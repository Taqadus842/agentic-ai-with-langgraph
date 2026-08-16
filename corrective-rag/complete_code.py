from typing import TypedDict
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_tavily import TavilySearch
from langgraph.graph import (
    StateGraph,
    START,
    END,
)

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)

docs = (
    PyPDFLoader("book1.pdf").load()
    + PyPDFLoader("book2.pdf").load()
    + PyPDFLoader("book3.pdf").load()
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150
)

chunks = splitter.split_documents(docs)

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

class State(TypedDict):

    question: str

    docs: list[Document]

    good_docs: list[Document]

    web_docs: list[Document]

    all_docs: list[Document]

    verdict: str

    score: float

    reason: str

    strips: list[str]

    kept_strips: list[str]

    refined_context: str

    answer: str

    validation: str

class RetrievalEvaluation(BaseModel):

    score: float = Field(
        description="Retrieval quality score from 0 to 1."
    )

    reason: str = Field(
        description="Why the retrieved documents received this score."
    )

retrieval_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict retrieval evaluator.

Evaluate whether the retrieved documents are useful for
answering the user's question.

Score retrieval quality from 0 to 1.

Scoring:

0.0 - 0.3:
Poor retrieval. Documents are mostly irrelevant.

0.3 - 0.7:
Uncertain retrieval. Some information is relevant,
but coverage is incomplete.

0.7 - 1.0:
Good retrieval. Documents are directly relevant and
contain enough information to answer the question.

Consider:

- Relevance
- Coverage
- Specificity
- Usefulness as evidence

Use ONLY the provided documents.

Do not use outside knowledge.

Return structured output.
"""
        ),
        (
            "human",
            """
Question:
{question}

Retrieved Documents:
{context}
"""
        )
    ]
)


retrieval_eval_chain = (
    retrieval_eval_prompt
    | llm.with_structured_output(
        RetrievalEvaluation
    )
)

def retrieve(state: State) -> dict:

    question = state["question"]

    docs = retriever.invoke(
        question
    )

    return {
        "docs": docs
    }

def evaluate_retrieval(state: State) -> dict:

    context = "\n\n".join(
        f"Document {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(
            state["docs"]
        )
    )

    result = retrieval_eval_chain.invoke(
        {
            "question": state["question"],
            "context": context
        }
    )

    score = max(
        0.0,
        min(1.0, result.score)
    )

    if score >= 0.7:

        verdict = "good"

    elif score <= 0.3:

        verdict = "poor"

    else:

        verdict = "uncertain"

    return {
        "score": score,
        "verdict": verdict,
        "reason": result.reason
    }

def route_retrieval(state: State):

    return state["verdict"]

def accept_retrieval(state: State) -> dict:

    return {
        "good_docs": state["docs"],
        "all_docs": state["docs"]
    }

class RewrittenQuery(BaseModel):

    query: str = Field(
        description="Improved search query."
    )


query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert search-query optimizer.

Rewrite the user's question into a better retrieval query.

The new query should:

- Preserve the user's intent
- Add useful terminology
- Remove unnecessary wording
- Be specific
- Improve document retrieval

Return ONLY the rewritten query.
"""
        ),
        (
            "human",
            """
Original question:
{question}

Retrieval problem:
{reason}
"""
        )
    ]
)


query_rewrite_chain = (
    query_rewrite_prompt
    | llm.with_structured_output(
        RewrittenQuery
    )
)

def corrective_retrieval(
    state: State
) -> dict:

    result = query_rewrite_chain.invoke(
        {
            "question": state["question"],
            "reason": state["reason"]
        }
    )

    new_query = result.query

    corrected_docs = retriever.invoke(
        new_query
    )

    return {
        "good_docs": corrected_docs,
        "all_docs": corrected_docs
    }

web_search = TavilySearch(
    max_results=5
)


def web_retrieval(state: State) -> dict:

    question = state["question"]

    results = web_search.invoke(
        {
            "query": question
        }
    )

    web_docs = []

    for result in results.get(
        "results",
        []
    ):

        web_docs.append(
            Document(
                page_content=result.get(
                    "content",
                    ""
                ),
                metadata={
                    "title": result.get(
                        "title"
                    ),
                    "url": result.get(
                        "url"
                    ),
                    "source": "web"
                }
            )
        )

    return {
        "web_docs": web_docs,
        "good_docs": web_docs,
        "all_docs": web_docs
    }

def decompose_to_sentences(
    text: str
) -> list[str]:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]

class KeepOrDrop(BaseModel):

    keep: bool = Field(
        description=(
            "True if sentence directly helps "
            "answer the question."
        )
    )

filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict relevance filter.

Determine whether the sentence directly helps answer
the user's question.

Return keep=true ONLY if the sentence contains information
directly useful for answering the question.

Return keep=false if it is:

- Irrelevant
- Only loosely related
- Too vague
- Not useful for answering the question

Use ONLY the provided sentence.

Do not use outside knowledge.

Return structured output.
"""
        ),
        (
            "human",
            """
Question:
{question}

Sentence:
{sentence}
"""
        )
    ]
)


filter_chain = (
    filter_prompt
    | llm.with_structured_output(
        KeepOrDrop
    )
)

def refine(state: State) -> dict:

    context = "\n\n".join(
        doc.page_content
        for doc in state["good_docs"]
    ).strip()

    strips = decompose_to_sentences(
        context
    )

    kept_strips = []

    for sentence in strips:

        result = filter_chain.invoke(
            {
                "question": state["question"],
                "sentence": sentence
            }
        )

        if result.keep:

            kept_strips.append(
                sentence
            )

    refined_context = "\n\n".join(
        kept_strips
    ).strip()

    return {
        "strips": strips,
        "kept_strips": kept_strips,
        "refined_context": refined_context
    }

def generate(state: State) -> dict:

    prompt = f"""
Answer the question using ONLY the provided context.

Question:
{state["question"]}

Context:
{state["refined_context"]}

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the context is insufficient, say so.
4. Give a clear and concise answer.
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }

class AnswerValidation(BaseModel):

    valid: bool

    reason: str


validation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict answer validator.

Determine whether the generated answer is fully supported
by the provided context.

Check for:

- Unsupported claims
- Hallucinations
- Information not present in the context
- Contradictions
- Misinterpretation

Return valid=true only if the answer is supported
by the context.
"""
        ),
        (
            "human",
            """
Question:
{question}

Context:
{context}

Generated Answer:
{answer}
"""
        )
    ]
)


validation_chain = (
    validation_prompt
    | llm.with_structured_output(
        AnswerValidation
    )
)


def validate_answer(state: State) -> dict:

    result = validation_chain.invoke(
        {
            "question": state["question"],
            "context": state["refined_context"],
            "answer": state["answer"]
        }
    )

    return {
        "validation": result.reason
    }

g = StateGraph(State)


g.add_node(
    "retrieve",
    retrieve
)

g.add_node(
    "evaluate",
    evaluate_retrieval
)

g.add_node(
    "accept",
    accept_retrieval
)

g.add_node(
    "corrective_retrieval",
    corrective_retrieval
)

g.add_node(
    "web_retrieval",
    web_retrieval
)

g.add_node(
    "refine",
    refine
)

g.add_node(
    "generate",
    generate
)

g.add_node(
    "validate",
    validate_answer
)

g.add_edge(
    START,
    "retrieve"
)

g.add_edge(
    "retrieve",
    "evaluate"
)


g.add_conditional_edges(
    "evaluate",
    route_retrieval,
    {
        "good": "accept",

        "uncertain": "corrective_retrieval",

        "poor": "web_retrieval"
    }
)


g.add_edge(
    "accept",
    "refine"
)

g.add_edge(
    "corrective_retrieval",
    "refine"
)

g.add_edge(
    "web_retrieval",
    "refine"
)

g.add_edge(
    "refine",
    "generate"
)

g.add_edge(
    "generate",
    "validate"
)

g.add_edge(
    "validate",
    END
)

app = g.compile()
result = app.invoke(
    {
        "question": "What is the main concept discussed in the book?",

        "docs": [],

        "good_docs": [],

        "web_docs": [],

        "all_docs": [],

        "verdict": "",

        "score": 0.0,

        "reason": "",

        "strips": [],

        "kept_strips": [],

        "refined_context": "",

        "answer": "",

        "validation": ""
    }
)


print(
    "\nRetrieval Score:",
    result["score"]
)

print(
    "\nVerdict:",
    result["verdict"]
)

print(
    "\nReason:",
    result["reason"]
)

print(
    "\nAnswer:"
)

print(
    result["answer"]
)

print(
    "\nValidation:"
)

print(
    result["validation"]
)