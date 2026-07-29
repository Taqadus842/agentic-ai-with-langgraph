from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    streaming=True
)


# State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Chat node
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# Build graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Memory
checkpointer = InMemorySaver()

# Compile graph
chatbot = graph.compile(checkpointer=checkpointer)