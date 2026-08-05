from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import (
    StateGraph,
    START,
    END,
)
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# ---------------------------------
# Load Environment Variables
# ---------------------------------
load_dotenv()


# ---------------------------------
# Initialize Gemini
# ---------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    streaming=True,
)


# ---------------------------------
# State
# ---------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------
# Chat Node
# ---------------------------------
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


# ---------------------------------
# Build Graph
# ---------------------------------
builder = StateGraph(ChatState)

builder.add_node("chat_node", chat_node)

builder.add_edge(START, "chat_node")
builder.add_edge("chat_node", END)


# ---------------------------------
# Memory
# ---------------------------------
conn = sqlite3.connect(
    "chatbot_memory.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)

# Create tables
checkpointer.setup()


# ---------------------------------
# Compile Graph
# ---------------------------------
chatbot = builder.compile(
    checkpointer=checkpointer
)


# ---------------------------------
# Optional Test
# ---------------------------------
if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "thread_1"
        }
    }

    while True:

        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            break

        response = chatbot.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config=config,
        )

        print("\nAssistant:", response["messages"][-1].content)

        print("\nConversation Memory")

        state = chatbot.get_state(config)

        for message in state.values["messages"]:
            print(type(message).__name__, ":", message.content)

        print("-" * 50)