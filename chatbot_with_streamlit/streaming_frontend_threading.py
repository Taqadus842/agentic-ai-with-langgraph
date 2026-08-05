import streamlit as st
import uuid

from lanngraph_database_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage


# -----------------------------
# Helper Functions
# -----------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def update_conversation_title(user_input):
    """Set the conversation title using the first user message."""

    title = user_input.strip()

    if len(title) > 30:
        title = title[:30] + "..."

    for thread in st.session_state["chat_threads"]:
        if thread["id"] == st.session_state["thread_id"]:
            if thread["title"] == "New Chat":
                thread["title"] = title
            break


def reset_chat():
    new_thread = generate_thread_id()

    st.session_state["thread_id"] = new_thread
    st.session_state["message_history"] = []

    st.session_state["chat_threads"].append(
        {
            "id": new_thread,
            "title": "New Chat",
        }
    )

    st.rerun()


def load_conversation(thread_id):
    """Load messages from LangGraph memory."""
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = chatbot.get_state(config)

    if not state.values:
        return

    messages = state.values.get("messages", [])

    for msg in messages:

        if isinstance(msg, HumanMessage):
            st.session_state["message_history"].append(
                {
                    "role": "user",
                    "content": msg.content,
                }
            )

        elif isinstance(msg, AIMessage):
            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": msg.content,
                }
            )


def response_generator(user_input, config):
    """Stream response from LangGraph."""

    for message, metadata in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config,
        stream_mode="messages",
    ):

        if not isinstance(message, AIMessage):
            continue

        content = message.content

        # New LangChain content block format
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and block.get("text")
                ):
                    yield block["text"]

        # Old string format
        elif isinstance(content, str):
            yield content

# -----------------------------
# Session State
# -----------------------------
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        {
            "id": st.session_state["thread_id"],
            "title": "New Chat"
        }
    ]


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat"):
    reset_chat()

st.sidebar.divider()
st.sidebar.subheader("My Conversations")

for thread in st.session_state["chat_threads"]:
    if st.sidebar.button(
        thread["title"],
        key=thread["id"],
        use_container_width=True,
    ):
        load_conversation(thread["id"])
        st.rerun()


# -----------------------------
# Main Page
# -----------------------------
st.title("LangGraph Chatbot")

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Type your message...")

if user_input:

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    if len(st.session_state["message_history"]) == 1:
        update_conversation_title(user_input)   

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    with st.chat_message("assistant"):

        ai_response = st.write_stream(
            response_generator(user_input, config)
        )

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response,
        }
    )