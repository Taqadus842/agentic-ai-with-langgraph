import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot

st.title("LangGraph Chatbot")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "message_history" not in st.session_state:
    st.session_state.message_history = []

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here...")

if user_input:

    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):

        placeholder = st.empty()
        full_response = ""

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config={
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            },
            stream_mode="messages",
        ):

            if hasattr(message_chunk, "content") and message_chunk.content:
                full_response += message_chunk.content
                placeholder.markdown(full_response)

    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )