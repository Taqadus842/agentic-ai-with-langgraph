import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        {   
            "configurable":{
                "thread_id":st.session_state.thread_id
        }}

    )

    ai_message = response["messages"][-1].content

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    with st.chat_message("assistant"):
        st.write(ai_message)