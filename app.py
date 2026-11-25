
import streamlit as st
from rag_pipeline import ask_question

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG AI Chatbot")
st.write("Ask questions about your uploaded PDF document.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user question
question = st.chat_input("Ask something about your document...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer = ask_question(question)
                st.markdown(answer)

            except Exception as e:
                answer = f"An error occurred: {e}"
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
