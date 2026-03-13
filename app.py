"""
Streamlit frontend for the Context-aware PDF Q&A Chatbot.

Run with:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
from rag import load_and_split, build_vectorstore, build_chain, ask
from config import MEMORY_WINDOW

# --- Page config ---
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="--",
    layout="centered",
)

st.title("Context-aware PDF Q&A Chatbot")
st.caption("Upload a PDF, then ask questions about its content.")

# --- Session state ---
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (human, ai) tuples
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# --- Sidebar: PDF upload ---
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and uploaded_file.name != st.session_state.doc_name:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                chunks = load_and_split(tmp_path)
                vectorstore = build_vectorstore(chunks)
                st.session_state.chain = build_chain(vectorstore)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.success(f"Ready. {len(chunks)} chunks indexed.")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
            finally:
                os.unlink(tmp_path)

    if st.session_state.doc_name:
        st.info(f"Active: {st.session_state.doc_name}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# --- Chat display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.write(src)

# --- Chat input ---
if prompt := st.chat_input("Ask a question about the document..."):
    if st.session_state.chain is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(
                    st.session_state.chain,
                    prompt,
                    st.session_state.chat_history[-MEMORY_WINDOW:],
                )
                answer = result["answer"]
                sources = result["sources"]
            except Exception as e:
                answer = f"Error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.write(src)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
    st.session_state.chat_history.append((prompt, answer))