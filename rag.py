"""
RAG pipeline: PDF parsing -> chunking -> embedding -> retrieval -> LLM response
LangChain 0.3 + Qdrant in-memory (Python 3.13 compatible, no C++ build required)
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from config import (
    OPENAI_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RETRIEVAL,
    MEMORY_WINDOW,
    LLM_MODEL,
    QA_PROMPT_TEMPLATE,
    CONDENSE_QUESTION_TEMPLATE,
)
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_and_split(pdf_path: str) -> list:
    """Load a PDF and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks: list) -> Qdrant:
    """Embed chunks and store in Qdrant in-memory instance."""
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",
        collection_name="rag_docs",
    )
    return vectorstore


def build_chain(vectorstore: Qdrant) -> ConversationalRetrievalChain:
    """Assemble the full RAG chain with prompt templates. Memory managed externally."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    qa_prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question", "chat_history"],
    )

    condense_prompt = PromptTemplate(
        template=CONDENSE_QUESTION_TEMPLATE,
        input_variables=["chat_history", "question"],
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RETRIEVAL},
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ask(chain: ConversationalRetrievalChain, question: str, chat_history: list) -> dict:
    """
    Send a question through the chain.
    chat_history: list of (human, ai) string tuples from session state.
    Returns: {"answer": str, "sources": list[str]}
    """
    result = chain.invoke({"question": question, "chat_history": chat_history})
    sources = list(
        {doc.metadata.get("source", "unknown") + f" (p.{doc.metadata.get('page', '?') + 1})"
         for doc in result.get("source_documents", [])}
    )
    return {"answer": result["answer"], "sources": sources}