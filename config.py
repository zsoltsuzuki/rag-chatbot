"""
Central configuration: model settings, chunking params, and prompt templates.
"""

import os

# --- API Key ---

# Set via environment variable or paste directly (not recommended for production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# --- LLM ---
PROVIDER = "groq"    # "openai" vagy "ollama"

GROQ_LLM_MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# OpenAI
LLM_MODEL = "gpt-4o-mini"  # cost-efficient; swap to gpt-4o for higher quality
EMBEDDING_MODEL = "text-embedding-3-small"

# Ollama (local)  
OLLAMA_LLM_MODEL = "llama3.2:3b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
#OLLAMA_EMBEDDING_MODEL = "bge-m3"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Chunking ---
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks

# --- Retrieval ---
TOP_K_RETRIEVAL = 8    # number of chunks retrieved per query

# --- Memory ---
MEMORY_WINDOW = 5      # how many past Q&A turns to keep in context

# --- Prompt Templates ---

QA_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about an uploaded document.

Use the context below to answer the question. If the question asks for a summary, \
summarize all the context provided. If the answer is not in the context, say so clearly \
but still try to be helpful based on what is available.

Context:
{context}

Chat history:
{chat_history}

Question: {question}

Answer:"""

CONDENSE_QUESTION_TEMPLATE = """Given the conversation history and a follow-up \
question, rephrase the follow-up question into a standalone question that can be \
understood without the chat history.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""