# Context-aware PDF Q&A Chatbot

A prototype RAG (Retrieval-Augmented Generation) chatbot that answers questions based on uploaded PDF documents. Built with Python and LangChain, with a switchable LLM layer: OpenAI, Groq (cloud) or a local model via Ollama.

## Features

- PDF parsing and automatic chunking
- Semantic search via an in-memory Qdrant vector store
- Switchable LLM provider: OpenAI, Groq or local Ollama
- Short-term conversational memory (sliding window)
- Prompt templates for consistent, grounded answers
- Source page references in every response
- Streamlit web interface

## Architecture

```
PDF upload
    |
    v
PyPDFLoader --> RecursiveCharacterTextSplitter
                    |
                    v
             Embeddings (OpenAI or Ollama) --> Qdrant (in-memory)
                                                    |
                    user question --> Retriever (top-k similarity)
                                                    |
                    chat history --> ConversationalRetrievalChain
                                                    |
                                   LLM (OpenAI / Groq / Ollama)
                                                    |
                                            Answer + sources
```

The vector store is rebuilt in memory on every upload, so switching providers never reuses vectors created by a different embedding model.

## LLM providers

The provider is selected with `PROVIDER` in `config.py`.

| PROVIDER | LLM | Embeddings | Requirements |
|---|---|---|---|
| `openai` | OpenAI (`LLM_MODEL`) | OpenAI (`EMBEDDING_MODEL`) | `OPENAI_API_KEY` |
| `groq` | Groq cloud (`GROQ_LLM_MODEL`) | local Ollama (`OLLAMA_EMBEDDING_MODEL`) | `GROQ_API_KEY` + Ollama running |
| `ollama` | local Ollama (`OLLAMA_LLM_MODEL`) | local Ollama (`OLLAMA_EMBEDDING_MODEL`) | Ollama running |

Groq does not provide an embedding model, so in `groq` mode the embeddings are computed locally with Ollama. This also means that `groq` and `ollama` modes share the same retrieval step and differ only in the LLM, which makes them directly comparable.

Groq is called through its OpenAI-compatible endpoint, so no extra package is needed for it. Available Groq model IDs change over time; check them in the Groq console before setting `GROQ_LLM_MODEL`.

All providers use `temperature=0`.

## Quickstart

### 1. Clone the repo

```
git clone https://github.com/zsoltsuzuki/rag-chatbot.git
cd rag-chatbot
```

### 2. Create a virtual environment and install dependencies

```
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
```

### 3. Set up the provider you want to use

**OpenAI**

```
$env:OPENAI_API_KEY="sk-..."       # PowerShell
# export OPENAI_API_KEY="sk-..."   # macOS / Linux
```

**Groq** (free tier available at console.groq.com)

```
$env:GROQ_API_KEY="gsk_..."
ollama pull nomic-embed-text
```

**Ollama (fully local)**

Install Ollama from ollama.com, make sure it is running, then:

```
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

API keys are read from environment variables. Do not paste them into `config.py` if you plan to commit the file.

### 4. Run the app

```
streamlit run app.py
```

Open http://localhost:8501 in your browser, upload a PDF, and start asking questions.

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `PROVIDER` | `ollama` | `openai`, `groq` or `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `GROQ_LLM_MODEL` | see `config.py` | Groq chat model ID |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq OpenAI-compatible endpoint |
| `OLLAMA_LLM_MODEL` | `llama3.2:3b` | Local chat model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Local embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `CHUNK_SIZE` | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `8` | Chunks retrieved per query |
| `MEMORY_WINDOW` | `5` | Past turns kept in context |

## Project structure

```
rag_chatbot/
├── app.py           # Streamlit UI
├── rag.py           # RAG pipeline (load, embed, retrieve, answer) and provider selection
├── config.py        # Settings and prompt templates
├── requirements.txt # Pinned dependency versions
└── README.md
```

## Tech stack

- LangChain - pipeline orchestration
- Qdrant - in-memory vector similarity search
- OpenAI API - embeddings and LLM
- Groq - cloud LLM inference (OpenAI-compatible API)
- Ollama - local LLM and embedding models
- Streamlit - web interface
- PyPDF - PDF parsing

## Notes

- `nomic-embed-text` is optimized for English; retrieval quality on non-English PDFs is lower.
- On the Groq free tier, rapid consecutive questions can hit the per-minute token limit (HTTP 429). Waiting a few seconds between questions avoids this.
- The legacy `ConversationalRetrievalChain` requires `langchain-core` below 1.0. The pinned versions in `requirements.txt` reflect a working combination.