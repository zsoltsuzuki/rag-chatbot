# Context-aware PDF Q&A Chatbot

A prototype RAG (Retrieval-Augmented Generation) chatbot that answers questions based on uploaded PDF documents. Built with Python, LangChain, and the OpenAI API.

## Features

- PDF parsing and automatic chunking
- Semantic search via FAISS vector store and OpenAI embeddings
- LLM-based response generation (GPT-4o-mini)
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
             OpenAIEmbeddings --> FAISS vectorstore
                                        |
                    user question --> Retriever (top-k similarity)
                                        |
                    chat history --> ConversationalRetrievalChain
                                        |
                                       LLM (ChatOpenAI)
                                        |
                                      Answer + sources
```

## Quickstart

**1. Clone the repo**

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Set your OpenAI API key**

```bash
export OPENAI_API_KEY="sk-..."
```

Or edit `config.py` directly.

**4. Run the app**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, upload a PDF, and start asking questions.

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model to use |
| `CHUNK_SIZE` | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `4` | Chunks retrieved per query |
| `MEMORY_WINDOW` | `5` | Past turns kept in context |

## Project structure

```
rag_chatbot/
├── app.py           # Streamlit UI
├── rag.py           # RAG pipeline (load, embed, retrieve, answer)
├── config.py        # Settings and prompt templates
├── requirements.txt
└── README.md
```

## Tech stack

- [LangChain](https://python.langchain.com/) - pipeline orchestration
- [OpenAI API](https://platform.openai.com/) - embeddings and LLM
- [FAISS](https://github.com/facebookresearch/faiss) - vector similarity search
- [Streamlit](https://streamlit.io/) - web interface
- [PyPDF](https://pypdf.readthedocs.io/) - PDF parsing
