# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# LangSmith / LangChain Observability
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "hust-rag-system")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGSMITH_ENABLED = LANGCHAIN_TRACING_V2.lower() == "true" and bool(LANGCHAIN_API_KEY)

# Propagate to os.environ so LangGraph native tracing activates at import time
if LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
if LANGCHAIN_PROJECT:
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_PATH = os.path.join(BASE_DIR, "data", "documents")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")
CHUNKED_DOCS_PATH = os.path.join(VECTOR_STORE_PATH, "chunked_docs.json")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "vector_db")
BM25_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "bm25_index.pkl")

# Model Names
EMBEDDING_MODEL = "google/embeddinggemma-300M"
CHAT_MODEL = "openai/gpt-oss-120b"
