# config/settings.py

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
