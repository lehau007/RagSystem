# config/settings.py

# This file centralizes configuration for the application.
# It's a good practice to store paths, model names, and other settings here.

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
HF_TOKEN = os.getenv("HF_TOKEN")
OSSAPI_KEY = os.getenv("OSSAPI_KEY")

# File Paths
# Use absolute paths to avoid issues with the current working directory.
# os.path.dirname(os.path.abspath(__file__)) gives you the directory of the current file (config).
# os.path.join(BASE_DIR, 'data', ...) constructs a path from the project root.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCUMENTS_PATH = os.path.join(BASE_DIR, "data", "documents")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")
CHUNKED_DOCS_PATH = os.path.join(VECTOR_STORE_PATH, "chunked_docs.json")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "vector_db")

# Model Names
EMBEDDING_MODEL = "google/embeddinggemma-300M"
CHAT_MODEL = "openai/gpt-oss-20b"
