"""
Pre-mock unavailable heavy packages in sys.modules before any test imports.

This allows core.chatbot, core.retriever, and core.cache to be imported
successfully in a lightweight test environment without GPU/ML dependencies.
The actual objects (Groq, HybridRetriever, SemanticCache) are then patched
inside individual test fixtures.
"""
import sys
from unittest.mock import MagicMock

_MOCK_MODULES = [
    "groq",
    "rank_bm25",
    "langchain_community",
    "langchain_community.vectorstores",
    "langchain_community.vectorstores.faiss",
]

for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
