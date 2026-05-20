import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def retriever_fixture():
    """
    HybridRetriever constructed via object.__new__ to skip __init__ entirely.
    All ML model loading (SentenceTransformer, CrossEncoder, FAISS, BM25) is
    bypassed; attributes are set directly as MagicMocks.
    """
    from core.retriever import HybridRetriever
    retriever = object.__new__(HybridRetriever)
    retriever.vectorstore = MagicMock()
    retriever.bm25 = MagicMock()
    retriever.reranker = MagicMock()
    retriever.chunked_docs = [
        Document(page_content=f"doc content {i}", metadata={"source": f"file{i}.pdf"})
        for i in range(5)
    ]
    retriever.model = MagicMock()
    retriever.embedder = MagicMock()
    return retriever


class TestHybridRetrieverRetrieve:
    def test_returns_list_of_documents(self, retriever_fixture):
        retriever_fixture.vectorstore.similarity_search_with_score.return_value = []
        retriever_fixture.bm25.get_scores.return_value = np.zeros(5)
        retriever_fixture.reranker.predict.return_value = np.array([0.9, 0.8, 0.7])
        result = retriever_fixture.retrieve("test query", top_k=5, rerank_top_n=3)
        assert isinstance(result, list)

    def test_deduplicates_vector_and_bm25_results(self, retriever_fixture):
        doc = Document(page_content="shared content", metadata={"source": "a.pdf"})
        retriever_fixture.vectorstore.similarity_search_with_score.return_value = [(doc, 0.1)]
        # BM25 returns a score for index 0, pointing to the same doc content
        scores = np.zeros(5)
        scores[0] = 1.0  # only first doc has a score
        retriever_fixture.bm25.get_scores.return_value = scores
        retriever_fixture.chunked_docs[0] = Document(
            page_content="shared content", metadata={"source": "a.pdf"}
        )
        retriever_fixture.reranker.predict.return_value = np.array([0.9])
        result = retriever_fixture.retrieve("query", top_k=5, rerank_top_n=1)
        # The same content appears in both vector and BM25 — should be deduplicated
        contents = [d.page_content for d in result]
        assert len(contents) == len(set(contents))

    def test_reranking_applied_and_limits_results(self, retriever_fixture):
        docs = [
            Document(page_content=f"content {i}", metadata={"source": f"{i}.pdf"})
            for i in range(3)
        ]
        retriever_fixture.vectorstore.similarity_search_with_score.return_value = [
            (docs[0], 0.1), (docs[1], 0.2), (docs[2], 0.3)
        ]
        retriever_fixture.bm25.get_scores.return_value = np.zeros(5)
        retriever_fixture.reranker.predict.return_value = np.array([0.5, 0.9, 0.1, 0.3, 0.2])
        result = retriever_fixture.retrieve("query", top_k=5, rerank_top_n=3)
        retriever_fixture.reranker.predict.assert_called_once()
        assert len(result) == 3

    def test_no_reranking_when_rerank_top_n_zero(self, retriever_fixture):
        doc = Document(page_content="content", metadata={"source": "a.pdf"})
        retriever_fixture.vectorstore.similarity_search_with_score.return_value = [(doc, 0.1)]
        retriever_fixture.bm25.get_scores.return_value = np.zeros(5)
        retriever_fixture.retrieve("query", top_k=5, rerank_top_n=0)
        retriever_fixture.reranker.predict.assert_not_called()
