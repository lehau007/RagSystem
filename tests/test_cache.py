import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def cache_fixture():
    """
    SemanticCache with SentenceTransformer mocked to avoid model downloads.
    FAISS is already a MagicMock via conftest.py sys.modules pre-mocking.
    After init, cache_index is replaced with a controlled mock for test assertions.
    """
    with patch("core.cache.SentenceTransformer") as MockST:
        MockST.return_value = MagicMock()
        from core.cache import SemanticCache
        cache = SemanticCache()
        cache.cache_index = MagicMock()
        yield cache


class TestSemanticCacheInit:
    def test_cache_index_created_on_init(self):
        with patch("core.cache.SentenceTransformer") as MockST:
            MockST.return_value = MagicMock()
            from core.cache import SemanticCache
            cache = SemanticCache()
        assert cache.cache_index is not None

    def test_threshold_default_value(self, cache_fixture):
        assert cache_fixture.threshold == 0.92


class TestSemanticCacheGet:
    def test_returns_none_on_empty_results(self, cache_fixture):
        cache_fixture.cache_index.similarity_search_with_score.return_value = []
        assert cache_fixture.get("some question") is None

    def test_returns_none_when_score_above_threshold(self, cache_fixture):
        doc = Document(page_content="cached question", metadata={"response": "answer"})
        cache_fixture.cache_index.similarity_search_with_score.return_value = [(doc, 0.5)]
        assert cache_fixture.get("some question") is None

    def test_returns_cached_response_on_hit(self, cache_fixture):
        doc = Document(page_content="cached question", metadata={"response": "cached answer"})
        cache_fixture.cache_index.similarity_search_with_score.return_value = [(doc, 0.05)]
        result = cache_fixture.get("some question")
        assert result == "cached answer"

    def test_returns_none_for_seed_init_doc(self, cache_fixture):
        """The 'init' seed document must never be returned as a cache hit."""
        doc = Document(page_content="init", metadata={"response": ""})
        cache_fixture.cache_index.similarity_search_with_score.return_value = [(doc, 0.01)]
        assert cache_fixture.get("some question") is None


class TestSemanticCacheUpdate:
    def test_update_adds_document_to_index(self, cache_fixture):
        cache_fixture.update("new question", "new answer")
        cache_fixture.cache_index.add_documents.assert_called_once()

    def test_update_saves_index_to_disk(self, cache_fixture):
        cache_fixture.update("new question", "new answer")
        cache_fixture.cache_index.save_local.assert_called_once()

    def test_update_document_contains_correct_content(self, cache_fixture):
        cache_fixture.update("my question", "my answer")
        call_args = cache_fixture.cache_index.add_documents.call_args
        docs = call_args[0][0]
        assert docs[0].page_content == "my question"
        assert docs[0].metadata["response"] == "my answer"
