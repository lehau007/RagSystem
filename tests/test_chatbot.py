import json
import pytest
from unittest.mock import MagicMock, patch


def groq_resp(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture
def chatbot_fixture():
    with patch("core.chatbot.SemanticCache") as MockCache, \
         patch("core.chatbot.HybridRetriever") as MockRetriever, \
         patch("core.chatbot.Groq") as MockGroq, \
         patch("core.chatbot.load_prompt", return_value="Template: {query}"):
        mock_groq = MagicMock()
        MockGroq.return_value = mock_groq
        mock_retriever = MagicMock()
        MockRetriever.return_value = mock_retriever
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        MockCache.return_value = mock_cache

        from core.chatbot import AgenticChatbot
        chatbot = AgenticChatbot()
        yield chatbot, mock_groq, mock_retriever, mock_cache


class TestInit:
    def test_creates_workflow(self, chatbot_fixture):
        chatbot, _, _, _ = chatbot_fixture
        assert chatbot.workflow is not None

    def test_instantiates_dependencies(self):
        with patch("core.chatbot.SemanticCache") as MC, \
             patch("core.chatbot.HybridRetriever") as MR, \
             patch("core.chatbot.Groq") as MG, \
             patch("core.chatbot.load_prompt", return_value="t {query} {contexts}"):
            MG.return_value = MagicMock()
            MR.return_value = MagicMock()
            MC.return_value = MagicMock()
            from core.chatbot import AgenticChatbot
            AgenticChatbot()
            MG.assert_called_once()
            MR.assert_called_once()
            MC.assert_called_once()


class TestDecomposeQuery:
    def test_returns_sub_queries(self, chatbot_fixture):
        chatbot, mock_groq, _, _ = chatbot_fixture
        mock_groq.chat.completions.create.return_value = groq_resp(
            json.dumps({"sub_queries": ["câu hỏi 1", "câu hỏi 2"]})
        )
        state = {"query": "Q?", "sub_queries": [], "contexts": [], "response": "", "history": []}
        result = chatbot.decompose_query(state)
        assert result["sub_queries"] == ["câu hỏi 1", "câu hỏi 2"]

    def test_fallback_on_invalid_json(self, chatbot_fixture):
        chatbot, mock_groq, _, _ = chatbot_fixture
        mock_groq.chat.completions.create.return_value = groq_resp("not json")
        state = {"query": "Q?", "sub_queries": [], "contexts": [], "response": "", "history": []}
        result = chatbot.decompose_query(state)
        assert result["sub_queries"] == ["Q?"]

    def test_fallback_on_missing_key(self, chatbot_fixture):
        chatbot, mock_groq, _, _ = chatbot_fixture
        mock_groq.chat.completions.create.return_value = groq_resp(
            json.dumps({"wrong_key": []})
        )
        state = {"query": "Q?", "sub_queries": [], "contexts": [], "response": "", "history": []}
        result = chatbot.decompose_query(state)
        assert result["sub_queries"] == ["Q?"]


class TestRetrieveContext:
    def test_deduplicates_identical_contexts(self, chatbot_fixture):
        from langchain_core.documents import Document
        chatbot, _, mock_retriever, _ = chatbot_fixture
        doc = Document(page_content="Same content", metadata={"source": "test.pdf"})
        mock_retriever.retrieve.return_value = [doc, doc]
        state = {"query": "q", "sub_queries": ["sub1"], "contexts": [], "response": "", "history": []}
        result = chatbot.retrieve_context(state)
        assert len(result["contexts"]) == 1

    def test_calls_retriever_for_each_sub_query(self, chatbot_fixture):
        from langchain_core.documents import Document
        chatbot, _, mock_retriever, _ = chatbot_fixture
        doc_a = Document(page_content="Content A", metadata={"source": "a.pdf"})
        doc_b = Document(page_content="Content B", metadata={"source": "b.pdf"})
        mock_retriever.retrieve.side_effect = [[doc_a], [doc_b]]
        state = {"query": "q", "sub_queries": ["s1", "s2"], "contexts": [], "response": "", "history": []}
        result = chatbot.retrieve_context(state)
        assert len(result["contexts"]) == 2
        assert mock_retriever.retrieve.call_count == 2


class TestSynthesizeResponse:
    def test_returns_llm_response(self, chatbot_fixture):
        chatbot, mock_groq, _, _ = chatbot_fixture
        mock_groq.chat.completions.create.return_value = groq_resp("Final answer")
        state = {"query": "Q?", "sub_queries": [], "contexts": ["ctx"], "response": "", "history": []}
        result = chatbot.synthesize_response(state)
        assert result["response"] == "Final answer"


class TestChat:
    def test_returns_cached_response(self, chatbot_fixture):
        chatbot, _, _, mock_cache = chatbot_fixture
        mock_cache.get.return_value = "Cached answer"
        result = chatbot.chat("Some question")
        assert result["response"] == "Cached answer"
        assert result["from_cache"] is True
        assert result["sub_queries"] == ["(Từ Cache)"]

    def test_runs_workflow_on_cache_miss(self, chatbot_fixture):
        from langchain_core.documents import Document
        chatbot, mock_groq, mock_retriever, mock_cache = chatbot_fixture
        mock_cache.get.return_value = None
        doc = Document(page_content="Regulation text", metadata={"source": "reg.pdf"})
        mock_retriever.retrieve.return_value = [doc]
        mock_groq.chat.completions.create.side_effect = [
            groq_resp(json.dumps({"sub_queries": ["sub1"]})),
            groq_resp("Synthesized answer"),
        ]
        result = chatbot.chat("Question about regulations")
        assert result["from_cache"] is False
        assert result["response"] == "Synthesized answer"

    def test_updates_cache_after_response(self, chatbot_fixture):
        from langchain_core.documents import Document
        chatbot, mock_groq, mock_retriever, mock_cache = chatbot_fixture
        mock_cache.get.return_value = None
        mock_retriever.retrieve.return_value = [
            Document(page_content="ctx", metadata={"source": "s.pdf"})
        ]
        mock_groq.chat.completions.create.side_effect = [
            groq_resp(json.dumps({"sub_queries": ["q1"]})),
            groq_resp("Answer"),
        ]
        chatbot.chat("Question")
        mock_cache.update.assert_called_once_with("Question", "Answer")
