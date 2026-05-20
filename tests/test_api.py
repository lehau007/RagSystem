import importlib
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(scope="module")
def client():
    """
    TestClient with AgenticChatbot fully mocked.
    api.main instantiates AgenticChatbot at module load time, so we must
    patch before importing (or reloading) the module.
    """
    mock_instance = MagicMock()

    with patch("core.chatbot.SemanticCache"), \
         patch("core.chatbot.HybridRetriever"), \
         patch("core.chatbot.Groq"), \
         patch("core.chatbot.load_prompt", return_value="t {query}"), \
         patch("core.chatbot.AgenticChatbot", return_value=mock_instance):
        import api.main as m
        importlib.reload(m)  # re-execute module-level chatbot = AgenticChatbot() with mock in place
        from fastapi.testclient import TestClient
        with TestClient(m.app) as c:
            c.mock = mock_instance
            yield c


def test_chat_success(client):
    client.mock.chat.return_value = {
        "response": "Sinh viên được đăng ký tối đa 24 tín chỉ.",
        "sub_queries": ["sub question"],
        "num_sources": 3,
        "from_cache": False,
    }
    resp = client.post("/chat", json={"user_input": "Hỏi về tín chỉ"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"] == "Sinh viên được đăng ký tối đa 24 tín chỉ."
    assert data["num_sources"] == 3
    assert data["from_cache"] is False


def test_chat_from_cache(client):
    client.mock.chat.return_value = {
        "response": "Cached answer",
        "sub_queries": ["(Từ Cache)"],
        "num_sources": 0,
        "from_cache": True,
    }
    resp = client.post("/chat", json={"user_input": "Repeated question"})
    assert resp.status_code == 200
    assert resp.json()["from_cache"] is True


def test_chat_with_history(client):
    client.mock.chat.return_value = {
        "response": "Answer with history context",
        "sub_queries": ["q"],
        "num_sources": 1,
        "from_cache": False,
    }
    history = [{"role": "user", "content": "Previous question"}]
    resp = client.post("/chat", json={"user_input": "Follow-up question", "history": history})
    assert resp.status_code == 200
    client.mock.chat.assert_called_with("Follow-up question", history=history)


def test_chat_internal_error(client):
    client.mock.chat.side_effect = Exception("LLM API timeout")
    resp = client.post("/chat", json={"user_input": "trigger error"})
    assert resp.status_code == 500
    assert "LLM API timeout" in resp.json()["detail"]
    client.mock.chat.side_effect = None  # reset for subsequent tests
