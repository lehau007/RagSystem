import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from langchain_core.documents import Document

from core import prompt_loader, retriever


class FakeBM25:
    def __init__(self, scores):
        self.scores = np.array(scores, dtype=float)

    def get_scores(self, _tokenized_query):
        return self.scores


def test_load_prompt_from_hub_when_enabled(monkeypatch):
    monkeypatch.setattr("config.settings.LANGSMITH_ENABLED", True)

    pulled = MagicMock()
    pulled.template = "Prompt from hub"

    with patch("langchain.hub.pull", return_value=pulled) as mock_pull:
        template = prompt_loader.load_prompt("ignored", hub_path="org/prompt")

    assert template == "Prompt from hub"
    mock_pull.assert_called_once_with("org/prompt")


def test_load_prompt_falls_back_to_local_yaml_when_hub_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.LANGSMITH_ENABLED", True)
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", Path(tmp_path))
    (tmp_path / "fallback.yaml").write_text('template: "Local fallback template"', encoding="utf-8")

    with patch("langchain.hub.pull", side_effect=RuntimeError("hub is unavailable")):
        template = prompt_loader.load_prompt("fallback", hub_path="org/fallback")

    assert template == "Local fallback template"


def test_load_prompt_reads_local_yaml_when_langsmith_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.LANGSMITH_ENABLED", False)
    monkeypatch.setattr(prompt_loader, "PROMPTS_DIR", Path(tmp_path))
    (tmp_path / "local_only.yaml").write_text('template: "Local only template"', encoding="utf-8")

    template = prompt_loader.load_prompt("local_only", hub_path="org/unused")
    assert template == "Local only template"


def _build_retriever(tmp_path, monkeypatch):
    bm25_path = tmp_path / "bm25.pkl"
    bm25_path.write_bytes(b"placeholder")

    docs_path = tmp_path / "chunked_docs.json"
    docs_path.write_text(
        json.dumps(
            [
                {"page_content": "dense-1", "metadata": {"source": "a.txt"}},
                {"page_content": "bm25-only", "metadata": {"source": "b.txt"}},
                {"page_content": "another-doc", "metadata": {"source": "c.txt"}},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(retriever, "BM25_INDEX_PATH", str(bm25_path))
    monkeypatch.setattr(retriever, "CHUNKED_DOCS_PATH", str(docs_path))
    monkeypatch.setattr(retriever, "FAISS_INDEX_PATH", str(tmp_path / "vector_db"))
    monkeypatch.setattr(retriever, "EMBEDDING_MODEL", "fake-model")

    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda texts, show_progress_bar=False: np.array(
        [[float(len(text))] for text in texts], dtype=float
    )

    fake_vectorstore = MagicMock()
    fake_vectorstore.similarity_search_with_score.return_value = [
        (Document(page_content="dense-1", metadata={"source": "dense.txt"}), 0.2),
        (Document(page_content="vector-only", metadata={"source": "vector.txt"}), 0.4),
    ]

    fake_reranker = MagicMock()
    fake_reranker.predict.side_effect = lambda pairs: np.array(
        [2.0 if "vector-only" in pair[1] else 1.0 if "dense-1" in pair[1] else 0.5 for pair in pairs]
    )

    with patch.object(retriever, "SentenceTransformer", return_value=fake_model) as mock_sentence_transformer, \
         patch.object(retriever.FAISS, "load_local", return_value=fake_vectorstore) as mock_load_local, \
         patch.object(retriever, "CrossEncoder", return_value=fake_reranker) as mock_cross_encoder, \
         patch.object(retriever.pickle, "load", return_value=FakeBM25([0.8, 0.9, 0.1])):
        instance = retriever.HybridRetriever("hf-test-token")

    return instance, fake_model, fake_vectorstore, fake_reranker, mock_sentence_transformer, mock_load_local, mock_cross_encoder


def test_sentence_transformer_embeddings_wrapper_methods():
    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda texts, show_progress_bar=False: np.array(
        [[float(len(text))] for text in texts], dtype=float
    )

    embedder = retriever.SentenceTransformerEmbeddings(fake_model)
    assert embedder.embed_documents(["ab", "abcd"]) == [[2.0], [4.0]]
    assert embedder.embed_query("abc") == [3.0]


def test_hybrid_retriever_init_loads_dependencies(tmp_path, monkeypatch):
    instance, _, _, _, mock_sentence_transformer, mock_load_local, mock_cross_encoder = _build_retriever(
        tmp_path, monkeypatch
    )

    assert len(instance.chunked_docs) == 3
    mock_sentence_transformer.assert_called_once_with("fake-model", use_auth_token="hf-test-token")
    assert mock_load_local.call_args.kwargs["allow_dangerous_deserialization"] is True
    mock_cross_encoder.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_hybrid_retriever_retrieve_fuses_and_reranks_results(tmp_path, monkeypatch):
    instance, _, fake_vectorstore, fake_reranker, _, _, _ = _build_retriever(tmp_path, monkeypatch)

    results = instance.retrieve("which doc", top_k=3, rerank_top_n=2)

    fake_vectorstore.similarity_search_with_score.assert_called_once_with("which doc", k=6)
    fake_reranker.predict.assert_called_once()
    assert len(results) == 2
    assert results[0].page_content == "vector-only"


def test_hybrid_retriever_retrieve_skips_rerank_when_disabled(tmp_path, monkeypatch):
    instance, _, _, fake_reranker, _, _, _ = _build_retriever(tmp_path, monkeypatch)

    results = instance.retrieve("which doc", top_k=2, rerank_top_n=0)

    assert len(results) == 2
    fake_reranker.predict.assert_not_called()
