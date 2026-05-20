# HUST Academic Regulations — Agentic RAG System v2.0

A chatbot that answers questions about Hanoi University of Science and Technology (HUST) academic regulations. Built on an **Agentic RAG** framework using LangGraph for multi-step reasoning over official regulation documents.

## Key Features

- **Query Decomposition (LangGraph):** Breaks complex questions into sub-queries and retrieves each independently before synthesizing a final answer.
- **Hybrid Search:** Combines FAISS vector search and BM25 keyword search via Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Reranking:** Reranks fused results with a cross-encoder model for higher-precision context selection.
- **Semantic Caching:** Detects similar repeated questions and returns cached answers instantly, reducing API cost and latency.
- **AI-Driven Data Pipeline:** Uses `gemma-3-27b-it` to convert PDFs to structured Markdown before chunking and indexing.
- **LangSmith Observability:** Full trace visibility into each LangGraph node, LLM call, and retrieval step (optional).
- **Prompt Versioning:** Prompts stored as YAML files in `langsmith_prompts/`, optionally synced with LangSmith Hub.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | `openai/gpt-oss-120b` via Groq Cloud |
| Orchestration | LangGraph + LangChain |
| Embeddings | `google/embeddinggemma-300M` (Sentence Transformers) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector DB | FAISS (CPU) |
| Keyword Search | BM25 (rank_bm25) |
| Backend | FastAPI |
| Observability | LangSmith |
| Evaluation | RAGAS |

---

## Installation

### 1. Clone the repository

```bash
git clone <repository_url>
cd RagSystem
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Configure environment variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
# Required
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_google_gemini_api_key
HF_TOKEN=your_hugging_face_token

# Optional — LangSmith observability (set to "true" to enable)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=hust-rag-system
```

> **Where to get keys:**
> - `GROQ_API_KEY` → [console.groq.com/keys](https://console.groq.com/keys)
> - `GEMINI_API_KEY` → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
> - `HF_TOKEN` → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
> - `LANGCHAIN_API_KEY` → [smith.langchain.com/settings](https://smith.langchain.com/settings) *(only needed if enabling LangSmith)*

---

## Usage

### 1. Preprocess documents (PDF → Markdown → Indexes)

Place regulation PDF files in `data/documents/` then run:

```bash
python -m scripts.preprocess
```

This runs the full pipeline: PDF loading → Gemini Markdown conversion → semantic chunking → FAISS + BM25 index creation.

> **Note:** AI-driven Markdown conversion includes 15-second pauses between batches to stay within free-tier rate limits.

### 2. Run the chatbot (CLI)

```bash
python run_cli_chat.py
```

### 3. Run the web API (FastAPI)

```bash
uvicorn api.main:app --reload
```

API docs available at `http://127.0.0.1:8000/docs`.

### 4. Evaluate with golden dataset (no LLM required)

Run keyword-match and retrieval-precision evaluation against the static golden dataset:

```bash
python scripts/evaluate_golden.py
```

### 5. Evaluate with RAGAS (LLM-as-judge)

Full automated evaluation using Gemini as the judge LLM:

```bash
python -m scripts.evaluate
```

> **Note:** Requires `GEMINI_API_KEY`. This is slow and costs API credits — not suitable for CI.

---

## LangSmith Observability

To enable full tracing in the [LangSmith](https://smith.langchain.com) dashboard:

1. Set `LANGCHAIN_TRACING_V2=true` in your `.env`
2. Add your `LANGCHAIN_API_KEY`
3. Run the chatbot — each request will appear as a trace under project `hust-rag-system`

LangGraph traces the full workflow graph automatically. Each LLM call (decomposition, synthesis) appears as a child run within the graph trace.

### Prompt versioning

Prompts are stored as YAML files in `langsmith_prompts/`:

```
langsmith_prompts/
├── decompose_query.yaml      # Query decomposition prompt
└── synthesize_response.yaml  # Answer synthesis prompt
```

Edit these files to update prompts without touching code. When LangSmith is enabled and `hub_path` is configured in `core/prompt_loader.py`, prompts can also be pulled from LangSmith Hub for centralized versioning across deployments.

---

## Running Tests

```bash
# Activate venv first
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate    # Windows

# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=core --cov=api --cov-report=term-missing
```

Tests mock all external dependencies (Groq API, FAISS, BM25, sentence-transformers) — no API keys or model downloads required.

---

## CI/CD

GitHub Actions runs automatically on every push to `main` or `develop` and on pull requests:

- **Lint job:** `ruff` static analysis
- **Test job:** `pytest` with ≥70% coverage gate

No API keys are needed in CI — all external calls are mocked.

---

## Project Structure

```
RagSystem/
├── .env                        # Your local API keys (gitignored)
├── .env.example                # Template — copy to .env and fill in keys
├── requirement.txt             # Python dependencies
├── run_cli_chat.py             # CLI entry point
│
├── api/
│   └── main.py                 # FastAPI app (POST /chat)
│
├── config/
│   └── settings.py             # Env var loading and path constants
│
├── core/
│   ├── chatbot.py              # LangGraph agentic workflow
│   ├── retriever.py            # Hybrid search + reranking
│   ├── cache.py                # Semantic FAISS cache
│   └── prompt_loader.py        # YAML / LangSmith Hub prompt loader
│
├── langsmith_prompts/
│   ├── decompose_query.yaml    # Versioned decomposition prompt
│   └── synthesize_response.yaml # Versioned synthesis prompt
│
├── data/
│   ├── documents/              # Raw PDF files
│   └── vector_store/           # FAISS indexes, BM25 index, cache
│
├── scripts/
│   ├── preprocess.py           # PDF → Markdown → FAISS + BM25 pipeline
│   ├── evaluate.py             # RAGAS evaluation (LLM-as-judge)
│   └── evaluate_golden.py      # Keyword-match evaluation (no LLM)
│
├── tests/
│   ├── conftest.py             # sys.modules pre-mocking for test isolation
│   ├── test_chatbot.py         # Unit tests for AgenticChatbot
│   ├── test_api.py             # Unit tests for FastAPI endpoints
│   └── fixtures/
│       └── golden_dataset.jsonl # 15 static Q&A samples for evaluation
│
└── .github/
    └── workflows/
        └── ci.yml              # Lint + test pipeline
```

---

## License

Developed for learning and research purposes to support HUST academic regulations lookup.
