# HUST Academic Regulations - Agentic RAG System v2.0

A chatbot system that helps users look up academic regulations of Hanoi University of Science and Technology (HUST). It uses an advanced **Agentic RAG** framework to handle complex questions.

## 🚀 Key Features

- **Agentic Framework (LangGraph):** Automatically decomposes complex questions (Query Decomposition) into sub-queries to retrieve each target precisely.
- **AI-Driven Data Pipeline:** Uses `gemma-3-27b-it` to convert PDFs into optimized Markdown while preserving table and heading structures before processing.
- **Hybrid Search:** Combines semantic search (FAISS Vector) and keyword search (BM25) to improve accuracy for academic terminology.
- **Reranking Stage:** Uses a Cross-Encoder to rerank results, ensuring the 3–5 most relevant passages are included in context.
- **Semantic Chunking:** Splits documents based on semantic shifts rather than fixed character counts.
- **Semantic Caching:** Stores and detects similar questions to respond instantly, reducing API cost and latency.
- **Automated Evaluation:** Integrates the RAGAS framework with the Gemma-3 model for automated response-quality evaluation.

## 🛠 Tech Stack

- **LLM Core:** `openai/gpt-oss-120b` (via Groq Cloud).
- **AI Parsing & Eval:** `gemma-3-27b-it` (via Google GenAI).
- **Orchestration:** LangChain, LangGraph.
- **Vector Database:** FAISS.
- **Embeddings:** `google/embeddinggemma-300M` (Sentence Transformers).
- **Backend:** FastAPI.

## 📋 System Requirements

- Python 3.10+
- API Keys:
  - `GROQ_API_KEY`: Used for the agent and answer synthesis.
  - `GEMINI_API_KEY`: Used for Markdown preprocessing and RAGAS evaluation.
  - `HF_TOKEN`: Used to download the embedding model.

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd RagSystem
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirement.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file at the project root:
   ```env
   HF_TOKEN="your_hugging_face_token"
   GROQ_API_KEY="your_groq_api_key"
   GEMINI_API_KEY="your_google_gemini_api_key"
   ```

## 📖 Usage Guide

### 1. Data preprocessing (PDF -> Markdown -> Index)
Place the academic regulation PDF files into `data/documents/` and run:
```bash
python -m scripts.preprocess
```
*Note: This process uses AI to parse Markdown and includes a 15s/batch pause to avoid Free Tier rate limits.*

### 2. Run the Chatbot (CLI Mode)
Interactive command-line interface:
```bash
python run_cli_chat.py
```

### 3. Run the Web API (FastAPI)
Start the API server for applications:
```bash
uvicorn api.main:app --reload
```
API docs at: `http://127.0.0.1:8000/docs`

### 4. System evaluation (RAGAS)
Run the automated evaluation pipeline:
```bash
python -m scripts.evaluate
```

## 📂 Project Structure

```text
├── api/              # FastAPI implementation
├── config/           # Centralized settings & keys
├── core/
│   ├── cache.py      # Semantic caching logic
│   ├── chatbot.py    # Agentic flow (LangGraph)
│   └── retriever.py  # Hybrid search & reranking
├── data/
│   ├── documents/    # Raw PDF files
│   └── vector_store/ # FAISS, BM25, and cache indexes
├── scripts/
│   ├── preprocess.py # Data pipeline (PDF to Markdown)
│   └── evaluate.py   # RAGAS evaluation script
└── run_cli_chat.py   # CLI entry point
```

## 📜 License
This project was developed for learning and research purposes to support HUST academic regulations lookup.
```