# HUST Academic Regulations - Agentic RAG System v2.0

Hệ thống chatbot hỗ trợ tra cứu quy chế học vụ Đại học Bách khoa Hà Nội (HUST), sử dụng framework **Agentic RAG** tiên tiến để xử lý các câu hỏi phức tạp và đa mục tiêu.

## 🚀 Tính năng nổi bật

-   **Agentic Framework (LangGraph):** Tự động phân rã câu hỏi phức tạp (Query Decomposition) thành các truy vấn con để tìm kiếm chính xác từng mục tiêu.
-   **AI-Driven Data Pipeline:** Sử dụng `gemma-3-27b-it` để chuyển đổi PDF sang Markdown tối ưu, bảo toàn cấu trúc bảng biểu và tiêu đề trước khi xử lý.
-   **Hybrid Search:** Kết hợp tìm kiếm ngữ nghĩa (FAISS Vector) và tìm kiếm từ khóa (BM25) để tối ưu hóa độ chính xác cho các thuật ngữ học thuật.
-   **Reranking Stage:** Sử dụng Cross-Encoder để xếp hạng lại kết quả, đảm bảo 3-5 đoạn văn bản liên quan nhất được đưa vào ngữ cảnh.
-   **Semantic Chunking:** Cắt tài liệu dựa trên sự thay đổi ý nghĩa thay vì số lượng ký tự cố định.
-   **Semantic Caching:** Lưu trữ và nhận diện các câu hỏi tương đồng để phản hồi ngay lập tức, tiết kiệm chi phí API và giảm latency.
-   **Automated Evaluation:** Tích hợp framework RAGAS với mô hình Gemma-3 để đánh giá tự động chất lượng phản hồi.

## 🛠 Tech Stack

-   **LLM Core:** `openai/gpt-oss-120b` (via Groq Cloud).
-   **AI Parsing & Eval:** `gemma-3-27b-it` (via Google GenAI).
-   **Orchestration:** LangChain, LangGraph.
-   **Vector Database:** FAISS.
-   **Embeddings:** `google/embeddinggemma-300M` (Sentence Transformers).
-   **Backend:** FastAPI.

## 📋 Yêu cầu hệ thống

-   Python 3.10+
-   API Keys:
    -   `GROQ_API_KEY`: Dùng cho Agent và tổng hợp câu trả lời.
    -   `GEMINI_API_KEY`: Dùng cho tiền xử lý Markdown và đánh giá RAGAS.
    -   `HF_TOKEN`: Dùng để tải model Embedding.

## ⚙️ Cài đặt

1.  **Clone repository:**
    ```bash
    git clone <repository_url>
    cd RagSystem
    ```

2.  **Khởi tạo môi trường ảo và cài đặt thư viện:**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirement.txt
    ```

3.  **Cấu hình biến môi trường:**
    Tạo file `.env` tại thư mục gốc:
    ```env
    HF_TOKEN="your_hugging_face_token"
    GROQ_API_KEY="your_groq_api_key"
    GEMINI_API_KEY="your_google_gemini_api_key"
    ```

## 📖 Hướng dẫn sử dụng

### 1. Tiền xử lý dữ liệu (PDF -> Markdown -> Index)
Đặt các file PDF quy chế vào `data/documents/` và chạy:
```bash
python -m scripts.preprocess
```
*Lưu ý: Quá trình này sử dụng AI để parse Markdown và có cơ chế dừng 15s/batch để tránh giới hạn Rate Limit của Free Tier.*

### 2. Chạy Chatbot (CLI Mode)
Giao diện dòng lệnh tương tác trực tiếp:
```bash
python run_cli_chat.py
```

### 3. Chạy Web API (FastAPI)
Khởi chạy server API cho ứng dụng:
```bash
uvicorn api.main:app --reload
```
Tài liệu API tại: `http://127.0.0.1:8000/docs`

### 4. Đánh giá hệ thống (RAGAS)
Chạy quy trình đánh giá tự động:
```bash
python -m scripts.evaluate
```

## 📂 Cấu trúc dự án

```text
├── api/              # FastAPI implementation
├── config/           # Centralized settings & keys
├── core/
│   ├── cache.py      # Semantic Caching logic
│   ├── chatbot.py    # Agentic flow (LangGraph)
│   └── retriever.py  # Hybrid search & Reranking
├── data/
│   ├── documents/    # Raw PDF files
│   └── vector_store/ # FAISS, BM25, and Cache indexes
├── scripts/
│   ├── preprocess.py # Data pipeline (PDF to Markdown)
│   └── evaluate.py   # RAGAS evaluation script
└── run_cli_chat.py   # CLI entry point
```

## 📜 Giấy phép
Dự án được phát triển phục vụ mục đích học tập và nghiên cứu tra cứu quy chế học vụ HUST.
