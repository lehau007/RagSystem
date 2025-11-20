# RAG System Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on a provided set of documents. It uses a language model that can decide whether to retrieve relevant context from the documents to answer a user's query.

The project is structured to be scalable and maintainable, with a clear separation between the core logic, the API, and the data processing scripts.

## Project Structure

```
.
├── api/
│   └── main.py           # FastAPI application
├── config/
│   └── settings.py       # Configuration for paths, models, and API keys
├── core/
│   ├── chatbot.py        # Core chatbot logic
│   └── retriever.py      # Document retrieval logic
├── data/
│   ├── documents/        # Raw PDF documents
│   └── vector_store/     # Stored vector database and chunked documents
├── scripts/
│   └── preprocess.py     # Data processing and vector store creation
├── tests/
│   ├── test_api.py       # Tests for the API
│   └── test_chatbot.py   # Tests for the chatbot logic
├── .gitignore
├── README.md
├── requirements.txt
└── run_cli_chat.py       # Script for command-line interaction
```

## Getting Started

### 1. Installation

First, clone the repository and install the required dependencies from `requirements.txt`:

```bash
git clone <your-repository-url>
cd RagSystem
pip install -r requirements.txt
```

### 2. Configuration

The project uses a `.env` file to manage secret keys.

1.  Create a file named `.env` in the root of the project.
2.  Add your API keys to the `.env` file. The project requires keys for Hugging Face (`HF_TOKEN`) and an OpenAI-compatible service (`OSSAPI_KEY`).

    ```
    HF_TOKEN="your_hugging_face_token_here"
    OSSAPI_KEY="your_oss_api_key_here"
    ```

### 3. Add Documents

Place the PDF documents you want the chatbot to use in the `data/documents/` directory.

### 4. Process Documents

Before running the chatbot, you need to process the documents and create the vector store. Run the preprocessing script:

```bash
python -m scripts.preprocess
```

This script will:
- Read the documents from `data/documents/`.
- Chunk the documents into smaller pieces.
- Create embeddings for the chunks.
- Save the chunked documents and the FAISS vector store in `data/vector_store/`.

## Usage

You can interact with the chatbot in two ways:

### 1. Command-Line Interface (CLI)

To chat with the bot directly in your terminal, run:

```bash
python run_cli_chat.py
```

### 2. FastAPI Application

To run the application as a web service, use Uvicorn:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can send a POST request to the `/chat` endpoint to interact with the chatbot.

You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Running Tests

The project includes a suite of tests to ensure the chatbot and API are working correctly. To run the tests, use `pytest`:

```bash
pytest
```

This will automatically discover and run the tests in the `tests/` directory.