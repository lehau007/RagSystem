import os
import json
import re
import pickle
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_experimental.text_splitter import SemanticChunker
from rank_bm25 import BM25Okapi
from google import genai

from config.settings import (
    DOCUMENTS_PATH, CHUNKED_DOCS_PATH, FAISS_INDEX_PATH, 
    BM25_INDEX_PATH, EMBEDDING_MODEL, HF_TOKEN, GEMINI_API_KEY
)

# 1. Hàm đọc và chuẩn hóa cơ bản
def load_and_clean_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        # Xoá khoảng trắng thừa, chuẩn hóa xuống dòng
        text = doc.page_content.replace('\r\n', '\n').replace('\r', '\n')
        doc.page_content = re.sub(r' {2,}', ' ', text)
    return docs

# 2. Hàm gọi AI để chuyển đổi sang Markdown (Batching)
def convert_to_markdown_with_ai(pages, batch_size=3):
    client = genai.Client(api_key=GEMINI_API_KEY)
    full_markdown = ""
    
    system_instruction = (
        "Bạn là một trợ lý chuyên xử lý văn bản quy chế học vụ. "
        "Hãy chuyển đổi phần văn bản PDF thô sau đây thành định dạng Markdown chuẩn xác. "
        "Giữ nguyên cấu trúc các Chương, Điều, khoản, điểm, danh sách và bảng biểu. "
        "Tuyệt đối không thêm lời chào, không bình luận, chỉ trả về nội dung Markdown."
    )

    print(f"Bắt đầu chuyển đổi {len(pages)} trang sang Markdown (Batch size: {batch_size})...")
    for i in range(0, len(pages), batch_size):
        batch_pages = pages[i:i+batch_size]
        batch_text = "\n".join([p.page_content for p in batch_pages])
        
        print(f"Đang xử lý batch trang {i+1} đến {min(i+batch_size, len(pages))}...")
        try:
            response = client.models.generate_content(
                model='gemma-3-27b-it', # Sử dụng đúng model gemma-3 như yêu cầu
                contents=f"{system_instruction}\n\nNội dung thô:\n{batch_text}"
            )
            full_markdown += response.text + "\n\n"
            time.sleep(15) # Nghỉ 15 giây để tránh lỗi 429 trên Free Tier
        except Exception as e:
            print(f"Lỗi khi xử lý batch {i+1}-{min(i+batch_size, len(pages))}: {e}")
            # Fallback: Nếu lỗi, lưu text thô của batch đó
            full_markdown += batch_text + "\n\n"
            
    return full_markdown

# Embeddings Wrapper
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

if __name__ == "__main__":
    print("--- BẮT ĐẦU PIPELINE TIỀN XỬ LÝ (MODEL: GEMMA-3) ---")
    
    # A. Load Documents
    all_markdown_docs = []
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"Lỗi: Thư mục {DOCUMENTS_PATH} không tồn tại.")
    else:
        pdf_files = [f for f in os.listdir(DOCUMENTS_PATH) if f.endswith(".pdf")]
        if not pdf_files:
            print("Không tìm thấy file PDF nào trong thư mục documents.")
        else:
            for doc_file in pdf_files:
                file_path = os.path.join(DOCUMENTS_PATH, doc_file)
                print(f"Đang đọc file: {doc_file}")
                raw_pages = load_and_clean_pdf(file_path)
                
                # B. AI-driven Markdown Conversion
                markdown_content = convert_to_markdown_with_ai(raw_pages)
                
                # Đóng gói thành Document
                all_markdown_docs.append(Document(
                    page_content=markdown_content,
                    metadata={"source": doc_file}
                ))

    if not all_markdown_docs:
        print("Không có dữ liệu để xử lý.")
    else:
        # C. Semantic Chunking
        print("\nKhởi tạo model Embedding cho Semantic Chunking...")
        model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
        embedder = SentenceTransformerEmbeddings(model)
        
        print("Đang cắt văn bản theo ngữ nghĩa (Semantic Chunking)...")
        text_splitter = SemanticChunker(embedder)
        chunked_docs = text_splitter.split_documents(all_markdown_docs)
        print(f"Đã tạo ra {len(chunked_docs)} chunks ngữ nghĩa.")

        # Lưu chunked_docs ra JSON
        docs_data = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunked_docs]
        os.makedirs(os.path.dirname(CHUNKED_DOCS_PATH), exist_ok=True)
        with open(CHUNKED_DOCS_PATH, 'w', encoding='utf-8') as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

        # D. Tạo FAISS Index (Vector Search)
        print("\nĐang tạo FAISS Vector Index...")
        vectorstore = FAISS.from_documents(chunked_docs, embedder)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"Đã lưu FAISS tại: {FAISS_INDEX_PATH}")

        # E. Tạo BM25 Index (Keyword Search)
        print("\nĐang tạo BM25 Keyword Index...")
        # Simple tokenization for BM25
        tokenized_corpus = [doc.page_content.lower().split() for doc in chunked_docs]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_INDEX_PATH, 'wb') as f:
            pickle.dump(bm25, f)
        print(f"Đã lưu BM25 tại: {BM25_INDEX_PATH}")
        
        print("--- HOÀN TẤT PIPELINE ---")
