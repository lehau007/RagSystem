import os
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from config.settings import VECTOR_STORE_PATH, EMBEDDING_MODEL, HF_TOKEN

CACHE_PATH = os.path.join(VECTOR_STORE_PATH, "semantic_cache")

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        """
        threshold: Độ tương đồng tối thiểu (0-1) để coi là trùng khớp cache.
        """
        self.model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
        self.embedder = SentenceTransformerEmbeddings(self.model)
        self.threshold = threshold
        self.cache_index = self._load_or_create_cache()

    def _load_or_create_cache(self):
        if os.path.exists(CACHE_PATH):
            return FAISS.load_local(CACHE_PATH, self.embedder, allow_dangerous_deserialization=True)
        else:
            # Tạo index trống với một document mồi (FAISS yêu cầu ít nhất 1 doc để khởi tạo/lưu)
            init_doc = Document(page_content="init", metadata={"response": ""})
            index = FAISS.from_documents([init_doc], self.embedder)
            return index

    def get(self, query: str):
        """Tìm kiếm câu trả lời trong cache"""
        if not self.cache_index:
            return None
        
        # Tìm kiếm 1 kết quả gần nhất kèm score (L2 distance trong FAISS, thấp là gần)
        # Lưu ý: similarity_search_with_score trả về distance. Cần chuyển đổi hoặc dùng threshold phù hợp.
        results = self.cache_index.similarity_search_with_score(query, k=1)
        
        if results:
            doc, score = results[0]
            # Với FAISS L2 distance, score càng nhỏ càng giống. 
            # Một ngưỡng an toàn cho semantic match thường là score < 0.1 - 0.2 tùy model.
            if score < 0.15 and doc.page_content != "init":
                print(f"--- Cache Hit (Score: {score:.4f}) ---")
                return doc.metadata.get("response")
        
        return None

    def update(self, query: str, response: str):
        """Cập nhật câu hỏi và câu trả lời mới vào cache"""
        new_doc = Document(page_content=query, metadata={"response": response})
        self.cache_index.add_documents([new_doc])
        self.cache_index.save_local(CACHE_PATH)
        print("--- Đã cập nhật Semantic Cache ---")
