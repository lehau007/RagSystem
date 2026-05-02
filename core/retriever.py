import os
import pickle
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from config.settings import (
    CHUNKED_DOCS_PATH, FAISS_INDEX_PATH, BM25_INDEX_PATH, 
    EMBEDDING_MODEL, HF_TOKEN
)

# Embeddings Wrapper (Giữ nguyên từ preprocess)
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

class HybridRetriever:
    def __init__(self, hf_token: str):
        # 1. Load Embedding Model & FAISS
        self.model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=hf_token)
        self.embedder = SentenceTransformerEmbeddings(self.model)
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, 
            self.embedder, 
            allow_dangerous_deserialization=True
        )
        
        # 2. Load BM25 Index
        with open(BM25_INDEX_PATH, 'rb') as f:
            self.bm25 = pickle.load(f)
            
        # 3. Load Original Chunks (để map kết quả BM25)
        with open(CHUNKED_DOCS_PATH, 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
            self.chunked_docs = [
                Document(page_content=d["page_content"], metadata=d["metadata"]) 
                for d in docs_data
            ]
            
        # 4. Load Reranker Model (Cross-Encoder)
        # Sử dụng model đa ngữ hiệu quả cho reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, top_k: int = 10, rerank_top_n: int = 5):
        """
        Luồng: Hybrid Search (FAISS + BM25) -> RRF Fusion -> Reranking
        """
        # A. Vector Search (Dense)
        vector_results = self.vectorstore.similarity_search_with_score(query, k=top_k * 2)
        
        # B. BM25 Search (Sparse)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # C. Fusion (Reciprocal Rank Fusion - đơn giản hóa)
        # Tạo dictionary để lưu score tổng hợp
        combined_results = {}
        
        # Thêm điểm từ Vector Search
        for i, (doc, score) in enumerate(vector_results):
            doc_id = doc.page_content
            combined_results[doc_id] = {"doc": doc, "score": 1 / (i + 60)}
            
        # Thêm/Cộng điểm từ BM25
        for i, idx in enumerate(top_bm25_indices):
            doc = self.chunked_docs[idx]
            doc_id = doc.page_content
            if doc_id in combined_results:
                combined_results[doc_id]["score"] += 1 / (i + 60)
            else:
                combined_results[doc_id] = {"doc": doc, "score": 1 / (i + 60)}
                
        # Sắp xếp lại theo Fusion Score
        sorted_results = sorted(
            combined_results.values(), 
            key=lambda x: x["score"], 
            reverse=True
        )[:top_k]
        
        final_docs = [item["doc"] for item in sorted_results]
        
        # D. Reranking (Cross-Encoder)
        if rerank_top_n > 0 and final_docs:
            print(f"--- Đang Reranking {len(final_docs)} kết quả ---")
            pairs = [[query, doc.page_content] for doc in final_docs]
            rerank_scores = self.reranker.predict(pairs)
            
            # Sắp xếp lại final_docs dựa trên rerank_scores
            reranked_indices = np.argsort(rerank_scores)[::-1]
            final_docs = [final_docs[i] for i in reranked_indices[:rerank_top_n]]
            
        return final_docs

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    retriever = HybridRetriever(token)
    
    query = "Điều kiện để đăng ký học vượt là gì?"
    results = retriever.retrieve(query)
    
    print(f"\nKết quả cho truy vấn: '{query}'")
    for i, doc in enumerate(results):
        print(f"\n[{i+1}] Nguồn: {doc.metadata.get('source')}")
        print(f"Nội dung: {doc.page_content[:200]}...")
