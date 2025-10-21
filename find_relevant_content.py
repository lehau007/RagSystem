## Load vector store

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import json

# Load environment variables
load_dotenv()

# For JSON format
def load_chunked_docs_json(filepath="DB/chunked_docs.json"):
    """Load chunked documents from JSON file"""
    from langchain_core.documents import Document
    
    with open(filepath, 'r', encoding='utf-8') as f:
        docs_data = json.load(f)
    
    chunked_docs = []
    for doc_data in docs_data:
        chunked_docs.append(Document(
            page_content=doc_data["page_content"],
            metadata=doc_data["metadata"]
        ))
    
    return chunked_docs

# Recreate the same embedder used during saving
model_id = "google/embeddinggemma-300M"
hf_token = os.getenv("HF_TOKEN")
model = SentenceTransformer(model_id, use_auth_token=hf_token)

# Create the same wrapper class
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embedder = SentenceTransformerEmbeddings(model)

# Load the saved vectorstore
vectorstore = FAISS.load_local("DB/vector_db", embedder, allow_dangerous_deserialization=True)
print(f"Loaded vectorstore with {vectorstore.index.ntotal} documents")

# Now you can use the vectorstore for similarity search
query = "your search query here"
results = vectorstore.similarity_search(query, k=5)
for result in results:
    print(f"Content: {result.page_content[:100]}...")
    print(f"Metadata: {result.metadata}")
    print("---")

def keyword_finding(keywords, chunked_docs=chunked_docs):
    """
    Find documents containing specific keywords
    
    Args:
        keywords: List of keywords to search for
        chunked_docs: List of document chunks to search in
    
    Returns:
        List of matching documents with scores
    """
    results = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        for doc in chunked_docs:
            content_lower = doc.page_content.lower()
            
            if keyword_lower in content_lower:
                # Count occurrences for scoring
                count = content_lower.count(keyword_lower)
                
                # Check if doc already in results
                existing = next((r for r in results if r['doc'] == doc), None)
                
                if existing:
                    existing['score'] += count
                    existing['matched_keywords'].append(keyword)
                else:
                    results.append({
                        'doc': doc,
                        'score': count,
                        'matched_keywords': [keyword]
                    })
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results


def similarity_finding(query, k=5, vectorstore=vectorstore):
    """
    Find documents using semantic similarity search
    
    Args:
        query: Search query string
        k: Number of results to return
        vectorstore: FAISS vector store
    
    Returns:
        List of similar documents
    """
    results = vectorstore.similarity_search(query, k=k)
    return results


def rag_tool(query, keywords=None, use_similarity=True, k=4):
    """
    Combined RAG tool that searches using both keywords and semantic similarity
    
    Args:
        query: User's search query
        keywords: Optional list of keywords for exact matching
        use_similarity: Whether to use semantic similarity search
        k: Number of results to return
    
    Returns:
        Dictionary with combined results and context
    """
    all_results = []
    
    # 1. Keyword search if keywords provided
    if keywords and len(keywords) > 0:
        keyword_results = keyword_finding(keywords)
        for item in keyword_results[:k]:
            all_results.append({
                'document': item['doc'],
                'source': 'keyword',
                'score': item['score'],
                'matched_keywords': item['matched_keywords']
            })
    
    # 2. Similarity search
    if use_similarity:
        similarity_results = similarity_finding(query, k=k)
        for doc in similarity_results:
            # Avoid duplicates
            if not any(r['document'] == doc for r in all_results):
                all_results.append({
                    'document': doc,
                    'source': 'similarity',
                    'score': None,
                    'matched_keywords': []
                })
    
    # 3. Combine and format context
    context_parts = []
    for i, result in enumerate(all_results[:k]):
        doc = result['document']
        source_info = f"[Source: {result['source']}"
        if result['matched_keywords']:
            source_info += f", Keywords: {', '.join(result['matched_keywords'])}"
        source_info += f", Page: {doc.metadata.get('page', 'N/A')}]"
        
        context_parts.append(f"{source_info}\n{doc.page_content}")
    
    return {
        'context': '\n\n---\n\n'.join(context_parts),
        'num_results': len(all_results[:k]),
        'results': all_results[:k]
    }

class DocumentFormatter:
    def __init__(self):
        
        self.vectorstore = 