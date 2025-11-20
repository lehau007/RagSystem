import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from config.settings import DOCUMENTS_PATH, CHUNKED_DOCS_PATH, FAISS_INDEX_PATH, EMBEDDING_MODEL, HF_TOKEN
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Load all documents from the documents directory
docs = []
for doc_file in os.listdir(DOCUMENTS_PATH):
    if doc_file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DOCUMENTS_PATH, doc_file))
        docs.extend(loader.load())
print(f"Loaded {len(docs)} pages from {len(os.listdir(DOCUMENTS_PATH))} documents")


"""" Format text """
def format_string(text):
    """
    Remove excessive spaces and newlines while preserving paragraph breaks.
    """
    import re
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Replace multiple consecutive spaces with single space
    result = re.sub(r' {2,}', ' ', text)
    
    # Replace 3+ newlines with double newline (preserve paragraphs)
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result

for i in range(len(docs)):
    docs[i].page_content = format_string(docs[i].page_content)
print(len(docs), "Formatted")

position_to_metadata = []
full_text = ""
current_position = 0

for doc in docs:
    page_content = doc.page_content
    page_length = len(page_content)
    
    position_to_metadata.append({
        "start": current_position,
        "end": current_position + page_length,
        "metadata": doc.metadata
    })
    
    full_text += page_content + "\n"
    current_position += page_length + 1  # +1 for the newline

header_pattern = re.compile(
    r"ABCXYZGHL\s+\d+"
)

def get_page_metadata(position):
    """Get the original page metadata for a given text position"""
    for mapping in position_to_metadata:
        if mapping["start"] <= position < mapping["end"]:
            return mapping["metadata"]
    return position_to_metadata[-1]["metadata"]  # Default to last page

structured_docs = []
headers = list(header_pattern.finditer(full_text))

if not headers:
    structured_docs.append(
        Document(
            page_content=full_text, 
            metadata={
                **docs[0].metadata,
                "header": "No Header",
                "page_range": f"1-{len(docs)}"
            }
        )
    )
else:
    for i, match in enumerate(headers):
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(full_text)
        chunk_text = full_text[start:end].strip()
        
        # Get metadata from the start position
        start_metadata = get_page_metadata(start)
        end_metadata = get_page_metadata(end - 1)
        
        # Determine page range
        start_page = start_metadata.get("page", 0)
        end_page = end_metadata.get("page", 0)
        page_range = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
        
        structured_docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    **start_metadata,  # Keep original metadata
                    "header": match.group(0),
                    "page_range": page_range,
                    "start_page": start_page,
                    "end_page": end_page
                }
            )
        )

print(f"Total structured docs: {len(structured_docs)}")


""" Split into chunks """
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_docs = splitter.split_documents(structured_docs)

def save_chunked_docs(chunked_docs, filepath=CHUNKED_DOCS_PATH):
    """Save chunked documents to JSON file"""
    docs_data = []
    for doc in chunked_docs:
        docs_data.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(docs_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(chunked_docs)} chunks to {filepath}")

# Save the chunked documents
save_chunked_docs(chunked_docs)

print(f"Final chunks: {len(chunked_docs)}")


""" Embedding by FAISS """
model = SentenceTransformer(EMBEDDING_MODEL, use_auth_token=HF_TOKEN)
# Create a wrapper so it matches LangChain's expected API
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embedder = SentenceTransformerEmbeddings(model)

vectorstore = FAISS.from_documents(chunked_docs, embedder)
print("Vector DB created with", vectorstore.index.ntotal, "documents")

# Create DB directory if it doesn't exist
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
vectorstore.save_local(FAISS_INDEX_PATH)
print(f"Vector database saved to {FAISS_INDEX_PATH}")