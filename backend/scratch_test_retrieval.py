import os
import sys
from dotenv import load_dotenv

# Set python path to backend to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_connection
from backend.embedding_manager import search_index

def run_test():
    # 1. Fetch documents from db
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM documents;")
    docs = cursor.fetchall()
    print("Documents in DB:")
    doc_ids = []
    for d in docs:
        print(f"ID: {d['id']} | Name: {d['name']}")
        doc_ids.append(d['id'])
    
    if not doc_ids:
        print("No documents found in DB. Let's search FAISS directory directly.")
        from backend.config import FAISS_DIR
        if os.path.exists(FAISS_DIR):
            doc_ids = [d for d in os.listdir(FAISS_DIR) if os.path.isdir(os.path.join(FAISS_DIR, d))]
            print(f"FAISS folders found: {doc_ids}")
            
    if not doc_ids:
        print("No indexes found.")
        return
        
    query = "What is Generative AI?"
    print(f"\n--- Running search_index for query: '{query}' ---")
    results = search_index(query, doc_ids, top_k=4)
    for i, (doc, score) in enumerate(results):
        relevance = max(0.0, min(1.0, 1.0 - (score / 2.0)))
        print(f"\nResult #{i+1}:")
        print(f"Page: {doc.metadata.get('page')}")
        print(f"Relevance: {relevance * 100:.1f}%")
        print(f"Content snippet: {doc.page_content[:300]}...")
        print("-" * 50)

if __name__ == "__main__":
    run_test()
