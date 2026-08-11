import os
import sys
from dotenv import load_dotenv

# Set python path to backend to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_connection
from backend.embedding_manager import get_embeddings_model, FAISS_DIR
from langchain_community.vectorstores import FAISS

def check_pages():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM documents;")
    docs = cursor.fetchall()
    doc_ids = [d['id'] for d in docs]
    
    embeddings = get_embeddings_model()
    for doc_id in doc_ids:
        path = os.path.join(FAISS_DIR, doc_id)
        if os.path.exists(path):
            db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            # Retrieve all documents stored in FAISS
            # FAISS doesn't have a direct "list all" but we can access it via index or docstore
            docs_in_store = list(db.docstore._dict.values())
            print(f"Total chunks in FAISS store: {len(docs_in_store)}")
            for idx, doc in enumerate(docs_in_store):
                page = doc.metadata.get("page")
                if page in [24, 36]:
                    print(f"\n--- Chunk {idx} | Page {page} ---")
                    print(doc.page_content)
                    print("="*60)

if __name__ == "__main__":
    check_pages()
