import os
import sys
import string
import math
from dotenv import load_dotenv

# Set python path to backend to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_connection
from backend.embedding_manager import get_embeddings_model, SimpleBM25, FAISS_DIR
from langchain_community.vectorstores import FAISS

def run_test():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM documents;")
    docs = cursor.fetchall()
    doc_ids = [d['id'] for d in docs]
    
    if not doc_ids:
        print("No documents found.")
        return
        
    embeddings = get_embeddings_model()
    main_vector_store = None
    for doc_id in doc_ids:
        path = os.path.join(FAISS_DIR, doc_id)
        if os.path.exists(path):
            db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            if main_vector_store is None:
                main_vector_store = db
            else:
                main_vector_store.merge_from(db)
                
    if main_vector_store is None:
        print("No index loaded.")
        return
        
    query = "What is Generative AI?"
    candidates = main_vector_store.similarity_search_with_score(query, k=15)
    
    corpus = [doc.page_content for doc, _ in candidates]
    bm25 = SimpleBM25(corpus)
    bm25_scores = [bm25.get_score(query, i) for i in range(len(candidates))]
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    
    q_lower = query.lower().strip()
    is_definition_query = q_lower.startswith(("what is", "what are", "define", "meaning of", "explain what", "describe"))
    query_content_words = [w for w in q_lower.split() if len(w) > 3 and w not in ["what", "with", "from", "that"]]
    
    print(f"Query: {query}")
    print(f"Is definition query: {is_definition_query}")
    print(f"Query content words: {query_content_words}")
    
    for i, (doc, vector_distance) in enumerate(candidates):
        vector_sim = max(0.0, min(1.0, 1.0 - (vector_distance / 2.0)))
        bm25_normalized = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
        hybrid_score = 0.6 * vector_sim + 0.4 * bm25_normalized
        
        # Boost calculation
        boost_reasons = []
        custom_boost = 0.0
        doc_text_lower = doc.page_content.lower()
        
        # 1. Definition pattern boost
        if is_definition_query:
            def_patterns = ["is a", "refers to", "defined as", "can be defined as", "means", "is the general term", "is a type of"]
            matched_patterns = [pat for pat in def_patterns if pat in doc_text_lower]
            if matched_patterns:
                custom_boost += 0.15
                boost_reasons.append(f"Definition patterns: {matched_patterns}")
                
        # 2. Section/Header match boost
        lines = doc.page_content.split("\n")
        matched_headers = []
        for line in lines:
            line_strip = line.strip()
            if len(line_strip) < 60 and line_strip.isupper():
                matched_words = [word for word in query_content_words if word in line_strip.lower()]
                if matched_words:
                    custom_boost += 0.10
                    matched_headers.append(f"'{line_strip}' matched {matched_words}")
                    break
        if matched_headers:
            boost_reasons.append(f"Section headers: {matched_headers}")
            
        final_score = min(1.0, hybrid_score + custom_boost)
        
        print(f"\nCandidate {i+1} (Page {doc.metadata.get('page')}) (Snippet: {doc.page_content[:60].replace('\n', ' ')})")
        print(f"  Vector Dist: {vector_distance:.4f} | Vector Sim: {vector_sim:.4f}")
        print(f"  BM25 Score: {bm25_scores[i]:.4f} | BM25 Norm: {bm25_normalized:.4f}")
        print(f"  Base Hybrid Score: {hybrid_score:.4f}")
        print(f"  Custom Boost: {custom_boost:.4f} (Reasons: {boost_reasons})")
        print(f"  Final Score: {final_score:.4f}")

if __name__ == "__main__":
    run_test()
