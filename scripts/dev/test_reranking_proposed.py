import os
import sys
import re
import string
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
    
    # Extract subject
    subject = q_lower
    if is_definition_query:
        for prefix in ["what is", "what are", "define", "meaning of", "explain what", "describe"]:
            if subject.startswith(prefix):
                subject = subject[len(prefix):].strip()
                break
        subject = subject.strip("? .!").strip()
        
    print(f"Query: {query}")
    print(f"Is definition query: {is_definition_query}")
    print(f"Subject: '{subject}'")
    print(f"Query content words: {query_content_words}")
    
    scored_candidates = []
    for i, (doc, vector_distance) in enumerate(candidates):
        vector_sim = max(0.0, min(1.0, 1.0 - (vector_distance / 2.0)))
        bm25_normalized = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
        hybrid_score = 0.6 * vector_sim + 0.4 * bm25_normalized
        
        # Proposed custom boosting
        custom_boost = 0.0
        boost_reasons = []
        doc_text_lower = doc.page_content.lower()
        
        # 1. Definition patterns
        if is_definition_query:
            # Base definition pattern check
            def_patterns = ["is a", "refers to", "defined as", "can be defined as", "means", "is the general term", "is a type of"]
            if any(pat in doc_text_lower for pat in def_patterns):
                custom_boost += 0.05
                boost_reasons.append("Base def patterns")
            
            # Subject proximity check
            if subject:
                subject_esc = re.escape(subject)
                pattern_regex = re.compile(
                    rf"\b{subject_esc}\b(?:[^\n]{{0,50}})?\b(is a|refers to|defined as|means|is the general term|is a type of|is a relatively new form)\b",
                    re.IGNORECASE
                )
                match = pattern_regex.search(doc.page_content)
                if match:
                    custom_boost += 0.25
                    boost_reasons.append(f"Subject proximity match: '{match.group(0)}'")
                    
        # 2. Section/Header match boost
        lines = doc.page_content.split("\n")
        for line in lines:
            parts = re.split(r'[|<>]', line)
            for part in parts:
                part_strip = part.strip()
                if 2 < len(part_strip) < 40 and part_strip.isupper():
                    if any(word in part_strip.lower() for word in query_content_words):
                        custom_boost += 0.10
                        boost_reasons.append(f"Section header: '{part_strip}'")
                        break
            else:
                continue
            break
            
        final_score = min(1.0, hybrid_score + custom_boost)
        scored_candidates.append((doc, vector_distance, final_score, vector_sim, bm25_normalized, custom_boost, boost_reasons))
        
    scored_candidates.sort(key=lambda x: x[2], reverse=True)
    
    print("\n--- Proposed Reranking Results ---")
    for idx, (doc, dist, score, v_sim, b_norm, boost, reasons) in enumerate(scored_candidates[:6]):
        print(f"Rank #{idx+1} (Page {doc.metadata.get('page')}) (Score: {score:.4f})")
        print(f"  Vector Sim: {v_sim:.4f} | BM25 Norm: {b_norm:.4f} | Boost: {boost:.4f} ({reasons})")
        print(f"  Snippet: {doc.page_content[:150].replace('\n', ' ')}")
        print("-" * 50)

if __name__ == "__main__":
    run_test()
