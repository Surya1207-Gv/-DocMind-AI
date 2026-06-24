import os
import sys
import re
import string
from dotenv import load_dotenv

# Set python path to backend to resolve imports correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import get_db_connection
from backend.embedding_manager import get_embeddings_model, FAISS_DIR
from langchain_community.vectorstores import FAISS
from typing import List
import math

class BetterBM25:
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_term_freqs = []
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "about", 
            "against", "between", "into", "through", "during", "before", "after", "above", "below", "from", 
            "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", 
            "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", 
            "very", "s", "t", "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself", 
            "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", 
            "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", 
            "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", 
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", 
            "does", "did", "doing", "would", "should", "could"
        }
        
        if self.corpus_size == 0:
            return
            
        def clean_text_to_words(text: str) -> List[str]:
            words = []
            for w in text.split():
                w_clean = w.strip(string.punctuation)
                if not w_clean:
                    continue
                # Split camelCase/PascalCase/merged words
                parts = re.findall(r'[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+', w_clean)
                if parts:
                    words.extend([p.lower() for p in parts])
                else:
                    words.append(w_clean.lower())
            return [w for w in words if w and w not in self.stop_words]
            
        total_len = 0
        for doc in corpus:
            words = clean_text_to_words(doc)
            total_len += len(words)
            self.doc_lens.append(len(words))
            
            tf = {}
            for word in words:
                tf[word] = tf.get(word, 0) + 1
            self.doc_term_freqs.append(tf)
            
        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 and total_len > 0 else 1.0
        
        # Calculate document frequency for IDF
        df = {}
        for tf in self.doc_term_freqs:
            for word in tf:
                df[word] = df.get(word, 0) + 1
                
        # Calculate standard BM25 IDF
        for word, freq in df.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            
    def get_score(self, query: str, doc_index: int) -> float:
        if self.corpus_size == 0 or doc_index >= self.corpus_size:
            return 0.0
            
        def clean_text_to_words(text: str) -> List[str]:
            words = []
            for w in text.split():
                w_clean = w.strip(string.punctuation)
                if not w_clean:
                    continue
                parts = re.findall(r'[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+', w_clean)
                if parts:
                    words.extend([p.lower() for p in parts])
                else:
                    words.append(w_clean.lower())
            return [w for w in words if w and w not in self.stop_words]
            
        query_words = clean_text_to_words(query)
        score = 0.0
        doc_len = self.doc_lens[doc_index]
        tf = self.doc_term_freqs[doc_index]
        
        for word in query_words:
            if word not in tf:
                continue
            word_tf = tf[word]
            idf = self.idf.get(word, 0.0)
            
            numerator = idf * word_tf * (self.k1 + 1)
            denominator = word_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += numerator / denominator
            
        return score

def run_test():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM documents;")
    docs = cursor.fetchall()
    doc_ids = [d['id'] for d in docs]
    
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
                
    query = "What is Generative AI?"
    candidates = main_vector_store.similarity_search_with_score(query, k=15)
    
    corpus = [doc.page_content for doc, _ in candidates]
    bm25 = BetterBM25(corpus)
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
    print(f"Subject: '{subject}'")
    
    scored_candidates = []
    for i, (doc, vector_distance) in enumerate(candidates):
        vector_sim = max(0.0, min(1.0, 1.0 - (vector_distance / 2.0)))
        bm25_normalized = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
        hybrid_score = 0.6 * vector_sim + 0.4 * bm25_normalized
        
        custom_boost = 0.0
        boost_reasons = []
        doc_text_lower = doc.page_content.lower()
        
        # 1. Definition patterns
        if is_definition_query:
            def_patterns = ["is a", "refers to", "defined as", "can be defined as", "means", "is the general term", "is a type of"]
            if any(pat in doc_text_lower for pat in def_patterns):
                custom_boost += 0.05
                boost_reasons.append("Base def patterns")
            
            if subject:
                subject_esc = re.escape(subject)
                # Regex allowing only parentheses or restricted short adverbs/fillers in between
                pattern_regex = re.compile(
                    rf"{subject_esc}\b" # Subject
                    rf"(?:\s*\([^)]*\))?" # Optional parenthetical
                    rf"(?:\s*,\s*[^,]+,\s*)?" # Optional appositive
                    rf"(?:\s*(?:sometimes|commonly|also|frequently|often|abbreviated\s+to\s+['\"\w\s.-]+|referred\s+to\s+as\s+['\"\w\s.-]+))*" # Optional adverbs
                    rf"\s+\b(is\s+a|refers\s+to|means|is\s+the\s+general\s+term|is\s+defined\s+as|can\s+be\s+defined\s+as|is\s+a\s+relatively\s+new\s+form|is\s+a\s+type\s+of)\b",
                    re.IGNORECASE
                )
                match = pattern_regex.search(doc.page_content)
                if match:
                    custom_boost += 0.45
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
        scored_candidates.append((doc, vector_distance, final_score, vector_sim, bm25_normalized, custom_boost, boost_reasons, bm25_scores[i]))
        
    scored_candidates.sort(key=lambda x: x[2], reverse=True)
    
    print("\n--- Better BM25 & New Boosting Results ---")
    for idx, (doc, dist, score, v_sim, b_norm, boost, reasons, raw_bm25) in enumerate(scored_candidates[:6]):
        print(f"Rank #{idx+1} (Page {doc.metadata.get('page')}) (Score: {score:.4f})")
        print(f"  Vector Sim: {v_sim:.4f} | BM25 Score (Raw/Norm): {raw_bm25:.4f}/{b_norm:.4f} | Boost: {boost:.4f} ({reasons})")
        print(f"  Snippet: {doc.page_content[:150].replace('\n', ' ')}")
        print("-" * 50)

if __name__ == "__main__":
    run_test()
