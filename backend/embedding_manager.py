import os
import shutil
import re
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
import requests
from langchain_core.embeddings import Embeddings
from backend.config import OPENROUTER_API_KEY, EMBEDDING_MODEL, FAISS_DIR

import time

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/embeddings"
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]
        
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def get_embeddings_model():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it in your .env file.")
    return OpenRouterEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY
    )

def create_and_save_index(chunks: List[Dict[str, Any]], doc_id: str) -> str:
    """
    Creates a FAISS index from document chunks and saves it locally.
    Uses batching and sleep intervals to respect Google Free Tier 429 rate limits.
    """
    if not chunks:
        return ""
    
    embeddings = get_embeddings_model()
    
    # Google AI Studio Free Tier has a rate limit of 100 requests per minute for embeddings.
    # We will process them in batches of 40 to minimize API requests and ensure reliability.
    BATCH_SIZE = 40
    
    # Initialize FAISS with first batch
    first_batch = chunks[:BATCH_SIZE]
    first_texts = [c["text"] for c in first_batch]
    first_metadatas = [c["metadata"] for c in first_batch]
    
    vector_store = FAISS.from_texts(texts=first_texts, embedding=embeddings, metadatas=first_metadatas)
    
    # Add subsequent batches with a short cooldown sleep
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        time.sleep(1.5) # 1.5s sleep = max 40 requests/min, well under the 100 RPM limit
        batch = chunks[i:i + BATCH_SIZE]
        batch_texts = [c["text"] for c in batch]
        batch_metadatas = [c["metadata"] for c in batch]
        vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)
    
    doc_index_path = os.path.join(FAISS_DIR, doc_id)
    vector_store.save_local(doc_index_path)
    
    return doc_index_path

def delete_index(doc_id: str):
    """
    Deletes the FAISS index files for a document.
    """
    doc_index_path = os.path.join(FAISS_DIR, doc_id)
    if os.path.exists(doc_index_path):
        shutil.rmtree(doc_index_path)

import math

import string

class SimpleBM25:
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
            
            # BM25 formula
            numerator = idf * word_tf * (self.k1 + 1)
            denominator = word_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += numerator / denominator
            
        return score

def search_index(query: str, doc_ids: List[str], top_k: int = 4) -> List[Tuple[Any, float]]:
    """
    Searches across specified doc_ids by loading and merging their FAISS indices.
    Retrieves candidates, re-ranks them using BM25 hybrid search, and returns top results.
    """
    embeddings = get_embeddings_model()
    
    # Identify which doc IDs to load
    target_ids = doc_ids
    if not target_ids:
        # Load all folders in FAISS_DIR
        if os.path.exists(FAISS_DIR):
            target_ids = [d for d in os.listdir(FAISS_DIR) if os.path.isdir(os.path.join(FAISS_DIR, d))]
            
    if not target_ids:
        return []
        
    main_vector_store = None
    
    for doc_id in target_ids:
        path = os.path.join(FAISS_DIR, doc_id)
        if os.path.exists(path):
            try:
                # FAISS load requires allow_dangerous_deserialization since it uses pickle
                db = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
                if main_vector_store is None:
                    main_vector_store = db
                else:
                    main_vector_store.merge_from(db)
            except Exception as e:
                print(f"Error loading index for {doc_id}: {e}")
                
    if main_vector_store is None:
        return []
        
    # Retrieve a larger set of candidates for re-ranking (e.g., max of 3x top_k and 15)
    candidate_count = max(top_k * 3, 15)
    candidates = main_vector_store.similarity_search_with_score(query, k=candidate_count)
    
    if not candidates:
        return []
        
    # Re-ranking using BM25
    corpus = [doc.page_content for doc, _ in candidates]
    bm25 = SimpleBM25(corpus)
    
    # Compute BM25 scores
    bm25_scores = [bm25.get_score(query, i) for i in range(len(candidates))]
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    
    q_lower = query.lower().strip()
    is_definition_query = q_lower.startswith(("what is", "what are", "define", "meaning of", "explain what", "describe"))
    
    # Extract query content words for title/section matching (ignore short words and stop words)
    query_content_words = [w for w in q_lower.split() if len(w) > 3 and w not in ["what", "with", "from", "that"]]

    # Extract subject for definition-query proximity matching
    subject = q_lower
    if is_definition_query:
        for prefix in ["what is", "what are", "define", "meaning of", "explain what", "describe"]:
            if subject.startswith(prefix):
                subject = subject[len(prefix):].strip()
                break
        subject = subject.strip("? .!").strip()

    scored_candidates = []
    for i, (doc, vector_distance) in enumerate(candidates):
        # Convert vector distance to a normalized 0-1 similarity score
        # FAISS uses L2 distance. score 0 means identical, >=2 means very distant.
        vector_sim = max(0.0, min(1.0, 1.0 - (vector_distance / 2.0)))
        
        # Normalize BM25 score
        bm25_normalized = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
        
        # Compute base hybrid score (0.6 weight on vector search, 0.4 weight on keyword match)
        hybrid_score = 0.6 * vector_sim + 0.4 * bm25_normalized
        
        # Custom boosting
        custom_boost = 0.0
        doc_text_lower = doc.page_content.lower()
        
        # 1. Definition pattern boost
        if is_definition_query:
            # Base definition pattern check
            def_patterns = ["is a", "refers to", "defined as", "can be defined as", "means", "is the general term", "is a type of"]
            if any(pat in doc_text_lower for pat in def_patterns):
                custom_boost += 0.05
            
            # Precise subject definition proximity check
            if subject:
                subject_esc = re.escape(subject)
                pattern_regex = re.compile(
                    rf"{subject_esc}\b"
                    rf"(?:\s*\([^)]*\))?"
                    rf"(?:\s*,\s*[^,]+,\s*)?"
                    rf"(?:\s*(?:sometimes|commonly|also|frequently|often|abbreviated\s+to\s+['\"\w\s.-]+|referred\s+to\s+as\s+['\"\w\s.-]+))*"
                    rf"\s+\b(is\s+a|refers\s+to|means|is\s+the\s+general\s+term|is\s+defined\s+as|can\s+be\s+defined\s+as|is\s+a\s+relatively\s+new\s+form|is\s+a\s+type\s+of)\b",
                    re.IGNORECASE
                )
                if pattern_regex.search(doc.page_content):
                    custom_boost += 0.45
                
        # 2. Section/Header match boost
        # Look for short uppercase headers in the chunk text matching query content words
        lines = doc.page_content.split("\n")
        for line in lines:
            parts = re.split(r'[|<>]', line)
            for part in parts:
                part_strip = part.strip()
                if 2 < len(part_strip) < 40 and part_strip.isupper():
                    if any(word in part_strip.lower() for word in query_content_words):
                        custom_boost += 0.10
                        break
            else:
                continue
            break
                    
        # Apply boost and cap at 1.0
        hybrid_score = min(1.0, hybrid_score + custom_boost)
        
        scored_candidates.append((doc, vector_distance, hybrid_score))
        
    # Sort candidates by combined hybrid score descending
    scored_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Log information about re-ranking
    if scored_candidates:
        orig_top = candidates[0][0].page_content[:50].replace("\n", " ")
        new_top = scored_candidates[0][0].page_content[:50].replace("\n", " ")
        print(f"[Reranking] Re-evaluated {len(candidates)} candidates. Orig top: '{orig_top}', New top: '{new_top}'")
        
    # Return formatted list matching (Document, score) where score is translated back 
    # to simulated L2 distance matching the hybrid score (for downstream confidence scores)
    # distance = 2.0 * (1.0 - hybrid_score)
    # Filter out low-relevance references below 50% Match
    final_results = []
    for doc, _, hybrid_score in scored_candidates[:top_k]:
        if hybrid_score < 0.50:
            continue
        simulated_distance = 2.0 * (1.0 - hybrid_score)
        final_results.append((doc, simulated_distance))
        
    return final_results

