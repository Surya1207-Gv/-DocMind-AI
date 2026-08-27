import os
import shutil
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores import FAISS
import requests
from langchain_core.embeddings import Embeddings
from backend.config import (
    OPENROUTER_API_KEY,
    EMBEDDING_MODEL,
    FAISS_DIR,
    RELEVANCE_THRESHOLD,
    VECTOR_WEIGHT,
)
from backend.logger import get_logger, log_rag_retrieval_event
from backend.reranker import rerank
from backend.text_utils import STOP_WORDS, tokenize

logger = get_logger(__name__)



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
        # Retry with exponential backoff for transient API failures
        for attempt in range(3):
            try:
                response = requests.post(self.url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise RuntimeError("Embedding API timed out after 3 attempts")
                time.sleep(2 ** attempt)  # 1s, 2s backoff
            except requests.exceptions.RequestException as e:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        
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
    Uses batching to send multiple texts per API request.
    """
    if not chunks:
        return ""
    
    embeddings = get_embeddings_model()
    
    # Process up to 500 chunks in a single API call to minimize requests and maximize concurrency
    BATCH_SIZE = 500
    
    # Initialize FAISS with first batch
    first_batch = chunks[:BATCH_SIZE]
    first_texts = [c["text"] for c in first_batch]
    first_metadatas = [c["metadata"] for c in first_batch]
    
    vector_store = FAISS.from_texts(texts=first_texts, embedding=embeddings, metadatas=first_metadatas)
    
    # Add subsequent batches
    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
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

# Module-level BM25 cache to avoid rebuilding on every query
# Key: hash of corpus content tuples → Value: SimpleBM25 instance
_bm25_cache: Dict[int, "SimpleBM25"] = {}

BM25_CACHE_MAX_SIZE = 50  # Evict oldest when cache grows too large

class SimpleBM25:
    def __init__(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        self.avg_doc_len = 0.0
        self.doc_lens = []
        self.doc_term_freqs = []
        self.idf = {}
        self.k1 = 1.5
        self.b = 0.75
        
        # Shared with claim verification (backend/text_utils.py) so retrieval and
        # grounding agree on what a "term" is. The previous private tokenizer
        # matched only ASCII letters and digits, so every Devanagari and Telugu
        # token was discarded and BM25 scored 0.0 for any non-Latin query --
        # hybrid search silently degraded to vector-only for those languages.
        self.stop_words = STOP_WORDS

        if self.corpus_size == 0:
            return

        total_len = 0
        for doc in corpus:
            words = tokenize(doc)
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
            
        query_words = tokenize(query)
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

def search_index(
    query: str,
    doc_ids: Optional[List[str]],
    top_k: int = 4,
    use_vector: bool = True,
    use_bm25: bool = True,
    use_boosts: bool = True,
    boost_def_pattern: bool = True,
    boost_proximity: bool = True,
    boost_header: bool = True,
    relevance_threshold: float = None,
    use_rerank: bool = True,
    rerank_llm: Any = None,
    trace: Dict[str, Any] = None,
) -> List[Tuple[Any, float]]:
    """
    Searches across specified doc_ids by loading and merging their FAISS indices.
    Retrieves candidates, re-ranks them using BM25 hybrid search, and returns top results.
    Optional flags allow empirical IR evaluation and ablation sweeps without altering production defaults.
    """
    if relevance_threshold is None:
        relevance_threshold = RELEVANCE_THRESHOLD

    embeddings = get_embeddings_model()
    
    # Identify which doc IDs to load.
    #
    # An empty list means "no documents selected" and returns nothing. Only an
    # explicit None means "every index on disk" -- a facility kept for offline
    # benchmarking scripts, never reachable from a request. The two used to be
    # the same case, so an authenticated chat with no document selected fell
    # through to loading every user's indices and answering from them. Defence
    # in depth: the API layer also resolves an empty selection to the caller's
    # own documents before it gets here.
    if doc_ids is None:
        target_ids = (
            [d for d in os.listdir(FAISS_DIR) if os.path.isdir(os.path.join(FAISS_DIR, d))]
            if os.path.exists(FAISS_DIR)
            else []
        )
    else:
        target_ids = list(doc_ids)

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
                logger.error("Error loading index for %s: %s", doc_id, e)
                
    if main_vector_store is None:
        return []
        
    # Retrieve a larger set of candidates for re-ranking (e.g., max of 3x top_k and 15)
    t_vec_start = time.perf_counter()
    candidate_count = max(top_k * 3, 15)
    candidates = main_vector_store.similarity_search_with_score(query, k=candidate_count)
    vector_ms = (time.perf_counter() - t_vec_start) * 1000.0
    
    if not candidates:
        log_rag_retrieval_event(query, 0, 0, 0.0, vector_ms, 0.0, None)
        return []
        
    # Re-ranking using BM25 (use cached index if corpus already seen)
    t_bm25_start = time.perf_counter()
    corpus = [doc.page_content for doc, _ in candidates]
    corpus_hash = hash(tuple(corpus))
    if corpus_hash not in _bm25_cache:
        if len(_bm25_cache) >= BM25_CACHE_MAX_SIZE:
            # Evict the oldest entry (first key in dict)
            oldest = next(iter(_bm25_cache))
            del _bm25_cache[oldest]
        _bm25_cache[corpus_hash] = SimpleBM25(corpus)
    bm25 = _bm25_cache[corpus_hash]
    
    # Compute BM25 scores
    bm25_scores = [bm25.get_score(query, i) for i in range(len(candidates))]
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0
    bm25_ms = (time.perf_counter() - t_bm25_start) * 1000.0

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
    applied_boost_types = []
    for i, (doc, vector_distance) in enumerate(candidates):
        # Convert vector distance to a normalized 0-1 similarity score
        # FAISS uses L2 distance. score 0 means identical, >=2 means very distant.
        vector_sim = max(0.0, min(1.0, 1.0 - (vector_distance / 2.0)))
        
        # Normalize BM25 score
        bm25_normalized = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
        
        # Compute base score depending on active retrieval components
        if use_vector and use_bm25 and max_bm25 <= 0.0:
            # BM25 matched nothing anywhere in the candidate set, so it has no
            # opinion to contribute. Still blending it in would multiply every
            # score by VECTOR_WEIGHT (0.6) uniformly -- not a re-ranking, just a
            # flat penalty that pushes genuine matches under the relevance
            # threshold. It bites hardest cross-lingually: a Hindi query against
            # an English document shares no surface forms by construction, so a
            # correct 0.925 vector match was being scored 0.555 and then
            # discarded. When the lexical stage abstains, defer to the vector.
            hybrid_score = vector_sim
        elif use_vector and use_bm25:
            hybrid_score = VECTOR_WEIGHT * vector_sim + (1.0 - VECTOR_WEIGHT) * bm25_normalized
        elif use_vector and not use_bm25:
            hybrid_score = vector_sim
        elif not use_vector and use_bm25:
            hybrid_score = bm25_normalized
        else:
            hybrid_score = 0.0
        
        # Custom boosting
        custom_boost = 0.0
        doc_text_lower = doc.page_content.lower()
        
        if use_boosts:
            # 1. Definition pattern boost
            if is_definition_query and boost_def_pattern:
                def_patterns = ["is a", "refers to", "defined as", "can be defined as", "means", "is the general term", "is a type of"]
                if any(pat in doc_text_lower for pat in def_patterns):
                    custom_boost += 0.05
                    applied_boost_types.append("def_pattern")
                
                # Precise subject definition proximity check
                if subject and boost_proximity:
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
                        applied_boost_types.append("proximity")
                    
            # 2. Section/Header match boost
            if boost_header:
                lines = doc.page_content.split("\n")
                for line in lines:
                    parts = re.split(r'[|<>]', line)
                    for part in parts:
                        part_strip = part.strip()
                        if 2 < len(part_strip) < 40 and part_strip.isupper():
                            if any(word in part_strip.lower() for word in query_content_words):
                                custom_boost += 0.10
                                applied_boost_types.append("header")
                                break
                    else:
                        continue
                    break
                        
        # Apply boost and cap at 1.0
        hybrid_score = min(1.0, hybrid_score + custom_boost)
        
        scored_candidates.append((doc, vector_distance, hybrid_score))
        
    # Sort candidates by combined hybrid score descending
    scored_candidates.sort(key=lambda x: x[2], reverse=True)

    # --- Second-stage reranking -------------------------------------------
    # Fusion above is a recall stage: it decides which ~15 passages are worth
    # considering at all. Reranking is a precision stage over that shortlist,
    # scoring each passage on whether it actually answers the question rather
    # than on how topically close it is. See backend/reranker.py for why this
    # is lexical rather than a cross-encoder.
    t_rerank_start = time.perf_counter()
    reranked = []
    if use_rerank:
        reranked = rerank(
            query,
            [(doc, hybrid) for doc, _dist, hybrid in scored_candidates],
            llm=rerank_llm,
        )
        scored_candidates = [
            (c.document, 2.0 * (1.0 - c.final_score), c.final_score)
            for c in reranked
        ]
    rerank_ms = (time.perf_counter() - t_rerank_start) * 1000.0
    
    # Return formatted list matching (Document, score) where score is translated back 
    # to simulated L2 distance matching the hybrid score (for downstream confidence scores)
    # Filter out low-relevance references below relevance threshold
    final_results = []
    seen_chunks = set()
    for doc, _, hybrid_score in scored_candidates[:top_k]:
        if hybrid_score < relevance_threshold:
            continue
            
        doc_id = doc.metadata.get("doc_id")
        chunk_idx = doc.metadata.get("chunk_index")
        
        if (doc_id, chunk_idx) in seen_chunks:
            continue
        seen_chunks.add((doc_id, chunk_idx))

        
        expanded_content = doc.page_content
        expanded_metadata = dict(doc.metadata)

        # Pull next adjacent chunk to make context cohesive and complete sentences
        if doc_id and chunk_idx is not None:
            next_idx = chunk_idx + 1
            next_chunk = None
            for d in main_vector_store.docstore._dict.values():
                if d.metadata.get("doc_id") == doc_id and d.metadata.get("chunk_index") == next_idx:
                    next_chunk = d
                    break
            if next_chunk:
                sep = "\n" if not expanded_content.endswith("\n") else ""
                expanded_content += sep + next_chunk.page_content
                seen_chunks.add((doc_id, next_idx))
                # Chunk indices run across the whole document, so the adjacent
                # chunk can sit on the following page. Without recording that,
                # the citation reads "Page 5" while quoting text from page 6 --
                # sending the reader to a page that does not contain the quote.
                next_page = next_chunk.metadata.get("page")
                this_page = expanded_metadata.get("page")
                if next_page is not None and this_page is not None and next_page != this_page:
                    expanded_metadata["page_end"] = next_page
                expanded_metadata["expanded_with_adjacent_chunk"] = True

        from langchain_core.documents import Document
        expanded_doc = Document(page_content=expanded_content, metadata=expanded_metadata)
        
        simulated_distance = 2.0 * (1.0 - hybrid_score)
        final_results.append((expanded_doc, simulated_distance))
        
    top_score = scored_candidates[0][2] if scored_candidates else 0.0
    boost_type = "proximity" if is_definition_query else ("header" if any(len(w) > 3 for w in query_content_words) else None)
    log_rag_retrieval_event(
        query=query,
        candidates_count=len(candidates),
        passed_threshold_count=len(final_results),
        top_score=top_score,
        vector_ms=vector_ms,
        bm25_ms=bm25_ms,
        boost_applied=boost_type
    )

    # Optional developer trace, populated in place so the caller can surface the
    # whole retrieval stage without this function needing a second return value
    # or a parallel code path.
    if trace is not None:
        trace.update({
            "query": query,
            "doc_ids": list(target_ids),
            "candidates": len(candidates),
            "passed_threshold": len(final_results),
            "relevance_threshold": relevance_threshold,
            "top_score": round(top_score, 4),
            "vector_ms": round(vector_ms, 2),
            "bm25_ms": round(bm25_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "reranked": bool(reranked),
            "boost_applied": boost_type or "none",
            "ranked": [c.to_trace() for c in reranked[:10]],
            "selected": [
                {
                    "doc_name": d.metadata.get("doc_name"),
                    "page": d.metadata.get("page"),
                    "page_end": d.metadata.get("page_end"),
                    "chunk_index": d.metadata.get("chunk_index"),
                    "score": round(1.0 - (dist / 2.0), 4),
                }
                for d, dist in final_results
            ],
        })

    return final_results


