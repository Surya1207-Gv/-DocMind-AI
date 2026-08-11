import os
import sys
import json
import argparse
from typing import List, Dict, Any

# Ensure repository root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.embedding_manager import search_index, get_embeddings_model, FAISS, SimpleBM25
from backend.config import FAISS_DIR

DATASET_PATH = os.path.join(BASE_DIR, "eval", "dataset.json")

def load_dataset() -> Dict[str, Any]:
    if os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"documents": [], "queries": []}

def save_dataset(data: Dict[str, Any]):
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved dataset to {DATASET_PATH}")

def preview_candidates(query: str, doc_id: str):
    """
    Retrieves and displays top 20 candidate chunks for a query to assist ground truth labeling.
    """
    doc_path = os.path.join(FAISS_DIR, doc_id)
    if not os.path.exists(doc_path):
        print(f"❌ Error: Index not found at {doc_path}")
        return

    embeddings = get_embeddings_model()
    db = FAISS.load_local(doc_path, embeddings, allow_dangerous_deserialization=True)
    candidates = db.similarity_search_with_score(query, k=20)
    
    corpus = [doc.page_content for doc, _ in candidates]
    bm25 = SimpleBM25(corpus)
    bm25_scores = [bm25.get_score(query, i) for i in range(len(candidates))]
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0

    print(f"\n{'='*80}")
    print(f"🔍 Query: '{query}' | Target Doc: {doc_id}")
    print(f"{'='*80}")
    print(f"{'#':<3} | {'Chunk ID':<10} | {'Page':<5} | {'Vec Dist':<9} | {'BM25':<7} | {'Hybrid':<7} | {'Preview (first 140 chars)'}")
    print("-" * 80)

    for i, (doc, dist) in enumerate(candidates):
        chunk_idx = doc.metadata.get("chunk_index", i)
        page = doc.metadata.get("page", 1)
        v_sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        bm25_norm = bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0.0
        hybrid = 0.6 * v_sim + 0.4 * bm25_norm
        preview = doc.page_content.replace("\n", " ")[:140]
        print(f"{i+1:<3} | Index {chunk_idx:<4} | P.{page:<4} | {dist:<9.4f} | {bm25_norm:<7.4f} | {hybrid:<7.4f} | {preview}...")

def add_labeled_query(query: str, doc_id: str, query_type: str, relevant_indices: List[int], notes: str):
    dataset = load_dataset()
    new_id = f"q{len(dataset.get('queries', [])) + 1:03d}"
    entry = {
        "id": new_id,
        "query": query,
        "doc_id": doc_id,
        "query_type": query_type,
        "relevant_chunk_indices": relevant_indices,
        "notes": notes
    }
    dataset.setdefault("queries", []).append(entry)
    save_dataset(dataset)
    print(f"✅ Added query [{new_id}]: '{query}' with relevant chunks {relevant_indices}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocMind Evaluation Dataset Label Helper")
    parser.add_argument("--query", typestr=str, help="Query text to inspect")
    parser.add_argument("--doc_id", type=str, help="Target document ID")
    parser.add_argument("--type", choices=["definitional", "factual", "comparative", "multi_hop"], default="factual")
    parser.add_argument("--relevant", type=int, nargs="+", help="Relevant chunk index numbers")
    parser.add_argument("--notes", type=str, default="", help="Notes explaining why chunks are ground truth")
    args = parser.parse_args()

    if args.query and args.doc_id:
        if args.relevant is not None:
            add_labeled_query(args.query, args.doc_id, args.type, args.relevant, args.notes)
        else:
            preview_candidates(args.query, args.doc_id)
    else:
        print("Usage:")
        print("  1. Preview candidates: python eval/label_helper.py --query 'What is X?' --doc_id <id>")
        print("  2. Add ground truth:   python eval/label_helper.py --query 'What is X?' --doc_id <id> --type definitional --relevant 0 1 --notes 'core definition'")
