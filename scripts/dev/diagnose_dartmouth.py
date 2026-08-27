"""
Retrieval diagnostic for the reported Dartmouth failure.

Builds a REAL FAISS index from a REAL PDF through the production ingest path
(process_document -> create_and_save_index), then runs the production
search_index and prints every intermediate score. Nothing about retrieval is
mocked except the embedding provider, which is replaced with a deterministic
local model so the run is reproducible and costs nothing.

    python scripts/dev/diagnose_dartmouth.py
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OPENROUTER_API_KEY", "diagnostic")
os.environ.setdefault("JWT_SECRET_KEY", "diagnostic")

TMP = tempfile.mkdtemp(prefix="docmind-diag-")
os.environ["DATA_DIR"] = TMP

import backend.config as config  # noqa: E402

config.UPLOAD_DIR = os.path.join(TMP, "uploads")
config.FAISS_DIR = os.path.join(TMP, "faiss_indices")
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.FAISS_DIR, exist_ok=True)

import backend.embedding_manager as em  # noqa: E402

em.FAISS_DIR = config.FAISS_DIR

from backend.tests.ai_corpus import DeterministicEmbeddings, PAGES, write_pdf  # noqa: E402


make_pdf = write_pdf


def main():
    from backend.document_processor import process_document
    from backend.embedding_manager import create_and_save_index, search_index

    embeddings = DeterministicEmbeddings()
    em.get_embeddings_model = lambda: embeddings

    doc_id = "acd64fe8-f40f-4179-a871-83dddbd110ac"
    pdf_path = os.path.join(config.UPLOAD_DIR, f"{doc_id}.pdf")
    make_pdf(pdf_path)

    print("=" * 100)
    print("STAGE 1 - PDF EXTRACTION AND CHUNKING")
    print("=" * 100)
    chunks = process_document(pdf_path, "AI.pdf", doc_id)
    print(f"document_id : {doc_id}")
    print(f"chunks      : {len(chunks)}")
    pages = sorted({c["metadata"]["page"] for c in chunks})
    print(f"pages       : {pages}")

    targets = {
        "dartmouth": ("Dartmouth", 4),
        "elements": ("Computing Power", 8),
        "vs": ("Veracity", 13),
        "tpu": ("TPU", 19),
        "rfc": ("RFC 8446", 21),
    }
    print("\nchunks containing the target phrases:")
    for key, (needle, page) in targets.items():
        hits = [c for c in chunks if needle.lower() in c["text"].lower()]
        for hit in hits:
            print(f"  [{key:9s}] chunk_index={hit['metadata']['chunk_index']:3d} "
                  f"page={hit['metadata']['page']:2d} (expected {page})  "
                  f"{hit['text'][:70]!r}")
        if not hits:
            print(f"  [{key:9s}] *** NOT PRESENT IN ANY CHUNK ***")

    print()
    print("=" * 100)
    print("STAGE 2 - INDEXING")
    print("=" * 100)
    create_and_save_index(chunks, doc_id)
    from langchain_community.vectorstores import FAISS

    store = FAISS.load_local(
        os.path.join(config.FAISS_DIR, doc_id), embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"FAISS index size  : {store.index.ntotal}")
    print(f"embedding dim     : {store.index.d}")
    print(f"docstore entries  : {len(store.docstore._dict)}")

    import numpy as np

    vectors = store.index.reconstruct_n(0, store.index.ntotal)
    norms = np.linalg.norm(np.asarray(vectors), axis=1)
    print(f"vector norms      : min={norms.min():.4f} max={norms.max():.4f} "
          f"(1.0 => unit-normalised, so squared-L2/2 is a valid cosine conversion)")

    queries = [
        "Dartmouth Conference 1956",
        "What year was the Dartmouth Conference?",
        "What are the three core elements that explain how modern AI systems work?",
        "What are the four Vs of Big Data?",
        "What is the difference between a GPU and a TPU?",
        "RFC 8446",
        "What was the price of OpenAI stock in 2018?",
    ]

    for query in queries:
        trace_stages(query, doc_id, store, embeddings)

    shutil.rmtree(TMP, ignore_errors=True)


def trace_stages(query, doc_id, store, embeddings):
    """Re-derive every intermediate score the production path computes."""
    import numpy as np

    from backend.config import RELEVANCE_THRESHOLD, VECTOR_WEIGHT
    from backend.embedding_manager import SimpleBM25, search_index
    from backend.reranker import RERANK_WEIGHT, lexical_rerank_score
    from backend.text_utils import tokenize

    print()
    print("=" * 100)
    print(f"QUERY: {query!r}")
    print("=" * 100)

    top_k = 3  # qa mode
    candidate_count = max(top_k * 3, 15)
    candidates = store.similarity_search_with_score(query, k=candidate_count)
    print(f"candidate_count requested={candidate_count}  returned={len(candidates)}")

    corpus = [d.page_content for d, _ in candidates]
    bm25 = SimpleBM25(corpus)
    print(f"BM25 corpus size  : {bm25.corpus_size}")
    print(f"BM25 query tokens : {tokenize(query)}")

    bm25_scores = [bm25.get_score(query, i) for i in range(len(candidates))]
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0

    rows = []
    for i, (doc, dist) in enumerate(candidates):
        vector_sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
        bm25_norm = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
        hybrid = vector_sim if max_bm25 <= 0.0 else (
            VECTOR_WEIGHT * vector_sim + (1.0 - VECTOR_WEIGHT) * bm25_norm
        )
        rr = lexical_rerank_score(query, doc.page_content)
        final = hybrid if rr < 0 else (1 - RERANK_WEIGHT) * hybrid + RERANK_WEIGHT * rr
        rows.append({
            "page": doc.metadata.get("page"),
            "chunk": doc.metadata.get("chunk_index"),
            "l2": dist,
            "vec": vector_sim,
            "bm25_raw": bm25_scores[i],
            "bm25_norm": bm25_norm,
            "hybrid": hybrid,
            "rerank": rr,
            "final": final,
            "text": doc.page_content[:48].replace("\n", " "),
        })

    rows.sort(key=lambda r: r["final"], reverse=True)

    print()
    print(f"{'rank':>4} {'page':>4} {'chunk':>5} {'L2':>8} {'vec':>6} {'bm25':>7} "
          f"{'bm25n':>6} {'hybrid':>7} {'rerank':>7} {'final':>7}  verdict")
    print("-" * 100)
    for rank, r in enumerate(rows[:8]):
        in_top_k = rank < top_k
        passes = r["final"] >= RELEVANCE_THRESHOLD
        if not in_top_k:
            verdict = f"DROPPED: outside top_k={top_k}"
        elif not passes:
            verdict = f"DROPPED: final {r['final']:.3f} < threshold {RELEVANCE_THRESHOLD}"
        else:
            verdict = "KEPT"
        rr = "abstain" if r["rerank"] < 0 else f"{r['rerank']:7.3f}"
        print(f"{rank:>4} {r['page']:>4} {r['chunk']:>5} {r['l2']:>8.4f} {r['vec']:>6.3f} "
              f"{r['bm25_raw']:>7.3f} {r['bm25_norm']:>6.3f} {r['hybrid']:>7.3f} "
              f"{rr:>7} {r['final']:>7.3f}  {verdict}")

    results = search_index(query, [doc_id], top_k=top_k)
    print(f"\nsearch_index() returned {len(results)} result(s):")
    for doc, dist in results:
        print(f"    page={doc.metadata.get('page')} "
              f"chunk={doc.metadata.get('chunk_index')} "
              f"score={1 - dist / 2:.3f}  {doc.page_content[:60]!r}")
    if not results:
        print("    *** EMPTY -> chat_engine emits "
              "'I cannot find any information related to your question' ***")


if __name__ == "__main__":
    main()
