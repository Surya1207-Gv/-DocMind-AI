"""
Retrieval regression tests against a REAL index.

These build an actual PDF, run it through the real ingest path, build a real
FAISS index and call the real `search_index`. Only the embedding provider is
substituted (for determinism and cost), and it is a genuine embedder producing
unit-normalised vectors -- not a canned result.

That distinction is the point. The reported failure was:

    Query "Dartmouth Conference 1956" against a document that plainly contains
    "The Dartmouth Conference in 1956 ..." on page 4 returned
    "I cannot find any information related to your question."

and it was invisible to every existing test because they all mocked
`search_index` and therefore skipped BM25 normalisation, hybrid fusion, and the
relevance gate -- the three stages that interacted to produce it.

Root cause: the fused score weights BM25 at 1 - VECTOR_WEIGHT = 0.4, which is
below RELEVANCE_THRESHOLD = 0.5. A passage that was the single unambiguous
exact match for the query still needed vector_sim >= 0.167 to survive the gate.
Lexical evidence alone could never admit a chunk -- precisely inverting the
purpose of hybrid retrieval for the queries BM25 exists to serve.
"""

import os
import shutil
import tempfile

import pytest

from backend.tests.ai_corpus import (
    DeterministicEmbeddings,
    OTHER_USER_PAGES,
    build_index,
)


DOC_ID = "acd64fe8-f40f-4179-a871-83dddbd110ac"
OTHER_DOC_ID = "b0000000-0000-0000-0000-00000000000b"


@pytest.fixture(scope="module")
def real_index():
    """
    A real FAISS index over the AI.pdf-like corpus, built once for the module.

    Yields a `search(query, doc_ids=None, top_k=3)` helper that calls the
    production `search_index`.
    """
    import backend.config as config
    import backend.embedding_manager as em

    tmp = tempfile.mkdtemp(prefix="docmind-retrieval-")
    saved = (config.UPLOAD_DIR, config.FAISS_DIR, em.FAISS_DIR, em.get_embeddings_model)

    config.UPLOAD_DIR = os.path.join(tmp, "uploads")
    config.FAISS_DIR = os.path.join(tmp, "faiss")
    em.FAISS_DIR = config.FAISS_DIR
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.FAISS_DIR, exist_ok=True)

    embeddings = DeterministicEmbeddings()
    em.get_embeddings_model = lambda: embeddings

    chunks = build_index(DOC_ID)
    build_index(OTHER_DOC_ID, "OtherTenant.pdf", OTHER_USER_PAGES)

    def search(query, doc_ids=None, top_k=3, **kwargs):
        return em.search_index(
            query, [DOC_ID] if doc_ids is None else doc_ids, top_k=top_k, **kwargs
        )

    yield {"search": search, "chunks": chunks, "embeddings": embeddings, "module": em}

    config.UPLOAD_DIR, config.FAISS_DIR, em.FAISS_DIR, em.get_embeddings_model = saved
    shutil.rmtree(tmp, ignore_errors=True)


def pages_of(results):
    return [doc.metadata.get("page") for doc, _ in results]


# ---------------------------------------------------------------------------
# Ingest: the text has to be in the index before retrieval can be blamed
# ---------------------------------------------------------------------------

def test_the_pdf_text_is_actually_indexed(real_index):
    chunks = real_index["chunks"]
    dartmouth = [c for c in chunks if "Dartmouth Conference in 1956" in c["text"]]

    assert dartmouth, "the source passage never survived extraction/chunking"
    assert dartmouth[0]["metadata"]["page"] == 4
    assert dartmouth[0]["metadata"]["doc_id"] == DOC_ID


def test_the_index_holds_one_vector_per_chunk(real_index):
    from langchain_community.vectorstores import FAISS
    import backend.config as config

    store = FAISS.load_local(
        os.path.join(config.FAISS_DIR, DOC_ID),
        real_index["embeddings"],
        allow_dangerous_deserialization=True,
    )
    assert store.index.ntotal == len(real_index["chunks"])
    assert len(store.docstore._dict) == len(real_index["chunks"])


def test_stored_vectors_are_unit_normalised(real_index):
    """
    The pipeline converts FAISS squared-L2 to cosine with `1 - d/2`, which is
    only valid for unit vectors. If a future embedding provider returned
    unnormalised vectors, every similarity would collapse toward zero and
    retrieval would fail silently.
    """
    import numpy as np
    from langchain_community.vectorstores import FAISS
    import backend.config as config

    store = FAISS.load_local(
        os.path.join(config.FAISS_DIR, DOC_ID),
        real_index["embeddings"],
        allow_dangerous_deserialization=True,
    )
    vectors = np.asarray(store.index.reconstruct_n(0, store.index.ntotal))
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# 1-5, 8. The reported queries
# ---------------------------------------------------------------------------

def test_dartmouth_keyword_query_retrieves_page_4(real_index):
    """The exact reported failure."""
    results = real_index["search"]("Dartmouth Conference 1956")

    assert results, "retrieval returned nothing for an exact match present on page 4"
    assert 4 in pages_of(results)
    assert "Dartmouth Conference in 1956" in results[0][0].page_content


def test_dartmouth_natural_language_query_retrieves_page_4(real_index):
    results = real_index["search"]("What year was the Dartmouth Conference?")

    assert results
    assert 4 in pages_of(results)
    assert "1956" in results[0][0].page_content


def test_three_core_elements_retrieves_page_8(real_index):
    results = real_index["search"](
        "What are the three core elements that explain how modern AI systems work?"
    )

    assert results
    assert 8 in pages_of(results)
    text = results[0][0].page_content
    for element in ("Algorithms", "Data", "Computing Power"):
        assert element in text


def test_four_vs_of_big_data_retrieves_page_13(real_index):
    results = real_index["search"]("What are the four Vs of Big Data?")

    assert results
    assert 13 in pages_of(results)
    assert "Veracity" in results[0][0].page_content


def test_gpu_versus_tpu_retrieves_the_hardware_pages(real_index):
    results = real_index["search"]("What is the difference between a GPU and a TPU?", top_k=4)

    assert results
    assert 19 in pages_of(results), "the passage stating the difference is on page 19"


def test_exact_identifier_rfc_8446_stays_searchable(real_index):
    """
    Identifier queries are pure lexical signal and near-zero semantic signal --
    the case the fused score handled worst.
    """
    results = real_index["search"]("RFC 8446")

    assert results, "an exact identifier present in the document was not retrievable"
    assert 21 in pages_of(results)
    # PDFs wrap lines wherever the layout demands, so an identifier can be split
    # across a newline in the extracted text. Retrieval is unaffected because the
    # tokenizer splits on whitespace either way, so the assertion normalises
    # whitespace rather than requiring the literal to survive intact.
    normalised = " ".join(results[0][0].page_content.split())
    assert "RFC 8446" in normalised


@pytest.mark.parametrize("query,expected", [
    ("SHA256", "SHA256"),
    ("GPT-4", "GPT-4"),
])
def test_other_exact_identifiers_stay_searchable(real_index, query, expected):
    results = real_index["search"](query)
    assert results, f"{query!r} returned nothing"
    assert expected in results[0][0].page_content


# ---------------------------------------------------------------------------
# 6. The unsupported question must still be refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query", [
    "What was the price of OpenAI stock in 2018?",
    "Who won the 2022 FIFA World Cup final?",
    "What is the capital of France?",
])
def test_unsupported_questions_retrieve_nothing(real_index, query):
    """
    The fix admits candidates on lexical evidence, so it must not have turned
    the relevance gate into a rubber stamp. These questions have no answer in
    the corpus and must still come back empty.
    """
    assert real_index["search"](query) == []


def test_a_query_sharing_only_common_words_is_still_refused(real_index):
    """
    "AI" and "data" appear throughout the document. Matching only low-IDF terms
    must not count as coverage, or every question would find evidence.
    """
    assert real_index["search"]("What is the AI data revenue forecast for 2027?") == []


# ---------------------------------------------------------------------------
# 7. Cross-user isolation
# ---------------------------------------------------------------------------

def test_retrieval_is_confined_to_the_requested_document(real_index):
    """
    The other tenant's document also mentions "Dartmouth Conference" and "1956"
    and would rank highly on lexical evidence. Scoping must exclude it.
    """
    results = real_index["search"]("Dartmouth Conference 1956", doc_ids=[DOC_ID], top_k=5)

    assert results
    for doc, _ in results:
        assert doc.metadata["doc_id"] == DOC_ID
        assert "another tenant" not in doc.page_content


def test_an_empty_document_selection_retrieves_nothing(real_index):
    """An empty list means "no documents", never "every index on disk"."""
    assert real_index["search"]("Dartmouth Conference 1956", doc_ids=[]) == []


def test_the_other_tenants_document_is_reachable_only_when_requested(real_index):
    """Sanity check on the fixture: the isolation test above is not vacuous."""
    results = real_index["search"]("Dartmouth Conference 1956", doc_ids=[OTHER_DOC_ID])
    assert results
    assert results[0][0].metadata["doc_id"] == OTHER_DOC_ID


# ---------------------------------------------------------------------------
# The specific scoring bug, pinned
# ---------------------------------------------------------------------------

def test_a_perfect_lexical_match_is_admitted_without_help_from_the_reranker(real_index):
    """
    The regression, isolated.

    With the reranker off, the Dartmouth chunk's fused score is ~0.486 -- below
    RELEVANCE_THRESHOLD (0.5) -- because BM25's weight is capped at 0.4 and the
    vector channel scores a three-word query against a paragraph poorly. It must
    still be admitted, on the strength of its lexical evidence.
    """
    results = real_index["search"]("Dartmouth Conference 1956", use_rerank=False)

    assert results, (
        "a passage containing every query term was discarded because the fused "
        "score could not reach the threshold on lexical evidence alone"
    )
    assert 4 in pages_of(results)


def test_lexical_coverage_separates_a_real_match_from_a_spurious_one(real_index):
    """
    Coverage is the measure the gate relies on, so its discrimination is
    asserted directly rather than only through its effect.
    """
    from backend.embedding_manager import SimpleBM25

    corpus = [c["text"] for c in real_index["chunks"]]
    bm25 = SimpleBM25(corpus)

    dartmouth_idx = next(
        i for i, text in enumerate(corpus) if "Dartmouth Conference in 1956" in text
    )
    hit = bm25.query_coverage("Dartmouth Conference 1956", dartmouth_idx)

    best_spurious = max(
        bm25.query_coverage("What was the price of OpenAI stock in 2018?", i)
        for i in range(len(corpus))
    )

    assert hit == pytest.approx(1.0), "every query term is present; coverage should be total"
    assert best_spurious < 0.5
    assert hit > best_spurious * 2


def test_coverage_counts_query_terms_absent_from_the_whole_corpus(real_index):
    """
    A term the collection does not contain is missing information, not
    irrelevant information. Excluding such terms from the denominator would let
    an unrelated question score a perfect match on its one common word.
    """
    from backend.embedding_manager import SimpleBM25

    corpus = [c["text"] for c in real_index["chunks"]]
    bm25 = SimpleBM25(corpus)

    ai_page = next(i for i, text in enumerate(corpus) if "Algorithms" in text)
    coverage = bm25.query_coverage("OpenAI stock price 2018 valuation", ai_page)

    assert coverage < 0.3


def test_the_relevance_threshold_itself_is_unchanged():
    """
    The fix adds an alternative admission route; it does not relax the existing
    one. A silent threshold drop would trade the reported bug for hallucinations.
    """
    from backend.config import LEXICAL_COVERAGE_THRESHOLD, RELEVANCE_THRESHOLD, VECTOR_WEIGHT

    assert RELEVANCE_THRESHOLD == 0.5
    assert VECTOR_WEIGHT == 0.6
    assert LEXICAL_COVERAGE_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# 9. Multilingual
# ---------------------------------------------------------------------------

def test_a_non_latin_query_does_not_crash_or_return_wrong_pages(real_index):
    """
    Cross-lingual matching is a property of the embedding model, and the
    deterministic embedder used here is not semantic -- so this asserts what is
    actually verifiable offline: the pipeline handles a Devanagari query without
    error and does not fabricate a match. True cross-lingual recall is covered
    against the live model, and is noted as a limitation.
    """
    results = real_index["search"]("डार्टमाउथ सम्मेलन 1956")
    for doc, _ in results:
        assert doc.metadata["doc_id"] == DOC_ID


def test_a_shared_numeral_alone_is_not_treated_as_coverage(real_index):
    """
    "1956" appears in the Dartmouth passage. A Hindi query containing only that
    numeral shares one low-information token, which must not by itself admit
    the chunk on lexical grounds.
    """
    from backend.embedding_manager import SimpleBM25

    corpus = [c["text"] for c in real_index["chunks"]]
    bm25 = SimpleBM25(corpus)
    dartmouth_idx = next(
        i for i, text in enumerate(corpus) if "Dartmouth Conference in 1956" in text
    )
    coverage = bm25.query_coverage("डार्टमाउथ सम्मेलन 1956", dartmouth_idx)
    assert coverage < 0.5


# ---------------------------------------------------------------------------
# Trace / observability
# ---------------------------------------------------------------------------

def test_the_trace_explains_why_candidates_were_rejected(real_index):
    trace = {}
    real_index["search"]("Dartmouth Conference 1956", trace=trace)

    assert trace["candidates"] > 0
    assert trace["passed_threshold"] >= 1
    assert trace["relevance_threshold"] == 0.5
    assert trace["lexical_coverage_threshold"] == 0.5
    assert trace["selected"]
    assert trace["selected"][0]["page"] == 4
    assert trace["selected"][0]["lexical_coverage"] == pytest.approx(1.0)

    for entry in trace["rejected"]:
        assert entry["reason"]
        assert "lexical_coverage" in entry


def test_the_trace_records_a_fully_rejected_query(real_index):
    trace = {}
    results = real_index["search"]("What was the price of OpenAI stock in 2018?", trace=trace)

    assert results == []
    assert trace["passed_threshold"] == 0
    assert trace["rejected"], "a refused query must say which candidates it rejected and why"
