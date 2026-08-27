"""
Regression tests for multilingual retrieval (Phases 2.9 and 10).

The defect: the BM25 tokenizer matched only ASCII letters and digits, so every
Devanagari and Telugu token was discarded. BM25 therefore scored 0.0 for any
Hindi or Telugu query, and "hybrid" retrieval silently collapsed to vector-only
for those languages while continuing to report itself as hybrid.

English retrieval must keep working exactly as before -- that is the other half
of the requirement, so it is asserted here too.
"""

import pytest

from backend.embedding_manager import SimpleBM25
from backend.text_utils import content_terms, tokenize


HINDI_DOC = "प्रमाणीकरण नीति के अनुसार सभी प्रशासनिक खातों के लिए बहु-कारक प्रमाणीकरण आवश्यक है।"
TELUGU_DOC = "ప్రామాణీకరణ విధానం ప్రకారం అన్ని నిర్వాహక ఖాతాలకు బహుళ-కారక ప్రామాణీకరణ అవసరం."
ENGLISH_DOC = "The authentication policy requires multi-factor authentication for all administrative accounts."
UNRELATED_DOC = "Quarterly revenue rose by twelve percent in the Frankfurt office cafeteria budget."


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_devanagari_survives_tokenization():
    terms = tokenize(HINDI_DOC)
    assert terms, "Hindi text tokenized to nothing"
    assert "प्रमाणीकरण" in terms
    assert "नीति" in terms


def test_telugu_survives_tokenization():
    terms = tokenize(TELUGU_DOC)
    assert terms, "Telugu text tokenized to nothing"
    assert "ప్రామాణీకరణ" in terms


def test_english_tokenization_still_splits_camel_case():
    assert tokenize("HTTPServer") == ["http", "server"]
    assert tokenize("camelCaseWord") == ["camel", "case", "word"]


def test_acronyms_glued_to_digits_keep_their_letters():
    """
    "GPT4" used to tokenize to just ["4"]: the pattern had no branch for a
    trailing all-caps run, so the acronym was dropped and the term became
    unsearchable.
    """
    assert tokenize("GPT4") == ["gpt", "4"]
    assert tokenize("SHA256") == ["sha", "256"]


def test_mixed_script_text_keeps_both_halves():
    terms = tokenize("Authentication प्रमाणीकरण policy")
    assert "authentication" in terms
    assert "प्रमाणीकरण" in terms


# ---------------------------------------------------------------------------
# BM25 over non-Latin scripts
# ---------------------------------------------------------------------------

def test_bm25_scores_a_hindi_query_above_zero():
    """The core regression: this used to be exactly 0.0 for every Hindi query."""
    bm25 = SimpleBM25([HINDI_DOC, UNRELATED_DOC])
    score = bm25.get_score("प्रमाणीकरण नीति", 0)
    assert score > 0.0


def test_bm25_scores_a_telugu_query_above_zero():
    bm25 = SimpleBM25([TELUGU_DOC, UNRELATED_DOC])
    assert bm25.get_score("ప్రామాణీకరణ విధానం", 0) > 0.0


def test_bm25_ranks_the_matching_hindi_document_first():
    corpus = [UNRELATED_DOC, HINDI_DOC]
    bm25 = SimpleBM25(corpus)
    scores = [bm25.get_score("प्रमाणीकरण नीति", i) for i in range(len(corpus))]
    assert scores[1] > scores[0]


def test_bm25_ranks_the_matching_telugu_document_first():
    corpus = [UNRELATED_DOC, TELUGU_DOC]
    bm25 = SimpleBM25(corpus)
    scores = [bm25.get_score("ప్రామాణీకరణ విధానం", i) for i in range(len(corpus))]
    assert scores[1] > scores[0]


def test_english_bm25_still_ranks_correctly():
    """The multilingual fix must not regress the English path."""
    corpus = [UNRELATED_DOC, ENGLISH_DOC]
    bm25 = SimpleBM25(corpus)
    scores = [bm25.get_score("multi-factor authentication policy", i) for i in range(len(corpus))]
    assert scores[1] > scores[0]


def test_scripts_do_not_collide_with_each_other():
    """A Hindi query must not match a Telugu document just for being non-Latin."""
    corpus = [TELUGU_DOC]
    bm25 = SimpleBM25(corpus)
    assert bm25.get_score("प्रमाणीकरण नीति", 0) == 0.0


def test_a_mixed_language_corpus_still_serves_english():
    """
    Cross-lingual retrieval leans on the multilingual embedding model for
    meaning; BM25's job is to not be dead weight. With Hindi and Telugu
    documents in the corpus, an English query must still find its English
    document lexically.
    """
    corpus = [HINDI_DOC, TELUGU_DOC, ENGLISH_DOC]
    bm25 = SimpleBM25(corpus)
    scores = [bm25.get_score("administrative accounts authentication", i) for i in range(len(corpus))]
    assert scores.index(max(scores)) == 2


def test_content_terms_are_shared_between_retrieval_and_verification():
    """
    Verification scores claims with the same tokenizer BM25 uses. If they
    diverged, a passage retrieval considered a match could be judged as not
    containing the claim it was retrieved for.
    """
    assert content_terms("प्रमाणीकरण नीति") <= set(tokenize(HINDI_DOC))


# ---------------------------------------------------------------------------
# The query/document language matrix, through the full hybrid pipeline
# ---------------------------------------------------------------------------
#
# What is and is not verifiable offline, stated plainly:
#
#   BM25 (lexical) is fully testable here, and it is where the bug was. It
#   matches only on shared surface forms, so it contributes signal when query
#   and document share a language and legitimately contributes nothing when
#   they do not.
#
#   Cross-lingual matching (Hindi query -> English document) is a property of
#   the EMBEDDING MODEL, not of this code. These tests mock the embedder, so
#   they verify that the pipeline carries a cross-lingual vector result through
#   to the answer without BM25 suppressing it -- not that the live embedding
#   model is good at Hindi. That requires the real API and is called out as a
#   limitation rather than asserted here.

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


def _store(docs_with_scores):
    store = MagicMock()
    store.similarity_search_with_score.return_value = docs_with_scores
    store.docstore._dict = {}
    return store


def _doc(text, doc_id="d1", page=1, chunk_index=0):
    return Document(
        page_content=text,
        metadata={"doc_id": doc_id, "doc_name": "policy.pdf",
                  "page": page, "chunk_index": chunk_index},
    )


def _search(query, docs_with_scores):
    from backend.embedding_manager import search_index

    with patch("backend.embedding_manager.get_embeddings_model"), \
         patch("backend.embedding_manager.FAISS.load_local", return_value=_store(docs_with_scores)), \
         patch("os.path.exists", return_value=True):
        return search_index(query, ["d1"], top_k=3)


def test_english_document_english_query_retrieves():
    results = _search("multi-factor authentication for administrative accounts",
                      [(_doc(ENGLISH_DOC), 0.2), (_doc(UNRELATED_DOC, chunk_index=1), 0.9)])
    assert results
    assert ENGLISH_DOC in results[0][0].page_content


def test_english_document_hindi_query_survives_the_pipeline():
    """
    The vector stage carries the cross-lingual match (mocked here). BM25 scores
    0 because there are no shared surface forms -- which is correct -- and the
    fused score must still clear the threshold rather than being dragged under
    it.
    """
    results = _search("प्रमाणीकरण नीति क्या है", [(_doc(ENGLISH_DOC), 0.15)])
    assert results, "a strong cross-lingual vector match was lost in fusion"


def test_english_document_telugu_query_survives_the_pipeline():
    results = _search("ప్రామాణీకరణ విధానం ఏమిటి", [(_doc(ENGLISH_DOC), 0.15)])
    assert results, "a strong cross-lingual vector match was lost in fusion"


def test_hindi_document_hindi_query_gets_lexical_support():
    """Same language on both sides: BM25 must contribute, not sit at zero."""
    bm25 = SimpleBM25([HINDI_DOC, UNRELATED_DOC])
    assert bm25.get_score("प्रमाणीकरण नीति", 0) > 0.0

    results = _search("प्रमाणीकरण नीति",
                      [(_doc(HINDI_DOC), 0.4), (_doc(UNRELATED_DOC, chunk_index=1), 0.45)])
    assert results
    assert HINDI_DOC in results[0][0].page_content


def test_telugu_document_telugu_query_gets_lexical_support():
    bm25 = SimpleBM25([TELUGU_DOC, UNRELATED_DOC])
    assert bm25.get_score("ప్రామాణీకరణ విధానం", 0) > 0.0

    results = _search("ప్రామాణీకరణ విధానం",
                      [(_doc(TELUGU_DOC), 0.4), (_doc(UNRELATED_DOC, chunk_index=1), 0.45)])
    assert results
    assert TELUGU_DOC in results[0][0].page_content


def test_lexical_support_reorders_a_same_language_tie():
    """
    Two passages the vector stage cannot separate; only BM25 can. Before the
    tokenizer fix this test could not have passed for any non-Latin script,
    because BM25 returned 0.0 for both and the tie stood.
    """
    results = _search(
        "प्रमाणीकरण नीति",
        [(_doc(UNRELATED_DOC), 0.4), (_doc(HINDI_DOC, chunk_index=1), 0.4)],
    )
    assert results
    assert HINDI_DOC in results[0][0].page_content


def test_a_mixed_language_corpus_does_not_break_english_retrieval():
    results = _search(
        "administrative accounts authentication",
        [
            (_doc(HINDI_DOC), 0.42),
            (_doc(TELUGU_DOC, chunk_index=1), 0.42),
            (_doc(ENGLISH_DOC, chunk_index=2), 0.42),
        ],
    )
    assert results
    assert ENGLISH_DOC in results[0][0].page_content
