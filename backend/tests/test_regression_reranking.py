"""
Regression tests for the second-stage reranker (Phase 2.5) and multi-document
retrieval (Phase 2.1).

Before this stage existed, hybrid fusion was the only ranking: every candidate
was scored once, independently, against the query. Fusion is a recall device --
good at getting the right passage into the top fifteen, weak at deciding which
of those fifteen answers the question. In particular a heading that repeats the
query's words outscores the paragraph that contains the answer.

The reranker's contract, pinned here: it REORDERS, it never drops. Filtering
belongs to the relevance threshold, which has exactly one owner.
"""

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from backend.reranker import RerankedCandidate, lexical_rerank_score, rerank


def doc(text, **metadata):
    return Document(page_content=text, metadata=metadata)


HEADING = "3.2 DATA RETENTION PERIOD"

ANSWER_PARAGRAPH = (
    "The data retention period is 30 days. After that window, audit logs are "
    "permanently deleted from primary storage and from all replicas, and the "
    "deletion is recorded in the compliance register for the following quarter."
)

OFF_TOPIC = (
    "Quarterly revenue in the Frankfurt office rose by twelve percent, driven "
    "largely by the new cafeteria subscription programme introduced last spring."
)


# ---------------------------------------------------------------------------
# The lexical scorer
# ---------------------------------------------------------------------------

def test_the_answer_paragraph_outscores_the_matching_heading():
    """
    The failure mode fusion cannot see: a heading matches every query term and
    contains no answer.
    """
    query = "What is the data retention period?"
    assert lexical_rerank_score(query, ANSWER_PARAGRAPH) > lexical_rerank_score(query, HEADING)


def test_an_off_topic_passage_scores_near_zero():
    assert lexical_rerank_score("What is the data retention period?", OFF_TOPIC) < 0.3


def test_contiguous_phrases_beat_scattered_words():
    query = "data retention period"
    contiguous = (
        "The data retention period is defined in section three and applies to "
        "every system that stores audit information for compliance purposes."
    )
    scattered = (
        "Retention of data is discussed below. The period covered by this "
        "document is the financial year, and data is one of several topics."
    )
    assert lexical_rerank_score(query, contiguous) > lexical_rerank_score(query, scattered)


def test_empty_inputs_abstain_rather_than_raising():
    """
    Nothing to compare means no opinion. Abstention is distinct from a score of
    0.0 ("compared, and it is a poor match") -- see the abstention tests below
    for why conflating the two broke cross-lingual retrieval.
    """
    assert lexical_rerank_score("", ANSWER_PARAGRAPH) < 0.0
    assert lexical_rerank_score("a query", "") < 0.0


def test_a_passage_sharing_no_terms_with_the_query_abstains():
    """
    A Hindi query against an English passage shares no surface forms even when
    the vector stage matched them correctly. Scoring that 0.0 asserted "bad
    match" and dragged a 0.925 vector hit under the relevance threshold.
    """
    assert lexical_rerank_score("प्रमाणीकरण नीति क्या है", ANSWER_PARAGRAPH) < 0.0


def test_an_abstained_candidate_keeps_its_retrieval_score_untouched():
    candidates = [(doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=0), 0.925)]
    reranked = rerank("प्रमाणीकरण नीति क्या है", candidates)

    assert reranked[0].final_score == 0.925, "abstention must not blend in a zero"


def test_abstention_does_not_disturb_ordering_when_no_scorer_has_an_opinion():
    candidates = [
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=0), 0.90),
        (doc(OFF_TOPIC, doc_id="d2", chunk_index=0), 0.60),
    ]
    reranked = rerank("प्रमाणीकरण नीति", candidates)
    assert [round(c.final_score, 3) for c in reranked] == [0.90, 0.60]


def test_scores_stay_within_range():
    score = lexical_rerank_score("data retention period", ANSWER_PARAGRAPH)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# The rerank stage
# ---------------------------------------------------------------------------

def test_reranking_promotes_the_answer_over_a_better_fused_heading():
    """
    The heading arrives ranked FIRST from fusion. Reranking must be able to
    overturn that, which is the entire reason for a second stage.
    """
    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
    ]
    reranked = rerank("What is the data retention period?", candidates)

    assert reranked[0].document.page_content == ANSWER_PARAGRAPH


def test_reranking_never_drops_candidates():
    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
        (doc(OFF_TOPIC, doc_id="d2", chunk_index=0), 0.55),
    ]
    reranked = rerank("data retention period", candidates)

    assert len(reranked) == len(candidates)
    assert {c.document.page_content for c in reranked} == {
        HEADING, ANSWER_PARAGRAPH, OFF_TOPIC
    }


def test_disabling_the_reranker_preserves_the_fused_order():
    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
    ]
    reranked = rerank("data retention period", candidates, strategy="none")
    assert reranked[0].document.page_content == HEADING
    assert reranked[0].final_score == 0.88


def test_zero_weight_preserves_the_fused_order():
    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
    ]
    reranked = rerank("data retention period", candidates, weight=0.0)
    assert reranked[0].document.page_content == HEADING


def test_reranking_an_empty_candidate_list_is_safe():
    assert rerank("anything", []) == []


def test_llm_reranker_scores_are_applied():
    llm = MagicMock()
    # Passage 2 is the answer; the LLM says so.
    llm.invoke.return_value = MagicMock(content="1: 2\n2: 9")

    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
    ]
    reranked = rerank("data retention period", candidates, llm=llm, strategy="llm", weight=0.6)

    assert reranked[0].document.page_content == ANSWER_PARAGRAPH


def test_llm_reranker_failure_falls_back_to_lexical():
    """A provider outage must degrade the ranking, not remove reranking."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("provider down")

    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
    ]
    reranked = rerank("What is the data retention period?", candidates, llm=llm, strategy="llm")

    assert len(reranked) == 2
    assert reranked[0].document.page_content == ANSWER_PARAGRAPH


def test_llm_reranker_handles_gemini_block_content():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=[{"type": "text", "text": "1: 1\n2: 8"}])

    candidates = [
        (doc(HEADING, doc_id="d1", chunk_index=0), 0.88),
        (doc(ANSWER_PARAGRAPH, doc_id="d1", chunk_index=1), 0.71),
    ]
    reranked = rerank("data retention period", candidates, llm=llm, strategy="llm", weight=0.6)
    assert reranked[0].document.page_content == ANSWER_PARAGRAPH


def test_trace_entries_expose_both_scores():
    candidate = RerankedCandidate(
        document=doc(ANSWER_PARAGRAPH, doc_name="policy.pdf", page=3, chunk_index=1),
        retrieval_score=0.71,
        rerank_score=0.9,
        final_score=0.79,
    )
    trace = candidate.to_trace()
    assert trace["doc_name"] == "policy.pdf"
    assert trace["page"] == 3
    assert trace["retrieval_score"] == 0.71
    assert trace["rerank_score"] == 0.9


# ---------------------------------------------------------------------------
# Multi-document retrieval
# ---------------------------------------------------------------------------

def test_evidence_from_several_documents_is_merged_and_deduplicated():
    from backend.chat_engine import _retrieve_evidence

    doc_a = doc("Policy A retains logs for 30 days.", doc_id="a", doc_name="A.pdf", chunk_index=0)
    doc_b = doc("Policy B retains logs for 90 days.", doc_id="b", doc_name="B.pdf", chunk_index=0)

    with patch("backend.chat_engine.search_index") as mock_search:
        mock_search.return_value = [(doc_a, 0.2), (doc_b, 0.3)]
        results, trace = _retrieve_evidence(["retention period"], ["a", "b"], 5)

    assert {d.metadata["doc_id"] for d, _ in results} == {"a", "b"}
    assert trace["merged_count"] == 2


def test_multi_query_legs_keep_the_best_score_per_passage():
    """
    A passage found only by the paraphrase must not be penalised for scoring
    badly on the original wording.
    """
    from backend.chat_engine import _retrieve_evidence

    shared = doc("Retention is 30 days.", doc_id="a", doc_name="A.pdf", chunk_index=0)

    with patch("backend.chat_engine.search_index") as mock_search:
        # Same passage, weaker on the first phrasing, stronger on the second.
        mock_search.side_effect = [[(shared, 0.9)], [(shared, 0.1)]]
        results, trace = _retrieve_evidence(["how long kept", "retention period"], ["a"], 5)

    assert len(results) == 1, "the same passage must not appear twice"
    assert results[0][1] == 0.1, "the best score across phrasings should win"
    assert len(trace["legs"]) == 2
