"""
Regression tests for claim-level verification, evidence gating and the
confidence model (Phases 2.3, 2.11, 3 and 9).

The behaviour under test is the fix for: retrieval returned one chunk at 0.712
and the system produced a confident-looking answer anyway.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from backend.models import ChatRequest
from backend.chat_engine import run_chat_stream
from backend.verification import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    ConfidenceResult,
    VerificationReport,
    apply_evidence_gate,
    compute_confidence,
    detect_contradictions,
    extract_claims,
    score_claim,
    verify_answer,
)


POLICY_TEXT = (
    "The authentication policy requires multi-factor authentication for all "
    "administrative accounts. Audit logs are retained for 30 days before deletion."
)


# ---------------------------------------------------------------------------
# Claim extraction and scoring
# ---------------------------------------------------------------------------

def test_extract_claims_ignores_headings_and_meta_commentary():
    answer = (
        "## Key findings\n"
        "The authentication policy requires multi-factor authentication for admins.\n"
        "Based on the document, this applies to every administrative account.\n"
        "Ok.\n"
    )
    claims = extract_claims(answer)

    assert any("multi-factor" in c for c in claims)
    assert not any(c.startswith("##") for c in claims)
    assert not any(c.lower().startswith("based on the document") for c in claims)
    assert "Ok." not in claims


def test_supported_claim_scores_high():
    support, index, missing = score_claim(
        "The authentication policy requires multi-factor authentication for administrative accounts.",
        [POLICY_TEXT],
    )
    assert support >= 0.8
    assert index == 0
    assert missing == []


def test_claim_with_a_fabricated_number_is_not_supported():
    """
    The single most damaging hallucination is an invented figure, and the
    easiest to catch: prose similarity must not be able to rescue it.
    """
    support, _index, missing = score_claim(
        "Audit logs are retained for 90 days before deletion.", [POLICY_TEXT]
    )
    assert missing == ["90"]
    assert support < 0.55, "a number absent from the evidence must fail the support threshold"


def test_claim_about_something_absent_scores_zero():
    support, index, missing = score_claim(
        "The company was founded in Berlin by three engineers in 1998.", [POLICY_TEXT]
    )
    assert support == 0.0
    # No passage supports it, so none is named -- a citation under an
    # unsupported claim would assert evidence that does not exist.
    assert index is None
    # And its invented specifics are still reported, which is what the gate acts on.
    assert "berlin" in missing
    assert "1998" in missing


def test_verify_answer_separates_supported_from_unsupported():
    answer = (
        "The authentication policy requires multi-factor authentication for "
        "administrative accounts. Audit logs are retained for 90 days."
    )
    report = verify_answer(answer, [POLICY_TEXT])

    assert report.supported_ratio == 0.5
    assert len(report.unsupported_claims) == 1
    assert "90 days" in report.unsupported_claims[0]


def test_a_refusal_is_treated_as_grounded_not_as_a_bad_claim():
    report = verify_answer(
        "I cannot find that information in the provided context.", [POLICY_TEXT]
    )
    assert report.is_refusal is True
    assert report.unsupported_claims == []


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def test_detects_conflicting_values_across_documents():
    evidence = [
        {"text": "The data retention period is 30 days.", "doc_id": "a",
         "doc_name": "PolicyA.pdf", "page": 2},
        {"text": "Our data retention period is 90 days.", "doc_id": "b",
         "doc_name": "PolicyB.pdf", "page": 7},
    ]
    conflicts = detect_contradictions(evidence)

    assert len(conflicts) == 1
    values = {v["value"] for v in conflicts[0]["values"]}
    assert values == {"30 days", "90 days"}
    # Both sides are citable.
    assert {v["doc_name"] for v in conflicts[0]["values"]} == {"PolicyA.pdf", "PolicyB.pdf"}


def test_differing_values_within_one_document_are_not_a_contradiction():
    """A table of tiers or a worked example is not a conflict."""
    evidence = [
        {"text": "Standard retention period is 30 days. Extended retention period is 90 days.",
         "doc_id": "a", "doc_name": "PolicyA.pdf", "page": 1},
    ]
    assert detect_contradictions(evidence) == []


def test_agreeing_documents_produce_no_contradiction():
    evidence = [
        {"text": "The data retention period is 30 days.", "doc_id": "a",
         "doc_name": "A.pdf", "page": 1},
        {"text": "The data retention period is 30 days.", "doc_id": "b",
         "doc_name": "B.pdf", "page": 1},
    ]
    assert detect_contradictions(evidence) == []


# ---------------------------------------------------------------------------
# Confidence model
# ---------------------------------------------------------------------------

def _report(ratio, refusal=False, contradictions=(), mean=None, specifics=()):
    report = VerificationReport(checked=True)
    report.supported_ratio = ratio
    # Defaults to the ratio so a test that cares only about the coarse verdict
    # does not have to state both.
    report.mean_support = ratio if mean is None else mean
    report.is_refusal = refusal
    report.contradictions = list(contradictions)
    report.unsupported_specifics = list(specifics)
    return report


def _confidence(score, label, band, retrieval_top=0.9):
    """
    Build a ConfidenceResult for gate tests.

    retrieval_top must be supplied because the weak-evidence rule reads it; an
    empty components dict means "no retrieval signal", which the gate correctly
    treats as insufficient evidence.
    """
    return ConfidenceResult(
        score=score, label=label, band=band,
        components={"retrieval_top": retrieval_top},
    )


def test_strong_retrieval_with_unsupported_claims_is_not_high_confidence():
    """
    The core Phase 3 regression: a good similarity score must no longer be able
    to carry an answer whose claims are not in the evidence.
    """
    result = compute_confidence(
        retrieval_scores=[0.92],
        report=_report(0.0),
        cited_source_count=1,
        retrieved_source_count=1,
        expected_source_count=3,
    )
    assert result.band in ("low", "very_low")


def test_one_weak_chunk_does_not_produce_a_confident_answer():
    """The literal reported case: candidates=15, passed=1, top_score=0.712."""
    result = compute_confidence(
        retrieval_scores=[0.712],
        report=_report(0.34),
        cited_source_count=1,
        retrieved_source_count=1,
        expected_source_count=3,
    )
    assert result.band != "high"
    assert result.score < 75


def test_strong_evidence_and_grounded_claims_is_high_confidence():
    result = compute_confidence(
        retrieval_scores=[0.9, 0.85, 0.8],
        report=_report(1.0),
        cited_source_count=3,
        retrieved_source_count=3,
        expected_source_count=3,
    )
    assert result.band == "high"


def test_no_evidence_forces_zero_confidence():
    result = compute_confidence(
        retrieval_scores=[],
        report=_report(1.0),
        cited_source_count=0,
        retrieved_source_count=0,
        expected_source_count=3,
    )
    assert result.score == 0
    assert result.band == "very_low"


def test_a_refusal_scores_zero_even_with_strong_retrieval():
    result = compute_confidence(
        retrieval_scores=[0.95],
        report=_report(1.0, refusal=True),
        cited_source_count=1,
        retrieved_source_count=1,
        expected_source_count=1,
    )
    assert result.score == 0


def test_contradictions_cap_confidence():
    conflicted = compute_confidence(
        retrieval_scores=[0.95, 0.93],
        report=_report(1.0, contradictions=[{"subject": "retention period", "values": []}]),
        cited_source_count=2,
        retrieved_source_count=2,
        expected_source_count=2,
    )
    assert conflicted.band != "high"


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------

def test_high_band_answer_passes_through_untouched():
    answer = "Multi-factor authentication is required for administrative accounts."
    final, gated = apply_evidence_gate(
        answer,
        _confidence(88, "High", "high"),
        _report(1.0),
    )
    assert final == answer
    assert gated is False


def test_medium_band_answer_is_kept_but_carries_a_warning():
    answer = "Multi-factor authentication is required for administrative accounts."
    final, gated = apply_evidence_gate(
        answer,
        _confidence(60, "Medium", "medium"),
        _report(0.6),
    )
    assert answer in final
    assert "verify" in final.lower()
    assert gated is False


@pytest.mark.parametrize("band", ["low", "very_low"])
def test_an_answer_with_invented_specifics_is_withheld(band):
    """
    Fabricated numbers are withheld outright, not shown with a caveat: a
    made-up figure with a warning next to it is still a made-up figure.
    """
    final, gated = apply_evidence_gate(
        "The retention period is 90 days and the policy was signed in 2019.",
        _confidence(20, "Low", band),
        _report(0.0, specifics=["90", "2019"]),
    )
    assert final == INSUFFICIENT_EVIDENCE_MESSAGE
    assert gated is True
    assert "90 days" not in final


def test_weak_retrieval_is_withheld_however_fluent_the_answer():
    """The weak-evidence rule reads the retrieval score, not the answer's prose."""
    final, gated = apply_evidence_gate(
        "Multi-factor authentication is required for administrative accounts.",
        _confidence(70, "Medium", "medium", retrieval_top=0.40),
        _report(1.0),
    )
    assert final == INSUFFICIENT_EVIDENCE_MESSAGE
    assert gated is True


def test_a_paraphrase_over_strong_evidence_is_shown_not_refused():
    """
    Low prose overlap is not evidence of invention. Refusing on it would reject
    every answer written in the user's own words, ELI5 mode included.
    """
    answer = "Administrators have to prove who they are in more than one way."
    final, gated = apply_evidence_gate(
        answer,
        _confidence(50, "Low", "low", retrieval_top=0.92),
        _report(0.0, mean=0.5),
    )
    assert gated is False
    assert answer in final
    assert "verify" in final.lower()


def test_contradictions_are_surfaced_above_a_high_confidence_answer():
    report = _report(1.0, contradictions=[{
        "subject": "retention period",
        "values": [
            {"value": "30 days", "doc_id": "a", "doc_name": "A.pdf", "page": 1},
            {"value": "90 days", "doc_id": "b", "doc_name": "B.pdf", "page": 2},
        ],
    }])
    final, _gated = apply_evidence_gate(
        "Retention is described in both policies.",
        _confidence(80, "High", "high"),
        report,
    )
    assert "Conflicting information found" in final
    assert "A.pdf" in final and "B.pdf" in final


# ---------------------------------------------------------------------------
# End-to-end through the streaming chat pipeline
# ---------------------------------------------------------------------------

def collect(gen):
    """Collect SSE events; the last metadata event is the authoritative one."""
    tokens, metadata = [], {}
    for chunk in gen:
        for line in chunk.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            try:
                data = json.loads(payload)
            except Exception:
                continue
            if data.get("type") == "metadata":
                metadata = data
            elif data.get("type") == "token":
                tokens.append(data.get("text", ""))
    return {"streamed": "".join(tokens), "metadata": metadata}


def _mock_llm(text):
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(content=text)]
    return llm


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_hallucinated_answer_over_one_weak_chunk_is_gated(mock_genai, mock_search):
    """
    The end-to-end Phase 3 regression.

    One passage retrieved, and an answer that asserts facts which are nowhere in
    it. The user must be told the evidence is insufficient rather than shown a
    fluent fabrication with a citation under it.
    """
    doc = Document(
        page_content=POLICY_TEXT,
        metadata={"page": 39, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 4},
    )
    mock_search.return_value = [(doc, 0.576)]  # hybrid 0.712 -> distance 0.576
    mock_genai.return_value = _mock_llm(
        "The system supports 5,000 concurrent users and was certified to "
        "ISO 27001 in 2019 by an external auditor in Frankfurt.\n"
        "Cited Source Indices: 0"
    )

    req = ChatRequest(question="How many concurrent users are supported?",
                      doc_ids=["d1"], history=[], mode="qa")
    result = collect(run_chat_stream(req, "user-a"))
    metadata = result["metadata"]

    assert metadata["evidence_gated"] is True
    assert metadata["content"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert metadata["sources"] == []
    assert metadata["confidence"] < 35
    # The fabricated specifics never reach the persisted answer.
    assert "5,000" not in metadata["content"]
    assert "ISO 27001" not in metadata["content"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_grounded_answer_over_strong_evidence_is_returned(mock_genai, mock_search):
    doc = Document(
        page_content=POLICY_TEXT,
        metadata={"page": 3, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 0},
    )
    mock_search.return_value = [(doc, 0.15)]
    mock_genai.return_value = _mock_llm(
        "The authentication policy requires multi-factor authentication for all "
        "administrative accounts.\nCited Source Indices: 0"
    )

    req = ChatRequest(question="What does the authentication policy require?",
                      doc_ids=["d1"], history=[], mode="qa")
    metadata = collect(run_chat_stream(req, "user-a"))["metadata"]

    assert metadata["evidence_gated"] is False
    assert "multi-factor authentication" in metadata["content"]
    assert metadata["confidence_band"] == "high"
    assert len(metadata["sources"]) == 1
    assert metadata["sources"][0]["supports_answer"] is True


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_no_evidence_returns_insufficient_information(mock_genai, mock_search):
    mock_search.return_value = []
    mock_genai.return_value = _mock_llm("should never be reached")

    req = ChatRequest(question="What is the parental leave allowance?",
                      doc_ids=["d1"], history=[], mode="qa")
    result = collect(run_chat_stream(req, "user-a"))

    assert result["metadata"]["confidence"] == 0
    assert result["metadata"]["sources"] == []


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_conflicting_documents_are_reported_not_merged(mock_genai, mock_search):
    """Document A says 30 days, Document B says 90 days -> say so, cite both."""
    doc_a = Document(
        page_content="The data retention period is 30 days for all audit logs.",
        metadata={"page": 2, "doc_id": "a", "doc_name": "PolicyA.pdf", "chunk_index": 0},
    )
    doc_b = Document(
        page_content="The data retention period is 90 days for all audit logs.",
        metadata={"page": 7, "doc_id": "b", "doc_name": "PolicyB.pdf", "chunk_index": 0},
    )
    mock_search.return_value = [(doc_a, 0.2), (doc_b, 0.25)]
    mock_genai.return_value = _mock_llm(
        "The data retention period is 30 days for all audit logs.\n"
        "Cited Source Indices: 0, 1"
    )

    req = ChatRequest(question="What is the data retention period?",
                      doc_ids=["a", "b"], history=[], mode="qa")
    metadata = collect(run_chat_stream(req, "user-a"))["metadata"]

    assert metadata["contradictions"], "the conflicting values must be detected"
    assert "Conflicting information found" in metadata["content"]
    assert "PolicyA.pdf" in metadata["content"]
    assert "PolicyB.pdf" in metadata["content"]
    # A conflict must not be presented with full confidence.
    assert metadata["confidence_band"] != "high"


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_trace_is_emitted_only_when_requested(mock_genai, mock_search):
    doc = Document(
        page_content=POLICY_TEXT,
        metadata={"page": 1, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 0},
    )
    mock_search.return_value = [(doc, 0.2)]
    mock_genai.return_value = _mock_llm(
        "The authentication policy requires multi-factor authentication for "
        "administrative accounts.\nCited Source Indices: 0"
    )

    without = collect(run_chat_stream(
        ChatRequest(question="What is required?", doc_ids=["d1"], mode="qa"), "u"
    ))["metadata"]
    assert "trace" not in without

    mock_genai.return_value = _mock_llm(
        "The authentication policy requires multi-factor authentication for "
        "administrative accounts.\nCited Source Indices: 0"
    )
    with_trace = collect(run_chat_stream(
        ChatRequest(question="What is required?", doc_ids=["d1"], mode="qa", trace=True), "u"
    ))["metadata"]

    trace = with_trace["trace"]
    assert trace["original_query"] == "What is required?"
    assert "retrieval" in trace
    assert "confidence" in trace
    assert "verification" in trace
