"""
The evidence gate, case by case, end to end.

One test per scenario in the production-readiness matrix, exercised through
``run_chat_stream`` rather than against the scoring functions directly, so what
is asserted is what a user would actually receive.

    A  strong evidence                  -> answer normally
    B  weak evidence                    -> withheld
    C  no evidence                      -> withheld
    D  evidence from the wrong document -> not answered from
    E  another user's document          -> never reaches retrieval
    F  contradictory evidence           -> conflict reported, both sides cited
    G  correct answer, paraphrased      -> still shown
    H  answer contains an invented fact -> withheld

The three gate rules and the signal each reads:

    no/weak evidence     -> retrieval score      (a property of the evidence)
    invented specifics   -> numbers/dates/names  (checkable literally)
    everything else      -> shown, with a caveat below the HIGH band
"""

import json
import os

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import backend.database as db
from backend.config import UPLOAD_DIR
from backend.chat_engine import run_chat_stream
from backend.models import ChatRequest
from backend.verification import INSUFFICIENT_EVIDENCE_MESSAGE


AUTH_POLICY = (
    "The authentication policy requires multi-factor authentication for all "
    "administrative accounts. Audit logs are retained for 30 days before deletion."
)

CATERING_POLICY = (
    "The catering budget covers refreshments for on-site meetings. Requests must "
    "be submitted to the office manager one week in advance."
)


def collect(gen):
    metadata = {}
    for chunk in gen:
        for line in chunk.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:].strip())
                except Exception:
                    continue
                if data.get("type") == "metadata":
                    metadata = data
    return metadata


def passage(text, doc_id="d1", doc_name="policy.pdf", page=1, chunk_index=0):
    return Document(
        page_content=text,
        metadata={"page": page, "doc_id": doc_id, "doc_name": doc_name,
                  "chunk_index": chunk_index},
    )


def run(mock_genai, mock_search, results, answer, question="What is required?", mode="qa"):
    mock_search.return_value = results
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(content=answer)]
    mock_genai.return_value = llm
    return collect(run_chat_stream(
        ChatRequest(question=question, doc_ids=["d1"], mode=mode), "user-a"
    ))


# ---------------------------------------------------------------------------
# A. Strong evidence -> answer normally
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_A_strong_evidence_is_answered_normally(mock_genai, mock_search):
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "The authentication policy requires multi-factor authentication for all "
        "administrative accounts.\nCited Source Indices: 0",
        question="What does the authentication policy require?",
    )
    assert metadata["evidence_gated"] is False
    assert metadata["confidence_band"] == "high"
    assert "multi-factor authentication" in metadata["content"]
    # No caveat on a fully grounded answer.
    assert "verify it against" not in metadata["content"]


# ---------------------------------------------------------------------------
# B. Weak evidence -> withheld
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_B_weak_evidence_is_withheld(mock_genai, mock_search):
    """
    A fluent, well-grounded-looking answer over a passage that barely matches
    the question. The refusal is driven by the retrieval score, which measures
    the evidence rather than the answer's wording.
    """
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 1.0)],   # relevance 0.50, below the answer floor
        "The authentication policy requires multi-factor authentication for "
        "administrative accounts.\nCited Source Indices: 0",
    )
    assert metadata["evidence_gated"] is True
    assert metadata["content"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert metadata["confidence"] == 0
    assert metadata["sources"] == []


# ---------------------------------------------------------------------------
# C. No evidence -> withheld
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_C_no_evidence_is_withheld(mock_genai, mock_search):
    metadata = run(
        mock_genai, mock_search, [],
        "Some confident-sounding answer built on nothing at all.",
    )
    assert metadata["confidence"] == 0
    assert metadata["sources"] == []
    assert "confident-sounding" not in metadata["content"]


# ---------------------------------------------------------------------------
# D. Evidence from the wrong document -> not answered from
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_D_answer_is_not_built_from_an_unrelated_document(mock_genai, mock_search):
    """
    Retrieval surfaced a passage about catering for a question about
    authentication. The model must not answer the authentication question from
    it, and the gate must catch it if it tries.
    """
    metadata = run(
        mock_genai, mock_search,
        [(passage(CATERING_POLICY, doc_id="d1", doc_name="catering.pdf"), 0.15)],
        "Administrative accounts must use multi-factor authentication and "
        "hardware security keys.\nCited Source Indices: 0",
        question="What authentication is required for admin accounts?",
    )
    assert metadata["evidence_gated"] is True
    assert metadata["content"] == INSUFFICIENT_EVIDENCE_MESSAGE
    # The catering passage is not offered as support for an authentication claim.
    assert metadata["sources"] == []


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_D_only_the_document_that_supports_the_claim_is_cited(mock_genai, mock_search):
    """Both documents retrieved; only the relevant one is marked as supporting."""
    metadata = run(
        mock_genai, mock_search,
        [
            (passage(AUTH_POLICY, doc_id="d1", doc_name="auth.pdf"), 0.15),
            (passage(CATERING_POLICY, doc_id="d2", doc_name="catering.pdf"), 0.30),
        ],
        "The authentication policy requires multi-factor authentication for all "
        "administrative accounts.\nCited Source Indices: 0, 1",
        question="What does the authentication policy require?",
    )
    supporting = [s for s in metadata["sources"] if s.get("supports_answer")]
    assert [s["doc_name"] for s in supporting] == ["auth.pdf"]


# ---------------------------------------------------------------------------
# E. Another user's document -> never reaches retrieval
# ---------------------------------------------------------------------------

def test_E_another_users_document_never_reaches_the_gate(client, auth_headers):
    """
    Isolation is enforced before retrieval, not by the gate. A document the
    caller does not own is rejected outright, so its text never enters a prompt
    and can never be quoted back.
    """
    other_user_id = "someone-else"
    db.create_user(other_user_id, "otheruser", "hash", "other@gmail.com", "Other")
    db.add_document("their-doc", other_user_id, "their-secret.pdf", 100, "2026-08-27 00:00:00")
    with open(os.path.join(UPLOAD_DIR, "their-doc.pdf"), "w") as handle:
        handle.write("confidential")

    resp = client.post("/api/chat", json={
        "question": "What does the document say?",
        "doc_ids": ["their-doc"],
        "history": [],
        "mode": "qa",
    }, headers=auth_headers)
    assert resp.status_code == 404


@patch("backend.chat_engine.search_index")
def test_E_an_empty_selection_cannot_reach_another_users_documents(
    mock_search, client, auth_headers
):
    other_user_id = "someone-else-2"
    db.create_user(other_user_id, "otheruser2", "hash", "other2@gmail.com", "Other")
    db.add_document("their-doc-2", other_user_id, "their-secret.pdf", 100, "2026-08-27 00:00:00")

    mock_search.return_value = []
    resp = client.post("/api/chat", json={
        "question": "What do my documents say?",
        "doc_ids": [],
        "history": [],
        "mode": "qa",
    }, headers=auth_headers)
    assert resp.status_code == 200
    resp.read()

    searched = mock_search.call_args[0][1]
    assert "their-doc-2" not in searched


# ---------------------------------------------------------------------------
# F. Contradictory evidence -> conflict reported
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_F_conflicting_evidence_is_reported_with_both_sides(mock_genai, mock_search):
    metadata = run(
        mock_genai, mock_search,
        [
            (passage("The data retention period is 30 days.", doc_id="a",
                     doc_name="PolicyA.pdf", page=2), 0.18),
            (passage("The data retention period is 90 days.", doc_id="b",
                     doc_name="PolicyB.pdf", page=7), 0.20),
        ],
        "The data retention period is 30 days.\nCited Source Indices: 0, 1",
        question="What is the data retention period?",
    )
    assert metadata["contradictions"]
    assert "Conflicting information found" in metadata["content"]
    assert "PolicyA.pdf" in metadata["content"]
    assert "PolicyB.pdf" in metadata["content"]
    assert "30 days" in metadata["content"] and "90 days" in metadata["content"]
    assert metadata["confidence_band"] != "high"


# ---------------------------------------------------------------------------
# G. Correct answer, paraphrased -> still shown
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_G_a_paraphrased_correct_answer_is_not_refused(mock_genai, mock_search):
    """
    Reworded, restructured, and correct. It keeps the domain terms -- which is
    what a model answering from supplied context actually produces -- so it is
    shown rather than refused, with a caveat reflecting the looser match.
    """
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "Every administrative account needs multi-factor authentication, and the "
        "audit logs get deleted once 30 days have passed.\n"
        "Cited Source Indices: 0",
        question="What does the policy require?",
    )
    assert metadata["evidence_gated"] is False
    assert "multi-factor authentication" in metadata["content"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_G_known_limitation_a_fully_disjoint_paraphrase_is_refused(mock_genai, mock_search):
    """
    KNOWN LIMITATION, asserted so it is a deliberate trade rather than a
    surprise.

    This answer is correct. It shares no vocabulary at all with the source --
    "administrators"/"administrative", "records"/"logs", "kept"/"retained" all
    miss, because the tokenizer does not stem. To a lexical scorer it is
    indistinguishable from an answer about a different document (case D), and
    the two cannot both be handled correctly without real entailment checking.

    The project's stated contract is to prefer "insufficient information" over
    an ungrounded answer, so in the modes that are told to use the context's own
    facts, this errs toward refusal. ELI5 -- the mode where full rewording is
    the instruction -- is exempted, and is covered by the test below.

    Lifting this properly means an entailment model or embedding-based claim
    similarity, which is the documented next step for VERIFY_MODE.
    """
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "Administrators must prove who they are in more than one way. Access "
        "records are kept for a limited period and then removed.\n"
        "Cited Source Indices: 0",
        question="What does the policy require?",
    )
    assert metadata["evidence_gated"] is True


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_G_an_eli5_answer_is_not_refused_for_avoiding_the_source_wording(
    mock_genai, mock_search
):
    """
    Regression. ELI5 mode is instructed to use everyday language and no
    technical jargon -- precisely the vocabulary a lexical verifier looks for.
    An earlier version of the gate therefore refused every ELI5 answer.
    """
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "Imagine your treehouse has a special lock. To get in you need a secret "
        "word AND a key. Grown-ups who look after it must use both, not just "
        "one. It also keeps a notebook of who visited and throws it away after "
        "a while.\nCited Source Indices: 0",
        question="Explain the login rules simply",
        mode="eli5",
    )
    assert metadata["evidence_gated"] is False, "a faithful ELI5 answer must not be withheld"
    assert "treehouse" in metadata["content"]


# ---------------------------------------------------------------------------
# H. Invented specifics -> withheld
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_H_an_invented_number_is_withheld(mock_genai, mock_search):
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "Audit logs are retained for 90 days before deletion.\nCited Source Indices: 0",
        question="How long are audit logs kept?",
    )
    assert metadata["evidence_gated"] is True
    assert "90 days" not in metadata["content"]
    assert "90" in metadata["verification"]["unsupported_specifics"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_H_an_invented_name_is_withheld(mock_genai, mock_search):
    """Fabricated organisations and places are as checkable as fabricated numbers."""
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "The system was certified to ISO 27001 in 2019 by an auditor in "
        "Frankfurt.\nCited Source Indices: 0",
        question="Is the system certified?",
    )
    assert metadata["evidence_gated"] is True
    assert "Frankfurt" not in metadata["content"]
    assert "frankfurt" in metadata["verification"]["unsupported_specifics"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_H_a_number_present_in_the_evidence_is_kept(mock_genai, mock_search):
    """The check must not punish an answer for quoting the source correctly."""
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "Audit logs are retained for 30 days before deletion.\nCited Source Indices: 0",
        question="How long are audit logs kept?",
    )
    assert metadata["evidence_gated"] is False
    assert "30 days" in metadata["content"]
    assert metadata["verification"]["unsupported_specifics"] == []


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_H_a_withheld_answer_reports_no_confidence(mock_genai, mock_search):
    """
    "Insufficient information" next to a 75% confidence badge is a contradiction
    the reader cannot resolve.
    """
    metadata = run(
        mock_genai, mock_search,
        [(passage(AUTH_POLICY), 0.15)],
        "Audit logs are retained for 90 days.\nCited Source Indices: 0",
    )
    assert metadata["evidence_gated"] is True
    assert metadata["confidence"] == 0
    assert metadata["confidence_band"] == "very_low"
    assert metadata["confidence_label"] == "Low"
