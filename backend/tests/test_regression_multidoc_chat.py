"""
Backend regression tests for multi-document chat.

The UI now sends several document ids on one request. Two properties have to
hold for that to be safe:

  1. A turn is still filed under ONE conversation. The backend persists a turn
     under ``doc_ids[0]``, so the anchor document must stay first and history
     must not fork when extra documents are added mid-conversation.
  2. Citations must name the document each passage actually came from, or a
     multi-document answer is unauditable.
"""

import json
import os

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import backend.database as db
from backend.config import UPLOAD_DIR
from backend.models import ChatRequest
from backend.chat_engine import run_chat_stream


POLICY_A = "Policy A states that audit logs are retained for 30 days."
POLICY_B = "Policy B requires multi-factor authentication for administrative accounts."


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


def _mock_llm(mock_genai, text):
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(content=text)]
    mock_genai.return_value = llm
    return llm


@pytest.fixture
def two_documents():
    """Two documents owned by the default admin user, with files on disk."""
    for doc_id, name in (("doc-a", "policy-a.pdf"), ("doc-b", "policy-b.pdf")):
        db.add_document(doc_id, "default_admin_id", name, 100, "2026-08-27 00:00:00")
        with open(os.path.join(UPLOAD_DIR, f"{doc_id}.pdf"), "w") as handle:
            handle.write("stored contents")
    return ["doc-a", "doc-b"]


def doc_a():
    return Document(
        page_content=POLICY_A,
        metadata={"page": 2, "doc_id": "doc-a", "doc_name": "policy-a.pdf", "chunk_index": 0},
    )


def doc_b():
    return Document(
        page_content=POLICY_B,
        metadata={"page": 7, "doc_id": "doc-b", "doc_name": "policy-b.pdf", "chunk_index": 0},
    )


# ---------------------------------------------------------------------------
# The API surface
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
def test_all_selected_documents_reach_retrieval(mock_search, client, auth_headers, two_documents):
    mock_search.return_value = [(doc_a(), 0.2), (doc_b(), 0.25)]

    resp = client.post("/api/chat", json={
        "question": "What do the policies require?",
        "doc_ids": ["doc-a", "doc-b"],
        "history": [],
        "mode": "qa",
    }, headers=auth_headers)
    assert resp.status_code == 200
    resp.read()

    assert mock_search.call_args[0][1] == ["doc-a", "doc-b"]


@patch("backend.chat_engine.search_index")
def test_a_single_document_request_is_unchanged(mock_search, client, auth_headers, two_documents):
    """The pre-existing workflow must behave exactly as before."""
    mock_search.return_value = [(doc_a(), 0.2)]

    resp = client.post("/api/chat", json={
        "question": "What does policy A say?",
        "doc_ids": ["doc-a"],
        "history": [],
        "mode": "qa",
    }, headers=auth_headers)
    assert resp.status_code == 200
    resp.read()

    assert mock_search.call_args[0][1] == ["doc-a"]


def test_one_unowned_id_rejects_the_whole_request(client, auth_headers, two_documents):
    """
    A multi-document selection is validated as a unit. Silently dropping the
    unauthorised id and answering from the rest would leak the fact that the
    other document exists.
    """
    resp = client.post("/api/chat", json={
        "question": "Compare these",
        "doc_ids": ["doc-a", "not-my-document"],
        "history": [],
        "mode": "qa",
    }, headers=auth_headers)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Conversation anchoring
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_history_is_filed_under_the_first_document(mock_genai, mock_search, two_documents):
    mock_search.return_value = [(doc_a(), 0.2), (doc_b(), 0.25)]
    _mock_llm(mock_genai, "Policy A states that audit logs are retained for 30 days.\nCited Source Indices: 0")

    list(run_chat_stream(
        ChatRequest(question="What is retained?", doc_ids=["doc-a", "doc-b"], mode="qa"),
        "default_admin_id",
    ))

    assert len(db.get_chat_history("default_admin_id", "doc-a")) == 2
    assert db.get_chat_history("default_admin_id", "doc-b") == []


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_adding_a_document_mid_conversation_does_not_fork_history(
    mock_genai, mock_search, two_documents
):
    """
    Turn 1 searches one document, turn 2 searches two. Both belong to the same
    thread, because the anchor stayed first.
    """
    mock_search.return_value = [(doc_a(), 0.2)]
    _mock_llm(mock_genai, "Policy A states that audit logs are retained for 30 days.\nCited Source Indices: 0")
    list(run_chat_stream(
        ChatRequest(question="What is retained?", doc_ids=["doc-a"], mode="qa"),
        "default_admin_id",
    ))

    mock_search.return_value = [(doc_a(), 0.2), (doc_b(), 0.25)]
    _mock_llm(mock_genai, "Policy B requires multi-factor authentication for administrative accounts.\nCited Source Indices: 1")
    list(run_chat_stream(
        ChatRequest(question="And authentication?", doc_ids=["doc-a", "doc-b"], mode="qa"),
        "default_admin_id",
    ))

    assert len(db.get_chat_history("default_admin_id", "doc-a")) == 4
    assert db.get_chat_history("default_admin_id", "doc-b") == []


# ---------------------------------------------------------------------------
# Citations across documents
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_each_citation_names_its_own_document(mock_genai, mock_search, two_documents):
    mock_search.return_value = [(doc_a(), 0.2), (doc_b(), 0.22)]
    _mock_llm(
        mock_genai,
        "Policy A states that audit logs are retained for 30 days. "
        "Policy B requires multi-factor authentication for administrative accounts.\n"
        "Cited Source Indices: 0, 1",
    )

    metadata = collect(run_chat_stream(
        ChatRequest(question="What do the policies say?", doc_ids=["doc-a", "doc-b"], mode="qa"),
        "default_admin_id",
    ))

    by_doc = {s["doc_id"]: s for s in metadata["sources"]}
    assert set(by_doc) == {"doc-a", "doc-b"}
    assert by_doc["doc-a"]["doc_name"] == "policy-a.pdf"
    assert by_doc["doc-a"]["page"] == 2
    assert by_doc["doc-b"]["doc_name"] == "policy-b.pdf"
    assert by_doc["doc-b"]["page"] == 7
    # Each passage is filed under the document it actually came from.
    assert POLICY_A in by_doc["doc-a"]["text"]
    assert POLICY_B in by_doc["doc-b"]["text"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_an_answer_drawn_from_one_document_does_not_cite_the_other(
    mock_genai, mock_search, two_documents
):
    """
    Both documents are in scope, but only one carries the answer. Verification
    marks which passages support it, so the unrelated document is not presented
    as evidence.
    """
    mock_search.return_value = [(doc_a(), 0.2), (doc_b(), 0.22)]
    _mock_llm(
        mock_genai,
        "Policy A states that audit logs are retained for 30 days.\nCited Source Indices: 0",
    )

    metadata = collect(run_chat_stream(
        ChatRequest(question="How long are logs retained?", doc_ids=["doc-a", "doc-b"], mode="qa"),
        "default_admin_id",
    ))

    supporting = [s for s in metadata["sources"] if s.get("supports_answer")]
    assert [s["doc_id"] for s in supporting] == ["doc-a"]
