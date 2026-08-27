"""
Regression tests for user isolation (Phase 12).

The defect found during the audit: the ownership check in the chat endpoints
iterated over the requested document ids, which does nothing when that list is
empty. An empty selection then reached ``search_index``, whose own fallback was
to load *every* FAISS index on disk. A user asking a question with no document
selected retrieved from other users' documents and got them back as cited
sources.

The fix has two layers, and both are tested here:
  1. the API resolves an empty selection to the caller's own documents;
  2. ``search_index`` treats an empty list as "no documents", and only an
     explicit ``None`` as "everything on disk".
"""

import json
import os

import pytest
from unittest.mock import patch

import backend.database as db
from backend.auth import hash_password
from backend.config import UPLOAD_DIR


def make_user(client, username, password="password123"):
    resp = client.post("/api/auth/register", json={
        "username": username,
        "password": password,
        "email": f"{username}@gmail.com",
        "full_name": f"{username.title()} Person",
    })
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    user = db.get_user_by_username(username)
    return user["id"], {"Authorization": f"Bearer {token}"}


def give_document(user_id, doc_id, name):
    """Attach a document (and its on-disk file) to a user."""
    db.add_document(doc_id, user_id, name, 100, "2026-08-27 00:00:00")
    with open(os.path.join(UPLOAD_DIR, f"{doc_id}.pdf"), "w") as handle:
        handle.write("stored document contents")
    return doc_id


@pytest.fixture
def two_users(client):
    alice_id, alice_headers = make_user(client, "alice")
    bob_id, bob_headers = make_user(client, "bob")
    give_document(alice_id, "alice-doc", "alice-secret.pdf")
    give_document(bob_id, "bob-doc", "bob-secret.pdf")
    return {
        "alice": (alice_id, alice_headers),
        "bob": (bob_id, bob_headers),
    }


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def test_document_listing_shows_only_your_own(client, two_users):
    _alice_id, alice_headers = two_users["alice"]
    names = [d["name"] for d in client.get("/api/documents", headers=alice_headers).json()]
    assert names == ["alice-secret.pdf"]


def test_cannot_delete_another_users_document(client, two_users):
    _bob_id, bob_headers = two_users["bob"]
    resp = client.delete("/api/documents/alice-doc", headers=bob_headers)
    assert resp.status_code == 404
    # And it is still there for its owner.
    _alice_id, alice_headers = two_users["alice"]
    assert len(client.get("/api/documents", headers=alice_headers).json()) == 1


def test_cannot_read_another_users_analytics(client, two_users):
    db.save_analytics(
        "alice-doc", 100, 2, 1, "Medium",
        ["Alice's confidential summary"], [], [], ["A question"],
    )
    _bob_id, bob_headers = two_users["bob"]
    resp = client.get("/api/analytics/alice-doc", headers=bob_headers)
    assert resp.status_code == 404


def test_cannot_read_another_users_chat_history(client, two_users):
    alice_id, _ = two_users["alice"]
    db.save_chat_message(
        "m1", alice_id, "alice-doc", "user", "Alice's private question", 0, [],
        "2026-08-27 00:00:00",
    )
    _bob_id, bob_headers = two_users["bob"]
    assert client.get("/api/chat/history/alice-doc", headers=bob_headers).status_code == 404


def test_cannot_clear_another_users_chat_history(client, two_users):
    _bob_id, bob_headers = two_users["bob"]
    assert client.delete("/api/chat/history/alice-doc", headers=bob_headers).status_code == 404


def test_active_chats_are_per_user(client, two_users):
    alice_id, alice_headers = two_users["alice"]
    bob_id, bob_headers = two_users["bob"]
    db.save_chat_message("m1", alice_id, "alice-doc", "user", "hi", 0, [], "2026-08-27 00:00:00")

    assert client.get("/api/chats/active", headers=alice_headers).json() == ["alice-doc"]
    assert client.get("/api/chats/active", headers=bob_headers).json() == []


# ---------------------------------------------------------------------------
# Retrieval scoping -- the cross-user leak
# ---------------------------------------------------------------------------

def test_chat_against_another_users_document_is_rejected(client, two_users):
    _bob_id, bob_headers = two_users["bob"]
    resp = client.post("/api/chat", json={
        "question": "What is in this document?",
        "doc_ids": ["alice-doc"],
        "history": [],
        "mode": "qa",
    }, headers=bob_headers)
    assert resp.status_code == 404


@patch("backend.chat_engine.search_index")
def test_empty_selection_is_scoped_to_the_callers_own_documents(mock_search, client, two_users):
    """
    The leak: an empty doc_ids list used to reach search_index unfiltered, whose
    fallback loaded every index on disk -- including other users'.
    """
    mock_search.return_value = []
    _bob_id, bob_headers = two_users["bob"]

    resp = client.post("/api/chat", json={
        "question": "What do my documents say?",
        "doc_ids": [],
        "history": [],
        "mode": "qa",
    }, headers=bob_headers)
    assert resp.status_code == 200
    resp.read()

    assert mock_search.called, "retrieval should still run for an empty selection"
    searched_ids = mock_search.call_args[0][1]
    assert searched_ids == ["bob-doc"]
    assert "alice-doc" not in searched_ids


@patch("backend.agent_engine.search_index")
def test_agent_endpoint_scopes_an_empty_selection_too(mock_search, client, two_users):
    mock_search.return_value = []
    _bob_id, bob_headers = two_users["bob"]

    resp = client.post("/api/agent/query", json={
        "question": "Summarise everything",
        "doc_ids": [],
        "mode": "deep",
    }, headers=bob_headers)
    assert resp.status_code == 200

    for call in mock_search.call_args_list:
        assert call[0][1] == ["bob-doc"]


def test_search_index_treats_an_empty_list_as_no_documents():
    """
    Defence in depth. Even if a future caller forgets to scope the selection,
    an empty list must not mean "load every index on disk".
    """
    from backend.embedding_manager import search_index

    assert search_index("anything at all", []) == []


def test_compare_rejects_another_users_document(client, two_users):
    _bob_id, bob_headers = two_users["bob"]
    resp = client.post("/api/compare", json={
        "doc_ids": ["alice-doc", "bob-doc"],
        "question": "Compare these",
    }, headers=bob_headers)
    assert resp.status_code == 404


def test_relationships_only_consider_your_own_documents(client, two_users):
    _bob_id, bob_headers = two_users["bob"]
    resp = client.get("/api/documents/relationships", headers=bob_headers)
    assert resp.status_code == 200
    # Bob owns exactly one document, so there is no pair to relate -- Alice's
    # document must not be pulled in to make one.
    assert resp.json()["document_count"] == 1
    assert resp.json()["relationships"] == []


def test_rag_trace_is_scoped_to_the_caller(client, two_users):
    _bob_id, bob_headers = two_users["bob"]
    resp = client.post("/api/rag/trace", json={
        "question": "What is in the documents?",
        "doc_ids": ["alice-doc"],
    }, headers=bob_headers)
    assert resp.status_code == 404
