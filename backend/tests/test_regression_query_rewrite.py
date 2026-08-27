"""
Regression tests for conversation-aware query rewriting (Phases 2.6 and 2.10).

The defect: chat history reached the generator but never the retriever. A
follow-up like "What are its limitations?" was embedded literally -- one content
word and a pronoun pointing at nothing -- so retrieval had almost no signal to
work with even though the antecedent was one message away.

The contract being pinned here: retrieval sees the resolved query, generation
sees the question the user actually typed.
"""

import pytest
from unittest.mock import MagicMock, patch

from backend.models import ChatMessage
from backend.query_rewriter import (
    RewriteResult,
    generate_query_variants,
    heuristic_rewrite,
    needs_context,
    rewrite_query,
)


HISTORY = [
    ChatMessage(role="user", content="What does the document say about authentication?"),
    ChatMessage(
        role="assistant",
        content="The document describes a multi-factor authentication policy for admin accounts.",
    ),
]


def _llm(reply):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=reply)
    return llm


# ---------------------------------------------------------------------------
# Deciding whether a question needs context at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What are its limitations?",
    "Why?",
    "And that one?",
    "Tell me more",
    "What about them?",
])
def test_follow_ups_are_detected_as_context_dependent(question):
    assert needs_context(question) is True


@pytest.mark.parametrize("question", [
    "What is the data retention period specified in the security policy?",
    "Which administrative accounts require multi-factor authentication?",
    "Summarise the onboarding checklist for new contractors in section four.",
])
def test_self_contained_questions_are_left_alone(question):
    assert needs_context(question) is False


def test_a_long_question_containing_a_pronoun_is_still_self_contained():
    """
    "the" and "its" appear constantly in normal prose. Only a question short
    enough that the pronoun is carrying the subject should be rewritten.
    """
    question = (
        "What are the limitations of the OAuth token refresh flow described in "
        "section four of the integration guide?"
    )
    assert needs_context(question) is False


# ---------------------------------------------------------------------------
# Rewriting
# ---------------------------------------------------------------------------

def test_follow_up_is_resolved_against_history_by_the_llm():
    llm = _llm("What are the limitations of the multi-factor authentication policy?")
    result = rewrite_query("What are its limitations?", HISTORY, llm=llm)

    assert result.was_rewritten is True
    assert result.strategy == "llm"
    assert "authentication" in result.search_query.lower()
    # The user's own wording is preserved for generation.
    assert result.original == "What are its limitations?"


def test_the_original_question_is_never_replaced():
    """Answering a question the user did not ask is worse than retrieving badly."""
    llm = _llm("Something entirely different about payroll")
    result = rewrite_query("What are its limitations?", HISTORY, llm=llm)
    assert result.original == "What are its limitations?"


def test_heuristic_rewrite_is_used_when_the_llm_fails():
    """An LLM failure must degrade to the free heuristic, not to no rewriting."""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("provider unavailable")

    result = rewrite_query("What are its limitations?", HISTORY, llm=llm)

    assert result.was_rewritten is True
    assert result.strategy == "heuristic"
    assert "authentication" in result.search_query.lower()


def test_heuristic_rewrite_adds_the_previous_subject():
    messages = [
        {"role": "user", "content": "What does the document say about authentication?"},
    ]
    rewritten = heuristic_rewrite("What are its limitations?", messages)
    assert rewritten is not None
    assert "limitations" in rewritten.lower()
    assert "authentication" in rewritten.lower()


def test_heuristic_rewrite_does_not_duplicate_terms_already_present():
    messages = [{"role": "user", "content": "Tell me about the authentication policy"}]
    rewritten = heuristic_rewrite("What are the authentication policy limits?", messages)
    # Nothing new to add, so no rewrite is offered.
    assert rewritten is None or rewritten.lower().count("authentication") == 1


def test_no_history_means_no_rewrite():
    result = rewrite_query("What are its limitations?", [], llm=_llm("anything"))
    assert result.was_rewritten is False
    assert result.search_query == "What are its limitations?"


def test_self_contained_question_skips_the_llm_entirely():
    """Most questions are self-contained; they must not pay for a round trip."""
    llm = _llm("should not be called")
    result = rewrite_query(
        "What is the data retention period in the security policy?", HISTORY, llm=llm
    )
    assert result.was_rewritten is False
    assert llm.invoke.call_count == 0


def test_an_empty_llm_rewrite_is_rejected():
    result = rewrite_query("What are its limitations?", HISTORY, llm=_llm("   "))
    # Falls back to the heuristic rather than searching for nothing.
    assert result.strategy in ("heuristic", "passthrough")
    assert result.search_query.strip()


def test_gemini_style_block_content_is_handled_in_rewriting():
    """The same list-vs-string content shape that broke streaming (Phase 4)."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content=[{"type": "text", "text": "What are the limitations of the authentication policy?"}]
    )
    result = rewrite_query("What are its limitations?", HISTORY, llm=llm)

    assert result.strategy == "llm"
    assert "authentication policy" in result.search_query.lower()


# ---------------------------------------------------------------------------
# Multi-query retrieval
# ---------------------------------------------------------------------------

def test_query_variants_are_parsed_from_a_line_list():
    llm = _llm(
        "1. What is the MFA requirement for admins?\n"
        "- Which accounts need two-factor authentication?\n"
        "Administrative account login security rules\n"
    )
    variants = generate_query_variants("What does the policy require?", llm, count=3)

    assert len(variants) == 3
    # Numbering and bullets are stripped.
    assert not any(v.startswith(("1.", "-", "*")) for v in variants)


def test_query_variant_generation_failure_returns_no_variants():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")
    assert generate_query_variants("anything", llm) == []


def test_all_queries_deduplicates_and_puts_the_primary_first():
    result = RewriteResult(
        original="q",
        search_query="authentication policy limits",
        variants=["Authentication Policy Limits", "mfa requirements"],
    )
    assert result.all_queries == ["authentication policy limits", "mfa requirements"]


def test_multi_query_can_be_enabled_per_call():
    llm = _llm("phrasing one\nphrasing two")
    result = rewrite_query(
        "What is the retention period for audit logs?",
        [],
        llm=llm,
        enable_multi_query=True,
    )
    assert result.variants
    assert len(result.all_queries) > 1


# ---------------------------------------------------------------------------
# End to end: the retriever sees the resolved query, the generator sees the
# question the user typed.
# ---------------------------------------------------------------------------

def _stream_llm(rewrite_reply, answer):
    """
    One mock standing in for both LLM roles in a turn.

    `.invoke` is the rewriting call; `.stream` is answer generation. Using one
    object mirrors production, where both go to the same configured provider.
    """
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=rewrite_reply)
    from langchain_core.messages import AIMessageChunk
    llm.stream.return_value = [AIMessageChunk(content=answer)]
    return llm


def test_a_follow_up_is_retrieved_on_the_resolved_query():
    """
    Q1 establishes the subject; Q2 refers to it with a pronoun. Retrieval must
    search for the resolved form -- searching for "What are its limitations?"
    literally gives the retriever one content word and a dangling pronoun.
    """
    from langchain_core.documents import Document
    from backend.chat_engine import run_chat_stream
    from backend.models import ChatRequest

    evidence = Document(
        page_content=(
            "The multi-factor authentication policy does not cover service "
            "accounts, and cannot be enforced on legacy VPN clients."
        ),
        metadata={"page": 4, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 2},
    )

    llm = _stream_llm(
        "What are the limitations of the multi-factor authentication policy?",
        "The policy does not cover service accounts and cannot be enforced on "
        "legacy VPN clients.\nCited Source Indices: 0",
    )

    with patch("backend.chat_engine.search_index") as mock_search, \
         patch("backend.chat_engine.get_llm_model", return_value=llm), \
         patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=llm):
        mock_search.return_value = [(evidence, 0.2)]

        request = ChatRequest(
            question="What are its limitations?",
            doc_ids=["d1"],
            history=HISTORY,
            mode="qa",
            trace=True,
        )
        events = list(run_chat_stream(request, "user-a"))

    searched_query = mock_search.call_args[0][0]
    assert "authentication" in searched_query.lower(), (
        f"retrieval searched for {searched_query!r}, which does not name the subject"
    )

    # The generator is asked the user's own question, not the rewritten one.
    generation_messages = llm.stream.call_args[0][0]
    assert generation_messages[-1].content == "What are its limitations?"


def test_the_visible_question_is_never_replaced_by_the_rewrite():
    """
    The rewrite is a retrieval device. What is stored and shown must remain what
    the user typed, or the transcript stops matching what they asked.
    """
    import json
    from langchain_core.documents import Document
    from backend.chat_engine import run_chat_stream
    from backend.models import ChatRequest
    import backend.database as db

    # chat_messages has a foreign key onto documents, so the row must exist for
    # the turn to persist at all.
    db.add_document("d1", "default_admin_id", "policy.pdf", 100, "2026-08-27 00:00:00")

    evidence = Document(
        page_content="The authentication policy does not cover service accounts.",
        metadata={"page": 4, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 2},
    )
    llm = _stream_llm(
        "What are the limitations of the authentication policy?",
        "The policy does not cover service accounts.\nCited Source Indices: 0",
    )

    with patch("backend.chat_engine.search_index") as mock_search, \
         patch("backend.chat_engine.get_llm_model", return_value=llm), \
         patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=llm):
        mock_search.return_value = [(evidence, 0.2)]
        list(run_chat_stream(
            ChatRequest(question="What are its limitations?", doc_ids=["d1"],
                        history=HISTORY, mode="qa"),
            "default_admin_id",
        ))

    stored = db.get_chat_history("default_admin_id", "d1")
    user_turns = [m for m in stored if m["role"] == "user"]
    assert user_turns, "the user turn was not persisted"
    assert user_turns[-1]["content"] == "What are its limitations?"


def test_the_trace_shows_both_the_original_and_the_rewritten_query():
    """A rewrite that changes retrieval must be inspectable, not invisible."""
    import json
    from langchain_core.documents import Document
    from backend.chat_engine import run_chat_stream
    from backend.models import ChatRequest

    evidence = Document(
        page_content="The authentication policy does not cover service accounts.",
        metadata={"page": 4, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 2},
    )
    llm = _stream_llm(
        "What are the limitations of the authentication policy?",
        "The policy does not cover service accounts.\nCited Source Indices: 0",
    )

    metadata = {}
    with patch("backend.chat_engine.search_index") as mock_search, \
         patch("backend.chat_engine.get_llm_model", return_value=llm), \
         patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=llm):
        mock_search.return_value = [(evidence, 0.2)]
        for chunk in run_chat_stream(
            ChatRequest(question="What are its limitations?", doc_ids=["d1"],
                        history=HISTORY, mode="qa", trace=True),
            "user-a",
        ):
            for line in chunk.split("\n"):
                if line.strip().startswith("data: "):
                    try:
                        data = json.loads(line.strip()[6:])
                    except Exception:
                        continue
                    if data.get("type") == "metadata":
                        metadata = data

    rewrite = metadata["trace"]["rewrite"]
    assert rewrite["original"] == "What are its limitations?"
    assert rewrite["was_rewritten"] is True
    assert "authentication" in rewrite["search_query"].lower()


def test_a_first_turn_question_is_retrieved_verbatim():
    """With no history there is nothing to resolve, and nothing is spent trying."""
    from langchain_core.documents import Document
    from backend.chat_engine import run_chat_stream
    from backend.models import ChatRequest

    evidence = Document(
        page_content="The authentication policy requires multi-factor authentication.",
        metadata={"page": 1, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 0},
    )
    llm = _stream_llm("unused", "The policy requires multi-factor authentication.\nCited Source Indices: 0")

    with patch("backend.chat_engine.search_index") as mock_search, \
         patch("backend.chat_engine.get_llm_model", return_value=llm), \
         patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=llm):
        mock_search.return_value = [(evidence, 0.2)]
        list(run_chat_stream(
            ChatRequest(question="What does the document say about authentication?",
                        doc_ids=["d1"], history=[], mode="qa"),
            "user-a",
        ))

    assert mock_search.call_args[0][0] == "What does the document say about authentication?"
    assert llm.invoke.call_count == 0, "no history means no rewriting round trip"
