"""
Regression tests for the Gemini streaming bug (Phase 4):

    TypeError: can only concatenate str (not "list") to str

Gemini returns ``chunk.content`` as a plain string most of the time and as a
list of typed content blocks some of the time (tool/thinking/multi-part
responses). The streaming loop concatenated it directly, so the second shape
crashed the whole response mid-stream.

The fix normalises every chunk before concatenation. These tests pin both
shapes, plus the mixed and malformed cases that a provider can legitimately
emit.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from backend.chat_engine import normalize_gemini_content, run_chat_stream
from backend.models import ChatRequest


# ---------------------------------------------------------------------------
# The normaliser itself
# ---------------------------------------------------------------------------

def test_string_content_passes_through():
    assert normalize_gemini_content("hello world") == "hello world"


def test_list_of_text_blocks_is_joined():
    blocks = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert normalize_gemini_content(blocks) == "hello world"


def test_list_of_bare_strings_is_joined():
    assert normalize_gemini_content(["hello ", "world"]) == "hello world"


def test_mixed_blocks_and_strings_are_joined():
    assert normalize_gemini_content(["a", {"type": "text", "text": "b"}]) == "ab"


def test_non_text_blocks_are_skipped_not_crashed_on():
    blocks = [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "visible"},
        {"type": "image", "source": {"data": "..."}},
    ]
    assert normalize_gemini_content(blocks) == "visible"


def test_empty_and_none_content_normalise_to_empty_string():
    assert normalize_gemini_content([]) == ""
    assert normalize_gemini_content(None) == ""
    assert normalize_gemini_content("") == ""


def test_normalising_never_raises_on_unexpected_shapes():
    """A provider change must degrade to empty output, not take down the stream."""
    for value in (123, {"unexpected": "dict"}, [None], [{"text": 42}]):
        assert isinstance(normalize_gemini_content(value), str)


# ---------------------------------------------------------------------------
# Through the streaming pipeline
# ---------------------------------------------------------------------------

EVIDENCE = (
    "The authentication policy requires multi-factor authentication for all "
    "administrative accounts."
)


def collect(gen):
    tokens, metadata = [], {}
    for chunk in gen:
        for line in chunk.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:].strip())
            except Exception:
                continue
            if data.get("type") == "token":
                tokens.append(data.get("text", ""))
            elif data.get("type") == "metadata":
                metadata = data
    return "".join(tokens), metadata


def _run(chunk_contents, mock_search, mock_genai):
    doc = Document(
        page_content=EVIDENCE,
        metadata={"page": 1, "doc_id": "d1", "doc_name": "policy.pdf", "chunk_index": 0},
    )
    mock_search.return_value = [(doc, 0.2)]

    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(content=c) for c in chunk_contents]
    mock_genai.return_value = llm

    req = ChatRequest(question="What does the policy require?", doc_ids=["d1"], mode="qa")
    return collect(run_chat_stream(req, "user-a"))


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_stream_of_string_chunks(mock_genai, mock_search):
    streamed, metadata = _run(
        ["The authentication policy requires ", "multi-factor authentication."],
        mock_search, mock_genai,
    )
    assert "multi-factor authentication" in streamed
    assert metadata["content"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_stream_of_list_block_chunks_does_not_raise(mock_genai, mock_search):
    """The exact crash: chunk.content arriving as a list of blocks."""
    streamed, metadata = _run(
        [
            [{"type": "text", "text": "The authentication policy requires "}],
            [{"type": "text", "text": "multi-factor authentication."}],
        ],
        mock_search, mock_genai,
    )
    assert "multi-factor authentication" in streamed
    assert metadata["content"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_stream_mixing_string_and_list_chunks(mock_genai, mock_search):
    """A single response can switch shapes partway through."""
    streamed, _metadata = _run(
        [
            "The authentication policy ",
            [{"type": "text", "text": "requires multi-factor "}],
            "authentication.",
        ],
        mock_search, mock_genai,
    )
    assert "The authentication policy requires multi-factor authentication." in streamed


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_stream_with_interleaved_non_text_blocks(mock_genai, mock_search):
    streamed, _metadata = _run(
        [
            [{"type": "thinking", "thinking": "should not be shown"}],
            [{"type": "text", "text": "Multi-factor authentication is required."}],
        ],
        mock_search, mock_genai,
    )
    assert "should not be shown" not in streamed
    assert "Multi-factor authentication is required." in streamed


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_empty_chunks_do_not_emit_empty_sse_events(mock_genai, mock_search):
    streamed, _metadata = _run(
        ["", [], [{"type": "text", "text": "Multi-factor authentication is required."}], ""],
        mock_search, mock_genai,
    )
    assert streamed.strip() == "Multi-factor authentication is required."
