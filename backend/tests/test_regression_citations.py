"""
Citation accuracy.

A citation is a promise that the quoted text is at the stated location. The
failure this suite guards against is quiet: retrieval stitches a chunk together
with the one after it to give the model complete sentences, and that stitched
passage can cross a page boundary. Citing only the first page then sends the
reader to a page that does not contain the words they were shown.
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from backend.chat_engine import run_chat_stream
from backend.document_processor import blocks_to_chunks, extract_text_blocks
from backend.models import ChatRequest, SourceChunk


PAGE_5_TEXT = "Access control is described in this section. Administrative accounts"
PAGE_6_TEXT = "must use multi-factor authentication at every login."


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


# ---------------------------------------------------------------------------
# Adjacent-chunk expansion and page spans
# ---------------------------------------------------------------------------

def _index_with_adjacent_pages():
    """A fake FAISS store holding two consecutive chunks that straddle a page break."""
    chunk_a = Document(
        page_content=PAGE_5_TEXT,
        metadata={"doc_id": "d1", "doc_name": "policy.pdf", "page": 5, "chunk_index": 10},
    )
    chunk_b = Document(
        page_content=PAGE_6_TEXT,
        metadata={"doc_id": "d1", "doc_name": "policy.pdf", "page": 6, "chunk_index": 11},
    )
    store = MagicMock()
    store.similarity_search_with_score.return_value = [(chunk_a, 0.1)]
    store.docstore._dict = {"a": chunk_a, "b": chunk_b}
    return store


@patch("backend.embedding_manager.get_embeddings_model")
@patch("backend.embedding_manager.FAISS.load_local")
@patch("os.path.exists", return_value=True)
def test_a_passage_crossing_a_page_break_records_both_pages(
    mock_exists, mock_load, mock_embeddings
):
    from backend.embedding_manager import search_index

    mock_load.return_value = _index_with_adjacent_pages()
    results = search_index("administrative accounts authentication", ["d1"], top_k=1)

    assert results, "expected the stitched passage to be returned"
    doc, _score = results[0]

    # Both halves are present in the text the model was shown...
    assert PAGE_5_TEXT in doc.page_content
    assert PAGE_6_TEXT in doc.page_content
    # ...so the citation must name the whole span, not just where it started.
    assert doc.metadata["page"] == 5
    assert doc.metadata["page_end"] == 6
    assert doc.metadata["expanded_with_adjacent_chunk"] is True


@patch("backend.embedding_manager.get_embeddings_model")
@patch("backend.embedding_manager.FAISS.load_local")
@patch("os.path.exists", return_value=True)
def test_expansion_within_one_page_records_no_page_span(
    mock_exists, mock_load, mock_embeddings
):
    """page_end is set only when the passage genuinely crosses a boundary."""
    from backend.embedding_manager import search_index

    chunk_a = Document(
        page_content="First half of a paragraph on page five.",
        metadata={"doc_id": "d1", "doc_name": "policy.pdf", "page": 5, "chunk_index": 10},
    )
    chunk_b = Document(
        page_content="Second half of the same paragraph, still on page five.",
        metadata={"doc_id": "d1", "doc_name": "policy.pdf", "page": 5, "chunk_index": 11},
    )
    store = MagicMock()
    store.similarity_search_with_score.return_value = [(chunk_a, 0.1)]
    store.docstore._dict = {"a": chunk_a, "b": chunk_b}
    mock_load.return_value = store

    doc, _ = search_index("paragraph page five", ["d1"], top_k=1)[0]
    assert doc.metadata["page"] == 5
    assert doc.metadata.get("page_end") is None


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_the_page_span_reaches_the_citation(mock_genai, mock_search):
    """The metadata is only useful if it survives all the way to the source card."""
    stitched = Document(
        page_content=f"{PAGE_5_TEXT} {PAGE_6_TEXT}",
        metadata={
            "doc_id": "d1", "doc_name": "policy.pdf", "page": 5, "page_end": 6,
            "chunk_index": 10, "expanded_with_adjacent_chunk": True,
        },
    )
    mock_search.return_value = [(stitched, 0.15)]
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(
        content="Administrative accounts must use multi-factor authentication at "
                "every login.\nCited Source Indices: 0"
    )]
    mock_genai.return_value = llm

    metadata = collect(run_chat_stream(
        ChatRequest(question="What must administrative accounts use?",
                    doc_ids=["d1"], mode="qa"),
        "user-a",
    ))

    source = metadata["sources"][0]
    assert source["page"] == 5
    assert source["page_end"] == 6
    # The text quoted to the user spans both pages, and the citation says so.
    assert PAGE_6_TEXT in source["text"]


@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_the_prompt_tells_the_model_the_real_page_span(mock_genai, mock_search):
    """
    The model is asked to cite, so it must be given the honest locator. Handing
    it "Page 5" for text spanning 5-6 makes an inaccurate citation the model's
    only option.
    """
    stitched = Document(
        page_content=f"{PAGE_5_TEXT} {PAGE_6_TEXT}",
        metadata={"doc_id": "d1", "doc_name": "policy.pdf", "page": 5,
                  "page_end": 6, "chunk_index": 10},
    )
    mock_search.return_value = [(stitched, 0.15)]
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(content="An answer.\nCited Source Indices: 0")]
    mock_genai.return_value = llm

    list(run_chat_stream(
        ChatRequest(question="What is required?", doc_ids=["d1"], mode="qa"), "user-a"
    ))

    system_message = llm.stream.call_args[0][0][0].content
    assert "Pages 5-6" in system_message
    assert "(Page 5)" not in system_message


# ---------------------------------------------------------------------------
# Other locators
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_section_and_source_url_reach_the_citation(mock_genai, mock_search):
    web_passage = Document(
        page_content="Audit logs are retained for 30 days before deletion.",
        metadata={
            "doc_id": "w1", "doc_name": "security-policy.html", "page": 2,
            "page_kind": "approximate", "chunk_index": 3, "section": "Retention",
            "source_url": "https://example.com/security-policy",
        },
    )
    mock_search.return_value = [(web_passage, 0.15)]
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(
        content="Audit logs are retained for 30 days before deletion.\n"
                "Cited Source Indices: 0"
    )]
    mock_genai.return_value = llm

    metadata = collect(run_chat_stream(
        ChatRequest(question="How long are logs kept?", doc_ids=["w1"], mode="qa"),
        "user-a",
    ))

    source = metadata["sources"][0]
    assert source["section"] == "Retention"
    assert source["source_url"] == "https://example.com/security-policy"
    assert source["chunk_index"] == 3
    assert source["doc_name"] == "security-policy.html"

    # The model was given the section and the URL, so it can cite either.
    system_message = llm.stream.call_args[0][0][0].content
    assert "Retention" in system_message
    assert "https://example.com/security-policy" in system_message


def test_chunk_index_is_stable_and_sequential_for_flat_formats():
    """chunk_index is the deep-link locator; it must be dense and ordered."""
    markdown = "\n".join([
        "# Policy", "", "First paragraph of the policy document text.", "",
        "## Retention", "", "Audit logs are retained for 30 days.", "",
        "## Access", "", "Administrative accounts require multi-factor authentication.",
    ])
    chunks = blocks_to_chunks(extract_text_blocks(markdown), "d1", "policy.md")
    indices = [c["metadata"]["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_flat_format_pages_are_flagged_as_approximate():
    """
    A .md has no pages. The number is a synthetic locator, and saying so is the
    difference between a useful ordering hint and a false promise.
    """
    long_text = "\n\n".join(f"Paragraph number {i} with some body text." for i in range(200))
    chunks = blocks_to_chunks(extract_text_blocks(long_text), "d1", "notes.md")

    assert all(c["metadata"]["page_kind"] == "approximate" for c in chunks)
    assert max(c["metadata"]["page"] for c in chunks) > 1


def test_pdf_pages_are_not_flagged_as_approximate():
    """Real page numbers must not be labelled synthetic."""
    from backend.pdf_processor import process_pdf

    import backend.pdf_processor as pdf_module
    with patch.object(pdf_module, "PdfReader") as mock_reader:
        page = MagicMock()
        page.extract_text.return_value = "Real page one content, extracted from a PDF."
        mock_reader.return_value.pages = [page]
        with patch("os.path.exists", return_value=True):
            chunks = process_pdf("/tmp/x.pdf", "real.pdf", "d1")

    assert chunks
    assert chunks[0]["metadata"]["page"] == 1
    assert "page_kind" not in chunks[0]["metadata"]


# ---------------------------------------------------------------------------
# Citation / claim correspondence
# ---------------------------------------------------------------------------

@patch("backend.chat_engine.search_index")
@patch("langchain_google_genai.ChatGoogleGenerativeAI")
def test_a_cited_passage_that_supports_nothing_is_not_marked_as_supporting(
    mock_genai, mock_search
):
    """
    The model may cite a passage it did not use. supports_answer is set by
    verification, not by the model's own claim, so the badge means something.
    """
    relevant = Document(
        page_content="Audit logs are retained for 30 days before deletion.",
        metadata={"doc_id": "d1", "doc_name": "a.pdf", "page": 1, "chunk_index": 0},
    )
    irrelevant = Document(
        page_content="The office cafeteria opens at 8am on weekdays.",
        metadata={"doc_id": "d1", "doc_name": "a.pdf", "page": 9, "chunk_index": 5},
    )
    mock_search.return_value = [(relevant, 0.15), (irrelevant, 0.35)]
    llm = MagicMock()
    llm.stream.return_value = [AIMessageChunk(
        content="Audit logs are retained for 30 days before deletion.\n"
                "Cited Source Indices: 0, 1"
    )]
    mock_genai.return_value = llm

    metadata = collect(run_chat_stream(
        ChatRequest(question="How long are logs kept?", doc_ids=["d1"], mode="qa"),
        "user-a",
    ))

    by_page = {s["page"]: s for s in metadata["sources"]}
    assert by_page[1]["supports_answer"] is True
    assert by_page[9]["supports_answer"] is False


def test_source_chunk_locators_are_optional_for_old_persisted_history():
    """Chat history saved before these fields existed must still deserialise."""
    legacy = {
        "text": "Some passage", "page": 3, "doc_id": "d1",
        "doc_name": "old.pdf", "relevance": 88.0,
    }
    source = SourceChunk(**legacy)
    assert source.page_end is None
    assert source.section is None
    assert source.supports_answer is None
