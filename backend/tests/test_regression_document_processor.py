"""
Regression tests for multi-format ingestion (Phase 5) and the SSRF guard around
URL fetching (Phase 11).

Every format must land in the *same* chunk representation the PDF path already
produced, because everything downstream -- embedding, hybrid retrieval,
citation rendering, claim verification -- reads that shape and nothing else.
"""

import io
import os
import zipfile

import pytest
from unittest.mock import MagicMock, patch

from backend.document_processor import (
    APPROX_PAGE_CHARS,
    DocumentExtractionError,
    UnsupportedDocumentError,
    _assert_public_url,
    blocks_to_chunks,
    detect_extension,
    extract_docx_blocks,
    extract_html_blocks,
    extract_text_blocks,
    process_document,
)


MARKDOWN = "\n".join([
    "# Authentication Policy",
    "",
    "All administrative accounts must use multi-factor authentication.",
    "",
    "## Retention",
    "",
    "Audit logs are retained for 30 days.",
])


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def test_extension_is_taken_from_the_filename():
    assert detect_extension("notes.md") == ".md"
    assert detect_extension("report.docx") == ".docx"
    assert detect_extension("page.html") == ".html"


def test_magic_bytes_override_a_mislabelled_extension():
    """A ZIP named .pdf must not be handed to the PDF parser."""
    assert detect_extension("report.pdf", b"%PDF-1.7") == ".pdf"
    assert detect_extension("report.docx", b"PK\x03\x04") == ".docx"


def test_unsupported_extension_is_rejected_clearly():
    with pytest.raises(UnsupportedDocumentError) as exc:
        detect_extension("malware.exe", b"MZ")
    assert "unsupported file type" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Text / Markdown
# ---------------------------------------------------------------------------

def test_markdown_headings_become_sections():
    blocks = extract_text_blocks(MARKDOWN)
    sections = {b["section"] for b in blocks}
    assert "Authentication Policy" in sections
    assert "Retention" in sections

    retention = [b for b in blocks if "30 days" in b["text"]]
    assert retention and retention[0]["section"] == "Retention"


def test_setext_headings_are_recognised():
    blocks = extract_text_blocks("Overview\n========\n\nSome body text here.")
    assert any(b["section"] == "Overview" for b in blocks)


def test_plain_text_splits_on_blank_lines():
    blocks = extract_text_blocks("First paragraph.\n\nSecond paragraph.")
    assert [b["text"] for b in blocks] == ["First paragraph.", "Second paragraph."]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def test_html_extraction_keeps_prose_and_drops_boilerplate():
    html = """
    <html><head><title>Security Policy</title>
    <style>body { color: red; }</style></head>
    <body>
      <nav>Home About Contact</nav>
      <h1>Access Control</h1>
      <p>All administrative accounts require multi-factor authentication.</p>
      <script>console.log('tracking');</script>
      <footer>Copyright 2026</footer>
    </body></html>
    """
    blocks, title = extract_html_blocks(html)
    text = " ".join(b["text"] for b in blocks)

    assert title == "Security Policy"
    assert "multi-factor authentication" in text
    assert "color: red" not in text
    assert "tracking" not in text
    assert "Home About Contact" not in text
    assert "Copyright 2026" not in text


def test_html_headings_become_sections():
    blocks, _title = extract_html_blocks(
        "<h2>Retention</h2><p>Audit logs are retained for 30 days.</p>"
    )
    retention = [b for b in blocks if "30 days" in b["text"]]
    assert retention and retention[0]["section"] == "Retention"


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def build_docx(tmp_path, paragraphs):
    """Build a minimal but real .docx (OPC package) for the extractor to read."""
    body = []
    for text, style in paragraphs:
        style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        body.append(f"<w:p>{style_xml}<w:r><w:t>{text}</w:t></w:r></w:p>")

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W}"><w:body>{"".join(body)}</w:body></w:document>'
    )
    path = os.path.join(str(tmp_path), "sample.docx")
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("word/document.xml", document_xml)
    return path


def test_docx_paragraphs_and_heading_styles_are_extracted(tmp_path):
    path = build_docx(tmp_path, [
        ("Authentication Policy", "Heading1"),
        ("All administrative accounts must use multi-factor authentication.", None),
        ("Retention", "Heading2"),
        ("Audit logs are retained for 30 days.", None),
    ])
    blocks = extract_docx_blocks(path)
    texts = [b["text"] for b in blocks]

    assert "All administrative accounts must use multi-factor authentication." in texts
    retention = [b for b in blocks if "30 days" in b["text"]]
    assert retention and retention[0]["section"] == "Retention"


def test_a_zip_that_is_not_a_docx_is_rejected_clearly(tmp_path):
    path = os.path.join(str(tmp_path), "notes.docx")
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("readme.txt", "not a word document")

    with pytest.raises(DocumentExtractionError) as exc:
        extract_docx_blocks(path)
    assert "word document" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# The common representation
# ---------------------------------------------------------------------------

def test_chunks_match_the_pdf_pipeline_shape():
    chunks = blocks_to_chunks(extract_text_blocks(MARKDOWN), "doc-1", "policy.md")
    assert chunks

    for index, chunk in enumerate(chunks):
        assert set(chunk) == {"text", "metadata"}
        metadata = chunk["metadata"]
        assert metadata["doc_id"] == "doc-1"
        assert metadata["doc_name"] == "policy.md"
        assert metadata["chunk_index"] == index
        assert isinstance(metadata["page"], int)


def test_flat_formats_mark_their_page_numbers_as_approximate():
    """
    A .md has no pages. Emitting a page number without saying it is synthetic
    would send a reader looking for a page that does not exist.
    """
    chunks = blocks_to_chunks(extract_text_blocks(MARKDOWN), "doc-1", "policy.md")
    assert all(c["metadata"]["page_kind"] == "approximate" for c in chunks)


def test_pseudo_pages_advance_through_a_long_document():
    long_blocks = [{"text": "x" * 1000, "section": None} for _ in range(10)]
    chunks = blocks_to_chunks(long_blocks, "doc-1", "long.txt")
    pages = {c["metadata"]["page"] for c in chunks}
    assert max(pages) > 1, "a document longer than one pseudo-page should span pages"
    assert max(pages) <= (10 * 1000 // APPROX_PAGE_CHARS) + 2


def test_source_url_is_carried_onto_every_chunk():
    chunks = blocks_to_chunks(
        extract_text_blocks("Some page content here."),
        "doc-1", "page.html", source_url="https://example.com/policy",
    )
    assert all(c["metadata"]["source_url"] == "https://example.com/policy" for c in chunks)


def test_process_document_dispatches_by_extension(tmp_path):
    path = os.path.join(str(tmp_path), "policy.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(MARKDOWN)

    chunks = process_document(path, "policy.md", "doc-1")
    assert any("multi-factor authentication" in c["text"] for c in chunks)


def test_process_document_delegates_pdfs_to_the_existing_reader(tmp_path):
    """PDFs keep their real page numbers -- that path must not be replaced."""
    path = os.path.join(str(tmp_path), "doc.pdf")
    with open(path, "wb") as handle:
        handle.write(b"%PDF-1.4 stub")

    with patch("backend.document_processor.process_pdf") as mock_pdf:
        mock_pdf.return_value = [{"text": "page one", "metadata": {"page": 1}}]
        chunks = process_document(path, "doc.pdf", "doc-1")

    mock_pdf.assert_called_once()
    assert chunks[0]["metadata"]["page"] == 1


def test_an_empty_file_raises_a_clear_extraction_error(tmp_path):
    path = os.path.join(str(tmp_path), "empty.txt")
    open(path, "w").close()

    with pytest.raises(DocumentExtractionError):
        process_document(path, "empty.txt", "doc-1")


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://localhost/admin",
    "http://127.0.0.1:8000/api/health",
    "http://169.254.169.254/latest/meta-data/",   # cloud instance metadata
    "http://10.0.0.5/internal",
    "http://192.168.1.1/router",
])
def test_private_and_loopback_addresses_are_refused(url):
    with pytest.raises(ValueError):
        _assert_public_url(url)


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/data",
    "gopher://example.com/",
])
def test_non_http_schemes_are_refused(url):
    with pytest.raises(ValueError) as exc:
        _assert_public_url(url)
    assert "http" in str(exc.value).lower()


def test_a_public_address_is_allowed():
    with patch("backend.document_processor.socket.getaddrinfo") as mock_resolve:
        mock_resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 0))]
        _assert_public_url("https://example.com/page")  # must not raise


def test_a_hostname_resolving_to_a_private_address_is_refused():
    """DNS rebinding: a public-looking name pointing at an internal address."""
    with patch("backend.document_processor.socket.getaddrinfo") as mock_resolve:
        mock_resolve.return_value = [(2, 1, 6, "", ("10.1.2.3", 0))]
        with pytest.raises(ValueError) as exc:
            _assert_public_url("https://totally-public.example.com/page")
    assert "non-public" in str(exc.value).lower()
