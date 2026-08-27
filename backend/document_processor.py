"""
=============================================================================
DocMind AI - Multi-format ingestion
=============================================================================
One entry point, one output shape, for every supported source.

    file or URL -> pages/sections -> chunks -> metadata -> (embeddings)

Every extractor here returns the exact structure ``process_pdf`` already
returned::

    {"text": str, "metadata": {doc_id, doc_name, page, chunk_index, ...}}

so the chunks flow into the *existing* embedding, retrieval, citation and
verification path unchanged. A second retrieval system for "the other formats"
would fragment relevance scoring and citation rendering for no benefit.

Two honest notes about the metadata:

* ``page`` is real for PDFs. For flat formats (DOCX, TXT, Markdown, HTML) there
  are no pages, so blocks are grouped into fixed-size pseudo-pages and
  ``page_kind`` is set to ``"approximate"``. Citations stay usable while being
  clear about what they mean.
* ``section`` carries the nearest preceding heading where the format exposes
  one. That is a better locator than an approximate page number, which is why
  the citation renderer prefers it.

Dependencies are deliberately stdlib-only. DOCX is a ZIP of XML and HTML has a
parser in the standard library, so supporting them costs no new package in the
deployed image.
"""

import io
import ipaddress
import os
import re
import socket
import zipfile
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_UPLOAD_BYTES
from backend.logger import get_logger
from backend.pdf_processor import process_pdf

logger = get_logger(__name__)

# Extensions accepted by the upload endpoint.
SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm")

# Characters of flat text treated as one pseudo-page. Roughly a dense A4 page,
# so citation page numbers land in a familiar range.
APPROX_PAGE_CHARS = 3000

# Leading bytes that identify a format regardless of the extension claimed.
_MAGIC = {
    b"%PDF": ".pdf",
    b"PK": ".docx",  # any ZIP; the DOCX reader verifies the internal structure
}


class UnsupportedDocumentError(ValueError):
    """The file type is not one we can extract text from."""


class DocumentExtractionError(ValueError):
    """The file is a supported type but yielded no usable text."""


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_extension(filename: str, header: bytes = b"") -> str:
    """
    Determine the format from the filename, cross-checked against magic bytes.

    The header wins when the two disagree: a file named ``report.pdf`` that
    starts with ``PK`` is a renamed archive, and handing it to the PDF reader
    produces a confusing parse error instead of a clear rejection.
    """
    extension = os.path.splitext(filename or "")[1].lower()

    for magic, magic_ext in _MAGIC.items():
        if header.startswith(magic):
            # A ZIP could legitimately be a .docx; anything else claiming to be
            # a PDF while starting with PK is mislabelled.
            if magic_ext == ".docx" and extension in (".docx",):
                return ".docx"
            if magic_ext == ".pdf":
                return ".pdf"
            if magic_ext == ".docx":
                return ".docx"

    if extension in SUPPORTED_EXTENSIONS:
        return extension

    raise UnsupportedDocumentError(
        f"Unsupported file type '{extension or filename}'. Supported formats: "
        + ", ".join(SUPPORTED_EXTENSIONS)
    )


# ---------------------------------------------------------------------------
# Block extraction, per format
# ---------------------------------------------------------------------------

# A "block" is one logical unit of source text with an optional heading.
Block = Dict[str, Any]   # {"text": str, "section": Optional[str]}


_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SETEXT_RE = re.compile(r"^[=-]{3,}\s*$")


def extract_text_blocks(text: str) -> List[Block]:
    """
    Split plain text or Markdown into blocks, tracking the current heading.

    Markdown ATX (``## Heading``) and Setext (underlined) headings both update
    the running section, so a chunk taken from deep inside a long document can
    still cite which section it came from.
    """
    blocks: List[Block] = []
    current_section: Optional[str] = None
    buffer: List[str] = []

    def flush():
        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                blocks.append({"text": body, "section": current_section})
            buffer.clear()

    lines = (text or "").splitlines()
    for index, line in enumerate(lines):
        atx = _MD_HEADING_RE.match(line)
        if atx:
            flush()
            current_section = atx.group(2).strip()
            blocks.append({"text": current_section, "section": current_section})
            continue

        # Setext: the *next* line is the underline.
        if (
            line.strip()
            and index + 1 < len(lines)
            and _SETEXT_RE.match(lines[index + 1])
        ):
            flush()
            current_section = line.strip()
            blocks.append({"text": current_section, "section": current_section})
            continue

        if _SETEXT_RE.match(line):
            continue  # already consumed as an underline

        if not line.strip():
            flush()
        else:
            buffer.append(line)

    flush()
    return blocks


# DOCX is Open Packaging Conventions: word/document.xml holds the body, with
# paragraph text split across <w:t> runs that must be concatenated.
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx_blocks(file_path: str) -> List[Block]:
    """
    Read paragraphs and table cells out of a .docx.

    Implemented against the OPC package directly (zipfile + ElementTree) rather
    than via python-docx: the format is a ZIP of XML, the subset we need is
    small, and this keeps a dependency out of the deployed image.
    """
    import xml.etree.ElementTree as ET

    try:
        with zipfile.ZipFile(file_path) as package:
            if "word/document.xml" not in package.namelist():
                raise DocumentExtractionError(
                    "File is a ZIP archive but not a Word document (no word/document.xml)."
                )
            xml_bytes = package.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise DocumentExtractionError(f"Not a readable .docx file: {exc}") from exc

    root = ET.fromstring(xml_bytes)

    blocks: List[Block] = []
    current_section: Optional[str] = None

    for paragraph in root.iter(f"{_W_NS}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{_W_NS}t")).strip()
        if not text:
            continue

        # Word marks headings with a paragraph style named Heading1..Heading9.
        style = paragraph.find(f"{_W_NS}pPr/{_W_NS}pStyle")
        style_name = style.get(f"{_W_NS}val", "") if style is not None else ""
        if style_name.lower().startswith("heading") or style_name.lower() == "title":
            current_section = text

        blocks.append({"text": text, "section": current_section})

    return blocks


class _HTMLTextExtractor(HTMLParser):
    """
    Pull readable prose out of HTML, tracking headings as sections.

    Script, style, nav and footer content is dropped: it is boilerplate that
    repeats on every page of a site and would otherwise dominate BM25 term
    statistics across a crawled corpus.
    """

    _SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "svg"}
    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
    _BLOCK_TAGS = {"p", "div", "li", "tr", "td", "th", "section", "article", "br", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: List[Block] = []
        self.title: Optional[str] = None
        self._skip_depth = 0
        self._in_title = False
        self._in_heading: Optional[str] = None
        self._current_section: Optional[str] = None
        self._buffer: List[str] = []

    def _flush(self):
        text = " ".join(" ".join(self._buffer).split())
        self._buffer.clear()
        if text:
            self.blocks.append({"text": text, "section": self._current_section})

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._HEADING_TAGS:
            self._flush()
            self._in_heading = tag
        elif tag in self._BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in self._HEADING_TAGS and self._in_heading == tag:
            heading = " ".join(" ".join(self._buffer).split())
            self._buffer.clear()
            self._in_heading = None
            if heading:
                self._current_section = heading
                self.blocks.append({"text": heading, "section": heading})
        elif tag in self._BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title = (self.title or "") + data.strip()
            return
        if data.strip():
            self._buffer.append(data)

    def close(self):
        super().close()
        self._flush()


def extract_html_blocks(html: str) -> Tuple[List[Block], Optional[str]]:
    """Return (blocks, page title) for an HTML document."""
    parser = _HTMLTextExtractor()
    parser.feed(html or "")
    parser.close()
    return parser.blocks, (parser.title or None)


# ---------------------------------------------------------------------------
# Web fetching
# ---------------------------------------------------------------------------

FETCH_TIMEOUT_SECONDS = 15
MAX_FETCH_BYTES = min(MAX_UPLOAD_BYTES, 10 * 1024 * 1024)


def _assert_public_url(url: str) -> None:
    """
    Reject URLs that could be used to reach the host's own network.

    The server fetches whatever URL a user supplies, which is a server-side
    request forgery primitive unless it is constrained: without this check a
    user could point it at ``http://169.254.169.254/`` (cloud instance
    metadata, often including credentials) or at an internal service that is
    unreachable from the internet but trivially reachable from inside the
    deployment. Only http(s) to a publicly routable address is allowed.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs can be ingested.")
    if not parsed.hostname:
        raise ValueError("URL is missing a hostname.")

    try:
        # Resolve every address the name maps to: a hostname can legitimately
        # resolve to one public and one private address.
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host '{parsed.hostname}'.") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError(
                f"Refusing to fetch '{parsed.hostname}': it resolves to the "
                f"non-public address {address}."
            )


def fetch_url(url: str) -> Tuple[str, str, Optional[str]]:
    """
    Download a web page.

    Returns (content, content_type, title). Raises ValueError for anything the
    SSRF guard rejects or that is not text.
    """
    _assert_public_url(url)

    response = requests.get(
        url,
        timeout=FETCH_TIMEOUT_SECONDS,
        headers={"User-Agent": "DocMind-AI/1.0 (+document ingestion)"},
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()

    # Re-check after redirects: an open redirect on a public host is a standard
    # way to smuggle a request to an internal address past the initial check.
    _assert_public_url(response.url)

    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and not (
        content_type.startswith("text/") or content_type in ("application/xhtml+xml",)
    ):
        raise ValueError(f"URL returned '{content_type}', which is not a text document.")

    chunks, total = [], 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        total += len(chunk)
        if total > MAX_FETCH_BYTES:
            raise ValueError(
                f"Page exceeds the {MAX_FETCH_BYTES // (1024 * 1024)} MB fetch limit."
            )
        chunks.append(chunk)

    encoding = response.encoding or "utf-8"
    content = b"".join(chunks).decode(encoding, errors="replace")
    return content, content_type or "text/html", response.url


# ---------------------------------------------------------------------------
# Blocks -> chunks
# ---------------------------------------------------------------------------

def blocks_to_chunks(
    blocks: List[Block],
    doc_id: str,
    doc_name: str,
    source_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Chunk flat blocks into the common representation.

    Blocks are grouped into pseudo-pages so citations have a stable, ordered
    locator, and each chunk keeps the section heading it fell under. Chunk size
    and overlap come from the same config the PDF path uses, so retrieval
    behaves identically across formats.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks: List[Dict[str, Any]] = []
    page = 1
    page_chars = 0

    for block in blocks:
        body = (block.get("text") or "").strip()
        if not body:
            continue

        if page_chars and page_chars + len(body) > APPROX_PAGE_CHARS:
            page += 1
            page_chars = 0
        page_chars += len(body)

        for piece in splitter.split_text(body):
            piece = piece.strip()
            if not piece:
                continue
            metadata: Dict[str, Any] = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page": page,
                # Flat formats have no pages; say so rather than implying the
                # number points at something the reader can turn to.
                "page_kind": "approximate",
                "chunk_index": len(chunks),
            }
            if block.get("section"):
                metadata["section"] = block["section"]
            if source_url:
                metadata["source_url"] = source_url
            chunks.append({"text": piece, "metadata": metadata})

    return chunks


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def process_document(
    file_path: str,
    doc_name: str,
    doc_id: str,
    extension: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract and chunk any supported file into the common representation.

    PDFs are delegated to the existing, working ``process_pdf`` unchanged --
    it already produces real page numbers, which none of the other formats can.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if extension is None:
        with open(file_path, "rb") as handle:
            header = handle.read(8)
        extension = detect_extension(doc_name or file_path, header)

    if extension == ".pdf":
        try:
            return process_pdf(file_path, doc_name, doc_id)
        except (DocumentExtractionError, UnsupportedDocumentError):
            raise
        except Exception as exc:
            # A truncated or corrupt PDF is a bad request, not a server fault.
            # pypdf raises its own error types (PdfStreamError, PdfReadError,
            # and assorted parse errors); translating them here keeps the
            # caller from having to know pypdf's exception taxonomy, and keeps
            # a malformed upload from being reported to the user as a 500.
            raise DocumentExtractionError(
                f"Could not read '{doc_name}' as a PDF: {exc}"
            ) from exc

    if extension == ".docx":
        blocks = extract_docx_blocks(file_path)
    elif extension in (".html", ".htm"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            blocks, _title = extract_html_blocks(handle.read())
    elif extension in (".txt", ".md", ".markdown"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            blocks = extract_text_blocks(handle.read())
    else:
        raise UnsupportedDocumentError(f"No extractor registered for '{extension}'.")

    chunks = blocks_to_chunks(blocks, doc_id, doc_name)
    if not chunks:
        raise DocumentExtractionError(
            f"No readable text found in '{doc_name}'. The file may be empty or image-only."
        )
    return chunks


def process_url(url: str, doc_id: str, doc_name: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Fetch a web page and chunk it.

    Returns (chunks, resolved_document_name, extracted_text). The text is
    returned so the caller can persist it alongside the other uploads and keep
    one storage layout for every source.
    """
    content, content_type, final_url = fetch_url(url)

    if content_type.startswith("text/html") or content_type == "application/xhtml+xml":
        blocks, title = extract_html_blocks(content)
    else:
        blocks, title = extract_text_blocks(content), None

    resolved_name = doc_name or title or urlparse(final_url).netloc or url
    if not resolved_name.lower().endswith((".html", ".htm", ".txt")):
        resolved_name = f"{resolved_name}.html"

    chunks = blocks_to_chunks(blocks, doc_id, resolved_name, source_url=final_url)
    if not chunks:
        raise DocumentExtractionError(
            f"No readable text found at {final_url}. The page may be rendered entirely by JavaScript."
        )

    plain_text = "\n\n".join(block["text"] for block in blocks)
    return chunks, resolved_name, plain_text
