"""
A real, multi-page PDF and a deterministic embedding provider, so retrieval
tests can exercise the ACTUAL indexing and search path instead of asserting on
a mocked `search_index` return value.

Mocking retrieval hides exactly the class of bug this exists to catch: the
Dartmouth failure was a scoring/threshold interaction between BM25
normalisation, hybrid fusion and the relevance gate, and every one of those
stages is skipped when the test hands `search_index` a canned answer.

The page contents mirror the document from the bug report: the Dartmouth
passage on page 4, the three core elements on page 8, the four Vs on page 13,
GPU/TPU across pages 18-19, and exact identifiers on page 21.
"""

import hashlib
import math
import os
from typing import List

from langchain_core.embeddings import Embeddings

EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Deterministic embeddings
# ---------------------------------------------------------------------------

def _feature_hashes(text: str) -> List[str]:
    lowered = (text or "").lower()
    tokens = [
        t for t in "".join(
            ch if (ch.isalnum() or ch.isspace()) else " " for ch in lowered
        ).split() if t
    ]
    features = list(tokens)
    features += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    compact = "".join(tokens)
    features += [compact[i:i + 4] for i in range(0, max(0, len(compact) - 3), 2)]
    return features


class DeterministicEmbeddings(Embeddings):
    """
    Hashing embedder with L2-normalised output.

    It reproduces the properties retrieval actually depends on: unit norm (so
    FAISS squared-L2 converts to cosine as `1 - d/2`, matching the production
    model), a fixed dimension, and graded similarity from shared vocabulary.

    It is NOT semantic. A paraphrase sharing no words scores low here where a
    real embedding model would score high, so tests that need genuine semantic
    matching say so explicitly rather than pretending this covers it. That
    limitation is useful here: it makes the vector channel weak, which is
    precisely the condition under which the lexical channel has to carry the
    query -- the condition that was broken.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * EMBEDDING_DIM
        for feature in _feature_hashes(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % EMBEDDING_DIM
            vector[index] += 1.0 if digest[4] & 1 else -1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

PAGES = {
    4: (
        "History of Artificial Intelligence. The Dartmouth Conference in 1956 is widely "
        "considered the start of AI as a distinct field of study. Organised by John "
        "McCarthy, Marvin Minsky, Nathaniel Rochester and Claude Shannon, the workshop "
        "brought together researchers who believed that every aspect of learning could "
        "in principle be described so precisely that a machine can be made to simulate "
        "it. The term artificial intelligence was coined for this event."
    ),
    8: (
        "How Modern AI Systems Work. Three core elements explain how modern AI systems "
        "work: Algorithms, Data, and Computing Power. Algorithms define the procedures "
        "a model follows to learn patterns. Data provides the examples from which those "
        "patterns are drawn. Computing Power determines how large a model can be and "
        "how quickly it can be trained. Progress in any one of the three shifts what is "
        "achievable overall."
    ),
    13: (
        "Big Data Characteristics. The four Vs of Big Data are Volume, Velocity, "
        "Variety and Veracity. Volume refers to the sheer quantity of records. Velocity "
        "describes the rate at which new data arrives. Variety covers the range of "
        "formats involved. Veracity concerns the trustworthiness of the data collected."
    ),
    18: (
        "Hardware for Machine Learning. A GPU is a general purpose parallel processor "
        "originally designed for graphics, and it remains the most flexible accelerator "
        "for training neural networks across many frameworks."
    ),
    19: (
        "A TPU is an application specific integrated circuit built by Google purely for "
        "tensor operations. The difference between a GPU and a TPU is that the GPU "
        "trades peak efficiency for flexibility, while the TPU delivers higher "
        "throughput per watt on the narrower set of operations it supports."
    ),
    21: (
        "Networking Standards. Transport security for model serving endpoints should "
        "follow RFC 8446, which defines TLS 1.3. Checksums are computed with SHA256, "
        "and GPT-4 class models are served behind authenticated gateways."
    ),
}

# A second, unrelated document used for cross-user isolation tests.
OTHER_USER_PAGES = {
    1: (
        "Confidential Payroll Review. The Dartmouth Conference budget line for 1956 was "
        "reconstructed for archival purposes and shows a total of forty thousand "
        "dollars allocated across eight researchers. This document belongs to another "
        "tenant and must never surface in anyone else's search results."
    ),
}


def write_pdf(path: str, pages: dict = None) -> str:
    """
    Write a minimal but genuine multi-page PDF that pypdf can extract text from.

    Hand-assembled rather than generated with a reporting library so the test
    suite gains no dependency: the PDF grammar needed for a page of Helvetica
    text is small.
    """
    pages = PAGES if pages is None else pages
    max_page = max(pages)

    content_streams = []
    for page_no in range(1, max_page + 1):
        text = pages.get(page_no, f"Page {page_no} contains no content relevant to these tests.")
        words, lines, current = text.split(), [], ""
        for word in words:
            if len(current) + len(word) + 1 > 90:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            lines.append(current)

        body = "BT /F1 10 Tf 40 750 Td 14 TL\n"
        for line in lines:
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            body += f"({escaped}) Tj T*\n"
        body += "ET"
        content_streams.append(body)

    out, offsets = [], {}

    def add(obj_num, payload):
        offsets[obj_num] = sum(len(part) for part in out)
        out.append(f"{obj_num} 0 obj\n{payload}\nendobj\n".encode("latin-1"))

    n_pages = len(content_streams)
    page_start = 4
    content_start = page_start + n_pages

    out.append(b"%PDF-1.4\n")
    kids = " ".join(f"{page_start + i} 0 R" for i in range(n_pages))
    add(1, "<< /Type /Catalog /Pages 2 0 R >>")
    add(2, f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    add(3, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i in range(n_pages):
        add(page_start + i,
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_start + i} 0 R >>")
    for i, stream in enumerate(content_streams):
        add(content_start + i, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    xref_pos = sum(len(part) for part in out)
    total = content_start + n_pages
    xref = [f"xref\n0 {total}\n", "0000000000 65535 f \n"]
    for num in range(1, total):
        xref.append(f"{offsets.get(num, 0):010d} 00000 n \n")
    out.append("".join(xref).encode("latin-1"))
    out.append(
        f"trailer\n<< /Size {total} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin-1")
    )

    with open(path, "wb") as handle:
        handle.write(b"".join(out))
    return path


def build_index(doc_id: str, doc_name: str = "AI.pdf", pages: dict = None):
    """
    Ingest a PDF through the real pipeline and build a real FAISS index.

    Returns the chunk list. The caller must have patched
    `backend.embedding_manager.get_embeddings_model` to DeterministicEmbeddings
    and pointed UPLOAD_DIR / FAISS_DIR at a temporary location.
    """
    from backend.config import UPLOAD_DIR
    from backend.document_processor import process_document
    from backend.embedding_manager import create_and_save_index

    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    write_pdf(pdf_path, pages)
    chunks = process_document(pdf_path, doc_name, doc_id)
    create_and_save_index(chunks, doc_id)
    return chunks
