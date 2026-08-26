"""
Generates the bundled demo PDF (assets/demo/sample.pdf).

The content is original, written for this project, so the repository ships a
document it is free to redistribute. Uses no third-party libraries -- it emits a
minimal, valid PDF using the base-14 Helvetica fonts.

Regenerate with:  python scripts/make_demo_pdf.py
"""

import os

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "demo", "sample.pdf",
)

# Each page is a list of (style, text) lines. Styles: h1, h2, body, bullet, cont.
PAGES = [
    [
        ("h1", "DocMind AI - Field Handbook"),
        ("body", "Version 2.0 | Retrieval-Augmented Generation over private documents"),
        ("body", ""),
        ("h2", "1. PURPOSE"),
        ("body", "This handbook is the reference document bundled with DocMind AI so that"),
        ("body", "the system can be demonstrated immediately, without uploading a file. It"),
        ("body", "is written to exercise every part of the retrieval pipeline: definitions,"),
        ("body", "numbered settings, section headers, and facts that span two pages."),
        ("body", ""),
        ("h2", "2. WHAT RETRIEVAL-AUGMENTED GENERATION IS"),
        ("body", "Retrieval-Augmented Generation is a technique that grounds a language"),
        ("body", "model answer in documents fetched at question time, rather than relying"),
        ("body", "on the parameters the model learned during training. The retrieved text is"),
        ("body", "placed into the prompt as context, and the model is instructed to answer"),
        ("body", "only from that context."),
        ("body", ""),
        ("body", "The benefit is twofold. Answers stay current, because updating the corpus"),
        ("body", "requires no retraining. Answers stay auditable, because every claim can be"),
        ("body", "traced to a specific page of a specific source document."),
        ("body", ""),
        ("h2", "3. WHY GROUNDING MATTERS"),
        ("body", "A language model asked a question outside its knowledge will often produce"),
        ("body", "a fluent but fabricated answer. This failure is called hallucination."),
        ("body", "Grounding reduces hallucination by making the correct answer available in"),
        ("body", "the prompt and by giving the system a defensible way to say it does not"),
        ("body", "know when retrieval returns nothing relevant."),
    ],
    [
        ("h1", "4. THE INGESTION PIPELINE"),
        ("body", ""),
        ("h2", "4.1 TEXT EXTRACTION"),
        ("body", "A PDF is read page by page. Page numbers are preserved at extraction time,"),
        ("body", "because a citation that cannot name a page is not verifiable. Pages that"),
        ("body", "yield no text are skipped; a document that yields no text at all is"),
        ("body", "rejected as scanned or empty rather than indexed as a silent failure."),
        ("body", ""),
        ("h2", "4.2 CHUNKING"),
        ("body", "Extracted text is split recursively on paragraph, sentence, and word"),
        ("body", "boundaries. The default chunk size is 1000 characters with an overlap of"),
        ("body", "150 characters, an overlap of fifteen percent."),
        ("body", ""),
        ("body", "Chunk size is a trade-off. Chunks that are too small lose the surrounding"),
        ("body", "argument and retrieve confidently but answer incompletely. Chunks that are"),
        ("body", "too large dilute the embedding, because a single vector must represent"),
        ("body", "several unrelated ideas at once. Overlap exists so that a sentence falling"),
        ("body", "across a boundary still appears whole in one of the two neighbours."),
        ("body", ""),
        ("h2", "4.3 EMBEDDING"),
        ("body", "Each chunk is converted into a dense vector by an embedding model. Two"),
        ("body", "passages with similar meaning receive vectors that sit close together, so"),
        ("body", "meaning-based search becomes a nearest-neighbour lookup. Chunks are"),
        ("body", "embedded in batches of up to 500 to reduce the number of API round trips."),
        ("body", ""),
        ("h2", "4.4 INDEXING"),
        ("body", "Vectors are stored in a FAISS index, one index per document, written to"),
        ("body", "disk. Because each document is a separate index, a document can be deleted"),
        ("body", "by removing its directory, with no global rebuild."),
    ],
    [
        ("h1", "5. THE QUERY PIPELINE"),
        ("body", ""),
        ("h2", "5.1 HYBRID RETRIEVAL"),
        ("body", "DocMind AI retrieves with two methods at once and merges the result."),
        ("bullet", "Dense vector search finds passages that mean the same thing as the"),
        ("cont", "question even when they share no vocabulary with it."),
        ("bullet", "BM25 keyword search finds passages containing the exact rare terms of"),
        ("cont", "the question, such as an identifier, an acronym, or a product name."),
        ("body", ""),
        ("body", "Each method fails where the other succeeds. Vector search misses exact"),
        ("body", "identifiers because embeddings blur precise tokens. Keyword search misses"),
        ("body", "paraphrase because it cannot match words it has never seen. Running both"),
        ("body", "and combining their scores is called hybrid retrieval."),
        ("body", ""),
        ("h2", "5.2 SCORE FUSION"),
        ("body", "The system fetches three times the requested number of candidates, scores"),
        ("body", "each candidate by both methods, and blends them. The default weighting is"),
        ("body", "0.6 for the vector score and 0.4 for the BM25 score."),
        ("body", ""),
        ("h2", "5.3 RELEVANCE THRESHOLD"),
        ("body", "A candidate whose blended score falls below 0.50 is discarded rather than"),
        ("body", "returned. This is deliberate. Returning the least-bad chunk when nothing"),
        ("body", "relevant exists is what causes a grounded system to hallucinate anyway."),
        ("body", "An empty result is a correct result when the corpus lacks the answer."),
        ("body", ""),
        ("h2", "5.4 CONTEXT EXPANSION"),
        ("body", "Once a chunk is selected, the chunk immediately following it is appended."),
        ("body", "A retrieved passage often ends mid-argument, and the sentence that"),
        ("body", "completes it lives in the next chunk."),
    ],
    [
        ("h1", "6. THE ANSWERING STAGE"),
        ("body", ""),
        ("body", "The surviving chunks are assembled into a context block and sent to the"),
        ("body", "language model with an instruction to answer only from that context and to"),
        ("body", "cite the page it used. The default number of chunks passed to the model is"),
        ("body", "8, controlled by the setting named TOP K."),
        ("body", ""),
        ("h2", "6.1 MULTI-HOP QUESTIONS"),
        ("body", "Some questions cannot be answered by a single retrieval. A question that"),
        ("body", "compares two things, or that requires one fact in order to look up a"),
        ("body", "second, needs more than one pass. For these, a planner first decomposes"),
        ("body", "the question into two or three focused sub-queries, each sub-query is"),
        ("body", "retrieved separately, and the merged evidence is synthesised into one"),
        ("body", "answer. A final verification step checks the drafted claims back against"),
        ("body", "the retrieved context before the answer is shown."),
        ("body", ""),
        ("h2", "7. EVALUATION"),
        ("body", "Retrieval quality is measured, not assumed. The benchmark corpus contains"),
        ("body", "1200 chunks and 60 questions with known correct answers. The metrics used"),
        ("body", "are Recall at k, which asks whether the correct passage appeared at all,"),
        ("body", "and Mean Reciprocal Rank, which asks how near the top it appeared."),
        ("body", ""),
        ("body", "Measuring both matters, because a change can improve one and damage the"),
        ("body", "other. Adding candidates usually raises recall while lowering precision."),
        ("body", ""),
        ("h2", "8. KNOWN LIMITATIONS"),
        ("bullet", "Scanned PDFs without a text layer cannot be read; no OCR is performed."),
        ("bullet", "Tables lose their column structure during text extraction."),
        ("bullet", "Answers are only as good as the corpus; the system cannot know a fact"),
        ("cont", "that no uploaded document contains."),
    ],
]

STYLE = {
    "h1": ("F2", 17, 26),
    "h2": ("F2", 12, 19),
    "body": ("F1", 10.5, 15),
    "bullet": ("F1", 10.5, 15),
    "cont": ("F1", 10.5, 15),   # continuation line of a bullet: indented, no dash
}

PAGE_W, PAGE_H = 612, 792
MARGIN_X, TOP_Y = 62, 730


def esc(text):
    """Escape the three characters that are special inside a PDF string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_content(lines):
    out = ["BT"]
    y = TOP_Y
    for style, text in lines:
        font, size, leading = STYLE[style]
        y -= leading
        if not text:
            continue
        x = MARGIN_X + (14 if style in ("bullet", "cont") else 0)
        body = ("- " + text) if style == "bullet" else text
        out.append("/%s %s Tf" % (font, size))
        out.append("1 0 0 1 %s %.1f Tm" % (x, y))
        out.append("(%s) Tj" % esc(body))
    out.append("ET")
    return "\n".join(out).encode("latin-1")


def main():
    objects = {}
    n_pages = len(PAGES)

    # Fixed object numbering:
    #   1 Catalog | 2 Pages | 3 Helvetica | 4 Helvetica-Bold
    #   5.. page objects, then content streams
    page_ids = [5 + i for i in range(n_pages)]
    content_ids = [5 + n_pages + i for i in range(n_pages)]

    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join("%d 0 R" % pid for pid in page_ids)
    objects[2] = ("<< /Type /Pages /Count %d /Kids [%s] >>" % (n_pages, kids)).encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
    objects[4] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"

    for i, lines in enumerate(PAGES):
        stream = build_content(lines)
        objects[page_ids[i]] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] "
            "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
            "/Contents %d 0 R >>" % (PAGE_W, PAGE_H, content_ids[i])
        ).encode()
        objects[content_ids[i]] = (
            ("<< /Length %d >>\nstream\n" % len(stream)).encode()
            + stream
            + b"\nendstream"
        )

    # Serialise with a correct cross-reference table.
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += ("%d 0 obj\n" % num).encode() + objects[num] + b"\nendobj\n"

    xref_pos = len(out)
    max_obj = max(objects)
    out += ("xref\n0 %d\n" % (max_obj + 1)).encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, max_obj + 1):
        out += ("%010d 00000 n \n" % offsets[num]).encode()
    out += (
        "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (max_obj + 1, xref_pos)
    ).encode()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        fh.write(out)
    print("Wrote %s (%d bytes, %d pages)" % (OUT_PATH, len(out), n_pages))


if __name__ == "__main__":
    main()
