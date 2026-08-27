"""
End-to-end check: real PDF -> real FAISS index -> real retrieval -> real
verification/evidence gate. Only the LLM and the embedding provider are stubbed.

The LLM stub answers strictly by quoting the retrieved context, which is what a
correctly behaving model does; that isolates the pipeline from model quality.

    python scripts/dev/diagnose_chat_e2e.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("OPENROUTER_API_KEY", "diagnostic")
os.environ.setdefault("GEMINI_API_KEY", "diagnostic")
os.environ.setdefault("JWT_SECRET_KEY", "diagnostic")

TMP = tempfile.mkdtemp(prefix="docmind-e2e-")
os.environ["DATA_DIR"] = TMP

import backend.config as config  # noqa: E402

config.UPLOAD_DIR = os.path.join(TMP, "uploads")
config.FAISS_DIR = os.path.join(TMP, "faiss_indices")
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.FAISS_DIR, exist_ok=True)

import backend.embedding_manager as em  # noqa: E402

em.FAISS_DIR = config.FAISS_DIR

from backend.tests.ai_corpus import DeterministicEmbeddings, write_pdf as make_pdf  # noqa: E402

DOC_ID = "acd64fe8-f40f-4179-a871-83dddbd110ac"


class ContextQuotingLLM:
    """
    Stands in for a well-behaved model: answers only by quoting the context it
    was given, and cites source 0. Any refusal therefore comes from the
    pipeline, not from the model.
    """

    def __init__(self):
        self.last_context = ""

    def _extract_context(self, messages):
        system = messages[0].content
        start = system.index("--- CONTEXT ---") + len("--- CONTEXT ---")
        end = system.index("--- END OF CONTEXT ---")
        return system[start:end].strip()

    def stream(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk

        context = self._extract_context(messages)
        self.last_context = context
        sentences = []
        for block in context.split("Content: ")[1:]:
            body = block.split("\n---")[0].strip()
            sentences.append(body)
        answer = " ".join(sentences)[:600] + "\nCited Source Indices: 0"
        yield AIMessageChunk(content=answer)

    def invoke(self, prompt, *args, **kwargs):
        from langchain_core.messages import AIMessage

        return AIMessage(content="")


def run(question, doc_ids):
    from backend.chat_engine import run_chat_stream
    from backend.models import ChatRequest

    metadata = {}
    request = ChatRequest(question=question, doc_ids=doc_ids, history=[], mode="qa", trace=True)
    for chunk in run_chat_stream(request, "diag-user"):
        for line in chunk.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except Exception:
                    continue
                if data.get("type") == "metadata":
                    metadata = data
    return metadata


def main():
    import backend.chat_engine as ce
    from backend.document_processor import process_document
    from backend.embedding_manager import create_and_save_index

    embeddings = DeterministicEmbeddings()
    em.get_embeddings_model = lambda: embeddings

    llm = ContextQuotingLLM()
    ce.get_llm_model = lambda *a, **k: llm

    import langchain_google_genai
    langchain_google_genai.ChatGoogleGenerativeAI = lambda *a, **k: llm
    ce.GEMINI_API_KEY = "diagnostic"

    pdf_path = os.path.join(config.UPLOAD_DIR, f"{DOC_ID}.pdf")
    make_pdf(pdf_path)
    chunks = process_document(pdf_path, "AI.pdf", DOC_ID)
    create_and_save_index(chunks, DOC_ID)

    cases = [
        ("Dartmouth Conference 1956", 4, True),
        ("What year was the Dartmouth Conference?", 4, True),
        ("What are the three core elements that explain how modern AI systems work?", 8, True),
        ("What are the four Vs of Big Data?", 13, True),
        ("What is the difference between a GPU and a TPU?", 19, True),
        ("RFC 8446", 21, True),
        ("What was the price of OpenAI stock in 2018?", None, False),
    ]

    print(f"{'question':56s} {'want':>4} {'got':>4} {'conf':>5} {'gated':>6}  outcome")
    print("-" * 112)
    failures = []
    for question, want_page, should_answer in cases:
        md = run(question, [DOC_ID])
        sources = md.get("sources") or []
        pages = sorted({s["page"] for s in sources})
        got = pages[0] if pages else None
        gated = md.get("evidence_gated")
        answered = bool(sources) and not gated
        content = (md.get("content") or "")[:60].replace("\n", " ")

        if should_answer:
            ok = answered and want_page in pages
        else:
            ok = not answered
        if not ok:
            failures.append((question, content))

        print(f"{question[:56]:56s} {str(want_page):>4} {str(got):>4} "
              f"{md.get('confidence', 0):>5} {str(gated):>6}  "
              f"{'OK ' if ok else 'FAIL'} {content}")

    print()
    if failures:
        print("FAILURES:")
        for question, content in failures:
            print(f"  {question}\n      -> {content}")
    else:
        print("All end-to-end cases behaved correctly.")

    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
