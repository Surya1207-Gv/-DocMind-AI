"""
Full chat path over a REAL index.

`test_regression_retrieval_pipeline.py` proves the right chunk is retrieved.
This proves the rest of the turn does the right thing with it: the answer is
grounded, the citation names the correct document and page, and an unsupported
question is still refused.

Only the LLM and the embedding provider are substituted. The LLM stub answers
strictly by quoting the context it was handed, which is what a well-behaved
model does -- so any refusal here is the pipeline's doing, not the model's.
"""

import json
import os
import shutil
import tempfile

import pytest

import backend.database as db
from backend.tests.ai_corpus import DeterministicEmbeddings, build_index


DOC_ID = "acd64fe8-f40f-4179-a871-83dddbd110ac"


class ContextQuotingLLM:
    """Answers only by quoting the retrieved context, and cites source 0."""

    def __init__(self):
        self.rewrite_reply = ""
        self.search_queries = []

    @staticmethod
    def _context_of(messages):
        system = messages[0].content
        start = system.index("--- CONTEXT ---") + len("--- CONTEXT ---")
        return system[start:system.index("--- END OF CONTEXT ---")].strip()

    def stream(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk

        context = self._context_of(messages)
        bodies = [block.split("\n---")[0].strip() for block in context.split("Content: ")[1:]]
        yield AIMessageChunk(content=" ".join(bodies)[:600] + "\nCited Source Indices: 0")

    def invoke(self, prompt, *args, **kwargs):
        from langchain_core.messages import AIMessage

        return AIMessage(content=self.rewrite_reply)


@pytest.fixture(scope="module")
def chat_env():
    import backend.chat_engine as ce
    import backend.config as config
    import backend.embedding_manager as em
    import langchain_google_genai

    tmp = tempfile.mkdtemp(prefix="docmind-chat-")
    saved = (
        config.UPLOAD_DIR, config.FAISS_DIR, em.FAISS_DIR,
        em.get_embeddings_model, ce.get_llm_model,
        langchain_google_genai.ChatGoogleGenerativeAI, ce.GEMINI_API_KEY,
    )

    config.UPLOAD_DIR = os.path.join(tmp, "uploads")
    config.FAISS_DIR = os.path.join(tmp, "faiss")
    em.FAISS_DIR = config.FAISS_DIR
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.FAISS_DIR, exist_ok=True)

    embeddings = DeterministicEmbeddings()
    em.get_embeddings_model = lambda: embeddings

    llm = ContextQuotingLLM()
    ce.get_llm_model = lambda *a, **k: llm
    langchain_google_genai.ChatGoogleGenerativeAI = lambda *a, **k: llm
    ce.GEMINI_API_KEY = "test"

    build_index(DOC_ID)

    # chat_messages has a foreign key onto documents.
    db.add_document(DOC_ID, "default_admin_id", "AI.pdf", 1024, "2026-08-27 00:00:00")

    yield {"llm": llm}

    (config.UPLOAD_DIR, config.FAISS_DIR, em.FAISS_DIR,
     em.get_embeddings_model, ce.get_llm_model,
     langchain_google_genai.ChatGoogleGenerativeAI, ce.GEMINI_API_KEY) = saved
    shutil.rmtree(tmp, ignore_errors=True)


def ask(question, history=(), mode="qa", trace=False):
    from backend.chat_engine import run_chat_stream
    from backend.models import ChatRequest

    metadata = {}
    request = ChatRequest(
        question=question, doc_ids=[DOC_ID], history=list(history), mode=mode, trace=trace
    )
    for chunk in run_chat_stream(request, "default_admin_id"):
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


# ---------------------------------------------------------------------------
# The reported question, end to end
# ---------------------------------------------------------------------------

def test_the_reported_query_no_longer_refuses(chat_env):
    """
    The exact user-visible failure: this returned "I cannot find any information
    related to your question in the uploaded documents."
    """
    metadata = ask("Dartmouth Conference 1956")

    assert "cannot find any information" not in metadata["content"]
    assert "Insufficient information" not in metadata["content"]
    assert metadata["sources"], "an answer with no citations is not grounded"


def test_the_dartmouth_year_question_is_answered_and_cited(chat_env):
    metadata = ask("What year was the Dartmouth Conference?")

    assert metadata["evidence_gated"] is False
    assert "1956" in metadata["content"]

    source = metadata["sources"][0]
    assert source["doc_name"] == "AI.pdf"
    assert source["doc_id"] == DOC_ID
    assert source["page"] == 4
    assert "Dartmouth Conference" in source["text"]


def test_the_answer_is_verified_against_the_cited_passage(chat_env):
    metadata = ask("What year was the Dartmouth Conference?")

    verification = metadata["verification"]
    assert verification["is_refusal"] is False
    # "1956" is in the evidence, so it must not be flagged as invented.
    assert "1956" not in verification["unsupported_specifics"]
    assert any(s.get("supports_answer") for s in metadata["sources"])


def test_the_three_core_elements_answer_cites_page_8(chat_env):
    metadata = ask("What are the three core elements that explain how modern AI systems work?")

    assert metadata["evidence_gated"] is False
    for element in ("Algorithms", "Data", "Computing Power"):
        assert element in metadata["content"]
    assert 8 in [s["page"] for s in metadata["sources"]]


def test_the_four_vs_answer_cites_page_13(chat_env):
    metadata = ask("What are the four Vs of Big Data?")

    assert metadata["evidence_gated"] is False
    assert "Veracity" in metadata["content"]
    assert 13 in [s["page"] for s in metadata["sources"]]


def test_the_gpu_tpu_answer_cites_page_19(chat_env):
    metadata = ask("What is the difference between a GPU and a TPU?")

    assert metadata["evidence_gated"] is False
    assert 19 in [s["page"] for s in metadata["sources"]]


# ---------------------------------------------------------------------------
# The gate still refuses what it should
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What was the price of OpenAI stock in 2018?",
    "Who won the 2022 FIFA World Cup final?",
])
def test_unsupported_questions_are_still_refused(chat_env, question):
    """
    Admitting candidates on lexical evidence must not have turned the gate into
    a rubber stamp. A confident-sounding answer here would be a hallucination.
    """
    metadata = ask(question)

    refused = (
        "cannot find any information" in metadata["content"]
        or "Insufficient information" in metadata["content"]
    )
    assert refused, f"an unanswerable question produced: {metadata['content'][:120]!r}"
    assert metadata["sources"] == []
    assert metadata["confidence"] == 0


# ---------------------------------------------------------------------------
# Conversation follow-up
# ---------------------------------------------------------------------------

def test_a_follow_up_is_resolved_against_the_conversation(chat_env):
    """
    Turn 2 refers to the subject with a pronoun. Retrieval must search for the
    resolved form, and the answer must still come from the right page.
    """
    from backend.models import ChatMessage

    chat_env["llm"].rewrite_reply = "Who organised the Dartmouth Conference in 1956?"

    history = [
        ChatMessage(role="user", content="What year was the Dartmouth Conference?"),
        ChatMessage(role="assistant", content="The Dartmouth Conference took place in 1956."),
    ]
    metadata = ask("Who organised it?", history=history, trace=True)

    rewrite = metadata["trace"]["rewrite"]
    assert rewrite["original"] == "Who organised it?"
    assert rewrite["was_rewritten"] is True
    assert "dartmouth" in rewrite["search_query"].lower()

    assert metadata["evidence_gated"] is False
    assert 4 in [s["page"] for s in metadata["sources"]]
    assert "McCarthy" in metadata["content"]


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def test_the_trace_reports_the_full_retrieval_story(chat_env):
    metadata = ask("Dartmouth Conference 1956", trace=True)
    trace = metadata["trace"]

    assert trace["original_query"] == "Dartmouth Conference 1956"
    leg = trace["retrieval"]["legs"][0]
    assert leg["candidates"] >= 1
    assert leg["passed_threshold"] >= 1
    assert leg["relevance_threshold"] == 0.5
    assert leg["lexical_coverage_threshold"] == 0.5
    assert leg["selected"][0]["page"] == 4
    assert leg["selected"][0]["lexical_coverage"] == pytest.approx(1.0)
    assert "rejected" in leg
    assert trace["confidence"]["components"]["retrieval_top"] > 0


def test_the_trace_is_not_emitted_unless_requested(chat_env):
    assert "trace" not in ask("Dartmouth Conference 1956", trace=False)
