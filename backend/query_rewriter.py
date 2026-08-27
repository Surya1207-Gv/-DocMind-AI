"""
=============================================================================
DocMind AI - Query rewriting and contextual resolution
=============================================================================
The retriever sees only the string it is handed. When a user asks

    "What does the document say about authentication?"
    "What are its limitations?"

the second query, taken literally, is a bag of two words -- "limitations" and a
pronoun that resolves to nothing. Embedding it retrieves passages about
limitations of anything at all, and BM25 has one usable term. The conversation
already contains the missing noun; it just never reached the retriever, because
history was only ever passed to the generator.

This module closes that gap:

    original question ---> search query (resolved, expanded)
                      \\--> unchanged, for answer generation

The original question is always preserved and is what the generator answers.
Rewriting is a *retrieval* concern; silently answering a question the user did
not ask would be worse than retrieving badly.

Rewriting is heuristic-first: a cheap check decides whether the query even needs
context, and only genuinely ambiguous follow-ups pay for an LLM call. Most
questions are self-contained and skip the round trip entirely.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.logger import get_logger
from backend.text_utils import content_terms, tokenize

logger = get_logger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Master switch for contextual resolution of follow-ups.
QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", True)

# Whether an ambiguous follow-up may spend an LLM call on rewriting. When off,
# the heuristic rewrite is used alone -- still a large improvement over the raw
# pronoun, and free.
QUERY_REWRITE_USE_LLM = _env_bool("QUERY_REWRITE_USE_LLM", True)

# Multi-query retrieval: issue N phrasings and merge the results. Off by
# default because it multiplies embedding calls per question; worth enabling
# where recall matters more than latency.
MULTI_QUERY_ENABLED = _env_bool("MULTI_QUERY_ENABLED", False)
MULTI_QUERY_COUNT = _env_int("MULTI_QUERY_COUNT", 3)

# How many prior turns to consider when resolving a reference.
REWRITE_HISTORY_TURNS = _env_int("REWRITE_HISTORY_TURNS", 4)

# Pronouns and deictic phrases that need an antecedent to mean anything.
_ANAPHORA_RE = re.compile(
    r"\b(?:it|its|it's|they|them|their|theirs|this|that|these|those|he|him|his|she|her|hers|"
    r"the\s+same|the\s+above|the\s+former|the\s+latter|there|such)\b",
    re.IGNORECASE,
)

# Openers that only make sense as a continuation of something already discussed.
_CONTINUATION_RE = re.compile(
    r"^(?:and|also|what\s+about|how\s+about|why|why\s+not|when|where|who|which|"
    r"any\s+others?|anything\s+else|tell\s+me\s+more|more\s+on|elaborate|expand|go\s+on|"
    r"summari[sz]e\s+(?:it|that|this))\b",
    re.IGNORECASE,
)

_REWRITE_PROMPT = """You rewrite follow-up questions for a document search engine.

Conversation so far:
{history}

Follow-up question: "{question}"

Rewrite the follow-up as a single standalone search query that a search engine
could answer without seeing the conversation. Replace every pronoun and
reference ("it", "its", "that", "the same") with the specific subject it refers
to, taken from the conversation above.

Rules:
- Keep it under 25 words.
- Do not answer the question.
- Do not add facts that are not in the conversation.
- If the follow-up is already standalone, repeat it unchanged.

Output only the rewritten query, with no quotes, prefix, or explanation."""

_MULTI_QUERY_PROMPT = """Generate {count} alternative phrasings of this document search query.

Query: "{query}"

Each phrasing should target the same information using different wording --
synonyms, the domain term instead of the everyday term, or a more specific form.

Rules:
- One phrasing per line.
- No numbering, bullets, or quotes.
- Do not answer the query."""


@dataclass
class RewriteResult:
    """What retrieval should search for, and why."""
    original: str
    search_query: str
    variants: List[str] = field(default_factory=list)
    was_rewritten: bool = False
    strategy: str = "passthrough"   # passthrough | heuristic | llm
    reason: str = ""

    @property
    def all_queries(self) -> List[str]:
        """Every query to retrieve for, primary first, deduplicated."""
        seen, queries = set(), []
        for q in [self.search_query, *self.variants]:
            key = q.strip().lower()
            if q.strip() and key not in seen:
                seen.add(key)
                queries.append(q.strip())
        return queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "search_query": self.search_query,
            "variants": self.variants,
            "was_rewritten": self.was_rewritten,
            "strategy": self.strategy,
            "reason": self.reason,
        }


def _history_messages(history: Sequence[Any]) -> List[Dict[str, str]]:
    """Accept both pydantic ChatMessage objects and plain dicts."""
    messages = []
    for message in history or []:
        if isinstance(message, dict):
            role, content = message.get("role", ""), message.get("content", "")
        else:
            role, content = getattr(message, "role", ""), getattr(message, "content", "")
        if content:
            messages.append({"role": role, "content": str(content)})
    return messages


def needs_context(question: str) -> bool:
    """
    Decide whether a question can stand on its own.

    A question is treated as context-dependent when it leans on a pronoun or
    opens like a continuation *and* is short enough that the pronoun is doing
    real work. "What are its limitations?" qualifies. "What are the limitations
    of the OAuth token refresh flow described in section 4?" does not, even
    though it contains "the", because it names its own subject.
    """
    text = (question or "").strip()
    if not text:
        return False

    terms = content_terms(text)

    if _ANAPHORA_RE.search(text) and len(terms) <= 8:
        return True
    # A continuation opener only signals dependence when the question is genuinely
    # short. "Which administrative accounts require MFA?" starts like a
    # continuation but names its own subject, and rewriting it would append
    # unrelated terms from the previous turn.
    if _CONTINUATION_RE.match(text) and len(terms) <= 4:
        return True
    # Very short questions rarely carry their own subject.
    if len(terms) <= 2:
        return True
    return False


def _subject_from_history(messages: Sequence[Dict[str, str]]) -> str:
    """
    Recover the topic under discussion from the most recent user turn.

    Uses the user's own words rather than the assistant's answer: the answer is
    long, and its vocabulary drifts toward whatever the retrieved chunks said,
    which would pull the rewritten query away from what the user is asking about.
    """
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        terms = [t for t in tokenize(message["content"]) if len(t) > 2]
        if terms:
            # Keep original order; the tail of a question usually holds the subject.
            return " ".join(terms[-6:])
    return ""


def heuristic_rewrite(question: str, messages: Sequence[Dict[str, str]]) -> Optional[str]:
    """
    Resolve a follow-up without an LLM by appending the previous subject.

    This does not produce elegant English -- "limitations authentication policy
    document" is not a sentence. It does not need to be: it is fed to an
    embedding model and BM25, both of which care about terms, not grammar. The
    user still sees their original wording.
    """
    subject = _subject_from_history(messages)
    if not subject:
        return None

    question_terms = content_terms(question)
    # Drop subject terms the question already contains, so we add signal
    # rather than duplicating it.
    additions = [t for t in subject.split() if t not in question_terms]
    if not additions:
        return None

    return f"{question.strip()} {' '.join(additions)}".strip()


def llm_rewrite(question: str, messages: Sequence[Dict[str, str]], llm: Any) -> Optional[str]:
    """Ask the model to resolve the reference. Returns None on any failure."""
    if llm is None:
        return None

    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}"
        for m in messages
    )
    prompt = _REWRITE_PROMPT.format(history=transcript, question=question)

    try:
        response = llm.invoke(prompt)
        rewritten = _coerce_text(response).strip().strip('"').strip()
    except Exception as exc:
        logger.warning("[QueryRewrite] LLM rewrite failed, falling back: %s", exc)
        return None

    if not rewritten or len(rewritten) > 300:
        return None
    # A "rewrite" that dropped every content word is worse than the original.
    if not content_terms(rewritten):
        return None
    return rewritten


def _coerce_text(response: Any) -> str:
    """
    Pull plain text out of an LLM response.

    Providers disagree about the shape of ``.content``: a string for most, a
    list of typed blocks for Gemini. Concatenating the latter directly is what
    produced `TypeError: can only concatenate str (not "list") to str`.
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def generate_query_variants(query: str, llm: Any, count: int = MULTI_QUERY_COUNT) -> List[str]:
    """Produce alternative phrasings for multi-query retrieval. Best-effort."""
    if llm is None or count < 1:
        return []

    try:
        response = llm.invoke(_MULTI_QUERY_PROMPT.format(count=count, query=query))
        text = _coerce_text(response)
    except Exception as exc:
        logger.warning("[QueryRewrite] Multi-query generation failed: %s", exc)
        return []

    variants = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if cleaned and cleaned.lower() != query.strip().lower():
            variants.append(cleaned)
    return variants[:count]


def rewrite_query(
    question: str,
    history: Sequence[Any] = (),
    llm: Any = None,
    enable_multi_query: Optional[bool] = None,
) -> RewriteResult:
    """
    Turn a user question into what retrieval should actually search for.

    The original question is preserved on the result and is what the generator
    is asked to answer.
    """
    original = (question or "").strip()
    result = RewriteResult(original=original, search_query=original)

    if not original or not QUERY_REWRITE_ENABLED:
        result.reason = "rewriting disabled" if original else "empty question"
        return result

    messages = _history_messages(history)[-(REWRITE_HISTORY_TURNS * 2):]

    if messages and needs_context(original):
        rewritten = None
        if QUERY_REWRITE_USE_LLM:
            rewritten = llm_rewrite(original, messages, llm)
            if rewritten:
                result.strategy = "llm"

        if not rewritten:
            rewritten = heuristic_rewrite(original, messages)
            if rewritten:
                result.strategy = "heuristic"

        if rewritten and rewritten.strip().lower() != original.lower():
            result.search_query = rewritten
            result.was_rewritten = True
            result.reason = "follow-up resolved against conversation history"
            logger.info("[QueryRewrite] %r -> %r (%s)", original, rewritten, result.strategy)
        else:
            result.reason = "context needed but no antecedent found"
    else:
        result.reason = "question is self-contained"

    use_multi = MULTI_QUERY_ENABLED if enable_multi_query is None else enable_multi_query
    if use_multi:
        result.variants = generate_query_variants(result.search_query, llm)

    return result
