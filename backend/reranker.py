"""
=============================================================================
DocMind AI - Second-stage reranking
=============================================================================
Retrieval ranking and reranking are different jobs, and the project previously
only had the first.

Hybrid fusion scores every candidate against the query *independently* and in
one pass -- a dense similarity blended with a BM25 term score. That is a recall
device: it is good at making sure the right passage is somewhere in the top
fifteen, and mediocre at deciding which of those fifteen actually answers the
question. Vector similarity in particular rewards passages that are *about* the
topic, which is why a heading listing the words of the question can outrank the
paragraph containing the answer.

A reranker looks at a smaller set with a sharper, more expensive criterion and
reorders it. This module provides two, chosen by the ``RERANKER`` env var:

    lexical (default)  A deterministic scorer over query/passage structure:
                       term coverage, phrase contiguity, answer-shape cues, and
                       a penalty for passages that are mostly boilerplate. Costs
                       microseconds and no API calls, which is what makes it
                       viable on a free-tier host where a cross-encoder is not.

    llm                One batched LLM call scoring the candidate pool. Sharper,
                       but adds a round trip to every question.

    none               Disable; fused order is used as-is.

Deliberately *not* a cross-encoder: a bge/MiniLM reranker means torch plus a
few hundred MB of weights in the image, which does not fit the deployment this
project targets. The lexical reranker is the honest version of that trade -- it
is not as good, and it is stated as such rather than described as one.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.logger import get_logger
from backend.text_utils import bigrams, content_terms, tokenize

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# none | lexical | llm
RERANKER = (os.getenv("RERANKER") or "lexical").strip().lower()

# How much the rerank score displaces the fused retrieval score. 0 keeps the
# original order, 1 ignores retrieval entirely. The default lets reranking
# reorder within a band without letting a lexical fluke promote an
# irrelevant passage above a strongly matching one.
RERANK_WEIGHT = _env_float("RERANK_WEIGHT", 0.4)

# Only the head of the fused list is reranked -- deeper candidates are already
# poor and reordering them changes nothing that reaches the LLM.
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 15)

# Passages shorter than this are usually headings or page furniture: they can
# match a query's words perfectly while containing no answer.
_MIN_SUBSTANTIVE_CHARS = 120

# How much a non-substantive passage is discounted. Chosen so a perfect-matching
# heading lands below a well-matching real paragraph, which is the ordering this
# whole stage exists to produce.
_SHORT_PASSAGE_PENALTY = 0.60
_NO_ASSERTION_PENALTY = 0.85

# Cues that a passage states something rather than merely naming a topic.
_ASSERTION_RE = re.compile(
    r"\b(?:is|are|was|were|must|shall|will|may|can|includes?|requires?|means?|"
    r"refers\s+to|defined\s+as|consists?\s+of|provides?|specifies?|states?)\b",
    re.IGNORECASE,
)

_QUESTION_WORD_RE = re.compile(r"^\s*(who|what|when|where|why|how|which|is|are|does|do|can|should)\b", re.IGNORECASE)

# Returned when the lexical scorer has no basis for an opinion. Distinct from
# 0.0, which means "judged, and judged poorly".
_ABSTAIN = -1.0


@dataclass
class RerankedCandidate:
    """A candidate plus the reranker's view of it."""
    document: Any
    retrieval_score: float
    rerank_score: float
    final_score: float

    def to_trace(self) -> Dict[str, Any]:
        metadata = getattr(self.document, "metadata", {}) or {}
        return {
            "doc_name": metadata.get("doc_name"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "retrieval_score": round(self.retrieval_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "final_score": round(self.final_score, 4),
        }


def lexical_rerank_score(query: str, passage: str) -> float:
    """
    Score how well a passage answers the query, on 0..1.

    Three additive signals, then a multiplicative penalty:

      coverage (0.50)  Fraction of the query's content terms present. The
                       primary signal, and the one BM25 already approximates --
                       recomputed here without BM25's length normalisation,
                       which systematically favours short fragments.

      phrase   (0.30)  Fraction of query bigrams present. This is what fused
                       scoring cannot see: it separates a passage containing
                       "retention period" from one that happens to contain
                       "retention" and "period" in unrelated sentences.

      density  (0.20)  How concentrated the matched terms are. A passage that
                       returns to the query terms repeatedly is more likely to
                       be about them than one that mentions them once in passing.

      substantiveness  A MULTIPLIER, not a bonus. A section heading like
                       "3.2 DATA RETENTION PERIOD" scores a perfect 1.0 on all
                       three signals above -- every query term present, in
                       order, at maximal density -- while containing no answer
                       whatsoever. That is the specific failure this stage
                       exists to correct, and an additive bonus of a tenth or
                       two cannot correct it: the heading has to be discounted,
                       not merely out-pointed. A passage too short to hold an
                       answer, or containing no assertion at all, is scaled
                       down accordingly.
    """
    query_terms = content_terms(query)
    if not query_terms or not passage:
        return _ABSTAIN

    passage_tokens = tokenize(passage)
    passage_terms = set(passage_tokens)
    if not passage_terms:
        return _ABSTAIN

    matched = query_terms & passage_terms
    if not matched:
        # No query term appears in this passage at all, so this scorer has no
        # evidence either way. Returning 0.0 would assert "bad match", which is
        # a different claim and a wrong one for a cross-lingual hit: a Hindi
        # query against an English passage shares no surface forms even when the
        # vector stage matched them correctly. Abstain instead, and let the
        # retrieval score stand.
        return _ABSTAIN

    coverage = len(matched) / len(query_terms)

    query_bigrams = bigrams(tokenize(query))
    phrase = 0.0
    if query_bigrams:
        phrase = len(query_bigrams & bigrams(passage_tokens)) / len(query_bigrams)

    # Occurrences, not distinct terms, so repetition counts.
    occurrences = sum(1 for token in passage_tokens if token in matched)
    density = min(1.0, (occurrences / max(len(passage_tokens), 1)) * 12.0)

    relevance = min(1.0, 0.50 * coverage + 0.30 * phrase + 0.20 * density)

    substantiveness = 1.0
    if len(passage) < _MIN_SUBSTANTIVE_CHARS:
        substantiveness *= _SHORT_PASSAGE_PENALTY
    if not _ASSERTION_RE.search(passage):
        substantiveness *= _NO_ASSERTION_PENALTY

    return relevance * substantiveness


_LLM_RERANK_PROMPT = """Rate how well each passage answers the question.

Question: {query}

{passages}

For each passage output one line: `<number>: <score>` where score is 0-10 for
how directly that passage answers the question. A passage that merely mentions
the topic scores low; one containing the answer scores high.
Output only those lines."""


def _llm_rerank_scores(query: str, passages: Sequence[str], llm: Any) -> Optional[List[float]]:
    """One batched call scoring every candidate. Returns None on any failure."""
    if llm is None or not passages:
        return None

    rendered = "\n\n".join(
        f"[{i + 1}] {p[:900]}" for i, p in enumerate(passages)
    )
    try:
        response = llm.invoke(_LLM_RERANK_PROMPT.format(query=query, passages=rendered))
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        text = str(content)
    except Exception as exc:
        logger.warning("[Rerank] LLM reranking failed, keeping fused order: %s", exc)
        return None

    scores = [0.0] * len(passages)
    seen_any = False
    for match in re.finditer(r"^\s*\[?(\d+)\]?\s*[:.\-]\s*(\d+(?:\.\d+)?)", text, re.MULTILINE):
        index = int(match.group(1)) - 1
        if 0 <= index < len(scores):
            scores[index] = max(0.0, min(1.0, float(match.group(2)) / 10.0))
            seen_any = True

    return scores if seen_any else None


def rerank(
    query: str,
    candidates: Sequence[Tuple[Any, float]],
    llm: Any = None,
    strategy: Optional[str] = None,
    weight: Optional[float] = None,
) -> List[RerankedCandidate]:
    """
    Reorder fused retrieval candidates with a sharper second-stage scorer.

    ``candidates`` is ``[(document, fused_score), ...]`` already sorted by fused
    score. The return is sorted by the blended final score, best first, and
    always contains every input candidate -- reranking reorders, it never drops.
    Filtering stays with the caller so the relevance threshold has one owner.
    """
    active = (strategy or RERANKER).strip().lower()
    blend = RERANK_WEIGHT if weight is None else weight

    if not candidates:
        return []

    if active == "none" or blend <= 0.0:
        return [
            RerankedCandidate(doc, score, 0.0, score)
            for doc, score in candidates
        ]

    head = list(candidates[:RERANK_CANDIDATES])
    tail = list(candidates[RERANK_CANDIDATES:])

    passages = [getattr(doc, "page_content", "") or "" for doc, _ in head]

    scores: Optional[List[float]] = None
    if active == "llm":
        scores = _llm_rerank_scores(query, passages, llm)
        if scores is None:
            # Fall back rather than skipping reranking altogether: the lexical
            # scorer is always available and never fails.
            active = "lexical"

    if scores is None:
        scores = [lexical_rerank_score(query, passage) for passage in passages]

    reranked = []
    for (doc, retrieval_score), rerank_score in zip(head, scores):
        if rerank_score < 0.0:
            # The scorer abstained: keep the retrieval score untouched rather
            # than blending in a zero it did not actually mean.
            reranked.append(
                RerankedCandidate(doc, retrieval_score, 0.0, retrieval_score)
            )
        else:
            reranked.append(
                RerankedCandidate(
                    document=doc,
                    retrieval_score=retrieval_score,
                    rerank_score=rerank_score,
                    final_score=(1.0 - blend) * retrieval_score + blend * rerank_score,
                )
            )
    reranked.sort(key=lambda c: c.final_score, reverse=True)

    # Untouched tail keeps its fused score and stays below the reranked head.
    reranked.extend(
        RerankedCandidate(doc, score, 0.0, score) for doc, score in tail
    )
    return reranked
