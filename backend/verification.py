"""
=============================================================================
DocMind AI - Answer verification, evidence gating and confidence
=============================================================================
This module answers a question the retrieval score cannot: *does the evidence
we retrieved actually support the answer the model just wrote?*

The distinction matters. A retrieval similarity of 0.71 says "this passage is
about roughly the same topic as the question". It says nothing about whether
the model's third sentence -- the one asserting a 30-day retention period --
appears anywhere in that passage. Treating the first number as if it answered
the second question is how a system ends up presenting a confident answer built
on one weak chunk.

The pipeline implemented here:

    answer -> split into claims
           -> score each claim against the retrieved evidence
           -> aggregate into a confidence band
           -> gate: answer / warn / refuse

Everything is lexical and deterministic by default. That is a deliberate
constraint, not a shortcut: verification runs on every single answer, so it has
to be fast enough to not double the latency and cheap enough to not double the
bill on a free-tier host. An optional LLM-based verifier (VERIFY_MODE=llm) is
available for higher fidelity where that trade is worth making.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config import LEXICAL_COVERAGE_THRESHOLD
from backend.text_utils import (
    bigrams,
    content_terms,
    extract_dates,
    extract_numbers,
    extract_proper_nouns,
    split_sentences,
    tokenize,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- Tunables ---------------------------------------------------------------
# A claim needs this much lexical grounding in the evidence to count as supported.
CLAIM_SUPPORT_THRESHOLD = _env_float("CLAIM_SUPPORT_THRESHOLD", 0.55)

# Credit given to a claim that invents no specifics but whose prose does not
# overlap the source closely -- the signature of a legitimate paraphrase.
#
# Lexical overlap cannot distinguish "restated in simpler words" from "made up",
# so demanding it would refuse every ELI5 answer, a mode explicitly instructed
# to avoid the document's vocabulary. What CAN be checked literally is the
# specifics: numbers, dates and names. A claim that asserts none the evidence
# lacks is not verified, but neither is it contradicted -- partial credit says
# exactly that, and lands the answer in the MEDIUM band where it is shown with
# a caveat rather than withheld.
PARAPHRASE_CREDIT = _env_float("PARAPHRASE_CREDIT", 0.5)

# Minimum topical connection before partial credit applies at all, so an answer
# about an unrelated subject gets none.
PARAPHRASE_MIN_OVERLAP = _env_float("PARAPHRASE_MIN_OVERLAP", 0.15)

# The best retrieved passage must reach this hybrid score before the system is
# willing to answer at all. This is the "weak evidence" gate, and it reads the
# retrieval score deliberately: retrieval quality is a property of the evidence,
# measurable before a single token is generated and unaffected by how the model
# chose to word its reply. RELEVANCE_THRESHOLD (0.50) decides what is worth
# putting in the prompt; this decides what is worth answering from.
EVIDENCE_MIN_TOP_SCORE = _env_float("EVIDENCE_MIN_TOP_SCORE", 0.55)

# Confidence band boundaries (0-100).
CONFIDENCE_HIGH = _env_float("CONFIDENCE_HIGH", 75.0)
CONFIDENCE_MEDIUM = _env_float("CONFIDENCE_MEDIUM", 55.0)
CONFIDENCE_LOW = _env_float("CONFIDENCE_LOW", 35.0)

# Master switch. Off restores the previous retrieval-score-only behaviour.
VERIFICATION_ENABLED = _env_bool("VERIFICATION_ENABLED", True)

# When true, a LOW-band answer is replaced with the insufficient-evidence
# message rather than shown with a caveat.
EVIDENCE_GATE_ENABLED = _env_bool("EVIDENCE_GATE_ENABLED", True)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient information in the selected documents to answer this reliably."
)

MEDIUM_CONFIDENCE_WARNING = (
    "_Note: this answer is only partially supported by the selected documents. "
    "Please verify it against the cited passages before relying on it._"
)

CONTRADICTION_NOTICE = "**Conflicting information found across the selected documents.**"

# Sentences that are meta-commentary rather than factual assertions. Scoring
# these as "unsupported claims" would punish an answer for being well-hedged.
_NON_CLAIM_PATTERNS = re.compile(
    r"^(?:"
    r"(?:the\s+)?(?:document|documents|context|text|passage|source|sources)\s+"
    r"(?:does\s+not|do\s+not|doesn't|don't)\b"
    r"|i\s+(?:cannot|can't|could not|couldn't|do not|don't|am\s+unable)\b"
    r"|(?:based|according)\s+(?:on|to)\s+the\s+(?:document|context|provided|selected)"
    r"|(?:in\s+)?summary\b"
    r"|however\b|therefore\b|note\s+that\b"
    r"|there\s+is\s+no\s+(?:information|mention|reference)\b"
    r")",
    re.IGNORECASE,
)

# Phrases that mean the model itself declined to answer.
REFUSAL_PHRASES = (
    "cannot find that information", "cannot find this information", "not find that information",
    "not find this information", "not present in the uploaded documents", "not mentioned in the provided",
    "information is not in the", "not found in the uploaded", "do not contain information",
    "does not contain information", "no information about", "unable to find", "cannot find information",
    "not found in the provided", "not mention this", "not mention that",
    "insufficient information in the selected",
)


# ---------------------------------------------------------------------------
# Claim extraction and scoring
# ---------------------------------------------------------------------------

@dataclass
class ClaimVerdict:
    """One factual assertion from the answer, scored against the evidence."""
    claim: str
    support: float
    supported: bool
    best_source_index: Optional[int] = None
    # Numbers, dates and names in this claim that the evidence does not contain.
    missing_specifics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "support": round(self.support, 3),
            "supported": self.supported,
            "source_index": self.best_source_index,
            "missing_specifics": self.missing_specifics,
        }


@dataclass
class VerificationReport:
    """Aggregate verdict over an answer."""
    claims: List[ClaimVerdict] = field(default_factory=list)
    supported_ratio: float = 0.0
    unsupported_claims: List[str] = field(default_factory=list)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    is_refusal: bool = False
    checked: bool = True
    # Numbers, dates and names asserted by the answer that appear nowhere in the
    # evidence. Tracked separately from supported_ratio because it means
    # something different: not "we could not confirm this" but "this was
    # invented". The gate treats the two differently.
    unsupported_specifics: List[str] = field(default_factory=list)
    # Average per-claim grounding. Unlike supported_ratio (a count of claims
    # over a threshold) this degrades smoothly, which matters because a faithful
    # paraphrase lands mid-scale rather than on either side of a cliff.
    mean_support: float = 0.0
    # The best-grounded single claim. Distinguishes "restated in other words"
    # (at least one claim connects) from "about something else entirely"
    # (nothing connects at all).
    max_support: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "claims_total": len(self.claims),
            "claims_supported": sum(1 for c in self.claims if c.supported),
            "supported_ratio": round(self.supported_ratio, 3),
            "mean_support": round(self.mean_support, 3),
            "max_support": round(self.max_support, 3),
            "unsupported_claims": self.unsupported_claims,
            "unsupported_specifics": self.unsupported_specifics,
            "contradictions": self.contradictions,
            "is_refusal": self.is_refusal,
            "claims": [c.to_dict() for c in self.claims],
        }


def is_refusal(answer: str) -> bool:
    """True when the model said it could not answer from the documents."""
    lowered = (answer or "").lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def extract_claims(answer: str) -> List[str]:
    """
    Pull the factual assertions out of a generated answer.

    Markdown headings, bullets and meta-commentary are dropped: a heading like
    "## Why this matters" asserts nothing, and counting it as an unsupported
    claim would make well-structured DEEP-mode answers score worse than terse
    ones for no good reason.
    """
    claims: List[str] = []
    for sentence in split_sentences(answer or ""):
        stripped = sentence.strip()
        if len(stripped) < 15:
            continue
        if _NON_CLAIM_PATTERNS.match(stripped):
            continue
        # A sentence with almost no content words carries no checkable fact.
        if len(content_terms(stripped)) < 3:
            continue
        claims.append(stripped)
    return claims


def score_claim(claim: str, evidence_texts: Sequence[str]) -> Tuple[float, Optional[int], List[str]]:
    """
    Score how well one claim is grounded in the evidence.

    Returns (support, best_evidence_index, specifics_absent_from_evidence).

    The scoring separates two questions that look similar and are not:

      1. Does the claim assert SPECIFICS the evidence does not contain?
         Numbers, dates and names can be checked literally. "Retained for 90
         days" when the source says 30, or "certified in Frankfurt" when no
         such place is mentioned, is a fabrication, and no amount of otherwise
         convincing prose should rescue it. These are capped hard.

      2. Does the claim's PROSE overlap the source?
         This is a much weaker signal than it appears. A faithful restatement
         in different words scores low, and ELI5 mode is explicitly instructed
         to produce exactly that. Treating low overlap as evidence of invention
         would refuse correct answers for being well written.

    So overlap drives the score, but a claim that invents no specifics and is
    at least topically connected earns partial credit rather than zero -- an
    honest report that it could not be verified lexically, not a verdict that
    it is false.
    """
    claim_terms = content_terms(claim)
    if not claim_terms:
        return 0.0, None, []

    claim_bigrams = bigrams(tokenize(claim))
    # The checkable specifics of this claim.
    claim_specifics = (
        extract_numbers(claim) | extract_dates(claim) | extract_proper_nouns(claim)
    )

    # Sentinel rather than 0.0: a claim that overlaps nothing scores exactly 0.0,
    # and starting at 0.0 meant `support > best_support` never fired for it, so
    # its fabricated specifics were never recorded. An off-topic sentence full
    # of invented names would then pass the gate reporting nothing wrong.
    best_support = -1.0
    best_index: Optional[int] = None
    best_missing: List[str] = []

    for index, text in enumerate(evidence_texts):
        passage_terms = content_terms(text)
        if not passage_terms:
            continue

        coverage = len(claim_terms & passage_terms) / len(claim_terms)

        passage_bigrams = bigrams(tokenize(text))
        phrase_bonus = 0.0
        if claim_bigrams:
            phrase_bonus = 0.15 * (len(claim_bigrams & passage_bigrams) / len(claim_bigrams))

        support = min(1.0, coverage + phrase_bonus)

        missing_specifics: List[str] = []
        if claim_specifics:
            passage_specifics = (
                extract_numbers(text) | extract_dates(text) | extract_proper_nouns(text)
            )
            missing_specifics = sorted(claim_specifics - passage_specifics)

        if missing_specifics:
            # Hard cap: an invented figure or name cannot be talked around.
            support = min(support, CLAIM_SUPPORT_THRESHOLD * 0.6)
        elif support < CLAIM_SUPPORT_THRESHOLD and coverage >= PARAPHRASE_MIN_OVERLAP:
            # Nothing fabricated, and the claim is on topic: credit it as
            # plausible-but-unverified rather than scoring it as a falsehood.
            support = max(support, PARAPHRASE_CREDIT)

        if support > best_support:
            best_support = support
            best_index = index
            best_missing = missing_specifics

    if best_support <= 0.0:
        # Either there were no usable passages, or none of them overlapped the
        # claim at all. Naming a "best" source for a claim nothing supports
        # would put a citation under an assertion that passage does not make.
        return 0.0, None, sorted(claim_specifics) if best_support < 0.0 else best_missing

    return best_support, best_index, best_missing



def verify_answer(answer: str, evidence_texts: Sequence[str]) -> VerificationReport:
    """
    Check every claim in the answer against the retrieved evidence.

    An answer with no evidence at all is not "unverified" — it is unsupported,
    and reported as such.
    """
    report = VerificationReport(checked=VERIFICATION_ENABLED)

    if not VERIFICATION_ENABLED:
        report.supported_ratio = 1.0
        report.mean_support = 1.0
        report.max_support = 1.0
        return report

    report.is_refusal = is_refusal(answer)
    if report.is_refusal:
        # A refusal makes no claims, so it is trivially and correctly grounded.
        report.supported_ratio = 1.0
        report.mean_support = 1.0
        report.max_support = 1.0
        return report

    claims = extract_claims(answer)
    if not claims:
        # Nothing checkable was asserted (a greeting, a clarifying question).
        report.supported_ratio = 1.0
        report.mean_support = 1.0
        report.max_support = 1.0
        return report

    for claim in claims:
        support, source_index, missing = score_claim(claim, evidence_texts)
        verdict = ClaimVerdict(
            claim=claim,
            support=support,
            supported=support >= CLAIM_SUPPORT_THRESHOLD,
            best_source_index=source_index,
            missing_specifics=missing,
        )
        report.claims.append(verdict)
        if not verdict.supported:
            report.unsupported_claims.append(claim)
        for specific in missing:
            if specific not in report.unsupported_specifics:
                report.unsupported_specifics.append(specific)

    report.supported_ratio = sum(1 for c in report.claims if c.supported) / len(report.claims)
    report.mean_support = sum(c.support for c in report.claims) / len(report.claims)
    report.max_support = max(c.support for c in report.claims)
    return report


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

# "retention period is 30 days" / "retained for 90 days" — capture the measured
# subject plus its value so two documents can be compared on the same subject.
_MEASURE_RE = re.compile(
    r"(?P<subject>(?:[A-Za-z][\w-]*\s+){0,4}?[A-Za-z][\w-]*)"
    r"\s*(?:is|are|was|were|of|:|=|shall\s+be|must\s+be|will\s+be|for)?\s*"
    r"(?P<value>\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|days?|weeks?|months?|years?|hours?|minutes?|usd|eur|inr|dollars?)",
    re.IGNORECASE,
)

# Words too generic to identify what is being measured.
_WEAK_SUBJECT_TERMS = {
    "is", "are", "was", "were", "the", "a", "an", "of", "for", "to", "in", "on", "at",
    "and", "or", "than", "then", "up", "over", "under", "least", "most", "more", "less",
    "be", "been", "has", "have", "had", "it", "this", "that", "these", "those", "within",
}


def _measure_subject_key(subject: str, unit: str) -> Optional[str]:
    """Reduce a subject phrase to a comparable key, or None if it is too vague."""
    terms = [t for t in tokenize(subject) if t not in _WEAK_SUBJECT_TERMS and not t.isdigit()]
    if not terms:
        return None
    # The last one or two content words carry the subject ("data retention
    # period" -> "retention period"), which is stable across rephrasings.
    key_terms = terms[-2:]
    unit_key = re.sub(r"s$", "", unit.lower())
    if unit_key in ("percent", "%"):
        unit_key = "%"
    return f"{' '.join(key_terms)}|{unit_key}"


def detect_contradictions(evidence: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find measurements that disagree across the retrieved evidence.

    Only cross-document disagreements are reported. Two different figures inside
    one document are usually a table of variants or a worked example, not a
    conflict; the same figure differing *between* documents is exactly the
    "Document A says 30 days, Document B says 90 days" case that must never be
    silently averaged into one confident answer.

    Each item names the subject, the conflicting values, and the documents and
    pages they came from, so the caller can cite both sides.
    """
    by_subject: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for item in evidence:
        text = item.get("text") or ""
        doc_name = item.get("doc_name") or "Unknown Document"
        doc_id = item.get("doc_id") or doc_name
        page = item.get("page")

        for match in _MEASURE_RE.finditer(text):
            key = _measure_subject_key(match.group("subject"), match.group("unit"))
            if not key:
                continue
            value = f"{match.group('value').replace(',', '')} {match.group('unit').lower()}"
            slot = by_subject.setdefault(key, {})
            # First sighting of this value wins the citation slot.
            slot.setdefault(value, {"doc_id": doc_id, "doc_name": doc_name, "page": page})

    contradictions: List[Dict[str, Any]] = []
    for key, values in by_subject.items():
        if len(values) < 2:
            continue
        distinct_docs = {v["doc_id"] for v in values.values()}
        if len(distinct_docs) < 2:
            continue

        subject = key.split("|", 1)[0]
        contradictions.append({
            "subject": subject,
            "values": [
                {
                    "value": value,
                    "doc_id": meta["doc_id"],
                    "doc_name": meta["doc_name"],
                    "page": meta["page"],
                }
                for value, meta in sorted(values.items())
            ],
        })

    return contradictions


def format_contradiction_notice(contradictions: Sequence[Dict[str, Any]]) -> str:
    """Render contradictions as a markdown block that cites every side."""
    if not contradictions:
        return ""

    lines = [CONTRADICTION_NOTICE, ""]
    for item in contradictions:
        lines.append(f"- **{item['subject']}**:")
        for value in item["values"]:
            page = f", page {value['page']}" if value.get("page") is not None else ""
            lines.append(f"    - `{value['value']}` — {value['doc_name']}{page}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    score: int
    label: str          # High | Medium | Low  (the three labels the UI knows)
    band: str           # high | medium | low | very_low
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "label": self.label,
            "band": self.band,
            "components": {k: round(v, 3) for k, v in self.components.items()},
        }


def compute_confidence(
    retrieval_scores: Sequence[float],
    report: VerificationReport,
    cited_source_count: int,
    retrieved_source_count: int,
    expected_source_count: int,
) -> ConfidenceResult:
    """
    Blend retrieval quality with answer/evidence agreement into one score.

    Retrieval strength alone was the old formula, and it is the component that
    lies most readily: it peaks whenever a passage is topically close, including
    when that passage does not contain the answer. It is kept, because a low
    retrieval score is still a genuine warning, but it is now outweighed by
    claim support -- the only component that looks at what the model actually
    wrote.

    Components:
      retrieval  (30%) -- top and mean hybrid score of the evidence
      coverage   (15%) -- did we find as much evidence as the mode asked for
      citation   (15%) -- did the model cite the passages it was given
      claims     (40%) -- how well the answer's claims are found in the evidence

    The number this returns is a HEURISTIC, not a probability. It is a weighted
    blend of four lexical and structural signals, and the weights were chosen by
    judgement, not fitted to labelled data. It is meaningful for ordering and
    for banding (is this answer better supported than that one?) and it should
    not be read as "87% likely to be correct". The bands are what the product
    acts on; the score exists to place an answer within them and to make the
    reasoning inspectable via `components` in the RAG trace.
    """
    scores = [s for s in retrieval_scores if s is not None]

    if scores:
        top = max(scores)
        mean = sum(scores) / len(scores)
        # Weighted toward the best passage: one strong piece of evidence is
        # worth more than several mediocre ones.
        retrieval = 0.65 * top + 0.35 * mean
    else:
        top = mean = retrieval = 0.0

    if expected_source_count > 0:
        coverage = min(1.0, retrieved_source_count / expected_source_count)
    else:
        coverage = 1.0 if retrieved_source_count else 0.0

    if retrieved_source_count > 0:
        citation = min(1.0, cited_source_count / retrieved_source_count)
    else:
        citation = 0.0

    # Two readings of the same evidence, blended because they answer different
    # questions: supported_ratio asks how many claims cleared the bar (a hard
    # verdict), mean_support asks how close they came (a smooth one). Using the
    # ratio alone made a faithful paraphrase indistinguishable from a
    # fabrication; using the mean alone let one filler word in an otherwise
    # verbatim answer drag a well-grounded response out of the HIGH band.
    if report.checked:
        claims = 0.5 * report.supported_ratio + 0.5 * report.mean_support
    else:
        claims = 1.0

    # A refusal is perfectly grounded and completely uninformative. Scoring it
    # on claim support would hand "I cannot find that information" a high
    # confidence, which is precisely backwards: the number the UI shows is
    # confidence in an ANSWER, and there isn't one.
    if report.is_refusal:
        return ConfidenceResult(
            score=0,
            label="Low",
            band="very_low",
            components={
                "retrieval": retrieval,
                "retrieval_top": top,
                "retrieval_mean": mean,
                "coverage": coverage,
                "citation": citation,
                "claims": 0.0,
                "refusal": 1.0,
            },
        )

    raw = (
        0.30 * retrieval
        + 0.15 * coverage
        + 0.15 * citation
        + 0.40 * claims
    )

    # Contradictions are a property of the evidence, not the answer, so they cap
    # confidence rather than zeroing it: the answer may be a faithful report
    # that the sources disagree.
    if report.contradictions:
        raw = min(raw, 0.60)

    # An answer that invents specifics is not low-confidence, it is wrong, and
    # the number shown next to it should say so. Note this keys on fabrication,
    # not on low prose overlap -- an answer written in the user's own words is
    # unverified, which is a different thing and is not capped here.
    if report.checked and report.unsupported_specifics:
        raw = min(raw, 0.30)

    # No evidence at all cannot yield a confident answer regardless of prose.
    if retrieved_source_count == 0:
        raw = 0.0

    score = int(round(max(0.0, min(1.0, raw)) * 100))

    if score >= CONFIDENCE_HIGH:
        band, label = "high", "High"
    elif score >= CONFIDENCE_MEDIUM:
        band, label = "medium", "Medium"
    elif score >= CONFIDENCE_LOW:
        band, label = "low", "Low"
    else:
        band, label = "very_low", "Low"

    return ConfidenceResult(
        score=score,
        label=label,
        band=band,
        components={
            "retrieval": retrieval,
            "retrieval_top": top,
            "retrieval_mean": mean,
            "coverage": coverage,
            "citation": citation,
            "claims": claims,
        },
    )


def apply_evidence_gate(
    answer: str,
    confidence: ConfidenceResult,
    report: VerificationReport,
    paraphrase_expected: bool = False,
    lexical_coverage: float = 0.0,
) -> Tuple[str, bool]:
    """
    Decide what the user is actually shown.

    Returns (final_answer, was_gated).

    Three things can cause an answer to be withheld, and each is triggered by
    the signal that actually measures it. That pairing is the important part:
    an earlier version refused on prose overlap, which measures vocabulary
    reuse rather than truthfulness, and consequently refused every correct
    answer written in the user's own words -- ELI5 mode, whose entire purpose
    is to avoid the document's phrasing, was refused unconditionally.

      1. No evidence at all.
         Nothing to be right about. Refuse.

      2. The answer asserts specifics the evidence does not contain -- a
         number, a date, a name. These are checkable literally, and one that is
         absent from every retrieved passage was invented. Refuse regardless of
         how confident the rest of the answer looks: showing a fabricated
         figure with a caveat still shows a fabricated figure.

      3. Retrieval itself was weak.
         The best passage we found barely matches the question, so whatever the
         model wrote is not standing on anything. Refuse. This is measured on
         the retrieval score, NOT on how closely the answer echoes the source.

    Anything else is shown: unchanged in the HIGH band, with an explicit caveat
    in MEDIUM and LOW. Low prose overlap lowers confidence -- it is real
    information -- but on its own it is not grounds for refusal, because the
    check cannot tell a paraphrase from a fabrication and must not pretend to.

    A refusal from the model is never re-gated: it already declined, and
    replacing one honest "I don't know" with a differently-worded one helps
    nobody.
    """
    if report.is_refusal:
        return answer, False

    contradiction_block = format_contradiction_notice(report.contradictions)

    def refuse() -> Tuple[str, bool]:
        parts = [p for p in (contradiction_block, INSUFFICIENT_EVIDENCE_MESSAGE) if p]
        return "\n\n".join(parts), True

    def show_with_caveat() -> Tuple[str, bool]:
        parts = [p for p in (contradiction_block, answer, MEDIUM_CONFIDENCE_WARNING) if p]
        return "\n\n".join(parts), False

    if EVIDENCE_GATE_ENABLED:
        # (1) and (3): no evidence, or evidence too weak to stand on.
        # Absent components means we have no retrieval signal to judge by, and
        # the conservative reading of "no signal" is "not enough evidence".
        #
        # retrieval_top comes from the FUSED score, which blends a semantic and
        # a lexical channel. On a short keyword question ("What year was the
        # Dartmouth Conference?") the semantic channel scores low simply because
        # a three-word query and a paragraph are not similar objects, dragging
        # the blend down even when the passage is an exact, unambiguous match.
        # Judging that as "weak evidence" refuses a correct answer sourced from
        # the right page. So the rule accepts either channel, on the same
        # principle the retrieval gate uses: strong lexical coverage is
        # evidence, not the absence of it.
        weak_fused = confidence.components.get("retrieval_top", 0.0) < EVIDENCE_MIN_TOP_SCORE
        weak_lexical = lexical_coverage < LEXICAL_COVERAGE_THRESHOLD
        if weak_fused and weak_lexical:
            return refuse()

        # (2) fabricated specifics.
        if report.unsupported_specifics:
            return refuse()

        # (4) The answer is not about this evidence at all.
        #
        # Not a single claim reaches even the paraphrase floor, which means the
        # answer shares essentially no vocabulary with any retrieved passage.
        # For a heavily reworded answer that is expected; for one the model was
        # told to build from "the direct facts explicitly stated in the CONTEXT"
        # it means the answer came from somewhere else.
        #
        # This is the one rule that depends on the mode, and honestly so:
        # lexical scoring cannot tell a faithful ELI5 analogy from an answer
        # about a different document -- both share no words with the source.
        # What distinguishes them is what the model was ASKED to produce, which
        # is knowable, so the caller passes it in rather than the scorer
        # guessing.
        if (
            not paraphrase_expected
            and report.claims
            and report.max_support < PARAPHRASE_MIN_OVERLAP
        ):
            return refuse()

    if confidence.band == "high":
        if contradiction_block:
            return f"{contradiction_block}\n\n{answer}", False
        return answer, False

    return show_with_caveat()
