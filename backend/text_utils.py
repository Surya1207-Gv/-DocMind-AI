"""
=============================================================================
DocMind AI - Shared text normalisation
=============================================================================
One tokenizer, used by BM25 lexical retrieval and by claim verification, so a
term that retrieval considers a match is the same term verification considers
evidence. Two divergent tokenizers would silently disagree about what a
document "says".

Script coverage matters here. The original BM25 tokenizer extracted words with
``re.findall(r'[A-Z]+(?=[A-Z][a-z]|\\b)|[A-Z]?[a-z]+|[0-9]+', w)``, a pattern
that only matches ASCII letters -- every Devanagari and Telugu token was
dropped on the floor, so BM25 scored 0.0 for any Hindi or Telugu query and the
"hybrid" search silently degraded to vector-only. This module keeps the
camelCase splitting that helped on technical PDFs while passing non-Latin
scripts through intact.
"""

import re
import string
import unicodedata
from typing import Iterable, List, Set

# English stop words. Deliberately English-only: applying them to a Hindi or
# Telugu query would be a no-op anyway, and there is no reliable cheap way to
# strip stop words in a language we have not detected.
STOP_WORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself",
    "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "would", "should", "could",
}

# Latin word / camelCase / number splitter, applied only to runs that are
# actually Latin script. The trailing `[A-Z]+` alternative is what keeps an
# all-caps term glued to a digit ("GPT4", "SHA256") from losing its letters:
# the original pattern matched only the digits in those and dropped the acronym.
_LATIN_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# A "word run": any run of letters, digits, marks and intra-word joiners. \w is
# Unicode-aware in Python 3, so this matches Devanagari, Telugu, CJK and more.
# Combining marks (category Mn) are explicitly included because Indic vowel
# signs are separate code points that \w already covers but that punctuation
# stripping must not sever.
_WORD_RUN_RE = re.compile(r"[\wऀ-ॿఀ-౿][\wऀ-ॿఀ-౿'’.-]*")

_ASCII_ONLY_RE = re.compile(r"^[\x00-\x7F]+$")

# Numbers, currency amounts, percentages, and dates. These are the tokens a
# hallucinating model gets wrong most visibly, so verification checks them
# separately from ordinary words.
_NUMERIC_RE = re.compile(
    r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:%|percent|per\s+cent)?",
    re.IGNORECASE,
)

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december|"
    "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_RE = re.compile(
    rf"(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{4}}"
    rf"|\d{{1,2}}\s+(?:{_MONTHS})\.?\s+\d{{4}}"
    rf"|\b(?:19|20)\d{{2}}\b)",
    re.IGNORECASE,
)


def is_latin(token: str) -> bool:
    """True when the token is pure ASCII and can safely go through camelCase splitting."""
    return bool(_ASCII_ONLY_RE.match(token))


def tokenize(text: str, drop_stop_words: bool = True) -> List[str]:
    """
    Split text into lowercase terms.

    Latin runs are additionally split on camelCase boundaries and letter/digit
    transitions ("HTTPServer" -> http, server; "GPT4" -> gpt, 4), which is what
    made technical PDFs searchable. Non-Latin runs are lowercased and kept
    whole, because the same splitting would erase them entirely.
    """
    if not text:
        return []

    terms: List[str] = []
    for run in _WORD_RUN_RE.findall(text):
        run = run.strip(string.punctuation + "’")
        if not run:
            continue

        if is_latin(run):
            parts = _LATIN_SPLIT_RE.findall(run)
            terms.extend(p.lower() for p in parts) if parts else terms.append(run.lower())
        else:
            # Normalise so that visually identical Indic sequences composed
            # differently compare equal.
            terms.append(unicodedata.normalize("NFC", run).lower())

    if drop_stop_words:
        return [t for t in terms if t and t not in STOP_WORDS]
    return [t for t in terms if t]


def content_terms(text: str) -> Set[str]:
    """Unique, stop-word-free terms — the unit of overlap scoring."""
    return set(tokenize(text, drop_stop_words=True))


def bigrams(terms: List[str]) -> Set[str]:
    """Adjacent term pairs, used to reward phrase-level (not just bag-of-words) matches."""
    return {f"{a} {b}" for a, b in zip(terms, terms[1:])}


def normalize_number(raw: str) -> str:
    """
    Canonicalise a numeric string so "1,200", "1200" and "1200.00" compare equal.

    Percent signs and the word "percent" collapse to a single marker so that
    "30%" and "30 percent" are recognised as the same claim.
    """
    text = raw.strip().lower()
    is_pct = "%" in text or "percent" in text or "per cent" in text
    digits = re.sub(r"[^\d.]", "", text)
    if not digits:
        return ""
    try:
        value = float(digits)
    except ValueError:
        return ""
    # Render integers without a trailing ".0" so 30 and 30.0 match.
    rendered = str(int(value)) if value == int(value) else str(value)
    return f"{rendered}%" if is_pct else rendered


def extract_numbers(text: str) -> Set[str]:
    """Canonical numeric tokens appearing in the text (years excluded — see extract_dates)."""
    found = set()
    for match in _NUMERIC_RE.findall(text or ""):
        canonical = normalize_number(match)
        if canonical:
            found.add(canonical)
    return found


def extract_dates(text: str) -> Set[str]:
    """Date-like strings, lowercased and whitespace-collapsed for comparison."""
    return {re.sub(r"\s+", " ", m).strip().lower() for m in _DATE_RE.findall(text or "")}


# Words that are capitalised for grammatical reasons rather than because they
# name something, so they must not be mistaken for proper nouns.
_NON_ENTITY_CAPITALS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its", "they", "their",
    "he", "she", "his", "her", "we", "our", "you", "your", "i", "there", "here",
    "and", "but", "or", "if", "when", "while", "however", "therefore", "also",
    "in", "on", "at", "for", "to", "of", "with", "by", "from", "as", "into",
    "all", "any", "each", "every", "some", "no", "not", "only", "both",
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "do", "does",
    "did", "can", "will", "would", "should", "could", "may", "might", "must",
    "what", "which", "who", "whom", "why", "how", "where", "based", "according",
    "imagine", "think", "note", "so", "then", "now", "first", "second", "third",
}

_CAPITALISED_RE = re.compile(r"\b[A-Z][A-Za-z0-9&.'-]{2,}\b")


def extract_proper_nouns(text: str) -> Set[str]:
    """
    Capitalised terms that look like they name a specific thing.

    Used alongside numbers and dates to identify the *checkable specifics* of a
    claim -- an organisation, standard, place or product name. These are what a
    model invents when it fabricates ("certified to ISO 27001 by an auditor in
    Frankfurt"), and unlike ordinary prose they can be checked literally,
    because a name that is not in the source is simply not in the source.

    A token at the start of a sentence is skipped: its capital says nothing
    about whether it is a name.
    """
    found: Set[str] = set()
    for sentence in re.split(r"(?<=[.!?।])\s+|\n+", text or ""):
        stripped = sentence.strip()
        if not stripped:
            continue
        for match in _CAPITALISED_RE.finditer(stripped):
            # Skip the sentence-initial word.
            if match.start() == 0:
                continue
            token = match.group().strip(".'-")
            if len(token) < 3:
                continue
            if token.lower() in _NON_ENTITY_CAPITALS:
                continue
            found.add(token.lower())
    return found


def split_sentences(text: str) -> List[str]:
    """
    Split prose into sentences.

    Deliberately regex-based rather than a model: verification runs on every
    answer, and pulling in an NLP model for sentence splitting would cost more
    than the rest of the pipeline. Handles the Devanagari danda (।) alongside
    Latin terminators.
    """
    if not text:
        return []

    # Protect common abbreviations from being treated as sentence ends.
    protected = re.sub(r"\b(e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|Fig|No)\.", r"\1<DOT>", text)
    pieces = re.split(r"(?<=[.!?।])\s+|\n{2,}", protected)

    sentences = []
    for piece in pieces:
        cleaned = piece.replace("<DOT>", ".").strip()
        # Strip markdown list/heading decoration so the claim text is prose.
        cleaned = re.sub(r"^[\s>*\-•]+", "", cleaned)
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Symmetric set overlap, used for document-level similarity."""
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
