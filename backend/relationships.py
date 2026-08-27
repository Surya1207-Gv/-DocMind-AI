"""
=============================================================================
DocMind AI - Cross-document relationship detection
=============================================================================
Answers "how do my documents relate to each other?" using only what the system
already stores: the per-document FAISS indices and the chunk text inside them.

No graph database. A relationship here is a similarity computation over vectors
that are already on disk, cached per request; standing up Neo4j to hold what is
effectively an NxN float matrix over a user's handful of documents would add a
service to deploy, back up and secure in exchange for nothing.

Two signals, because they disagree in useful ways:

  * centroid cosine  -- the mean embedding of each document. Captures topical
                        relatedness: two documents about authentication score
                        highly even with no shared wording.
  * chunk overlap    -- the fraction of chunks in the smaller document that have
                        a near-identical counterpart in the larger. Captures
                        literal reuse, which is what distinguishes a duplicate
                        or a revised version from a document that is merely
                        about the same subject.

High cosine with low overlap is "related". High cosine WITH high overlap is a
duplicate or a version. That distinction is the whole point of computing both.
"""

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config import FAISS_DIR
from backend.logger import get_logger
from backend.text_utils import content_terms, jaccard
from backend.verification import detect_contradictions

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Relationship thresholds, from strongest to weakest.
DUPLICATE_SIMILARITY = _env_float("REL_DUPLICATE_SIMILARITY", 0.97)
DUPLICATE_OVERLAP = _env_float("REL_DUPLICATE_OVERLAP", 0.80)
VERSION_SIMILARITY = _env_float("REL_VERSION_SIMILARITY", 0.90)
VERSION_OVERLAP = _env_float("REL_VERSION_OVERLAP", 0.35)
SIMILAR_SIMILARITY = _env_float("REL_SIMILAR_SIMILARITY", 0.85)
RELATED_SIMILARITY = _env_float("REL_RELATED_SIMILARITY", 0.65)

# Two chunks count as "the same chunk" at this lexical overlap.
CHUNK_MATCH_JACCARD = _env_float("REL_CHUNK_MATCH_JACCARD", 0.80)

# Comparing every chunk against every chunk is quadratic; sample instead. Enough
# to detect substantial reuse, cheap enough to run on request.
MAX_CHUNKS_SAMPLED = 60

# Version markers in filenames: "policy_v2.pdf" vs "policy_v3.pdf",
# "report-2023.docx" vs "report-2024.docx".
_VERSION_TOKEN_RE = re.compile(
    r"(?:[_\-\s]v?\d+(?:\.\d+)*|[_\-\s](?:19|20)\d{2}|[_\-\s](?:draft|final|rev\d*|copy))",
    re.IGNORECASE,
)


@dataclass
class DocumentProfile:
    """Everything needed to compare one document, loaded once."""
    doc_id: str
    doc_name: str
    centroid: Optional[List[float]]
    chunk_terms: List[frozenset]
    texts: List[str]
    pages: List[Any]


def _normalize_name(name: str) -> str:
    """Strip extension and version markers so 'policy_v2.pdf' ~ 'policy_v3.pdf'."""
    stem = os.path.splitext(name or "")[0]
    return _VERSION_TOKEN_RE.sub("", stem).strip(" _-").lower()


def load_document_profile(doc_id: str, doc_name: str, embeddings: Any) -> Optional[DocumentProfile]:
    """
    Load a document's stored vectors and chunk text from its FAISS index.

    Returns None when the index is missing -- a document whose index was never
    built, or was removed, simply has no relationships rather than raising.
    """
    path = os.path.join(FAISS_DIR, doc_id)
    if not os.path.isdir(path):
        return None

    try:
        from langchain_community.vectorstores import FAISS

        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    except Exception as exc:
        logger.warning("[Relationships] Could not load index for %s: %s", doc_id, exc)
        return None

    documents = list(store.docstore._dict.values())
    if not documents:
        return None

    # Reconstruct the stored vectors rather than re-embedding: the vectors are
    # already paid for, and re-embedding would cost an API call per document.
    centroid = None
    try:
        import numpy as np

        vectors = store.index.reconstruct_n(0, store.index.ntotal)
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.size:
            mean = matrix.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                centroid = (mean / norm).tolist()
    except Exception as exc:
        logger.warning("[Relationships] Could not reconstruct vectors for %s: %s", doc_id, exc)

    sampled = documents[:MAX_CHUNKS_SAMPLED]
    return DocumentProfile(
        doc_id=doc_id,
        doc_name=doc_name,
        centroid=centroid,
        chunk_terms=[frozenset(content_terms(d.page_content)) for d in sampled],
        texts=[d.page_content for d in sampled],
        pages=[(d.metadata or {}).get("page") for d in sampled],
    )


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Cosine similarity of two already-normalised centroids."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def chunk_overlap_ratio(left: DocumentProfile, right: DocumentProfile) -> float:
    """
    Fraction of the smaller document's chunks that appear in the larger.

    Normalising by the smaller side is what makes a short document excerpted
    wholesale from a long one register as heavily overlapping, instead of being
    diluted to near zero by the long document's other content.
    """
    smaller, larger = (left, right) if len(left.chunk_terms) <= len(right.chunk_terms) else (right, left)
    if not smaller.chunk_terms or not larger.chunk_terms:
        return 0.0

    matches = 0
    for terms in smaller.chunk_terms:
        if not terms:
            continue
        if any(jaccard(terms, other) >= CHUNK_MATCH_JACCARD for other in larger.chunk_terms):
            matches += 1

    return matches / len(smaller.chunk_terms)


def classify(
    similarity: float,
    overlap: float,
    name_match: bool,
    has_conflict: bool,
) -> Optional[str]:
    """
    Name the relationship, strongest first.

    ``conflicting`` outranks everything: two documents that give different values
    for the same thing are the case a user most needs told about, and calling
    them merely "similar" would bury it.
    """
    if has_conflict:
        return "conflicting"
    if similarity >= DUPLICATE_SIMILARITY and overlap >= DUPLICATE_OVERLAP:
        return "duplicate"
    if similarity >= VERSION_SIMILARITY and (overlap >= VERSION_OVERLAP or name_match):
        return "possible-version-of"
    if similarity >= SIMILAR_SIMILARITY:
        return "similar"
    if similarity >= RELATED_SIMILARITY:
        return "related"
    return None


def detect_relationships(
    documents: Sequence[Dict[str, Any]],
    embeddings: Any,
) -> List[Dict[str, Any]]:
    """
    Compare every pair of the caller's documents.

    ``documents`` must already be scoped to one user -- this function does no
    authorisation of its own, and comparing across users would leak the
    existence and titles of other people's files.
    """
    profiles: List[DocumentProfile] = []
    for document in documents:
        profile = load_document_profile(document["id"], document.get("name", "Unknown"), embeddings)
        if profile:
            profiles.append(profile)

    relationships: List[Dict[str, Any]] = []

    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            left, right = profiles[i], profiles[j]

            similarity = cosine(left.centroid, right.centroid)
            # Only pay for the quadratic chunk comparison when the cheap signal
            # already suggests the documents are worth comparing.
            overlap = chunk_overlap_ratio(left, right) if similarity >= RELATED_SIMILARITY else 0.0

            name_match = (
                bool(_normalize_name(left.doc_name))
                and _normalize_name(left.doc_name) == _normalize_name(right.doc_name)
            )

            conflicts: List[Dict[str, Any]] = []
            if similarity >= RELATED_SIMILARITY:
                evidence = [
                    {"text": text, "doc_id": profile.doc_id, "doc_name": profile.doc_name, "page": page}
                    for profile in (left, right)
                    for text, page in zip(profile.texts, profile.pages)
                ]
                conflicts = detect_contradictions(evidence)

            relationship = classify(similarity, overlap, name_match, bool(conflicts))
            if not relationship:
                continue

            relationships.append({
                "type": relationship,
                "similarity": round(similarity, 4),
                "chunk_overlap": round(overlap, 4),
                "name_match": name_match,
                "documents": [
                    {"doc_id": left.doc_id, "doc_name": left.doc_name},
                    {"doc_id": right.doc_id, "doc_name": right.doc_name},
                ],
                "conflicts": conflicts,
            })

    # Strongest relationships first, so the UI can show the important ones.
    order = {"conflicting": 0, "duplicate": 1, "possible-version-of": 2, "similar": 3, "related": 4}
    relationships.sort(key=lambda r: (order.get(r["type"], 9), -r["similarity"]))
    return relationships
