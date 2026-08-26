"""
=============================================================================
DocMind AI - Demo Seeding
=============================================================================
Free hosting tiers give you an ephemeral filesystem: when the instance sleeps
and wakes, uploads, indices and the SQLite database are gone. A reviewer opening
the public URL would then land on an empty app.

When DEMO_SEED is enabled, this module ingests a small bundled PDF on boot --
through exactly the same pipeline as a real upload (extract -> chunk -> embed ->
index) -- so there is always something to ask questions about.

Every failure here is non-fatal: seeding is a convenience, never a reason for
the API to fail to start.
"""

import os
import shutil
import uuid
from datetime import datetime

from backend.config import (
    DEMO_PDF_PATH,
    DEMO_USERNAME,
    DEMO_PASSWORD,
    UPLOAD_DIR,
)
from backend.logger import get_logger

logger = get_logger(__name__)

DEMO_DOC_NAME = "DocMind AI - Field Handbook.pdf"


def _demo_user_id(db, hash_password) -> str:
    """Return the demo user's id, creating the account on first boot."""
    user = db.get_user_by_username(DEMO_USERNAME)
    if user:
        return user["id"]

    user_id = str(uuid.uuid4())
    db.create_user(user_id, DEMO_USERNAME, hash_password(DEMO_PASSWORD))
    logger.info("[Demo] Created demo account '%s'", DEMO_USERNAME)
    return user_id


def seed_demo_document() -> None:
    """
    Ingest the bundled demo PDF for the demo user, if it is not already present.

    Idempotent: re-running with the document already indexed is a no-op, so a
    restart with a persistent disk attached will not duplicate the corpus.
    """
    # Imported lazily so importing this module never drags in the DB/embedding
    # stack -- keeps `import backend.demo_seed` cheap and side-effect free.
    import backend.database as db
    from backend.auth import hash_password
    from backend.embedding_manager import create_and_save_index
    from backend.pdf_processor import process_pdf

    if not os.path.isfile(DEMO_PDF_PATH):
        logger.warning("[Demo] Seed PDF not found at %s - skipping.", DEMO_PDF_PATH)
        return

    user_id = _demo_user_id(db, hash_password)

    # Idempotency guard: skip if this user already has the demo document.
    for existing in db.list_documents(user_id) or []:
        if existing.get("name") == DEMO_DOC_NAME:
            logger.info("[Demo] Demo document already indexed - skipping seed.")
            return

    doc_id = str(uuid.uuid4())
    dest_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

    try:
        shutil.copyfile(DEMO_PDF_PATH, dest_path)

        chunks = process_pdf(dest_path, DEMO_DOC_NAME, doc_id)
        if not chunks:
            logger.warning("[Demo] Seed PDF produced no text - skipping.")
            os.remove(dest_path)
            return

        # This is the only step that needs a working API key.
        create_and_save_index(chunks, doc_id)

        page_count = max(c["metadata"].get("page", 1) for c in chunks)
        word_count = len(" ".join(c["text"] for c in chunks).split())

        db.add_document(
            doc_id,
            user_id,
            DEMO_DOC_NAME,
            os.path.getsize(dest_path),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        db.save_analytics(
            doc_id=doc_id,
            word_count=word_count,
            page_count=page_count,
            read_time_mins=max(1, round(word_count / 200)),
            complexity_score="Medium",
            summary=[
                "A short handbook describing how DocMind AI ingests documents and answers questions about them.",
                "Covers chunking, embeddings, hybrid retrieval, score fusion, and evaluation.",
            ],
            entities=[],
            alerts=[{
                "type": "insight",
                "content": "This sample document ships with the app so you can try it without uploading anything.",
                "page": 1,
            }],
            suggested_questions=[
                "What is Retrieval-Augmented Generation?",
                "Why does the system use hybrid retrieval instead of vector search alone?",
                "What is the default chunk size and overlap, and why?",
                "What happens when no chunk passes the relevance threshold?",
            ],
        )

        logger.info(
            "[Demo] Seeded '%s' (%d chunks across %d pages) for user '%s'.",
            DEMO_DOC_NAME, len(chunks), page_count, DEMO_USERNAME,
        )

    except Exception as exc:
        # Most likely cause: no/invalid API key, so embeddings cannot be built.
        # The app must still start -- uploads will simply fail with a clear error.
        logger.error("[Demo] Seeding failed (%s): %s", type(exc).__name__, exc)
        if os.path.exists(dest_path):
            os.remove(dest_path)
