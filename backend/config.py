"""
Central configuration for DocMind AI.

Every value is environment-driven with a sane default, so the same codebase runs
unchanged on a laptop, in Docker, and on a cloud host. No secrets live in here —
they are read from the environment (loaded from backend/.env in local dev).
"""

import os

from dotenv import load_dotenv

# --- Paths -----------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Load local .env first (never present in production — the host injects env vars).
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(REPO_ROOT, ".env"))


def _env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to the default if unset or malformed."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# DATA_DIR is the single writable root for all runtime state. Point it at a
# mounted volume in production to make uploads/indices/DB survive restarts.
DATA_DIR = os.getenv("DATA_DIR") or BASE_DIR

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
FAISS_DIR = os.path.join(DATA_DIR, "faiss_indices")
DB_FILE = os.getenv("DB_FILE") or os.path.join(DATA_DIR, "docmind.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

# --- Persistence awareness --------------------------------------------------
# ALL durable state -- user accounts, uploaded files, FAISS indices, chat
# history, analytics and quizzes -- lives under DATA_DIR. If DATA_DIR is not a
# mounted volume, every one of those is lost when the container is replaced,
# which on Render's free plan happens whenever the instance sleeps. That is the
# real cause behind "I registered yesterday and cannot log in today": the
# account was not overwritten, the database file no longer exists.
#
# Recorded at import so /api/health can report it honestly instead of leaving
# operators to infer it.
DB_EXISTED_AT_BOOT = os.path.isfile(DB_FILE)

# Set automatically by Render when a Postgres instance is attached. This
# application has no Postgres support (see DEPLOY.md, "Persistence"), so the
# variable being present means someone believes their data is being persisted
# to a database that is not being written to. Surfaced rather than ignored.
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_BACKEND = "sqlite"

# Directory holding the compiled React SPA. Served by FastAPI in production so
# the whole app is a single origin (no CORS, one deployable service).
FRONTEND_DIST_DIR = os.path.abspath(
    os.getenv("FRONTEND_DIST_DIR") or os.path.join(REPO_ROOT, "frontend", "dist")
)

# --- Secrets ---------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Models ----------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")

# --- RAG tuning ------------------------------------------------------------
CHUNK_SIZE = _env_int("CHUNK_SIZE", 1000)          # ~150 words — one coherent paragraph
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 150)     # 15% overlap — avoids boundary fragmentation
TOP_K = _env_int("TOP_K", 8)                       # chunks passed to the LLM as context
RELEVANCE_THRESHOLD = _env_float("RELEVANCE_THRESHOLD", 0.50)  # below this, a chunk is dropped
VECTOR_WEIGHT = _env_float("VECTOR_WEIGHT", 0.6)   # hybrid score = VECTOR_WEIGHT*vec + (1-w)*bm25

# A passage containing at least this fraction of the query's information content
# (IDF-weighted term coverage) is admitted on lexical evidence alone, whatever
# the fused score says.
#
# Why this exists: the fused score weights BM25 at 1-VECTOR_WEIGHT = 0.4, which
# is BELOW RELEVANCE_THRESHOLD (0.5). A passage that is the single unambiguous
# exact match for the query therefore could not clear the gate on lexical
# evidence alone -- it still needed vector_sim >= 0.167 to survive. That
# inverted the point of hybrid retrieval precisely for the queries BM25 exists
# to serve: short strings of proper nouns and identifiers ("Dartmouth Conference
# 1956", "RFC 8446"), which carry the most lexical signal and the least semantic
# signal for an embedding model to work with.
#
# The fused score and its 0.6/0.4 weights are untouched: they still decide the
# ORDER. This governs admission only, and only in the direction of admitting
# evidence that was being wrongly discarded.
LEXICAL_COVERAGE_THRESHOLD = _env_float("LEXICAL_COVERAGE_THRESHOLD", 0.5)

# --- Uploads ---------------------------------------------------------------
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 25)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# --- Demo mode -------------------------------------------------------------
# On free hosting the disk is ephemeral, so a cold start would show an empty app.
# When enabled, a bundled sample PDF is indexed on boot so the demo is never blank.
DEMO_SEED = _env_bool("DEMO_SEED", False)
DEMO_PDF_PATH = os.getenv("DEMO_PDF_PATH") or os.path.join(REPO_ROOT, "assets", "demo", "sample.pdf")
DEMO_USERNAME = os.getenv("DEMO_USERNAME", "demo")
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo1234")
