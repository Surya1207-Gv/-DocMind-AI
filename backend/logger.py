"""
=============================================================================
DocMind AI — Centralized Structured Logging & RAG Telemetry
=============================================================================
Provides rotating file and console logging alongside structured RAG metrics:
  - Query latency breakdown (vector_ms, bm25_ms, total_ms)
  - Retrieval yield & threshold passage rate
  - Zero-hit rate monitoring (leading indicator of retrieval degradation)
"""

import logging
import sys
import os
import time
import hashlib
import json
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Keep logs beside the rest of the runtime state so a mounted volume captures them.
LOG_DIR = os.getenv("DATA_DIR") or BASE_DIR
LOG_FILE = os.path.join(LOG_DIR, "docmind.log")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ---------------------------------------------------------------------------
# In-Memory RAG Telemetry Tracker
# ---------------------------------------------------------------------------
class RAGTelemetry:
    def __init__(self):
        self.total_queries = 0
        self.zero_hit_queries = 0
        self.total_latency_ms = 0.0
        self.boost_triggered_count = 0
        self.events = []

    def record_event(self, event_data: Dict[str, Any]):
        self.total_queries += 1
        if event_data.get("passed_threshold", 0) == 0:
            self.zero_hit_queries += 1
        if event_data.get("boost_applied"):
            self.boost_triggered_count += 1
        self.total_latency_ms += event_data.get("total_ms", 0.0)
        
        # Keep last 50 events in circular memory
        self.events.append(event_data)
        if len(self.events) > 50:
            self.events.pop(0)

    def get_summary(self) -> Dict[str, Any]:
        avg_lat = (self.total_latency_ms / self.total_queries) if self.total_queries > 0 else 0.0
        zero_hit_rate = (self.zero_hit_queries / self.total_queries * 100.0) if self.total_queries > 0 else 0.0
        boost_rate = (self.boost_triggered_count / self.total_queries * 100.0) if self.total_queries > 0 else 0.0
        return {
            "total_queries": self.total_queries,
            "zero_hit_queries": self.zero_hit_queries,
            "zero_hit_rate_pct": round(zero_hit_rate, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "boost_triggered_rate_pct": round(boost_rate, 2),
            "recent_events_count": len(self.events)
        }

telemetry = RAGTelemetry()


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger for the given module name."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_fmt)
    logger.addHandler(console_handler)
    
    # Rotating file handler — max 5MB per file, keep 3 backups
    try:
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(_fmt)
        logger.addHandler(file_handler)
    except (OSError, PermissionError):
        pass
    
    logger.propagate = False
    return logger


def log_rag_retrieval_event(
    query: str,
    candidates_count: int,
    passed_threshold_count: int,
    top_score: float,
    vector_ms: float,
    bm25_ms: float,
    boost_applied: Optional[str] = None
):
    """
    Logs structured telemetry for a RAG retrieval execution and records in telemetry tracker.
    """
    q_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    total_ms = round(vector_ms + bm25_ms, 2)
    
    event_data = {
        "event": "rag_retrieval",
        "query_hash": q_hash,
        "candidates": candidates_count,
        "passed_threshold": passed_threshold_count,
        "top_score": round(top_score, 3),
        "vector_ms": round(vector_ms, 2),
        "bm25_ms": round(bm25_ms, 2),
        "total_ms": total_ms,
        "boost_applied": boost_applied or "none"
    }
    
    telemetry.record_event(event_data)
    
    rag_logger = get_logger("docmind.rag_telemetry")
    rag_logger.info("[TELEMETRY] %s", json.dumps(event_data))
    
    if passed_threshold_count == 0:
        rag_logger.warning("[DEGRADATION ALERT] Query hash %s produced 0 chunks above 0.50 threshold!", q_hash)
