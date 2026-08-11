# Development & Diagnostic Scripts

This directory contains one-off diagnostic and verification scripts used during local development and testing.

> **Note:** These scripts are developer utilities and diagnostics. They are **not** part of the production application runtime or the automated CI test suite. All automated regression tests reside in `backend/tests/` and are executed via `pytest`.

### Script Inventory
- `test_reranking_proposed_bm25.py`: Prototype exploration of BM25 scoring with CamelCase tokenization.
- `test_reranking_proposed.py`: Prototype exploration of 60/40 hybrid score fusion with definitional boosting.
- `test_reranking_detail.py`: Detailed candidate-level score inspection script.
- `test_openrouter.py`: Diagnostic checks for OpenRouter API endpoints and latency.
- `test_chat_e2e.py`: Terminal-based end-to-end chat simulation.
- `scratch_test_retrieval.py`: Standalone FAISS index query test.
- `test_embeddings.py`: Validates embedding dimensions (1536) from OpenRouter.
- `scratch_check_pages.py`: Verifies PDF page text extraction fidelity.
- `find_free_models.py`: Queries OpenRouter API for active free tier models.
- `test_db_sources.py`: Inspects SQLite JSON source citation payloads.
