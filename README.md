# DocMind AI — Retrieval-Augmented Document Intelligence

> Document intelligence platform delivering grounded Q&A with page-level citations, multi-hop reasoning, and evidence gating.

[![CI](https://github.com/Surya1207-Gv/-DocMind-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/Surya1207-Gv/-DocMind-AI/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Demo

* **Live Demo:** `Coming soon`
* **Workflow:**
  $$\text{Upload a Document} \longrightarrow \text{Ask a Question} \longrightarrow \text{Retrieve Relevant Passages} \longrightarrow \text{Generate Cited Answer}$$

> **Pre-Seeded Sample:** A bundled handbook (`assets/demo/sample.pdf`) is indexed on boot when `DEMO_SEED=true`, enabling immediate testing without requiring an initial upload.

---

## Overview

Large Language Models (LLMs) are fluent but ungrounded. When asked about domain-specific contracts, compliance policies, or technical papers they were never trained on, they frequently confabulate confident, plausible-sounding falsehoods. This failure mode makes naive LLMs unusable for high-stakes document analysis.

**DocMind AI solves this using verifiable Retrieval-Augmented Generation (RAG):**
- **Grounded Answers:** Uploaded documents are parsed, chunked, embedded, and indexed per user. When a query is submitted, only verified relevant passages are injected into the prompt.
- **Page-Level Provenance:** Every factual statement cites its exact source document and page number, expandable in the UI to inspect the underlying passage.
- **Honest Refusals:** When a query falls outside the corpus or fails the relevance threshold, the system explicitly refuses to answer rather than guessing.
- **Hybrid Retrieval:** Dense vector search collapses on rare acronyms and exact codes (`RFC 8446`), while sparse keyword search collapses on conceptual paraphrasing. DocMind runs **both** FAISS vector similarity and Okapi BM25, fusing scores with heuristic boosts to optimize precision and recall.

---

## Key Features

### Retrieval Quality
- **Hybrid Search Fusion:** Weighted blending of dense vector similarity (`0.6`) and Okapi BM25 sparse keyword scores (`0.4`).
- **Custom Okapi BM25 Engine:** In-memory implementation (`k₁=1.5`, `b=0.75`) featuring CamelCase token splitting and multilingual normalization.
- **Heuristic Ranking Boosts:** Pattern-based boosts for definitional queries (`+0.05`), exact subject-definition proximity (`+0.45`), and section headers (`+0.10`).
- **Shortlist Reranking:** Deterministic lexical reranker scoring candidates on phrase contiguity and query term coverage.
- **Lexical Coverage Admission:** Independent threshold route (`LEXICAL_COVERAGE_THRESHOLD=0.50`) preventing pure keyword queries from being penalized by vector distance.
- **Adjacent-Chunk Expansion:** Automatically stitches the adjacent successor chunk (`chunk_index + 1`) to preserve sentence completion and context cohesion.

### Grounding and Trust
- **Verifiable Citations:** Exact document name and page number citations attached to every claim.
- **Confidence Scoring:** Derived from multi-factor retrieval similarity and claim-support verification.
- **Strict Evidence Gating:** Rejection threshold (`0.50`) that halts generation when candidate evidence is insufficient.
- **Post-Hoc Citation Pruning:** System inspects model outputs and strips unreferenced sources before returning responses.
- **Multi-Document Conflict Detection:** Flags contradictory metrics or values across comparative documents.

### Product Experience
- **Task-Specific Modes:** Four optimized prompt configurations: Q&A (`temp=0.2`), Summary (`temp=0.3`), Deep Analysis (`temp=0.5`), and ELI5 (`temp=0.6`).
- **Background Document Analytics:** Automated extraction of summaries, key entities, critical alerts, and suggested follow-up questions.
- **Assessment Generation:** Generates multiple-choice quizzes with page references for review and compliance verification.
- **Cross-Document Comparison:** Side-by-side comparative analysis across multiple uploaded documents.
- **Multi-Format Ingestion:** Native extraction for PDF, Markdown, Plain Text, HTML, DOCX, and web URLs.
- **Session Continuity:** Persistent chat history stored in SQLite with PDF export capabilities.

### Engineering
- **Multi-Tenant Isolation:** Per-user document ownership, isolated SQLite records, and independent FAISS directory indices.
- **Production-Grade Testing:** 317 backend tests (unit, API integration, and regression suites) and 46 frontend tests executed in CI.
- **Real-Time Streaming:** Chunked responses delivered via Server-Sent Events (SSE).
- **Observability & Telemetry:** Per-request retrieval telemetry and aggregated performance counters (`GET /api/metrics`).
- **Zero-Secret Codebase:** 100% environment-driven configuration with strict startup validation.

---

## Architecture

DocMind AI is engineered as **a single-origin service**: FastAPI serves both the JSON/SSE API endpoints and the compiled React SPA. This eliminates cross-origin configuration, simplifies network routing, and enables single-container deployments.

```
                              ┌──────────────────────────────┐
   Browser  ───────────────►  │  React SPA (served by FastAPI)│
                              └───────────────┬──────────────┘
                                              │  POST /api/chat  (same origin)
                                              ▼
                              ┌──────────────────────────────┐
                              │   FastAPI  ·  JWT auth       │
                              └───────────────┬──────────────┘
                                              │
                    ┌─────────────────────────┴────────────────────────┐
                    ▼                                                  ▼
          ┌───────────────────┐                              ┌──────────────────┐
          │  QUERY PROCESSING │  classify · typo-correct     │  SQLite (WAL)    │
          └─────────┬─────────┘                              │  users, docs,    │
                    ▼                                        │  chat history    │
          ┌───────────────────┐                              └──────────────────┘
          │  EMBED THE QUERY  │  text-embedding-3-small
          └─────────┬─────────┘
                    ▼
          ┌───────────────────────────────────────────┐
          │  HYBRID RETRIEVAL                         │
          │   • FAISS dense vector search  (weight .6)│ ◄── per-document
          │   • Okapi BM25 keyword search  (weight .4)│     FAISS indices
          │   • heuristic boosts (definition/header)  │     on disk
          │   • drop anything below 0.50              │
          │   • expand with the adjacent chunk        │
          └─────────────────┬─────────────────────────┘
                            ▼
          ┌───────────────────────────────────────────┐
          │  CONTEXT  → top-k passages + page metadata │
          └─────────────────┬─────────────────────────┘
                            ▼
          ┌───────────────────────────────────────────┐
          │  LLM  (OpenRouter, Gemini fallback)       │
          │  "answer only from this context"          │
          └─────────────────┬─────────────────────────┘
                            ▼
          ┌───────────────────────────────────────────┐
          │  RESPONSE  streamed over SSE, with cited  │
          │  pages, confidence, retrieval telemetry   │
          └───────────────────────────────────────────┘
```

For comparative and multi-step reasoning, execution transitions through a **LangGraph `StateGraph`**:

```
Query ──► Planner ──► Retriever ──► Synthesizer ──► Verifier ──► Verified Answer
         (split into   (hybrid      (cross-source    (verifies claims
          2-3 sub-Qs)   search per   synthesis +      against context &
                        sub-query)   citations)       prunes citations)
```

### Observability & Telemetry

Every retrieval cycle logs a structured telemetry event tracking query hash, candidate volumes, component latencies, and applied boosts:

```json
{
  "event": "rag_retrieval",
  "query_hash": "a4f8c12e",
  "candidates": 15,
  "passed_threshold": 4,
  "top_score": 0.892,
  "vector_ms": 0.84,
  "bm25_ms": 0.41,
  "boost_applied": "proximity"
}
```

Aggregated metrics—including total queries, average latency, zero-hit frequency, and boost application rates—are exposed via `GET /api/metrics`. In the UI, every response includes metadata showing retrieval duration, chunk count, and the active LLM.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 · JavaScript (ES2022) |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| Frontend | React 19, Vite, Axios |
| LLM | `nvidia/nemotron-3-nano-30b-a3b:free` via OpenRouter (fallback: `gemini-3.6-flash`) |
| Embeddings | `openai/text-embedding-3-small` (1536 dim) via OpenRouter |
| Vector Store | FAISS (`faiss-cpu`), local per-document index files |
| Keyword Search | Custom Okapi BM25 (`k₁=1.5`, `b=0.75`) with CamelCase splitting |
| Agent Framework | LangGraph (`StateGraph` planner, retriever, synthesizer, verifier) |
| Database | SQLite in WAL mode (`docmind.db`) |
| Auth | PyJWT + bcrypt with user-scoped isolation |
| Tests | pytest, pytest-cov, Vitest, GitHub Actions CI |
| Deployment | Multi-stage Docker container → Render free tier (single web service) |

---

## How RAG Works

The ingestion and retrieval lifecycle is divided into six deterministic phases:

1. **Ingest:** Documents are extracted page-by-page via `pypdf` (or specialized parsers for Markdown, TXT, HTML, and DOCX). Original page numbers and section headers are preserved as immutable chunk metadata. Scanned documents devoid of text layers are rejected immediately.
2. **Chunk:** `RecursiveCharacterTextSplitter` segments text on paragraph, sentence, and word boundaries at **1,000 characters with 150 characters of overlap (15%)**. Each chunk retains `doc_id`, `doc_name`, `page`, and `chunk_index`.
3. **Embed:** Chunks are vectorized using `text-embedding-3-small` in batches of 700 chunks per request and written to a dedicated per-document FAISS index on disk.
4. **Retrieve:** The user query is vectorized, and FAISS fetches an expanded candidate shortlist:
   $$\text{candidate\_count} = \max(\text{TOP\_K} \times 3, 15)$$
   Each candidate is scored across dense and sparse channels, normalized, boosted, and reranked:
   $$\text{Hybrid Score} = 0.6 \cdot \text{Vector Similarity} + 0.4 \cdot \text{Normalized BM25} + \text{Heuristic Boosts}$$
5. **Gate:** Chunks falling below `RELEVANCE_THRESHOLD=0.50` are dropped unless they satisfy the independent lexical coverage route (`LEXICAL_COVERAGE_THRESHOLD=0.50`). If no candidates survive, the request terminates early with an honest refusal.
6. **Generate:** Qualifying passages are expanded with their adjacent chunk and injected into a numbered context block. The LLM synthesizes the response, explicitly citing used sources. Unreferenced sources are pruned prior to streaming.

---

## Local Setup

### Prerequisites
- Python 3.12+
- Node.js 20+
- An [OpenRouter API key](https://openrouter.ai/keys) (free tier supported)

### 1. Clone & Configure Secrets

```bash
git clone https://github.com/Surya1207-Gv/-DocMind-AI.git
# Note: The './' prefix prevents shells from interpreting the leading dash as a command flag
cd ./-DocMind-AI

# Windows
copy .env.example backend\.env

# Linux/macOS
cp .env.example backend/.env
```

Generate a secure secret key for JWT token signing:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Open `backend/.env` and assign the generated string to `JWT_SECRET_KEY`, and add your `OPENROUTER_API_KEY`.

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv backend/venv

# Activate virtual environment
# Windows (Command Prompt):
backend\venv\Scripts\activate
# Windows (PowerShell):
.\backend\venv\Scripts\Activate.ps1
# Linux/macOS:
source backend/venv/bin/activate

# Install dependencies
pip install -r backend/requirements-dev.txt

# Start backend dev server
uvicorn backend.main:app --reload --port 8000
```

### 3. Frontend Setup (Second Terminal)

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173`. Vite proxies `/api` requests directly to `http://127.0.0.1:8000`.

### 4. Running Production Build Locally (Single Origin)

To test the exact single-origin deployment served entirely by FastAPI:

```bash
# Build frontend
cd frontend
npm run build
cd ..

# Serve via FastAPI
uvicorn backend.main:app --port 8000
# Open http://localhost:8000
```

### 5. Running with Docker Compose

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env

# Start container
docker compose up --build
# Open http://localhost:8000
```

---

## Environment Variables

> **Security Note:** Never commit `.env` or credential files to version control. The repository `.gitignore` explicitly excludes `.env`, `backend/.env`, and local SQLite/FAISS state.

### Required Variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | API key for LLM generation and embedding inference via [OpenRouter](https://openrouter.ai/keys). |
| `JWT_SECRET_KEY` | — | Cryptographic secret used to sign and verify JWT session tokens. |

### Optional Variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini fallback key (`gemini-3.6-flash`). Preferred over OpenRouter when set. |
| `DATA_DIR` | `backend/` | Single writable directory for uploads, FAISS indices, SQLite DB, and log files. |
| `ALLOWED_ORIGINS` | localhost dev origins | Comma-separated CORS origins. Leave empty in production (same-origin). |
| `FRONTEND_DIST_DIR` | `frontend/dist` | Path to compiled React production assets served by FastAPI. |
| `LLM_MODEL` | `nvidia/nemotron-3-nano-30b-a3b:free` | Default model identifier for generation. |
| `EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Default embedding model identifier (1536 dimensions). |
| `CHUNK_SIZE` | `1000` | Target character size per document chunk. |
| `CHUNK_OVERLAP` | `150` | Character overlap shared between adjacent chunks. |
| `TOP_K` | `8` | Maximum number of context chunks delivered to the LLM. |
| `RELEVANCE_THRESHOLD` | `0.5` | Minimum hybrid similarity score required for candidate inclusion. |
| `VECTOR_WEIGHT` | `0.6` | Dense vector weight in hybrid scoring (BM25 weight is `1.0 - VECTOR_WEIGHT`). |
| `LEXICAL_COVERAGE_THRESHOLD` | `0.5` | Threshold for independent lexical coverage admission route. |
| `MAX_UPLOAD_MB` | `25` | Maximum allowed file upload size in megabytes. |
| `DEMO_SEED` | `false` | Automatically indexes bundled sample PDF on startup when set to `true`. |
| `DEMO_USERNAME` | `demo` | Username for the pre-seeded public demo account. |
| `DEMO_PASSWORD` | `demo1234` | Default password for local development and public demo instances (do not use in production). |
| `RERANKER` | `lexical` | Second-stage reranker algorithm (`lexical`, `llm`, or `none`). |
| `RERANK_WEIGHT` | `0.4` | Displacement weight applied by the reranker on fused scores. |
| `RERANK_CANDIDATES` | `15` | Shortlist candidate count evaluated by the reranker. |
| `VERIFICATION_ENABLED` | `true` | Enables lexical claim support and hallucination verification. |
| `EVIDENCE_GATE_ENABLED` | `true` | Gates answers below confidence thresholds with explicit warnings or refusals. |
| `LOG_LEVEL` | `INFO` | Application log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |

---

## Running Tests

### Backend Test Suite
The backend test suite contains **317 tests** across unit, API integration, and regression suites:

```bash
# Windows (PowerShell)
$env:PYTHONPATH="."
python -m pytest backend/tests/ -v --cov=backend

# Linux/macOS
PYTHONPATH=. python -m pytest backend/tests/ -v --cov=backend
```

### Frontend Test Suite
The React application contains **46 tests** covering UI components, confidence meters, and document management:

```bash
cd frontend
npm run test
```

### Retrieval Benchmark Suite
To execute the empirical retrieval benchmark across the 1,200-chunk corpus:

```bash
python eval/run_eval.py
```

---

## API Endpoints

Interactive Swagger UI documentation is available at `/docs` (and ReDoc at `/redoc`) when the backend is running.

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | No | System health probe checking database connectivity, storage writability, and LLM configuration. |
| `GET` | `/api/info` | No | Service descriptor returning active model names, upload limits, and default retrieval parameters. |
| `POST` | `/api/auth/register` | No | Register a new user account with email and username validation. |
| `POST` | `/api/auth/login` | No | Authenticate credentials and issue a signed JWT access token. |
| `PUT` | `/api/users/me` | Yes | Update current user profile attributes. |
| `GET` | `/api/chats/active` | Yes | List active conversation sessions belonging to the caller. |
| `POST` | `/api/upload` | Yes | Upload and index a document (magic-byte validation and size enforcement). |
| `POST` | `/api/documents/from-url` | Yes | Ingest and index a document from a public URL (with SSRF protection). |
| `GET` | `/api/documents` | Yes | List all indexed documents owned by the caller. |
| `DELETE` | `/api/documents/{doc_id}` | Yes | Delete a document, its database records, and its on-disk FAISS index. |
| `GET` | `/api/documents/relationships` | Yes | Detect duplicate, version, and topical relationships across documents. |
| `POST` | `/api/chat` | Yes | Single-shot RAG conversation streamed via Server-Sent Events (SSE). |
| `POST` | `/api/chat/agent` | Yes | Multi-hop LangGraph agent Q&A streamed via SSE. |
| `GET` | `/api/chat/history/{doc_id}` | Yes | Retrieve persistent conversation history for a document. |
| `DELETE` | `/api/chat/history/{doc_id}` | Yes | Clear conversation history associated with a document. |
| `GET` | `/api/analytics/{doc_id}` | Yes | Fetch background document analytics (summary, entities, alerts, suggested questions). |
| `POST` | `/api/quiz/{doc_id}` | Yes | Generate a multiple-choice assessment with page citations. |
| `POST` | `/api/compare` | Yes | Perform multi-document comparison and synthesis. |
| `POST` | `/api/agent/query` | Yes | Execute synchronous multi-hop query decomposition (JSON response). |
| `POST` | `/api/rag/trace` | Yes | Retrieve step-by-step scoring, candidate rankings, and gating decisions for a query. |
| `GET` | `/api/telemetry` | Yes | Fetch aggregate RAG operational telemetry. |
| `GET` | `/api/metrics` | Yes | Fetch operational metrics (latencies, zero-hit rate, boost counts). |

---

## Evaluation

The retrieval pipeline was evaluated on the project's controlled 1,200-chunk benchmark across four document archetypes (technical RFCs, AI research papers, regulatory compliance guidelines, and cloud infrastructure architecture) using 60 discriminating labeled queries.

**Key Metrics:**
- **Recall@4:** 98.3% (95.0% at operating threshold 0.50)
- **nDCG@4:** 0.9095 (0.8885 at operating threshold 0.50)
- **MRR (Mean Reciprocal Rank):** 0.8839
- **Zero-Hit Rate:** 0.0%
- **LangGraph Multi-Hop Agent Recall@4:** 100.0% (+9.1% improvement over single-shot 90.9%)

See [Retrieval Benchmark](docs/retrieval_benchmark.md) for full methodology, ablation sweeps, and failure analysis.  
For technical interview deep dives and architectural rationale, see [Interview Notes](docs/interview-notes.md).

---

## Deployment

DocMind AI is deployed as **a single Docker web service on Render's free tier**. A multi-stage Docker build compiles the Vite React SPA in a Node environment, then copies static assets into the Python runtime where FastAPI serves them from a single origin.

### Render Blueprint Deploy

1. Push the repository to GitHub.
2. In the [Render Dashboard](https://dashboard.render.com): click **New → Blueprint** and select the repository. Render reads [`render.yaml`](render.yaml) automatically.
3. Configure prompted secrets:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key.
   - `DEMO_PASSWORD`: Desired password for the public demo account.
   - `GEMINI_API_KEY`: (Optional) Google Gemini API key.
   - *Note:* `JWT_SECRET_KEY` is automatically generated by Render (`generateValue: true`).
4. Click **Deploy**. The app will be live at `https://<service-name>.onrender.com`.

### Operational Notes (Free Tier)
- **Cold Starts:** Free instances spin down after ~15 minutes of inactivity. Initial wake-up requests take 30–60 seconds.
- **Ephemeral Storage:** Free instances do not include persistent disks; local SQLite state and FAISS indices reset upon restarts. Setting `DEMO_SEED=true` re-indexes `assets/demo/sample.pdf` on boot so the platform is always demo-ready.
- **Production Persistence:** For production use, uncomment the `disk:` configuration in `render.yaml` and switch the plan to `starter` to mount persistent storage under `/data`.

---

## Limitations

- **No OCR Support:** Scanned PDFs without an embedded text layer cannot be parsed. The upload is rejected with a clear error rather than creating an empty index.
- **Table Structure Flattening:** Standard PDF text extraction flattens tabular structures, reducing retrieval precision on matrix data.
- **Free-Tier LLM Constraints:** The default free model (`nemotron-3-nano-30b`) has limited contextual capacity compared to frontier models; switching to frontier models (e.g., GPT-4) is configurable via `LLM_MODEL`, or by enabling Google Gemini via `GEMINI_API_KEY`.
- **In-Process Vector Store:** FAISS indices run in-process on the local host. While fast for demo and per-user collections, horizontal multi-replica scaling requires transitioning to a distributed vector database (e.g., Qdrant or pgvector).
- **Ephemeral Free-Tier Storage:** State resets upon instance reboot unless attached to a persistent volume.
- **English-Centric Regex Heuristics:** Definition and header boosts utilize English linguistic markers and require tuning for other languages.
- **External Embedding Latency:** Overall retrieval latency is dominated by the external embedding API round-trip (~1.0–1.5s), while local FAISS similarity search executes in ~30ms.

---

## Future Improvements

- **Cross-Encoder Reranker:** Integrate a lightweight cross-encoder model to rescore the top candidate pool for higher precision.
- **OCR Pipeline Integration:** Implement Tesseract or PyMuPDF OCR fallback for scanned and image-heavy documents.
- **Managed Vector Store Adapter:** Add pluggable support for Qdrant or pgvector to enable horizontal scaling and distributed persistence.
- **Multi-Turn Query Rewriting:** Integrate historical conversation context into query reformulation for complex multi-turn follow-ups.
- **Streaming Document Ingestion:** Provide real-time chunking and vectorization progress indicators over WebSockets or SSE.
- **Automated CI Regression Gating:** Add automated gates in GitHub Actions to block pull requests that cause retrieval recall degradation on benchmark datasets.

---

## Project Structure

```text
├── Dockerfile                  # Multi-stage build: compiles React SPA and runs FastAPI
├── render.yaml                 # Render Infrastructure-as-Code blueprint
├── docker-compose.yml          # Local containerized orchestration
├── .env.example                # Documented configuration template
├── assets/
│   └── demo/
│       └── sample.pdf          # Bundled sample document for demo seeding
├── backend/
│   ├── main.py                 # FastAPI application, route controllers, and SPA hosting
│   ├── config.py               # Environment variable parsing and default parameters
│   ├── auth.py                 # JWT token generation, bcrypt hashing, and authentication
│   ├── database.py             # SQLite WAL database schema, connection, and queries
│   ├── logger.py               # Structured logging and RAG retrieval telemetry
│   ├── pdf_processor.py        # PDF text extraction and chunking pipeline
│   ├── document_processor.py   # Multi-format parsing (MD, TXT, HTML, DOCX, URLs)
│   ├── embedding_manager.py    # OpenRouter embeddings, FAISS indices, and BM25 hybrid search
│   ├── chat_engine.py          # SSE streaming, system prompts, and citation pruning
│   ├── agent_engine.py         # LangGraph multi-hop StateGraph implementation
│   ├── analytics_engine.py     # Background document summary and entity extraction
│   ├── quiz_engine.py          # Automated assessment generation
│   ├── compare_engine.py       # Cross-document comparison logic
│   ├── reranker.py             # Deterministic lexical candidate reranker
│   ├── verification.py         # Evidence gating and claim verification engine
│   ├── demo_seed.py            # Startup indexing for demo document
│   └── tests/                  # 317 automated pytest test cases
├── frontend/
│   ├── src/                    # React 19 application source code
│   └── src/__tests__/          # 46 Vitest component and utility tests
├── eval/
│   ├── dataset.json            # 60 ground-truth labeled benchmark queries
│   └── run_eval.py             # Retrieval benchmark evaluation runner
└── docs/
    ├── retrieval_benchmark.md  # 1,200-chunk empirical benchmark report
    ├── interview-notes.md      # Technical interview preparation and architectural deep dive
    └── v1_architecture_legacy.md # Architectural history and evolution
```

---

## License

Distributed under the [MIT License](LICENSE). Built by Surya Sasank.
