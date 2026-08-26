# DocMind AI — Retrieval-Augmented Document Intelligence

[![Backend Tests](https://github.com/Surya1207-Gv/-DocMind-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/Surya1207-Gv/-DocMind-AI/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61DAFB.svg)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Ask questions about your own PDFs and get answers that cite the exact page they
> came from — or an honest "that isn't in the document."

---

## Overview

Large language models are fluent but ungrounded. Ask one about a contract, a
policy, or a research paper it has never seen and it will usually invent a
confident, wrong answer. That is unusable for any document you actually care
about being right.

**DocMind AI solves this with retrieval-augmented generation.** Your PDF is
parsed, chunked, embedded and indexed. When you ask a question, the system
retrieves the passages that genuinely answer it, hands *only those passages* to
the model, and shows you which page each claim came from. If nothing relevant is
found, it says so rather than guessing.

The interesting engineering is in the retrieval step. Naive vector search misses
exact identifiers; naive keyword search misses paraphrase. DocMind runs **both**
and fuses the scores, a design choice validated on a 1,200-chunk benchmark
(details below).

---

## Architecture

The entire product ships as **one service on one origin** — FastAPI serves both
the JSON API and the compiled React SPA. There is no separate frontend
deployment, no CORS negotiation, and one URL to share.

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

For multi-hop and comparative questions, the request is routed through a
**LangGraph `StateGraph`** instead:

```
Query ─► Planner ─► Retriever ─► Synthesizer ─► Verifier ─► Verified answer
         (split     (hybrid      (cross-source   (checks claims
          into 2-3   search per   synthesis      back against
          sub-Qs)    sub-query)   + citations)   the context)
```

---

## Features

**Retrieval quality**
- Hybrid dense + sparse retrieval with weighted score fusion
- Custom Okapi BM25 implementation (`k₁=1.5`, `b=0.75`) with CamelCase splitting
- Heuristic boosts for definition-style questions and section headers
- Relevance gating — low-scoring chunks are dropped, not passed to the model
- Adjacent-chunk expansion so retrieved passages do not end mid-sentence

**Grounding and trust**
- Page-level citations on every answer, expandable to the exact source text
- Confidence score derived from the fused retrieval score
- Explicit refusal when nothing passes the relevance threshold
- Post-hoc citation pruning: sources the model did not use are removed

**Product**
- Four answer modes: Q&A, Summary, Deep Analysis, ELI5
- Background document analysis: summary, entities, alerts, suggested questions
- Auto-generated multiple-choice quizzes with page references
- Multi-document comparison
- Conversation history and PDF export
- One-click demo login with a sample document pre-indexed

**Engineering**
- JWT auth with bcrypt hashing; per-user isolation of documents and indices
- 71 automated tests (62 backend, 9 frontend) running in CI
- Structured RAG telemetry (`/api/metrics`) — latency, zero-hit rate, boost rate
- Health endpoint probing database, index writability and LLM configuration
- Fully environment-driven configuration; no secrets in code

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 · JavaScript (ES2022) |
| Backend | FastAPI 0.110, Uvicorn, Pydantic v2 |
| Frontend | React 19, Vite 8, Axios |
| LLM | `nvidia/nemotron-3-nano-30b-a3b:free` via OpenRouter (Gemini 2.5 Flash fallback) |
| Embeddings | `openai/text-embedding-3-small` via OpenRouter |
| Vector store | FAISS (`faiss-cpu`), one index per document, persisted to disk |
| Keyword search | Custom Okapi BM25 (no external dependency) |
| Agent | LangGraph `StateGraph` |
| Database | SQLite in WAL mode |
| Auth | PyJWT + bcrypt |
| Tests | pytest, Vitest, GitHub Actions |
| Deployment | Docker (multi-stage) → Render free tier |

---

## How RAG Works Here

**1 · Ingest.** `pypdf` extracts text page by page. Page numbers are captured at
extraction time, because a citation that cannot name a page is not verifiable. A
PDF that yields no text is rejected as scanned rather than indexed as an empty
document.

**2 · Chunk.** `RecursiveCharacterTextSplitter` splits on paragraph → sentence →
word boundaries at **1000 characters with 150 characters of overlap**. Each chunk
carries `doc_id`, `doc_name`, `page` and `chunk_index` as metadata.

**3 · Embed.** Chunks are embedded in batches of up to 500 per API call and
written to a FAISS index named after the document, so deleting a document is a
directory removal with no global rebuild.

**4 · Retrieve.** The query is embedded, then FAISS returns `max(3·k, 15)`
candidates. Each candidate is scored by both methods and fused:

```
hybrid = 0.6 · vector_similarity + 0.4 · normalised_BM25   (+ heuristic boosts)
```

**5 · Gate.** Candidates below `0.50` are discarded. This is the step that
prevents hallucination: returning the least-bad chunk when nothing relevant
exists is exactly how a "grounded" system ends up making things up anyway.

**6 · Generate.** Surviving chunks become a numbered context block. The system
prompt instructs the model to answer only from that block and to list the source
indices it used. Unused sources are pruned before the final SSE event.

---

## Local Setup

**Prerequisites:** Python 3.12, Node.js 20, and an
[OpenRouter API key](https://openrouter.ai/keys) (the free tier is sufficient).

```bash
git clone https://github.com/Surya1207-Gv/-DocMind-AI.git
cd -DocMind-AI

# 1. Configure secrets
cp .env.example backend/.env          # Windows: copy .env.example backend\.env
#    then set OPENROUTER_API_KEY and JWT_SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"   # value for JWT_SECRET_KEY

# 2. Backend
python -m venv backend/venv
source backend/venv/Scripts/activate  # Linux/macOS: source backend/venv/bin/activate
pip install -r backend/requirements-dev.txt
uvicorn backend.main:app --reload --port 8000

# 3. Frontend (second terminal)
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to port 8000, so the frontend
uses the same same-origin paths it will use in production.

**To run exactly what gets deployed** (SPA compiled and served by FastAPI):

```bash
cd frontend && npm run build && cd ..
uvicorn backend.main:app --port 8000
# open http://localhost:8000
```

**Or with Docker:**

```bash
cp .env.example .env      # fill in OPENROUTER_API_KEY and JWT_SECRET_KEY
docker compose up --build
# open http://localhost:8000
```

### Tests

```bash
python -m pytest backend/tests/ -v --cov=backend    # 62 backend tests
cd frontend && npm run test                          # 9 frontend tests
python eval/run_eval.py                              # retrieval benchmark
```

---

## Environment Variables

Copy `.env.example` and fill it in. **Never commit `.env`** — it is gitignored.

### Required

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Serves both the chat model and the embedding model. Get one at [openrouter.ai/keys](https://openrouter.ai/keys). |
| `JWT_SECRET_KEY` | Signs session tokens. Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. The app refuses to start without it. |

### Optional

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini 2.5 Flash fallback LLM. Preferred over OpenRouter when set. |
| `DATA_DIR` | `backend/` | Single writable root for uploads, indices, database and logs. Point at a mounted volume to persist state. |
| `ALLOWED_ORIGINS` | localhost dev origins | CORS allowlist. Leave empty in production — the SPA is same-origin. |
| `DEMO_SEED` | `false` | Index the bundled sample PDF at startup so a fresh instance is never empty. |
| `DEMO_USERNAME` / `DEMO_PASSWORD` | `demo` / `demo1234` | Credentials for the one-click demo account. |
| `TOP_K` | `8` | Chunks passed to the LLM as context. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Chunking parameters. |
| `RELEVANCE_THRESHOLD` | `0.5` | Below this fused score a chunk is discarded. |
| `VECTOR_WEIGHT` | `0.6` | Vector share of the hybrid score; BM25 gets the remainder. |
| `MAX_UPLOAD_MB` | `25` | Upload size limit. |
| `LLM_MODEL` / `EMBEDDING_MODEL` | see table above | Swap models without touching code. |
| `ADMIN_PASSWORD` | random | Only needed to log in as the legacy `admin` account. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

---

## Deployment

Deployed as **a single Docker web service on Render's free tier**. The image
compiles the React SPA in one stage and serves it from FastAPI in the next, so
there is one service, one URL and no CORS configuration.

### Steps

1. **Push to GitHub.**

2. **Create the service.** In the [Render dashboard](https://dashboard.render.com):
   *New → Blueprint* → select this repository. Render reads `render.yaml` and
   configures the Docker service automatically.
   (Or *New → Web Service* → Runtime **Docker**, Dockerfile path `./Dockerfile`,
   health check path `/api/health`.)

3. **Set the secrets** when prompted:
   - `OPENROUTER_API_KEY` — your key
   - `DEMO_PASSWORD` — any password for the demo account
   - `JWT_SECRET_KEY` — Render generates this automatically
   - `GEMINI_API_KEY` — optional, may be left blank

4. **Deploy.** The first build takes roughly 5–10 minutes. Your app is then live at
   `https://<service-name>.onrender.com`.

### Free-tier behaviour worth knowing

- **Cold starts.** The instance sleeps after ~15 minutes idle; the next request
  takes 30–60 seconds to wake it. Load the URL a minute before a demo.
- **Ephemeral disk.** Free instances have no persistent volume, so uploads and
  chat history reset when the instance restarts. `DEMO_SEED=true` re-indexes the
  bundled sample PDF on every boot, so the app is never empty. To persist state,
  move to a paid instance and uncomment the `disk:` block in `render.yaml`.

---

## Example Questions

The demo account has a sample handbook pre-indexed. Try:

- *What is Retrieval-Augmented Generation?* — definition-style retrieval
- *What is the default chunk size and overlap, and why?* — exact numeric facts
- *Why does the system use hybrid retrieval instead of vector search alone?* — reasoning across a full section
- *What happens when no chunk passes the relevance threshold?* — tests the grounding behaviour
- *How does chunking differ from indexing?* — comparative, routes through the agent
- *What was Tesla's 2019 revenue in Norway?* — **should be refused**; nothing in the corpus answers it

That last one is the important demo. A system that answers it is broken.

---

## Limitations

Stated honestly, because these are the questions an interviewer will ask:

- **No OCR.** Scanned PDFs without a text layer cannot be read. The upload is
  rejected with a clear message rather than silently indexed as empty.
- **Tables lose structure.** Text extraction flattens columns, so tabular data
  retrieves poorly.
- **Free-tier LLM.** The default model is a small free model; answer quality is
  below GPT-4-class models. Swapping `LLM_MODEL` is a one-variable change.
- **In-process FAISS.** Indices are loaded and merged per query. This is fast and
  free for demo-scale corpora but will not scale to millions of chunks or
  horizontal replicas — that needs a managed vector database.
- **No persistence on the free tier.** See the deployment note above.
- **Heuristic boosts are English-specific.** The definition and header boosts use
  English regex patterns and would need rework for other languages.
- **Retrieval latency is dominated by the embedding API call** (~1–1.5 s), not by
  FAISS (~30 ms).

---

## Future Improvements

- Cross-encoder re-ranking on the top ~20 candidates for better precision
- OCR fallback (Tesseract) for scanned documents
- Swap FAISS for Qdrant or pgvector to support replicas and metadata filtering
- Query rewriting from conversation history for better follow-up handling
- Streaming ingestion with progress feedback for large uploads
- Automated regression gate: fail CI if benchmark recall drops

---

## How I Would Explain This Project in an Interview

**The problem.** LLMs hallucinate on documents they were never trained on. If you
ask one about your own contract or policy, you get a fluent answer with no way to
tell whether it is true. I wanted a system where every claim is traceable to a
page, and where "I don't know" is a valid, reachable answer.

**The architecture.** A FastAPI backend and a React SPA, shipped as a single
Docker image where FastAPI serves the compiled frontend. One service, one origin,
no CORS. Ingestion extracts text per page, chunks it, embeds it, and writes a
FAISS index per document. Queries embed, retrieve, gate, and prompt.

**Why RAG rather than fine-tuning.** Fine-tuning bakes knowledge into weights: it
is expensive, must be redone whenever a document changes, and still cannot cite a
source. RAG keeps knowledge in an index — updating means re-indexing one file —
and because the evidence is in the prompt, every answer is auditable. For a
document-QA product, auditability is the whole requirement.

**Why FAISS.** The corpus is per-user and demo-scale. FAISS is a library, not a
service: no extra container, no network hop, no monthly bill, and it persists to
disk as two files I can delete per document. A managed vector DB like Pinecone
would add cost and an availability dependency for capabilities — replication,
metadata filtering at scale — that this workload does not need. I know where the
ceiling is: once indices exceed memory or I need multiple replicas, FAISS-in-
process stops being the right answer and I would move to Qdrant or pgvector.

**Chunking strategy.** 1000 characters with 150 characters (15%) of overlap,
split recursively on paragraph then sentence boundaries. The trade-off is
concrete: chunks that are too small retrieve confidently but answer
incompletely, because the surrounding argument is gone; chunks that are too large
dilute the embedding, because one vector has to represent several ideas. The
overlap exists so a sentence straddling a boundary still appears whole in one
neighbour. I also attach the *next* chunk to every retrieved chunk, because a
retrieved passage often ends mid-argument.

**Embedding strategy.** `text-embedding-3-small` — 1536 dimensions, strong
quality per dollar, and reached through OpenRouter so the same key serves both
embeddings and generation. Chunks are batched 500 per request, which turned
ingestion from many round trips into a handful.

**Retrieval strategy — the part I'd want to be asked about.** I use hybrid
retrieval because the two methods fail in complementary ways, and I measured it
rather than assumed it. On a 1,200-chunk benchmark with 60 labelled queries,
pure BM25 collapsed to 25% Recall@4 on paraphrase queries with no vocabulary
overlap, while pure vector search dropped to 66.7% on rare identifiers like
`RFC 8446` — embeddings blur precise tokens. Fusing them at 0.6/0.4 lifted
Recall@4 from 75.0% (vector) and 73.3% (BM25) to 86.7%, and adding the
definition-proximity boost took it to 95.0% with nDCG@4 of 0.90 and mean rank
1.52. The cost is latency: 27 ms → 53 ms for retrieval, which is negligible
next to the ~1.4 s embedding call and multi-second generation.

**Prompt strategy.** Retrieved chunks are injected as a numbered context block
with document name and page. The system prompt constrains the model to that
context and requires it to emit the source indices it actually used, which I then
use to prune unused citations — so the UI never shows a source the answer did not
rely on. There are four prompt variants (Q&A, summary, deep, ELI5) with different
temperatures.

**Evaluation and limitations.** I measure Recall@k and MRR, not vibes: recall
asks whether the right passage was retrieved at all, MRR asks how near the top it
was, and they can move in opposite directions — widening the candidate set raises
recall while hurting precision. The relevance threshold is the honesty mechanism:
if nothing scores above 0.50, the system refuses instead of answering from the
least-bad chunk. Known weaknesses: no OCR, tables lose structure, a small free
LLM, and FAISS-in-process won't scale horizontally.

**Deployment.** One Docker image on Render's free tier. The multi-stage build
compiles the SPA with Node then copies the static build into the Python runtime,
so the deployed artifact is byte-identical to what I test locally. Configuration
is entirely environment-driven — chunk size, top-k, threshold, and model names
are all env vars, so tuning retrieval does not require a code change. All runtime
state lives under a single `DATA_DIR` so attaching a persistent volume is a
one-variable change. I know the free tier's constraints — cold starts and an
ephemeral disk — so the app re-seeds a bundled sample document on boot and is
never empty when someone opens the link.

---

## API Surface

18 endpoints. Interactive docs at `/docs` when running.

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | No | DB, index-writability and LLM configuration probe |
| `GET` | `/api/info` | No | Service descriptor: active model names and top-k |
| `POST` | `/api/auth/register` | No | Register a user |
| `POST` | `/api/auth/login` | No | Obtain a JWT |
| `PUT` | `/api/users/me` | Yes | Update profile |
| `GET` | `/api/chats/active` | Yes | List active conversations |
| `POST` | `/api/upload` | Yes | Upload a PDF (magic-byte + size validated) |
| `GET` | `/api/documents` | Yes | List the caller's documents |
| `DELETE` | `/api/documents/{doc_id}` | Yes | Delete a document and its index |
| `POST` | `/api/chat` | Yes | Single-shot RAG chat, streamed over SSE |
| `POST` | `/api/chat/agent` | Yes | Multi-hop LangGraph agent, streamed over SSE |
| `GET` | `/api/chat/history/{doc_id}` | Yes | Conversation history |
| `DELETE` | `/api/chat/history/{doc_id}` | Yes | Clear conversation history |
| `GET` | `/api/analytics/{doc_id}` | Yes | Background document analytics |
| `POST` | `/api/quiz/{doc_id}` | Yes | Generate a multiple-choice quiz |
| `POST` | `/api/compare` | Yes | Cross-document comparison |
| `POST` | `/api/agent/query` | Yes | Multi-hop agent, JSON response |
| `GET` | `/api/telemetry` · `/api/metrics` | Yes | Aggregate RAG operational metrics |

---

## Empirical Retrieval Benchmark

> **Corpus note.** To measure retrieval under repeatable, un-confounded
> conditions, performance is evaluated on a controlled 1,200-chunk corpus across
> four document archetypes (dense RFC specifications, deep-learning papers,
> banking/compliance regulations, cloud infrastructure guides) with 60
> ground-truth labelled queries. Top-4 retrieval inspects 0.33% of the corpus.

| Retrieval configuration | nDCG@4 | Mean rank | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR | Score separation | Latency |
|---|---|---|---|---|---|---|---|---|---|
| 1 · Pure vector (FAISS only) | 0.6974 | 9.23 | 58.3% | 75.0% | 85.0% | 18.8% | 0.6854 | −0.0821 | 27.67 ms |
| 2 · Pure BM25 (keyword only) | 0.7121 | 24.87 | 66.7% | 73.3% | 75.0% | 18.3% | 0.7093 | −0.1245 | 26.54 ms |
| 3 · Naive hybrid (60/40) | 0.8149 | 3.45 | 73.3% | 86.7% | 96.7% | 21.7% | 0.8032 | +0.0412 | 47.92 ms |
| **4 · DocMind boosted hybrid (prod)** | **0.9023** | **1.52** | **86.7%** | **95.0%** | **100.0%** | **23.8%** | **0.8942** | **+0.2315** | **52.88 ms** |

**Findings**

- **BM25's blind spot.** On paraphrase queries with zero vocabulary overlap,
  pure BM25 falls to 25.0% Recall@4 while vector search holds 83.3%.
- **Vector's blind spot.** On rare identifiers (`RFC 8446`, `PCI-DSS 4.0 Req 3.4`),
  pure vector search drops to 66.7% Recall@4 while BM25 scores 100%.
- **Definition-proximity boost (+0.45).** Lifts definitional queries from 80.0%
  to 100.0% Recall@4 and improves nDCG@4 from 0.8149 to 0.8917.
- **Score separation (+0.2315).** In production configuration the relevant chunk
  scores 0.23 above the best distractor, which is what makes a fixed 0.50
  threshold dependable.

Reproduce with `python eval/run_eval.py`. Full method in
[`docs/retrieval_benchmark.md`](docs/retrieval_benchmark.md).

---

## Observability

Every retrieval emits a structured event:

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

Aggregates — query count, mean latency, zero-hit rate, boost rate — are exposed
at `GET /api/metrics`. The UI additionally shows, under each answer, how many
chunks were retrieved, how long retrieval took, and which model responded.

A rising **zero-hit rate** is the leading indicator that retrieval has degraded.

---

## Project Structure

```
├── Dockerfile              # multi-stage: build SPA → serve from FastAPI
├── render.yaml             # Render blueprint (one web service)
├── docker-compose.yml      # local container run
├── .env.example            # every configurable variable, documented
├── assets/demo/sample.pdf  # bundled demo document (original content)
├── backend/
│   ├── main.py             # FastAPI app, routes, SPA hosting
│   ├── config.py           # all configuration, environment-driven
│   ├── pdf_processor.py    # extraction + chunking
│   ├── embedding_manager.py# embeddings, FAISS, BM25, hybrid retrieval
│   ├── chat_engine.py      # prompting, SSE streaming, citation pruning
│   ├── agent_engine.py     # LangGraph multi-hop agent
│   ├── demo_seed.py        # boot-time sample indexing
│   ├── auth.py database.py logger.py
│   └── tests/              # 62 tests
├── frontend/src/           # React SPA
├── eval/                   # retrieval benchmark harness
└── docs/                   # benchmark method, legacy architecture notes
```

---

## License

[MIT](LICENSE) · Built by Surya Sasank
