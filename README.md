# DocMind AI — Production-Grade RAG Document Intelligence Platform

[![Backend Tests](https://github.com/Surya1207-Gv/-DocMind-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/Surya1207-Gv/-DocMind-AI/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.2-61DAFB.svg)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.74-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

DocMind AI is an enterprise-grade Retrieval-Augmented Generation (RAG) platform engineered from the ground up. It goes far beyond basic "chat with PDF" scripts by implementing **empirically benchmarked hybrid retrieval**, a **LangGraph multi-hop agent layer**, **streaming LLM responses (SSE)** with post-hoc citation pruning, **structured RAG telemetry**, multi-tenant JWT auth, and a responsive glassmorphic React UI.

**Tech Stack:** Python 3.12 · FastAPI · React 19 (Vite) · LangChain · LangGraph · FAISS · OpenRouter API · Pydantic v2 · SQLite (WAL) · jsPDF · Docker

---

## What Makes This Different

| Dimension | Standard Tutorial RAG | DocMind AI Platform |
|---|---|---|
| **Retrieval Strategy** | Pure vector distance (top-k) | **Hybrid BM25 + FAISS Vector** with subject proximity regex boosting |
| **Tokenization** | Naive whitespace / lowercase split | **CamelCase-aware regex tokenizer** + 124 curated stop-words |
| **Multi-Hop Reasoning** | Single-shot query (fails multi-topic) | **LangGraph StateGraph Agent** (Planner → Retriever → Synthesizer → Verifier) |
| **Boundary Handling** | Truncated sentences at chunk edges | **Adjacent Chunk Expansion** (`chunk_index + 1` docstore lookups) |
| **Streaming & Citations**| Naive raw streaming | **Three-phase SSE** with post-hoc citation pruning |
| **Empirical Evaluation** | Unmeasured assertions | **45-Query Benchmark Harness** measuring Recall@K, Precision@K, and MRR |
| **Observability** | Console `print()` statements | **Structured JSON Telemetry** + Zero-hit rate degradation alerts |
| **Fault Tolerance** | Crashing on rate limit | **Gemini → OpenRouter Provider Failover** + 3-attempt exponential backoff |
| **Testing** | 0 to 5 unit tests | **68 Automated Tests** (59 Backend Pytest + 9 Frontend Vitest + CI) |

---

## 📊 Empirical Retrieval Benchmark (900 Chunks, 60 Labeled Queries)

Retrieval performance measured across 6 multi-domain technical documents (**900 chunks**, top-4 retrieval inspects **0.44%** of corpus) using a standardized 60-query evaluation suite (`backend/evaluate_retrieval.py`):

| Retrieval Configuration | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR | nDCG@4 | Mean Rank | Zero-Hit % | Latency |
|---|---|---|---|---|---|---|---|---|---|
| **1. Pure Vector (FAISS only)** | 63.3% | 76.7% | 85.0% | 19.2% | 0.7137 | 0.7288 | 6.88 | 0.0% | 15.65 ms |
| **2. Pure BM25 (Keyword only)** | 70.0% | 75.0% | 75.0% | 18.8% | 0.7381 | 0.7423 | 19.33 | 0.0% | 20.30 ms |
| **3. Naive Hybrid (60/40)** | 76.7% | 88.3% | 96.7% | 22.1% | 0.8354 | 0.8490 | 2.67 | 0.0% | 34.61 ms |
| **4. DocMind Boosted Hybrid** | **88.3%** | **95.0%** | **100.0%** | **23.8%** | **0.9238** | **0.9317** | **1.32** | **0.0%** | **35.80 ms** |

### Key Empirical Findings from Large-Scale Corpus
- **Why Pure Vector Fails Alone (76.7% Recall@4):** Dense semantic embeddings diffuse rare numerical codes, RFC tags, and exact acronyms (e.g. `RFC 8446`, `PCI-DSS 4.0 Req 3.4`), yielding lower precision on technical identifiers.
- **Why Pure BM25 Fails Alone (75.0% Recall@4):** BM25 achieves only **33.3% Recall@4 on synonym/conceptual queries** where user phrasing differs from document vocabulary.
- **Why Hybrid Fusion Wins (88.3% Recall@4):** Linearly weighting Vector (60%) and BM25 (40%) combines semantic generalization with exact keyword matching, boosting Recall@4 by **+11.6%** over vector alone.
- **Why Proximity Regex Boosting is Essential (+0.45):** In a large corpus, technical subjects appear across dozens of operational sections. The proximity regex boost specifically promotes the canonical definition to Rank #1 on definitional queries, lifting Definitional Recall@4 from 86.7% to **100.0%** and improving overall MRR from 0.8354 to **0.9238** (Mean Rank drops from 2.67 to **1.32**).


---

## 🤖 LangGraph Multi-Hop Agent Architecture

For complex, multi-topic, or comparative questions (e.g. *"How does the payment policy differ from the refund policy?"*), DocMind routes execution through a compiled **LangGraph `StateGraph`**:

```
                  ┌──────────────────────┐
                  │ User Multi-Hop Query │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Planner Node      │ ──> Decomposes into 2-3 focused sub-queries
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Retriever Node    │ ──> Hybrid search per sub-query + deduplication
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Synthesizer Node   │ ──> Cross-source synthesis with explicit citations
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Verifier Node     │ ──> Fact verification & hallucination check
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Verified Output    │
                  └──────────────────────┘
```

---

## 🪵 Observability & Structured RAG Telemetry

DocMind integrates centralized structured telemetry (`backend/logger.py`) logging retrieval events in real time:

```json
{
  "event": "rag_retrieval",
  "query_hash": "a4f8c12e",
  "candidates": 15,
  "passed_threshold": 4,
  "top_score": 0.892,
  "vector_ms": 0.84,
  "bm25_ms": 0.41,
  "total_ms": 1.25,
  "boost_applied": "proximity"
}
```

- **Zero-Hit Rate Monitoring:** Emits proactive warnings whenever zero candidate chunks pass the `0.50` relevance threshold (leading indicator of document retrieval degradation).
- **Telemetry Endpoint:** Real-time metrics queryable at `GET /api/telemetry`.

---

## 🚀 Key Features

### 1. Advanced Hybrid Retrieval & Reranking
- **Custom BM25 Engine:** Implementation of Okapi BM25 ($k_1=1.5, b=0.75$) with CamelCase splitting.
- **Adjacent Chunk Expansion:** Concatenates neighboring chunk text (`chunk_index + 1`) to preserve semantic coherence across boundaries.
- **Relevance Gating:** Drops chunks below 50% hybrid score before prompting the LLM.

### 2. Streaming SSE & Citation Pruning
- Three-phase Server-Sent Events stream: Initial metadata → Token deltas → Pruned citations.
- Post-hoc pruning: LLM appends source indices; backend strips unreferenced documents before final event.

### 3. Background Document Intelligence Pipeline
- Automatic entity extraction, executive 5-bullet summary, readability complexity classification, proactive smart alerts, and suggested questions on every PDF upload.

### 4. Enterprise Security & Multi-Tenancy
- JWT authentication with bcrypt password hashing (12 rounds).
- Per-user data isolation: all SQLite queries and FAISS indices partitioned by `user_id`.
- Parameterized SQL queries and zero `innerHTML` usage across the frontend.

---

## 📡 API Surface (17 Endpoints)

| Method | Endpoint | Auth | Purpose |
|---|---|---|---|
| `GET` | `/api/health` | No | Health check and server status |
| `POST` | `/api/auth/register` | No | User registration with email validation |
| `POST` | `/api/auth/login` | No | JWT access token authentication |
| `PUT` | `/api/users/me` | Yes | Update user profile |
| `GET` | `/api/chats/active` | Yes | List active conversation sessions |
| `POST` | `/api/upload` | Yes | Upload and process PDF document |
| `GET` | `/api/documents` | Yes | List authenticated user documents |
| `DELETE` | `/api/documents/{doc_id}` | Yes | Delete document and FAISS vector index |
| `POST` | `/api/chat` | Yes | SSE streaming RAG chat endpoint |
| `GET` | `/api/chat/history/{doc_id}` | Yes | Retrieve conversation history |
| `DELETE` | `/api/chat/history/{doc_id}` | Yes | Clear conversation history |
| `GET` | `/api/analytics/{doc_id}` | Yes | Retrieve background document analytics |
| `POST` | `/api/quiz/{doc_id}` | Yes | Generate 10-question MCQ assessment |
| `POST` | `/api/compare` | Yes | Cross-document comparative analysis |
| `POST` | `/api/agent/query` | Yes | LangGraph multi-hop reasoning query |
| `GET` | `/api/telemetry` | Yes | Real-time RAG operational metrics |

---

## 🐳 Quick Start & Docker Deployment

### Prerequisites
- Docker & Docker Compose (or Python 3.12 + Node.js 20)
- OpenRouter API Key (or Google Gemini API key)

### Running with Docker Compose
```bash
# 1. Clone repository
git clone https://github.com/Surya1207-Gv/-DocMind-AI.git
cd -DocMind-AI

# 2. Setup environment variables
cp backend/.env.example backend/.env
# Edit backend/.env with your OPENROUTER_API_KEY and JWT_SECRET_KEY

# 3. Launch full stack
docker-compose up --build
```
- **Frontend UI:** `http://localhost:3000`
- **Backend API:** `http://localhost:8000`
- **API Docs:** `http://localhost:8000/docs`

---

## 🧪 Testing & Verification

```bash
# Run all 50 backend tests with coverage
python -m pytest backend/tests/ -v --cov=backend

# Run frontend tests (Vitest)
cd frontend && npm run test

# Run the 45-query retrieval benchmark suite
python backend/evaluate_retrieval.py
```

---

## 📄 License
MIT License. Created by [Surya G](https://github.com/Surya1207-Gv).
