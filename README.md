# DocMind AI — Production-Grade RAG Document Intelligence Platform

## Overview

DocMind AI is a complete, enterprise-ready Retrieval-Augmented Generation (RAG) platform engineered from the ground up. It goes far beyond basic "chat with PDF" scripts — implementing hybrid retrieval, agentic document intelligence, streaming LLM responses, multi-tenant JWT auth, and a production-grade React UI.

**Tech Stack:** Python · FastAPI · React (Vite) · LangChain · FAISS · OpenRouter API · Pydantic · SQLite · jsPDF

---

## What Makes This Different

Most "Chat with PDF" projects are simple wrappers: embed → store → retrieve → answer. DocMind AI implements what real enterprise RAG systems use:

| Feature | Basic RAG Demo | DocMind AI |
|---|---|---|
| Vector retrieval | Yes | Yes |
| Hybrid BM25 + Vector reranking | No | Yes |
| Definition-query boosting | No | Yes |
| Subject proximity regex matching | No | Yes |
| Streaming SSE token delivery | No | Yes |
| Citation-level source pruning | No | Yes |
| JWT Auth + User isolation | No | Yes |
| Agentic analytics pipeline | No | Yes |
| Session-aware conversation memory | No | Yes |
| PDF export of chat logs | No | Yes |
| MCQ quiz auto-generation | No | Yes |
| Multi-document cross-comparison | No | Yes |
| Confidence scoring | No | Yes |

---

## Key Engineering Achievements

- **Engineered a production-grade RAG platform** implementing semantic chunking, embedding pipelines, FAISS vector retrieval, and grounded LLM generation with confidence scoring for reliable document intelligence.

- **Architected an autonomous agentic workflow** that performs entity extraction, executive summarization, complexity analysis, proactive alert generation, and suggested-question creation on every document upload.

- **Built a modular 6-engine AI backend** exposed through 8 REST APIs, incorporating Pydantic validation, fault-tolerant processing, rate-limit resilience, and scalable multi-document analysis.

- **Developed multi-document reasoning capabilities**, including cross-document comparison, structured MCQ generation, session-aware conversational memory, and a complete SaaS-ready React platform.

---

## Features

### Advanced RAG Pipeline

- **Hybrid Retrieval**: Combines FAISS vector similarity with a custom BM25 keyword scoring engine for superior recall over pure embedding-based search
- **Intelligent Re-Ranking**: Three-stage scoring: vector similarity (60%) + BM25 normalized (40%) + custom boosts — reranks candidates before LLM generation
- **Definition-Query Boosting**: Detects definition-type queries ("What is X?") and applies:
  - Subject proximity regex matching — boosts chunks where the exact subject noun precedes a definitional verb (is a, refers to, means) within a tightly scoped window
  - Section header matching — boosts chunks containing uppercase headers matching query terms
- **Grounded Generation**: System prompts enforce sentence-level citation — every claim must be supported by a retrieved chunk
- **Citation Pruning**: After generation, the LLM returns cited source indices; backend strips unused sources before reaching the frontend
- **Relevance Threshold Filtering**: Chunks scoring below 50% hybrid relevance are excluded from context

### Agentic Document Intelligence

Auto-runs on every PDF upload — no user action needed:

- **Entity Extraction**: Identifies people, organizations, dates, locations, and key concepts
- **Executive Summarization**: 5-bullet executive summary of key themes
- **Complexity Analysis**: Readability scoring and classification (Beginner / Intermediate / Expert)
- **Proactive Smart Alerts**: Detects deadlines, regulations, and warnings and surfaces them instantly
- **Suggested Questions**: AI proposes the 5 most insightful questions a reader should ask

### Multi-Mode Chat Engine

Four distinct chat personalities with tailored system prompts:

- **Q&A Mode** — Precise, factual, citation-grounded answers
- **Summary Mode** — Concise bullet-point synthesized summaries
- **Deep Analysis Mode** — Step-by-step reasoning, implications, and inference chains
- **ELI5 Mode** — Plain-language explanations using analogies

### Complete Auth System

- JWT-based authentication with bcrypt password hashing
- User registration with email validation
- Per-user document isolation — users only access their own uploads
- Auto-expiry token refresh with 401 interception in the frontend

### Additional Features

- **Confidence Scoring**: FAISS L2 distances converted to 0-100% certainty with animated radial meter UI
- **Auto MCQ Quiz Generator**: 10 questions per document with difficulty ratings, 4 options, and page references
- **Multi-Document Cross-Comparison**: Merged FAISS index search across 2+ documents with AI synthesis
- **Chat Export to PDF**: Full conversation history exported via jsPDF
- **Glassmorphism Dark UI**: Frosted glass panels, animated particle network background, smooth transitions

---

## System Architecture

```
User Query
    |
    v
Intent Classification (Definition / General / Comparative)
    |
    v
FAISS Similarity Search (Top-15 candidates)
    |
    v
BM25 Re-scoring (CamelCase-aware tokenizer, stop-word filtered)
    |
    v
Custom Boost Layer:
  |-- Definition pattern boost (+0.05 if "is a", "refers to", etc.)
  |-- Subject proximity regex (+0.45 if subject directly precedes definitional verb)
  +-- Section header match (+0.10 if uppercase header contains query terms)
    |
    v
Relevance Threshold (drop if hybrid_score < 0.50)
    |
    v
Top-4 chunks --> Context prompt assembly
    |
    v
LLM Generation (Streaming SSE with grounding constraints)
    |
    v
Citation Index Extraction --> Source pruning
    |
    v
Final SSE metadata event with clean response + cited sources only
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 18 + Vite | SPA with SSE streaming support |
| Styling | Vanilla CSS | Glassmorphism dark UI, particle canvas |
| Backend | FastAPI + Uvicorn | 8 REST APIs + SSE streaming |
| Language | Python 3.12 | Backend runtime |
| RAG Core | LangChain | Document chunking, vector store abstraction |
| Embeddings | OpenRouter (text-embedding-3-small) | Semantic vector generation |
| LLM | OpenRouter (nvidia/nemotron-3-nano-30b-a3b:free) | Grounded answer generation |
| Vector DB | FAISS (faiss-cpu) | Per-document similarity search |
| Keyword Search | Custom BM25 (built from scratch) | Hybrid retrieval re-ranking |
| PDF Parsing | PyPDF | Text extraction + page metadata |
| Validation | Pydantic v2 | Request/response schema enforcement |
| Auth | JWT (python-jose) + bcrypt (passlib) | Secure authentication |
| Database | SQLite | User data, chat history, analytics cache |
| PDF Export | jsPDF | Client-side chat export |
| HTTP Client | Axios | Frontend API layer |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- OpenRouter API Key — Get a free key at https://openrouter.ai/

### 1. Clone the Repository

```bash
git clone https://github.com/Surya1207-Gv/DocMind-AI.git
cd DocMind-AI
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create `backend/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Run the Application

**Option A — One-click launch (Windows):**

```bash
start.bat
```

**Option B — Manual:**

```bash
# Terminal 1: Backend
backend\venv\Scripts\python.exe -m uvicorn backend.main:app --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

---

## API Reference

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| POST | /api/auth/register | No | Register a new user |
| POST | /api/auth/login | No | Login and receive JWT |
| POST | /api/upload | Yes | Upload PDF, run full agentic pipeline |
| GET | /api/documents | Yes | List user uploaded documents |
| DELETE | /api/documents/{doc_id} | Yes | Delete document + FAISS index |
| POST | /api/chat | Yes | Stream chat response (SSE) |
| GET | /api/chat/history/{doc_id} | Yes | Fetch chat history |
| DELETE | /api/chat/history/{doc_id} | Yes | Clear chat history |
| GET | /api/analytics/{doc_id} | Yes | Get document analytics |
| POST | /api/quiz/{doc_id} | Yes | Generate / return cached MCQ quiz |
| POST | /api/compare | Yes | Multi-document cross-comparison |

---

## Project Structure

```
DocMind-AI/
|-- backend/
|   |-- main.py               # FastAPI app + all API routes
|   |-- config.py             # Model settings, paths, env loading
|   |-- models.py             # Pydantic request/response schemas
|   |-- auth.py               # JWT token creation + verification
|   |-- database.py           # SQLite CRUD operations
|   |-- embedding_manager.py  # FAISS index + BM25 hybrid reranker
|   |-- chat_engine.py        # LLM chat + SSE streaming + citation pruning
|   |-- pdf_processor.py      # PyPDF extraction + LangChain chunking
|   |-- analytics_engine.py   # Agentic summarization + entity extraction
|   |-- quiz_engine.py        # MCQ question generation
|   |-- compare_engine.py     # Multi-document comparison
|   +-- requirements.txt
|
|-- frontend/
|   +-- src/
|       |-- App.jsx                       # Root app, auth state, routing
|       |-- api.js                        # Axios API client layer
|       +-- components/
|           |-- ChatWindow.jsx            # SSE stream consumer, message rendering
|           |-- Sidebar.jsx               # Document list, upload zone
|           |-- AnalyticsDash.jsx         # Analytics dashboard
|           |-- QuizPanel.jsx             # Interactive MCQ quiz UI
|           |-- CompareMode.jsx           # Multi-doc comparison panel
|           |-- SourceCard.jsx            # Collapsible citation cards
|           |-- ConfidenceMeter.jsx       # Animated radial confidence meter
|           |-- ChatModeSelector.jsx      # Q&A / Summary / Deep / ELI5 toggle
|           |-- ExportButton.jsx          # jsPDF chat export
|           |-- SmartAlerts.jsx           # Proactive alert display
|           |-- MessageBubble.jsx         # Markdown-rendered message
|           |-- ParticleBackground.jsx    # Animated WebGL particle canvas
|           |-- TypingIndicator.jsx       # Streaming loading indicator
|           +-- UploadZone.jsx            # Drag-and-drop PDF upload
|
|-- start.bat                 # Windows one-click launcher
|-- .gitignore
+-- README.md
```

---

## Technical Deep-Dives

### How does the hybrid BM25 + Vector reranker work?

The system retrieves the top-15 FAISS candidates by vector cosine similarity, then independently scores them with a custom BM25 implementation (built from scratch with CamelCase-aware tokenization and stop-word filtering). The final hybrid score:

```
hybrid_score = 0.6 * vector_sim + 0.4 * bm25_normalized + custom_boost
```

For definition queries, a subject proximity regex applies a +0.45 boost to chunks where the exact query subject directly precedes a definitional predicate. This reliably promotes definitional chunks above application/popularity chunks that score higher on pure embedding similarity.

### Why does citation pruning matter?

Without pruning, the system shows all 4 retrieved chunks as citations even if the LLM only used 1 or 2 to answer. After generation, the LLM appends "Cited Source Indices: 0, 2" to its response. The backend extracts these indices, strips the raw citation tag, and replaces the sources array with only actually-cited chunks before streaming the final metadata event.

### How is streaming implemented end-to-end?

The FastAPI endpoint returns a StreamingResponse with media_type="text/event-stream". The backend generator yields SSE events in three phases:

1. **Initial metadata event** — confidence score and pre-retrieved sources
2. **Token events** — one {"type": "token", "text": "..."} per LLM token delta  
3. **Final metadata event** — updated sources (post-citation-pruning) + clean answer

The React frontend reads the SSE stream via fetch() with ReadableStream, accumulating tokens to progressively render the response.

### How does the agentic analytics pipeline work?

On every PDF upload, the backend automatically:
1. Chunks the document with 2000-char chunks + 300-char overlap
2. Passes representative chunks to the LLM with a structured JSON-schema prompt
3. LLM returns a validated Pydantic model: {summary, entities, alerts, suggested_questions}
4. Complexity is computed via word-per-sentence and syllable count heuristics
5. All results are cached in SQLite — subsequent analytics requests are instant

### Why FAISS instead of a cloud vector database?

FAISS operates fully in-memory on local disk, eliminating network latency, cloud costs, and rate limits. DocMind manages one FAISS index directory per document for user isolation. For multi-document queries, indices are merged in-memory using FAISS merge_from() before search. This is trivially migrated to pgvector or Pinecone by swapping the search_index() function.

### How is user data isolated in a multi-tenant system?

Every upload is associated with the authenticated user's ID from the JWT. All queries include WHERE user_id = ?. The /api/documents endpoint only returns documents matching the authenticated caller. Attempting to access another user's document returns a 404.

---

## RAG Optimization Techniques

| Technique | Implementation | Impact |
|---|---|---|
| Hybrid Retrieval | BM25 + Vector with weighted fusion | Higher recall for keyword-heavy queries |
| Intent Classification | Query prefix detection (definition / general) | Higher precision for definitional QA |
| Subject Proximity Matching | Regex: subject + optional parenthetical + definitional verb | Strong definition chunk promotion |
| Section Header Boosting | Uppercase line detection + query word overlap | Better structural document navigation |
| Relevance Threshold | Drop chunks below 50% hybrid score | Less noise in LLM context |
| Citation-level Grounding | LLM returns cited indices, backend prunes unused | Less citation noise and hallucination |
| Per-claim support rule | System prompt: every sentence must have supporting chunk | Lower hallucination rate |
| CamelCase BM25 Tokenizer | Splits GenerativeAI into Generative + AI | Better BM25 term matching accuracy |

---

## License

Licensed under the [MIT License](LICENSE).

---

Built by Surya Sasank — A production-grade RAG system demonstrating semantic retrieval, hybrid reranking, agentic workflows, and grounded LLM generation.
