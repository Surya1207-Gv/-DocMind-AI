# DocMind AI — Technical Interview Notes & Deep Dive

This document contains detailed engineering rationale and talking points for technical interviews, explaining key architectural decisions, retrieval trade-offs, and empirical findings behind DocMind AI.

---

## 1. The Problem

Large Language Models (LLMs) are fluent and articulate, but inherently ungrounded. When queried about specialized internal documents—such as contracts, compliance guidelines, technical RFCs, or legal policies—they frequently hallucinate confident but factually incorrect answers.

In enterprise and high-stakes document analysis, an unverifiable answer is unacceptable. DocMind AI was built with two strict guarantees:
1. **Auditability:** Every factual claim must cite the specific document and page number it originated from, with the user able to view the underlying passage.
2. **Honesty:** When relevant evidence does not exist in the uploaded documents, the system must refuse to answer rather than guessing or fabricating.

---

## 2. System Architecture

DocMind AI is deployed as **a single container on a single origin**:
- **Backend:** FastAPI (Python 3.12) exposing asynchronous REST and Server-Sent Event (SSE) endpoints.
- **Frontend:** React 19 SPA built with Vite, served directly from FastAPI's static mount in production to eliminate CORS overhead.
- **Storage:** SQLite configured in Write-Ahead Logging (WAL) mode for transactional user metadata, document tracking, and conversation history.
- **Retrieval Engine:** Hybrid architecture combining FAISS dense vector search with an in-memory Okapi BM25 keyword index.
- **Agent Orchestration:** LangGraph `StateGraph` for multi-hop decomposition and cross-document comparative analysis.

---

## 3. Why RAG Rather Than Fine-Tuning?

| Dimension | Retrieval-Augmented Generation (RAG) | Model Fine-Tuning |
|---|---|---|
| **Knowledge Updates** | Instantaneous: add, replace, or delete documents by modifying index files without retraining. | Slow and expensive: requires dataset preparation and continuous fine-tuning runs. |
| **Auditability & Citations** | Deterministic: retrieved chunks are passed in context and directly cited by page number. | Black-box: factual knowledge is baked into parametric weights without source provenance. |
| **Hallucination Prevention** | System prompts and evidence gating can enforce strict grounding in provided context. | Models still confabulate when uncertain. |
| **Access Control** | Per-user index scoping enforces multi-tenant document isolation. | Hard to isolate tenant data baked into shared model weights. |

For document Q&A, auditability and instant updates make RAG the superior architecture.

---

## 4. Why FAISS?

- **Zero Extra Infrastructure:** FAISS runs in-process as a Python C++ extension. There is no separate vector database container to manage, no additional cloud bill, and zero network hop during vector search.
- **Per-Document Index Isolation:** Indices are saved as discrete files named after each document (`FAISS_DIR/{doc_id}`). Deleting a document is an $O(1)$ filesystem delete rather than an expensive partition rebuild.
- **Target Scale:** For single users and demo collections (thousands of chunks), FAISS is sub-millisecond in search execution (~30 ms).
- **Architectural Ceiling:** In an interview, acknowledge when this architecture reaches its limits:
  - When document indices exceed available container memory.
  - When horizontal multi-replica scaling with shared writes is required.
  - *Next step:* Transition to an external vector database like **Qdrant** or **pgvector** with HNSW indexing and metadata filtering.

---

## 5. Chunking Strategy

- **Parameters:** 1,000 characters with 150 characters (15%) overlap using `RecursiveCharacterTextSplitter`.
- **Rationale:**
  - *Too small (<400 chars):* High retrieval vector similarity, but answers lack surrounding semantic context and nuance.
  - *Too large (>2,000 chars):* Embeddings become diluted across multiple topics, decreasing retrieval precision.
  - *Overlap (150 chars):* Prevents key sentences from being split across chunk boundaries.
- **Adjacent Chunk Expansion:** Retrieved top chunks are dynamically expanded with their immediate successor (`chunk_index + 1`). This ensures complete sentences and paragraphs are available to the LLM during generation.

---

## 6. Embedding Strategy

- **Model:** `openai/text-embedding-3-small` (1,536 dimensions) accessed via OpenRouter.
- **Cost vs. Performance:** High semantic fidelity, low token cost, and standardized dimensionality.
- **Ingestion Concurrency:** Chunks are batched (700 chunks per API request) rather than sent individually, reducing ingestion round-trips from minutes to seconds.

---

## 7. Hybrid Retrieval Strategy

### Complementary Failure Modes

Naive retrieval methods fail in predictable, complementary ways:
- **Pure Vector Search Collapse:** Fails on exact keywords, rare acronyms, product SKUs, and specifications (e.g., `RFC 8446`, `PCI-DSS 4.0`). Dense embeddings blur precise character tokens into semantic neighborhoods.
- **Pure BM25 Collapse:** Fails on conceptual paraphrasing and vocabulary mismatches (e.g., user asks about "mitigating fabrications" while the text discusses "hallucination suppression").

### Fusion & Heuristic Boosts

DocMind fuses dense and sparse retrieval scores:

$$\text{Hybrid Score} = 0.6 \cdot \text{Vector Similarity} + 0.4 \cdot \text{Normalized BM25} + \text{Heuristic Boosts}$$

- **Definition Pattern Boost (+0.05):** Applied when query indicates definitional intent (`what is`, `define`, `meaning of`) and candidate contains definitional phrasing (`is defined as`, `refers to`).
- **Proximity Regex Boost (+0.45):** Applied when the query subject appears immediately before a definitional verb in the candidate text, separating actual definitions from passing mentions.
- **Header Match Boost (+0.10):** Applied when uppercase section titles match significant query terms.
- **Second-Stage Reranking:** Lexical reranking shortlists candidates and evaluates term contiguity and query coverage.

### Empirical Validation (1,200-Chunk Benchmark)

On a controlled benchmark of 1,200 chunks across 4 document archetypes with 60 labeled queries:
- **Pure Vector:** Recall@4 = 91.7%, nDCG@4 = 0.8518, Latency = 701 ms
- **Pure BM25:** Recall@4 = 100.0%, nDCG@4 = 0.9486 (collapses on synonym queries to 25%)
- **DocMind Boosted Hybrid:** Recall@4 = 98.3%, nDCG@4 = 0.9095, Score Separation = +0.2827

---

## 8. Prompt Strategy & Grounding

- **Context Injection:** Retrieved chunks are injected as an enumerated context block:
  ```text
  [Source 1] (Document: policy.pdf, Page: 4)
  Passage text...
  ```
- **Constraint Instructions:** The system prompt explicitly commands the LLM to answer *only* using information from the context block, refusing questions whose answers are absent.
- **Source Citation & Pruning:** The model outputs cited source indices (e.g., `[Source 1]`). Post-processing prunes unreferenced sources before streaming the final SSE event to the client.
- **Mode Temperatures:**
  - `qa`: `0.2` (strict adherence)
  - `summary`: `0.3`
  - `deep`: `0.5`
  - `eli5`: `0.6` (creative simplification)

---

## 9. Evaluation Methodology

Rather than relying on qualitative impressions, DocMind measures information retrieval metrics:
- **Recall@k:** Fraction of relevant chunks retrieved within the top $k$ positions.
- **nDCG@k (Normalized Discounted Cumulative Gain):** Evaluates ranking quality, heavily penalizing relevant chunks placed lower in the candidate list.
- **MRR (Mean Reciprocal Rank):** Average reciprocal rank of the first relevant passage.
- **Zero-Hit Rate:** Percentage of queries that return zero results above the relevance threshold (target: 0.0%).
- **Relevance Gate:** A strict 0.50 cutoff score acts as the primary defense against hallucination.

---

## 10. Known Limitations

Be prepared to discuss these real-world constraints candidly:
- **No OCR:** Scanned PDFs without an embedded text layer cannot be parsed; they are rejected with an explicit error instead of creating an empty index.
- **Table Flattening:** Standard PDF text extraction flattens multi-column tables, degrading retrieval precision on matrix data.
- **In-Process FAISS:** Requires sufficient server RAM to hold indices in memory; cannot scale horizontally across multiple instances without a distributed vector store.
- **Embedding Latency:** End-to-end retrieval latency is dominated by the external embedding API call (~1.0–1.5s), not FAISS search (~30ms).
- **Ephemeral Storage on Free Tier:** Free cloud instances wipe local indices on restart; mitigated by boot-time demo seeding.

---

## 11. Deployment Architecture

- **Docker Multi-Stage Build:**
  - Stage 1: Node 20 environment compiles the Vite React SPA to static assets.
  - Stage 2: Python 3.12 slim runtime installs wheels, copies the frontend build into `frontend/dist`, and starts FastAPI with Uvicorn.
- **Environment-Driven Configuration:** Every operational parameter (`TOP_K`, `CHUNK_SIZE`, `RELEVANCE_THRESHOLD`, model identifiers) is configured via environment variables.
- **Demo Seeding:** With `DEMO_SEED=true`, the application indexes a sample PDF on startup, ensuring demo environments are immediately usable.

---

## 12. Discriminative Demo Questions

When demonstrating the platform, use questions that highlight specific pipeline strengths:

1. **Definition Retrieval:**
   - *Query:* "What is Retrieval-Augmented Generation?"
   - *Validates:* Triggers the proximity and definition heuristic boosts.
2. **Exact Numeric Fact:**
   - *Query:* "What is the default chunk size and overlap, and why?"
   - *Validates:* Tests precise token recall and adjacent chunk expansion.
3. **Comparative Analysis:**
   - *Query:* "How does chunking differ from indexing?"
   - *Validates:* LangGraph multi-hop routing and cross-passage synthesis.
4. **Honest Refusal (Critical Test Case):**
   - *Query:* "What was Tesla's 2019 revenue in Norway?"
   - *Validates:* Out-of-corpus query triggers the relevance threshold gate, producing an honest refusal ("Insufficient information in the document") rather than a hallucination.
