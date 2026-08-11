# DocMind AI — Empirical Retrieval Benchmark Report

**Generated:** 2026-08-11 07:25:04 UTC  
**Corpus Size:** **900 chunks** across 6 technical documents (top-4 retrieves **0.44%** of corpus)  
**Evaluation Dataset:** **60 ground-truth labeled queries** across 4 stratified categories  

---

## 1. Primary Configuration Comparison

| Retrieval Configuration | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR | nDCG@4 | Mean Rank | Zero-Hit % | Latency |
|---|---|---|---|---|---|---|---|---|---|
| **Config A: Pure Vector (FAISS only)** | 71.7% | 93.3% | 93.3% | 23.3% | 0.8124 | 0.8413 | 7.72 | 55.0% | 121.65 ms |
| **Config B: Pure BM25 (Keyword only)** | 73.3% | 91.7% | 93.3% | 22.9% | 0.8180 | 0.8391 | 12.50 | 0.0% | 108.89 ms |
| **Config C: Naive Hybrid (60/40, No Boosts)** | 78.3% | 93.3% | 93.3% | 23.3% | 0.8537 | 0.8714 | 7.20 | 0.0% | 126.31 ms |
| **Config D: DocMind Boosted Hybrid (Production)** | 75.0% | 95.0% | 96.7% | 23.8% | 0.8391 | 0.8641 | 6.98 | 0.0% | 98.50 ms |

---

## 2. Recall@4 Breakdown by Query Category

| Query Category | Pure Vector | Pure BM25 | Naive Hybrid (60/40) | DocMind Boosted Hybrid |
|---|---|---|---|---|
| **Definitional** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Keyword/Exact** | 100.0% | 100.0% | 100.0% | **100.0%** |
| **Synonym/Conceptual** | 86.7% | 66.7% | 80.0% | **80.0%** |
| **Multi-Hop/Comparative** | 86.7% | 100.0% | 93.3% | **100.0%** |

---

## 3. Boost Component Ablation Study

| Ablation Variant | Recall@1 | Recall@4 | MRR | nDCG@4 |
|---|---|---|---|---|
| **Base Hybrid (No Boosts)** | 78.3% | 93.3% | 0.8537 | 0.8714 |
| **Hybrid + Pattern Boost (+0.05)** | 78.3% | 93.3% | 0.8537 | 0.8714 |
| **Hybrid + Proximity Regex (+0.45)** | 78.3% | 93.3% | 0.8537 | 0.8714 |
| **Hybrid + Section Header (+0.10)** | 75.0% | 95.0% | 0.8391 | 0.8641 |
| **Full DocMind Boost Pipeline** | 75.0% | 95.0% | 0.8391 | 0.8641 |

---

## 4. Relevance Threshold Sensitivity Sweep

| Relevance Cutoff Threshold | Recall@4 | Zero-Hit Rate % |
|---|---|---|
| **0.30** | 95.0% | 0.0% |
| **0.35** | 95.0% | 0.0% |
| **0.40** | 95.0% | 0.0% |
| **0.45** | 95.0% | 0.0% |
| **0.50** | 95.0% | 0.0% |
| **0.55** | 93.3% | 0.0% |
| **0.60** | 93.3% | 0.0% |
| **0.65** | 88.3% | 5.0% |
| **0.70** | 83.3% | 10.0% |
