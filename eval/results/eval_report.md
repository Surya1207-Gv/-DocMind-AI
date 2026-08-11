# DocMind AI — Empirical Retrieval Benchmark & Evaluation Report

**Generated:** 2026-08-11 07:14:59 UTC  
**Benchmark Scope:** 45 labeled queries across 4 documents (25 ground-truth corpus chunks)

## 1. Primary Configuration Comparison

| Configuration | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR | Zero-Hit % | Latency |
|---|---|---|---|---|---|---|---|
| **Config A: Pure Vector (FAISS only)** | 80.0% | 97.8% | 100.0% | 24.4% | 0.8852 | 8.9% | 1.96 ms |
| **Config B: Pure BM25 (Keyword only)** | 91.1% | 100.0% | 100.0% | 25.0% | 0.9519 | 0.0% | 0.83 ms |
| **Config C: Naive Hybrid (60/40, No Boosts)** | 91.1% | 100.0% | 100.0% | 25.0% | 0.9519 | 0.0% | 0.85 ms |
| **Config D: DocMind Boosted Hybrid (Production)** | 91.1% | 100.0% | 100.0% | 25.0% | 0.9519 | 0.0% | 1.20 ms |

## 2. Recall@4 Breakdown by Query Category

| Category | Pure Vector | Pure BM25 | Naive Hybrid | DocMind Boosted |
|---|---|---|---|---|
| Definitional | 100.0% | 100.0% | 100.0% | 100.0% |
| Keyword/Exact | 100.0% | 100.0% | 100.0% | 100.0% |
| Conceptual | 100.0% | 100.0% | 100.0% | 100.0% |
| Multi-Hop/Section | 90.0% | 100.0% | 100.0% | 100.0% |

## 3. Boost Ablation Analysis

| Ablation Variant | Recall@1 | Recall@4 | MRR | Mean Score |
|---|---|---|---|---|
| Base Hybrid (No Boosts) | 91.1% | 100.0% | 0.9519 | 0.4275 |
| Hybrid + Pattern Boost (+0.05) | 91.1% | 100.0% | 0.9519 | 0.4430 |
| Hybrid + Proximity Regex (+0.45) | 91.1% | 100.0% | 0.9519 | 0.4439 |
| Hybrid + Section Header (+0.10) | 91.1% | 100.0% | 0.9519 | 0.4486 |
| Full DocMind Boost Pipeline | 91.1% | 100.0% | 0.9519 | 0.4739 |

## 4. Relevance Threshold Sensitivity Sweep

| Relevance Cutoff Threshold | Recall@4 | Zero-Hit Rate % | Mean Retrieved Score |
|---|---|---|---|
| **0.30** | 100.0% | 0.0% | 0.4739 |
| **0.35** | 100.0% | 0.0% | 0.4739 |
| **0.40** | 100.0% | 0.0% | 0.4739 |
| **0.45** | 100.0% | 0.0% | 0.4739 |
| **0.50** | 100.0% | 0.0% | 0.4739 |
| **0.55** | 100.0% | 0.0% | 0.4739 |
| **0.60** | 100.0% | 0.0% | 0.4739 |
| **0.65** | 100.0% | 0.0% | 0.4739 |
| **0.70** | 100.0% | 6.7% | 0.4739 |
