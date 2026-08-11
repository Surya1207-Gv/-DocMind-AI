import os
import sys
import json
import time
import math
from typing import List, Dict, Any, Tuple

# Ensure repository root in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.evaluate_retrieval import EvaluationEngine, BENCHMARK_CORPUS, LABELED_QUERIES

REPORT_PATH = os.path.join(BASE_DIR, "eval", "results", "eval_report.md")
DOCS_BENCHMARK_PATH = os.path.join(BASE_DIR, "docs", "retrieval_benchmark.md")
DATASET_PATH = os.path.join(BASE_DIR, "eval", "dataset.json")

def sync_dataset_json():
    dataset_obj = {
        "metadata": {
            "total_corpus_chunks": len(BENCHMARK_CORPUS),
            "total_labeled_queries": len(LABELED_QUERIES),
            "documents_count": 6
        },
        "queries": LABELED_QUERIES
    }
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_obj, f, indent=2)

def run_evaluation_harness():
    sync_dataset_json()
    engine = EvaluationEngine(BENCHMARK_CORPUS)

    configs = [
        ("Config A: Pure Vector (FAISS only)", "vector", {}),
        ("Config B: Pure BM25 (Keyword only)", "bm25", {}),
        ("Config C: Naive Hybrid (60/40, No Boosts)", "naive_hybrid", {}),
        ("Config D: DocMind Boosted Hybrid (Production)", "docmind_boosted", {})
    ]

    results = {}
    print(f"\n{'='*120}")
    print(f"   DOCMIND AI — COMPREHENSIVE RAG BENCHMARK ({len(BENCHMARK_CORPUS)} CHUNKS, {len(LABELED_QUERIES)} LABELED QUERIES)")
    print(f"{'='*120}")

    for label, mode, kwargs in configs:
        hits_1 = 0
        hits_4 = 0
        hits_10 = 0
        prec_4_sum = 0.0
        mrr_sum = 0.0
        ndcg_4_sum = 0.0
        first_rank_sum = 0.0
        latencies = []
        zero_hits = 0

        cat_stats = {
            "Definitional": {"total": 0, "hits_1": 0, "hits_4": 0, "mrr_sum": 0.0},
            "Keyword/Exact": {"total": 0, "hits_1": 0, "hits_4": 0, "mrr_sum": 0.0},
            "Synonym/Conceptual": {"total": 0, "hits_1": 0, "hits_4": 0, "mrr_sum": 0.0},
            "Multi-Hop/Comparative": {"total": 0, "hits_1": 0, "hits_4": 0, "mrr_sum": 0.0}
        }

        for q in LABELED_QUERIES:
            q_text = q["query"]
            target_id = q["target"]
            cat = q["category"]
            cat_stats[cat]["total"] += 1

            t_start = time.perf_counter()
            retrieved_scores = engine.retrieve(q_text, mode=mode, **kwargs)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

            # Check zero-hit at 0.50 threshold
            passed = [cid for cid, s in retrieved_scores if s >= 0.50]
            if not passed:
                zero_hits += 1

            retrieved_ids = [cid for cid, _ in retrieved_scores]

            rank = None
            for idx, cid in enumerate(retrieved_ids, 1):
                if cid == target_id:
                    rank = idx
                    break

            if rank is not None:
                first_rank_sum += rank
                rr = 1.0 / rank
                mrr_sum += rr
                cat_stats[cat]["mrr_sum"] += rr

                if rank == 1:
                    hits_1 += 1
                    cat_stats[cat]["hits_1"] += 1
                if rank <= 4:
                    hits_4 += 1
                    cat_stats[cat]["hits_4"] += 1
                    prec_4_sum += 0.25
                    ndcg_4_sum += 1.0 / math.log2(rank + 1)
                if rank <= 10:
                    hits_10 += 1
            else:
                first_rank_sum += len(retrieved_ids)

        n = len(LABELED_QUERIES)
        results[label] = {
            "recall_1": hits_1 / n,
            "recall_4": hits_4 / n,
            "recall_10": hits_10 / n,
            "prec_4": prec_4_sum / n,
            "mrr": mrr_sum / n,
            "ndcg_4": ndcg_4_sum / n,
            "mean_rank": first_rank_sum / n,
            "zero_hit_rate": (zero_hits / n) * 100.0,
            "latency_ms": sum(latencies) / len(latencies),
            "cat_stats": cat_stats
        }

    # 1. Main comparison table
    print(f"\n{'Retrieval Configuration':<44} | {'Recall@1':<9} | {'Recall@4':<9} | {'Recall@10':<10} | {'Prec@4':<7} | {'MRR':<7} | {'nDCG@4':<7} | {'Mean Rank':<10} | {'Zero-Hit %':<10} | {'Latency'}")
    print("-" * 135)
    for label, res in results.items():
        print(f"{label:<44} | {res['recall_1']*100:>7.1f}% | {res['recall_4']*100:>7.1f}% | {res['recall_10']*100:>8.1f}% | {res['prec_4']*100:>5.1f}% | {res['mrr']:>7.4f} | {res['ndcg_4']:>7.4f} | {res['mean_rank']:>9.2f} | {res['zero_hit_rate']:>9.1f}% | {res['latency_ms']:>5.2f} ms")

    # 2. Category table
    print(f"\n--- Recall@4 by Query Category Across Configurations ---")
    print(f"{'Category':<24} | {'Pure Vector':<12} | {'Pure BM25':<12} | {'Naive Hybrid':<14} | {'DocMind Boosted'}")
    print("-" * 85)
    for cat in ["Definitional", "Keyword/Exact", "Synonym/Conceptual", "Multi-Hop/Comparative"]:
        v_res = results["Config A: Pure Vector (FAISS only)"]["cat_stats"][cat]
        b_res = results["Config B: Pure BM25 (Keyword only)"]["cat_stats"][cat]
        h_res = results["Config C: Naive Hybrid (60/40, No Boosts)"]["cat_stats"][cat]
        d_res = results["Config D: DocMind Boosted Hybrid (Production)"]["cat_stats"][cat]
        v_pct = (v_res["hits_4"] / v_res["total"] * 100) if v_res["total"] else 0.0
        b_pct = (b_res["hits_4"] / b_res["total"] * 100) if b_res["total"] else 0.0
        h_pct = (h_res["hits_4"] / h_res["total"] * 100) if h_res["total"] else 0.0
        d_pct = (d_res["hits_4"] / d_res["total"] * 100) if d_res["total"] else 0.0
        print(f"{cat:<24} | {v_pct:>10.1f}% | {b_pct:>10.1f}% | {h_pct:>12.1f}% | {d_pct:>13.1f}%")

    # 3. Boost Ablation Study
    print(f"\n--- Boost Component Ablation on Hybrid Retrieval ---")
    ablation_configs = [
        ("Base Hybrid (No Boosts)", "naive_hybrid", {}),
        ("Hybrid + Pattern Boost (+0.05)", "ablation", {"boost_proximity": False, "boost_header": False}),
        ("Hybrid + Proximity Regex (+0.45)", "ablation", {"boost_def_pattern": False, "boost_header": False}),
        ("Hybrid + Section Header (+0.10)", "ablation", {"boost_def_pattern": False, "boost_proximity": False}),
        ("Full DocMind Boost Pipeline", "docmind_boosted", {})
    ]
    ablation_results = {}
    print(f"{'Ablation Variant':<38} | {'Recall@1':<9} | {'Recall@4':<9} | {'MRR':<8} | {'nDCG@4':<8}")
    print("-" * 80)
    for name, mode, kwargs in ablation_configs:
        hits_1 = 0
        hits_4 = 0
        mrr_sum = 0.0
        ndcg_4_sum = 0.0
        for q in LABELED_QUERIES:
            retrieved = engine.retrieve(q["query"], mode=mode, **kwargs)
            retrieved_ids = [cid for cid, _ in retrieved]
            target_id = q["target"]
            rank = retrieved_ids.index(target_id) + 1 if target_id in retrieved_ids else 999
            if rank == 1: hits_1 += 1
            if rank <= 4:
                hits_4 += 1
                ndcg_4_sum += 1.0 / math.log2(rank + 1)
            if rank <= len(retrieved_ids):
                mrr_sum += 1.0 / rank
        n = len(LABELED_QUERIES)
        ablation_results[name] = {
            "recall_1": hits_1 / n,
            "recall_4": hits_4 / n,
            "mrr": mrr_sum / n,
            "ndcg_4": ndcg_4_sum / n
        }
        print(f"{name:<38} | {hits_1/n*100:>7.1f}% | {hits_4/n*100:>7.1f}% | {mrr_sum/n:>8.4f} | {ndcg_4_sum/n:>8.4f}")

    # 4. Threshold Sensitivity Sweep
    print(f"\n--- Relevance Threshold Sensitivity Sweep (0.30 - 0.70) ---")
    print(f"{'Threshold':<12} | {'Recall@4':<10} | {'Zero-Hit %':<12}")
    print("-" * 40)
    threshold_sweep = []
    for th in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        zero_count = 0
        hits_4 = 0
        for q in LABELED_QUERIES:
            scores = engine.retrieve(q["query"], mode="docmind_boosted")
            passed = [cid for cid, s in scores if s >= th]
            if not passed:
                zero_count += 1
            if q["target"] in [cid for cid, _ in scores[:4]] and any(s >= th for cid, s in scores[:4] if cid == q["target"]):
                hits_4 += 1
        n = len(LABELED_QUERIES)
        rec4 = hits_4 / n
        zh_pct = (zero_count / n) * 100.0
        threshold_sweep.append((th, rec4, zh_pct))
        print(f"{th:<12.2f} | {rec4*100:>8.1f}% | {zh_pct:>10.1f}%")

    # Generate Markdown Report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(DOCS_BENCHMARK_PATH), exist_ok=True)
    
    report_content = f"""# DocMind AI — Empirical Retrieval Benchmark Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Corpus Size:** **{len(BENCHMARK_CORPUS)} chunks** across 6 technical documents (top-4 retrieves **0.44%** of corpus)  
**Evaluation Dataset:** **{len(LABELED_QUERIES)} ground-truth labeled queries** across 4 stratified categories  

---

## 1. Primary Configuration Comparison

| Retrieval Configuration | Recall@1 | Recall@4 | Recall@10 | Precision@4 | MRR | nDCG@4 | Mean Rank | Zero-Hit % | Latency |
|---|---|---|---|---|---|---|---|---|---|
"""
    for label, res in results.items():
        report_content += f"| **{label}** | {res['recall_1']*100:.1f}% | {res['recall_4']*100:.1f}% | {res['recall_10']*100:.1f}% | {res['prec_4']*100:.1f}% | {res['mrr']:.4f} | {res['ndcg_4']:.4f} | {res['mean_rank']:.2f} | {res['zero_hit_rate']:.1f}% | {res['latency_ms']:.2f} ms |\n"

    report_content += """
---

## 2. Recall@4 Breakdown by Query Category

| Query Category | Pure Vector | Pure BM25 | Naive Hybrid (60/40) | DocMind Boosted Hybrid |
|---|---|---|---|---|
"""
    for cat in ["Definitional", "Keyword/Exact", "Synonym/Conceptual", "Multi-Hop/Comparative"]:
        v_res = results["Config A: Pure Vector (FAISS only)"]["cat_stats"][cat]
        b_res = results["Config B: Pure BM25 (Keyword only)"]["cat_stats"][cat]
        h_res = results["Config C: Naive Hybrid (60/40, No Boosts)"]["cat_stats"][cat]
        d_res = results["Config D: DocMind Boosted Hybrid (Production)"]["cat_stats"][cat]
        v_pct = (v_res["hits_4"] / v_res["total"] * 100) if v_res["total"] else 0.0
        b_pct = (b_res["hits_4"] / b_res["total"] * 100) if b_res["total"] else 0.0
        h_pct = (h_res["hits_4"] / h_res["total"] * 100) if h_res["total"] else 0.0
        d_pct = (d_res["hits_4"] / d_res["total"] * 100) if d_res["total"] else 0.0
        report_content += f"| **{cat}** | {v_pct:.1f}% | {b_pct:.1f}% | {h_pct:.1f}% | **{d_pct:.1f}%** |\n"

    report_content += """
---

## 3. Boost Component Ablation Study

| Ablation Variant | Recall@1 | Recall@4 | MRR | nDCG@4 |
|---|---|---|---|---|
"""
    for name, res in ablation_results.items():
        report_content += f"| **{name}** | {res['recall_1']*100:.1f}% | {res['recall_4']*100:.1f}% | {res['mrr']:.4f} | {res['ndcg_4']:.4f} |\n"

    report_content += """
---

## 4. Relevance Threshold Sensitivity Sweep

| Relevance Cutoff Threshold | Recall@4 | Zero-Hit Rate % |
|---|---|---|
"""
    for th, rec4, zh_pct in threshold_sweep:
        report_content += f"| **{th:.2f}** | {rec4*100:.1f}% | {zh_pct:.1f}% |\n"

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    with open(DOCS_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[Evaluation Complete] Reports saved to:\n  - {REPORT_PATH}\n  - {DOCS_BENCHMARK_PATH}\n")

if __name__ == "__main__":
    run_evaluation_harness()
