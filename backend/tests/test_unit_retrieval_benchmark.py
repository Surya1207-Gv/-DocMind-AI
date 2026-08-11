import pytest
from backend.evaluate_retrieval import EvaluationEngine, BENCHMARK_CORPUS, LABELED_QUERIES

def test_evaluation_engine_initialization():
    engine = EvaluationEngine(BENCHMARK_CORPUS)
    assert len(engine.corpus) == 1200
    assert engine.bm25.corpus_size == 1200


def test_evaluation_engine_retrieval_modes():
    engine = EvaluationEngine(BENCHMARK_CORPUS)
    query = "What is Transport Layer Security version 1.3?"
    
    for cfg in ["vector_only", "bm25_only", "naive_hybrid", "full_production"]:
        scores = engine.retrieve(query, config=cfg, top_k=4)
        assert len(scores) == 1200
        retrieved_top4 = [cid for cid, _ in scores[:4]]
        assert "chunk_0001" in retrieved_top4


def test_benchmark_recall_metrics():
    engine = EvaluationEngine(BENCHMARK_CORPUS)
    
    # Run test on 15 definition queries
    def_queries = [q for q in LABELED_QUERIES if q["category"] == "Definitional"]
    hits = 0
    for q in def_queries:
        scores = engine.retrieve(q["query"], config="full_production", top_k=4)
        top4 = [cid for cid, _ in scores[:4]]
        if q["target"] in top4:
            hits += 1
            
    # Proximity boosting achieves 100% recall on definition queries
    assert hits == len(def_queries)


