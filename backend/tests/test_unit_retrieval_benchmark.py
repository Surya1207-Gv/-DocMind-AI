import pytest
from backend.evaluate_retrieval import EvaluationEngine, BENCHMARK_CORPUS, LABELED_QUERIES

def test_evaluation_engine_initialization():
    engine = EvaluationEngine(BENCHMARK_CORPUS)
    assert len(engine.corpus) == 25
    assert engine.bm25.corpus_size == 25


def test_evaluation_engine_retrieval_modes():
    engine = EvaluationEngine(BENCHMARK_CORPUS)
    query = "What is Artificial Intelligence?"
    
    for mode in ["vector", "bm25", "naive_hybrid", "docmind_boosted"]:
        retrieved = engine.retrieve(query, mode=mode, top_k=3)
        assert len(retrieved) == 3
        # Top-1 or top-3 chunk for AI definition should be c01
        assert "c01" in retrieved

def test_benchmark_recall_metrics():
    engine = EvaluationEngine(BENCHMARK_CORPUS)
    
    # Run test on 10 definition queries
    def_queries = [q for q in LABELED_QUERIES if q["category"] == "Definitional"]
    hits = 0
    for q in def_queries:
        retrieved = engine.retrieve(q["query"], mode="docmind_boosted", top_k=3)
        if q["target"] in retrieved:
            hits += 1
            
    # Should achieve 100% recall on definition queries with proximity boosting
    assert hits == len(def_queries)
