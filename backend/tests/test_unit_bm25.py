import pytest
from backend.embedding_manager import SimpleBM25

def test_bm25_empty_corpus():
    bm25 = SimpleBM25([])
    assert bm25.corpus_size == 0
    assert bm25.get_score("test", 0) == 0.0

def test_bm25_score_matching():
    corpus = [
        "Generative Artificial Intelligence is changing the tech world.",
        "Simple keyword search engines rely heavily on term frequencies.",
        "RAG pipelines retrieve relevant document chunks from databases."
    ]
    bm25 = SimpleBM25(corpus)
    assert bm25.corpus_size == 3
    
    # Score for word "Generative" in doc 0 should be higher than doc 1 or 2
    score_doc0 = bm25.get_score("Generative", 0)
    score_doc1 = bm25.get_score("Generative", 1)
    
    assert score_doc0 > 0.0
    assert score_doc1 == 0.0

def test_bm25_case_insensitivity_and_punctuation():
    corpus = [
        "Hello, World! This is a test."
    ]
    bm25 = SimpleBM25(corpus)
    # Search with lowercase and punctuation removed should still match
    assert bm25.get_score("hello", 0) > 0.0
    assert bm25.get_score("world", 0) > 0.0
    assert bm25.get_score("WORLD!", 0) > 0.0

def test_bm25_stop_words_ignored():
    corpus = [
        "The quick brown fox jumps over a lazy dog."
    ]
    bm25 = SimpleBM25(corpus)
    # Common stop-words like 'the', 'a', 'over' should have score 0.0
    assert bm25.get_score("the", 0) == 0.0
    assert bm25.get_score("over", 0) == 0.0
    assert bm25.get_score("fox", 0) > 0.0

def test_bm25_out_of_bounds_index():
    corpus = ["First doc", "Second doc"]
    bm25 = SimpleBM25(corpus)
    assert bm25.get_score("doc", 5) == 0.0
    assert bm25.get_score("doc", -1) == bm25.get_score("doc", 1)

def test_bm25_camel_case_splitting():
    corpus = [
        "GenerativeAI and ChatGPT are large language models."
    ]
    bm25 = SimpleBM25(corpus)
    # The clean_text_to_words splits camelCase: "GenerativeAI" -> "generative", "ai"
    assert bm25.get_score("generative", 0) > 0.0
    assert bm25.get_score("ai", 0) > 0.0

@pytest.mark.parametrize("query,doc_idx,expected_non_zero", [
    ("Artificial", 0, True),
    ("RAG", 2, True),
    ("models", 1, False),
])
def test_bm25_parameterized(query, doc_idx, expected_non_zero):
    corpus = [
        "Generative Artificial Intelligence is amazing.",
        "Search engines index documents.",
        "A RAG pipeline uses vectors and retrieves chunks."
    ]
    bm25 = SimpleBM25(corpus)
    score = bm25.get_score(query, doc_idx)
    if expected_non_zero:
        assert score > 0.0
    else:
        assert score == 0.0
