import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from backend.embedding_manager import search_index
from backend.chat_engine import run_chat
from backend.models import ChatRequest

@patch("backend.embedding_manager.get_embeddings_model")
@patch("backend.embedding_manager.FAISS.load_local")
@patch("os.path.exists", return_value=True)
def test_search_index_filters_low_relevance(mock_exists, mock_load_local, mock_get_embeddings):
    # Setup mock FAISS vector store
    mock_db = MagicMock()
    mock_doc_low = Document(page_content="Unrelated text", metadata={"doc_id": "doc1", "chunk_index": 0})
    
    # FAISS returns list of tuples: (Document, score)
    # L2 distance > 1.0 means relevance < 50%
    mock_db.similarity_search_with_score.return_value = [(mock_doc_low, 1.2)]
    mock_load_local.return_value = mock_db
    
    # search_index should filter out items below 0.50 hybrid relevance threshold
    results = search_index("Generative AI", ["doc1"])
    assert len(results) == 0

@patch("backend.chat_engine.search_index")
@patch("backend.chat_engine.get_llm_model")
def test_confidence_score_calculation(mock_get_llm, mock_search):
    # Setup mock search results with a good score (L2 distance of 0.4 -> 80% relevance)
    mock_doc = Document(page_content="Factual info", metadata={"page": 1, "doc_id": "d1", "doc_name": "test.pdf"})
    mock_search.return_value = [(mock_doc, 0.4)]
    
    # Mock LLM returns content with cited sources index
    mock_llm_inst = MagicMock()
    from langchain_core.messages import AIMessage
    mock_llm_inst.invoke.return_value = AIMessage(content="Factual info answered.\nCited Source Indices: 0")
    mock_get_llm.return_value = mock_llm_inst
    
    req = ChatRequest(question="What is the fact?", doc_ids=["d1"], history=[], mode="qa")
    resp = run_chat(req)
    
    # Confidence: (1.0 - (0.4 / 2.0)) * 100 = 80%
    assert resp.confidence == 80
    assert resp.confidence_label == "High"
    assert len(resp.sources) == 1

@patch("backend.chat_engine.search_index")
@patch("backend.chat_engine.get_llm_model")
def test_fallback_phrase_rejection(mock_get_llm, mock_search):
    # Search returns valid documents
    mock_doc = Document(page_content="Some context", metadata={"page": 1, "doc_id": "d1", "doc_name": "test.pdf"})
    mock_search.return_value = [(mock_doc, 0.4)]
    
    # LLM returns a fallback response indicating it cannot find the information
    mock_llm_inst = MagicMock()
    from langchain_core.messages import AIMessage
    mock_llm_inst.invoke.return_value = AIMessage(content="I cannot find that information in the provided context.")
    mock_get_llm.return_value = mock_llm_inst
    
    req = ChatRequest(question="Secret question?", doc_ids=["d1"], history=[], mode="qa")
    resp = run_chat(req)
    
    # Response should have 0 confidence and empty sources lists
    assert resp.confidence == 0
    assert resp.confidence_label == "Low"
    assert len(resp.sources) == 0

@patch("backend.chat_engine.get_llm_model")
def test_conversational_bypass_faiss(mock_get_llm):
    # Set mock conversational response
    mock_llm_inst = MagicMock()
    from langchain_core.messages import AIMessage
    mock_llm_inst.invoke.return_value = AIMessage(content="Hello! How can I help you today?")
    mock_get_llm.return_value = mock_llm_inst
    
    req = ChatRequest(question="Hello", doc_ids=["d1"], history=[], mode="qa")
    resp = run_chat(req)
    
    # Bypasses vector DB search, returns conversational reply
    assert resp.answer == "Hello! How can I help you today?"
    assert resp.confidence == 0
    assert len(resp.sources) == 0

def test_out_of_scope_query():
    # Production classification bypasses standard queries, but we can verify run_chat
    # rejects out of scope queries when the classifier labels them as OUT_OF_SCOPE.
    with patch("backend.chat_engine.classify_and_normalize_question") as mock_classify:
        mock_classify.return_value = {"classification": "OUT_OF_SCOPE", "corrected_query": "capital of france"}
        
        req = ChatRequest(question="capital of france", doc_ids=["d1"], history=[], mode="qa")
        resp = run_chat(req)
        
        assert "out of scope" in resp.answer.lower()
        assert resp.confidence == 0
        assert len(resp.sources) == 0
