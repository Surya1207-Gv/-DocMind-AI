import pytest
import json
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from backend.agent_engine import (
    build_multi_hop_agent, run_agent_query, run_agent_stream,
    planner_node, retriever_node, synthesizer_node, verifier_node
)

def test_planner_node_decomposition_multi_hop():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='["What is payment policy?", "What is refund policy?"]')
    
    with patch("backend.agent_engine.get_llm_model", return_value=mock_llm):
        state = {"original_query": "Compare payment and refund policies"}
        res = planner_node(state)
        assert len(res["sub_queries"]) == 2
        assert "What is payment policy?" in res["sub_queries"]
        assert "What is refund policy?" in res["sub_queries"]

def test_planner_node_single_query_routing():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='["What is Artificial Intelligence?"]')
    
    with patch("backend.agent_engine.get_llm_model", return_value=mock_llm):
        state = {"original_query": "What is Artificial Intelligence?"}
        res = planner_node(state)
        assert len(res["sub_queries"]) == 1
        assert res["sub_queries"][0] == "What is Artificial Intelligence?"

def test_planner_node_decomposition_output_shape_fallback():
    # When LLM returns non-JSON or invalid schema, verify fallback to original query
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='I am not able to output valid json')
    
    with patch("backend.agent_engine.get_llm_model", return_value=mock_llm):
        state = {"original_query": "Explain Zero Trust Architecture"}
        res = planner_node(state)
        assert isinstance(res["sub_queries"], list)
        assert len(res["sub_queries"]) == 1
        assert res["sub_queries"][0] == "Explain Zero Trust Architecture"

def test_verifier_node_citation_pruning():
    chunks = [
        {"text": "Payment is due in 30 days.", "doc_name": "policy.pdf", "page": 1},
        {"text": "Refunds are processed in 5 days.", "doc_name": "policy.pdf", "page": 2},
        {"text": "Unrelated bonus clause.", "doc_name": "policy.pdf", "page": 3}
    ]
    draft = "Payment is due in 30 days and refunds take 5 days.\nCited Source Indices: 0, 1"
    state = {
        "draft_answer": draft,
        "retrieved_chunks": chunks,
        "confidence": 85,
        "confidence_label": "High"
    }
    
    res = verifier_node(state)
    assert res["verification_status"]["grounded"] is True
    assert res["verification_status"]["cited_sources_count"] == 2
    assert len(res["sources"]) == 2
    assert "Cited Source Indices:" not in res["verified_answer"]

def test_verifier_node_detects_unsupported_claim_and_zeros_confidence():
    chunks = [{"text": "Company history overview.", "doc_name": "about.pdf", "page": 1}]
    refusal_draft = "I cannot find that information in the uploaded documents."
    state = {
        "draft_answer": refusal_draft,
        "retrieved_chunks": chunks,
        "confidence": 75,
        "confidence_label": "Medium"
    }
    
    res = verifier_node(state)
    assert res["verification_status"]["grounded"] is False
    assert res["confidence"] == 0
    assert res["confidence_label"] == "Low"

def test_verifier_node_handles_invalid_out_of_bounds_indices():
    chunks = [{"text": "Single available chunk.", "doc_name": "test.pdf", "page": 1}]
    draft = "Answer referencing missing source.\nCited Source Indices: 99, 105"
    state = {
        "draft_answer": draft,
        "retrieved_chunks": chunks,
        "confidence": 80,
        "confidence_label": "High"
    }
    
    res = verifier_node(state)
    # If all indices are out of bounds, should gracefully fall back to all available retrieved chunks
    assert len(res["sources"]) == 1
    assert "Cited Source Indices:" not in res["verified_answer"]

@patch("backend.agent_engine.search_index")
@patch("backend.agent_engine.get_llm_model")
def test_full_langgraph_agent_execution(mock_get_llm, mock_search):
    from langchain_core.documents import Document
    mock_doc = Document(page_content="Policy clause content", metadata={"page": 1, "doc_id": "d1", "doc_name": "test.pdf"})
    mock_search.return_value = [(mock_doc, 0.4)]
    
    mock_llm_inst = MagicMock()
    mock_llm_inst.invoke.side_effect = [
        AIMessage(content='["sub-query 1", "sub-query 2"]'), # planner
        AIMessage(content="Synthesized policy comparison.\nCited Source Indices: 0") # synthesizer
    ]
    mock_get_llm.return_value = mock_llm_inst
    
    result = run_agent_query("Compare policies", ["d1"], mode="deep")
    
    assert "Synthesized policy comparison" in result["answer"]
    assert len(result["sub_queries"]) == 2
    assert result["verification_status"]["grounded"] is True
    assert len(result["sources"]) == 1

@patch("backend.agent_engine.search_index")
@patch("backend.agent_engine.get_llm_model")
def test_run_agent_stream_generator_events(mock_get_llm, mock_search):
    from langchain_core.documents import Document
    mock_doc = Document(page_content="Chunk content", metadata={"page": 1, "doc_id": "d1", "doc_name": "test.pdf"})
    mock_search.return_value = [(mock_doc, 0.3)]
    
    mock_llm_inst = MagicMock()
    mock_llm_inst.invoke.side_effect = [
        AIMessage(content='["sub-q1", "sub-q2"]'),
        AIMessage(content="Agent stream verified output.\nCited Source Indices: 0")
    ]
    mock_get_llm.return_value = mock_llm_inst
    
    generator = run_agent_stream("Compare topics", ["d1"], mode="deep")
    events = []
    for item in generator:
        if item.startswith("data: "):
            payload = json.loads(item[6:].strip())
            events.append(payload)
            
    event_types = [e["type"] for e in events]
    assert "step" in event_types
    assert "token" in event_types
    assert "metadata" in event_types
    assert "done" in event_types
    
    # Check step sequence
    step_names = [e.get("step") for e in events if e["type"] == "step"]
    assert "planning" in step_names
    assert "retrieving" in step_names
    assert "synthesizing" in step_names
    assert "verifying" in step_names

@patch("backend.agent_engine.search_index")
def test_retriever_node_deduplication(mock_search):
    from langchain_core.documents import Document
    doc1 = Document(page_content="Common chunk", metadata={"page": 1, "doc_id": "d1", "doc_name": "test.pdf"})
    doc2 = Document(page_content="Distinct chunk", metadata={"page": 2, "doc_id": "d1", "doc_name": "test.pdf"})
    # Both sub-queries return doc1, but second also returns doc2
    mock_search.side_effect = [
        [(doc1, 0.3)],
        [(doc1, 0.3), (doc2, 0.4)]
    ]
    
    state = {
        "sub_queries": ["sub1", "sub2"],
        "doc_ids": ["d1"],
        "mode": "deep"
    }
    res = retriever_node(state)
    assert len(res["retrieved_chunks"]) == 2
    contents = [c["text"] for c in res["retrieved_chunks"]]
    assert "Common chunk" in contents
    assert "Distinct chunk" in contents

@patch("backend.agent_engine.get_llm_model")
def test_synthesizer_node_combines_context(mock_get_llm):
    mock_llm_inst = MagicMock()
    mock_llm_inst.invoke.return_value = AIMessage(content="Grounded synthesis.\nCited Source Indices: 0")
    mock_get_llm.return_value = mock_llm_inst
    
    state = {
        "original_query": "What is the policy?",
        "sub_queries": ["What is the policy?"],
        "retrieved_chunks": [
            {"text": "Policy clause 1.", "doc_name": "p.pdf", "page": 1, "relevance": 0.8}
        ],
        "mode": "qa"
    }
    res = synthesizer_node(state)
    assert "draft_answer" in res
    assert "Grounded synthesis" in res["draft_answer"]

def test_verifier_node_empty_draft_fallback():
    state = {
        "draft_answer": "",
        "retrieved_chunks": [],
        "confidence": 0,
        "confidence_label": "Low"
    }
    res = verifier_node(state)
    assert res["verification_status"]["grounded"] is False
    assert res["verified_answer"] == "I could not find sufficient information to answer your question."

