import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage
from backend.agent_engine import build_multi_hop_agent, run_agent_query, planner_node, verifier_node

def test_planner_node_decomposition():
    # Mock LLM decomposition response
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content='["What is payment policy?", "What is refund policy?"]')
    
    with patch("backend.agent_engine.get_llm_model", return_value=mock_llm):
        state = {"original_query": "Compare payment and refund policies"}
        res = planner_node(state)
        assert len(res["sub_queries"]) == 2
        assert "What is payment policy?" in res["sub_queries"]
        assert "What is refund policy?" in res["sub_queries"]

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

@patch("backend.agent_engine.search_index")
@patch("backend.agent_engine.get_llm_model")
def test_full_langgraph_agent_execution(mock_get_llm, mock_search):
    # Mock search index
    from langchain_core.documents import Document
    mock_doc = Document(page_content="Policy clause content", metadata={"page": 1, "doc_id": "d1", "doc_name": "test.pdf"})
    mock_search.return_value = [(mock_doc, 0.4)]
    
    # Mock LLM calls (planner returns sub-queries, synthesizer returns draft)
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
