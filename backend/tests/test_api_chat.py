import json
import pytest
from unittest.mock import patch, MagicMock
from backend.models import QuizQuestion
from langchain_core.documents import Document

@pytest.fixture
def mock_document():
    # Insert a document in DB and create local file
    from backend.database import add_document
    from backend.config import UPLOAD_DIR
    import os
    doc_id = "test_doc_id"
    add_document(doc_id, "default_admin_id", "test.pdf", 500, "2026-07-09 22:00:00")
    
    mock_file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(mock_file_path, "w") as f:
        f.write("mock pdf file contents")
        
    return doc_id

@patch("backend.chat_engine.search_index")
def test_chat_success(mock_search, client, auth_headers, mock_document):
    # Setup mock search index output (doc, score)
    mock_doc = Document(
        page_content="Generative AI is a relatively new form of AI that creates new content.",
        metadata={"page": 1, "doc_id": mock_document, "doc_name": "test.pdf", "chunk_index": 0}
    )
    mock_search.return_value = [(mock_doc, 0.2)] # L2 distance of 0.2
    
    chat_payload = {
        "question": "What is Generative AI?",
        "doc_ids": [mock_document],
        "history": [],
        "mode": "qa"
    }
    
    resp = client.post("/api/chat", json=chat_payload, headers=auth_headers)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    
    # Read streaming response chunks
    content = resp.text
    assert "data:" in content
    assert "metadata" in content
    assert "token" in content
    assert "done" in content

@patch("backend.main.generate_document_quiz")
@patch("backend.main.process_pdf")
def test_quiz_generation(mock_process, mock_gen_quiz, client, auth_headers, mock_document):
    # Setup mock quiz questions
    mock_process.return_value = [{"text": "text chunk", "metadata": {"page": 1}}]
    mock_gen_quiz.return_value = [
        QuizQuestion(id=1, question="Test Question?", options=["A", "B", "C", "D"], correct="A", difficulty="Easy", page_ref=1)
    ]
    
    resp = client.post(f"/api/quiz/{mock_document}", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == mock_document
    assert len(data["questions"]) == 1
    assert data["questions"][0]["question"] == "Test Question?"

@patch("backend.main.compare_documents")
def test_compare_success(mock_compare_docs, client, auth_headers, mock_document):
    from backend.models import CompareResponse, DocumentCompareResult
    mock_compare_docs.return_value = CompareResponse(
        comparison_answer="Compare answer text",
        documents=[
            DocumentCompareResult(doc_id=mock_document, doc_name="test.pdf", summary="Summary text")
        ]
    )
    
    compare_payload = {
        "doc_ids": [mock_document],
        "question": "Compare doc content"
    }
    
    resp = client.post("/api/compare", json=compare_payload, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["comparison_answer"] == "Compare answer text"
    assert data["documents"][0]["doc_id"] == mock_document

def test_chat_history_lifecycle(client, auth_headers, mock_document):
    # Verify initially history is empty
    history_resp = client.get(f"/api/chat/history/{mock_document}", headers=auth_headers)
    assert history_resp.status_code == 200
    assert history_resp.json() == []
    
    # Save a chat message directly to DB
    from backend.database import save_chat_message
    save_chat_message("msg_1", "default_admin_id", mock_document, "user", "User question", 0, [], "2026-07-09 22:00:00")
    save_chat_message("msg_2", "default_admin_id", mock_document, "assistant", "Assistant answer", 90, [], "2026-07-09 22:00:05")
    
    # Check history contains message
    history_resp = client.get(f"/api/chat/history/{mock_document}", headers=auth_headers)
    assert len(history_resp.json()) == 2
    assert history_resp.json()[0]["content"] == "User question"
    assert history_resp.json()[1]["content"] == "Assistant answer"
    assert history_resp.json()[1]["confidence_label"] == "High"
    
    # Clear history
    clear_resp = client.delete(f"/api/chat/history/{mock_document}", headers=auth_headers)
    assert clear_resp.status_code == 200
    
    # Verify empty history
    history_resp = client.get(f"/api/chat/history/{mock_document}", headers=auth_headers)
    assert history_resp.json() == []
