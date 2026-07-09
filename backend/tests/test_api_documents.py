import os
import pytest
from unittest.mock import patch, MagicMock

def test_list_documents_empty(client, auth_headers):
    resp = client.get("/api/documents", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []

@patch("backend.main.process_pdf")
@patch("backend.main.create_and_save_index")
def test_upload_document_success(mock_create_index, mock_process_pdf, client, auth_headers):
    # Setup mock chunks
    mock_process_pdf.return_value = [
        {"text": "Chunk 1", "metadata": {"page": 1, "doc_id": "test_id", "doc_name": "test.pdf", "chunk_index": 0}}
    ]
    mock_create_index.return_value = "/path/to/faiss/index"
    
    # Upload a fake PDF file
    file_content = b"%PDF-1.4 mock pdf content"
    files = {"file": ("test.pdf", file_content, "application/pdf")}
    
    resp = client.post("/api/upload", files=files, headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "processed successfully" in data["message"]
    assert data["document"]["name"] == "test.pdf"
    
    # Verify the document is added to the list
    list_resp = client.get("/api/documents", headers=auth_headers)
    assert list_resp.status_code == 200
    docs = list_resp.json()
    assert len(docs) == 1
    assert docs[0]["name"] == "test.pdf"

def test_upload_document_non_pdf(client, auth_headers):
    file_content = b"plain text content"
    files = {"file": ("test.txt", file_content, "text/plain")}
    resp = client.post("/api/upload", files=files, headers=auth_headers)
    assert resp.status_code == 400
    assert "only pdf files are supported" in resp.json()["detail"].lower()

@patch("backend.main.delete_index")
def test_delete_document_success(mock_delete_index, client, auth_headers):
    # First, insert a document directly in SQLite to delete
    from backend.database import add_document
    doc_id = "doc_to_delete"
    # Create the mock PDF file on disk in temp directory to satisfy checks in main.py
    from backend.config import UPLOAD_DIR
    mock_file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(mock_file_path, "w") as f:
        f.write("mock pdf file contents")
        
    add_document(doc_id, "default_admin_id", "test_delete.pdf", 100, "2026-07-09 22:00:00")
    
    # Verify list shows the doc
    list_resp = client.get("/api/documents", headers=auth_headers)
    assert len(list_resp.json()) == 1
    
    # Delete the document
    del_resp = client.delete(f"/api/documents/{doc_id}", headers=auth_headers)
    assert del_resp.status_code == 200
    
    # Verify list is empty
    list_resp = client.get("/api/documents", headers=auth_headers)
    assert len(list_resp.json()) == 0
    assert not os.path.exists(mock_file_path)

def test_delete_document_not_found(client, auth_headers):
    resp = client.delete("/api/documents/non_existent_doc_id", headers=auth_headers)
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

def test_get_analytics_not_found(client, auth_headers):
    resp = client.get("/api/analytics/non_existent_doc_id", headers=auth_headers)
    assert resp.status_code == 404
