import os
import pytest
from unittest.mock import patch, MagicMock
from backend.pdf_processor import process_pdf

def test_process_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        process_pdf("non_existent_file.pdf", "test.pdf", "doc_123")

@patch("backend.pdf_processor.PdfReader")
@patch("backend.pdf_processor.os.path.exists", return_value=True)
def test_process_pdf_success(mock_exists, mock_pdf_reader):
    # Setup mock PdfReader with pages
    mock_reader_inst = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "This is the content of page 1."
    
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "This is the content of page 2."
    
    mock_reader_inst.pages = [mock_page1, mock_page2]
    mock_pdf_reader.return_value = mock_reader_inst
    
    chunks = process_pdf("mock_file.pdf", "test.pdf", "doc_123")
    
    # We should have chunks returned
    assert len(chunks) > 0
    # Inspect first chunk
    first_chunk = chunks[0]
    assert "text" in first_chunk
    assert first_chunk["metadata"]["doc_id"] == "doc_123"
    assert first_chunk["metadata"]["doc_name"] == "test.pdf"
    assert first_chunk["metadata"]["page"] == 1
    assert first_chunk["metadata"]["chunk_index"] == 0

@patch("backend.pdf_processor.PdfReader")
@patch("backend.pdf_processor.os.path.exists", return_value=True)
def test_process_pdf_empty(mock_exists, mock_pdf_reader):
    # Setup mock PdfReader with empty pages (scanned/no text)
    mock_reader_inst = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "" # No text extracted
    
    mock_reader_inst.pages = [mock_page]
    mock_pdf_reader.return_value = mock_reader_inst
    
    chunks = process_pdf("mock_file.pdf", "test.pdf", "doc_123")
    assert len(chunks) == 0
