import os
import uuid
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

def process_pdf(file_path: str, doc_name: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page and splits into chunks with page metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    reader = PdfReader(file_path)
    
    pages_data = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_data.append({
                "text": text,
                "page": page_num + 1
            })
            
    # If no text could be extracted, return empty
    if not pages_data:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = []
    for page_data in pages_data:
        page_text = page_data["text"]
        page_num = page_data["page"]
        
        # Split text on this page
        page_chunks = splitter.split_text(page_text)
        
        for i, chunk_text in enumerate(page_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page_num,
                    "chunk_index": len(chunks)
                }
            })
            
    return chunks
