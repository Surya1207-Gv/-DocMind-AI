import os
import shutil
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock environmental keys before imports to prevent errors
os.environ["OPENROUTER_API_KEY"] = "mock_openrouter_key"
os.environ["GEMINI_API_KEY"] = "mock_gemini_key"
os.environ["JWT_SECRET_KEY"] = "mock_secret_key"

# Create temp directories for test assets
TEST_DIR = tempfile.mkdtemp()
TEST_UPLOAD_DIR = os.path.join(TEST_DIR, "uploads")
TEST_FAISS_DIR = os.path.join(TEST_DIR, "faiss_indices")
TEST_DB_FILE = os.path.join(TEST_DIR, "docmind_test.db")

os.makedirs(TEST_UPLOAD_DIR, exist_ok=True)
os.makedirs(TEST_FAISS_DIR, exist_ok=True)

# Redirect all sqlite3 connections for docmind.db to TEST_DB_FILE
import sqlite3
_original_connect = sqlite3.connect
def mock_connect(database, *args, **kwargs):
    if isinstance(database, str) and (database.endswith("docmind.db") or "docmind.db" in database):
        return _original_connect(TEST_DB_FILE, *args, **kwargs)
    return _original_connect(database, *args, **kwargs)
sqlite3.connect = mock_connect

# Patch config and DB file before importing backend components
import backend.config
backend.config.UPLOAD_DIR = TEST_UPLOAD_DIR
backend.config.FAISS_DIR = TEST_FAISS_DIR

import backend.database
backend.database.DB_FILE = TEST_DB_FILE

# Re-init the database schema on the test database file
backend.database.init_db()

# Now import the app and other engines
from backend.main import app
from backend.models import ChatMessage, QuizQuestion

# Reusable mock LLM class
class MockLLM:
    def __init__(self, content="Generative AI is a type of artificial intelligence that can create new content, such as text, images, or code. It is trained on large datasets to learn patterns."):
        self.content = content
        
    def invoke(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessage
        prompt_str = str(messages)
        res = self.content
        
        # If it is a quiz generation prompt, return valid Quiz JSON
        if "generate_document_quiz" in prompt_str or "multiple-choice quiz" in prompt_str:
            res = """[
              {
                "id": 1,
                "question": "What is the primary topic?",
                "options": ["AI", "Food", "History", "Music"],
                "correct": "AI",
                "difficulty": "Easy",
                "page_ref": 1
              }
            ]"""
        elif "compare_documents" in prompt_str or "DOCUMENT TEXTS" in prompt_str:
            res = "Similarities: Both documents discuss AI. Differences: Doc A focuses on models, Doc B on metrics. Conclusion: AI is key."
        elif "Cited Source Indices:" in prompt_str or "Cited Source" in prompt_str or "Index" in prompt_str:
            # Append cited source index for qa/summary mode parsing
            res += "\nCited Source Indices: 0"
            
        return AIMessage(content=res)
        
    def stream(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk
        prompt_str = str(messages)
        res = self.content
        if "Cited Source" in prompt_str or "Index" in prompt_str:
            res += "\nCited Source Indices: 0"
            
        for chunk in res.split(" "):
            yield AIMessageChunk(content=chunk + " ")

class MockEmbeddings:
    def embed_documents(self, texts):
        # Return dummy 1536 dimension vector
        return [[0.1] * 1536 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 1536

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    # Setup mocks globally for LLM and embeddings to prevent network calls
    llm_patcher = patch("backend.chat_engine.get_llm_model", return_value=MockLLM())
    emb_patcher = patch("backend.embedding_manager.get_embeddings_model", return_value=MockEmbeddings())
    
    mock_llm = llm_patcher.start()
    mock_emb = emb_patcher.start()
    
    yield
    
    llm_patcher.stop()
    emb_patcher.stop()
    
    # Cleanup temp directory
    try:
        shutil.rmtree(TEST_DIR)
    except Exception:
        pass

@pytest.fixture(autouse=True)
def clean_db():
    """Wipes the database tables before each test to ensure test isolation."""
    with backend.database.get_db_connection() as conn:
        conn.execute("DELETE FROM chat_messages;")
        conn.execute("DELETE FROM quizzes;")
        conn.execute("DELETE FROM analytics;")
        conn.execute("DELETE FROM documents;")
        conn.execute("DELETE FROM users;")
        conn.commit()
        
    # Ensure there is a default admin user for testing
    from backend.auth import hash_password
    backend.database.create_user(
        user_id="default_admin_id",
        username="admin",
        password_hash=hash_password("admin123"),
        email="admin@gmail.com",
        full_name="Admin User"
    )

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    login_payload = {"username": "admin", "password": "admin123"}
    resp = client.post("/api/auth/login", json=login_payload)
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
