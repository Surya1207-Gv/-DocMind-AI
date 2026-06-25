import os
import json
import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.config import UPLOAD_DIR, BASE_DIR
from backend.models import (
    ChatRequest, ChatResponse, DocumentInfo, DocumentAnalytics, 
    QuizResponse, CompareRequest, CompareResponse
)
from backend.pdf_processor import process_pdf
from backend.embedding_manager import create_and_save_index, delete_index
from backend.chat_engine import run_chat, run_chat_stream
from backend.analytics_engine import analyze_document
from backend.quiz_engine import generate_document_quiz
from backend.compare_engine import compare_documents

# Authenticated & Database layers
from backend.auth import get_current_user, hash_password, verify_password, create_access_token
import backend.database as db

app = FastAPI(title="DocMind - Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication Request & Response schemas
class UserAuthRequest(BaseModel):
    username: str
    password: str

class UserRegisterRequest(BaseModel):
    username: str
    password: str
    email: str
    full_name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None

# --- Database Migration helper on Startup ---
def migrate_metadata_json():
    # Ensure there is at least one default user "admin" to link existing documents to
    admin = db.get_user_by_username("admin")
    admin_id = "default_admin_id"
    if not admin:
        # Default password is "admin123"
        hashed = hash_password("admin123")
        db.create_user(admin_id, "admin", hashed)
    else:
        admin_id = admin["id"]
        
    meta_path = os.path.join(BASE_DIR, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
            
            # Migrate documents
            docs = data.get("documents", {})
            for doc_id, doc_info in docs.items():
                existing_doc = db.get_document(doc_id, admin_id)
                if not existing_doc:
                    db.add_document(doc_id, admin_id, doc_info.get("name", "Unknown"), doc_info.get("size", 0), doc_info.get("upload_time", ""))
            
            # Migrate analytics
            analytics_data = data.get("analytics", {})
            for doc_id, ana in analytics_data.items():
                existing_ana = db.get_analytics(doc_id)
                if not existing_ana:
                    db.save_analytics(
                        doc_id,
                        ana.get("word_count", 0),
                        ana.get("page_count", 0),
                        ana.get("read_time_mins", 0),
                        ana.get("complexity_score", "Medium"),
                        ana.get("summary", []),
                        ana.get("entities", []),
                        ana.get("alerts", []),
                        ana.get("suggested_questions", [])
                    )
            
            # Migrate quizzes
            quizzes_data = data.get("quizzes", {})
            for doc_id, quiz_qs in quizzes_data.items():
                existing_quiz = db.get_quiz(doc_id)
                if not existing_quiz:
                    db.save_quiz(doc_id, quiz_qs)
            
            # Rename file to metadata_migrated.json to prevent re-running next time
            os.rename(meta_path, os.path.join(BASE_DIR, "metadata_migrated.json"))
            print("[Migration] Successfully migrated metadata.json to SQLite database.")
        except Exception as e:
            print(f"[Migration] Error migrating metadata.json: {e}")

migrate_metadata_json()

# --- Health Check ---
@app.get("/api/health")
def health_check():
    return {"status": "healthy", "time": datetime.utcnow().isoformat()}

@app.get("/")
def read_root():
    return {
        "message": "Welcome to DocMind API.",
        "web_interface": "http://localhost:5173",
        "api_docs": "/docs",
        "health_check": "/api/health"
    }

# --- Auth Routes ---

@app.post("/api/auth/register")
def register_user(request: UserRegisterRequest):
    username = request.username.strip()
    password = request.password.strip()
    email = request.email.strip()
    full_name = request.full_name.strip()
    
    if len(username) < 3 or len(password) < 4:
        raise HTTPException(status_code=400, detail="Username must be >= 3 chars, password >= 4 chars.")
    if len(full_name) < 2:
        raise HTTPException(status_code=400, detail="Full name must be >= 2 characters.")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email address structure.")
    lower_email = email.lower()
    if not (lower_email.endswith("@gmail.com") or lower_email.endswith("@google.com") or lower_email.endswith("@googlemail.com")):
        raise HTTPException(status_code=400, detail="Registration requires a Google email account (@gmail.com or @google.com).")
        
    # Check if username exists
    existing_uname = db.get_user_by_username(username)
    if existing_uname:
        raise HTTPException(status_code=400, detail="Username is already taken.")
        
    # Check if email exists
    with db.get_db_connection() as conn:
        existing_email = conn.execute("SELECT id FROM users WHERE email = ?;", (email,)).fetchone()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email is already registered.")
            
    user_id = str(uuid.uuid4())
    hashed = hash_password(password)
    success = db.create_user(user_id, username, hashed, email, full_name)
    if not success:
        raise HTTPException(status_code=500, detail="Database write failed.")
        
    # Generate token
    token = create_access_token({"sub": user_id})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=username,
        email=email,
        full_name=full_name
    )

@app.post("/api/auth/login")
def login_user(request: UserAuthRequest):
    username = request.username.strip()
    password = request.password.strip()
    
    user = db.get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
        
    token = create_access_token({"sub": user["id"]})
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=username,
        email=user["email"],
        full_name=user["full_name"]
    )

class UserUpdateRequest(BaseModel):
    username: str
    email: str
    full_name: str
    password: str = ""

@app.put("/api/users/me")
def update_current_user(request: UserUpdateRequest, current_user: dict = Depends(get_current_user)):
    username = request.username.strip()
    email = request.email.strip()
    full_name = request.full_name.strip()
    password = request.password.strip()

    if len(username) < 3:
        raise HTTPException(status_code=400, detail="Username must be >= 3 characters.")
    if len(full_name) < 2:
        raise HTTPException(status_code=400, detail="Full name must be >= 2 characters.")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Invalid email address structure.")
    
    lower_email = email.lower()
    if not (lower_email.endswith("@gmail.com") or lower_email.endswith("@google.com") or lower_email.endswith("@googlemail.com")):
        raise HTTPException(status_code=400, detail="Profile requires a Google email account (@gmail.com or @google.com).")
    
    # Check if username exists on another user
    existing_uname = db.get_user_by_username(username)
    if existing_uname and existing_uname["id"] != current_user["id"]:
        raise HTTPException(status_code=400, detail="Username is already taken.")
        
    # Check if email exists on another user
    with db.get_db_connection() as conn:
        existing_email = conn.execute("SELECT id FROM users WHERE email = ? AND id != ?;", (email, current_user["id"])).fetchone()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email is already in use by another user.")
            
    pwd_hash = None
    if password:
        if len(password) < 4:
            raise HTTPException(status_code=400, detail="Password must be >= 4 characters.")
        pwd_hash = hash_password(password)

    success = db.update_user(
        user_id=current_user["id"],
        username=username,
        email=email,
        full_name=full_name,
        password_hash=pwd_hash
    )
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update user profile.")
        
    return {
        "username": username,
        "email": email,
        "full_name": full_name
    }

@app.get("/api/chats/active")
def get_active_chats_list(current_user: dict = Depends(get_current_user)):
    return db.get_active_chats(current_user["id"])

# --- Protected Document Inventory Routes ---

def background_analyze_task(chunks: List[Dict[str, Any]], doc_id: str, filename: str, page_count: int):
    try:
        analytics = analyze_document(chunks, doc_id, filename, page_count)
        db.save_analytics(
            doc_id,
            analytics.word_count,
            analytics.page_count,
            analytics.read_time_mins,
            analytics.complexity_score,
            analytics.summary,
            [e.dict() for e in analytics.entities],
            [a.dict() for a in analytics.alerts],
            analytics.suggested_questions
        )
        print(f"[Background Task] Analytics generated and saved for doc_id: {doc_id}")
    except Exception as e:
        print(f"[Background Task] Error generating analytics for doc_id {doc_id}: {e}")

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    
    # Save PDF locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    file_size = os.path.getsize(file_path)
    
    try:
        # Extract and chunk text
        chunks = process_pdf(file_path, file.filename, doc_id)
        if not chunks:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF. It might be scanned or empty.")
            
        page_count = max([c["metadata"].get("page", 1) for c in chunks])
        
        # Create FAISS Vector Index (using the optimized batching implementation)
        create_and_save_index(chunks, doc_id)
        
        # Compute basic word stats for placeholder analytics
        total_text = " ".join([c["text"] for c in chunks])
        word_count = len(total_text.split())
        read_time_mins = max(1, round(word_count / 200))
        
        # Save to SQLite doc inventory
        upload_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db.add_document(doc_id, current_user["id"], file.filename, file_size, upload_time_str)
        
        # Save placeholder analytics to SQLite first
        placeholder_summary = [
            "Analyzing document content to extract key summaries...",
            "Please wait a moment while details are extracted."
        ]
        placeholder_alerts = [
            {"type": "insight", "content": "Analyzing document content in the background...", "page": 1}
        ]
        db.save_analytics(
            doc_id=doc_id,
            word_count=word_count,
            page_count=page_count,
            read_time_mins=read_time_mins,
            complexity_score="Medium",
            summary=placeholder_summary,
            entities=[],
            alerts=placeholder_alerts,
            suggested_questions=["What is the main topic of this document?"]
        )
        
        # Spawn background task to generate real analytics
        background_tasks.add_task(background_analyze_task, chunks, doc_id, file.filename, page_count)
        
        doc_info = DocumentInfo(
            id=doc_id,
            name=file.filename,
            size=file_size,
            upload_time=upload_time_str
        )
        
        from backend.models import SmartAlert
        analytics = DocumentAnalytics(
            doc_id=doc_id,
            doc_name=file.filename,
            word_count=word_count,
            page_count=page_count,
            read_time_mins=read_time_mins,
            complexity_score="Medium",
            summary=placeholder_summary,
            entities=[],
            alerts=[SmartAlert(type="insight", content="Analyzing document content in the background...", page=1)],
            suggested_questions=["What is the main topic of this document?"]
        )
        
        return {
            "message": "File processed successfully. Analytics will populate in the background.",
            "document": doc_info,
            "analytics": analytics
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/api/documents", response_model=List[DocumentInfo])
def list_documents(current_user: dict = Depends(get_current_user)):
    user_docs = db.list_documents(current_user["id"])
    docs = []
    for d in user_docs:
        file_path = os.path.join(UPLOAD_DIR, f"{d['id']}.pdf")
        if os.path.exists(file_path):
            docs.append(DocumentInfo(**d))
    return docs

@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    # Verify ownership
    doc = db.get_document(doc_id, current_user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # Delete file from disk
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error removing PDF file: {e}")
            
    # Delete FAISS vector index
    try:
        delete_index(doc_id)
    except Exception as e:
        print(f"Error removing FAISS index: {e}")
        
    # Remove from SQLite (will cascade delete analytics, quizzes, chat_messages)
    db.delete_document(doc_id, current_user["id"])
    
    return {"message": "Document deleted successfully"}

# --- Protected Chat & Streaming Routes ---

@app.post("/api/chat")
def chat_document(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    print(f"[API Chat] User: {current_user.get('username')} | Query: '{request.question}' | Docs: {request.doc_ids}")
    # Verify ownership of target documents
    for doc_id in request.doc_ids:
        doc = db.get_document(doc_id, current_user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found or unauthorized.")
            
    try:
        # Return text/event-stream SSE chunk streams
        return StreamingResponse(
            run_chat_stream(request, current_user["id"]),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat processing: {str(e)}")

@app.get("/api/chat/history/{doc_id}")
def get_chat_messages(doc_id: str, current_user: dict = Depends(get_current_user)):
    # Verify document ownership if specific doc ID is given (or "global")
    if doc_id != "global":
        doc = db.get_document(doc_id, current_user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
            
    history = db.get_chat_history(current_user["id"], doc_id)
    return history

@app.delete("/api/chat/history/{doc_id}")
def clear_chat_messages(doc_id: str, current_user: dict = Depends(get_current_user)):
    if doc_id != "global":
        doc = db.get_document(doc_id, current_user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found.")
            
    db.clear_chat_history(current_user["id"], doc_id)
    return {"message": "Chat history cleared successfully"}

# --- Protected Analytics & Assessments ---

@app.get("/api/analytics/{doc_id}", response_model=DocumentAnalytics)
def get_document_analytics(doc_id: str, current_user: dict = Depends(get_current_user)):
    # Verify document ownership
    doc = db.get_document(doc_id, current_user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    analytics = db.get_analytics(doc_id)
    if not analytics:
        raise HTTPException(status_code=404, detail="Analytics not found for this document.")
    analytics["doc_name"] = doc["name"]
    return DocumentAnalytics(**analytics)

@app.post("/api/quiz/{doc_id}", response_model=QuizResponse)
def get_document_quiz(doc_id: str, current_user: dict = Depends(get_current_user)):
    # Verify document ownership
    doc = db.get_document(doc_id, current_user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
        
    # Check if quiz is already cached in database
    cached = db.get_quiz(doc_id)
    if cached:
        return QuizResponse(doc_id=doc_id, questions=cached)
        
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    doc_name = doc["name"]
    
    try:
        chunks = process_pdf(file_path, doc_name, doc_id)
        questions = generate_document_quiz(chunks, doc_id)
        
        # Save to database cache
        db.save_quiz(doc_id, [q.dict() for q in questions])
        
        return QuizResponse(doc_id=doc_id, questions=questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate quiz: {str(e)}")

@app.post("/api/compare", response_model=CompareResponse)
def compare_docs(request: CompareRequest, current_user: dict = Depends(get_current_user)):
    # Verify ownership of target documents
    documents_dict = {}
    for doc_id in request.doc_ids:
        doc = db.get_document(doc_id, current_user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found or unauthorized.")
        documents_dict[doc_id] = doc
            
    try:
        response = compare_documents(request, documents_dict)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing documents: {str(e)}")
