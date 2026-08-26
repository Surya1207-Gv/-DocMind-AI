import os
import re
import json
import shutil
import uuid
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel


from backend.config import (
    UPLOAD_DIR,
    BASE_DIR,
    FRONTEND_DIST_DIR,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    DEMO_SEED,
)
from backend.models import (
    ChatRequest, ChatResponse, DocumentInfo, DocumentAnalytics, 
    QuizResponse, CompareRequest, CompareResponse,
    AgentQueryRequest, AgentQueryResponse
)
from backend.pdf_processor import process_pdf
from backend.embedding_manager import create_and_save_index, delete_index
from backend.chat_engine import run_chat_stream
from backend.analytics_engine import analyze_document
from backend.quiz_engine import generate_document_quiz
from backend.compare_engine import compare_documents
from backend.agent_engine import run_agent_query, run_agent_stream

# Authenticated & Database layers
from backend.auth import get_current_user, hash_password, verify_password, create_access_token
import backend.database as db
from backend.logger import get_logger, telemetry
from backend.config import FAISS_DIR

logger = get_logger(__name__)


app = FastAPI(title="DocMind - Backend API")

# Configure environment-driven CORS
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS") or os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_ENV.split(",") if origin.strip()]
if "*" in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response




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
        # This account exists to own documents migrated from the legacy
        # metadata.json. It must never ship with a guessable password: on a
        # public deployment that would be an open door. If ADMIN_PASSWORD is not
        # supplied, the account gets an unguessable random secret and is simply
        # not log-in-able, which is the safe default.
        admin_password = os.getenv("ADMIN_PASSWORD")
        if not admin_password:
            admin_password = secrets.token_urlsafe(32)
            logger.info(
                "[Startup] Created 'admin' record with a random password "
                "(set ADMIN_PASSWORD to enable logging in as admin)."
            )
        db.create_user(admin_id, "admin", hash_password(admin_password))
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
            logger.info("[Migration] Successfully migrated metadata.json to SQLite database.")
        except Exception as e:
            logger.error("[Migration] Error migrating metadata.json: %s", e)

migrate_metadata_json()


# --- Optional demo seeding ---------------------------------------------------
@app.on_event("startup")
def seed_demo_on_startup():
    """
    On free hosting the disk is ephemeral, so a restart leaves the app empty.
    When DEMO_SEED=true, index a bundled sample PDF in a background thread so
    boot is never blocked by embedding API calls (and never fails because of them).
    """
    if not DEMO_SEED:
        return

    import threading

    from backend.demo_seed import seed_demo_document

    threading.Thread(target=seed_demo_document, name="demo-seed", daemon=True).start()


# --- Health Check ---
@app.get("/api/health")
def health_check():
    db_ok = False
    try:
        with db.get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            db_ok = True
    except Exception as e:
        logger.error("Health check DB probe failed: %s", e)

    faiss_writable = os.access(FAISS_DIR, os.W_OK) if os.path.exists(FAISS_DIR) else True
    llm_configured = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY"))

    status_str = "healthy" if (db_ok and faiss_writable and llm_configured) else "degraded"
    return {
        "status": status_str,
        "database": "connected" if db_ok else "disconnected",
        "faiss_indices_writable": faiss_writable,
        "llm_provider": "configured" if llm_configured else "missing_api_key",
        "time": datetime.utcnow().isoformat()
    }


@app.get("/api/info")
def api_info():
    """Machine-readable service descriptor (the SPA is served from '/')."""
    return {
        "name": "DocMind AI",
        "description": "Hybrid-retrieval RAG over your PDFs, with citations.",
        "api_docs": "/docs",
        "health_check": "/api/health",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": TOP_K,
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
    email_pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(email_pattern, email):
        raise HTTPException(status_code=400, detail="Invalid email address structure.")
        
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
        logger.info("[Background Task] Analytics generated and saved for doc_id: %s", doc_id)
    except Exception as e:
        logger.error("[Background Task] Error generating analytics for doc_id %s: %s", doc_id, e)

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    # Magic-byte check: verify binary starts with %PDF
    header = await file.read(4)
    await file.seek(0)
    if header != b"%PDF":
        raise HTTPException(status_code=400, detail="Invalid PDF file format. File must start with '%PDF' header.")
        
    doc_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    
    # Save PDF locally
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
        
    file_size = os.path.getsize(file_path)

    # Reject oversized uploads after the write so we never hold a large body in
    # memory; the partial file is removed immediately.
    if file_size > MAX_UPLOAD_BYTES:
        os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File is {file_size / 1024 / 1024:.1f} MB. Maximum allowed size is {MAX_UPLOAD_MB} MB.",
        )

    if file_size == 0:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

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
            logger.error("Error removing PDF file: %s", e)
            
    # Delete FAISS vector index
    try:
        delete_index(doc_id)
    except Exception as e:
        logger.error("Error removing FAISS index: %s", e)
        
    # Remove from SQLite (will cascade delete analytics, quizzes, chat_messages)
    db.delete_document(doc_id, current_user["id"])
    
    return {"message": "Document deleted successfully"}

# --- Protected Chat & Streaming Routes ---

@app.post("/api/chat")
def chat_document(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    logger.info("[API Chat] User: %s | Query: '%s' | Docs: %s", current_user.get('username'), request.question, request.doc_ids)

    # Reject blank questions before spending a retrieval + LLM round trip.
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

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

# --- LangGraph Agent & Observability Routes ---

@app.post("/api/agent/query", response_model=AgentQueryResponse)
def agent_query_endpoint(request: AgentQueryRequest, current_user: dict = Depends(get_current_user)):
    """
    Executes multi-hop query decomposition, hybrid retrieval, synthesis, and
    fact-verification using the LangGraph state graph.
    """
    logger.info("[API Agent Query] User: %s | Query: '%s' | Docs: %s", current_user.get('username'), request.question, request.doc_ids)

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    for doc_id in request.doc_ids:
        doc = db.get_document(doc_id, current_user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document ID {doc_id} not found or unauthorized.")
            
    try:
        result = run_agent_query(request.question, request.doc_ids, mode=request.mode)
        return AgentQueryResponse(
            answer=result["answer"],
            sub_queries=result["sub_queries"],
            confidence=result["confidence"],
            confidence_label=result["confidence_label"],
            sources=result["sources"],
            verification_status=result["verification_status"]
        )
    except Exception as e:
        logger.error("Error executing LangGraph agent query: %s", e)
        raise HTTPException(status_code=500, detail=f"Error executing agent query: {str(e)}")

@app.post("/api/chat/agent")
async def chat_agent_endpoint(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    """
    Executes LangGraph multi-hop agent reasoning and streams intermediate steps and final answer over SSE.
    """
    target_doc_ids = request.doc_ids if request.doc_ids else []
    logger.info("[API Chat Agent Stream] User: %s | Query: '%s' | Docs: %s", current_user.get('username'), request.question, target_doc_ids)

    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    for d_id in target_doc_ids:
        doc = db.get_document(d_id, current_user["id"])
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {d_id} not found or unauthorized.")
        
    return StreamingResponse(
        run_agent_stream(request.question, target_doc_ids, mode=request.mode),
        media_type="text/event-stream"
    )


@app.get("/api/telemetry")
def get_telemetry_summary(current_user: dict = Depends(get_current_user)):
    """
    Returns real-time RAG operational metrics including zero-hit rates,
    average query latencies, and threshold passage counts.
    """
    return telemetry.get_summary()

@app.get("/api/metrics")
def get_metrics_summary(current_user: dict = Depends(get_current_user)):
    """
    Returns aggregate RAG operational metrics and counters for production observability.
    """
    return telemetry.get_summary()




# ---------------------------------------------------------------------------
# Static SPA hosting
# ---------------------------------------------------------------------------
# Mounted last, after every /api route, so the API always wins the path match.
# This is what lets the whole product ship as ONE service on ONE origin:
# no CORS negotiation, no second deployment, no cross-origin cookie rules.
# If the React build is absent (e.g. backend-only dev), the API still runs fine.
if os.path.isdir(FRONTEND_DIST_DIR):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    _INDEX_HTML = os.path.join(FRONTEND_DIST_DIR, "index.html")

    # Hashed build assets — safe to cache aggressively.
    _assets_dir = os.path.join(FRONTEND_DIST_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        """Serve a real static file when it exists, else fall back to index.html."""
        # Never let the SPA fallback swallow an unmatched API call — that would
        # return HTML to a fetch() expecting JSON and produce a confusing error.
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = os.path.normpath(os.path.join(FRONTEND_DIST_DIR, full_path))
        # Guard against path traversal escaping the build directory.
        if candidate.startswith(os.path.abspath(FRONTEND_DIST_DIR)) and os.path.isfile(candidate):
            return FileResponse(candidate)

        return FileResponse(_INDEX_HTML)

    logger.info("[Startup] Serving React SPA from %s", FRONTEND_DIST_DIR)
else:
    logger.warning(
        "[Startup] No frontend build at %s — running API-only. "
        "Run 'npm run build' in frontend/ to serve the UI from this process.",
        FRONTEND_DIST_DIR,
    )
