import os
import sqlite3
import json
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "docmind.db")

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables if they do not exist."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Enable Foreign Keys
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # 1. Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT UNIQUE,
                full_name TEXT
            );
        """)
        
        # Migration: Add columns to users table if they don't exist
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN email TEXT;")
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT;")
        except sqlite3.OperationalError:
            pass
            
        try:
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users (email);")
        except sqlite3.OperationalError:
            pass
        
        # 2. Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                size INTEGER NOT NULL,
                upload_time TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
        """)
        
        # 3. Analytics Table (caches AI analytics report)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analytics (
                doc_id TEXT PRIMARY KEY,
                word_count INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                read_time_mins INTEGER NOT NULL,
                complexity_score TEXT NOT NULL,
                summary TEXT NOT NULL,          -- JSON string list
                entities TEXT NOT NULL,         -- JSON object list
                alerts TEXT NOT NULL,           -- JSON object list
                suggested_questions TEXT NOT NULL, -- JSON string list
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        
        # 4. Quizzes Table (caches AI generated quizzes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quizzes (
                doc_id TEXT PRIMARY KEY,
                questions TEXT NOT NULL,        -- JSON object list of QuizQuestions
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        
        # 5. Chat History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                role TEXT NOT NULL,             -- 'user' or 'assistant'
                content TEXT NOT NULL,
                confidence INTEGER NOT NULL,
                sources TEXT NOT NULL,          -- JSON object list of SourceChunks
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        
        conn.commit()

# --- Auth Operations ---

def create_user(user_id: str, username: str, password_hash: str, email: str = None, full_name: str = None) -> bool:
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, email, full_name) VALUES (?, ?, ?, ?, ?);",
                (user_id, username, password_hash, email, full_name)
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        return False

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?;", (username,)).fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?;", (user_id,)).fetchone()
        return dict(row) if row else None

# --- Document Operations ---

def add_document(doc_id: str, user_id: str, name: str, size: int, upload_time: str):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO documents (id, user_id, name, size, upload_time) VALUES (?, ?, ?, ?, ?);",
            (doc_id, user_id, name, size, upload_time)
        )
        conn.commit()

def list_documents(user_id: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM documents WHERE user_id = ?;", (user_id,)).fetchall()
        return [dict(r) for r in rows]

def get_document(doc_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?;", (doc_id, user_id)).fetchone()
        return dict(row) if row else None

def delete_document(doc_id: str, user_id: str):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ? AND user_id = ?;", (doc_id, user_id))
        conn.commit()

# --- Analytics Operations ---

def save_analytics(
    doc_id: str, word_count: int, page_count: int, read_time_mins: int, complexity_score: str,
    summary: List[str], entities: List[Dict[str, Any]], alerts: List[Dict[str, Any]], suggested_questions: List[str]
):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analytics (
                doc_id, word_count, page_count, read_time_mins, complexity_score, 
                summary, entities, alerts, suggested_questions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                doc_id, word_count, page_count, read_time_mins, complexity_score,
                json.dumps(summary), json.dumps(entities), json.dumps(alerts), json.dumps(suggested_questions)
            )
        )
        conn.commit()

def get_analytics(doc_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM analytics WHERE doc_id = ?;", (doc_id,)).fetchone()
        if not row:
            return None
        data = dict(row)
        # Parse JSON fields
        data["summary"] = json.loads(data["summary"])
        data["entities"] = json.loads(data["entities"])
        data["alerts"] = json.loads(data["alerts"])
        data["suggested_questions"] = json.loads(data["suggested_questions"])
        return data

# --- Quiz Operations ---

def save_quiz(doc_id: str, questions: List[Dict[str, Any]]):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO quizzes (doc_id, questions) VALUES (?, ?);",
            (doc_id, json.dumps(questions))
        )
        conn.commit()

def get_quiz(doc_id: str) -> Optional[List[Dict[str, Any]]]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT questions FROM quizzes WHERE doc_id = ?;", (doc_id,)).fetchone()
        return json.loads(row["questions"]) if row else None

# --- Chat Messages Operations ---

def save_chat_message(
    msg_id: str, user_id: str, doc_id: str, role: str, content: str, confidence: int, sources: List[Dict[str, Any]], timestamp: str
):
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (id, user_id, doc_id, role, content, confidence, sources, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (msg_id, user_id, doc_id, role, content, confidence, json.dumps(sources), timestamp)
        )
        conn.commit()

def get_chat_history(user_id: str, doc_id: str) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, role, content, confidence, sources, timestamp 
            FROM chat_messages 
            WHERE user_id = ? AND doc_id = ? 
            ORDER BY rowid ASC;
            """,
            (user_id, doc_id)
        ).fetchall()
        
        history = []
        for r in rows:
            msg = dict(r)
            msg["sources"] = json.loads(msg["sources"])
            history.append(msg)
        return history

def clear_chat_history(user_id: str, doc_id: str):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM chat_messages WHERE user_id = ? AND doc_id = ?;", (user_id, doc_id))
        conn.commit()

# Run table initialization on import
init_db()
