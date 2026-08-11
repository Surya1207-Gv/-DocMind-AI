import sqlite3
import json

def test():
    conn = sqlite3.connect('backend/docmind.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM chat_messages WHERE role = 'assistant';").fetchall()
    for row in rows:
        print("Content Snippet:", row["content"][:60])
        print("Confidence:", row["confidence"])
        try:
            sources = json.loads(row["sources"])
            print("Sources (Page, Relevance):", [(s["page"], s["relevance"]) for s in sources])
        except Exception as e:
            print("Error parsing sources:", e)
        print("-" * 40)

if __name__ == "__main__":
    test()
