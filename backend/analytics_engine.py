import json
import re
from typing import List, Dict, Any
from backend.models import DocumentAnalytics, EntityInfo, SmartAlert
from backend.chat_engine import get_llm_model

def analyze_document(chunks: List[Dict[str, Any]], doc_id: str, doc_name: str, page_count: int) -> DocumentAnalytics:
    """
    Analyzes document text, extracts metadata, stats, summaries, entities, and smart alerts.
    """
    # 1. Compute basic stats
    total_text = " ".join([c["text"] for c in chunks])
    words = total_text.split()
    word_count = len(words)
    
    # Reading time (average 200 words per minute)
    read_time_mins = max(1, round(word_count / 200))
    
    # 2. Extract sample for LLM analysis (first 3 pages + last page if large, or just general text)
    # This prevents token overflow on huge documents while maintaining quality context.
    sample_texts = []
    
    # Sort chunks by chunk index/page
    sorted_chunks = sorted(chunks, key=lambda x: (x["metadata"].get("page", 0), x["metadata"].get("chunk_index", 0)))
    
    # Take first 5 chunks and last 3 chunks
    if len(sorted_chunks) <= 8:
        sample_texts = [c["text"] for c in sorted_chunks]
    else:
        sample_texts = [c["text"] for c in sorted_chunks[:5]] + ["\n[... CONTIGUOUS CONTENT TRUNCATED FOR SUMMARIZATION ...] \n"] + [c["text"] for c in sorted_chunks[-3:]]
        
    sample_context = "\n\n".join(sample_texts)
    
    # 3. Request Gemini for advanced insights (Summary, Entities, Alerts, Suggested Questions)
    prompt = (
        "You are a professional Document Intelligence System.\n"
        "Your task is to analyze the following document excerpt and generate a detailed report in structured JSON format.\n"
        f"Document Name: {doc_name}\n"
        f"Total Pages: {page_count}\n"
        f"Total Word Count: {word_count}\n\n"
        "--- DOCUMENT EXCERPT ---\n"
        f"{sample_context[:12000]}\n"
        "--- END EXCERPT ---\n\n"
        "Generate a JSON response matching the following schema. Make sure it is strictly valid JSON without comments.\n"
        "{\n"
        "  \"complexity_score\": \"Easy\" | \"Medium\" | \"Hard\",\n"
        "  \"summary\": [\"Point 1\", \"Point 2\", \"Point 3\", \"Point 4\", \"Point 5\"],\n"
        "  \"entities\": [\n"
        "     {\"name\": \"Entity Name\", \"type\": \"Person|Organization|Location|Date|Concept\", \"description\": \"Short context\"}\n"
        "  ],\n"
        "  \"alerts\": [\n"
        "     {\"type\": \"warning|date|stat|insight\", \"content\": \"Alert details\", \"page\": null|number}\n"
        "  ],\n"
        "  \"suggested_questions\": [\"Question 1\", \"Question 2\", \"Question 3\", \"Question 4\", \"Question 5\"]\n"
        "}\n\n"
        "Rules:\n"
        "- Generate exactly 5 relevant summary bullet points.\n"
        "- Extract up to 6 key entities (important names, organizations, concepts or dates).\n"
        "- Create 3-5 smart alerts (e.g., key deadlines/dates as type 'date', warning or critical terms as 'warning', key metrics as 'stat', interesting points as 'insight').\n"
        "- Generate 5 highly relevant, specific questions that a user can ask about this document.\n"
        "- Return ONLY the JSON object. Do not include markdown code block syntax (like ```json) in your raw response, or if you do, ensure it is the ONLY output."
    )
    
    llm = get_llm_model()
    try:
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        
        # Clean up code blocks if LLM outputted them
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        raw_content = raw_content.strip()
        
        data = json.loads(raw_content)
    except Exception as e:
        print(f"Error in document analytics generation: {e}")
        # Fail-safe data
        data = {
            "complexity_score": "Medium",
            "summary": [
                f"Document uploaded: {doc_name}.",
                "Analysis generation experienced a slight hiccup.",
                "You can still ask questions about this document in the chat panel.",
                "Check the details of this document page by page in the main viewer.",
                "Try refreshing or re-uploading if you continue to see this template summary."
            ],
            "entities": [
                {"name": doc_name, "type": "Document", "description": "The uploaded PDF document file"}
            ],
            "alerts": [
                {"type": "insight", "content": "Document loaded successfully. Ask a question to begin.", "page": 1}
            ],
            "suggested_questions": [
                "What is the main topic of this document?",
                "Can you summarize the main findings?",
                "Are there any key dates or deadlines mentioned?",
                "Who are the main parties or organizations in this document?",
                "What are the major conclusions or action items?"
            ]
        }
        
    # Defensive parsing for Entities
    entities = []
    for ent in data.get("entities", []):
        if not isinstance(ent, dict):
            continue
        entities.append(EntityInfo(
            name=str(ent.get("name", "Unknown")),
            type=str(ent.get("type", "Concept")),
            description=str(ent.get("description", ""))
        ))
        
    # Defensive parsing for Alerts
    alerts = []
    for al in data.get("alerts", []):
        if not isinstance(al, dict):
            continue
            
        # Find alert text content defensively
        content = al.get("content") or al.get("text") or al.get("description") or al.get("message")
        if not content:
            # Pick first available string key that isn't metadata
            str_keys = [k for k, v in al.items() if k not in ["type", "page"] and isinstance(v, str)]
            if str_keys:
                content = al[str_keys[0]]
            else:
                content = "Key metric highlighted in document."
                
        # Parse page reference safely
        page = al.get("page")
        try:
            page = int(page) if page is not None else None
        except (ValueError, TypeError):
            page = None
            
        alerts.append(SmartAlert(
            type=str(al.get("type", "insight")),
            content=str(content),
            page=page
        ))
    
    return DocumentAnalytics(
        doc_id=doc_id,
        doc_name=doc_name,
        word_count=word_count,
        page_count=page_count,
        read_time_mins=read_time_mins,
        complexity_score=data.get("complexity_score", "Medium"),
        summary=data.get("summary", []),
        entities=entities,
        alerts=alerts,
        suggested_questions=data.get("suggested_questions", [])
    )
