import json
import random
from typing import List, Dict, Any
from backend.models import QuizQuestion, QuizResponse
from backend.chat_engine import get_llm_model

def generate_document_quiz(chunks: List[Dict[str, Any]], doc_id: str, count: int = 10) -> List[QuizQuestion]:
    """
    Generates a list of multiple-choice questions from document chunks using Gemini.
    """
    if not chunks:
        return []
        
    # Sort chunks by chunk index/page
    sorted_chunks = sorted(chunks, key=lambda x: (x["metadata"].get("page", 0), x["metadata"].get("chunk_index", 0)))
    
    # Select up to 10 chunks evenly spaced across the document
    step = max(1, len(sorted_chunks) // count)
    selected_chunks = [sorted_chunks[i] for i in range(0, len(sorted_chunks), step)][:count]
    
    context_parts = []
    for c in selected_chunks:
        page = c["metadata"].get("page", 1)
        text = c["text"]
        context_parts.append(f"[PAGE {page}]: {text}")
        
    context_str = "\n\n---\n\n".join(context_parts)
    
    prompt = (
        "You are an educational assessment assistant.\n"
        "Your task is to generate a multiple-choice quiz based on the provided document pages.\n"
        f"Generate exactly {count} multiple-choice questions.\n\n"
        "--- DOCUMENT TEXT ---\n"
        f"{context_str[:12000]}\n"
        "--- END DOCUMENT TEXT ---\n\n"
        "Output a strictly valid JSON list matching the schema below. Do not include comments or explanations outside the JSON structure.\n"
        "[\n"
        "  {\n"
        "    \"id\": 1,\n"
        "    \"question\": \"Question text here?\",\n"
        "    \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
        "    \"correct\": \"Option A\",\n"
        "    \"difficulty\": \"Easy\" | \"Medium\" | \"Hard\",\n"
        "    \"page_ref\": 1\n"
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- Generate questions that are factual and directly answered by the text.\n"
        "- The 'correct' field must exactly match one of the items in the 'options' list.\n"
        "- The 'page_ref' field must be the number of the page where the answer can be verified.\n"
        "- Ensure options are diverse and plausible but with only one clear correct answer.\n"
        "- Return ONLY the raw JSON list (wrapped in ```json ... ``` code blocks is okay, but raw JSON is preferred)."
    )
    
    llm = get_llm_model()
    questions = []
    
    try:
        response = llm.invoke(prompt)
        raw_content = response.content.strip()
        
        # Clean up markdown code block notation if present
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:]
        if raw_content.endswith("```"):
            raw_content = raw_content[:-3]
        raw_content = raw_content.strip()
        
        raw_questions = json.loads(raw_content)
        for i, q in enumerate(raw_questions):
            questions.append(QuizQuestion(
                id=q.get("id", i + 1),
                question=q.get("question", ""),
                options=q.get("options", []),
                correct=q.get("correct", ""),
                difficulty=q.get("difficulty", "Medium"),
                page_ref=q.get("page_ref", 1)
            ))
    except Exception as e:
        print(f"Error generating quiz: {e}")
        # Fall-safe placeholder questions
        questions = [
            QuizQuestion(
                id=1,
                question="What is the primary topic of this document?",
                options=["Options could not be generated.", "An error occurred with the AI API.", "Ensure your API key is correct.", "All of the above"],
                correct="All of the above",
                difficulty="Easy",
                page_ref=1
            )
        ]
        
    return questions
