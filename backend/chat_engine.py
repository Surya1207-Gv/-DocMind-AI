import os
import json
import re
import time
import uuid
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from backend.logger import get_logger
logger = get_logger(__name__)

# Monkey patch langchain_google_genai to prevent infinite exponential backoff retry loops on quota/rate-limits
try:
    import langchain_google_genai.chat_models
    from tenacity import retry, stop_after_attempt
    def _no_retry_decorator() -> Callable[[Any], Any]:
        return retry(reraise=True, stop=stop_after_attempt(1))
    langchain_google_genai.chat_models._create_retry_decorator = _no_retry_decorator
except Exception as e:
    logger.warning("Error applying langchain_google_genai retry monkey patch: %s", e)

from backend.config import GEMINI_API_KEY, OPENROUTER_API_KEY, LLM_MODEL, TOP_K
from backend.models import ChatRequest, ChatResponse, SourceChunk, ChatMessage
from backend.embedding_manager import search_index
from backend.database import save_chat_message

# System Prompts for Different Modes
SYSTEM_PROMPTS = {
    "qa": (
        "You are a document question-answering assistant. Answer ONLY the user's question using the retrieved document context.\n"
        "Rules:\n"
        "- Keep the answer between 50 and 120 words.\n"
        "- Do not explain unrelated concepts.\n"
        "- Do not add background information unless necessary.\n"
        "- Answer directly.\n"
        "- If the answer isn't in the document, say so.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line.\n"
    ),
    "summary": (
        "You are a document summarizer. Provide an ultra‑concise summary of the retrieved information.\n"
        "Rules:\n"
        "- Limit to 2 sentences maximum.\n"
        "- Capture only the core idea.\n"
        "- Merge related points.\n"
        "- Do not include definitions or quotations unless essential.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line."
    ),
    "deep": (
        "You are an expert AI researcher. Using ONLY the retrieved document, provide a comprehensive, multi-section analysis.\n"
        "Rules:\n"
        "- Structure the answer using proper Markdown headers and bullet points exactly like this:\n"
        "  ## [Section title describing WHY]\n"
        "  ...\n"
        "  ## [Section title describing HOW]\n"
        "  ...\n"
        "  ### [Subsection title describing related concepts / details]\n"
        "  ...\n"
        "  ### Examples\n"
        "  ...\n"
        "- Connect information from multiple retrieved chunks.\n"
        "- Length: 250-500 words.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line.\n"
    ),
    "eli5": (
        "Explain the answer as if speaking to a 10-year-old using ONLY the retrieved document.\n"
        "Rules:\n"
        "- Use everyday language.\n"
        "- Use one analogy.\n"
        "- No technical jargon.\n"
        "- Maximum 150 words.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line.\n"
    )
}

class OpenRouterChat(BaseChatModel):
    model: str
    api_key: str
    temperature: float = 0.2
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        formatted_messages = []
        for msg in messages:
            role = "user"
            if msg.type == "assistant" or msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": msg.content})
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": self.temperature
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

def normalize_gemini_content(content: Any) -> str:
    """
    Normalize Gemini/LangChain chunk content to plain text.
    Gemini may return a string or a list of content blocks.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_text = item.get("text")
                if isinstance(item_text, str):
                    parts.append(item_text)
        return "".join(parts)

    return ""


def get_mode_temperature(mode: str) -> float:
    if mode == "qa":
        return 0.2
    elif mode == "summary":
        return 0.3
    elif mode == "deep":
        return 0.5
    elif mode == "eli5":
        return 0.6
    return 0.2

def get_llm_model(temperature: float = 0.2):
    if GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=GEMINI_API_KEY,
            max_retries=0,
            convert_system_message_to_human=True
        )
    else:
        if not OPENROUTER_API_KEY:
            raise ValueError("Neither GEMINI_API_KEY nor OPENROUTER_API_KEY is configured in your .env file.")
        return OpenRouterChat(
            model=LLM_MODEL,
            api_key=OPENROUTER_API_KEY,
            temperature=temperature
        )

def check_conversational(query: str) -> bool:
    q = query.strip().lower().replace("?", "").replace("!", "").replace(".", "").replace(",", "")
    words = q.split()
    if not words:
        return True
        
    greetings = {
        "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon", "good evening",
        "how are you", "how is it going", "how's it going", "what's up", "yo", "sup", "howdy",
        "who are you", "what are you", "your name", "what is your name", "who built you",
        "thank you", "thanks", "bye", "goodbye", "help", "what can you do", "hi there", "hello there", "hey there",
        "ok", "okay", "got it", "i see", "understood", "okay thanks", "thanks a lot", "thank you so much",
        "never mind", "nevermind", "no thanks", "no thank you", "fine", "cool", "great", "awesome",
        "it is out of the document", "it's out of the document", "out of the document", "not in the document",
        "it is not in the document", "it's not in the document", "that is not in the document",
        "that's not in the document", "not in the doc", "it is not in the doc", "it's not in the doc",
        "out of the doc", "this is out of the document", "it is out of the documents", "it's out of the documents",
        "is it out of the document", "that is out of the document", "that's out of the document",
        "not there", "it is not there", "it's not there", "is it in the document", "is this in the document",
        "it is not there in the document", "it is not in the documents", "that is out of scope", "this is out of scope"
    }
    
    # If the exact text is in greetings
    if q in greetings:
        return True
        
    # If it's a single word greeting/acknowledgement
    if len(words) == 1 and words[0] in {
        "hi", "hello", "hey", "hola", "yo", "howdy", "thanks", "help", 
        "ok", "okay", "fine", "cool", "great", "understood", "awesome", "yes", "no"
    }:
        return True

    # Check for phrases that contain "out of the document", "not in the document", etc.
    phrases = [
        "out of the document", "not in the document", "out of document", "not in document",
        "out of the documents", "not in the documents", "not there in the document",
        "not present in the document", "not present in the documents"
    ]
    if any(phrase in q for phrase in phrases):
        return True
        
    return False

CLASSIFY_PROMPT = """You are an AI assistant classifying a user's question for a Document QA system.
Analyze the user's question and classify it into one of the following categories:
- CONVERSATIONAL: Greetings, thanks, polite conversation, asking about your identity, or short follow-up acknowledgements / corrections (e.g., "hi", "hello", "who are you?", "thank you", "ok I see", "never mind", "It is out of the document").
- TYPO: The query contains typos, spelling mistakes, or poor formatting, but the intent is clear (e.g., "salry and place", "applicaton deadline").
- AMBIGUOUS: The query is extremely vague, short, or incomplete search term, making it unclear what references are requested (e.g., "roles?", "v", "India?", "salary").
- OUT_OF_SCOPE: The query is a general knowledge question, asking to write code, or completely unrelated to analyzing uploaded documents/information (e.g., "what is the capital of France", "how do I cook pasta", "write a python script").
- FACTUAL: The query asks for specific, direct details, facts, or data points (e.g., "what is the company name?", "when is the deadline?").
- SUMMARY: The query asks for an overview, summary, or bullet points of the contents (e.g., "summarize this page", "what are the main takeaways?").
- REASONING: The query asks for explanation of "why", "how", deep analysis, synthesis of ideas, or comparison (e.g., "compare the candidate requirements across documents").

Response format: You must return ONLY a valid JSON object with the following keys:
- "classification": one of the category names above in uppercase.
- "corrected_query": if the category is TYPO, provide the corrected and clean version of the query. If the category is NOT TYPO, set this to the original user question.
- "explanation": a very brief one-sentence reason for this classification.

Do not include any other text, markdown formatting (outside of valid JSON structure), or explanation.

User Question: "{question}"
"""

def classify_and_normalize_question(question: str) -> Dict[str, Any]:
    # 1. Quick conversational check to bypass LLM call and avoid latency/cost
    if check_conversational(question):
        return {
            "classification": "CONVERSATIONAL",
            "corrected_query": question,
            "explanation": "Conversational greeting/outro/acknowledgement."
        }

    # 2. Ambiguity check: if question is extremely short or meaningless
    q_clean = question.strip().lower().strip("?.! ")
    if len(q_clean) <= 2 or q_clean in ["v", "india", "roles", "salary", "home"]:
        return {
            "classification": "AMBIGUOUS",
            "corrected_query": question,
            "explanation": "Extremely short or single-word query is ambiguous."
        }

    # Bypass LLM-based classification call directly to save latency and avoid false OUT_OF_SCOPE / AMBIGUOUS categorizations
    return {
        "classification": "FACTUAL",
        "corrected_query": question,
        "explanation": "Bypassed LLM classification to reduce latency."
    }

# Maximum number of conversation turns to send to the LLM
# Older turns are dropped to prevent token overflow
MAX_HISTORY_TURNS = 5  # 5 user + 5 assistant = 10 messages max


def run_chat_stream(request: ChatRequest, user_id: str):
    """
    Generator yielding Server-Sent Events (SSE) representing token stream and citations metadata.
    Saves message exchanges in SQLite.
    """
    # 1. Classify the question
    cls_data = classify_and_normalize_question(request.question)
    cls_type = cls_data["classification"]
    normalized_q = cls_data["corrected_query"]
    
    sources = []
    confidence = 0
    confidence_label = "Low"
    full_answer = ""
    prefix_note = ""
    retrieval_ms = None      # populated only on the retrieval path
    
    if cls_type == "CONVERSATIONAL":
        system_prompt = (
            "You are DocMind, a friendly and intelligent document analysis assistant.\n"
            "You help users analyze documents, extract summaries, generate quizzes, and compare cross-references.\n"
            "Since the user just greeted you or asked a general conversational question, respond in a friendly, polite, and brief manner.\n"
            "Let them know you are DocMind and are ready to help them analyze the uploaded documents once they select or ask about them."
        )
        messages_list = [SystemMessage(content=system_prompt)]
        # Sliding window: only inject last MAX_HISTORY_TURNS*2 messages to prevent token overflow
        recent_history = (request.history or [])[-(MAX_HISTORY_TURNS * 2):]
        for msg in recent_history:
            messages_list.append(HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content))
        messages_list.append(HumanMessage(content=request.question))
        confidence = 0
        confidence_label = "High"

        # Stream metadata first
        metadata_event = {
            "type": "metadata",
            "confidence": confidence,
            "confidence_label": confidence_label,
            "sources": [],
            "mode": request.mode
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"

    elif cls_type == "OUT_OF_SCOPE":
        # Stream metadata first
        metadata_event = {
            "type": "metadata",
            "confidence": 0,
            "confidence_label": "Low",
            "sources": [],
            "mode": request.mode
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"
        
        answer_text = "I'm sorry, but that query is out of scope for the uploaded documents. Please ask a question related to the documents you have uploaded."
        # Stream tokens
        for token in answer_text.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'text': token + ' '})}\n\n"
        full_answer = answer_text
        
        # Save to DB and close stream
        try:
            user_msg_id = str(uuid.uuid4())
            asst_msg_id = str(uuid.uuid4())
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_id = request.doc_ids[0] if request.doc_ids else "global"
            save_chat_message(user_msg_id, user_id, doc_id, "user", request.question, 0, [], timestamp_str)
            save_chat_message(asst_msg_id, user_id, doc_id, "assistant", full_answer, 0, [], timestamp_str)
        except Exception as db_err:
            logger.error("Error persisting to SQLite: %s", db_err)
            
        yield f"data: {json.dumps({'type': 'metadata', 'confidence': 0, 'confidence_label': 'Low', 'sources': [], 'content': full_answer, 'mode': request.mode})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"
        return

    elif cls_type == "AMBIGUOUS":
        # Stream metadata first
        metadata_event = {
            "type": "metadata",
            "confidence": 0,
            "confidence_label": "Low",
            "sources": [],
            "mode": request.mode
        }
        yield f"data: {json.dumps(metadata_event)}\n\n"
        
        answer_text = "Your question is a bit ambiguous. Could you please specify which document or section you are referring to, or clarify what information you are looking for?"
        # Stream tokens
        for token in answer_text.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'text': token + ' '})}\n\n"
        full_answer = answer_text
        
        # Save to DB and close stream
        try:
            user_msg_id = str(uuid.uuid4())
            asst_msg_id = str(uuid.uuid4())
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            doc_id = request.doc_ids[0] if request.doc_ids else "global"
            save_chat_message(user_msg_id, user_id, doc_id, "user", request.question, 0, [], timestamp_str)
            save_chat_message(asst_msg_id, user_id, doc_id, "assistant", full_answer, 0, [], timestamp_str)
        except Exception as db_err:
            logger.error("Error persisting to SQLite: %s", db_err)
            
        yield f"data: {json.dumps({'type': 'metadata', 'confidence': 0, 'confidence_label': 'Low', 'sources': [], 'content': full_answer, 'mode': request.mode})}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"
        return

    else:
        # Determine dynamic top_k based on mode
        if request.mode == "qa":
            mode_top_k = 3
        elif request.mode == "summary":
            mode_top_k = 5
        elif request.mode == "deep":
            mode_top_k = 9  # 8 to 10 chunks
        else:
            mode_top_k = 5  # default/eli5

        # Search vector DB with BM25 hybrid ranking re-scoring
        _t_retrieval = time.perf_counter()
        search_results = search_index(normalized_q, request.doc_ids, top_k=mode_top_k)
        retrieval_ms = round((time.perf_counter() - _t_retrieval) * 1000.0, 1)
        
        prefix_note = ""
        if cls_type == "TYPO" and normalized_q.strip().lower() != request.question.strip().lower():
            prefix_note = f"*(Interpreted as: \"{normalized_q}\")*\n\n"
            
        if not search_results:
            # Rejection message due to unrelated question
            metadata_event = {
                "type": "metadata",
                "confidence": 0,
                "confidence_label": "Low",
                "sources": [],
                "mode": request.mode
            }
            yield f"data: {json.dumps(metadata_event)}\n\n"
            
            # Send the typo prefix note first if applicable
            if prefix_note:
                yield f"data: {json.dumps({'type': 'token', 'text': prefix_note})}\n\n"
                
            answer_text = "I cannot find any information related to your question in the uploaded documents. Please ask a question directly related to the documents."
            # Stream tokens
            for token in answer_text.split(" "):
                yield f"data: {json.dumps({'type': 'token', 'text': token + ' '})}\n\n"
            full_answer = prefix_note + answer_text
            
            # Save to DB and close stream
            try:
                user_msg_id = str(uuid.uuid4())
                asst_msg_id = str(uuid.uuid4())
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                doc_id = request.doc_ids[0] if request.doc_ids else "global"
                save_chat_message(user_msg_id, user_id, doc_id, "user", request.question, 0, [], timestamp_str)
                save_chat_message(asst_msg_id, user_id, doc_id, "assistant", full_answer, 0, [], timestamp_str)
            except Exception as db_err:
                logger.error("Error persisting to SQLite: %s", db_err)
                
            yield f"data: {json.dumps({'type': 'metadata', 'confidence': 0, 'confidence_label': 'Low', 'sources': [], 'content': full_answer, 'mode': request.mode})}\n\n"
            yield "data: {\"type\": \"done\"}\n\n"
            return
            
        context_parts = []
        raw_distances = []
        
        for idx, (doc, score) in enumerate(search_results):
            raw_distances.append(score)
            relevance = max(0.0, min(1.0, 1.0 - (score / 2.0)))
            
            sources.append(SourceChunk(
                text=doc.page_content,
                page=doc.metadata.get("page", 1),
                doc_id=doc.metadata.get("doc_id", ""),
                doc_name=doc.metadata.get("doc_name", "Unknown Document"),
                relevance=round(relevance * 100, 1)
            ))
            
            context_parts.append(
                f"Source Index: {idx}\n"
                f"Document: {doc.metadata.get('doc_name')} (Page {doc.metadata.get('page')})\n"
                f"Content: {doc.page_content}\n"
                f"---"
            )
            
        context_str = "\n".join(context_parts)
        if raw_distances:
            avg_score = sum(raw_distances) / len(raw_distances)
            confidence = int(max(0, min(100, (1.0 - (avg_score / 2.0)) * 100)))
        else:
            confidence = 0
            
        confidence_label = "High" if confidence >= 80 else ("Medium" if confidence >= 65 else "Low")
        
        system_prompt = SYSTEM_PROMPTS.get(request.mode, SYSTEM_PROMPTS["qa"])
        
        if request.mode == "eli5":
            critical_rule = (
                "CRITICAL RULE: Explain the facts from the CONTEXT above using very simple child-friendly analogies "
                "(such as comparing AI to training a child or a robot). You may use these analogies to make it simple, "
                "but do not introduce outside factual details, statistics, or metrics not in the context. "
                "All core facts must remain strictly grounded in the context provided."
            )
        else:
            critical_rule = (
                "CRITICAL RULE: Answer using ONLY the direct facts explicitly stated in the CONTEXT above. "
                "Do NOT introduce general facts, external descriptions, or general knowledge not present in the CONTEXT. "
                "If the CONTEXT does not contain a specific fact or detail, omit it completely. "
                "Keep your explanation strictly limited to the facts provided."
            )
            
        system_content = (
            f"{system_prompt}\n"
            f"--- CONTEXT ---\n"
            f"{context_str}\n"
            f"--- END OF CONTEXT ---\n"
            f"{critical_rule}"
        )
        messages_list = [SystemMessage(content=system_content)]
        # Sliding window: only inject last MAX_HISTORY_TURNS*2 messages to prevent token overflow
        recent_history = (request.history or [])[-(MAX_HISTORY_TURNS * 2):]
        for msg in recent_history:
            messages_list.append(HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content))
        messages_list.append(HumanMessage(content=normalized_q))

    # stream metadata
    metadata_event = {
        "type": "metadata",
        "confidence": confidence,
        "confidence_label": confidence_label,
        "sources": [s.dict() for s in sources],
        "mode": request.mode,
        # Observability for the UI: what was actually retrieved, how long the
        # retrieval stage took, and which model answered. Never any credentials.
        "retrieved_count": len(sources),
        "retrieval_ms": retrieval_ms,
        "model": LLM_MODEL,
    }
    yield f"data: {json.dumps(metadata_event)}\n\n"
    
    # Send typo note if we corrected a typo
    if prefix_note:
        yield f"data: {json.dumps({'type': 'token', 'text': prefix_note})}\n\n"
        full_answer += prefix_note

    # REST stream
    stream_answer = ""
    if GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=GEMINI_API_KEY,
                max_retries=0,
                convert_system_message_to_human=True
            )
            for chunk in llm.stream(messages_list):
                delta = normalize_gemini_content(chunk.content)
                if delta:
                    stream_answer += delta
                    yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
        except Exception as e:
            logger.exception("Gemini generation failed: %s", e)
            if OPENROUTER_API_KEY:
                try:
                    yield "data: " + json.dumps({'type': 'token', 'text': "*(Gemini generation failed. Falling back to OpenRouter...)*\n\n"}) + "\n\n"
                    formatted_messages = []
                    for msg in messages_list:
                        role = "user"
                        if msg.type == "assistant" or msg.type == "ai":
                            role = "assistant"
                        elif msg.type == "system":
                            role = "system"
                        formatted_messages.append({"role": role, "content": msg.content})

                    headers = {
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    temp = get_mode_temperature(request.mode)
                    payload = {
                        "model": LLM_MODEL,
                        "messages": formatted_messages,
                        "temperature": temp,
                        "stream": True
                    }

                    url = "https://openrouter.ai/api/v1/chat/completions"
                    response = requests.post(url, headers=headers, json=payload, stream=True)
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode("utf-8").strip()
                            if line_str.startswith("data: "):
                                data_content = line_str[6:]
                                if data_content == "[DONE]":
                                    break
                                try:
                                    chunk_json = json.loads(data_content)
                                    delta = chunk_json["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        stream_answer += delta
                                        yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
                                except Exception:
                                    pass
                except Exception as e_fallback:
                    err_msg = f"Error communicating with DocMind: {str(e_fallback)}"
                    yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
                    stream_answer += err_msg
            else:
                err_msg = f"Error communicating with Gemini: {str(e)}"
                yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
                stream_answer += err_msg
    else:
        formatted_messages = []
        for msg in messages_list:
            role = "user"
            if msg.type == "assistant" or msg.type == "ai":
                role = "assistant"
            elif msg.type == "system":
                role = "system"
            formatted_messages.append({"role": role, "content": msg.content})

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        temp = get_mode_temperature(request.mode)
        payload = {
            "model": LLM_MODEL,
            "messages": formatted_messages,
            "temperature": temp,
            "stream": True
        }

        url = "https://openrouter.ai/api/v1/chat/completions"
        
        try:
            response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_content)
                            delta = chunk_json["choices"][0]["delta"].get("content", "")
                            if delta:
                                stream_answer += delta
                                yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
                        except Exception:
                            pass
        except Exception as e:
            err_msg = f"Error communicating with DocMind: {str(e)}"
            yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
            stream_answer += err_msg

    full_answer += stream_answer

    # Extract cited source indices and clean up full_answer
    cited_indices = []
    match = re.search(r"Cited Source Indices:\s*([\d\s,]+)", stream_answer, re.IGNORECASE)
    if match:
        idx_str = match.group(1)
        cited_indices = [int(i.strip()) for i in idx_str.split(",") if i.strip().isdigit()]
    
    # Strip Cited Source Indices from full_answer
    cleaned_stream_answer = re.sub(r"\n*Cited Source Indices:\s*.*", "", stream_answer, flags=re.IGNORECASE).strip()
    full_answer = prefix_note + cleaned_stream_answer
    
    if cited_indices:
        filtered_sources = []
        for idx in cited_indices:
            if 0 <= idx < len(sources):
                filtered_sources.append(sources[idx])
        if filtered_sources:
            sources = filtered_sources
            
    # Always yield an updated metadata event at the end to clean the text and filter sources
    update_metadata_event = {
        "type": "metadata",
        "confidence": confidence,
        "confidence_label": confidence_label,
        "sources": [s.dict() for s in sources],
        "content": full_answer,
        "mode": request.mode
    }
    yield f"data: {json.dumps(update_metadata_event)}\n\n"

    # Verify if fallback phrases are present in the response
    fallback_phrases = [
        "cannot find that information", "cannot find this information", "not find that information",
        "not find this information", "not present in the uploaded documents", "not mentioned in the provided",
        "information is not in the", "not found in the uploaded", "do not contain information",
        "does not contain information", "no information about", "unable to find", "cannot find information",
        "not found in the provided", "not mention this", "not mention that"
    ]
    if any(phrase in cleaned_stream_answer.lower() for phrase in fallback_phrases):
        confidence = 0
        confidence_label = "Low"
        sources = []
        update_metadata_event = {
            "type": "metadata",
            "confidence": 0,
            "confidence_label": "Low",
            "sources": [],
            "content": full_answer,
            "mode": request.mode
        }
        yield f"data: {json.dumps(update_metadata_event)}\n\n"

    # 2. Write User and AI messages to SQLite history
    try:
        user_msg_id = str(uuid.uuid4())
        asst_msg_id = str(uuid.uuid4())
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        doc_id = request.doc_ids[0] if request.doc_ids else "global"
        
        # Save user query
        save_chat_message(
            user_msg_id, user_id, doc_id, "user", request.question, 0, [], timestamp_str
        )
        # Save assistant streaming output
        save_chat_message(
            asst_msg_id, user_id, doc_id, "assistant", full_answer, confidence, [s.dict() for s in sources], timestamp_str
        )
    except Exception as db_err:
        logger.error("Error persisting conversation to SQLite: %s", db_err)

    # 3. Yield done event
    yield "data: {\"type\": \"done\"}\n\n"

