import os
from typing import List, Dict, Any, Tuple, Optional
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from backend.config import OPENROUTER_API_KEY, LLM_MODEL, TOP_K
from backend.models import ChatRequest, ChatResponse, SourceChunk, ChatMessage
from backend.embedding_manager import search_index

# System Prompts for Different Modes
SYSTEM_PROMPTS = {
    "qa": (
        "You are an expert document assistant. Answer the user's question accurately based ONLY on the provided document contexts.\n"
        "CRITICAL: Do NOT make up facts or introduce any outside/external knowledge. Relate your answer strictly and directly to the facts, definitions, and wording found in the CONTEXT.\n"
        "If the answer cannot be found in the context, say 'I cannot find that information in the uploaded documents.'\n"
        "Keep your answer clear, informative, direct, and factual. Do not refer to 'the context' or 'the document' directly unless necessary.\n"
        "TECHNICAL PRECISION:\n"
        "1. Do not equate Generative AI with foundation models. State clearly that foundation models are an example of generative AI models, trained on a broad data set and applicable to a wide range of tasks.\n"
        "2. Focus strictly on defining the concept and do not append detailed lists of specific applications (such as website building, video editing, or venture capital investment figures) unless the user explicitly asks for applications.\n"
        "3. Every sentence in your response must be supported by at least one retrieved chunk. If no chunk supports a statement, omit it completely.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line.\n"
    ),
    "summary": (
        "You are an expert summarizer. Answer the user's question by summarizing the key points based ONLY on the provided document contexts.\n"
        "CRITICAL: Rely strictly on facts directly mentioned in the CONTEXT. Do not use outside/external knowledge.\n"
        "If the answer cannot be found in the context, say 'I cannot find that information in the uploaded documents.'\n"
        "Present your response using clean bullet points and bold headers. Focus on high-level takeaways.\n"
        "TECHNICAL PRECISION:\n"
        "1. Do not equate Generative AI with foundation models. State clearly that foundation models are an example of generative AI models, trained on a broad data set and applicable to a wide range of tasks.\n"
        "2. Focus strictly on defining the concept and do not append detailed lists of specific applications (such as website building, video editing, or venture capital investment figures) unless the user explicitly asks for applications.\n"
        "3. Every sentence in your response must be supported by at least one retrieved chunk. If no chunk supports a statement, omit it completely.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line.\n"
    ),
    "deep": (
        "You are an advanced analyst. Provide a deep, step-by-step analytical answer based ONLY on the provided document contexts.\n"
        "CRITICAL: Rely strictly on facts directly mentioned in the CONTEXT. Do not use outside/external knowledge.\n"
        "If the answer cannot be found in the context, say 'I cannot find that information in the uploaded documents.'\n"
        "Explain the reasoning, cite sections, and explore implications of the facts found in the documents.\n"
        "TECHNICAL PRECISION:\n"
        "1. Do not equate Generative AI with foundation models. State clearly that foundation models are an example of generative AI models, trained on a broad data set and applicable to a wide range of tasks.\n"
        "2. Focus strictly on defining the concept and do not append detailed lists of specific applications (such as website building, video editing, or venture capital investment figures) unless the user explicitly asks for applications.\n"
        "3. Every sentence in your response must be supported by at least one retrieved chunk. If no chunk supports a statement, omit it completely.\n"
        "At the very end of your response, on a new line, write 'Cited Source Indices: ' followed by a comma-separated list of the Source Index numbers you actually used in your answer (e.g. Cited Source Indices: 1, 2). Do not include any other text on this line.\n"
    ),
    "eli5": (
        "You are a helpful teacher. Explain the answer to the user's question like they are 5 years old, based ONLY on the provided document contexts.\n"
        "CRITICAL: Rely strictly on facts directly mentioned in the CONTEXT. Do not use outside/external knowledge.\n"
        "If the answer cannot be found in the context, say 'I cannot find that information in the uploaded documents.'\n"
        "Use simple terms, plain language, and clear analogies. Break down complex jargon into basic ideas.\n"
        "TECHNICAL PRECISION:\n"
        "1. Do not equate Generative AI with foundation models. State clearly that foundation models are an example of generative AI models, trained on a broad data set and applicable to a wide range of tasks.\n"
        "2. Focus strictly on defining the concept and do not append detailed lists of specific applications (such as website building, video editing, or venture capital investment figures) unless the user explicitly asks for applications.\n"
        "3. Every sentence in your response must be supported by at least one retrieved chunk. If no chunk supports a statement, omit it completely.\n"
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
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

def get_llm_model():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it in your .env file.")
    return OpenRouterChat(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        temperature=0.2
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
        "thank you", "thanks", "bye", "goodbye", "help", "what can you do", "hi there", "hello there", "hey there"
    }
    
    # If the exact text is in greetings
    if q in greetings:
        return True
        
    # If it's a single word greeting
    if len(words) == 1 and words[0] in {"hi", "hello", "hey", "hola", "yo", "howdy", "thanks", "help"}:
        return True
        
    return False

import json
import uuid
from datetime import datetime
from backend.database import save_chat_message

def run_chat(request: ChatRequest) -> ChatResponse:
    # Maintain synchronous run_chat for fallback compatibility
    if check_conversational(request.question):
        sources = []
        confidence = 0
        system_prompt = (
            "You are DocMind AI, a friendly and intelligent document analysis assistant.\n"
            "You help users analyze documents, extract summaries, generate quizzes, and compare cross-references.\n"
            "Since the user just greeted you or asked a general conversational question, respond in a friendly, polite, and brief manner.\n"
            "Let them know you are DocMind AI and are ready to help them analyze the uploaded documents once they select or ask about them."
        )
        messages = [SystemMessage(content=system_prompt)]
        if request.history:
            for msg in request.history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))
        messages.append(HumanMessage(content=request.question))
        
        llm = get_llm_model()
        try:
            response = llm.invoke(messages)
            answer = response.content
        except Exception as e:
            answer = f"Error communicating with DocMind AI: {str(e)}"
            
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            sources=sources,
            mode=request.mode
        )

    search_results = search_index(request.question, request.doc_ids, top_k=TOP_K)
    context_parts = []
    sources = []
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
        
    system_prompt = SYSTEM_PROMPTS.get(request.mode, SYSTEM_PROMPTS["qa"])
    system_content = (
        f"{system_prompt}\n"
        f"--- CONTEXT ---\n"
        f"{context_str}\n"
        f"--- END OF CONTEXT ---\n"
        f"CRITICAL RULE: Answer using ONLY the direct facts explicitly stated in the CONTEXT above. "
        f"Do NOT paraphrase to introduce general facts, external descriptions, or general knowledge. "
        f"If the CONTEXT does not contain a specific fact or detail, omit it completely. "
        f"Keep your explanation strictly limited to the text provided."
    )
    
    messages = [SystemMessage(content=system_content)]
    if request.history:
        for msg in request.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            else:
                messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=request.question))
    
    llm = get_llm_model()
    try:
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        answer = f"Error communicating with DocMind AI: {str(e)}"
        
    # Extract cited source indices and clean up answer
    import re
    cited_indices = []
    match = re.search(r"Cited Source Indices:\s*([\d\s,]+)", answer, re.IGNORECASE)
    if match:
        idx_str = match.group(1)
        cited_indices = [int(i.strip()) for i in idx_str.split(",") if i.strip().isdigit()]
        
    # Strip Cited Source Indices from answer
    answer = re.sub(r"\n*Cited Source Indices:\s*.*", "", answer, flags=re.IGNORECASE).strip()
    
    if cited_indices:
        filtered_sources = []
        for idx in cited_indices:
            if 0 <= idx < len(sources):
                filtered_sources.append(sources[idx])
        if filtered_sources:
            sources = filtered_sources
            
    fallback_phrases = [
        "cannot find that information", "cannot find this information", "not find that information",
        "not find this information", "not present in the uploaded documents", "not mentioned in the provided",
        "information is not in the", "not found in the uploaded", "do not contain information",
        "does not contain information", "no information about", "unable to find", "cannot find information",
        "not found in the provided", "not mention this", "not mention that"
    ]
    if any(phrase in answer.lower() for phrase in fallback_phrases):
        sources = []
        confidence = 0
        
    return ChatResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        mode=request.mode
    )

def run_chat_stream(request: ChatRequest, user_id: str):
    """
    Generator yielding Server-Sent Events (SSE) representing token stream and citations metadata.
    Saves message exchanges in SQLite.
    """
    is_conv = check_conversational(request.question)
    
    sources = []
    confidence = 0
    
    if is_conv:
        system_prompt = (
            "You are DocMind AI, a friendly and intelligent document analysis assistant.\n"
            "You help users analyze documents, extract summaries, generate quizzes, and compare cross-references.\n"
            "Since the user just greeted you or asked a general conversational question, respond in a friendly, polite, and brief manner.\n"
            "Let them know you are DocMind AI and are ready to help them analyze the uploaded documents once they select or ask about them."
        )
        messages_list = [SystemMessage(content=system_prompt)]
        if request.history:
            for msg in request.history:
                if msg.role == "user":
                    messages_list.append(HumanMessage(content=msg.content))
                else:
                    messages_list.append(AIMessage(content=msg.content))
        messages_list.append(HumanMessage(content=request.question))
    else:
        # Search vector DB with BM25 hybrid ranking re-scoring
        search_results = search_index(request.question, request.doc_ids, top_k=TOP_K)
        
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
            
        system_prompt = SYSTEM_PROMPTS.get(request.mode, SYSTEM_PROMPTS["qa"])
        system_content = (
            f"{system_prompt}\n"
            f"--- CONTEXT ---\n"
            f"{context_str}\n"
            f"--- END OF CONTEXT ---\n"
            f"CRITICAL RULE: Answer using ONLY the direct facts explicitly stated in the CONTEXT above. "
            f"Do NOT paraphrase to introduce general facts, external descriptions, or general knowledge. "
            f"If the CONTEXT does not contain a specific fact or detail, omit it completely. "
            f"Keep your explanation strictly limited to the text provided."
        )
        messages_list = [SystemMessage(content=system_content)]
        if request.history:
            for msg in request.history:
                if msg.role == "user":
                    messages_list.append(HumanMessage(content=msg.content))
                else:
                    messages_list.append(AIMessage(content=msg.content))
        messages_list.append(HumanMessage(content=request.question))

    # 1. Send metadata event first so frontend gets confidence and source citations immediately
    metadata_event = {
        "type": "metadata",
        "confidence": confidence,
        "sources": [s.dict() for s in sources],
        "mode": request.mode
    }
    yield f"data: {json.dumps(metadata_event)}\n\n"

    # Assemble payload for direct REST call to OpenRouter with streaming enabled
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
    payload = {
        "model": LLM_MODEL,
        "messages": formatted_messages,
        "temperature": 0.2,
        "stream": True
    }

    url = "https://openrouter.ai/api/v1/chat/completions"
    full_answer = ""
    
    try:
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
                            full_answer += delta
                            yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
                    except Exception:
                        pass
    except Exception as e:
        err_msg = f"Error communicating with DocMind AI: {str(e)}"
        yield f"data: {json.dumps({'type': 'token', 'text': err_msg})}\n\n"
        full_answer += err_msg

    # Extract cited source indices and clean up full_answer
    import re
    cited_indices = []
    match = re.search(r"Cited Source Indices:\s*([\d\s,]+)", full_answer, re.IGNORECASE)
    if match:
        idx_str = match.group(1)
        cited_indices = [int(i.strip()) for i in idx_str.split(",") if i.strip().isdigit()]
    
    # Strip Cited Source Indices from full_answer
    full_answer = re.sub(r"\n*Cited Source Indices:\s*.*", "", full_answer, flags=re.IGNORECASE).strip()
    
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
    if any(phrase in full_answer.lower() for phrase in fallback_phrases):
        confidence = 0
        sources = []
        update_metadata_event = {
            "type": "metadata",
            "confidence": 0,
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
        print(f"Error persisting conversation to SQLite: {db_err}")

    # 3. Yield done event
    yield "data: {\"type\": \"done\"}\n\n"

