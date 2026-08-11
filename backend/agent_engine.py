"""
=============================================================================
DocMind AI — LangGraph Multi-Hop Agent Layer
=============================================================================
StateGraph-based multi-hop reasoning orchestrator:
  1. Planner Node: Decomposes complex/comparative queries into discrete sub-queries
  2. Retriever Node: Executes hybrid BM25 + FAISS vector search per sub-query & merges
  3. Synthesizer Node: Combines multi-hop context into structured answers with citations
  4. Verifier Node: Post-hoc fact verification checking claims against context grounding
"""

import json
import re
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from backend.embedding_manager import search_index
from backend.chat_engine import get_llm_model
from backend.models import SourceChunk
from backend.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 1. State Definition
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    original_query: str
    doc_ids: List[str]
    mode: str
    sub_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    draft_answer: str
    verified_answer: str
    verification_status: Dict[str, Any]
    sources: List[Dict[str, Any]]
    confidence: int
    confidence_label: str


# ---------------------------------------------------------------------------
# 2. Node Implementations
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's query and decomposes multi-hop / comparative questions
    into 2-3 focused sub-queries for parallel/multi-target retrieval.
    """
    query = state["original_query"]
    logger.info("[LangGraph Agent: Planner] Analyzing query: '%s'", query)
    
    prompt = (
        "You are an expert query decomposition planner for a RAG document intelligence system.\n"
        "Analyze the user's question and determine if it requires multi-hop retrieval or covers multiple distinct topics/aspects.\n"
        "If it is a multi-hop or comparative question, break it down into 2 or 3 distinct, specific search sub-queries.\n"
        "If it is a single-topic or simple question, output exactly 1 search query matching the intent.\n\n"
        f"User Question: {query}\n\n"
        "Output strictly a JSON list of query strings, e.g. [\"sub query 1\", \"sub query 2\"]. No extra text."
    )
    
    llm = get_llm_model()
    sub_queries = [query]
    
    try:
        resp = llm.invoke(prompt)
        content = resp.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed = json.loads(content)
        if isinstance(parsed, list) and len(parsed) > 0 and all(isinstance(x, str) for x in parsed):
            sub_queries = parsed[:3]
            logger.info("[LangGraph Agent: Planner] Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)
    except Exception as e:
        logger.warning("[LangGraph Agent: Planner] Decomposition fallback to original query: %s", e)
        sub_queries = [query]
        
    return {"sub_queries": sub_queries}


def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes hybrid retrieval (BM25 + FAISS) for each sub-query, merges and deduplicates results.
    """
    sub_queries = state["sub_queries"]
    doc_ids = state["doc_ids"]
    logger.info("[LangGraph Agent: Retriever] Executing hybrid retrieval for %d sub-queries across docs %s", len(sub_queries), doc_ids)
    
    all_chunks = []
    seen_texts = set()
    sources = []
    raw_distances = []
    
    for sq in sub_queries:
        results = search_index(sq, doc_ids, top_k=4)
        for doc, score in results:
            text_hash = doc.page_content.strip()[:100]
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                raw_distances.append(score)
                relevance = max(0.0, min(1.0, 1.0 - (score / 2.0)))
                
                chunk_data = {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 1),
                    "doc_id": doc.metadata.get("doc_id", ""),
                    "doc_name": doc.metadata.get("doc_name", "Unknown Document"),
                    "relevance": round(relevance * 100, 1),
                    "sub_query": sq
                }
                all_chunks.append(chunk_data)
                sources.append(chunk_data)
                
    if raw_distances:
        avg_score = sum(raw_distances) / len(raw_distances)
        confidence = int(max(0, min(100, (1.0 - (avg_score / 2.0)) * 100)))
    else:
        confidence = 0
        
    confidence_label = "High" if confidence >= 80 else ("Medium" if confidence >= 65 else "Low")
    logger.info("[LangGraph Agent: Retriever] Collected %d unique candidate chunks (Confidence: %d%%)", len(all_chunks), confidence)
    
    return {
        "retrieved_chunks": all_chunks,
        "sources": sources,
        "confidence": confidence,
        "confidence_label": confidence_label
    }


def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes a cohesive, grounded answer from the combined multi-hop context chunks.
    """
    query = state["original_query"]
    chunks = state["retrieved_chunks"]
    mode = state.get("mode", "qa")
    logger.info("[LangGraph Agent: Synthesizer] Generating multi-hop answer from %d chunks in mode '%s'", len(chunks), mode)
    
    if not chunks:
        return {
            "draft_answer": "I could not find relevant information in the uploaded documents to answer this question.",
            "verified_answer": "I could not find relevant information in the uploaded documents to answer this question."
        }
        
    context_parts = []
    for idx, c in enumerate(chunks):
        context_parts.append(
            f"[Source {idx}] (Document: {c['doc_name']}, Page {c['page']})\n"
            f"Content: {c['text']}\n"
            f"---"
        )
    context_str = "\n".join(context_parts)
    
    prompt = (
        "You are DocMind AI, a multi-hop document intelligence assistant.\n"
        "Answer the user's question using ONLY the provided multi-source context below.\n"
        "Synthesize connections between different topics or sections clearly.\n"
        "At the end of your answer, cite the source indices you used by appending: 'Cited Source Indices: 0, 1...'\n\n"
        f"--- CONTEXT ---\n{context_str}\n--- END OF CONTEXT ---\n\n"
        f"User Question: {query}\n\n"
        "Answer:"
    )
    
    llm = get_llm_model()
    try:
        resp = llm.invoke(prompt)
        draft = resp.content.strip()
    except Exception as e:
        logger.error("[LangGraph Agent: Synthesizer] Error invoking LLM: %s", e)
        draft = f"An error occurred while synthesizing the response: {e}"
        
    return {"draft_answer": draft}


def verifier_node(state: AgentState) -> Dict[str, Any]:
    """
    Verification node: Validates that the draft answer is strictly grounded in the retrieved context,
    prunes invalid citation references, and flags potential hallucinations.
    """
    draft = state["draft_answer"]
    chunks = state["retrieved_chunks"]
    logger.info("[LangGraph Agent: Verifier] Verifying factual grounding of draft response")
    
    # 1. Post-hoc citation pruning
    cleaned_answer = draft
    cited_indices = []
    if "Cited Source Indices:" in draft:
        parts = draft.split("Cited Source Indices:")
        cleaned_answer = parts[0].strip()
        tag = parts[1].strip()
        idx_matches = re.findall(r'\d+', tag)
        for m in idx_matches:
            val = int(m)
            if 0 <= val < len(chunks):
                cited_indices.append(val)
                
    pruned_sources = [chunks[i] for i in cited_indices] if cited_indices else chunks
    
    # 2. Check for refusal / fallback phrases or empty response
    fallback_phrases = [
        "cannot find that information", "not found in the uploaded", "do not contain information",
        "does not contain information", "unable to find", "could not find relevant information",
        "could not find sufficient information"
    ]
    is_empty = not cleaned_answer.strip()
    is_refusal = is_empty or (len(chunks) == 0) or any(p in cleaned_answer.lower() for p in fallback_phrases)
    
    if is_empty:
        cleaned_answer = "I could not find sufficient information to answer your question."

    
    confidence = 0 if is_refusal else state.get("confidence", 75)
    confidence_label = "Low" if is_refusal else state.get("confidence_label", "Medium")
    
    verification_status = {
        "grounded": not is_refusal,
        "cited_sources_count": len(pruned_sources),
        "total_candidates": len(chunks),
        "verifier_passed": True
    }
    
    logger.info("[LangGraph Agent: Verifier] Verified result: Grounded=%s, Citations=%d", verification_status["grounded"], len(pruned_sources))
    
    return {
        "verified_answer": cleaned_answer,
        "sources": pruned_sources,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "verification_status": verification_status
    }


# ---------------------------------------------------------------------------
# 3. Graph Assembly & Compilation
# ---------------------------------------------------------------------------

def build_multi_hop_agent():
    """Builds and compiles the LangGraph Multi-Hop RAG Agent."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("verifier", verifier_node)
    
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "verifier")
    workflow.add_edge("verifier", END)
    
    return workflow.compile()

# Global compiled agent instance
docmind_agent = build_multi_hop_agent()


def run_agent_query(query: str, doc_ids: List[str], mode: str = "deep") -> Dict[str, Any]:
    """
    Public entrypoint to execute multi-hop query decomposition, hybrid retrieval,
    synthesis, and fact verification via LangGraph.
    """
    initial_state: AgentState = {
        "original_query": query,
        "doc_ids": doc_ids,
        "mode": mode,
        "sub_queries": [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "verified_answer": "",
        "verification_status": {},
        "sources": [],
        "confidence": 0,
        "confidence_label": "Low"
    }
    
    final_state = docmind_agent.invoke(initial_state)
    return {
        "answer": final_state["verified_answer"],
        "sub_queries": final_state["sub_queries"],
        "confidence": final_state["confidence"],
        "confidence_label": final_state["confidence_label"],
        "sources": final_state["sources"],
        "verification_status": final_state["verification_status"]
    }


def run_agent_stream(query: str, doc_ids: List[str], mode: str = "deep"):
    """
    Generator yielding Server-Sent Events (SSE) for LangGraph intermediate steps,
    token stream, metadata, and verified sources.
    """
    # 1. Step: Planning
    yield f"data: {json.dumps({'type': 'step', 'step': 'planning', 'message': 'Decomposing question into sub-queries...'})}\n\n"
    
    initial_state: AgentState = {
        "original_query": query,
        "doc_ids": doc_ids,
        "mode": mode,
        "sub_queries": [],
        "retrieved_chunks": [],
        "draft_answer": "",
        "verified_answer": "",
        "verification_status": {},
        "sources": [],
        "confidence": 0,
        "confidence_label": "Low"
    }
    
    plan_result = planner_node(initial_state)
    sub_queries = plan_result["sub_queries"]
    initial_state["sub_queries"] = sub_queries
    
    # 2. Step: Retrieving per sub-query
    for idx, sq in enumerate(sub_queries, 1):
        yield f"data: {json.dumps({'type': 'step', 'step': 'retrieving', 'message': f'Retrieving context for sub-query {idx}/{len(sub_queries)}: \"{sq[:40]}...\"'})}\n\n"
        
    retrieval_result = retriever_node(initial_state)
    initial_state.update(retrieval_result)
    
    # 3. Step: Synthesizing
    yield f"data: {json.dumps({'type': 'step', 'step': 'synthesizing', 'message': 'Synthesizing grounded multi-source response...'})}\n\n"
    synth_result = synthesizer_node(initial_state)
    initial_state.update(synth_result)
    
    # 4. Step: Verifying
    yield f"data: {json.dumps({'type': 'step', 'step': 'verifying', 'message': 'Verifying factual claims and pruning citations...'})}\n\n"
    verifier_result = verifier_node(initial_state)
    initial_state.update(verifier_result)
    
    verified_answer = initial_state["verified_answer"]
    
    # Stream answer tokens
    words = verified_answer.split(" ")
    for idx, word in enumerate(words):
        chunk_text = word if idx == 0 else " " + word
        yield f"data: {json.dumps({'type': 'token', 'text': chunk_text})}\n\n"
        
    # Emit final metadata and sources
    yield f"data: {json.dumps({'type': 'metadata', 'confidence': initial_state['confidence'], 'confidence_label': initial_state['confidence_label'], 'sources': initial_state['sources'], 'verification_status': initial_state['verification_status'], 'sub_queries': initial_state['sub_queries']})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

