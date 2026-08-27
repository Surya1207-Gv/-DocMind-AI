from typing import List, Dict, Any
from backend.models import CompareRequest, CompareResponse, DocumentCompareResult, SourceChunk
from backend.embedding_manager import search_index
from backend.chat_engine import UNTRUSTED_CONTENT_GUARD, get_llm_model
from backend.logger import get_logger
from backend.verification import detect_contradictions, format_contradiction_notice

logger = get_logger(__name__)



def compare_documents(request: CompareRequest, documents_meta: Dict[str, Dict[str, Any]]) -> CompareResponse:
    """
    Retrieves comparative context from multiple documents and queries Gemini for differences/similarities.
    """
    # 1. Search vector DB across selected doc_ids
    search_results = search_index(request.question, request.doc_ids, top_k=6)
    
    # 2. Format context by document, keeping every passage as a citable source
    #    so a stated difference can be traced back to the text it came from.
    context_by_doc = {}
    sources: List[SourceChunk] = []
    for doc, score in search_results:
        metadata = doc.metadata or {}
        d_id = metadata.get("doc_id", "")
        d_name = metadata.get("doc_name", "Unknown Document")
        page = metadata.get("page", 1)
        relevance = max(0.0, min(1.0, 1.0 - (score / 2.0)))

        sources.append(SourceChunk(
            text=doc.page_content,
            page=page,
            page_end=metadata.get("page_end"),
            chunk_index=metadata.get("chunk_index"),
            section=metadata.get("section"),
            source_url=metadata.get("source_url"),
            doc_id=d_id,
            doc_name=d_name,
            relevance=round(relevance * 100, 1),
        ))

        if d_id not in context_by_doc:
            context_by_doc[d_id] = {
                "name": d_name,
                "chunks": []
            }
        context_by_doc[d_id]["chunks"].append(f"[Page {page}]: {doc.page_content}")
        
    context_str = ""
    for d_id, doc_info in context_by_doc.items():
        chunks_str = "\n".join(doc_info["chunks"])
        context_str += f"=== DOCUMENT: {doc_info['name']} ===\n{chunks_str}\n=====================================\n\n"
        
    if not context_str:
        return CompareResponse(
            comparison_answer="No relevant content found in the selected documents to answer this question.",
            documents=[],
            sources=[],
            contradictions=[],
        )

    # Values that disagree across documents are the single most useful output of
    # a comparison, and the easiest for a fluent summary to smooth over. Detect
    # them deterministically and tell the model not to reconcile them.
    contradictions = detect_contradictions([
        {"text": s.text, "doc_id": s.doc_id, "doc_name": s.doc_name, "page": s.page}
        for s in sources
    ])
        
    # 3. Call Gemini to perform comparison
    prompt = (
        "You are an expert comparative analyst.\n"
        "Analyze the excerpted text from the different documents below and answer the user's comparison question.\n"
        "Present your response using clear comparative sections: highlight similarities, list key differences, and provide a concluding summary.\n\n"
        "--- DOCUMENT TEXTS ---\n"
        f"{context_str[:12000]}\n"
        "--- END DOCUMENT TEXTS ---\n\n"
        f"Comparison Question: {request.question}\n\n"
        "Response Structure:\n"
        "1. **Summary Table or Overview** (if helpful)\n"
        "2. **Similarities**\n"
        "3. **Key Differences**\n"
        "4. **Conclusion/Key Takeaway**\n\n"
        "Be factual and only cite what is present in the text.\n"
        "Cite the document name and page for every difference you state.\n\n"
        + (
            "IMPORTANT - these values conflict across the documents. Report the "
            "conflict explicitly with both values and their documents; do not "
            "reconcile, average, or choose between them: "
            + "; ".join(c["subject"] for c in contradictions) + "\n\n"
            if contradictions else ""
        )
        + UNTRUSTED_CONTENT_GUARD
    )
    
    llm = get_llm_model()
    try:
        response = llm.invoke(prompt)
        comparison_answer = response.content
    except Exception as e:
        comparison_answer = f"Error performing comparison: {str(e)}"
        
    # 4. Generate quick, individual comparative summaries for each doc
    doc_results = []
    for d_id in request.doc_ids:
        doc_meta = documents_meta.get(d_id, {})
        d_name = doc_meta.get("name", "Unknown")
        
        # Get chunks just for this doc
        doc_chunks = context_by_doc.get(d_id, {}).get("chunks", [])
        if doc_chunks:
            doc_context = "\n".join(doc_chunks)[:3000]
            summary_prompt = (
                f"Based on the following text from {d_name}, write a short 2-sentence summary of what this document says regarding: '{request.question}'\n\n"
                f"Text:\n{doc_context}\n\n"
                "Summary (exactly 2 sentences):"
            )
            try:
                summary_resp = llm.invoke(summary_prompt)
                doc_summary = summary_resp.content.strip()
            except Exception:
                doc_summary = f"Contains relevant text about {request.question}."
        else:
            doc_summary = "This document does not contain direct mentions of the comparison topic."
            
        doc_results.append(DocumentCompareResult(
            doc_id=d_id,
            doc_name=d_name,
            summary=doc_summary
        ))
        
    if contradictions:
        # Lead with the conflict rather than trusting it to survive inside the
        # model's prose.
        comparison_answer = (
            format_contradiction_notice(contradictions) + "\n\n" + comparison_answer
        )

    return CompareResponse(
        comparison_answer=comparison_answer,
        documents=doc_results,
        sources=sources,
        contradictions=contradictions,
    )
