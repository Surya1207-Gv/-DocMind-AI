from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    question: str
    doc_ids: List[str] = Field(default_factory=list)
    history: List[ChatMessage] = Field(default_factory=list)
    mode: str = "qa" # "qa", "summary", "deep", "eli5"
    # Ask the backend to emit the full RAG trace alongside the answer. Opt-in so
    # normal responses stay small.
    trace: bool = False

class SourceChunk(BaseModel):
    """
    One cited passage.

    The extra locators are optional so older persisted chat history (which was
    stored without them) still deserialises.
    """
    text: str
    page: int
    doc_id: str
    doc_name: str
    relevance: float
    # Set when the passage was expanded with its following chunk and that chunk
    # sits on a later page, so the citation can honestly say "pages 5-6".
    page_end: Optional[int] = None
    # Position of the passage within the document, for stable deep-linking.
    chunk_index: Optional[int] = None
    # Nearest preceding heading, when the source format exposes one.
    section: Optional[str] = None
    # Origin URL, for documents ingested from the web.
    source_url: Optional[str] = None
    # Whether claim verification found this passage actually supporting the
    # answer, as opposed to merely being retrieved alongside it.
    supports_answer: Optional[bool] = None

class ChatResponse(BaseModel):
    answer: str
    confidence: int
    confidence_label: str = "Medium"
    sources: List[SourceChunk]
    mode: str
    # high | medium | low | very_low -- drives the evidence gate.
    confidence_band: Optional[str] = None
    # Claim-level grounding report (see backend/verification.py).
    verification: Optional[Dict[str, Any]] = None
    # Cross-document conflicts found in the evidence.
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)

class QuizQuestion(BaseModel):
    id: int
    question: str
    options: List[str]
    correct: str # option text or index
    difficulty: str
    page_ref: int

class QuizResponse(BaseModel):
    doc_id: str
    questions: List[QuizQuestion]

class CompareRequest(BaseModel):
    doc_ids: List[str]
    question: str

class DocumentCompareResult(BaseModel):
    doc_id: str
    doc_name: str
    summary: str

class CompareResponse(BaseModel):
    comparison_answer: str
    documents: List[DocumentCompareResult]
    # Passages the comparison was drawn from, so a difference can be checked
    # against the text that produced it.
    sources: List[SourceChunk] = Field(default_factory=list)
    # Values that disagree across the compared documents.
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)

class EntityInfo(BaseModel):
    name: str
    type: str # Person, Date, Organization, Location, etc.
    description: str

class SmartAlert(BaseModel):
    type: str # warning, date, stat, insight
    content: str
    page: Optional[int] = None

class DocumentAnalytics(BaseModel):
    doc_id: str
    doc_name: str
    word_count: int
    page_count: int
    read_time_mins: int
    complexity_score: str # Easy, Medium, Hard
    summary: List[str] # 5 bullet points
    entities: List[EntityInfo]
    alerts: List[SmartAlert]
    suggested_questions: List[str]

class DocumentInfo(BaseModel):
    id: str
    name: str
    size: int
    upload_time: str

class AgentQueryRequest(BaseModel):
    question: str
    doc_ids: List[str] = Field(default_factory=list)
    mode: str = "deep"

class AgentQueryResponse(BaseModel):
    answer: str
    sub_queries: List[str]
    confidence: int
    confidence_label: str
    sources: List[Dict[str, Any]]
    verification_status: Dict[str, Any]

