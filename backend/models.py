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

class SourceChunk(BaseModel):
    text: str
    page: int
    doc_id: str
    doc_name: str
    relevance: float

class ChatResponse(BaseModel):
    answer: str
    confidence: int
    sources: List[SourceChunk]
    mode: str

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
