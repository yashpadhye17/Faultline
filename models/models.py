from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Prompt_Type(str, Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    DOCUMENT_SUMMARY = "document_summary"
    DOCUMENT_QUESTION_ANSWER = "document_question_answer"
    CONTEXT_QUESTION_ANSWER = "context_question_answer"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    MAINTENANCE_TROUBLESHOOTING = "maintenance_troubleshooting"
    ENGINEERING_COMPARISON = "engineering_comparison"
    NASA_RESEARCH = "nasa_research"


class FeatureName(str, Enum):
    RCA = "rca"
    MAINTENANCE = "maintenance"
    COMPARISON = "comparison"
    RESEARCH = "research"


class Metadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Report(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    content: Optional[str] = None


class Citation(BaseModel):
    doc_id: str
    title: str = ""
    citation_url: str = ""
    snippet: str = ""
    score: float = 0.0
    feature_tags: str = ""
    center_code: str = ""
    year: Optional[int] = None


class RunMeta(BaseModel):
    request_id: str
    feature: str
    model_name: str
    embedding_model: str = ""
    prompt_id: str
    prompt_version: str
    top_k: int
    fetch_k: int = 0
    reranker_model: str = ""
    chunk_overlap: int = 0
    latency_ms: float
    num_contexts: int = 0
    created_at: str = Field(default_factory=utc_now_iso)


class RCARequest(BaseModel):
    incident_description: str
    system_context: str = ""
    session_id: Optional[str] = None


class RCACause(BaseModel):
    rank: int
    cause: str
    evidence: str
    confidence: str = "medium"


class RCAReport(BaseModel):
    problem_statement: str
    likely_causes: List[RCACause]
    recommended_checks: List[str]
    corrective_actions: List[str]
    summary: str
    citations: List[Citation]
    run_meta: RunMeta
    raw_answer: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    steps: List[str] = Field(default_factory=list)
    citations: List[Citation]
    run_meta: RunMeta


class ComparisonRequest(BaseModel):
    topic_a: str
    topic_b: str
    focus: str = "design, performance, risks, and recommendations"
    session_id: Optional[str] = None


class ComparisonReport(BaseModel):
    topic_a: str
    topic_b: str
    similarities: List[str]
    differences: List[str]
    risks: List[str]
    recommendations: List[str]
    summary: str
    citations: List[Citation]
    run_meta: RunMeta
    raw_answer: str = ""


class ResearchRequest(BaseModel):
    question: str
    center_code: Optional[str] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    session_id: Optional[str] = None


class ResearchResponse(BaseModel):
    answer: str
    citations: List[Citation]
    run_meta: RunMeta


class EvalSample(BaseModel):
    sample_id: str
    feature: FeatureName
    question: str
    ground_truth: str = ""
    relevant_doc_ids: List[str] = Field(default_factory=list)
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)


class RagasMetricResult(BaseModel):
    sample_id: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None


class EvaluationRunManifest(BaseModel):
    run_id: str
    created_at: str
    dataset_path: str
    model_name: str
    embedding_model: str
    top_k: int
    chunk_size: int
    sample_count: int
    aggregate_metrics: Dict[str, float] = Field(default_factory=dict)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    passed: bool = False


class HealthStatus(BaseModel):
    status: str
    app: str
    version: str
    vectorstore_loaded: bool = False
    doc_count: int = 0


class IndexBuildManifest(BaseModel):
    created_at: str
    doc_count: int
    chunk_count: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    corpus_hash: str
    path: str
    index_type: str = "hnsw"
    bm25_enabled: bool = True
    hybrid_search: bool = True
