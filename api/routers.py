from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from pydantic import BaseModel, Field

from logger.logger import get_app_logger
from models.models import (
    ChatRequest,
    ComparisonRequest,
    RCARequest,
    ResearchRequest,
)
from src.document_analyze.service import RootCauseAnalysisService
from src.document_comparison.service import ComparisonService
from src.multidoc_chat.service import ResearchCopilotService
from src.single_docchat.service import MaintenanceCopilotService
from utils.evaluation_artifacts import EvaluationArtifactsService
from utils.evaluation_runner import EvaluationJobManager, ExperimentConfig
from utils.model_loader import get_model_catalog
from utils.upload_context import UploadContextManager
from utils.vectorstore_manager import VectorStoreManager

log = get_app_logger("api_routers")
templates = Jinja2Templates(directory="templates")
router = APIRouter()


class EvaluationRunRequest(BaseModel):
    feature: str = "all"
    limit: int | None = Field(default=None, ge=1, le=50)
    llm_models: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    top_k: int | None = Field(default=None, ge=1, le=20)
    fetch_k: int | None = Field(default=None, ge=1, le=50)


@lru_cache(maxsize=1)
def get_upload_manager() -> UploadContextManager:
    return UploadContextManager()


@lru_cache(maxsize=1)
def get_vectorstore() -> VectorStoreManager:
    return VectorStoreManager()


@lru_cache(maxsize=1)
def get_rca_service() -> RootCauseAnalysisService:
    vs = get_vectorstore()
    uploads = get_upload_manager()
    return RootCauseAnalysisService(vectorstore=vs, upload_manager=uploads)


@lru_cache(maxsize=1)
def get_maintenance_service() -> MaintenanceCopilotService:
    vs = get_vectorstore()
    uploads = get_upload_manager()
    return MaintenanceCopilotService(vectorstore=vs, upload_manager=uploads)


@lru_cache(maxsize=1)
def get_comparison_service() -> ComparisonService:
    vs = get_vectorstore()
    uploads = get_upload_manager()
    return ComparisonService(vectorstore=vs, upload_manager=uploads)


@lru_cache(maxsize=1)
def get_research_service() -> ResearchCopilotService:
    vs = get_vectorstore()
    uploads = get_upload_manager()
    return ResearchCopilotService(vectorstore=vs, upload_manager=uploads)


@lru_cache(maxsize=1)
def get_eval_artifacts() -> EvaluationArtifactsService:
    return EvaluationArtifactsService()


@lru_cache(maxsize=1)
def get_eval_jobs() -> EvaluationJobManager:
    return EvaluationJobManager()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.get("/rca", response_class=HTMLResponse)
async def rca_page(request: Request):
    return templates.TemplateResponse("rca.html", {"request": request})


@router.get("/maintenance", response_class=HTMLResponse)
async def maintenance_page(request: Request):
    return templates.TemplateResponse("maintenance.html", {"request": request})


@router.get("/comparison", response_class=HTMLResponse)
async def comparison_page(request: Request):
    return templates.TemplateResponse("comparison.html", {"request": request})


@router.get("/research", response_class=HTMLResponse)
async def research_page(request: Request):
    return templates.TemplateResponse("research.html", {"request": request})


@router.get("/metrics-dashboard", response_class=HTMLResponse)
async def metrics_dashboard(request: Request):
    return templates.TemplateResponse("metrics.html", {"request": request})


@router.post("/api/v1/uploads")
async def upload_pdf(session_id: str = Form(None), file: UploadFile = File(...)):
    manager = get_upload_manager()
    sid = session_id or manager.create_session()
    content = await file.read()
    chunks = manager.add_pdf(sid, file.filename or "upload.pdf", content)
    log.info("Upload API success", session_id=sid, chunks=len(chunks))
    return {"session_id": sid, "chunks": len(chunks), "filename": file.filename}


@router.post("/api/v1/rca")
async def api_rca(payload: RCARequest):
    result = get_rca_service().analyze(payload)
    return result.model_dump()


@router.post("/api/v1/maintenance")
async def api_maintenance(payload: ChatRequest):
    result = get_maintenance_service().chat(payload)
    return result.model_dump()


@router.post("/api/v1/comparison")
async def api_comparison(payload: ComparisonRequest):
    result = get_comparison_service().compare(payload)
    return result.model_dump()


@router.post("/api/v1/research")
async def api_research(payload: ResearchRequest):
    result = get_research_service().ask(payload)
    return result.model_dump()


@router.get("/api/v1/evaluation/catalog")
async def api_evaluation_catalog():
    return get_model_catalog()


@router.get("/api/v1/evaluation/runs")
async def api_evaluation_runs():
    return {"runs": get_eval_artifacts().list_runs()}


@router.get("/api/v1/evaluation/latest")
async def api_evaluation_latest(run_id: str | None = None):
    payload = get_eval_artifacts().load_run(run_id)
    if not payload:
        return {"run_id": None, "detail": "No evaluation runs found"}
    return payload


@router.get("/api/v1/evaluation/status")
async def api_evaluation_status():
    return get_eval_jobs().status()


@router.post("/api/v1/evaluation/run")
async def api_evaluation_run(payload: EvaluationRunRequest):
    config = ExperimentConfig(
        feature=payload.feature,
        limit=payload.limit,
        llm_models=payload.llm_models,
        metrics=payload.metrics,
        temperature=payload.temperature,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
    )
    state = get_eval_jobs().start(config)
    log.info(
        "Evaluation job requested",
        status=state.get("status"),
        feature=payload.feature,
        limit=payload.limit,
        llms=payload.llm_models,
    )
    return state
