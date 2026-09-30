from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.routers import router
from exception.exceptions import ReportsException
from logger.logger import get_app_logger
from models.models import HealthStatus
from utils.config_loader import get_config_loader
from utils.vectorstore_manager import VectorStoreManager

load_dotenv()
log = get_app_logger("main")
config = get_config_loader()

app = FastAPI(
    title=config.get("app", "name", default="Engineering Intelligence & Reliability Copilot"),
    version=config.get("app", "version", default="1.1.0"),
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.include_router(router)


@app.exception_handler(ReportsException)
async def reports_exception_handler(_: Request, exc: ReportsException):
    log.error("ReportsException", error=str(exc))
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz():
    return HealthStatus(
        status="ok",
        app=str(config.get("app", "name", default="")),
        version=str(config.get("app", "version", default="")),
    )


@app.get("/readyz")
async def readyz():
    vs = VectorStoreManager()
    loaded = vs.exists()
    manifest = vs.read_manifest() if loaded else None
    return HealthStatus(
        status="ready" if loaded else "not_ready",
        app=str(config.get("app", "name", default="")),
        version=str(config.get("app", "version", default="")),
        vectorstore_loaded=loaded,
        doc_count=manifest.doc_count if manifest else 0,
    )


@app.get("/metrics")
async def metrics():
    vs = VectorStoreManager()
    manifest = vs.read_manifest() if vs.exists() else None
    reranker = config.get("retriever", "reranker", default={}) or {}
    hybrid = config.get("retriever", "hybrid", default={}) or {}
    cite = config.get("retriever", "citation_enforcement", default={}) or {}
    return {
        "app": config.get("app", "name"),
        "version": config.get("app", "version"),
        "vectorstore": manifest.model_dump() if manifest else None,
        "retriever_top_k": config.get("retriever", "top_k"),
        "retriever_fetch_k": config.get("retriever", "fetch_k"),
        "hybrid_search": hybrid.get("enabled", False),
        "bm25_enabled": manifest.bm25_enabled if manifest else False,
        "index_type": manifest.index_type if manifest else "",
        "reranker_enabled": reranker.get("enabled", False),
        "reranker_model": reranker.get("model_name") if reranker.get("enabled", False) else "",
        "citation_enforcement": cite.get("enabled", False),
        "chunk_overlap": config.get("chunking", "overlap"),
        "llm_model": config.get("llm", "model_name"),
        "embedding_model": config.get("embedding_model", "model_name"),
    }


log.info(
    "Application created",
    app=config.get("app", "name"),
    version=config.get("app", "version"),
)
