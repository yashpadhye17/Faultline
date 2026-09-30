from __future__ import annotations

from abc import ABC
from typing import Any

from logger.logger import get_app_logger
from models.models import Citation, RunMeta
from utils.config_loader import get_config_loader
from utils.model_loader import ModelLoader
from utils.prompt_manager import PromptManager
from utils.retriever_service import RetrieverService
from utils.telemetry import TelemetryService
from utils.citation_enforcer import enforce_citations
from utils.upload_context import UploadContextManager
from utils.vectorstore_manager import VectorStoreManager


class BaseFeatureService(ABC):
    """Shared dependencies and helpers for all product features."""

    feature_name: str = "base"
    prompt_id: str = ""

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        vectorstore: VectorStoreManager | None = None,
        retriever: RetrieverService | None = None,
        prompt_manager: PromptManager | None = None,
        upload_manager: UploadContextManager | None = None,
    ):
        self.log = get_app_logger(self.__class__.__name__)
        self.config = get_config_loader()
        self.telemetry = TelemetryService(self.__class__.__name__)
        self.model_loader = model_loader or ModelLoader()
        self.vectorstore = vectorstore or VectorStoreManager(self.model_loader)
        self.retriever = retriever or RetrieverService(self.vectorstore)
        self.prompt_manager = prompt_manager or PromptManager()
        self.upload_manager = upload_manager or UploadContextManager()
        self.llm = self.model_loader.load_llm()
        self.log.info("Feature service initialized", feature=self.feature_name)

    def build_run_meta(
        self,
        *,
        request_id: str,
        prompt_id: str,
        prompt_version: str,
        latency_ms: float,
        num_contexts: int,
    ) -> RunMeta:
        reranker_cfg = self.config.get("retriever", "reranker", default={}) or {}
        return RunMeta(
            request_id=request_id,
            feature=self.feature_name,
            model_name=str(self.config.get("llm", "model_name", default="")),
            embedding_model=str(self.config.get("embedding_model", "model_name", default="")),
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            top_k=int(self.config.get("retriever", "top_k", default=6)),
            fetch_k=int(self.config.get("retriever", "fetch_k", default=20)),
            reranker_model=str(reranker_cfg.get("model_name", "")) if reranker_cfg.get("enabled", True) else "",
            chunk_overlap=int(self.config.get("chunking", "overlap", default=200)),
            latency_ms=latency_ms,
            num_contexts=num_contexts,
        )

    def invoke_llm(self, prompt: str) -> str:
        self.log.info("Invoking LLM", feature=self.feature_name, prompt_chars=len(prompt))
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(str(part) for part in content)
        return str(content).strip()

    def finalize_answer(self, answer: str, hits) -> tuple[str, list[Citation]]:
        citations = self.retriever.to_citations(hits)
        cite_cfg = self.config.get("retriever", "citation_enforcement", default={}) or {}
        if cite_cfg.get("enabled", True):
            answer, citations = enforce_citations(
                answer,
                citations,
                enabled=True,
                append_sources_if_missing=bool(cite_cfg.get("append_sources_if_missing", True)),
            )
        return answer, citations

    @staticmethod
    def parse_bullet_section(text: str, header: str) -> list[str]:
        lines = text.splitlines()
        capturing = False
        items: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith(header.lower()):
                capturing = True
                continue
            if capturing:
                if stripped.startswith("##") or (stripped.endswith(":") and not stripped.startswith(("-", "*", "1"))):
                    if items:
                        break
                if stripped.startswith(("-", "*", "•")):
                    items.append(stripped.lstrip("-*• ").strip())
                elif stripped[:2].isdigit() and "." in stripped[:4]:
                    items.append(stripped.split(".", 1)[1].strip())
        return items
