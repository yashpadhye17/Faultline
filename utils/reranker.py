from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from logger.logger import get_app_logger
from utils.config_loader import get_config_loader
from utils.telemetry import TelemetryService


class CrossEncoderReranker:
    """Local HuggingFace cross-encoder reranker for retrieval precision uplift."""

    def __init__(self, model_name: str | None = None):
        self.log = get_app_logger("reranker")
        self.telemetry = TelemetryService("reranker")
        loader = get_config_loader()
        self.model_name = model_name or str(
            loader.get("retriever", "reranker", "model_name", default="BAAI/bge-reranker-base")
        )
        self.enabled = bool(loader.get("retriever", "reranker", "enabled", default=True))
        self._model: CrossEncoder | None = None
        self.log.info(
            "Reranker configured",
            enabled=self.enabled,
            model_name=self.model_name,
        )

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self.log.info("Loading cross-encoder reranker", model=self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        *,
        top_n: int,
    ) -> list[tuple[Document, float]]:
        if not documents:
            return []
        if not self.enabled or len(documents) == 1:
            return [(doc, 0.0) for doc in documents[:top_n]]

        timer = self.telemetry.timed()
        pairs = [(query, doc.page_content[:4000]) for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(documents, [float(s) for s in scores]),
            key=lambda item: item[1],
            reverse=True,
        )
        kept = ranked[:top_n]
        self.log.info(
            "Rerank complete",
            candidate_count=len(documents),
            kept=len(kept),
            latency_ms=timer.latency_ms,
            model=self.model_name,
            top_score=kept[0][1] if kept else None,
        )
        return kept


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()
