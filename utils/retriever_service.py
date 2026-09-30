from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from logger.logger import get_app_logger
from models.models import Citation
from utils.config_loader import get_config_loader
from utils.reranker import CrossEncoderReranker, get_reranker
from utils.telemetry import TelemetryService
from utils.vectorstore_manager import VectorStoreManager


class RetrieverService:
    """BM25 + HNSW hybrid recall → cross-encoder rerank → fused ranking."""

    def __init__(
        self,
        vectorstore: VectorStoreManager | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.log = get_app_logger("retriever_service")
        self.loader = get_config_loader()
        self.vectorstore = vectorstore or VectorStoreManager()
        self.reranker = reranker or get_reranker()
        self.telemetry = TelemetryService("retriever")
        self.default_top_k: int | None = None
        self.default_fetch_k: int | None = None

    def feature_filters(self, feature: str) -> dict[str, Any]:
        mapping = self.loader.get("retriever", "feature_tag_filters", default={}) or {}
        tags = mapping.get(feature) or []
        return {"feature_tags": tags} if tags else {}

    def retrieve(
        self,
        query: str,
        *,
        feature: str | None = None,
        k: int | None = None,
        fetch_k: int | None = None,
        extra_filters: dict[str, Any] | None = None,
        upload_docs: list[Document] | None = None,
    ) -> list[tuple[Document, float]]:
        timer = self.telemetry.timed()
        top_k = k or self.default_top_k or int(self.loader.get("retriever", "top_k", default=6))
        fetch_k = (
            fetch_k
            if fetch_k is not None
            else self.default_fetch_k
            if self.default_fetch_k is not None
            else int(self.loader.get("retriever", "fetch_k", default=20))
        )
        filters = self.feature_filters(feature) if feature else {}
        if extra_filters:
            filters.update(extra_filters)

        hybrid_hits = self.vectorstore.hybrid_search(query, k=fetch_k, filters=filters)
        faiss_hits = self._dedupe_hits(hybrid_hits)

        if self.reranker.enabled and faiss_hits:
            docs = [doc for doc, _ in faiss_hits]
            faiss_scores = {id(doc): score for doc, score in faiss_hits}
            reranked = self.reranker.rerank(query, docs, top_n=min(top_k * 2, len(docs)))
            fused = self._fuse_scores(reranked, faiss_scores)
            hits = fused[:top_k]
        else:
            hits = faiss_hits[:top_k]

        if upload_docs:
            upload_scored = [(doc, 1.05) for doc in upload_docs[: max(1, top_k // 2)]]
            hits = self._dedupe_hits(upload_scored + hits)[:top_k]

        self.log.info(
            "Retriever finished",
            feature=feature,
            latency_ms=timer.latency_ms,
            fetch_k=fetch_k,
            top_k=top_k,
            pipeline="bm25+hnsw_rrf→rerank",
            reranker_enabled=self.reranker.enabled,
            reranker_model=self.reranker.model_name if self.reranker.enabled else "",
            hits=len(hits),
            doc_ids=[h[0].metadata.get("doc_id") for h in hits],
            chunk_overlap=self.loader.get("chunking", "overlap", default=200),
        )
        return hits

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            return [0.5 for _ in values]
        return [(v - lo) / (hi - lo) for v in values]

    def _fuse_scores(
        self,
        reranked: list[tuple[Document, float]],
        faiss_scores: dict[int, float],
    ) -> list[tuple[Document, float]]:
        rerank_vals = [score for _, score in reranked]
        faiss_vals = [faiss_scores.get(id(doc), 0.0) for doc, _ in reranked]
        norm_rerank = self._normalize(rerank_vals)
        norm_faiss = self._normalize(faiss_vals)
        fused: list[tuple[Document, float]] = []
        for idx, (doc, _) in enumerate(reranked):
            # Prefer semantic rerank, keep some dense-retrieval signal.
            score = 0.7 * norm_rerank[idx] + 0.3 * norm_faiss[idx]
            fused.append((doc, float(score)))
        fused.sort(key=lambda item: item[1], reverse=True)
        return fused

    @staticmethod
    def _dedupe_hits(hits: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        seen_keys: set[str] = set()
        deduped: list[tuple[Document, float]] = []
        for doc, score in hits:
            doc_id = str(doc.metadata.get("doc_id", ""))
            fingerprint = re.sub(r"\s+", " ", doc.page_content[:240].lower()).strip()
            key = f"{doc_id}|{fingerprint[:120]}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append((doc, score))
        return deduped

    @staticmethod
    def to_citations(hits: list[tuple[Document, float]], snippet_chars: int = 320) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for doc, score in hits:
            doc_id = str(doc.metadata.get("doc_id", ""))
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            year = doc.metadata.get("year")
            try:
                year_i = int(year) if year not in (None, "", -1, "-1") else None
            except (TypeError, ValueError):
                year_i = None
            citations.append(
                Citation(
                    doc_id=doc_id,
                    title=str(doc.metadata.get("title", "")),
                    citation_url=str(doc.metadata.get("citation_url", "")),
                    snippet=doc.page_content[:snippet_chars].replace("\n", " ").strip(),
                    score=float(score),
                    feature_tags=str(doc.metadata.get("feature_tags", "")),
                    center_code=str(doc.metadata.get("center_code", "")),
                    year=year_i,
                )
            )
        return citations

    @staticmethod
    def format_context(hits: list[tuple[Document, float]]) -> str:
        blocks = []
        for idx, (doc, score) in enumerate(hits, start=1):
            blocks.append(
                f"[{idx}] doc_id={doc.metadata.get('doc_id')} "
                f"title={doc.metadata.get('title')} score={score:.3f}\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def context_texts(hits: list[tuple[Document, float]]) -> list[str]:
        return [doc.page_content for doc, _ in hits]
