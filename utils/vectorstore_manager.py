from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from exception.exceptions import ReportsException
from logger.logger import get_app_logger
from models.models import IndexBuildManifest, utc_now_iso
from utils.bm25_index import BM25Index
from utils.config_loader import get_config_loader
from utils.document_pipeline import DocumentLoader, TextChunker
from utils.model_loader import ModelLoader

_CHUNK_KEY = re.compile(r"\s+")


class VectorStoreManager:
    """HNSW FAISS + BM25 hybrid vector store with metadata filters."""

    def __init__(self, model_loader: ModelLoader | None = None):
        self.log = get_app_logger("vectorstore_manager")
        self.loader = get_config_loader()
        self.model_loader = model_loader or ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.store_path = self.loader.resolve_path(self.loader.get("faiss_db", "path"))
        self.manifest_path = self.store_path.parent / "build_manifest.json"
        self._store: FAISS | None = None
        self._bm25: BM25Index | None = None

    @property
    def store(self) -> FAISS:
        if self._store is None:
            self._store = self.load()
        return self._store

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            index = BM25Index()
            if not index.load(self.store_path):
                index.build([])
            self._bm25 = index
        return self._bm25

    def exists(self) -> bool:
        return (self.store_path / "index.faiss").exists()

    def load(self) -> FAISS:
        if not self.exists():
            raise ReportsException(
                f"Vectorstore not found at {self.store_path}. Run scripts/build_vectorstore.py first."
            )
        self.log.info("Loading FAISS vectorstore", path=str(self.store_path))
        self._store = FAISS.load_local(
            str(self.store_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._prepare_hnsw_search()
        return self._store

    def _faiss_cfg(self) -> dict[str, Any]:
        return self.loader.get("faiss_db", default={}) or {}

    def _prepare_hnsw_search(self) -> None:
        if self._store is None:
            return
        idx = self._store.index
        if hasattr(idx, "hnsw"):
            idx.hnsw.efSearch = int(self._faiss_cfg().get("hnsw_ef_search", 64))

    def _build_hnsw_store(self, chunks: list[Document]) -> FAISS:
        cfg = self._faiss_cfg()
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectors: list[list[float]] = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self.embeddings.embed_documents(batch))
            if start == 0 or (start // batch_size) % 10 == 0:
                self.log.info("Embedding progress", done=min(start + batch_size, len(texts)), total=len(texts))
        arr = np.asarray(vectors, dtype=np.float32)
        dim = arr.shape[1]

        if cfg.get("use_hnsw", True):
            m = int(cfg.get("hnsw_m", 32))
            index = faiss.IndexHNSWFlat(dim, m)
            index.hnsw.efConstruction = int(cfg.get("hnsw_ef_construction", 200))
            index_type = "hnsw"
        else:
            index = faiss.IndexFlatL2(dim)
            index_type = "flat"

        index.add(arr)
        ids = [f"chunk-{i}" for i in range(len(chunks))]
        docstore = InMemoryDocstore(
            {
                ids[i]: Document(page_content=texts[i], metadata=metadatas[i])
                for i in range(len(chunks))
            }
        )
        index_to_docstore_id = {i: ids[i] for i in range(len(chunks))}
        self.log.info("Dense index built", index_type=index_type, chunks=len(chunks), dim=dim)
        return FAISS(
            self.embeddings,
            index,
            docstore,
            index_to_docstore_id,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

    def build(self, force: bool = False) -> IndexBuildManifest:
        if self.exists() and not force:
            self.log.info("Vectorstore already exists; use force=True to rebuild")
            return self.read_manifest()

        doc_loader = DocumentLoader()
        chunker = TextChunker()
        max_chars = int(self.loader.get("chunking", "max_chars_per_doc", default=12000))
        documents = doc_loader.iter_corpus_documents(max_chars_per_doc=max_chars)
        if not documents:
            raise ReportsException("No corpus documents found to index")
        chunks = chunker.split(documents)
        self.log.info("Building hybrid index", docs=len(documents), chunks=len(chunks))

        store = self._build_hnsw_store(chunks)
        self.store_path.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.store_path))
        self._store = store

        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save(self.store_path)
        self._bm25 = bm25

        corpus_hash = hashlib_corpus(documents)
        hybrid_cfg = self.loader.get("retriever", "hybrid", default={}) or {}
        manifest = IndexBuildManifest(
            created_at=utc_now_iso(),
            doc_count=len(documents),
            chunk_count=len(chunks),
            embedding_model=self.loader.get("embedding_model", "model_name", default=""),
            chunk_size=chunker.chunk_size,
            chunk_overlap=chunker.chunk_overlap,
            corpus_hash=corpus_hash,
            path=str(self.store_path),
            index_type="hnsw" if self._faiss_cfg().get("use_hnsw", True) else "flat",
            bm25_enabled=True,
            hybrid_search=bool(hybrid_cfg.get("enabled", True)),
        )
        self.manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        self.log.info("Hybrid build complete", **manifest.model_dump())
        return manifest

    def read_manifest(self) -> IndexBuildManifest:
        if not self.manifest_path.exists():
            return IndexBuildManifest(
                created_at="",
                doc_count=0,
                chunk_count=0,
                embedding_model="",
                chunk_size=0,
                chunk_overlap=0,
                corpus_hash="",
                path=str(self.store_path),
            )
        return IndexBuildManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _chunk_key(doc: Document) -> str:
        doc_id = str(doc.metadata.get("doc_id", ""))
        fp = _CHUNK_KEY.sub(" ", doc.page_content[:220].lower()).strip()
        return f"{doc_id}|{fp[:100]}"

    @staticmethod
    def _rrf_fuse(
        dense: list[tuple[Document, float]],
        sparse: list[tuple[Document, float]],
        *,
        k: int,
        rrf_k: int,
    ) -> list[tuple[Document, float]]:
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for rank, (doc, _) in enumerate(dense):
            key = VectorStoreManager._chunk_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, (doc, _) in enumerate(sparse):
            key = VectorStoreManager._chunk_key(doc)
            docs[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(docs[key], float(score)) for key, score in ordered[:k]]

    def hybrid_search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        top_k = k or int(self.loader.get("retriever", "top_k", default=6))
        hybrid_cfg = self.loader.get("retriever", "hybrid", default={}) or {}
        use_hybrid = hybrid_cfg.get("enabled", True) and (
            self.store_path / "bm25_corpus.jsonl"
        ).exists()
        if not use_hybrid:
            return self.similarity_search(query, k=top_k, filters=filters)

        fetch = max(top_k * 4, top_k)
        self._prepare_hnsw_search()
        dense_raw = self.store.similarity_search_with_score(query, k=min(fetch * 3, 500))
        dense: list[tuple[Document, float]] = []
        for doc, score in dense_raw:
            if self._passes_filters(doc.metadata, filters or {}):
                dense.append((doc, float(1.0 / (1.0 + float(score)))))
        sparse = self.bm25.search(
            query,
            k=min(fetch * 3, 500),
            filter_fn=lambda meta: self._passes_filters(meta, filters or {}),
        )
        rrf_k = int(hybrid_cfg.get("rrf_k", 60))
        fused = self._rrf_fuse(dense, sparse, k=fetch, rrf_k=rrf_k)[:top_k]
        self.log.info(
            "Hybrid search complete",
            query_preview=query[:120],
            requested_k=top_k,
            dense_candidates=len(dense),
            sparse_candidates=len(sparse),
            returned=len(fused),
            rrf_k=rrf_k,
        )
        return fused

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        top_k = k or int(self.loader.get("retriever", "top_k", default=6))
        self._prepare_hnsw_search()
        results = self.store.similarity_search_with_score(query, k=max(top_k * 4, top_k))
        filtered: list[tuple[Document, float]] = []
        for doc, score in results:
            if self._passes_filters(doc.metadata, filters or {}):
                sim = float(1.0 / (1.0 + float(score)))
                filtered.append((doc, sim))
            if len(filtered) >= top_k:
                break
        return filtered

    @staticmethod
    def _passes_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        if not filters:
            return True
        feature_tags = filters.get("feature_tags")
        if feature_tags:
            tags = {t.strip() for t in str(metadata.get("feature_tags", "")).split(",") if t.strip()}
            wanted = set(feature_tags) if isinstance(feature_tags, list) else {str(feature_tags)}
            if tags.isdisjoint(wanted):
                return False
        center = filters.get("center_code")
        if center and str(metadata.get("center_code", "")).upper() != str(center).upper():
            return False
        year_from = filters.get("year_from")
        year_to = filters.get("year_to")
        year = metadata.get("year", -1)
        try:
            year_i = int(year)
        except (TypeError, ValueError):
            year_i = -1
        if year_from is not None and year_i != -1 and year_i < int(year_from):
            return False
        if year_to is not None and year_i != -1 and year_i > int(year_to):
            return False
        return True


def hashlib_corpus(documents: list[Document]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: str(d.metadata.get("doc_id", ""))):
        digest.update(str(doc.metadata.get("doc_id", "")).encode())
        digest.update(str(doc.metadata.get("content_hash", "")).encode())
    return digest.hexdigest()[:16]
