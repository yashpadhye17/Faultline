"""Sparse BM25 index persisted alongside the dense vector store."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from logger.logger import get_app_logger

log = get_app_logger("bm25_index")

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


class BM25Index:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def build(self, documents: list[Document]) -> None:
        self.documents = list(documents)
        corpus = [tokenize(doc.page_content) for doc in self.documents]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        log.info("BM25 index built", chunks=len(self.documents))

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "bm25_corpus.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for doc in self.documents:
                row = {"page_content": doc.page_content, "metadata": doc.metadata}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.info("BM25 corpus saved", path=str(path), chunks=len(self.documents))

    def load(self, directory: Path) -> bool:
        path = directory / "bm25_corpus.jsonl"
        if not path.exists():
            return False
        documents: list[Document] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            documents.append(
                Document(page_content=row["page_content"], metadata=row.get("metadata") or {})
            )
        self.build(documents)
        return True

    def search(
        self,
        query: str,
        k: int,
        *,
        filter_fn: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[tuple[Document, float]]:
        if not self._bm25 or not self.documents:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        hits: list[tuple[Document, float]] = []
        for idx in ranked:
            if float(scores[idx]) <= 0:
                break
            doc = self.documents[idx]
            if filter_fn and not filter_fn(doc.metadata):
                continue
            hits.append((doc, float(scores[idx])))
            if len(hits) >= k:
                break
        return hits
