from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from exception.exceptions import ReportsException
from logger.logger import get_app_logger
from utils.config_loader import get_config_loader


class DocumentLoader:
    """Load NASA PDFs with sidecar metadata."""

    def __init__(self):
        self.log = get_app_logger("document_loader")
        self.loader = get_config_loader()
        self.pdf_dir = self.loader.resolve_path(self.loader.get("corpus", "pdf_dir"))
        self.metadata_dir = self.loader.resolve_path(self.loader.get("corpus", "metadata_dir"))
        self.metadata_jsonl = self.loader.resolve_path(self.loader.get("corpus", "metadata_jsonl"))

    def load_metadata_index(self) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        if self.metadata_jsonl.exists():
            for line in self.metadata_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = str(row.get("doc_id") or row.get("id") or "")
                if doc_id:
                    index[doc_id] = row
        for path in self.metadata_dir.glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            doc_id = str(row.get("doc_id") or path.stem)
            index[doc_id] = row
        self.log.info("Metadata index loaded", count=len(index))
        return index

    def extract_pdf_text(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                parts = [page.get_text("text") for page in doc]
            return "\n".join(parts).strip()
        except Exception as exc:
            raise ReportsException(f"Failed to read PDF {pdf_path.name}: {exc}") from exc

    def iter_corpus_documents(self, max_chars_per_doc: int = 12000) -> list[Document]:
        meta_index = self.load_metadata_index()
        documents: list[Document] = []
        for pdf_path in sorted(self.pdf_dir.glob("*.pdf")):
            doc_id = pdf_path.name.split("_", 1)[0]
            meta = meta_index.get(doc_id, {})
            text = self.extract_pdf_text(pdf_path)
            if not text:
                self.log.info("Skipping empty PDF", pdf=pdf_path.name)
                continue
            abstract = str(meta.get("abstract") or "").strip()
            # Prefer abstract + leading body for scalable indexing of large technical PDFs.
            body = text[:max_chars_per_doc]
            combined = f"Title: {meta.get('title') or pdf_path.stem}\n\nAbstract: {abstract}\n\nBody:\n{body}".strip()
            feature_tags = meta.get("feature_tags") or []
            if isinstance(feature_tags, list):
                feature_tags_str = ",".join(str(t) for t in feature_tags)
            else:
                feature_tags_str = str(feature_tags)
            metadata = {
                "doc_id": doc_id,
                "title": meta.get("title") or pdf_path.stem,
                "source": meta.get("source") or "NASA NTRS",
                "citation_url": meta.get("citation_url") or f"https://ntrs.nasa.gov/citations/{doc_id}",
                "sti_type": meta.get("sti_type") or "",
                "center_code": meta.get("center_code") or "",
                "center_name": meta.get("center_name") or "",
                "year": meta.get("year") if meta.get("year") is not None else -1,
                "feature_tags": feature_tags_str,
                "query_tag": meta.get("query_tag") or "",
                "subject_categories": ",".join(meta.get("subject_categories") or []),
                "keywords": ",".join(meta.get("keywords") or []),
                "pdf_filename": meta.get("pdf_filename") or pdf_path.name,
                "content_hash": hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16],
            }
            documents.append(Document(page_content=combined, metadata=metadata))
        self.log.info("Corpus documents loaded", count=len(documents), max_chars_per_doc=max_chars_per_doc)
        return documents

    def load_upload(self, pdf_path: Path, session_id: str) -> Document:
        text = self.extract_pdf_text(pdf_path)
        if not text:
            raise ReportsException(f"Uploaded PDF has no extractable text: {pdf_path.name}")
        return Document(
            page_content=text,
            metadata={
                "doc_id": f"upload-{session_id}-{pdf_path.stem}",
                "title": pdf_path.name,
                "source": "user_upload",
                "citation_url": "",
                "sti_type": "UPLOAD",
                "center_code": "",
                "center_name": "",
                "year": -1,
                "feature_tags": "upload",
                "query_tag": "upload",
                "subject_categories": "",
                "keywords": "",
                "pdf_filename": pdf_path.name,
                "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            },
        )


class TextChunker:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.log = get_app_logger("text_chunker")
        loader = get_config_loader()
        self.chunk_size = chunk_size or int(loader.get("chunking", "size", default=1000))
        self.chunk_overlap = chunk_overlap or int(loader.get("chunking", "overlap", default=150))
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'doc')}-{idx}"
        self.log.info("Documents chunked", input_docs=len(documents), chunks=len(chunks))
        return chunks
