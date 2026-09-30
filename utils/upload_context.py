from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from langchain_core.documents import Document

from exception.exceptions import ReportsException
from logger.logger import get_app_logger
from utils.config_loader import get_config_loader
from utils.document_pipeline import DocumentLoader, TextChunker


class UploadContextManager:
    """Session-scoped optional PDF uploads (never mutates NASA FAISS)."""

    def __init__(self):
        self.log = get_app_logger("upload_context")
        self.loader = get_config_loader()
        self.temp_dir = self.loader.resolve_path(
            self.loader.get("uploads", "temp_dir", default="data/uploads")
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_mb = float(self.loader.get("uploads", "max_mb", default=25))
        self.allowed = set(self.loader.get("uploads", "allowed_extensions", default=[".pdf"]))
        self.doc_loader = DocumentLoader()
        self.chunker = TextChunker()
        self._sessions: dict[str, list[Document]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        (self.temp_dir / session_id).mkdir(parents=True, exist_ok=True)
        self.log.info("Upload session created", session_id=session_id)
        return session_id

    def add_pdf(self, session_id: str, filename: str, content: bytes) -> list[Document]:
        if not session_id:
            session_id = self.create_session()
        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed:
            raise ReportsException(f"Unsupported upload type: {suffix}")
        size_mb = len(content) / (1024 * 1024)
        if size_mb > self.max_mb:
            raise ReportsException(f"Upload exceeds {self.max_mb} MB limit")

        session_dir = self.temp_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        dest = session_dir / Path(filename).name
        dest.write_bytes(content)
        doc = self.doc_loader.load_upload(dest, session_id)
        chunks = self.chunker.split([doc])
        self._sessions.setdefault(session_id, []).extend(chunks)
        self.log.info(
            "Upload ingested",
            session_id=session_id,
            filename=filename,
            chunks=len(chunks),
        )
        return chunks

    def get_chunks(self, session_id: str | None) -> list[Document]:
        if not session_id:
            return []
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        session_dir = self.temp_dir / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        self.log.info("Upload session cleared", session_id=session_id)
