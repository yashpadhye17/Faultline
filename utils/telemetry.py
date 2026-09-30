from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from logger.logger import get_app_logger


class TelemetryService:
    """Request-scoped telemetry helpers for LLMOps logging."""

    def __init__(self, component: str = "telemetry"):
        self.log = get_app_logger(component)

    def new_request_id(self) -> str:
        return str(uuid.uuid4())

    def bind(self, **kwargs: Any) -> None:
        structlog.contextvars.bind_contextvars(**kwargs)

    def clear(self) -> None:
        structlog.contextvars.clear_contextvars()

    def timed(self) -> "_Timer":
        return _Timer()

    def info(self, event: str, **kwargs: Any) -> None:
        self.log.info(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self.log.error(event, **kwargs)


class _Timer:
    def __init__(self):
        self._start = time.perf_counter()

    @property
    def latency_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)
