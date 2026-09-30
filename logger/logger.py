from __future__ import annotations

import logging
import os
from datetime import datetime
from functools import lru_cache

import structlog


class CustomLogger:
    """Production structured logger factory (JSON console + file)."""

    _configured = False
    _log_file_path: str | None = None

    def __init__(self, log_dir: str = "logs"):
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        if CustomLogger._log_file_path is None:
            log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            CustomLogger._log_file_path = os.path.join(self.logs_dir, log_file)
        self.log_file_path = CustomLogger._log_file_path

    def get_logger(self, name: str = __file__):
        logger_name = os.path.basename(str(name))
        if not CustomLogger._configured:
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            root = logging.getLogger()
            root.handlers.clear()
            root.setLevel(logging.INFO)
            root.addHandler(console_handler)
            root.addHandler(file_handler)

            structlog.configure(
                processors=[
                    structlog.contextvars.merge_contextvars,
                    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    structlog.processors.add_log_level,
                    structlog.processors.EventRenamer(to="event"),
                    structlog.processors.JSONRenderer(),
                ],
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            CustomLogger._configured = True

        return structlog.get_logger(logger_name)


@lru_cache(maxsize=64)
def get_app_logger(name: str = "app"):
    return CustomLogger().get_logger(name)
