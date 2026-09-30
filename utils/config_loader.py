from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigLoader:
    """Load and resolve project configuration paths."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path
        self._config = self._read()

    def _read(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def resolve_path(self, relative: str | Path) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self._config
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    return ConfigLoader()


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        return get_config_loader().config
    return ConfigLoader(config_path).config


loaded_config = load_config()
