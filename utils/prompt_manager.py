from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from exception.exceptions import ReportsException
from logger.logger import get_app_logger
from utils.config_loader import PROJECT_ROOT, get_config_loader


class PromptManager:
    """Versioned prompt registry backed by prompts/manifest.yaml and template files."""

    def __init__(self, manifest_path: str | Path | None = None):
        self.log = get_app_logger("prompt_manager")
        loader = get_config_loader()
        rel = manifest_path or loader.get("llmops", "prompts_manifest", default="prompts/manifest.yaml")
        self.manifest_path = loader.resolve_path(rel) if not isinstance(rel, Path) else rel
        if not self.manifest_path.is_absolute():
            self.manifest_path = PROJECT_ROOT / self.manifest_path
        self.manifest = self._load_manifest()
        self.log.info("PromptManager initialized", prompts=list(self.manifest.get("prompts", {}).keys()))

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise ReportsException(f"Prompt manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def get(self, prompt_id: str, version: str | None = None) -> dict[str, Any]:
        prompts = self.manifest.get("prompts", {})
        if prompt_id not in prompts:
            raise ReportsException(f"Unknown prompt_id: {prompt_id}")
        meta = prompts[prompt_id]
        use_version = version or meta.get("version", "v1")
        template_rel = meta.get("path")
        template_path = PROJECT_ROOT / template_rel
        if not template_path.exists():
            raise ReportsException(f"Prompt template missing: {template_path}")
        text = template_path.read_text(encoding="utf-8")
        self.log.info("Prompt loaded", prompt_id=prompt_id, version=use_version)
        return {
            "prompt_id": prompt_id,
            "version": use_version,
            "template": text,
            "description": meta.get("description", ""),
        }

    def render(self, prompt_id: str, **kwargs: Any) -> tuple[str, str, str]:
        payload = self.get(prompt_id)
        try:
            rendered = payload["template"].format(**kwargs)
        except KeyError as exc:
            raise ReportsException(f"Missing prompt variable {exc} for {prompt_id}") from exc
        return rendered, payload["prompt_id"], payload["version"]
