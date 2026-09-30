from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from logger.logger import get_app_logger
from utils.config_loader import get_config_loader


class EvaluationArtifactsService:
    """Load evaluation run artifacts for metrics UI / LLMOps dashboards."""

    def __init__(self):
        self.log = get_app_logger("evaluation_artifacts")
        self.loader = get_config_loader()
        self.runs_dir = self.loader.resolve_path(
            self.loader.get("evaluation", "runs_dir", default="data/evaluation/runs")
        )

    def list_runs(self) -> list[dict[str, Any]]:
        if not self.runs_dir.exists():
            return []
        runs = []
        for path in sorted(self.runs_dir.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            manifest_path = path / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            runs.append(
                {
                    "run_id": path.name,
                    "created_at": manifest.get("created_at"),
                    "sample_count": manifest.get("sample_count"),
                    "passed": manifest.get("passed"),
                    "aggregate_metrics": manifest.get("aggregate_metrics", {}),
                    "model_name": manifest.get("model_name"),
                    "embedding_model": manifest.get("embedding_model"),
                    "experiment": bool(manifest.get("experiment")),
                    "cell_count": manifest.get("cell_count"),
                }
            )
        self.log.info("Listed evaluation runs", count=len(runs))
        return runs

    def load_run(self, run_id: str | None = None) -> dict[str, Any] | None:
        runs = self.list_runs()
        if not runs:
            return None
        target = run_id or runs[0]["run_id"]
        run_dir = self.runs_dir / target
        if not run_dir.exists():
            return None
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        metrics = {}
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        report = ""
        report_path = run_dir / "report.md"
        if report_path.exists():
            report = report_path.read_text(encoding="utf-8")
        payload = {
            "run_id": target,
            "manifest": manifest,
            "metrics": metrics,
            "report_markdown": report,
            "thresholds": manifest.get("thresholds")
            or self.loader.get("evaluation", "thresholds", default={}),
        }
        self.log.info("Loaded evaluation run", run_id=target)
        return payload
