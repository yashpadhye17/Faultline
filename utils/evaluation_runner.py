"""Offline evaluation runner (RAGAS-oriented deterministic metrics)."""

from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from logger.logger import get_app_logger
from models.models import (
    ChatRequest,
    ComparisonRequest,
    EvalSample,
    EvaluationRunManifest,
    FeatureName,
    RCARequest,
    ResearchRequest,
    utc_now_iso,
)
from src.document_analyze.service import RootCauseAnalysisService
from src.document_comparison.service import ComparisonService
from src.multidoc_chat.service import ResearchCopilotService
from src.single_docchat.service import MaintenanceCopilotService
from utils.config_loader import get_config_loader
from utils.model_loader import ModelLoader
from utils.retriever_service import RetrieverService
from utils.vectorstore_manager import VectorStoreManager

log = get_app_logger("evaluation_runner")


def tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def overlap_score(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def faithfulness(answer: str, contexts: list[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    ctx = " ".join(contexts)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        return 0.0
    supported = sum(1 for s in sentences if overlap_score(s, ctx) >= 0.05)
    return supported / len(sentences)


def context_precision(question: str, contexts: list[str], embed_fn) -> float:
    if not contexts:
        return 0.0
    q = embed_fn(question)
    scores = [cosine(q, embed_fn(c)) for c in contexts]
    return sum(1 for s in scores if s >= 0.25) / len(scores)


def context_recall(ground_truth: str, contexts: list[str], embed_fn) -> float | None:
    if not ground_truth.strip():
        return None
    if not contexts:
        return 0.0
    return cosine(embed_fn(ground_truth), embed_fn(" ".join(contexts)))


@dataclass
class ExperimentConfig:
    feature: str = "all"
    limit: int | None = None
    llm_models: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    temperature: float | None = None
    top_k: int | None = None
    fetch_k: int | None = None
    reranker_enabled: bool | None = None


class EvaluationRunner:
    """Run gold-set queries through feature services and score RAGAS-oriented metrics."""

    def __init__(
        self,
        *,
        llm_model: str | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        fetch_k: int | None = None,
    ):
        self.loader = get_config_loader()
        self.llm_model = llm_model or str(self.loader.get("llm", "model_name"))
        self.embedding_model = str(self.loader.get("embedding_model", "model_name"))
        self.temperature = (
            temperature
            if temperature is not None
            else float(self.loader.get("llm", "temperature", default=0.2))
        )
        self.top_k = top_k if top_k is not None else int(self.loader.get("retriever", "top_k", default=6))
        self.fetch_k = (
            fetch_k if fetch_k is not None else int(self.loader.get("retriever", "fetch_k", default=20))
        )

        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings(self.embedding_model)
        # Retrieval index stays on the built FAISS corpus embedding.
        self.vectorstore = VectorStoreManager(ModelLoader())
        self.retriever = RetrieverService(self.vectorstore)
        self.retriever.default_top_k = self.top_k
        self.retriever.default_fetch_k = self.fetch_k

        self.rca = RootCauseAnalysisService(
            model_loader=self.model_loader, vectorstore=self.vectorstore, retriever=self.retriever
        )
        self.maintenance = MaintenanceCopilotService(
            model_loader=self.model_loader, vectorstore=self.vectorstore, retriever=self.retriever
        )
        self.comparison = ComparisonService(
            model_loader=self.model_loader, vectorstore=self.vectorstore, retriever=self.retriever
        )
        self.research = ResearchCopilotService(
            model_loader=self.model_loader, vectorstore=self.vectorstore, retriever=self.retriever
        )
        llm = self.model_loader.load_llm(self.llm_model, temperature=self.temperature)
        for svc in (self.rca, self.maintenance, self.comparison, self.research):
            svc.llm = llm

        self._embed_cache: dict[str, list[float]] = {}

    def embed(self, text: str) -> list[float]:
        key = text[:2000]
        if key not in self._embed_cache:
            self._embed_cache[key] = list(self.embeddings.embed_query(text[:2000]))
        return self._embed_cache[key]

    def load_dataset(self, path: Path) -> list[EvalSample]:
        samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(EvalSample.model_validate_json(line))
        return samples

    def _retrieve(self, query: str, feature: str):
        return self.retriever.retrieve(
            query, feature=feature, k=self.top_k, fetch_k=self.fetch_k
        )

    def run_sample(self, sample: EvalSample) -> dict:
        if sample.feature == FeatureName.RCA:
            result = self.rca.analyze(RCARequest(incident_description=sample.question))
            answer = result.raw_answer or result.summary
            citations = result.citations
            hits = self._retrieve(sample.question, "rca")
            contexts = RetrieverService.context_texts(hits)
            run_meta = result.run_meta
        elif sample.feature == FeatureName.MAINTENANCE:
            result = self.maintenance.chat(ChatRequest(message=sample.question))
            answer = result.answer
            citations = result.citations
            hits = self._retrieve(sample.question, "maintenance")
            contexts = RetrieverService.context_texts(hits)
            run_meta = result.run_meta
        elif sample.feature == FeatureName.COMPARISON:
            parts = [p.strip() for p in sample.question.split(" vs ", 1)]
            topic_a = parts[0] if parts else sample.question
            topic_b = parts[1] if len(parts) > 1 else "baseline approach"
            result = self.comparison.compare(ComparisonRequest(topic_a=topic_a, topic_b=topic_b))
            answer = result.raw_answer or result.summary
            citations = result.citations
            hits = self._retrieve(f"{topic_a} versus {topic_b}", "comparison")
            contexts = RetrieverService.context_texts(hits)
            run_meta = result.run_meta
        else:
            result = self.research.ask(ResearchRequest(question=sample.question))
            answer = result.answer
            citations = result.citations
            hits = self._retrieve(sample.question, "research")
            contexts = RetrieverService.context_texts(hits)
            run_meta = result.run_meta

        # Stamp selected experiment models onto run meta for artifact transparency.
        run_meta.model_name = self.llm_model
        run_meta.embedding_model = self.embedding_model

        all_metrics = {
            "faithfulness": faithfulness(answer, contexts),
            "answer_relevancy": cosine(self.embed(sample.question), self.embed(answer)),
            "context_precision": context_precision(sample.question, contexts, self.embed),
            "context_recall": context_recall(sample.ground_truth, contexts, self.embed),
            "answer_correctness": (
                cosine(self.embed(sample.ground_truth), self.embed(answer))
                if sample.ground_truth.strip()
                else None
            ),
        }
        return {
            "sample_id": sample.sample_id,
            "feature": sample.feature.value,
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "answer": answer,
            "contexts": contexts,
            "citations": [c.model_dump() for c in citations],
            "metrics": all_metrics,
            "run_meta": run_meta.model_dump(),
            "llm_model": self.llm_model,
        }

    def run(
        self,
        dataset_path: Path,
        feature: str | None,
        limit: int | None,
        *,
        metric_keys: list[str] | None = None,
        run_id: str | None = None,
        parent_experiment_id: str | None = None,
    ) -> Path:
        samples = self.load_dataset(dataset_path)
        if feature and feature != "all":
            samples = [s for s in samples if s.feature.value == feature]
        if limit:
            samples = samples[:limit]

        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        runs_dir = self.loader.resolve_path(self.loader.get("evaluation", "runs_dir"))
        out_dir = runs_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        predictions = []
        for sample in samples:
            log.info(
                "Evaluating sample",
                sample_id=sample.sample_id,
                feature=sample.feature.value,
                llm=self.llm_model,
            )
            predictions.append(self.run_sample(sample))

        available = [
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]
        keys = [k for k in (metric_keys or available) if k in available] or available
        aggregates: dict[str, float] = {}
        for key in keys:
            vals = [p["metrics"][key] for p in predictions if p["metrics"].get(key) is not None]
            aggregates[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

        thresholds = self.loader.get("evaluation", "thresholds", default={}) or {}
        gated = {k: float(v) for k, v in thresholds.items() if k in aggregates}
        passed = all(aggregates.get(k, 0) >= thr for k, thr in gated.items()) if gated else True

        manifest = EvaluationRunManifest(
            run_id=run_id,
            created_at=utc_now_iso(),
            dataset_path=str(dataset_path),
            model_name=self.llm_model,
            embedding_model=self.embedding_model,
            top_k=self.top_k,
            chunk_size=int(self.loader.get("chunking", "size", default=1000)),
            sample_count=len(predictions),
            aggregate_metrics=aggregates,
            thresholds=gated,
            passed=passed,
        )
        manifest_payload = json.loads(manifest.model_dump_json())
        manifest_payload.update(
            {
                "retrieval_note": (
                    "Retrieval + metrics use the same corpus embedding model configured in config.yaml."
                ),
                "hybrid_search": bool(
                    (self.loader.get("retriever", "hybrid", default={}) or {}).get("enabled", True)
                ),
                "reranker_model": str(
                    (self.loader.get("retriever", "reranker", default={}) or {}).get("model_name", "")
                ),
                "temperature": self.temperature,
                "fetch_k": self.fetch_k,
                "parent_experiment_id": parent_experiment_id,
                "selected_metrics": keys,
            }
        )

        (out_dir / "manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        scoring_notes = {
            "answer_relevancy": "cosine(question, answer) via corpus embedding model",
            "context_precision": "fraction of contexts with cosine(question, context) >= 0.25",
            "context_recall": "cosine(ground_truth, joined_contexts)",
            "answer_correctness": "cosine(ground_truth, answer)",
        }
        metrics_payload = {
            "aggregate": aggregates,
            "per_sample": [
                {"sample_id": p["sample_id"], **{k: p["metrics"].get(k) for k in keys}}
                for p in predictions
            ],
            "passed": passed,
            "thresholds": gated,
            "llm_model": self.llm_model,
            "scoring": {k: scoring_notes[k] for k in keys if k in scoring_notes},
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        (out_dir / "report.md").write_text(
            self._render_report(manifest_payload, metrics_payload), encoding="utf-8"
        )
        log.info("Evaluation complete", run_id=run_id, passed=passed, aggregates=aggregates)
        return out_dir

    @staticmethod
    def _render_report(manifest: dict, metrics_payload: dict) -> str:
        lines = [
            f"# EIRC Evaluation Report — {manifest['run_id']}",
            "",
            f"- Created: `{manifest.get('created_at')}`",
            f"- Dataset: `{manifest.get('dataset_path')}`",
            f"- Samples: **{manifest.get('sample_count')}**",
            f"- Generation LLM: `{manifest.get('model_name')}`",
            f"- Corpus embedding: `{manifest.get('embedding_model')}`",
            f"- top_k: `{manifest.get('top_k')}` · fetch_k: `{manifest.get('fetch_k')}` · temp: `{manifest.get('temperature')}`",
            f"- Gate passed: **{manifest.get('passed')}**",
            "",
            "## Aggregate metrics",
            "",
            "| Metric | Score | Threshold |",
            "|---|---:|---:|",
        ]
        for key, value in (manifest.get("aggregate_metrics") or {}).items():
            thr = (manifest.get("thresholds") or {}).get(key, "")
            lines.append(f"| {key} | {value:.4f} | {thr} |")
        lines.extend(
            [
                "",
                "## What each metric means",
                "",
                "- **Faithfulness** — how grounded answer sentences are in retrieved context (anti-hallucination).",
                "- **Answer relevancy** — semantic similarity of answer to the question.",
                "- **Context precision** — fraction of retrieved chunks that are on-topic for the question.",
                "- **Context recall** — how well retrieved context covers the gold ground-truth.",
                "- **Answer correctness** — semantic similarity of answer to gold ground-truth.",
                "",
                "## Scoring notes",
                "",
                "- Retrieval uses the prebuilt FAISS + reranker pipeline.",
                "- Cosine metrics use the same corpus embedding model as retrieval indexing.",
                "",
            ]
        )
        return "\n".join(lines)

    @classmethod
    def run_experiment(cls, config: ExperimentConfig) -> Path:
        """Compare multiple generation LLMs on the gold set (single corpus embedding)."""
        loader = get_config_loader()
        dataset = loader.resolve_path(loader.get("evaluation", "dataset_path"))
        sample_limit = config.limit or int(loader.get("evaluation", "sample_limit", default=8))
        default_llm = str(loader.get("llm", "model_name"))
        llms = config.llm_models or [default_llm]
        metric_keys = config.metrics or list(loader.get("evaluation", "metrics", default=[]) or [])
        embedding_model = str(loader.get("embedding_model", "model_name"))

        experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_experiment"
        runs_dir = loader.resolve_path(loader.get("evaluation", "runs_dir"))
        exp_dir = runs_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        cells = []
        for llm in llms:
            cell_id = f"{experiment_id}__{cls._slug(llm)}"
            log.info("Experiment cell", llm=llm, cell_id=cell_id)
            runner = cls(
                llm_model=llm,
                temperature=config.temperature,
                top_k=config.top_k,
                fetch_k=config.fetch_k,
            )
            out = runner.run(
                dataset,
                config.feature,
                sample_limit,
                metric_keys=metric_keys,
                run_id=cell_id,
                parent_experiment_id=experiment_id,
            )
            metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            cells.append(
                {
                    "run_id": cell_id,
                    "llm_model": llm,
                    "aggregate": metrics.get("aggregate", {}),
                    "passed": metrics.get("passed"),
                    "path": str(out),
                }
            )

        summary = {
            "experiment_id": experiment_id,
            "created_at": utc_now_iso(),
            "config": asdict(config),
            "cells": cells,
            "best_by_metric": cls._best_by_metric(cells),
            "embedding_model": embedding_model,
        }
        (exp_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": experiment_id,
                    "created_at": summary["created_at"],
                    "sample_count": sample_limit,
                    "passed": all(c.get("passed") for c in cells) if cells else False,
                    "aggregate_metrics": {},
                    "model_name": ",".join(llms),
                    "embedding_model": embedding_model,
                    "experiment": True,
                    "cell_count": len(cells),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (exp_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (exp_dir / "report.md").write_text(cls._render_experiment_report(summary), encoding="utf-8")
        return exp_dir

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")[:48]

    @staticmethod
    def _best_by_metric(cells: list[dict]) -> dict[str, dict]:
        best: dict[str, dict] = {}
        for cell in cells:
            for metric, score in (cell.get("aggregate") or {}).items():
                cur = best.get(metric)
                if cur is None or float(score) > float(cur["score"]):
                    best[metric] = {
                        "score": score,
                        "llm_model": cell["llm_model"],
                        "run_id": cell["run_id"],
                    }
        return best

    @staticmethod
    def _render_experiment_report(summary: dict) -> str:
        lines = [
            f"# EIRC Multi-Model Experiment — {summary['experiment_id']}",
            "",
            "Compares generation LLMs on the gold evaluation set (shared BM25+HNSW retrieval + reranker).",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(summary.get("config") or {}, indent=2),
            "```",
            "",
            "## Matrix results",
            "",
            "| LLM | Relevancy | Ctx precision | Ctx recall | Correctness | Passed |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for cell in summary.get("cells") or []:
            agg = cell.get("aggregate") or {}
            lines.append(
                "| {llm} | {r:.3f} | {p:.3f} | {cr:.3f} | {ac:.3f} | {passed} |".format(
                    llm=cell.get("llm_model"),
                    r=float(agg.get("answer_relevancy") or 0),
                    p=float(agg.get("context_precision") or 0),
                    cr=float(agg.get("context_recall") or 0),
                    ac=float(agg.get("answer_correctness") or 0),
                    passed=cell.get("passed"),
                )
            )
        lines.extend(["", "## Best cell per metric", ""])
        for metric, row in (summary.get("best_by_metric") or {}).items():
            lines.append(
                f"- **{metric}**: {row['score']:.4f} — `{row['llm_model']}`"
            )
        lines.append("")
        return "\n".join(lines)


class EvaluationJobManager:
    """Single-flight background evaluation jobs for the Metrics UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "run_id": None,
            "message": "No evaluation running",
            "error": None,
            "feature": None,
            "limit": None,
            "config": None,
            "started_at": None,
            "finished_at": None,
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def start(self, config: ExperimentConfig | None = None, **legacy) -> dict[str, Any]:
        if config is None:
            config = ExperimentConfig(
                feature=str(legacy.get("feature") or "all"),
                limit=legacy.get("limit"),
            )
        with self._lock:
            if self._state["status"] == "running":
                return dict(self._state)
            self._state = {
                "status": "running",
                "run_id": None,
                "message": "Evaluation started",
                "error": None,
                "feature": config.feature,
                "limit": config.limit,
                "config": asdict(config),
                "started_at": utc_now_iso(),
                "finished_at": None,
            }
            thread = threading.Thread(
                target=self._worker,
                kwargs={"config": config},
                daemon=True,
                name="eirc-evaluation",
            )
            self._thread = thread
            thread.start()
            return dict(self._state)

    def _worker(self, *, config: ExperimentConfig) -> None:
        try:
            multi = len(config.llm_models or []) > 1
            if multi or config.llm_models:
                out = EvaluationRunner.run_experiment(config)
            else:
                loader = get_config_loader()
                dataset = loader.resolve_path(loader.get("evaluation", "dataset_path"))
                sample_limit = config.limit or int(loader.get("evaluation", "sample_limit", default=8))
                out = EvaluationRunner(
                    temperature=config.temperature,
                    top_k=config.top_k,
                    fetch_k=config.fetch_k,
                ).run(dataset, config.feature, sample_limit, metric_keys=config.metrics or None)
            run_id = out.name
            with self._lock:
                self._state.update(
                    {
                        "status": "completed",
                        "run_id": run_id,
                        "message": f"Evaluation complete: {run_id}",
                        "error": None,
                        "finished_at": utc_now_iso(),
                    }
                )
        except Exception as exc:  # noqa: BLE001 — surface to UI
            log.exception("Evaluation job failed")
            with self._lock:
                self._state.update(
                    {
                        "status": "failed",
                        "message": "Evaluation failed",
                        "error": str(exc),
                        "finished_at": utc_now_iso(),
                    }
                )
