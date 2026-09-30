#!/usr/bin/env python3
"""CLI entrypoint for offline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.config_loader import get_config_loader
from utils.evaluation_runner import EvaluationRunner, ExperimentConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EIRC evaluation suite")
    parser.add_argument("--feature", default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument(
        "--llm",
        action="append",
        dest="llms",
        default=None,
        help="Generation LLM id (repeatable for multi-LLM experiments)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--fetch-k", type=int, default=None)
    args = parser.parse_args()

    loader = get_config_loader()
    limit = args.limit or int(loader.get("evaluation", "sample_limit", default=8))
    config = ExperimentConfig(
        feature=args.feature,
        limit=limit,
        llm_models=args.llms or [],
        temperature=args.temperature,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
    )
    if config.llm_models:
        out = EvaluationRunner.run_experiment(config)
    else:
        dataset = Path(args.dataset) if args.dataset else loader.resolve_path(
            loader.get("evaluation", "dataset_path")
        )
        out = EvaluationRunner(
            temperature=args.temperature,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
        ).run(dataset, args.feature, limit)
    print(f"Evaluation artifacts written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
