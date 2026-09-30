#!/usr/bin/env python3
"""Backfill feature_tags on NASA NTRS metadata for filtered retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from logger.logger import get_app_logger
from utils.config_loader import get_config_loader

log = get_app_logger("enrich_feature_tags")

RULES = {
    "rca": [
        "failure",
        "fault",
        "anomaly",
        "mishap",
        "reliability",
        "fmea",
        "root cause",
        "safety",
        "fracture",
        "fatigue",
    ],
    "maintenance": [
        "maintenance",
        "troubleshoot",
        "diagnostic",
        "health management",
        "operations",
        "repair",
        "inspection",
        "prognostic",
    ],
    "comparison": [
        "trade",
        "comparison",
        "evaluation",
        "design",
        "test",
        "characterization",
        "benchmark",
    ],
    "research": [
        "propulsion",
        "aerodynamic",
        "spacecraft",
        "material",
        "structure",
        "avionics",
        "thermal",
        "communication",
    ],
}

QUERY_TAG_MAP = {
    '"failure investigation"': ["rca", "maintenance"],
    '"failure analysis"': ["rca"],
    '"anomaly investigation"': ["rca", "maintenance"],
    '"mishap investigation"': ["rca"],
    '"root cause"': ["rca"],
    '"corrective action"': ["rca", "maintenance"],
    '"lessons learned" failure': ["rca", "maintenance"],
    "hardware anomaly": ["rca", "maintenance"],
    "component repair": ["maintenance", "rca"],
    "fault isolation": ["maintenance", "rca"],
    "problem reporting corrective": ["rca", "maintenance"],
    "bearing failure": ["rca", "maintenance"],
    "leak investigation": ["rca"],
    "fire test report": ["rca", "comparison"],
    "damage assessment repair": ["rca", "maintenance"],
    "nonconformance": ["rca", "maintenance"],
    "incident investigation": ["rca"],
    "troubleshoot maintenance": ["maintenance", "rca"],
}


def infer_tags(row: dict) -> list[str]:
    blob = " ".join(
        [
            str(row.get("title") or ""),
            str(row.get("abstract") or ""),
            " ".join(row.get("keywords") or []),
            " ".join(row.get("subject_categories") or []),
            str(row.get("query_tag") or ""),
        ]
    ).lower()
    tags: set[str] = set(row.get("feature_tags") or [])
    for feature, needles in RULES.items():
        if any(n in blob for n in needles):
            tags.add(feature)
    qt = (row.get("query_tag") or "").lower()
    for key, mapped in QUERY_TAG_MAP.items():
        if qt == key.lower():
            tags.update(mapped)
    # Repair-focused corpus: default to RCA/maintenance, not open research.
    if not tags:
        tags.update({"rca", "maintenance"})
    return sorted(tags)


def main() -> int:
    loader = get_config_loader()
    jsonl_path = loader.resolve_path(loader.get("corpus", "metadata_jsonl"))
    meta_dir = loader.resolve_path(loader.get("corpus", "metadata_dir"))
    rows = []
    if jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    updated = 0
    by_id = {}
    for row in rows:
        row["feature_tags"] = infer_tags(row)
        by_id[str(row.get("doc_id"))] = row
        updated += 1

    for path in meta_dir.glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["feature_tags"] = infer_tags(row)
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        by_id[str(row.get("doc_id") or path.stem)] = row

    ordered = sorted(by_id.values(), key=lambda r: str(r.get("doc_id")))
    jsonl_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + ("\n" if ordered else ""),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in ordered:
        for tag in row.get("feature_tags") or []:
            counts[tag] = counts.get(tag, 0) + 1
    log.info("Feature tags enriched", docs=len(ordered), tag_counts=counts)
    print(json.dumps({"docs": len(ordered), "tag_counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
