#!/usr/bin/env python3
"""Build or rebuild the NASA FAISS vectorstore."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from logger.logger import get_app_logger
from utils.vectorstore_manager import VectorStoreManager

log = get_app_logger("build_vectorstore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NASA FAISS vectorstore")
    parser.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    args = parser.parse_args()
    manager = VectorStoreManager()
    manifest = manager.build(force=args.force)
    log.info("Build finished", **manifest.model_dump())
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
