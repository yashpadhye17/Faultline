#!/usr/bin/env python3
"""
Download public NASA NTRS technical reports into data/nasa_ntrs/ for RAG ingestion.

For each PDF we also write a sidecar metadata JSON so vector DBs can attach
filterable fields (sti_type, center, year, subjects, keywords, authors, etc.).

API: https://ntrs.nasa.gov/api
Rate limit: 500 requests / 15 minutes (no auth required for public data).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests

BASE_URL = "https://ntrs.nasa.gov/api"
# Operational / engineering reliability corpus — not conference papers or journal research.
DEFAULT_TYPES = [
    "TECHNICAL_MEMORANDUM",
    "TECHNICAL_PUBLICATION",
    "CONTRACTOR_REPORT",
    "SPECIAL_PUBLICATION",
]
DEFAULT_QUERIES = [
    '"failure investigation"',
    '"failure analysis"',
    '"anomaly investigation"',
    '"mishap investigation"',
    '"root cause"',
    '"corrective action"',
    '"lessons learned" failure',
    "hardware anomaly",
    "component repair",
    "fault isolation",
    "problem reporting corrective",
    "bearing failure",
    "leak investigation",
    "fire test report",
    "damage assessment repair",
    "nonconformance",
    "incident investigation",
    "troubleshoot maintenance",
]
USER_AGENT = "DocumentPortal-NTRS-Fetcher/1.0 (educational RAG; contact local)"

# Title/abstract must look like a repair, failure, mishap, or anomaly report.
_REPAIR_TITLE = re.compile(
    r"\b("
    r"failure\s+(investigation|analysis|report)|"
    r"anomaly\s+(investigation|report|analysis)|"
    r"mishap|"
    r"root\s+cause|"
    r"corrective\s+action|"
    r"lessons\s+learned|"
    r"problem\s+report|"
    r"incident\s+investigation|"
    r"accident\s+investigation|"
    r"fault\s+(isolation|detection|analysis)|"
    r"troubleshoot|"
    r"nonconformance|"
    r"hardware\s+anomaly|"
    r"damage\s+(assessment|investigation|report)|"
    r"leak\s+(investigation|failure)|"
    r"fire\s+(investigation|test\s+report)|"
    r"bearing\s+failure|"
    r"nozzle\s+failure|"
    r"component[- ]level.*repair|"
    r"electronic[- ]assembly\s+repair|"
    r"repair\s+(procedure|report|of|analysis)|"
    r"post[- ](flight|test)\s+(anomaly|failure|investigation)|"
    r"fracture\s+(of|investigation|failure)|"
    r"crack(ed|ing)?\s+(investigation|failure|analysis)"
    r")\b",
    re.I,
)
_REPAIR_HINT = re.compile(
    r"\b(failure|anomaly|mishap|repair|fracture|crack|leak|fault|corrective|"
    r"nonconformance|troubleshoot|incident|accident|damage)\b",
    re.I,
)
_EXCLUDE_TITLE = re.compile(
    r"\b("
    r"conference\s+paper|symposium|workshop\s+proceedings|journal\s+article|"
    r"literature\s+review|cfd\s+study|numerical\s+simulation\s+of|"
    r"optimization\s+of\s+aerodynamic|conceptual\s+design\s+study|"
    r"preliminary\s+design\s+study|wind[- ]tunnel\s+test\s+technique"
    r")\b",
    re.I,
)


def is_repair_report(record: dict[str, Any]) -> bool:
    """Keep technical repair / failure / anomaly reports; drop research fluff."""
    title = str(record.get("title") or "")
    abstract = str(record.get("abstract") or "")
    if _EXCLUDE_TITLE.search(title):
        return False
    if _REPAIR_TITLE.search(title):
        return True
    # Abstract-only: only if the title already hints at ops/failure content.
    if _REPAIR_HINT.search(title) and _REPAIR_TITLE.search(abstract[:1200]):
        return True
    return False


def search_citations(
    session: requests.Session,
    *,
    query: str | None,
    sti_type: str | None,
    page_size: int,
    offset: int,
) -> dict[str, Any]:
    params: list[tuple[str, Any]] = [
        ("disseminated", "DOCUMENT_AND_METADATA"),
        ("page.size", page_size),
        ("page.from", offset),
        ("sort.field", "published"),
        ("sort.order", "desc"),
    ]
    if query:
        params.append(("q", query))
    if sti_type:
        params.append(("stiType", sti_type))

    response = session.get(
        f"{BASE_URL}/citations/search",
        params=params,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def extract_pdf_info(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return (filename, relative_api_path) for the first STI PDF download."""
    if not record.get("downloadsAvailable"):
        return None
    for item in record.get("downloads") or []:
        if item.get("type") not in (None, "STI"):
            continue
        links = item.get("links") or {}
        pdf_path = links.get("pdf") or links.get("original")
        name = item.get("name")
        if pdf_path and name and str(name).lower().endswith(".pdf"):
            return str(name), str(pdf_path)
        if pdf_path and str(pdf_path).lower().endswith(".pdf"):
            return Path(pdf_path).name, str(pdf_path)
    return None


def _author_name(affiliation: dict[str, Any]) -> str | None:
    meta = affiliation.get("meta") or {}
    author = meta.get("author") or affiliation.get("author") or {}
    if isinstance(author.get("name"), str) and author["name"].strip():
        return author["name"].strip()
    name = " ".join(
        part
        for part in [
            author.get("givenName"),
            author.get("middleName"),
            author.get("familyName"),
        ]
        if part
    ).strip()
    return name or None


def _parse_year(value: Any) -> int | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else None


def build_metadata(
    record: dict[str, Any],
    *,
    pdf_filename: str,
    query_tag: str | None = None,
) -> dict[str, Any]:
    """Flat, filter-friendly metadata for vector DB payloads."""
    authors: list[str] = []
    organizations: list[str] = []
    for affiliation in record.get("authorAffiliations") or []:
        name = _author_name(affiliation)
        if name and name not in authors:
            authors.append(name)
        org = ((affiliation.get("meta") or {}).get("organization") or {}).get("name")
        if org and org not in organizations:
            organizations.append(org)

    center = record.get("center") or {}
    if isinstance(center, str):
        center = {"name": center}
    published = (
        record.get("publishedDate")
        or record.get("distributionDate")
        or record.get("created")
    )
    year = _parse_year(published)
    doc_id = str(record.get("id"))
    subjects = [str(s) for s in (record.get("subjectCategories") or [])]
    keywords = [str(k) for k in (record.get("keywords") or [])]
    report_numbers = [str(n) for n in (record.get("otherReportNumbers") or [])]
    funding = []
    for item in record.get("fundingNumbers") or []:
        if isinstance(item, dict) and item.get("number"):
            funding.append(str(item["number"]))
        elif item:
            funding.append(str(item))

    meta = {
        # Primary identity / file linkage
        "doc_id": doc_id,
        "source": "NASA NTRS",
        "pdf_filename": pdf_filename,
        "pdf_path": f"pdfs/{pdf_filename}",
        "metadata_path": f"metadata/{doc_id}.json",
        "citation_url": f"https://ntrs.nasa.gov/citations/{doc_id}",
        # Filterable scalars
        "title": record.get("title") or "",
        "sti_type": record.get("stiType") or "",
        "sti_type_details": record.get("stiTypeDetails") or "",
        "center_code": center.get("code") or "",
        "center_name": center.get("name") or "",
        "year": year,
        "published": published,
        "distribution": record.get("distribution") or "",
        "disseminated": record.get("disseminated") or "",
        "status": record.get("status") or "",
        # Filterable lists (Chroma/Qdrant/Pinecone-style where filters)
        "authors": authors,
        "organizations": organizations,
        "subject_categories": subjects,
        "keywords": keywords,
        "report_numbers": report_numbers,
        "funding_numbers": funding,
        # Useful for retrieval display / optional embedding context
        "abstract": record.get("abstract") or "",
        "query_tag": query_tag or "",
    }
    return meta


def download_pdf(
    session: requests.Session,
    relative_path: str,
    dest: Path,
    *,
    force: bool,
) -> bool:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False
    url = f"https://ntrs.nasa.gov{relative_path}"
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)
    return True


def write_sidecar(meta_dir: Path, meta: dict[str, Any]) -> Path:
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{meta['doc_id']}.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_existing_index(metadata_path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if not metadata_path.exists():
        return by_id
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or row.get("id") or "")
            if doc_id:
                by_id[doc_id] = row
    return by_id


def write_metadata_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def iter_records_for_query(
    session: requests.Session,
    *,
    query: str | None,
    sti_types: list[str],
    per_query_limit: int,
    seen_ids: set[str],
    repair_only: bool = True,
    max_scan: int = 400,
):
    """Yield downloadable repair/failure reports for one query across STI types."""
    yielded = 0
    for sti_type in sti_types:
        if yielded >= per_query_limit:
            return
        offset = 0
        scanned = 0
        page_size = min(100, max(per_query_limit, 25))
        while yielded < per_query_limit and scanned < max_scan:
            payload = search_citations(
                session,
                query=query,
                sti_type=sti_type,
                page_size=page_size,
                offset=offset,
            )
            results = payload.get("results") or []
            total = (payload.get("stats") or {}).get("total", 0)
            if offset == 0:
                print(f"    [{sti_type}] raw matches: {total}")
            if not results:
                break

            for record in results:
                scanned += 1
                doc_id = str(record.get("id"))
                if doc_id in seen_ids:
                    continue
                if repair_only and not is_repair_report(record):
                    continue
                if not extract_pdf_info(record):
                    continue
                seen_ids.add(doc_id)
                yield record
                yielded += 1
                if yielded >= per_query_limit:
                    return
                if scanned >= max_scan:
                    break

            offset += len(results)
            if offset >= total:
                break
        if yielded:
            print(f"    [{sti_type}] kept after repair filter: {yielded} (scanned≈{scanned})")


def refresh_existing_pdfs(
    session: requests.Session,
    *,
    pdf_dir: Path,
    meta_dir: Path,
    collected: dict[str, dict[str, Any]],
    delay: float,
) -> int:
    """Rebuild filterable metadata + sidecars for PDFs already on disk."""
    refreshed = 0
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc_id = pdf.name.split("_", 1)[0]
        if not doc_id.isdigit():
            continue
        existing = collected.get(doc_id)
        needs_refresh = (
            existing is None
            or "doc_id" not in existing
            or "sti_type" not in existing
            or "center_code" not in existing
            or not (meta_dir / f"{doc_id}.json").exists()
        )
        if not needs_refresh:
            continue
        try:
            response = session.get(f"{BASE_URL}/citations/{doc_id}", timeout=60)
            response.raise_for_status()
            record = response.json()
        except requests.RequestException as exc:
            print(f"  ! refresh failed {doc_id}: {exc}", file=sys.stderr)
            time.sleep(delay)
            continue
        meta = build_metadata(record, pdf_filename=pdf.name, query_tag=(existing or {}).get("query_tag", ""))
        write_sidecar(meta_dir, meta)
        collected[doc_id] = meta
        refreshed += 1
        print(f"  ~ refreshed metadata {doc_id}")
        time.sleep(delay)
    return refreshed


def clear_corpus(out_dir: Path) -> None:
    """Remove existing PDFs and metadata so a fetch starts clean."""
    pdf_dir = out_dir / "pdfs"
    meta_dir = out_dir / "metadata"
    for path in (out_dir / "metadata.jsonl", out_dir / "manifest.json"):
        if path.exists():
            path.unlink()
    if pdf_dir.exists():
        for pdf in pdf_dir.glob("*"):
            if pdf.is_file():
                pdf.unlink()
    if meta_dir.exists():
        for side in meta_dir.glob("*.json"):
            side.unlink()
    print(f"Cleared prior corpus under {out_dir}")


def fetch_corpus(
    *,
    out_dir: Path,
    queries: list[str | None],
    sti_types: list[str],
    total_limit: int,
    per_query_limit: int,
    delay: float,
    force: bool,
    replace: bool = False,
    repair_only: bool = True,
) -> dict[str, Any]:
    pdf_dir = out_dir / "pdfs"
    meta_dir = out_dir / "metadata"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.jsonl"
    manifest_path = out_dir / "manifest.json"

    if replace:
        clear_corpus(out_dir)

    existing = {} if replace else load_existing_index(metadata_path)
    collected: dict[str, dict[str, Any]] = dict(existing)
    seen_ids = set(collected.keys())

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    if not replace:
        print("Refreshing metadata for existing PDFs (if needed)...")
        refresh_existing_pdfs(
            session,
            pdf_dir=pdf_dir,
            meta_dir=meta_dir,
            collected=collected,
            delay=delay,
        )
        seen_ids = set(collected.keys())

    downloaded = 0
    reused = 0
    metadata_written = 0
    failed = 0
    skipped_non_repair = 0
    starting_count = len(collected)

    print("NASA NTRS corpus fetch (technical repair / failure / anomaly reports)")
    print(f"  output={out_dir}")
    print(f"  queries={queries}")
    print(f"  types={sti_types}")
    print(f"  repair_only={repair_only} replace={replace}")
    print(f"  total_limit={total_limit} per_query_limit={per_query_limit}")
    print(f"  already_indexed={starting_count}")

    for query in queries:
        if len(collected) >= total_limit:
            break
        remaining = total_limit - len(collected)
        q_limit = min(per_query_limit, remaining)
        q_label = query if query else "(no query)"
        print(f"\n== query: {q_label!r} (want up to {q_limit} new) ==")

        for record in iter_records_for_query(
            session,
            query=query,
            sti_types=sti_types,
            per_query_limit=q_limit,
            seen_ids=seen_ids,
            repair_only=repair_only,
        ):
            if len(collected) >= total_limit:
                break

            pdf_info = extract_pdf_info(record)
            if not pdf_info:
                continue
            filename, relative_path = pdf_info
            dest_name = f"{record.get('id')}_{filename}"
            dest = pdf_dir / dest_name

            try:
                was_new = download_pdf(session, relative_path, dest, force=force)
            except requests.RequestException as exc:
                failed += 1
                print(f"  ! failed {record.get('id')}: {exc}", file=sys.stderr)
                time.sleep(delay)
                continue

            meta = build_metadata(
                record,
                pdf_filename=dest_name,
                query_tag=query or "",
            )
            meta["corpus_focus"] = "technical_repair_reports"
            write_sidecar(meta_dir, meta)
            collected[meta["doc_id"]] = meta
            metadata_written += 1

            if was_new:
                downloaded += 1
                print(f"  + {meta['doc_id']} | {meta['sti_type']} | {meta['title'][:70]}")
            else:
                reused += 1
                print(f"  = {meta['doc_id']} | cached pdf + refreshed metadata")
            time.sleep(delay)

    # Drop any leftover non-repair docs if replace was not used but repair_only is on.
    if repair_only and not replace:
        before = len(collected)
        pruned: dict[str, dict[str, Any]] = {}
        for doc_id, meta in collected.items():
            title = str(meta.get("title") or "")
            abstract = str(meta.get("abstract") or "")
            fake = {"title": title, "abstract": abstract}
            if is_repair_report(fake):
                pruned[doc_id] = meta
            else:
                skipped_non_repair += 1
                pdf_name = meta.get("pdf_filename")
                if pdf_name:
                    pdf_path = pdf_dir / str(pdf_name)
                    if pdf_path.exists():
                        pdf_path.unlink()
                side = meta_dir / f"{doc_id}.json"
                if side.exists():
                    side.unlink()
        collected = pruned
        if skipped_non_repair:
            print(f"Pruned {skipped_non_repair} non-repair docs from prior corpus")

    ordered = sorted(collected.values(), key=lambda r: str(r.get("doc_id")))
    write_metadata_jsonl(metadata_path, ordered)

    # Compact manifest (no abstracts) for quick inventory.
    manifest = {
        "source": "NASA NTRS OpenAPI",
        "base_url": BASE_URL,
        "corpus_focus": "technical_repair_reports",
        "queries": queries,
        "sti_types": sti_types,
        "repair_only": repair_only,
        "total_limit": total_limit,
        "per_query_limit": per_query_limit,
        "saved_count": len(ordered),
        "new_since_run_start": max(0, len(ordered) - starting_count),
        "downloaded": downloaded,
        "reused_existing_pdfs": reused,
        "metadata_written": metadata_written,
        "pruned_non_repair": skipped_non_repair,
        "failed": failed,
        "pdf_dir": str(pdf_dir),
        "metadata_dir": str(meta_dir),
        "metadata_file": str(metadata_path),
        "filterable_fields": [
            "doc_id",
            "sti_type",
            "center_code",
            "center_name",
            "year",
            "authors",
            "subject_categories",
            "keywords",
            "report_numbers",
            "source",
            "query_tag",
            "corpus_focus",
        ],
        "attribution": (
            "Data provided by NASA Scientific and Technical Information Program "
            "(https://ntrs.nasa.gov/api/openapi/)"
        ),
        "records": [
            {
                "doc_id": row.get("doc_id"),
                "title": row.get("title"),
                "sti_type": row.get("sti_type"),
                "center_code": row.get("center_code"),
                "year": row.get("year"),
                "pdf_filename": row.get("pdf_filename"),
                "citation_url": row.get("citation_url"),
            }
            for row in ordered
        ],
    }
    write_manifest(manifest_path, manifest)

    print(
        f"\nDone. total={len(ordered)} downloaded={downloaded} "
        f"cached={reused} pruned={skipped_non_repair} failed={failed}"
    )
    print(f"PDFs:      {pdf_dir}")
    print(f"Sidecars:  {meta_dir}/<doc_id>.json")
    print(f"Index:     {metadata_path}")
    return manifest


def backfill_sidecars_from_jsonl(out_dir: Path) -> int:
    """Ensure every metadata.jsonl row has a sidecar JSON file."""
    metadata_path = out_dir / "metadata.jsonl"
    meta_dir = out_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    rows = load_existing_index(metadata_path)
    count = 0
    for meta in rows.values():
        # Normalize legacy rows if needed.
        if "doc_id" not in meta and "id" in meta:
            meta = {
                **meta,
                "doc_id": str(meta["id"]),
                "sti_type": meta.get("stiType") or meta.get("sti_type") or "",
            }
        write_sidecar(meta_dir, meta)
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch NASA NTRS technical repair / failure / anomaly reports "
            "+ filterable metadata for RAG (excludes research-paper noise)"
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "nasa_ntrs",
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        dest="queries",
        default=None,
        help="Free-text query (repeatable). Default: repair/failure/anomaly set.",
    )
    parser.add_argument(
        "--sti-type",
        action="append",
        dest="sti_types",
        default=None,
        help="STI type filter (repeatable). Defaults to TM/TP/CR/SP.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=250,
        help="Max total documents to keep in the corpus (default: 250)",
    )
    parser.add_argument(
        "--per-query-limit",
        type=int,
        default=40,
        help="Max new docs to pull per query before moving on (default: 40)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between downloads",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download PDFs even if they already exist",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Wipe existing PDFs/metadata before fetching a fresh repair-report corpus",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Disable repair/failure title filter (not recommended for EIRC)",
    )
    parser.add_argument(
        "--backfill-only",
        action="store_true",
        help="Only write sidecars from existing metadata.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.backfill_only:
        n = backfill_sidecars_from_jsonl(args.out_dir)
        print(f"Wrote {n} sidecar metadata files under {args.out_dir / 'metadata'}")
        return 0

    sti_types = args.sti_types or list(DEFAULT_TYPES)
    if args.queries is None:
        queries: list[str | None] = list(DEFAULT_QUERIES)
    else:
        queries = [q.strip() or None for q in args.queries]

    fetch_corpus(
        out_dir=args.out_dir,
        queries=queries,
        sti_types=sti_types,
        total_limit=args.limit,
        per_query_limit=args.per_query_limit,
        delay=args.delay,
        force=args.force,
        replace=args.replace,
        repair_only=not args.include_all,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
