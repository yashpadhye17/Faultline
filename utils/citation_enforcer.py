"""Ensure answers cite only retrieved NASA doc_ids."""

from __future__ import annotations

import re

from models.models import Citation

_DOC_ID = re.compile(r"\b(20\d{8})\b")
_BRACKET_ID = re.compile(r"\[(20\d{8})\]")


def extract_cited_doc_ids(text: str) -> set[str]:
    ids = set(_DOC_ID.findall(text))
    ids.update(_BRACKET_ID.findall(text))
    return ids


def enforce_citations(
    answer: str,
    citations: list[Citation],
    *,
    enabled: bool = True,
    append_sources_if_missing: bool = True,
) -> tuple[str, list[Citation]]:
    if not enabled or not citations:
        return answer, citations

    valid = {c.doc_id for c in citations if c.doc_id}
    cited = extract_cited_doc_ids(answer)
    grounded_cited = cited & valid
    orphans = cited - valid

    text = answer.rstrip()
    if orphans:
        text += (
            "\n\n> **Citation check:** The following ids were not in retrieved context and were "
            f"not treated as grounded: {', '.join(sorted(orphans))}."
        )

    if append_sources_if_missing and not grounded_cited:
        if "## Key Sources" not in text and "## key sources" not in text.lower():
            lines = ["\n\n## Key Sources"]
            for c in citations[:6]:
                title = c.title or c.doc_id
                lines.append(f"- **{c.doc_id}** — {title}")
            text = text + "\n".join(lines)
        grounded_cited = valid

    used = [c for c in citations if c.doc_id in grounded_cited]
    if not used:
        used = citations[: min(6, len(citations))]
    return text, used
