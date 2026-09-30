from __future__ import annotations

from models.models import RCACause, RCAReport, RCARequest
from utils.base_feature_service import BaseFeatureService


class RootCauseAnalysisService(BaseFeatureService):
    feature_name = "rca"
    prompt_id = "root_cause_analysis"

    def analyze(self, request: RCARequest) -> RCAReport:
        request_id = self.telemetry.new_request_id()
        self.telemetry.bind(request_id=request_id, feature=self.feature_name)
        timer = self.telemetry.timed()
        self.log.info("RCA analyze started", incident_chars=len(request.incident_description))

        upload_chunks = self.upload_manager.get_chunks(request.session_id)
        query = f"{request.incident_description}\n{request.system_context}".strip()
        hits = self.retriever.retrieve(
            query,
            feature=self.feature_name,
            upload_docs=upload_chunks,
        )
        context = self.retriever.format_context(hits)
        prompt, prompt_id, prompt_version = self.prompt_manager.render(
            self.prompt_id,
            incident_description=request.incident_description,
            system_context=request.system_context or "N/A",
            context=context or "No relevant NASA context retrieved.",
        )
        raw = self.invoke_llm(prompt)
        raw, citations = self.finalize_answer(raw, hits)
        run_meta = self.build_run_meta(
            request_id=request_id,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            latency_ms=timer.latency_ms,
            num_contexts=len(hits),
        )

        causes = self._parse_causes(raw)
        report = RCAReport(
            problem_statement=self._section_text(raw, "## Problem Statement", "## Likely Causes")
            or request.incident_description[:400],
            likely_causes=causes,
            recommended_checks=self.parse_bullet_section(raw, "## Recommended Checks"),
            corrective_actions=self.parse_bullet_section(raw, "## Corrective Actions"),
            summary=self._section_text(raw, "## Summary", None) or raw[:500],
            citations=citations,
            run_meta=run_meta,
            raw_answer=raw,
        )
        self.log.info(
            "RCA analyze completed",
            latency_ms=run_meta.latency_ms,
            citations=len(citations),
            causes=len(causes),
        )
        self.telemetry.clear()
        return report

    def _parse_causes(self, text: str) -> list[RCACause]:
        section = self._section_text(text, "## Likely Causes", "## Recommended Checks")
        causes: list[RCACause] = []
        if not section:
            return causes
        for line in section.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit() or stripped.startswith(("-", "*")):
                body = stripped.lstrip("0123456789.-*) ").strip()
                parts = [p.strip() for p in body.replace("—", "-").split(" - ")]
                cause = parts[0] if parts else body
                evidence = parts[1] if len(parts) > 1 else ""
                confidence = parts[2] if len(parts) > 2 else "medium"
                causes.append(
                    RCACause(
                        rank=len(causes) + 1,
                        cause=cause,
                        evidence=evidence,
                        confidence=confidence.lower().replace("confidence:", "").strip() or "medium",
                    )
                )
        return causes

    @staticmethod
    def _section_text(text: str, start: str, end: str | None) -> str:
        lower = text
        start_idx = lower.lower().find(start.lower())
        if start_idx < 0:
            return ""
        start_idx = start_idx + len(start)
        if end:
            end_idx = lower.lower().find(end.lower(), start_idx)
            if end_idx < 0:
                return text[start_idx:].strip()
            return text[start_idx:end_idx].strip()
        return text[start_idx:].strip()
