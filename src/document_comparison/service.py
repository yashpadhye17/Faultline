from __future__ import annotations

from models.models import ComparisonReport, ComparisonRequest
from utils.base_feature_service import BaseFeatureService


class ComparisonService(BaseFeatureService):
    feature_name = "comparison"
    prompt_id = "engineering_comparison"

    def compare(self, request: ComparisonRequest) -> ComparisonReport:
        request_id = self.telemetry.new_request_id()
        self.telemetry.bind(request_id=request_id, feature=self.feature_name)
        timer = self.telemetry.timed()
        self.log.info("Comparison started", topic_a=request.topic_a, topic_b=request.topic_b)

        upload_chunks = self.upload_manager.get_chunks(request.session_id)
        query = f"{request.topic_a} versus {request.topic_b} {request.focus}"
        hits = self.retriever.retrieve(
            query,
            feature=self.feature_name,
            upload_docs=upload_chunks,
        )
        context = self.retriever.format_context(hits)
        prompt, prompt_id, prompt_version = self.prompt_manager.render(
            self.prompt_id,
            topic_a=request.topic_a,
            topic_b=request.topic_b,
            focus=request.focus,
            context=context or "No relevant NASA context retrieved.",
        )
        raw = self.invoke_llm(prompt)
        raw, citations = self.finalize_answer(raw, hits)
        report = ComparisonReport(
            topic_a=request.topic_a,
            topic_b=request.topic_b,
            similarities=self.parse_bullet_section(raw, "## Similarities"),
            differences=self.parse_bullet_section(raw, "## Differences"),
            risks=self.parse_bullet_section(raw, "## Risks"),
            recommendations=self.parse_bullet_section(raw, "## Recommendations"),
            summary=self._section_text(raw, "## Summary", None) or raw[:500],
            citations=citations,
            run_meta=self.build_run_meta(
                request_id=request_id,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                latency_ms=timer.latency_ms,
                num_contexts=len(hits),
            ),
            raw_answer=raw,
        )
        self.log.info(
            "Comparison completed",
            latency_ms=report.run_meta.latency_ms,
            citations=len(report.citations),
        )
        self.telemetry.clear()
        return report

    @staticmethod
    def _section_text(text: str, start: str, end: str | None) -> str:
        start_idx = text.lower().find(start.lower())
        if start_idx < 0:
            return ""
        start_idx += len(start)
        if end:
            end_idx = text.lower().find(end.lower(), start_idx)
            if end_idx < 0:
                return text[start_idx:].strip()
            return text[start_idx:end_idx].strip()
        return text[start_idx:].strip()
