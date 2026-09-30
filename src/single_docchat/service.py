from __future__ import annotations

from models.models import ChatRequest, ChatResponse
from utils.base_feature_service import BaseFeatureService


class MaintenanceCopilotService(BaseFeatureService):
    feature_name = "maintenance"
    prompt_id = "maintenance_troubleshooting"

    def chat(self, request: ChatRequest) -> ChatResponse:
        request_id = self.telemetry.new_request_id()
        self.telemetry.bind(request_id=request_id, feature=self.feature_name)
        timer = self.telemetry.timed()
        self.log.info("Maintenance chat started", message_chars=len(request.message))

        history = "\n".join(f"{m.role}: {m.content}" for m in request.history[-6:]) or "None"
        upload_chunks = self.upload_manager.get_chunks(request.session_id)
        hits = self.retriever.retrieve(
            request.message,
            feature=self.feature_name,
            upload_docs=upload_chunks,
        )
        context = self.retriever.format_context(hits)
        prompt, prompt_id, prompt_version = self.prompt_manager.render(
            self.prompt_id,
            history=history,
            message=request.message,
            context=context or "No relevant NASA context retrieved.",
        )
        raw = self.invoke_llm(prompt)
        raw, citations = self.finalize_answer(raw, hits)
        answer = self._section_text(raw, "## Answer", "## Troubleshooting Steps") or raw
        steps = self.parse_bullet_section(raw, "## Troubleshooting Steps")
        if not steps:
            # numbered lines fallback
            for line in raw.splitlines():
                s = line.strip()
                if s[:2].isdigit() and "." in s[:4]:
                    steps.append(s.split(".", 1)[1].strip())

        response = ChatResponse(
            answer=answer.strip(),
            steps=steps,
            citations=citations,
            run_meta=self.build_run_meta(
                request_id=request_id,
                prompt_id=prompt_id,
                prompt_version=prompt_version,
                latency_ms=timer.latency_ms,
                num_contexts=len(hits),
            ),
        )
        self.log.info(
            "Maintenance chat completed",
            latency_ms=response.run_meta.latency_ms,
            steps=len(steps),
            citations=len(response.citations),
        )
        self.telemetry.clear()
        return response

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
