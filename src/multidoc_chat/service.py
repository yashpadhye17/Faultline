from __future__ import annotations

from models.models import ResearchRequest, ResearchResponse
from utils.base_feature_service import BaseFeatureService


class ResearchCopilotService(BaseFeatureService):
    feature_name = "research"
    prompt_id = "nasa_research"

    def ask(self, request: ResearchRequest) -> ResearchResponse:
        request_id = self.telemetry.new_request_id()
        self.telemetry.bind(request_id=request_id, feature=self.feature_name)
        timer = self.telemetry.timed()
        self.log.info("Research ask started", question_chars=len(request.question))

        extra = {}
        if request.center_code:
            extra["center_code"] = request.center_code
        if request.year_from is not None:
            extra["year_from"] = request.year_from
        if request.year_to is not None:
            extra["year_to"] = request.year_to

        upload_chunks = self.upload_manager.get_chunks(request.session_id)
        hits = self.retriever.retrieve(
            request.question,
            feature=self.feature_name,
            extra_filters=extra or None,
            upload_docs=upload_chunks,
        )
        context = self.retriever.format_context(hits)
        prompt, prompt_id, prompt_version = self.prompt_manager.render(
            self.prompt_id,
            question=request.question,
            context=context or "No relevant NASA context retrieved.",
        )
        answer = self.invoke_llm(prompt)
        answer, citations = self.finalize_answer(answer, hits)
        response = ResearchResponse(
            answer=answer,
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
            "Research ask completed",
            latency_ms=response.run_meta.latency_ms,
            citations=len(response.citations),
        )
        self.telemetry.clear()
        return response
