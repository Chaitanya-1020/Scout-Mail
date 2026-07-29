"""
Unit tests for the outreach composer agent.

Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 2 - Email
Composer, Task 1.

Uses a fake `LLMProvider` (per docs/coding_guidelines.md §6 — no test hits a
real external API or real LLM by default) to verify: successful composition,
handling of missing/empty LLM output fields, propagation of LLM provider
failures, and that a custom `OutreachPromptTemplate` is actually used when
supplied.
"""

from __future__ import annotations

import logging

import pytest

from app.agents.outreach.composer_agent import (
    ComposedEmail,
    OutreachComposerAgent,
    OutreachComposerError,
    OutreachPromptTemplate,
)
from app.agents.outreach.context_builder import ContactContext, JobContext, OutreachContext
from app.agents.resume_parser.parser_agent import ExperienceEntry, ResumeProfile
from app.llm.ollama_client import LLMProvider, LLMProviderError


class FakeLLMProvider(LLMProvider):
    """Test double for `LLMProvider` that returns a pre-configured response
    or raises a pre-configured error, and records the last call's arguments
    for assertions.
    """

    def __init__(
        self,
        json_response: dict | None = None,
        raise_error: LLMProviderError | None = None,
    ) -> None:
        self._json_response = json_response
        self._raise_error = raise_error
        self.last_task: str | None = None
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def generate(self, task: str, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError("Not used by OutreachComposerAgent.")

    def generate_json(self, task: str, prompt: str, system: str | None = None) -> dict:
        self.last_task = task
        self.last_prompt = prompt
        self.last_system = system

        if self._raise_error is not None:
            raise self._raise_error

        assert self._json_response is not None
        return self._json_response


@pytest.fixture
def valid_context() -> OutreachContext:
    resume_profile = ResumeProfile(
        full_name="Jane Doe",
        email="jane.doe@example.com",
        phone=None,
        skills=["Python", "SQL"],
        target_roles=["Backend Engineer"],
        experience=[
            ExperienceEntry(
                company="Acme Corp",
                title="Software Engineer",
                start_date="2020",
                end_date="2023",
                summary="Built backend services.",
            )
        ],
        education=[],
        years_of_experience=4.0,
    )
    job = JobContext(
        company_name="Acme Inc",
        role_title="Senior Backend Engineer",
        jd_url="https://boards.greenhouse.io/acme/jobs/12345",
        jd_snapshot_text="We are looking for a Senior Backend Engineer to join our team.",
    )
    contact = ContactContext(
        name="John Smith",
        title="Engineering Recruiter",
        email="john.smith@acme.com",
        confidence_level="high",
    )
    return OutreachContext(
        resume_profile=resume_profile, job=job, contact=contact, match_score=0.82
    )


@pytest.fixture
def valid_llm_response() -> dict:
    return {
        "subject": "Interest in the Senior Backend Engineer role",
        "greeting": "Hi John,",
        "body": "I'm reaching out about the Senior Backend Engineer role at Acme Inc. "
        "My experience as a Software Engineer at Acme Corp building backend "
        "services aligns well with what you're looking for.",
        "closing": "Best regards,",
    }


class TestComposeSuccess:
    def test_returns_composed_email_from_valid_llm_response(
        self, valid_context: OutreachContext, valid_llm_response: dict
    ) -> None:
        fake_llm = FakeLLMProvider(json_response=valid_llm_response)
        agent = OutreachComposerAgent(
            llm_provider=fake_llm, logger=logging.getLogger("test.composer")
        )

        result = agent.compose(valid_context)

        assert isinstance(result, ComposedEmail)
        assert result.subject == valid_llm_response["subject"]
        assert result.greeting == valid_llm_response["greeting"]
        assert result.body == valid_llm_response["body"]
        assert result.closing == valid_llm_response["closing"]

    def test_uses_generation_task_for_llm_call(
        self, valid_context: OutreachContext, valid_llm_response: dict
    ) -> None:
        fake_llm = FakeLLMProvider(json_response=valid_llm_response)
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        agent.compose(valid_context)

        assert fake_llm.last_task == "generation"

    def test_prompt_includes_resume_and_job_details(
        self, valid_context: OutreachContext, valid_llm_response: dict
    ) -> None:
        fake_llm = FakeLLMProvider(json_response=valid_llm_response)
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        agent.compose(valid_context)

        assert fake_llm.last_prompt is not None
        assert "Acme Inc" in fake_llm.last_prompt
        assert "Senior Backend Engineer" in fake_llm.last_prompt
        assert "Python" in fake_llm.last_prompt
        assert "John Smith" in fake_llm.last_prompt

    def test_uses_default_system_prompt_by_default(
        self, valid_context: OutreachContext, valid_llm_response: dict
    ) -> None:
        fake_llm = FakeLLMProvider(json_response=valid_llm_response)
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        agent.compose(valid_context)

        assert fake_llm.last_system is not None
        assert "Do not invent" in fake_llm.last_system

    def test_custom_prompt_template_is_used(
        self, valid_context: OutreachContext, valid_llm_response: dict
    ) -> None:
        custom_template = OutreachPromptTemplate(
            system_prompt="CUSTOM SYSTEM PROMPT MARKER",
            instruction_template="CUSTOM INSTRUCTION for {company_name} / {role_title}",
        )
        fake_llm = FakeLLMProvider(json_response=valid_llm_response)
        agent = OutreachComposerAgent(llm_provider=fake_llm, prompt_template=custom_template)

        agent.compose(valid_context)

        assert fake_llm.last_system == "CUSTOM SYSTEM PROMPT MARKER"
        assert fake_llm.last_prompt == "CUSTOM INSTRUCTION for Acme Inc / Senior Backend Engineer"


class TestComposeFailures:
    def test_raises_when_llm_provider_fails(self, valid_context: OutreachContext) -> None:
        fake_llm = FakeLLMProvider(raise_error=LLMProviderError("model unavailable"))
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        with pytest.raises(OutreachComposerError, match="Acme Inc"):
            agent.compose(valid_context)

    @pytest.mark.parametrize(
        "missing_field", ["subject", "greeting", "body", "closing"]
    )
    def test_raises_when_response_missing_required_field(
        self, valid_context: OutreachContext, valid_llm_response: dict, missing_field: str
    ) -> None:
        broken_response = dict(valid_llm_response)
        del broken_response[missing_field]

        fake_llm = FakeLLMProvider(json_response=broken_response)
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        with pytest.raises(OutreachComposerError, match=missing_field):
            agent.compose(valid_context)

    @pytest.mark.parametrize(
        "empty_field", ["subject", "greeting", "body", "closing"]
    )
    def test_raises_when_response_has_empty_field(
        self, valid_context: OutreachContext, valid_llm_response: dict, empty_field: str
    ) -> None:
        broken_response = dict(valid_llm_response)
        broken_response[empty_field] = "   "

        fake_llm = FakeLLMProvider(json_response=broken_response)
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        with pytest.raises(OutreachComposerError, match=empty_field):
            agent.compose(valid_context)

    def test_reports_all_missing_fields_together(self, valid_context: OutreachContext) -> None:
        fake_llm = FakeLLMProvider(json_response={})
        agent = OutreachComposerAgent(llm_provider=fake_llm)

        with pytest.raises(OutreachComposerError) as exc_info:
            agent.compose(valid_context)

        message = str(exc_info.value)
        assert "subject" in message
        assert "greeting" in message
        assert "body" in message
        assert "closing" in message