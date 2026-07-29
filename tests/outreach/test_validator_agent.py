"""
Unit tests for the outreach validator agent.

Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 3 - Email
Validation, Task 1.

Uses a fake `LLMProvider` (per docs/coding_guidelines.md §6 — no test hits a
real external API or real LLM by default) to verify: deterministic checks
(recipient name, company name, job title, personalization length) and
LLM-based fact-checking (fabricated claims, unsupported score claims),
independently and combined. Also verifies the agent never mutates the
`ComposedEmail` it validates.
"""

from __future__ import annotations

import logging

import pytest

from app.agents.outreach.composer_agent import ComposedEmail
from app.agents.outreach.context_builder import ContactContext, JobContext, OutreachContext
from app.agents.outreach.validator_agent import (
    OutreachValidationError,
    OutreachValidatorAgent,
    ValidationIssueCategory,
    ValidationSeverity,
)
from app.agents.resume_parser.parser_agent import ExperienceEntry, ResumeProfile
from app.llm.ollama_client import LLMProvider, LLMProviderError


class FakeLLMProvider(LLMProvider):
    """Test double for `LLMProvider` returning a pre-configured fact-check
    response or raising a pre-configured error."""

    def __init__(
        self,
        json_response: dict | None = None,
        raise_error: LLMProviderError | None = None,
    ) -> None:
        self._json_response = json_response if json_response is not None else {
            "fabricated_claims": [],
            "unsupported_score_claims": [],
        }
        self._raise_error = raise_error
        self.last_task: str | None = None

    def generate(self, task: str, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError("Not used by OutreachValidatorAgent.")

    def generate_json(self, task: str, prompt: str, system: str | None = None) -> dict:
        self.last_task = task
        if self._raise_error is not None:
            raise self._raise_error
        return self._json_response


@pytest.fixture
def context() -> OutreachContext:
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
def clean_email() -> ComposedEmail:
    return ComposedEmail(
        subject="Interest in the Senior Backend Engineer role at Acme Inc",
        greeting="Hi John,",
        body=(
            "I'm reaching out about the Senior Backend Engineer role at Acme Inc. "
            "My experience as a Software Engineer at Acme Corp building backend "
            "services with Python and SQL aligns well with what you're looking for."
        ),
        closing="Best regards,",
    )


class TestValidatePassing:
    def test_clean_email_passes_with_no_issues(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm, logger=logging.getLogger("test"))

        result = agent.validate(context, clean_email)

        assert result.passed is True
        assert result.issues == []
        assert result.has_errors is False

    def test_uses_validation_task_for_llm_call(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        agent.validate(context, clean_email)

        assert fake_llm.last_task == "validation"

    def test_does_not_mutate_composed_email(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        original_subject = clean_email.subject
        original_body = clean_email.body

        agent.validate(context, clean_email)

        assert clean_email.subject == original_subject
        assert clean_email.body == original_body


class TestDeterministicChecks:
    def test_flags_missing_recipient_name(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        broken_email = ComposedEmail(
            subject=clean_email.subject,
            greeting="Hello,",
            body=clean_email.body.replace("John", ""),
            closing=clean_email.closing,
        )
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, broken_email)

        categories = [issue.category for issue in result.issues]
        assert ValidationIssueCategory.MISSING_RECIPIENT_NAME in categories
        assert result.passed is False

    def test_flags_missing_company_name(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        broken_email = ComposedEmail(
            subject="Interest in the Senior Backend Engineer role",
            greeting=clean_email.greeting,
            body="I'm reaching out about the Senior Backend Engineer role. "
            "My experience as a Software Engineer at Acme Corp fits well.",
            closing=clean_email.closing,
        )
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, broken_email)

        categories = [issue.category for issue in result.issues]
        assert ValidationIssueCategory.MISSING_COMPANY_NAME in categories
        assert result.passed is False

    def test_flags_missing_job_title_as_warning_not_error(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        broken_email = ComposedEmail(
            subject="Interest in your open role at Acme Inc",
            greeting=clean_email.greeting,
            body="I'm reaching out about your open role at Acme Inc. "
            "My experience as a Software Engineer at Acme Corp fits well "
            "with the team's needs based on the description.",
            closing=clean_email.closing,
        )
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, broken_email)

        job_title_issues = [
            issue
            for issue in result.issues
            if issue.category == ValidationIssueCategory.MISSING_JOB_TITLE
        ]
        assert len(job_title_issues) == 1
        assert job_title_issues[0].severity == ValidationSeverity.WARNING

    def test_flags_missing_personalization_for_short_body(
        self, context: OutreachContext
    ) -> None:
        short_email = ComposedEmail(
            subject="Interest in the Senior Backend Engineer role at Acme Inc",
            greeting="Hi John,",
            body="Interested in this Senior Backend Engineer role at Acme Inc.",
            closing="Best regards,",
        )
        fake_llm = FakeLLMProvider()
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, short_email)

        categories = [issue.category for issue in result.issues]
        assert ValidationIssueCategory.MISSING_PERSONALIZATION in categories
        personalization_issue = next(
            issue
            for issue in result.issues
            if issue.category == ValidationIssueCategory.MISSING_PERSONALIZATION
        )
        assert personalization_issue.severity == ValidationSeverity.WARNING


class TestLlmFactChecking:
    def test_flags_fabricated_claims_as_errors(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider(
            json_response={
                "fabricated_claims": ["Claims candidate led a team of 20 engineers"],
                "unsupported_score_claims": [],
            }
        )
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, clean_email)

        fabricated_issues = [
            issue
            for issue in result.issues
            if issue.category == ValidationIssueCategory.FABRICATED_CLAIM
        ]
        assert len(fabricated_issues) == 1
        assert fabricated_issues[0].severity == ValidationSeverity.ERROR
        assert result.passed is False

    def test_flags_unsupported_score_claims_as_errors(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider(
            json_response={
                "fabricated_claims": [],
                "unsupported_score_claims": ["States a 95% match with the role"],
            }
        )
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, clean_email)

        score_issues = [
            issue
            for issue in result.issues
            if issue.category == ValidationIssueCategory.UNSUPPORTED_MATCH_SCORE_CLAIM
        ]
        assert len(score_issues) == 1
        assert score_issues[0].severity == ValidationSeverity.ERROR
        assert result.passed is False

    def test_ignores_empty_claim_strings(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider(
            json_response={"fabricated_claims": ["   "], "unsupported_score_claims": [""]}
        )
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, clean_email)

        assert result.issues == []
        assert result.passed is True

    def test_raises_when_llm_fact_check_fails(
        self, context: OutreachContext, clean_email: ComposedEmail
    ) -> None:
        fake_llm = FakeLLMProvider(raise_error=LLMProviderError("model timeout"))
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        with pytest.raises(OutreachValidationError, match="Acme Inc"):
            agent.validate(context, clean_email)


class TestCombinedIssues:
    def test_reports_deterministic_and_llm_issues_together(
        self, context: OutreachContext
    ) -> None:
        broken_email = ComposedEmail(
            subject="Interest in your open role",
            greeting="Hello,",
            body="I'm reaching out about your open role. "
            "My experience as a Software Engineer fits well with this position "
            "based on the description provided.",
            closing="Best regards,",
        )
        fake_llm = FakeLLMProvider(
            json_response={
                "fabricated_claims": ["Claims candidate has 10 years of experience"],
                "unsupported_score_claims": [],
            }
        )
        agent = OutreachValidatorAgent(llm_provider=fake_llm)

        result = agent.validate(context, broken_email)

        categories = {issue.category for issue in result.issues}
        assert ValidationIssueCategory.MISSING_RECIPIENT_NAME in categories
        assert ValidationIssueCategory.MISSING_COMPANY_NAME in categories
        assert ValidationIssueCategory.FABRICATED_CLAIM in categories
        assert result.passed is False
        assert result.has_errors is True