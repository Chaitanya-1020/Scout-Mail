"""
Unit tests for the outreach context builder.

Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 1 - Email
Context Builder, Task 1.

Covers successful context assembly and every individual validation failure
path in `OutreachContextBuilder.build_context`, per
docs/coding_guidelines.md §6 (every new module ships with unit tests; test
names reference behavior, not implementation). No external dependencies are
exercised — `OutreachContextBuilder` is pure logic, so no fakes/stubs beyond
a plain `logging.Logger` are required.
"""

from __future__ import annotations

import logging

import pytest

from app.agents.outreach.context_builder import (
    ContactContext,
    JobContext,
    OutreachContext,
    OutreachContextBuildError,
    OutreachContextBuilder,
)
from app.agents.resume_parser.parser_agent import ExperienceEntry, ResumeProfile


@pytest.fixture
def builder() -> OutreachContextBuilder:
    return OutreachContextBuilder(logger=logging.getLogger("test.outreach.context_builder"))


@pytest.fixture
def valid_resume_profile() -> ResumeProfile:
    return ResumeProfile(
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


@pytest.fixture
def valid_job() -> JobContext:
    return JobContext(
        company_name="Acme Inc",
        role_title="Senior Backend Engineer",
        jd_url="https://boards.greenhouse.io/acme/jobs/12345",
        jd_snapshot_text="We are looking for a Senior Backend Engineer to join our team.",
    )


@pytest.fixture
def valid_contact() -> ContactContext:
    return ContactContext(
        name="John Smith",
        title="Engineering Recruiter",
        email="john.smith@acme.com",
        confidence_level="high",
    )


class TestBuildContextSuccess:
    def test_builds_context_with_all_valid_inputs(
        self,
        builder: OutreachContextBuilder,
        valid_resume_profile: ResumeProfile,
        valid_job: JobContext,
        valid_contact: ContactContext,
    ) -> None:
        context = builder.build_context(
            resume_profile=valid_resume_profile,
            job=valid_job,
            contact=valid_contact,
            match_score=0.82,
        )

        assert isinstance(context, OutreachContext)
        assert context.resume_profile is valid_resume_profile
        assert context.job is valid_job
        assert context.contact is valid_contact
        assert context.match_score == 0.82

    def test_accepts_boundary_match_scores(
        self,
        builder: OutreachContextBuilder,
        valid_resume_profile: ResumeProfile,
        valid_job: JobContext,
        valid_contact: ContactContext,
    ) -> None:
        for boundary_score in (0.0, 1.0):
            context = builder.build_context(
                resume_profile=valid_resume_profile,
                job=valid_job,
                contact=valid_contact,
                match_score=boundary_score,
            )
            assert context.match_score == boundary_score

    def test_accepts_resume_with_experience_but_no_skills(
        self,
        builder: OutreachContextBuilder,
        valid_job: JobContext,
        valid_contact: ContactContext,
    ) -> None:
        profile = ResumeProfile(
            full_name="Jane Doe",
            email=None,
            phone=None,
            skills=[],
            target_roles=[],
            experience=[
                ExperienceEntry(company="Acme Corp", title="Engineer")
            ],
            education=[],
            years_of_experience=None,
        )

        context = builder.build_context(
            resume_profile=profile, job=valid_job, contact=valid_contact, match_score=0.5
        )

        assert context.resume_profile.experience[0].company == "Acme Corp"


class TestBuildContextValidationFailures:
    def test_rejects_none_resume_profile(
        self, builder: OutreachContextBuilder, valid_job: JobContext, valid_contact: ContactContext
    ) -> None:
        with pytest.raises(OutreachContextBuildError, match="resume_profile is required"):
            builder.build_context(
                resume_profile=None, job=valid_job, contact=valid_contact, match_score=0.5
            )

    def test_rejects_resume_profile_with_no_skills_or_experience(
        self, builder: OutreachContextBuilder, valid_job: JobContext, valid_contact: ContactContext
    ) -> None:
        empty_profile = ResumeProfile(
            full_name="Jane Doe",
            email=None,
            phone=None,
            skills=[],
            target_roles=[],
            experience=[],
            education=[],
            years_of_experience=None,
        )

        with pytest.raises(OutreachContextBuildError, match="at least skills or experience"):
            builder.build_context(
                resume_profile=empty_profile, job=valid_job, contact=valid_contact, match_score=0.5
            )

    def test_rejects_none_job(
        self, builder: OutreachContextBuilder, valid_resume_profile: ResumeProfile, valid_contact: ContactContext
    ) -> None:
        with pytest.raises(OutreachContextBuildError, match="job is required"):
            builder.build_context(
                resume_profile=valid_resume_profile, job=None, contact=valid_contact, match_score=0.5
            )

    @pytest.mark.parametrize(
        "field_name",
        ["company_name", "role_title", "jd_url", "jd_snapshot_text"],
    )
    def test_rejects_job_with_empty_required_field(
        self,
        builder: OutreachContextBuilder,
        valid_resume_profile: ResumeProfile,
        valid_contact: ContactContext,
        field_name: str,
    ) -> None:
        job_kwargs = {
            "company_name": "Acme Inc",
            "role_title": "Senior Backend Engineer",
            "jd_url": "https://boards.greenhouse.io/acme/jobs/12345",
            "jd_snapshot_text": "We are looking for a Senior Backend Engineer.",
        }
        job_kwargs[field_name] = "   "
        invalid_job = JobContext(**job_kwargs)

        with pytest.raises(OutreachContextBuildError, match=f"job.{field_name}"):
            builder.build_context(
                resume_profile=valid_resume_profile,
                job=invalid_job,
                contact=valid_contact,
                match_score=0.5,
            )

    def test_rejects_none_contact(
        self, builder: OutreachContextBuilder, valid_resume_profile: ResumeProfile, valid_job: JobContext
    ) -> None:
        with pytest.raises(OutreachContextBuildError, match="contact is required"):
            builder.build_context(
                resume_profile=valid_resume_profile, job=valid_job, contact=None, match_score=0.5
            )

    def test_rejects_contact_with_empty_name(
        self, builder: OutreachContextBuilder, valid_resume_profile: ResumeProfile, valid_job: JobContext
    ) -> None:
        invalid_contact = ContactContext(
            name="  ", title="Recruiter", email="john@acme.com", confidence_level="high"
        )

        with pytest.raises(OutreachContextBuildError, match="contact.name"):
            builder.build_context(
                resume_profile=valid_resume_profile,
                job=valid_job,
                contact=invalid_contact,
                match_score=0.5,
            )

    def test_rejects_contact_with_invalid_email(
        self, builder: OutreachContextBuilder, valid_resume_profile: ResumeProfile, valid_job: JobContext
    ) -> None:
        invalid_contact = ContactContext(
            name="John Smith", title="Recruiter", email="not-an-email", confidence_level="high"
        )

        with pytest.raises(OutreachContextBuildError, match="contact.email"):
            builder.build_context(
                resume_profile=valid_resume_profile,
                job=valid_job,
                contact=invalid_contact,
                match_score=0.5,
            )

    def test_rejects_contact_with_empty_confidence_level(
        self, builder: OutreachContextBuilder, valid_resume_profile: ResumeProfile, valid_job: JobContext
    ) -> None:
        invalid_contact = ContactContext(
            name="John Smith", title="Recruiter", email="john@acme.com", confidence_level=""
        )

        with pytest.raises(OutreachContextBuildError, match="contact.confidence_level"):
            builder.build_context(
                resume_profile=valid_resume_profile,
                job=valid_job,
                contact=invalid_contact,
                match_score=0.5,
            )

    def test_rejects_none_match_score(
        self,
        builder: OutreachContextBuilder,
        valid_resume_profile: ResumeProfile,
        valid_job: JobContext,
        valid_contact: ContactContext,
    ) -> None:
        with pytest.raises(OutreachContextBuildError, match="match_score is required"):
            builder.build_context(
                resume_profile=valid_resume_profile,
                job=valid_job,
                contact=valid_contact,
                match_score=None,
            )

    @pytest.mark.parametrize("out_of_range_score", [-0.1, 1.1, 5.0])
    def test_rejects_out_of_range_match_score(
        self,
        builder: OutreachContextBuilder,
        valid_resume_profile: ResumeProfile,
        valid_job: JobContext,
        valid_contact: ContactContext,
        out_of_range_score: float,
    ) -> None:
        with pytest.raises(OutreachContextBuildError, match="match_score must be between"):
            builder.build_context(
                resume_profile=valid_resume_profile,
                job=valid_job,
                contact=valid_contact,
                match_score=out_of_range_score,
            )

    def test_reports_multiple_validation_errors_together(
        self, builder: OutreachContextBuilder
    ) -> None:
        with pytest.raises(OutreachContextBuildError) as exc_info:
            builder.build_context(
                resume_profile=None, job=None, contact=None, match_score=None
            )

        message = str(exc_info.value)
        assert "resume_profile is required" in message
        assert "job is required" in message
        assert "contact is required" in message
        assert "match_score is required" in message