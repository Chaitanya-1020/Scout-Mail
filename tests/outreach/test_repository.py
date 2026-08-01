"""
Unit tests for the outreach draft repository.

Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 4 - Draft
Repository, Task 1.

Exercises `OutreachRepository` against an in-memory SQLite database built
from the application's real `Base.metadata` (app/db/base.py,
app/db/models.py), per docs/coding_guidelines.md §6 — no test hits a real
Postgres instance; SQLite-in-memory stands in as a lightweight fake for the
same declarative schema.
"""

from __future__ import annotations

import json
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.agents.outreach.composer_agent import ComposedEmail
from app.agents.outreach.repository import OutreachRepository, OutreachRepositoryError
from app.agents.outreach.validator_agent import (
    ValidationIssue,
    ValidationIssueCategory,
    ValidationResult,
    ValidationSeverity,
)
from app.db.base import Base
from app.db.models import ConfidenceLevel, Contact, JobPosting, OutreachStatus


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, future=True)
    db_session = session_factory()
    yield db_session
    db_session.close()


@pytest.fixture
def job_posting(session: Session) -> JobPosting:
    posting = JobPosting(
        source_connector="greenhouse",
        external_id="12345",
        company_name="Acme Inc",
        role_title="Senior Backend Engineer",
        jd_url="https://boards.greenhouse.io/acme/jobs/12345",
        jd_snapshot_text="We are looking for a Senior Backend Engineer.",
        jd_hash="a" * 64,
    )
    session.add(posting)
    session.flush()
    return posting


@pytest.fixture
def contact(session: Session, job_posting: JobPosting) -> Contact:
    contact_row = Contact(
        job_posting_id=job_posting.id,
        name="John Smith",
        title="Engineering Recruiter",
        email="john.smith@acme.com",
        confidence_level=ConfidenceLevel.HIGH,
    )
    session.add(contact_row)
    session.flush()
    return contact_row


@pytest.fixture
def composed_email() -> ComposedEmail:
    return ComposedEmail(
        subject="Interest in the Senior Backend Engineer role",
        greeting="Hi John,",
        body="I'm reaching out about the Senior Backend Engineer role at Acme Inc.",
        closing="Best regards,",
    )


@pytest.fixture
def passing_validation_result() -> ValidationResult:
    return ValidationResult(passed=True, issues=[])


@pytest.fixture
def failing_validation_result() -> ValidationResult:
    return ValidationResult(
        passed=False,
        issues=[
            ValidationIssue(
                category=ValidationIssueCategory.FABRICATED_CLAIM,
                severity=ValidationSeverity.ERROR,
                description="Claims candidate led a team of 20 engineers",
            ),
            ValidationIssue(
                category=ValidationIssueCategory.MISSING_JOB_TITLE,
                severity=ValidationSeverity.WARNING,
                description="Role title does not appear verbatim in the email",
            ),
        ],
    )


class TestSaveDraftSuccess:
    def test_persists_draft_with_composed_email_content(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )

        assert saved.id is not None
        assert saved.subject == composed_email.subject
        assert "Hi John," in saved.body
        assert "Senior Backend Engineer role at Acme Inc" in saved.body
        assert "Best regards," in saved.body

    def test_persists_validation_passed_flag(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )

        assert saved.validation_passed is True

    def test_persists_failing_validation_and_serializes_issues(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        failing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=failing_validation_result,
        )

        assert saved.validation_passed is False
        parsed_notes = json.loads(saved.validation_notes)
        assert len(parsed_notes) == 2
        assert parsed_notes[0]["category"] == "fabricated_claim"
        assert parsed_notes[0]["severity"] == "error"
        assert "20 engineers" in parsed_notes[0]["description"]
        assert parsed_notes[1]["category"] == "missing_job_title"
        assert parsed_notes[1]["severity"] == "warning"

    def test_new_draft_defaults_to_draft_status_and_not_approved(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )

        assert saved.status == OutreachStatus.DRAFT
        assert saved.approved_by_human is False
        assert saved.sent_at is None

    def test_draft_even_with_failing_validation_is_not_auto_approved(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
        failing_validation_result: ValidationResult,
    ) -> None:
        # Validation outcome (pass or fail) must never itself set
        # approved_by_human -- only a human approval action can (PRD §13.2).
        repository = OutreachRepository(session)

        passing_draft = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )
        failing_draft = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=failing_validation_result,
        )

        assert passing_draft.approved_by_human is False
        assert failing_draft.approved_by_human is False

    def test_confidence_level_accessible_via_contact_relationship(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        # Confidence is not duplicated onto OutreachEmail -- it remains a
        # single source of truth on Contact, reachable via contact_id.
        repository = OutreachRepository(session)

        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )

        linked_contact = session.get(Contact, saved.contact_id)
        assert linked_contact is not None
        assert linked_contact.confidence_level == ConfidenceLevel.HIGH

    def test_created_at_timestamp_is_set(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )
        session.commit()
        session.refresh(saved)

        assert saved.created_at is not None


class TestSaveDraftValidationErrors:
    def test_raises_when_job_posting_does_not_exist(
        self,
        session: Session,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        with pytest.raises(OutreachRepositoryError, match="No job posting found"):
            repository.save_draft(
                job_posting_id=uuid.uuid4(),
                contact_id=contact.id,
                composed_email=composed_email,
                validation_result=passing_validation_result,
            )

    def test_raises_when_contact_does_not_exist(
        self,
        session: Session,
        job_posting: JobPosting,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)

        with pytest.raises(OutreachRepositoryError, match="No contact found"):
            repository.save_draft(
                job_posting_id=job_posting.id,
                contact_id=uuid.uuid4(),
                composed_email=composed_email,
                validation_result=passing_validation_result,
            )


class TestRetrieval:
    def test_get_draft_returns_persisted_row(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)
        saved = repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )

        fetched = repository.get_draft(saved.id)

        assert fetched is not None
        assert fetched.id == saved.id

    def test_get_draft_returns_none_for_unknown_id(self, session: Session) -> None:
        repository = OutreachRepository(session)

        assert repository.get_draft(uuid.uuid4()) is None

    def test_get_drafts_for_job_posting_returns_all_matching(
        self,
        session: Session,
        job_posting: JobPosting,
        contact: Contact,
        composed_email: ComposedEmail,
        passing_validation_result: ValidationResult,
    ) -> None:
        repository = OutreachRepository(session)
        repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )
        repository.save_draft(
            job_posting_id=job_posting.id,
            contact_id=contact.id,
            composed_email=composed_email,
            validation_result=passing_validation_result,
        )

        drafts = repository.get_drafts_for_job_posting(job_posting.id)

        assert len(drafts) == 2
        assert all(draft.job_posting_id == job_posting.id for draft in drafts)

    def test_get_drafts_for_job_posting_returns_empty_list_when_none_exist(
        self, session: Session, job_posting: JobPosting
    ) -> None:
        repository = OutreachRepository(session)

        assert repository.get_drafts_for_job_posting(job_posting.id) == []