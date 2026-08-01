"""
Outreach draft repository.

Implements: PRD §5 (Outreach Composer + Validator Agent — persists drafted
emails pending human approval), §6a.1 (Layered Confidence Pipeline —
outreach drafts are tied to the contact's confidence level and its evidence
trail, not a standalone claim), §9 (Auditability — every stored fact carries
source and confidence, with a timestamp).
Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 4 - Draft
Repository, Task 1.

Persists a composed outreach email (app/agents/outreach/composer_agent.py)
and its validation result (app/agents/outreach/validator_agent.py) into the
`OutreachEmail` ORM model (app/db/models.py). Deliberately does not
duplicate `Contact.confidence_level` or `ConfidenceEvidence` rows onto
`OutreachEmail`: since `OutreachEmail.contact_id` already references the
`Contact` row, its confidence level and full evidence trail (via
`Contact.evidence`) remain a single source of truth (per
docs/coding_guidelines.md §4 — every stored fact carries its source and
confidence alongside the value, never duplicated). Validation issues are
serialized into `OutreachEmail.validation_notes` so the full evidence of
*why* validation passed or failed is retrievable, not just the pass/fail
boolean.
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy.orm import Session

from app.agents.outreach.composer_agent import ComposedEmail
from app.agents.outreach.validator_agent import ValidationResult
from app.db.models import Contact, JobPosting, OutreachEmail, OutreachStatus


class OutreachRepositoryError(Exception):
    """Raised when an outreach draft cannot be persisted or retrieved."""


def _compose_full_body(composed_email: ComposedEmail) -> str:
    """Join the composer's structured parts into the single `body` field
    stored on `OutreachEmail` (the ORM model has no separate greeting/closing
    columns; PRD §5 requires only the final email as reviewed/approved).
    """
    return "\n\n".join(
        part
        for part in (composed_email.greeting, composed_email.body, composed_email.closing)
        if part.strip()
    )


def _serialize_validation_notes(validation_result: ValidationResult) -> str:
    """Serialize every validation issue (category, severity, description) as
    JSON, so the full evidence behind a pass/fail outcome is retrievable, not
    just the boolean (PRD §9 — auditability).
    """
    return json.dumps(
        [
            {
                "category": issue.category.value,
                "severity": issue.severity.value,
                "description": issue.description,
            }
            for issue in validation_result.issues
        ]
    )


class OutreachRepository:
    """Persistence for drafted outreach emails and their validation results.

    Every draft is created in `OutreachStatus.DRAFT` with
    `approved_by_human = False` (the ORM model's default), regardless of
    whether validation passed — validation outcome informs, but never
    substitutes for, the human approval gate enforced downstream in
    `app/services/email_sender.py` (PRD §13.2).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_draft(
        self,
        job_posting_id: uuid.UUID,
        contact_id: uuid.UUID,
        composed_email: ComposedEmail,
        validation_result: ValidationResult,
    ) -> OutreachEmail:
        """Persist a newly composed and validated outreach draft.

        Args:
            job_posting_id: id of the `JobPosting` this draft targets.
            contact_id: id of the `Contact` this draft is addressed to; the
                contact's `confidence_level` and evidence trail remain
                accessible via this reference rather than being duplicated
                here.
            composed_email: the drafted email produced by
                `OutreachComposerAgent`.
            validation_result: the outcome produced by
                `OutreachValidatorAgent` for this draft.

        Returns:
            The persisted `OutreachEmail` row.

        Raises:
            OutreachRepositoryError: if `job_posting_id` or `contact_id`
                does not reference an existing row.
        """
        self._require_job_posting_exists(job_posting_id)
        self._require_contact_exists(contact_id)

        outreach_email = OutreachEmail(
            job_posting_id=job_posting_id,
            contact_id=contact_id,
            subject=composed_email.subject,
            body=_compose_full_body(composed_email),
            validation_passed=validation_result.passed,
            validation_notes=_serialize_validation_notes(validation_result),
            status=OutreachStatus.DRAFT,
        )
        self._session.add(outreach_email)
        self._session.flush()

        return outreach_email

    def get_draft(self, outreach_email_id: uuid.UUID) -> OutreachEmail | None:
        """Fetch a persisted outreach draft by id."""
        return self._session.get(OutreachEmail, outreach_email_id)

    def get_drafts_for_job_posting(self, job_posting_id: uuid.UUID) -> list[OutreachEmail]:
        """Fetch all outreach drafts for a given job posting."""
        return list(
            self._session.query(OutreachEmail)
            .filter(OutreachEmail.job_posting_id == job_posting_id)
            .all()
        )

    def _require_job_posting_exists(self, job_posting_id: uuid.UUID) -> None:
        if self._session.get(JobPosting, job_posting_id) is None:
            raise OutreachRepositoryError(
                f"No job posting found with id '{job_posting_id}'."
            )

    def _require_contact_exists(self, contact_id: uuid.UUID) -> None:
        if self._session.get(Contact, contact_id) is None:
            raise OutreachRepositoryError(
                f"No contact found with id '{contact_id}'."
            )