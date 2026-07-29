"""
Outreach context builder.

Implements: PRD §5 (Outreach Composer + Validator Agent — drafts a
personalized email using resume + job description + contact info), §6.4
(Outreach personalization must be grounded in the candidate's actual resume
and the job posting's actual JD, not generic boilerplate), §9 (Auditability
— every fact used to draft an email is traceable back to its source: parsed
resume, JD snapshot, match score, and resolved contact).
Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 1 - Email
Context Builder, Task 1.

Combines a candidate's parsed resume profile, a job posting's description,
its resume-match score, and a resolved contact into a single, strongly
typed `OutreachContext` consumed by the outreach composer agent
(app/agents/outreach/composer_agent.py, a later task). This module performs
no LLM calls, no persistence, and no HTTP I/O — it is pure assembly and
validation logic (Single Responsibility, per docs/coding_guidelines.md).
Depends only on `ResumeProfile` (app/agents/resume_parser/parser_agent.py)
and a logger, injected at construction (Dependency Inversion, per
docs/architecture.md) — never on ORM models directly, so callers translate
`Resume`/`JobPosting`/`Contact` rows into the plain input dataclasses defined
here at the call site (e.g. the graph layer, Epic 7).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agents.resume_parser.parser_agent import ResumeProfile


class OutreachContextBuildError(Exception):
    """Raised when an `OutreachContext` cannot be built from the given inputs
    because one or more required fields are missing or invalid.
    """


@dataclass(frozen=True)
class JobContext:
    """The job-posting facts an outreach email may be grounded in.

    `role_title` and `jd_snapshot_text` must be the verbatim strings sourced
    from `app.db.models.JobPosting` (PRD §6.2 — never paraphrased before
    reaching this builder).
    """

    company_name: str
    role_title: str
    jd_url: str
    jd_snapshot_text: str


@dataclass(frozen=True)
class ContactContext:
    """The resolved contact facts an outreach email may be addressed to.

    `confidence_level` is carried through so the composer/validator agent
    (Epic 6, Story 2) and the review dashboard (Epic 8) can factor it into
    downstream decisions; this builder does not interpret or gate on it.
    """

    name: str
    title: str | None
    email: str
    confidence_level: str
    """String form of `app.utils.confidence.ConfidenceLevel` (e.g. 'high',
    'medium', 'low'), kept as a plain string here to avoid this domain-layer
    module depending on that module's specific enum type."""


@dataclass(frozen=True)
class OutreachContext:
    """Strongly typed, fully validated context for drafting one outreach email.

    An instance of this type is only ever constructed via
    `OutreachContextBuilder.build_context`, which guarantees every field is
    present and non-empty — the outreach composer agent can consume this
    type without re-validating its contents.
    """

    resume_profile: ResumeProfile
    job: JobContext
    contact: ContactContext
    match_score: float


class OutreachContextBuilder:
    """Assembles and validates an `OutreachContext` from its constituent parts.

    A `logging.Logger` is injected at construction (defaulting to this
    module's logger) rather than imported globally, so callers can supply a
    contextual/child logger (e.g. bound with a job-posting id) without this
    class depending on a specific logging framework beyond the standard
    library (per docs/coding_guidelines.md).
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger(__name__)

    def build_context(
        self,
        resume_profile: ResumeProfile,
        job: JobContext,
        contact: ContactContext,
        match_score: float,
    ) -> OutreachContext:
        """Build a validated `OutreachContext`.

        Args:
            resume_profile: the candidate's structured resume profile
                (app.agents.resume_parser.parser_agent.ResumeProfile).
            job: the target job posting's facts.
            contact: the resolved contact to address the email to.
            match_score: resume-to-job similarity score (0.0-1.0) from the
                Resume Match Agent (app.agents.resume_match.scorer).

        Returns:
            A fully validated `OutreachContext`.

        Raises:
            OutreachContextBuildError: if any required field is missing,
                empty, or out of range. The error message enumerates every
                validation failure found, not just the first, so a caller
                can fix all issues in one pass.
        """
        self._logger.debug(
            "Building outreach context",
            extra={
                "company_name": job.company_name if job else None,
                "role_title": job.role_title if job else None,
                "contact_email": contact.email if contact else None,
                "match_score": match_score,
            },
        )

        errors = [
            *self._validate_resume_profile(resume_profile),
            *self._validate_job(job),
            *self._validate_contact(contact),
            *self._validate_match_score(match_score),
        ]

        if errors:
            self._logger.warning(
                "Outreach context build failed validation",
                extra={"validation_errors": errors},
            )
            raise OutreachContextBuildError(
                "Cannot build outreach context due to the following issue(s): "
                + "; ".join(errors)
            )

        context = OutreachContext(
            resume_profile=resume_profile,
            job=job,
            contact=contact,
            match_score=match_score,
        )

        self._logger.info(
            "Outreach context built successfully",
            extra={
                "company_name": job.company_name,
                "role_title": job.role_title,
                "contact_email": contact.email,
                "contact_confidence_level": contact.confidence_level,
                "match_score": match_score,
            },
        )

        return context

    def _validate_resume_profile(self, resume_profile: ResumeProfile) -> list[str]:
        if resume_profile is None:
            return ["resume_profile is required"]

        errors: list[str] = []
        if not resume_profile.skills and not resume_profile.experience:
            errors.append(
                "resume_profile must have at least skills or experience to "
                "ground a personalized email"
            )
        return errors

    def _validate_job(self, job: JobContext) -> list[str]:
        if job is None:
            return ["job is required"]

        errors: list[str] = []
        if not job.company_name or not job.company_name.strip():
            errors.append("job.company_name must not be empty")
        if not job.role_title or not job.role_title.strip():
            errors.append("job.role_title must not be empty")
        if not job.jd_url or not job.jd_url.strip():
            errors.append("job.jd_url must not be empty")
        if not job.jd_snapshot_text or not job.jd_snapshot_text.strip():
            errors.append("job.jd_snapshot_text must not be empty")
        return errors

    def _validate_contact(self, contact: ContactContext) -> list[str]:
        if contact is None:
            return ["contact is required"]

        errors: list[str] = []
        if not contact.name or not contact.name.strip():
            errors.append("contact.name must not be empty")
        if not contact.email or "@" not in contact.email:
            errors.append("contact.email must be a non-empty, syntactically valid address")
        if not contact.confidence_level or not contact.confidence_level.strip():
            errors.append("contact.confidence_level must not be empty")
        return errors

    def _validate_match_score(self, match_score: float) -> list[str]:
        if match_score is None:
            return ["match_score is required"]
        if not 0.0 <= match_score <= 1.0:
            return [f"match_score must be between 0.0 and 1.0, got {match_score}"]
        return []