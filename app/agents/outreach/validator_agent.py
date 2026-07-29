"""
Outreach email validator agent.

Implements: PRD §5 (Outreach Composer + Validator Agent — validates a
drafted email before it can be approved), §6.4 (Outreach personalization
must be grounded in the candidate's actual resume and the job posting's
actual JD; fabricated claims and missing personalization must be caught
before a human reviews the draft), §6a.3 (Modular LLM Pipeline — validation
routed to Mistral 7B, the "validation" task, a different model than
composition to reduce correlated errors).
Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 3 - Email
Validation, Task 1.

Validates a `ComposedEmail` (app/agents/outreach/composer_agent.py) against
its `OutreachContext` (app/agents/outreach/context_builder.py) using a
combination of deterministic string checks (recipient name, company name,
job title presence) and an injected `LLMProvider` for fact-checking
(fabricated claims not supported by the resume, unsupported match-score
claims). This agent never modifies the email it validates — it only
produces a structured `ValidationResult` for the caller (e.g. the outreach
repository, Epic 6 Story 4, or the review dashboard, Epic 8) to act on.
Depends only on the `LLMProvider` abstraction, never a concrete LLM backend
(Dependency Inversion, per docs/architecture.md).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from app.agents.outreach.composer_agent import ComposedEmail
from app.agents.outreach.context_builder import OutreachContext
from app.llm.ollama_client import LLMProvider, LLMProviderError

_FACT_CHECK_SYSTEM_PROMPT = """\
You are a strict fact-checking assistant. You are given a candidate's resume
profile facts and a drafted outreach email written on that candidate's
behalf. Identify any claim in the email about the candidate's skills,
experience, employers, job titles, achievements, or metrics that is NOT
directly supported by the resume profile facts provided. Also identify any
claim in the email about the candidate's resume match score (e.g. a stated
percentage or qualitative strength claim) that is not consistent with the
match score value provided. Do not flag generic, unverifiable statements of
interest or enthusiasm (e.g. "I'm excited about this role") as fabricated --
only flag specific factual claims about the candidate's background or match
strength that are not supported by the provided facts.
"""

_FACT_CHECK_INSTRUCTION_TEMPLATE = """\
Resume profile facts:
- Skills: {skills}
- Experience: {experience_summary}
- Target roles: {target_roles}

Actual resume match score: {match_score:.2f} (0.0-1.0 scale, not typically
disclosed to the recipient as a raw number or percentage).

Drafted email:
Subject: {subject}
{greeting}

{body}

{closing}

Return a JSON object with exactly these keys:
{{
  "fabricated_claims": [string, ...],
  "unsupported_score_claims": [string, ...]
}}
Each entry should be a short quote or paraphrase of the unsupported claim
from the email. Return empty lists if none are found.
"""

_MIN_PERSONALIZED_BODY_LENGTH_CHARS = 120


class ValidationIssueCategory(str, Enum):
    """The category of a single validation issue found in a drafted email."""

    MISSING_RECIPIENT_NAME = "missing_recipient_name"
    MISSING_COMPANY_NAME = "missing_company_name"
    MISSING_JOB_TITLE = "missing_job_title"
    MISSING_PERSONALIZATION = "missing_personalization"
    FABRICATED_CLAIM = "fabricated_claim"
    UNSUPPORTED_MATCH_SCORE_CLAIM = "unsupported_match_score_claim"


class ValidationSeverity(str, Enum):
    """Severity of a validation issue."""

    ERROR = "error"
    """Blocks the draft from being considered validated
    (`ValidationResult.passed = False`)."""
    WARNING = "warning"
    """Surfaced for human review but does not by itself fail validation."""


class OutreachValidationError(Exception):
    """Raised when validation cannot be attempted at all (e.g. the LLM fails
    in a way that leaves fact-checking entirely unavailable)."""


@dataclass(frozen=True)
class ValidationIssue:
    """A single issue found while validating a drafted outreach email."""

    category: ValidationIssueCategory
    severity: ValidationSeverity
    description: str


@dataclass(frozen=True)
class ValidationResult:
    """Structured outcome of validating a `ComposedEmail`.

    `passed` is True only if no `ERROR`-severity issues were found;
    `WARNING`-severity issues (e.g. missing personalization) do not block
    validation but are always included for the reviewer's benefit.
    """

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)


class OutreachValidatorAgent:
    """Validates a drafted outreach email against its source context.

    Combines cheap, deterministic checks (recipient name / company name /
    job title presence, minimum personalization length) with LLM-based fact
    checking (fabricated claims, unsupported match-score claims) via a
    validation-task model distinct from the composition-task model, so a
    single model's blind spot is less likely to pass its own fabrication
    through unnoticed (PRD §6a.3). This agent is read-only with respect to
    the email: it never returns a modified `ComposedEmail`, only a
    `ValidationResult`.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm_provider
        self._logger = logger or logging.getLogger(__name__)

    def validate(
        self, context: OutreachContext, composed_email: ComposedEmail
    ) -> ValidationResult:
        """Validate a composed email against its context.

        Args:
            context: the `OutreachContext` the email was drafted from.
            composed_email: the drafted email to validate.

        Returns:
            A `ValidationResult` enumerating every issue found. Never
            raises for issues found in the email itself — only
            `OutreachValidationError` is raised, and only if fact-checking
            could not be attempted at all.
        """
        self._logger.debug(
            "Validating outreach email",
            extra={
                "company_name": context.job.company_name,
                "role_title": context.job.role_title,
                "contact_email": context.contact.email,
            },
        )

        issues: list[ValidationIssue] = []
        issues.extend(self._check_personalization_markers(context, composed_email))
        issues.extend(self._check_facts_with_llm(context, composed_email))

        passed = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)

        self._logger.info(
            "Outreach email validation complete",
            extra={
                "company_name": context.job.company_name,
                "role_title": context.job.role_title,
                "passed": passed,
                "issue_count": len(issues),
            },
        )

        return ValidationResult(passed=passed, issues=issues)

    def _check_personalization_markers(
        self, context: OutreachContext, composed_email: ComposedEmail
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        greeting_and_body = f"{composed_email.greeting}\n{composed_email.body}"

        first_name = context.contact.name.strip().split()[0] if context.contact.name.strip() else ""
        if first_name and not self._contains_case_insensitive(greeting_and_body, first_name):
            issues.append(
                ValidationIssue(
                    category=ValidationIssueCategory.MISSING_RECIPIENT_NAME,
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Recipient's first name '{first_name}' does not appear in the "
                        "greeting or body."
                    ),
                )
            )

        full_email_text = (
            f"{composed_email.subject}\n{composed_email.greeting}\n"
            f"{composed_email.body}\n{composed_email.closing}"
        )

        if not self._contains_case_insensitive(full_email_text, context.job.company_name):
            issues.append(
                ValidationIssue(
                    category=ValidationIssueCategory.MISSING_COMPANY_NAME,
                    severity=ValidationSeverity.ERROR,
                    description=(
                        f"Company name '{context.job.company_name}' does not appear "
                        "anywhere in the email."
                    ),
                )
            )

        if not self._contains_case_insensitive(full_email_text, context.job.role_title):
            issues.append(
                ValidationIssue(
                    category=ValidationIssueCategory.MISSING_JOB_TITLE,
                    severity=ValidationSeverity.WARNING,
                    description=(
                        f"Role title '{context.job.role_title}' does not appear verbatim "
                        "in the email (may be paraphrased)."
                    ),
                )
            )

        if len(composed_email.body.strip()) < _MIN_PERSONALIZED_BODY_LENGTH_CHARS:
            issues.append(
                ValidationIssue(
                    category=ValidationIssueCategory.MISSING_PERSONALIZATION,
                    severity=ValidationSeverity.WARNING,
                    description=(
                        f"Email body is only {len(composed_email.body.strip())} characters, "
                        f"below the {_MIN_PERSONALIZED_BODY_LENGTH_CHARS}-character "
                        "personalization threshold."
                    ),
                )
            )

        return issues

    def _check_facts_with_llm(
        self, context: OutreachContext, composed_email: ComposedEmail
    ) -> list[ValidationIssue]:
        prompt = self._build_fact_check_prompt(context, composed_email)

        try:
            raw = self._llm.generate_json(
                task="validation",
                prompt=prompt,
                system=_FACT_CHECK_SYSTEM_PROMPT,
            )
        except LLMProviderError as exc:
            self._logger.warning(
                "LLM fact-check failed for outreach email; treating as unvalidated",
                extra={
                    "company_name": context.job.company_name,
                    "role_title": context.job.role_title,
                },
            )
            raise OutreachValidationError(
                f"Fact-checking failed for outreach email to "
                f"'{context.job.company_name}' / '{context.job.role_title}'."
            ) from exc

        issues: list[ValidationIssue] = []

        for claim in raw.get("fabricated_claims") or []:
            claim_text = str(claim).strip()
            if claim_text:
                issues.append(
                    ValidationIssue(
                        category=ValidationIssueCategory.FABRICATED_CLAIM,
                        severity=ValidationSeverity.ERROR,
                        description=f"Unsupported claim about the candidate: {claim_text}",
                    )
                )

        for claim in raw.get("unsupported_score_claims") or []:
            claim_text = str(claim).strip()
            if claim_text:
                issues.append(
                    ValidationIssue(
                        category=ValidationIssueCategory.UNSUPPORTED_MATCH_SCORE_CLAIM,
                        severity=ValidationSeverity.ERROR,
                        description=f"Unsupported match-score claim: {claim_text}",
                    )
                )

        return issues

    def _build_fact_check_prompt(
        self, context: OutreachContext, composed_email: ComposedEmail
    ) -> str:
        skills = ", ".join(context.resume_profile.skills) or "Not specified"
        target_roles = ", ".join(context.resume_profile.target_roles) or "Not specified"
        experience_summary = (
            "; ".join(
                f"{entry.title} at {entry.company}"
                + (f" ({entry.summary})" if entry.summary else "")
                for entry in context.resume_profile.experience
            )
            or "Not specified"
        )

        return _FACT_CHECK_INSTRUCTION_TEMPLATE.format(
            skills=skills,
            experience_summary=experience_summary,
            target_roles=target_roles,
            match_score=context.match_score,
            subject=composed_email.subject,
            greeting=composed_email.greeting,
            body=composed_email.body,
            closing=composed_email.closing,
        )

    def _contains_case_insensitive(self, haystack: str, needle: str) -> bool:
        if not needle.strip():
            return True
        pattern = re.escape(needle.strip())
        return re.search(pattern, haystack, re.IGNORECASE) is not None