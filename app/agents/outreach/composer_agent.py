"""
Outreach email composer agent.

Implements: PRD §5 (Outreach Composer + Validator Agent — drafts a
personalized email using resume + job description + contact info), §6.4
(Outreach personalization must be grounded in the candidate's actual resume
and the job posting's actual JD; must not fabricate experience or
achievements), §6a.3 (Modular LLM Pipeline — email drafting routed to
Llama 3.1 8B, the "generation" task).
Roadmap: Epic 6 - Outreach Composer + Validator Agent, Story 2 - Email
Composer, Task 1.

Consumes an `OutreachContext` (app/agents/outreach/context_builder.py) and
produces a structured, personalized outreach email via an injected
`LLMProvider` (app/llm/ollama_client.py). Depends only on the `LLMProvider`
abstraction, never on a concrete LLM backend (Dependency Inversion, per
docs/architecture.md). The prompt template is injectable
(`OutreachPromptTemplate`), so tone/structure can be tuned without modifying
this agent's code (Open/Closed, per docs/architecture.md). Fact-checking the
composed email against the resume/JD (ensuring no fabricated experience) is
a separate, later responsibility
(app/agents/outreach/validator_agent.py, Epic 6, Story 3) — this agent's
system prompt instructs the model not to fabricate, but does not itself
verify the output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agents.outreach.context_builder import OutreachContext
from app.llm.ollama_client import LLMProvider, LLMProviderError

_DEFAULT_SYSTEM_PROMPT = """\
You are an assistant that drafts concise, professional job-outreach emails on
behalf of a job candidate. You are given the candidate's resume profile, a
specific job posting, the resume's match score for that job, and the name
and title of the person the email will be sent to.

Ground every claim about the candidate strictly in the provided resume
profile. Do not invent, exaggerate, or imply skills, experience, employers,
titles, achievements, or metrics that are not explicitly present in the
resume profile. If the resume profile does not support a strong claim,
write a more modest, accurate one instead. Do not fabricate any detail about
the company or role beyond what is stated in the job description text
provided.

Keep the tone professional, warm, and concise. The email should read as
written by a real candidate, not a template. Avoid generic filler phrases.
"""

_DEFAULT_INSTRUCTION_TEMPLATE = """\
Draft a personalized outreach email using the information below.

Candidate resume profile:
- Target roles: {target_roles}
- Skills: {skills}
- Experience: {experience_summary}

Job posting:
- Company: {company_name}
- Role title: {role_title}
- Job description excerpt: {jd_excerpt}
- Resume match score: {match_score:.2f} (0.0-1.0 scale)

Recipient:
- Name: {contact_name}
- Title: {contact_title}

Return a JSON object with exactly these keys:
{{
  "subject": string,
  "greeting": string,
  "body": string,
  "closing": string
}}

"greeting" should address the recipient by name (e.g. "Hi Jane,"). "body"
should be 2-4 short paragraphs: why the candidate is reaching out, how their
actual background (from the resume profile above) fits this specific role,
and a clear, low-pressure ask (e.g. a brief call or considering their
application). "closing" should be a short sign-off line (e.g. "Best regards,")
without the candidate's name, since the name is appended separately.
"""

_JD_EXCERPT_MAX_CHARS = 1500


class OutreachComposerError(Exception):
    """Raised when a personalized outreach email cannot be composed."""


@dataclass(frozen=True)
class ComposedEmail:
    """A structured, personalized outreach email draft."""

    subject: str
    greeting: str
    body: str
    closing: str


@dataclass(frozen=True)
class OutreachPromptTemplate:
    """Configurable prompt template for the outreach composer.

    `instruction_template` must be a `str.format`-compatible template
    accepting the placeholders used in `_DEFAULT_INSTRUCTION_TEMPLATE`:
    target_roles, skills, experience_summary, company_name, role_title,
    jd_excerpt, match_score, contact_name, contact_title.
    """

    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    instruction_template: str = _DEFAULT_INSTRUCTION_TEMPLATE


class OutreachComposerAgent:
    """Drafts a personalized outreach email from an `OutreachContext`.

    Depends only on `LLMProvider` (Dependency Inversion, per
    docs/architecture.md), so the underlying model/backend can be swapped
    via configuration without changing this agent. The prompt template is
    injected as well, allowing tone/structure experimentation without a
    code change (Open/Closed).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_template: OutreachPromptTemplate | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._llm = llm_provider
        self._prompt_template = prompt_template or OutreachPromptTemplate()
        self._logger = logger or logging.getLogger(__name__)

    def compose(self, context: OutreachContext) -> ComposedEmail:
        """Compose a personalized outreach email for the given context.

        Args:
            context: fully validated `OutreachContext`
                (app.agents.outreach.context_builder.OutreachContext).

        Returns:
            A `ComposedEmail` with subject, greeting, body, and closing.

        Raises:
            OutreachComposerError: if the LLM fails to produce a usable
                structured draft.
        """
        prompt = self._build_prompt(context)

        self._logger.info(
            "Composing outreach email",
            extra={
                "company_name": context.job.company_name,
                "role_title": context.job.role_title,
                "contact_email": context.contact.email,
                "match_score": context.match_score,
            },
        )

        try:
            raw = self._llm.generate_json(
                task="generation",
                prompt=prompt,
                system=self._prompt_template.system_prompt,
            )
        except LLMProviderError as exc:
            self._logger.warning(
                "LLM failed to compose outreach email",
                extra={
                    "company_name": context.job.company_name,
                    "role_title": context.job.role_title,
                },
            )
            raise OutreachComposerError(
                f"LLM failed to compose an outreach email for "
                f"'{context.job.company_name}' / '{context.job.role_title}'."
            ) from exc

        composed = self._to_composed_email(raw)

        self._logger.info(
            "Outreach email composed",
            extra={
                "company_name": context.job.company_name,
                "role_title": context.job.role_title,
                "subject": composed.subject,
            },
        )

        return composed

    def _build_prompt(self, context: OutreachContext) -> str:
        target_roles = ", ".join(context.resume_profile.target_roles) or "Not specified"
        skills = ", ".join(context.resume_profile.skills) or "Not specified"

        experience_summary = (
            "; ".join(
                f"{entry.title} at {entry.company}"
                + (f" ({entry.summary})" if entry.summary else "")
                for entry in context.resume_profile.experience
            )
            or "Not specified"
        )

        jd_excerpt = context.job.jd_snapshot_text.strip()[:_JD_EXCERPT_MAX_CHARS]

        return self._prompt_template.instruction_template.format(
            target_roles=target_roles,
            skills=skills,
            experience_summary=experience_summary,
            company_name=context.job.company_name,
            role_title=context.job.role_title,
            jd_excerpt=jd_excerpt,
            match_score=context.match_score,
            contact_name=context.contact.name,
            contact_title=context.contact.title or "Not specified",
        )

    def _to_composed_email(self, raw: dict) -> ComposedEmail:
        subject = str(raw.get("subject", "")).strip()
        greeting = str(raw.get("greeting", "")).strip()
        body = str(raw.get("body", "")).strip()
        closing = str(raw.get("closing", "")).strip()

        missing_fields = [
            field_name
            for field_name, value in (
                ("subject", subject),
                ("greeting", greeting),
                ("body", body),
                ("closing", closing),
            )
            if not value
        ]

        if missing_fields:
            raise OutreachComposerError(
                f"LLM response is missing required field(s): {', '.join(missing_fields)}."
            )

        return ComposedEmail(subject=subject, greeting=greeting, body=body, closing=closing)