"""
Resume parser agent — structured profile extraction.

Implements: PRD §8.1 (Upload/parse resume (PDF/DOCX) → structured profile),
§6a.3 (Modular LLM Pipeline — Parsing/extraction routed to Phi-3 Mini, fast,
cheap, structured-output friendly).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 2 - Structured
Profile Extraction, Task 2.

Turns raw resume text into a structured profile (skills, experience, target
roles) using an injected `LLMProvider` (app/llm/ollama_client.py). Per
docs/architecture.md, this module depends on the `LLMProvider` abstraction,
not on `OllamaClient` concretely — the concrete client is instantiated and
injected at the composition root (app/main.py / app/graph/runner.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.ollama_client import LLMProvider, LLMProviderError

_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise resume-parsing assistant. Given raw resume text, extract a
structured profile. Do not invent information that is not present in the
text. If a field cannot be determined, use an empty list or null as
appropriate. Preserve exact wording for job titles and company names as they
appear in the resume.
"""

_EXTRACTION_JSON_SCHEMA_HINT = """\
Return a JSON object with exactly these keys:
{
  "full_name": string or null,
  "email": string or null,
  "phone": string or null,
  "skills": [string, ...],
  "target_roles": [string, ...],
  "experience": [
    {
      "company": string,
      "title": string,
      "start_date": string or null,
      "end_date": string or null,
      "summary": string or null
    }
  ],
  "education": [
    {
      "institution": string,
      "degree": string or null,
      "field_of_study": string or null,
      "end_date": string or null
    }
  ],
  "years_of_experience": number or null
}
"""


class ResumeParsingError(Exception):
    """Raised when a resume cannot be parsed into a structured profile."""


@dataclass(frozen=True)
class ExperienceEntry:
    company: str
    title: str
    start_date: str | None = None
    end_date: str | None = None
    summary: str | None = None


@dataclass(frozen=True)
class EducationEntry:
    institution: str
    degree: str | None = None
    field_of_study: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class ResumeProfile:
    """Structured candidate profile derived from resume text."""

    full_name: str | None
    email: str | None
    phone: str | None
    skills: list[str] = field(default_factory=list)
    target_roles: list[str] = field(default_factory=list)
    experience: list[ExperienceEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    years_of_experience: float | None = None

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON/DB storage
        (e.g. `Resume.parsed_profile`)."""
        return {
            "full_name": self.full_name,
            "email": self.email,
            "phone": self.phone,
            "skills": list(self.skills),
            "target_roles": list(self.target_roles),
            "experience": [
                {
                    "company": e.company,
                    "title": e.title,
                    "start_date": e.start_date,
                    "end_date": e.end_date,
                    "summary": e.summary,
                }
                for e in self.experience
            ],
            "education": [
                {
                    "institution": ed.institution,
                    "degree": ed.degree,
                    "field_of_study": ed.field_of_study,
                    "end_date": ed.end_date,
                }
                for ed in self.education
            ],
            "years_of_experience": self.years_of_experience,
        }


class ResumeParserAgent:
    """Extracts a structured `ResumeProfile` from raw resume text via an LLM.

    Depends only on the `LLMProvider` interface (Dependency Inversion), so
    the underlying model/backend can be swapped via configuration without
    changing this agent (per docs/architecture.md — Open/Closed, Extensibility).
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def parse(self, resume_text: str) -> ResumeProfile:
        """Parse raw resume text into a structured profile.

        Raises:
            ResumeParsingError: if the text is empty or the LLM output cannot
                be turned into a valid profile.
        """
        if not resume_text or not resume_text.strip():
            raise ResumeParsingError("Cannot parse an empty resume text.")

        prompt = (
            f"{_EXTRACTION_JSON_SCHEMA_HINT}\n\n"
            f"Resume text:\n\"\"\"\n{resume_text.strip()}\n\"\"\""
        )

        try:
            raw = self._llm.generate_json(
                task="extraction",
                prompt=prompt,
                system=_EXTRACTION_SYSTEM_PROMPT,
            )
        except LLMProviderError as exc:
            raise ResumeParsingError(
                "LLM failed to produce a structured resume profile."
            ) from exc

        return self._to_profile(raw)

    def _to_profile(self, raw: dict) -> ResumeProfile:
        try:
            experience = [
                ExperienceEntry(
                    company=str(item.get("company", "")).strip(),
                    title=str(item.get("title", "")).strip(),
                    start_date=self._clean_optional_str(item.get("start_date")),
                    end_date=self._clean_optional_str(item.get("end_date")),
                    summary=self._clean_optional_str(item.get("summary")),
                )
                for item in raw.get("experience") or []
                if isinstance(item, dict) and item.get("company") and item.get("title")
            ]

            education = [
                EducationEntry(
                    institution=str(item.get("institution", "")).strip(),
                    degree=self._clean_optional_str(item.get("degree")),
                    field_of_study=self._clean_optional_str(item.get("field_of_study")),
                    end_date=self._clean_optional_str(item.get("end_date")),
                )
                for item in raw.get("education") or []
                if isinstance(item, dict) and item.get("institution")
            ]

            skills = [str(s).strip() for s in raw.get("skills") or [] if str(s).strip()]
            target_roles = [
                str(r).strip() for r in raw.get("target_roles") or [] if str(r).strip()
            ]

            years_of_experience = raw.get("years_of_experience")
            if years_of_experience is not None:
                try:
                    years_of_experience = float(years_of_experience)
                except (TypeError, ValueError):
                    years_of_experience = None

            return ResumeProfile(
                full_name=self._clean_optional_str(raw.get("full_name")),
                email=self._clean_optional_str(raw.get("email")),
                phone=self._clean_optional_str(raw.get("phone")),
                skills=skills,
                target_roles=target_roles,
                experience=experience,
                education=education,
                years_of_experience=years_of_experience,
            )
        except (AttributeError, TypeError) as exc:
            raise ResumeParsingError(
                "LLM output did not match the expected resume profile structure."
            ) from exc

    @staticmethod
    def _clean_optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None