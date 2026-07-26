"""
Resume parser repository.

Implements: PRD §8.1 (Upload/parse resume (PDF/DOCX) → structured profile),
§6a.3 (Cache parsed resumes, keyed by resume hash, reused across runs).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 2 - Structured
Profile Extraction, Task 3.

Persists parsed resume profiles to `Resume.parsed_profile` and provides a
cache-aware lookup keyed by file hash, so re-uploading or re-processing an
identical resume skips a redundant LLM extraction call. Depends on the
`CacheProvider` abstraction (app/services/cache.py), not a concrete cache
backend (Dependency Inversion, per docs/architecture.md).
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.resume_parser.parser_agent import ResumeProfile
from app.db.models import Resume
from app.services.cache import CacheProvider

_CACHE_NAMESPACE = "resume_profile"


class ResumeRepositoryError(Exception):
    """Raised when a resume record cannot be found or persisted."""


class ResumeParserRepository:
    """Persistence and caching for parsed resume profiles.

    Cache-first lookups avoid redundant local-LLM extraction work per
    PRD §6a.3: a resume's parsed profile is cached by its file hash, so
    repeated runs against the same resume reuse the cached structured
    profile instead of re-invoking the extraction agent.
    """

    def __init__(self, session: Session, cache: CacheProvider) -> None:
        self._session = session
        self._cache = cache

    def get_cached_profile(self, file_hash: str) -> dict | None:
        """Return a previously cached parsed profile for this resume hash, if any."""
        return self._cache.get(_CACHE_NAMESPACE, file_hash)

    def save_parsed_profile(self, resume_id, profile: ResumeProfile) -> None:  # noqa: ANN001
        """Persist a structured profile onto its `Resume` record and cache it
        by file hash for reuse.

        Raises:
            ResumeRepositoryError: if no `Resume` exists with `resume_id`.
        """
        resume = self._session.get(Resume, resume_id)
        if resume is None:
            raise ResumeRepositoryError(f"No resume found with id '{resume_id}'.")

        profile_dict = profile.to_dict()
        resume.parsed_profile = profile_dict
        self._session.flush()

        self._cache.set(_CACHE_NAMESPACE, resume.file_hash, profile_dict)

    def get_resume_by_hash(self, file_hash: str) -> Resume | None:
        """Fetch a stored `Resume` record by its content hash."""
        return self._session.execute(
            select(Resume).where(Resume.file_hash == file_hash)
        ).scalar_one_or_none()