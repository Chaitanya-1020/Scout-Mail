"""
Resume match repository.

Implements: PRD §5 (Resume Match Agent (RAG) — computes and surfaces ranked
postings by similarity), §8.4 (Resume Match Agent filters postings by a
configurable minimum similarity threshold).
Roadmap: Epic 4 - Resume Match Agent (RAG), Story 2 - Similarity Scoring,
Task 2.

Persists similarity scores produced by `ResumeMatchScorer`
(app/agents/resume_match/scorer.py) into the `ResumeJobMatch` ORM model
(app/db/models.py), and provides ranked-retrieval for downstream consumers
(e.g. Contact Finder Agent, dashboard). Upserts by (resume_id, job_posting_id)
so re-scoring an unchanged pair across runs updates the existing row rather
than creating duplicates, consistent with the unique constraint already
defined on `ResumeJobMatch`.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.resume_match.scorer import ScoredJobPosting
from app.db.models import ResumeJobMatch


class ResumeMatchRepositoryError(Exception):
    """Raised when a resume-job match cannot be persisted or retrieved."""


class ResumeMatchRepository:
    """Persistence for resume-to-job-posting similarity scores."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_matches(
        self, resume_id: uuid.UUID, scored_postings: list[ScoredJobPosting]
    ) -> list[ResumeJobMatch]:
        """Persist a batch of scored postings for a single resume.

        Upserts by (resume_id, job_posting_id): an existing match is updated
        with the new score rather than duplicated, since re-running the
        match agent against an unchanged resume/JD pair should not create
        additional rows.

        Returns:
            The list of `ResumeJobMatch` rows, in the same order as
            `scored_postings`.
        """
        results: list[ResumeJobMatch] = []

        for scored in scored_postings:
            results.append(self._save_one(resume_id, scored))

        self._session.flush()
        return results

    def _save_one(
        self, resume_id: uuid.UUID, scored: ScoredJobPosting
    ) -> ResumeJobMatch:
        job_posting_id = uuid.UUID(str(scored.job_posting_id))

        existing = self._session.execute(
            select(ResumeJobMatch).where(
                ResumeJobMatch.resume_id == resume_id,
                ResumeJobMatch.job_posting_id == job_posting_id,
            )
        ).scalar_one_or_none()

        if existing is not None:
            existing.similarity_score = scored.similarity_score
            self._session.flush()
            return existing

        match = ResumeJobMatch(
            resume_id=resume_id,
            job_posting_id=job_posting_id,
            similarity_score=scored.similarity_score,
        )
        self._session.add(match)
        self._session.flush()
        return match

    def get_ranked_matches(
        self, resume_id: uuid.UUID, limit: int | None = None
    ) -> list[ResumeJobMatch]:
        """Return this resume's matches ordered by descending similarity score.

        Args:
            resume_id: id of the resume to fetch matches for.
            limit: optional maximum number of matches to return.
        """
        query = (
            select(ResumeJobMatch)
            .where(ResumeJobMatch.resume_id == resume_id)
            .order_by(ResumeJobMatch.similarity_score.desc())
        )
        if limit is not None:
            query = query.limit(limit)

        return list(self._session.execute(query).scalars().all())

    def get_match(
        self, resume_id: uuid.UUID, job_posting_id: uuid.UUID
    ) -> ResumeJobMatch | None:
        """Fetch a single resume-job match, if it exists."""
        return self._session.execute(
            select(ResumeJobMatch).where(
                ResumeJobMatch.resume_id == resume_id,
                ResumeJobMatch.job_posting_id == job_posting_id,
            )
        ).scalar_one_or_none()