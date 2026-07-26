"""
Job Scout repository.

Implements: PRD §8.3 (Job Scout Agent pulls new postings on a schedule),
§6.2 (Role Title Accuracy — store the JD URL + snapshot text as the
source-of-truth reference shown to the user during approval), §6a.3
(Caching — resolved company domains and JD content are hashed/keyed for
reuse across runs).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 5 - Job Scout Agent
Orchestration, Task 3.

Persists `RawJobPosting` values fetched by `JobScoutAgent`
(app/agents/job_scout/agent.py) into the `JobPosting` ORM model
(app/db/models.py). Deduplicates by a hash of the JD snapshot text so
re-fetching an unchanged posting across scheduled runs does not create
duplicate rows (PRD §6a.3 — cache/reuse keyed by JD hash).
"""

from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.connectors.base import RawJobPosting
from app.db.models import JobPosting


class JobScoutRepositoryError(Exception):
    """Raised when a job posting cannot be persisted."""


def compute_jd_hash(jd_snapshot_text: str) -> str:
    """Compute a stable content hash for a JD snapshot, used for dedup and
    as the cache key referenced elsewhere in the pipeline (PRD §6a.3).
    """
    return hashlib.sha256(jd_snapshot_text.strip().encode("utf-8")).hexdigest()


class JobScoutRepository:
    """Persistence for raw job postings fetched by the Job Scout Agent.

    Each `RawJobPosting` is stored verbatim — `role_title` and
    `jd_snapshot_text` are never paraphrased or normalized here (PRD §6.2)
    — and deduplicated by JD content hash so identical postings discovered
    across multiple scheduled runs are not duplicated.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_postings(self, postings: list[RawJobPosting]) -> list[JobPosting]:
        """Persist a batch of raw postings, skipping any already stored under
        the same JD content hash.

        Returns:
            The list of `JobPosting` rows corresponding to `postings`, in
            the same order — either freshly created or the pre-existing
            matching row on a dedup hit.

        Raises:
            JobScoutRepositoryError: if a posting is missing required fields.
        """
        results: list[JobPosting] = []

        for posting in postings:
            results.append(self._save_one(posting))

        self._session.flush()
        return results

    def _save_one(self, posting: RawJobPosting) -> JobPosting:
        if not posting.role_title.strip() or not posting.jd_snapshot_text.strip():
            raise JobScoutRepositoryError(
                f"Posting from '{posting.source_connector}' "
                f"(external_id='{posting.external_id}') is missing role_title "
                "or jd_snapshot_text."
            )

        jd_hash = compute_jd_hash(posting.jd_snapshot_text)

        existing = self._session.execute(
            select(JobPosting).where(JobPosting.jd_hash == jd_hash)
        ).scalar_one_or_none()

        if existing is not None:
            return existing

        job_posting = JobPosting(
            source_connector=posting.source_connector,
            external_id=posting.external_id,
            company_name=posting.company_name,
            company_domain=posting.company_domain,
            role_title=posting.role_title,
            jd_url=posting.jd_url,
            jd_snapshot_text=posting.jd_snapshot_text,
            jd_hash=jd_hash,
            posted_at=posting.posted_at,
        )
        self._session.add(job_posting)
        self._session.flush()
        return job_posting

    def get_by_jd_hash(self, jd_hash: str) -> JobPosting | None:
        """Fetch a stored `JobPosting` by its JD content hash, if present."""
        return self._session.execute(
            select(JobPosting).where(JobPosting.jd_hash == jd_hash)
        ).scalar_one_or_none()