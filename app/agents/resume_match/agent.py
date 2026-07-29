"""
Resume Match Agent.

Implements: PRD §5 (Resume Match Agent (RAG) — computes similarity between
resume and job description embeddings, surfaces ranked matching postings),
§8.4 (Resume Match Agent filters postings by a configurable minimum
similarity threshold).
Roadmap: Epic 4 - Resume Match Agent (RAG), Story 3 - Resume Match Agent
Orchestration, Task 1.

Orchestrates JD embedding (app/agents/resume_match/jd_embedder.py), resume
embedding lookup (via `VectorStoreProvider`, app/vectorstore/chroma_client.py),
similarity scoring and threshold filtering (app/agents/resume_match/scorer.py),
and persistence (app/agents/resume_match/repository.py) as a single callable
use case. Depends only on these collaborators' abstractions/public
interfaces, never on concrete infrastructure directly (Dependency Inversion,
per docs/architecture.md). Per docs/architecture.md §3, this agent does not
call other agents (e.g. Job Scout, Contact Finder) directly — cross-agent
coordination belongs to the graph layer (Epic 7).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.agents.resume_match.jd_embedder import JdEmbedder, JdEmbeddingError
from app.agents.resume_match.repository import ResumeMatchRepository
from app.agents.resume_match.scorer import ResumeMatchScorer, ScoredJobPosting
from app.vectorstore.chroma_client import VectorStoreError, VectorStoreProvider


class ResumeMatchAgentError(Exception):
    """Raised when the Resume Match Agent cannot complete a matching run."""


@dataclass(frozen=True)
class CandidateJobPosting:
    """A job posting to be scored against a resume in this run."""

    job_posting_id: str
    jd_hash: str
    jd_snapshot_text: str


class ResumeMatchAgent:
    """Matches a single resume against a batch of candidate job postings.

    Coordinates: (1) look up the resume's cached embedding, (2) embed each
    candidate JD (cache-aware, via `JdEmbedder`), (3) score similarity and
    filter by the configured minimum threshold (via `ResumeMatchScorer`),
    and (4) persist the resulting matches (via `ResumeMatchRepository`).
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        resume_collection_name: str,
        jd_embedder: JdEmbedder,
        scorer: ResumeMatchScorer,
        repository: ResumeMatchRepository,
    ) -> None:
        self._vector_store = vector_store
        self._resume_collection_name = resume_collection_name
        self._jd_embedder = jd_embedder
        self._scorer = scorer
        self._repository = repository

    def match(
        self,
        resume_id: uuid.UUID,
        resume_file_hash: str,
        candidate_postings: list[CandidateJobPosting],
    ) -> list[ScoredJobPosting]:
        """Match a resume against candidate postings and persist the results.

        Args:
            resume_id: id of the `Resume` row being matched.
            resume_file_hash: content hash of the resume, used as its
                embedding cache key (must already be embedded by
                `app.agents.resume_parser.embedder.ResumeEmbedder`).
            candidate_postings: postings to score against this resume.

        Returns:
            Postings scoring at or above the configured minimum threshold,
            sorted by descending similarity score, already persisted.

        Raises:
            ResumeMatchAgentError: if the resume has no cached embedding, or
                if JD embedding/scoring fails for any candidate.
        """
        resume_embedding = self._get_resume_embedding(resume_file_hash)

        scored_candidates: list[tuple[str, list[float]]] = []
        for posting in candidate_postings:
            jd_embedding = self._embed_candidate(posting)
            scored_candidates.append((posting.job_posting_id, jd_embedding))

        try:
            scored_postings = self._scorer.score_and_filter(
                resume_embedding=resume_embedding,
                candidate_postings=scored_candidates,
            )
        except Exception as exc:  # noqa: BLE001 - normalize scoring failure
            raise ResumeMatchAgentError(
                f"Failed to score candidate postings against resume '{resume_id}'."
            ) from exc

        self._repository.save_matches(resume_id=resume_id, scored_postings=scored_postings)

        return scored_postings

    def _get_resume_embedding(self, resume_file_hash: str) -> list[float]:
        try:
            cached = self._vector_store.get(self._resume_collection_name, resume_file_hash)
        except VectorStoreError as exc:
            raise ResumeMatchAgentError(
                f"Failed to look up embedding for resume '{resume_file_hash}'."
            ) from exc

        if cached is None or not cached.get("embedding"):
            raise ResumeMatchAgentError(
                f"No cached embedding found for resume '{resume_file_hash}'. "
                "The resume must be embedded before matching."
            )

        return cached["embedding"]

    def _embed_candidate(self, posting: CandidateJobPosting) -> list[float]:
        try:
            return self._jd_embedder.embed_job_description(
                job_posting_id=posting.job_posting_id,
                jd_hash=posting.jd_hash,
                jd_snapshot_text=posting.jd_snapshot_text,
            )
        except JdEmbeddingError as exc:
            raise ResumeMatchAgentError(
                f"Failed to embed job description for posting '{posting.job_posting_id}'."
            ) from exc