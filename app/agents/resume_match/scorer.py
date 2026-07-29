"""
Resume-to-job similarity scorer.

Implements: PRD §5 (Resume Match Agent (RAG) — computes similarity between
resume and job description embeddings), §8.4 (Resume Match Agent filters
postings by a configurable minimum similarity threshold).
Roadmap: Epic 4 - Resume Match Agent (RAG), Story 2 - Similarity Scoring,
Task 1.

Computes cosine similarity between a resume embedding (Epic 2, Story 3) and
a job description embedding (Epic 4, Story 1), and filters candidate
postings against a configurable minimum score (`resume_match_min_score`,
app/config.py). Depends only on plain vectors (`list[float]`) passed in by
its caller — no dependency on `VectorStoreProvider`, `EmbeddingProvider`, or
any infrastructure concretely (Single Responsibility / Dependency Inversion,
per docs/architecture.md): this module is pure scoring math, with no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class SimilarityScoringError(Exception):
    """Raised when a similarity score cannot be computed."""


@dataclass(frozen=True)
class ScoredJobPosting:
    """A job posting's id paired with its similarity score against a resume."""

    job_posting_id: str
    similarity_score: float


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns:
        A float in [-1.0, 1.0]; higher means more similar.

    Raises:
        SimilarityScoringError: if the vectors are empty, of mismatched
            length, or either has zero magnitude.
    """
    if not vector_a or not vector_b:
        raise SimilarityScoringError("Cannot compute similarity for an empty vector.")

    if len(vector_a) != len(vector_b):
        raise SimilarityScoringError(
            f"Vector length mismatch: {len(vector_a)} vs {len(vector_b)}."
        )

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        raise SimilarityScoringError(
            "Cannot compute cosine similarity when a vector has zero magnitude."
        )

    return dot_product / (magnitude_a * magnitude_b)


class ResumeMatchScorer:
    """Scores and filters job postings by similarity to a resume embedding.

    The minimum-score threshold is injected at construction (sourced from
    `Settings.resume_match_min_score`, app/config.py, by the caller/composition
    root) rather than read from configuration directly, keeping this module
    free of any infrastructure dependency (per docs/architecture.md).
    """

    def __init__(self, min_score_threshold: float) -> None:
        if not 0.0 <= min_score_threshold <= 1.0:
            raise ValueError(
                f"min_score_threshold must be between 0.0 and 1.0, got {min_score_threshold}."
            )
        self._min_score_threshold = min_score_threshold

    def score(self, resume_embedding: list[float], jd_embedding: list[float]) -> float:
        """Return the similarity score between a resume and a single JD embedding."""
        return cosine_similarity(resume_embedding, jd_embedding)

    def score_and_filter(
        self,
        resume_embedding: list[float],
        candidate_postings: list[tuple[str, list[float]]],
    ) -> list[ScoredJobPosting]:
        """Score a resume against multiple candidate postings and return only
        those meeting or exceeding the configured minimum threshold.

        Args:
            resume_embedding: embedding vector for the candidate's resume.
            candidate_postings: list of (job_posting_id, jd_embedding) pairs.

        Returns:
            `ScoredJobPosting` entries for postings scoring at or above the
            threshold, sorted by descending similarity score.

        Raises:
            SimilarityScoringError: if any embedding is invalid (see
                `cosine_similarity`).
        """
        scored: list[ScoredJobPosting] = []

        for job_posting_id, jd_embedding in candidate_postings:
            similarity_score = self.score(resume_embedding, jd_embedding)
            if similarity_score >= self._min_score_threshold:
                scored.append(
                    ScoredJobPosting(
                        job_posting_id=job_posting_id,
                        similarity_score=similarity_score,
                    )
                )

        scored.sort(key=lambda s: s.similarity_score, reverse=True)
        return scored