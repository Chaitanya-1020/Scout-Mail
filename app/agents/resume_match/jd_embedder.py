"""
Job description embedding agent.

Implements: PRD §5 (Resume Match Agent (RAG) — embeds resume + job
description into ChromaDB), §6a.3 (Caching — JD embeddings cached, keyed by
JD hash, reused across runs).
Roadmap: Epic 4 - Resume Match Agent (RAG), Story 1 - JD Embedding, Task 1.

Embeds a job posting's JD snapshot text into the vector store so the Resume
Match Agent (Epic 4, Story 2) can compute similarity against resume
embeddings. Depends only on the `VectorStoreProvider` abstraction
(app/vectorstore/chroma_client.py) and the `EmbeddingProvider` abstraction
already defined in app/agents/resume_parser/embedder.py — never on
`ChromaVectorStore` or a specific embedding backend concretely (Dependency
Inversion, per docs/architecture.md). Reuses `EmbeddingProvider` rather than
redefining an equivalent interface here, keeping one embedding-generation
abstraction in the codebase (Single Responsibility / no duplicate
abstractions, per docs/coding_guidelines.md).
"""

from __future__ import annotations

from app.agents.resume_parser.embedder import EmbeddingProvider, ResumeEmbeddingError
from app.vectorstore.chroma_client import VectorStoreError, VectorStoreProvider


class JdEmbeddingError(Exception):
    """Raised when a job description cannot be embedded into the vector store."""


class JdEmbedder:
    """Embeds a job posting's JD snapshot text into the vector store, keyed
    by the JD's content hash so re-embedding an unchanged posting is skipped
    (PRD §6a.3 — cache JD embeddings, reused across runs).
    """

    def __init__(
        self,
        vector_store: VectorStoreProvider,
        embedding_provider: EmbeddingProvider,
        collection_name: str,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._collection_name = collection_name

    def embed_job_description(
        self, job_posting_id: str, jd_hash: str, jd_snapshot_text: str
    ) -> list[float]:
        """Embed the JD snapshot text and upsert it into the vector store.

        Uses `jd_hash` as the vector store id so an unchanged JD (identical
        content, e.g. re-fetched on a later Job Scout run) is a cache hit —
        the existing vector is reused without invoking the embedding model
        again.

        Args:
            job_posting_id: id of the corresponding `JobPosting` row, stored
                as metadata for traceability.
            jd_hash: content hash of `jd_snapshot_text`
                (see `app.agents.job_scout.repository.compute_jd_hash`).
            jd_snapshot_text: verbatim JD text to embed (PRD §6.2 — never
                paraphrased before embedding).

        Returns:
            The embedding vector (freshly generated, or the cached one on
            hit).

        Raises:
            JdEmbeddingError: if embedding generation or storage fails.
        """
        if not jd_snapshot_text or not jd_snapshot_text.strip():
            raise JdEmbeddingError("Cannot embed an empty job description.")

        try:
            cached = self._vector_store.get(self._collection_name, jd_hash)
        except VectorStoreError as exc:
            raise JdEmbeddingError(
                f"Failed to check vector store cache for JD '{jd_hash}'."
            ) from exc

        if cached is not None and cached.get("embedding"):
            return cached["embedding"]

        try:
            embedding = self._embedding_provider.embed(jd_snapshot_text)
        except ResumeEmbeddingError as exc:
            raise JdEmbeddingError(
                f"Failed to generate embedding for JD '{jd_hash}'."
            ) from exc

        try:
            self._vector_store.upsert(
                collection_name=self._collection_name,
                ids=[jd_hash],
                embeddings=[embedding],
                documents=[jd_snapshot_text],
                metadatas=[{"job_posting_id": str(job_posting_id), "jd_hash": jd_hash}],
            )
        except VectorStoreError as exc:
            raise JdEmbeddingError(
                f"Failed to store embedding for JD '{jd_hash}'."
            ) from exc

        return embedding