"""
Resume embedding agent.

Implements: PRD §5 (Resume Match Agent (RAG) — embeds resume + job description
into ChromaDB), §6a.3 (Caching — resume embeddings cached, keyed by resume
hash, reused across runs).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 3 - Resume
Embedding, Task 1.

Embeds a resume's structured profile text into the vector store so the
Resume Match Agent (Epic 4) can compute similarity against job descriptions.
Depends only on the `VectorStoreProvider` abstraction (app/vectorstore/
chroma_client.py) and a local `EmbeddingProvider` abstraction defined here for
generating vectors — never on `ChromaVectorStore` or a specific embedding
backend concretely (Dependency Inversion, per docs/architecture.md).

Note: if a dedicated embedding-provider module is later preferred (mirroring
`app/llm/ollama_client.py`'s split of interface + implementation), the
`EmbeddingProvider` abstraction defined here could move there — kept local
for now since only this file was requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import ollama

from app.agents.resume_parser.parser_agent import ResumeProfile
from app.config import get_settings
from app.vectorstore.chroma_client import VectorStoreError, VectorStoreProvider


class ResumeEmbeddingError(Exception):
    """Raised when a resume cannot be embedded into the vector store."""


class EmbeddingProvider(ABC):
    """Abstraction over generating a vector embedding for a piece of text."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for `text`.

        Raises:
            ResumeEmbeddingError: if the embedding cannot be generated.
        """
        raise NotImplementedError


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Concrete `EmbeddingProvider` backed by a local Ollama embedding model."""

    def __init__(self, model: str = "nomic-embed-text") -> None:
        settings = get_settings()
        self._client = ollama.Client(host=settings.ollama_base_url)
        self._model = model

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ResumeEmbeddingError("Cannot embed empty text.")

        try:
            response = self._client.embeddings(model=self._model, prompt=text)
        except Exception as exc:  # noqa: BLE001 - normalize third-party failure
            raise ResumeEmbeddingError(
                f"Failed to generate embedding with model '{self._model}'."
            ) from exc

        embedding = response.get("embedding")
        if not embedding:
            raise ResumeEmbeddingError(
                f"Embedding model '{self._model}' returned an empty vector."
            )
        return embedding


def _profile_to_embedding_text(profile: ResumeProfile) -> str:
    """Flatten a structured resume profile into a single text blob suitable
    for embedding (skills, target roles, and experience summaries carry the
    most signal for job-match similarity).
    """
    parts: list[str] = []

    if profile.target_roles:
        parts.append("Target roles: " + ", ".join(profile.target_roles))

    if profile.skills:
        parts.append("Skills: " + ", ".join(profile.skills))

    for entry in profile.experience:
        experience_line = f"{entry.title} at {entry.company}"
        if entry.summary:
            experience_line += f": {entry.summary}"
        parts.append(experience_line)

    for entry in profile.education:
        education_line = entry.institution
        if entry.degree:
            education_line = f"{entry.degree}, {education_line}"
        if entry.field_of_study:
            education_line += f" ({entry.field_of_study})"
        parts.append(education_line)

    text = "\n".join(parts).strip()
    if not text:
        raise ResumeEmbeddingError(
            "Resume profile has no embeddable content (no skills, roles, or experience)."
        )
    return text


class ResumeEmbedder:
    """Embeds a resume's structured profile into the vector store, keyed by
    the resume's content hash so re-embedding an unchanged resume is skipped
    (PRD §6a.3 — cache resume embeddings, reused across runs).
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

    def embed_resume(
        self, resume_id: str, file_hash: str, profile: ResumeProfile
    ) -> list[float]:
        """Embed the resume profile and upsert it into the vector store.

        Uses `file_hash` as the vector store id so an unchanged resume
        (identical content) is a cache hit — the existing vector is reused
        without invoking the embedding model again.

        Returns:
            The embedding vector (freshly generated, or the cached one on
            hit).

        Raises:
            ResumeEmbeddingError: if embedding generation or storage fails.
        """
        try:
            cached = self._vector_store.get(self._collection_name, file_hash)
        except VectorStoreError as exc:
            raise ResumeEmbeddingError(
                f"Failed to check vector store cache for resume '{file_hash}'."
            ) from exc

        if cached is not None and cached.get("embedding"):
            return cached["embedding"]

        embedding_text = _profile_to_embedding_text(profile)
        embedding = self._embedding_provider.embed(embedding_text)

        try:
            self._vector_store.upsert(
                collection_name=self._collection_name,
                ids=[file_hash],
                embeddings=[embedding],
                documents=[embedding_text],
                metadatas=[{"resume_id": str(resume_id), "file_hash": file_hash}],
            )
        except VectorStoreError as exc:
            raise ResumeEmbeddingError(
                f"Failed to store embedding for resume '{file_hash}'."
            ) from exc

        return embedding