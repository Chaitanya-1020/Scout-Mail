"""
ChromaDB-backed vector store provider.

Implements: PRD §7 (Tech Stack — ChromaDB, local embedded mode, no hosting
cost), §6a.3 (Caching — JD/resume embeddings cached and reused across runs).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 3 - Local LLM + Vector
Store Bootstrap, Task 2.

Defines the `VectorStoreProvider` interface (domain-facing abstraction) and
its concrete Chroma implementation. Per docs/architecture.md, `app/agents/*`
must depend on `VectorStoreProvider`, never on `ChromaVectorStore` directly —
concrete wiring happens only in `app/main.py` / `app/graph/runner.py`
(Dependency Inversion).

Note: if a dedicated `app/vectorstore/base.py` is later preferred to host the
`VectorStoreProvider` interface separately (mirroring `connectors/base.py`),
that split should be its own task — this file keeps both together for now
since only this file was requested.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings


class VectorStoreError(Exception):
    """Raised when a vector store operation fails."""


class VectorStoreProvider(ABC):
    """Abstraction over a vector store, scoped to a named collection.

    Callers (Resume Match Agent, resume/JD embedders) reference collections
    by logical name (e.g. "resumes", "job_descriptions") rather than knowing
    about Chroma specifically.
    """

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update vectors in the named collection."""
        raise NotImplementedError

    @abstractmethod
    def get(self, collection_name: str, id_: str) -> dict[str, Any] | None:
        """Fetch a single stored record by id, or None if absent.

        Used for cache-style lookups (e.g. "is this JD hash already
        embedded?") per PRD §6a.3.
        """
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a similarity query against the named collection."""
        raise NotImplementedError


class ChromaVectorStore(VectorStoreProvider):
    """Concrete `VectorStoreProvider` implementation backed by local, embedded
    ChromaDB (persistent on disk, no external service required).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_cache: dict[str, Collection] = {}

    def _collection(self, name: str) -> Collection:
        if name not in self._collection_cache:
            try:
                self._collection_cache[name] = self._client.get_or_create_collection(
                    name=name
                )
            except Exception as exc:  # noqa: BLE001 - normalize backend failure
                raise VectorStoreError(
                    f"Failed to get or create Chroma collection '{name}'."
                ) from exc
        return self._collection_cache[name]

    def upsert(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not (len(ids) == len(embeddings) == len(documents)):
            raise VectorStoreError(
                "ids, embeddings, and documents must be the same length for upsert."
            )
        collection = self._collection(collection_name)
        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:  # noqa: BLE001 - normalize backend failure
            raise VectorStoreError(
                f"Failed to upsert {len(ids)} record(s) into collection "
                f"'{collection_name}'."
            ) from exc

    def get(self, collection_name: str, id_: str) -> dict[str, Any] | None:
        collection = self._collection(collection_name)
        try:
            result = collection.get(ids=[id_], include=["embeddings", "documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001 - normalize backend failure
            raise VectorStoreError(
                f"Failed to fetch id '{id_}' from collection '{collection_name}'."
            ) from exc

        ids = result.get("ids") or []
        if not ids:
            return None

        return {
            "id": ids[0],
            "embedding": (result.get("embeddings") or [None])[0],
            "document": (result.get("documents") or [None])[0],
            "metadata": (result.get("metadatas") or [None])[0],
        }

    def query(
        self,
        collection_name: str,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        collection = self._collection(collection_name)
        try:
            return collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
            )
        except Exception as exc:  # noqa: BLE001 - normalize backend failure
            raise VectorStoreError(
                f"Query against collection '{collection_name}' failed."
            ) from exc