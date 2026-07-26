"""
Generic cache provider.

Implements: PRD §6a.3 (Cache aggressively to cut redundant local-LLM/API
work: parsed resumes, JD embeddings, resolved company domains, and
previously-verified contacts are all cached, keyed by domain/company/JD hash,
and reused across runs), §9 (Non-Functional Requirements — Extensibility).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 2 - Structured
Profile Extraction, Task 3.

Defines the `CacheProvider` interface (domain-facing abstraction) and a
concrete Postgres-backed implementation, keyed by an arbitrary string key
(e.g. a resume file hash, JD hash, or company domain). Per
docs/architecture.md, `app/agents/*` depend on `CacheProvider`, never on a
concrete cache backend directly — concrete wiring happens only at the
composition root.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.db.base import Base


class CacheError(Exception):
    """Raised when a cache operation fails."""


class CacheProvider(ABC):
    """Abstraction over a namespaced key-value cache.

    `namespace` scopes keys logically (e.g. "resume_profile", "jd_embedding",
    "company_domain", "contact") so different cache consumers cannot collide
    on the same key.
    """

    @abstractmethod
    def get(self, namespace: str, key: str) -> dict | None:
        """Return the cached JSON-serializable value for `key`, or None if absent."""
        raise NotImplementedError

    @abstractmethod
    def set(self, namespace: str, key: str, value: dict) -> None:
        """Store a JSON-serializable value under `key`, overwriting any prior value."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, namespace: str, key: str) -> bool:
        """Return True if a value is cached under `key`."""
        raise NotImplementedError


class CacheEntry(Base):
    """Persisted cache entry, namespaced and keyed for reuse across runs."""

    __tablename__ = "cache_entries"
    __table_args__ = (
        UniqueConstraint("namespace", "cache_key", name="uq_cache_namespace_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(256), nullable=False)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class PostgresCache(CacheProvider):
    """Concrete `CacheProvider` implementation backed by a Postgres table.

    Reuses the application's existing Postgres database (PRD §7) instead of
    introducing a separate cache store, keeping the system free-tier
    compatible (no additional paid infrastructure).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, namespace: str, key: str) -> dict | None:
        entry = self._session.execute(
            select(CacheEntry).where(
                CacheEntry.namespace == namespace, CacheEntry.cache_key == key
            )
        ).scalar_one_or_none()

        if entry is None:
            return None

        try:
            return json.loads(entry.value_json)
        except json.JSONDecodeError as exc:
            raise CacheError(
                f"Corrupt cache entry for namespace='{namespace}', key='{key}'."
            ) from exc

    def set(self, namespace: str, key: str, value: dict) -> None:
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise CacheError(
                f"Value for namespace='{namespace}', key='{key}' is not JSON-serializable."
            ) from exc

        entry = self._session.execute(
            select(CacheEntry).where(
                CacheEntry.namespace == namespace, CacheEntry.cache_key == key
            )
        ).scalar_one_or_none()

        if entry is None:
            entry = CacheEntry(namespace=namespace, cache_key=key, value_json=serialized)
            self._session.add(entry)
        else:
            entry.value_json = serialized

        self._session.flush()

    def exists(self, namespace: str, key: str) -> bool:
        result = self._session.execute(
            select(CacheEntry.id).where(
                CacheEntry.namespace == namespace, CacheEntry.cache_key == key
            )
        ).scalar_one_or_none()
        return result is not None