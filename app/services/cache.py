"""
Generic cache provider.

Implements: PRD §6a.3 (Cache aggressively to cut redundant local-LLM/API
work: parsed resumes, JD embeddings, resolved company domains, and
previously-verified contacts are all cached, keyed by domain/company/JD hash,
and reused across runs), §9 (Non-Functional Requirements — Extensibility).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 2 - Structured
Profile Extraction, Task 3 (original); extended by Epic 5 - Contact Finder
Agent, Story 1 - Company Domain Resolution, Task 2 (TTL expiration, cache
invalidation, and a company→domain convenience API).

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
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, String, Text, UniqueConstraint, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.db.base import Base

_COMPANY_DOMAIN_NAMESPACE = "company_domain"


class CacheError(Exception):
    """Raised when a cache operation fails."""


class CacheProvider(ABC):
    """Abstraction over a namespaced key-value cache with optional TTL expiry.

    `namespace` scopes keys logically (e.g. "resume_profile", "jd_embedding",
    "company_domain", "contact") so different cache consumers cannot collide
    on the same key.
    """

    @abstractmethod
    def get(self, namespace: str, key: str) -> dict | None:
        """Return the cached JSON-serializable value for `key`, or None if
        absent or expired.
        """
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        namespace: str,
        key: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store a JSON-serializable value under `key`, overwriting any prior
        value.

        Args:
            ttl_seconds: if provided, the entry expires and is treated as
                absent after this many seconds. If None, the entry never
                expires.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, namespace: str, key: str) -> bool:
        """Return True if a non-expired value is cached under `key`."""
        raise NotImplementedError

    @abstractmethod
    def invalidate(self, namespace: str, key: str) -> bool:
        """Remove a cached value under `key`, regardless of expiry.

        Returns:
            True if an entry was found and removed, False if none existed.
        """
        raise NotImplementedError


class CacheEntry(Base):
    """Persisted cache entry, namespaced and keyed for reuse across runs.

    `expires_at` is nullable: a null value means the entry never expires.
    """

    __tablename__ = "cache_entries"
    __table_args__ = (
        UniqueConstraint("namespace", "cache_key", name="uq_cache_namespace_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    namespace: Mapped[str] = mapped_column(String(128), nullable=False)
    cache_key: Mapped[str] = mapped_column(String(256), nullable=False)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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
        entry = self._get_entry(namespace, key)
        if entry is None:
            return None

        if self._is_expired(entry):
            self._delete_entry(entry)
            return None

        try:
            return json.loads(entry.value_json)
        except json.JSONDecodeError as exc:
            raise CacheError(
                f"Corrupt cache entry for namespace='{namespace}', key='{key}'."
            ) from exc

    def set(
        self,
        namespace: str,
        key: str,
        value: dict,
        ttl_seconds: int | None = None,
    ) -> None:
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise CacheError(
                f"Value for namespace='{namespace}', key='{key}' is not JSON-serializable."
            ) from exc

        if ttl_seconds is not None and ttl_seconds <= 0:
            raise CacheError(
                f"ttl_seconds must be positive if provided, got {ttl_seconds}."
            )

        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
            if ttl_seconds is not None
            else None
        )

        entry = self._get_entry(namespace, key)

        if entry is None:
            entry = CacheEntry(
                namespace=namespace,
                cache_key=key,
                value_json=serialized,
                expires_at=expires_at,
            )
            self._session.add(entry)
        else:
            entry.value_json = serialized
            entry.expires_at = expires_at

        self._session.flush()

    def exists(self, namespace: str, key: str) -> bool:
        entry = self._get_entry(namespace, key)
        if entry is None:
            return False

        if self._is_expired(entry):
            self._delete_entry(entry)
            return False

        return True

    def invalidate(self, namespace: str, key: str) -> bool:
        entry = self._get_entry(namespace, key)
        if entry is None:
            return False

        self._delete_entry(entry)
        return True

    def _get_entry(self, namespace: str, key: str) -> CacheEntry | None:
        return self._session.execute(
            select(CacheEntry).where(
                CacheEntry.namespace == namespace, CacheEntry.cache_key == key
            )
        ).scalar_one_or_none()

    def _delete_entry(self, entry: CacheEntry) -> None:
        self._session.delete(entry)
        self._session.flush()

    @staticmethod
    def _is_expired(entry: CacheEntry) -> bool:
        if entry.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= entry.expires_at

    # --- Company → Domain convenience API (PRD §6a.3) ---
    #
    # Thin, semantically-named wrappers over get/set/invalidate scoped to the
    # "company_domain" namespace, so callers in app/agents/contact_finder/*
    # do not need to know the namespace string used for domain caching.

    def get_company_domain(self, company_name: str) -> str | None:
        """Return the cached resolved domain for `company_name`, or None if
        absent or expired.
        """
        cached = self.get(_COMPANY_DOMAIN_NAMESPACE, company_name)
        if cached is None:
            return None
        return cached.get("domain")

    def set_company_domain(
        self, company_name: str, domain: str, ttl_seconds: int | None = None
    ) -> None:
        """Cache a resolved domain for `company_name`.

        Args:
            ttl_seconds: optional expiry; company domains rarely change, so
                callers may omit this to cache indefinitely, or provide a
                long TTL (e.g. 30+ days) to allow periodic re-resolution.
        """
        self.set(
            _COMPANY_DOMAIN_NAMESPACE,
            company_name,
            {"domain": domain},
            ttl_seconds=ttl_seconds,
        )

    def invalidate_company_domain(self, company_name: str) -> bool:
        """Remove a cached domain resolution for `company_name`, forcing
        re-resolution on next lookup.
        """
        return self.invalidate(_COMPANY_DOMAIN_NAMESPACE, company_name)