"""
Hunter.io fallback client.

Implements: PRD §6.3 (Contact Finder Agent — Hunter.io free tier used as a
fallback contact-verification source), §6a.1 (Layered Confidence Pipeline —
Hunter.io is queried only when cheaper/free layers (SMTP handshake) fail or
leave confidence at LOW, since it consumes a scarce monthly quota), §9
(Auditability — every fact stored with source and confidence; Hunter.io
results carry explicit provenance).
Roadmap: Epic 5 - Contact Finder Agent, Story 6 - Hunter.io Integration,
Task 1.

Wraps the Hunter.io Email Finder API with quota tracking so the configured
free-tier monthly limit (`hunter_io_monthly_quota`, app/config.py, default
25) is never exceeded. Depends only on the `CacheProvider` abstraction
(app/services/cache.py) for quota-usage tracking, not a concrete cache
backend (Dependency Inversion, per docs/architecture.md). Enforces the
"fallback only" policy explicitly: callers must state why they are invoking
Hunter.io (SMTP validation failed, or confidence remains LOW), and a request
missing a valid trigger reason is rejected before any quota is consumed or
any HTTP call is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import requests

from app.services.cache import CacheProvider

_API_BASE_URL = "https://api.hunter.io/v2/email-finder"
_REQUEST_TIMEOUT_SECONDS = 15
_QUOTA_NAMESPACE = "hunter_io_quota"


class HunterFallbackTrigger(str, Enum):
    """Justification for invoking the Hunter.io fallback layer.

    Per PRD §6a.1, Hunter.io is queried only after cheaper/free evidence
    layers have run and either failed or left confidence at LOW — never as
    a first-choice lookup.
    """

    SMTP_VALIDATION_FAILED = "smtp_validation_failed"
    CONFIDENCE_REMAINS_LOW = "confidence_remains_low"


class HunterClientError(Exception):
    """Raised when a Hunter.io request fails or is not permitted."""


class HunterQuotaExceededError(HunterClientError):
    """Raised when the configured monthly Hunter.io quota has been exhausted."""


@dataclass(frozen=True)
class HunterEvidence:
    """Structured, provenance-carrying result of a Hunter.io lookup.

    `found` distinguishes "queried Hunter.io and it had no match" from a
    request that could not be made at all (which raises instead) — both are
    meaningful for confidence scoring and auditability (PRD §9).
    """

    email: str | None
    found: bool
    hunter_confidence_score: int | None
    """Hunter.io's own 0-100 confidence score for the match, if found."""
    source_description: str
    trigger_reason: HunterFallbackTrigger
    queried_at: datetime


class HunterQuotaTracker:
    """Tracks Hunter.io monthly quota usage via the `CacheProvider` abstraction.

    Usage is keyed by calendar month (e.g. "2026-07") so the counter resets
    naturally each month without a separate scheduled reset job.
    """

    def __init__(self, cache: CacheProvider, monthly_quota: int) -> None:
        self._cache = cache
        self._monthly_quota = monthly_quota

    def _current_period_key(self) -> str:
        now = datetime.now(timezone.utc)
        return f"{now.year:04d}-{now.month:02d}"

    def used_this_period(self) -> int:
        period_key = self._current_period_key()
        cached = self._cache.get(_QUOTA_NAMESPACE, period_key)
        if cached is None:
            return 0
        return int(cached.get("count", 0))

    def remaining_this_period(self) -> int:
        return max(0, self._monthly_quota - self.used_this_period())

    def has_quota_remaining(self) -> bool:
        return self.remaining_this_period() > 0

    def record_usage(self) -> None:
        period_key = self._current_period_key()
        used = self.used_this_period()
        self._cache.set(_QUOTA_NAMESPACE, period_key, {"count": used + 1})


class HunterClient:
    """Optional fallback contact-verification client backed by Hunter.io's
    free-tier Email Finder API.

    Every call must declare a `HunterFallbackTrigger` justifying the lookup
    (PRD §6a.1 — fallback only, after cheaper evidence layers). Quota is
    checked and recorded via `HunterQuotaTracker` so the configured monthly
    limit is never exceeded, even across process restarts (quota state is
    persisted, not held in memory).
    """

    def __init__(
        self,
        api_key: str | None,
        quota_tracker: HunterQuotaTracker,
    ) -> None:
        self._api_key = api_key
        self._quota_tracker = quota_tracker

    def find_email(
        self,
        first_name: str,
        last_name: str,
        domain: str,
        trigger_reason: HunterFallbackTrigger,
    ) -> HunterEvidence:
        """Query Hunter.io's Email Finder for a person at a company domain.

        Args:
            first_name: person's first name.
            last_name: person's last name.
            domain: bare company domain (e.g. "acme.com").
            trigger_reason: why this fallback lookup is being made. Required
                to make the fallback-only policy explicit and enforceable at
                the call site, per PRD §6a.1.

        Returns:
            Structured evidence of the lookup outcome, with full provenance.

        Raises:
            HunterClientError: if no API key is configured, required inputs
                are missing, or the API call fails.
            HunterQuotaExceededError: if the configured monthly quota has
                already been exhausted.
        """
        if not self._api_key:
            raise HunterClientError(
                "Hunter.io API key is not configured; cannot query the fallback layer."
            )

        if not first_name.strip() or not last_name.strip() or not domain.strip():
            raise HunterClientError(
                "first_name, last_name, and domain are all required to query Hunter.io."
            )

        if not self._quota_tracker.has_quota_remaining():
            raise HunterQuotaExceededError(
                "Hunter.io monthly free-tier quota has been exhausted; "
                "skipping fallback lookup."
            )

        try:
            response = requests.get(
                _API_BASE_URL,
                params={
                    "domain": domain.strip(),
                    "first_name": first_name.strip(),
                    "last_name": last_name.strip(),
                    "api_key": self._api_key,
                },
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise HunterClientError(
                f"Hunter.io request failed for '{first_name} {last_name}' at '{domain}'."
            ) from exc

        self._quota_tracker.record_usage()

        try:
            payload = response.json()
        except ValueError as exc:
            raise HunterClientError("Hunter.io returned a non-JSON response.") from exc

        return self._to_evidence(payload, trigger_reason)

    def _to_evidence(self, payload: dict, trigger_reason: HunterFallbackTrigger) -> HunterEvidence:
        data = payload.get("data") or {}
        email = data.get("email")
        hunter_score = data.get("score")

        queried_at = datetime.now(timezone.utc)

        if not email:
            return HunterEvidence(
                email=None,
                found=False,
                hunter_confidence_score=None,
                source_description="Hunter.io Email Finder API returned no match.",
                trigger_reason=trigger_reason,
                queried_at=queried_at,
            )

        return HunterEvidence(
            email=str(email),
            found=True,
            hunter_confidence_score=int(hunter_score) if hunter_score is not None else None,
            source_description="Hunter.io Email Finder API match.",
            trigger_reason=trigger_reason,
            queried_at=queried_at,
        )