"""
RSS feed job connector.

Implements: PRD §5 (Job Scout Agent — queries RSS feeds for postings),
§6a.2 (Job Discovery Source Policy — RSS feeds are an allowed, modular,
swappable connector), §6.1 (Company Name Accuracy — resolved from
per-feed configuration rather than an aggregator guess), §12 (Risks —
prefer stable structured feeds over HTML scraping to reduce fragility).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 3 - Additional
Sources, Task 2.

Implements `ConnectorBase` (app/connectors/base.py) against a configured RSS
feed URL using `feedparser` (already declared in requirements.txt /
pyproject.toml). Depends only on the `ConnectorBase` / `RawJobPosting`
abstractions — never special-cased by `app/agents/job_scout/agent.py`, which
treats this connector interchangeably with any other registered connector
(Liskov Substitution, per docs/architecture.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import struct_time

import feedparser

from app.connectors.base import ConnectorBase, ConnectorFetchError, RawJobPosting


class RssConnector(ConnectorBase):
    """Fetches job postings from a single RSS/Atom feed.

    Because a generic RSS feed does not reliably identify the company
    publishing a given entry, each connector instance is scoped to one feed
    URL with an explicit `company_name` supplied by configuration (PRD §6.1
    — company name must be resolved from a canonical source, not guessed
    from feed content). The Job Scout Agent (or its registry) constructs one
    instance per configured `rss_feed_urls` entry (PRD §7, app/config.py).
    """

    def __init__(self, feed_url: str, company_name: str) -> None:
        if not feed_url or not feed_url.strip():
            raise ValueError("RSS feed_url must be a non-empty string.")
        if not company_name or not company_name.strip():
            raise ValueError("RSS company_name must be a non-empty string.")
        self._feed_url = feed_url.strip()
        self._company_name = company_name.strip()

    @property
    def source_name(self) -> str:
        return "rss"

    def fetch_postings(self) -> list[RawJobPosting]:
        parsed = feedparser.parse(self._feed_url)

        if parsed.get("bozo"):
            bozo_exception = parsed.get("bozo_exception")
            raise ConnectorFetchError(
                f"Failed to parse RSS feed '{self._feed_url}': {bozo_exception}"
            )

        entries = parsed.get("entries")
        if entries is None:
            raise ConnectorFetchError(
                f"RSS feed '{self._feed_url}' returned no 'entries' field."
            )

        postings: list[RawJobPosting] = []
        for entry in entries:
            posting = self._to_raw_posting(entry)
            if posting is not None:
                postings.append(posting)

        return postings

    def _to_raw_posting(self, entry: dict) -> RawJobPosting | None:
        external_id = entry.get("id") or entry.get("link")
        title = entry.get("title")
        link = entry.get("link")
        snapshot_text = (
            entry.get("summary")
            or entry.get("description")
            or (entry.get("content", [{}])[0].get("value") if entry.get("content") else None)
        )

        if not external_id or not title or not link or not snapshot_text:
            # Skip malformed entries rather than failing the whole feed fetch,
            # so one bad posting does not block discovery of the rest
            # (PRD §12 — scraping/data fragility mitigation).
            return None

        posted_at = self._parse_published(entry.get("published_parsed"))

        return RawJobPosting(
            source_connector=self.source_name,
            external_id=str(external_id),
            company_name=self._company_name,
            role_title=str(title),
            jd_url=str(link),
            jd_snapshot_text=str(snapshot_text),
            posted_at=posted_at,
            company_domain=None,
        )

    @staticmethod
    def _parse_published(value: struct_time | None) -> datetime | None:
        if value is None:
            return None
        try:
            return datetime.fromtimestamp(
                __import__("calendar").timegm(value), tz=timezone.utc
            )
        except (TypeError, ValueError, OverflowError):
            return None