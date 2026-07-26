"""
Lever job board connector.

Implements: PRD §5 (Job Scout Agent — queries Lever API for postings),
§6a.2 (Job Discovery Source Policy — Lever API is an allowed, modular,
swappable connector), §6.1 (Company Name Accuracy — resolved from the ATS
itself rather than an aggregator site), §12 (Risks — prefer stable ATS JSON
APIs over HTML scraping to reduce fragility).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 2 - ATS Connectors,
Task 2.

Implements `ConnectorBase` (app/connectors/base.py) against Lever's public
postings JSON API. Depends only on the `ConnectorBase` / `RawJobPosting`
abstractions and the `requests` HTTP library — never special-cased by
`app/agents/job_scout/agent.py`, which treats this connector interchangeably
with any other registered connector (Liskov Substitution, per
docs/architecture.md).
"""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from app.connectors.base import ConnectorBase, ConnectorFetchError, RawJobPosting

_API_BASE_URL = "https://api.lever.co/v0/postings"
_REQUEST_TIMEOUT_SECONDS = 15


class LeverConnector(ConnectorBase):
    """Fetches job postings from a company's public Lever postings API.

    A Lever "company slug" identifies the company's postings feed
    (e.g. `api.lever.co/v0/postings/{company_slug}`). One connector instance
    is scoped to a single company slug; the Job Scout Agent (or its
    registry) constructs one instance per configured
    `lever_company_slugs` entry (PRD §7, app/config.py).
    """

    def __init__(self, company_slug: str) -> None:
        if not company_slug or not company_slug.strip():
            raise ValueError("Lever company_slug must be a non-empty string.")
        self._company_slug = company_slug.strip()

    @property
    def source_name(self) -> str:
        return "lever"

    def fetch_postings(self) -> list[RawJobPosting]:
        url = f"{_API_BASE_URL}/{self._company_slug}"

        try:
            response = requests.get(
                url,
                params={"mode": "json"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectorFetchError(
                f"Failed to fetch Lever postings for company '{self._company_slug}'."
            ) from exc

        try:
            postings_payload = response.json()
        except ValueError as exc:
            raise ConnectorFetchError(
                f"Lever company '{self._company_slug}' returned a non-JSON response."
            ) from exc

        if not isinstance(postings_payload, list):
            raise ConnectorFetchError(
                f"Lever company '{self._company_slug}' response was not a list of postings."
            )

        postings: list[RawJobPosting] = []
        for posting in postings_payload:
            raw_posting = self._to_raw_posting(posting)
            if raw_posting is not None:
                postings.append(raw_posting)

        return postings

    def _to_raw_posting(self, posting: dict) -> RawJobPosting | None:
        external_id = posting.get("id")
        title = posting.get("text")
        hosted_url = posting.get("hostedUrl")
        description_html = posting.get("descriptionPlain") or posting.get("description")

        if not external_id or not title or not hosted_url or not description_html:
            # Skip malformed entries rather than failing the whole feed fetch,
            # so one bad posting does not block discovery of the rest
            # (PRD §12 — scraping/data fragility mitigation).
            return None

        posted_at = self._parse_created_at(posting.get("createdAt"))

        return RawJobPosting(
            source_connector=self.source_name,
            external_id=str(external_id),
            company_name=self._extract_company_name(posting),
            role_title=str(title),
            jd_url=str(hosted_url),
            jd_snapshot_text=str(description_html),
            posted_at=posted_at,
            company_domain=None,
        )

    def _extract_company_name(self, posting: dict) -> str:
        # Lever's per-posting payload does not include an explicit company
        # name field; the canonical company identifier from this ATS is its
        # own company slug (PRD §6.1 — resolve from canonical ATS source
        # rather than an aggregator).
        categories = posting.get("categories")
        if isinstance(categories, dict) and categories.get("team"):
            # "team" is a department, not a company name — never substituted
            # for company_name; kept unused intentionally to avoid mislabeling.
            pass
        return self._company_slug

    @staticmethod
    def _parse_created_at(value: object) -> datetime | None:
        if value is None:
            return None
        try:
            # Lever returns createdAt as epoch milliseconds.
            return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            return None