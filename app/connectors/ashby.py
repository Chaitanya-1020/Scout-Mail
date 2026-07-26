"""
Ashby job board connector.

Implements: PRD §5 (Job Scout Agent — queries Ashby public job boards for
postings), §6a.2 (Job Discovery Source Policy — Ashby public job boards are
an allowed, modular, swappable connector), §6.1 (Company Name Accuracy —
resolved from the ATS itself rather than an aggregator site), §12 (Risks —
prefer stable ATS JSON APIs over HTML scraping to reduce fragility).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 2 - ATS Connectors,
Task 3.

Implements `ConnectorBase` (app/connectors/base.py) against Ashby's public
job board posting API. Depends only on the `ConnectorBase` / `RawJobPosting`
abstractions and the `requests` HTTP library — never special-cased by
`app/agents/job_scout/agent.py`, which treats this connector interchangeably
with any other registered connector (Liskov Substitution, per
docs/architecture.md).
"""

from __future__ import annotations

from datetime import datetime

import requests

from app.connectors.base import ConnectorBase, ConnectorFetchError, RawJobPosting

_API_BASE_URL = "https://api.ashbyhq.com/posting-api/job-board"
_REQUEST_TIMEOUT_SECONDS = 15


class AshbyConnector(ConnectorBase):
    """Fetches job postings from a company's public Ashby job board.

    An Ashby "org slug" identifies the company's public job board
    (e.g. `api.ashbyhq.com/posting-api/job-board/{org_slug}`). One connector
    instance is scoped to a single org slug; the Job Scout Agent (or its
    registry) constructs one instance per configured `ashby_org_slugs` entry
    (PRD §7, app/config.py).
    """

    def __init__(self, org_slug: str) -> None:
        if not org_slug or not org_slug.strip():
            raise ValueError("Ashby org_slug must be a non-empty string.")
        self._org_slug = org_slug.strip()

    @property
    def source_name(self) -> str:
        return "ashby"

    def fetch_postings(self) -> list[RawJobPosting]:
        url = f"{_API_BASE_URL}/{self._org_slug}"

        try:
            response = requests.get(
                url,
                params={"includeCompensation": "false"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectorFetchError(
                f"Failed to fetch Ashby postings for org '{self._org_slug}'."
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ConnectorFetchError(
                f"Ashby org '{self._org_slug}' returned a non-JSON response."
            ) from exc

        jobs = payload.get("jobs")
        if jobs is None:
            raise ConnectorFetchError(
                f"Ashby org '{self._org_slug}' response missing 'jobs' field."
            )

        postings: list[RawJobPosting] = []
        for job in jobs:
            posting = self._to_raw_posting(job)
            if posting is not None:
                postings.append(posting)

        return postings

    def _to_raw_posting(self, job: dict) -> RawJobPosting | None:
        external_id = job.get("id")
        title = job.get("title")
        job_url = job.get("jobUrl") or job.get("applyUrl")
        description_html = job.get("descriptionHtml") or job.get("descriptionPlain")

        if not external_id or not title or not job_url or not description_html:
            # Skip malformed entries rather than failing the whole board fetch,
            # so one bad posting does not block discovery of the rest
            # (PRD §12 — scraping/data fragility mitigation).
            return None

        posted_at = self._parse_published_at(job.get("publishedDate"))

        return RawJobPosting(
            source_connector=self.source_name,
            external_id=str(external_id),
            company_name=self._extract_company_name(job),
            role_title=str(title),
            jd_url=str(job_url),
            jd_snapshot_text=str(description_html),
            posted_at=posted_at,
            company_domain=None,
        )

    def _extract_company_name(self, job: dict) -> str:
        # Ashby's per-job payload does not consistently include a distinct
        # company name field on the job itself; the canonical company
        # identifier from this ATS is its own org slug (PRD §6.1 — resolve
        # from canonical ATS source rather than an aggregator).
        organization_name = job.get("organizationName")
        if organization_name:
            return str(organization_name)
        return self._org_slug

    @staticmethod
    def _parse_published_at(value: object) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None