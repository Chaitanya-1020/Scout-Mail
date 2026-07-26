"""
Greenhouse job board connector.

Implements: PRD §5 (Job Scout Agent — queries Greenhouse API for postings),
§6a.2 (Job Discovery Source Policy — Greenhouse API is an allowed, modular,
swappable connector), §6.1 (Company Name Accuracy — resolved from the ATS
itself rather than an aggregator site), §12 (Risks — prefer stable ATS JSON
APIs over HTML scraping to reduce fragility).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 2 - ATS Connectors,
Task 1.

Implements `ConnectorBase` (app/connectors/base.py) against Greenhouse's
public job board JSON API. Depends only on the `ConnectorBase` /
`RawJobPosting` abstractions and the `requests` HTTP library — never
special-cased by `app/agents/job_scout/agent.py`, which treats this
connector interchangeably with any other registered connector (Liskov
Substitution, per docs/architecture.md).
"""

from __future__ import annotations

from datetime import datetime

import requests

from app.connectors.base import ConnectorBase, ConnectorFetchError, RawJobPosting

_API_BASE_URL = "https://boards-api.greenhouse.io/v1/boards"
_REQUEST_TIMEOUT_SECONDS = 15


class GreenhouseConnector(ConnectorBase):
    """Fetches job postings from a company's public Greenhouse job board.

    A Greenhouse "board token" identifies the company's board
    (e.g. `boards-api.greenhouse.io/v1/boards/{board_token}/jobs`). One
    connector instance is scoped to a single board token; the Job Scout Agent
    (or its registry) constructs one instance per configured
    `greenhouse_board_tokens` entry (PRD §7, app/config.py).
    """

    def __init__(self, board_token: str) -> None:
        if not board_token or not board_token.strip():
            raise ValueError("Greenhouse board_token must be a non-empty string.")
        self._board_token = board_token.strip()

    @property
    def source_name(self) -> str:
        return "greenhouse"

    def fetch_postings(self) -> list[RawJobPosting]:
        url = f"{_API_BASE_URL}/{self._board_token}/jobs"

        try:
            response = requests.get(
                url,
                params={"content": "true"},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectorFetchError(
                f"Failed to fetch Greenhouse jobs for board '{self._board_token}'."
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise ConnectorFetchError(
                f"Greenhouse board '{self._board_token}' returned a non-JSON response."
            ) from exc

        jobs = payload.get("jobs")
        if jobs is None:
            raise ConnectorFetchError(
                f"Greenhouse board '{self._board_token}' response missing 'jobs' field."
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
        absolute_url = job.get("absolute_url")
        content_html = job.get("content")
        company_name = self._extract_company_name(job)

        if external_id is None or not title or not absolute_url or not content_html:
            # Skip malformed entries rather than failing the whole board fetch,
            # so one bad posting does not block discovery of the rest
            # (PRD §12 — scraping/data fragility mitigation).
            return None

        posted_at = self._parse_updated_at(job.get("updated_at"))

        return RawJobPosting(
            source_connector=self.source_name,
            external_id=str(external_id),
            company_name=company_name,
            role_title=str(title),
            jd_url=str(absolute_url),
            jd_snapshot_text=str(content_html),
            posted_at=posted_at,
            company_domain=None,
        )

    def _extract_company_name(self, job: dict) -> str:
        # Greenhouse's per-job payload does not always include a company
        # name field directly; when present under "company" or via
        # departments/offices metadata is not company-identifying, fall back
        # to the board token as the canonical company identifier from the
        # ATS itself (PRD §6.1 — resolve from canonical ATS source).
        company = job.get("company")
        if isinstance(company, dict) and company.get("name"):
            return str(company["name"])
        return self._board_token

    @staticmethod
    def _parse_updated_at(value: object) -> datetime | None:
        if not value or not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None