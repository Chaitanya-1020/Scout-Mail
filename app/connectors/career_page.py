"""
Company career-page connector (HTML fallback).

Implements: PRD §5 (Job Scout Agent — scrapes company career pages for
postings), §6a.2 (Job Discovery Source Policy — company career pages are an
allowed, modular, swappable connector), §6.1 (Company Name Accuracy —
resolves company name from the canonical career page rather than an
aggregator), §12 (Risks — scraping fragility: career-page HTML structure
changes break connectors; mitigated here by isolating parsing behind
per-company, configurable selectors rather than hardcoded page structure).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 3 - Additional
Sources, Task 1.

Implements `ConnectorBase` (app/connectors/base.py) as a generic HTML
scraper driven by a per-company `CareerPageConfig` (base URL + CSS-like
selector hints), since career pages have no common schema unlike ATS JSON
APIs. Uses only the standard library `html.parser` to avoid adding a new
third-party dependency not already declared in requirements.txt/pyproject.toml
(per docs/coding_guidelines.md — no dependency added without updating those
files first).

Note: robust, general-purpose HTML scraping typically benefits from a
dedicated parsing library (e.g. BeautifulSoup); introducing one would be a
dependency change and, per docs/architecture.md §5 (Deviation Process),
should be proposed and added to requirements.txt/pyproject.toml as its own
task before this connector depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

from app.connectors.base import ConnectorBase, ConnectorFetchError, RawJobPosting

_REQUEST_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class CareerPageConfig:
    """Per-company configuration for scraping a career page.

    Career pages have no shared schema (unlike Greenhouse/Lever/Ashby JSON
    APIs), so each company requires its own listing-link and title-container
    markers. This config is the only thing that varies between companies —
    the parsing/fetch logic itself does not change (Open/Closed, per
    docs/architecture.md).
    """

    company_name: str
    careers_url: str
    job_link_path_marker: str
    """A substring that identifies an <a href> as a job posting link on this
    page (e.g. '/jobs/', '/careers/positions/'). Links not containing this
    marker are ignored."""


class _JobLinkExtractor(HTMLParser):
    """Extracts anchor tags and their link text from a career listing page."""

    def __init__(self) -> None:
        super().__init__()
        self._current_href: str | None = None
        self._current_text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []  # (href, link_text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = next((value for name, value in attrs if name == "href" and value), None)
        if href:
            self._current_href = href
            self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href is not None:
            text = "".join(self._current_text_parts).strip()
            self.links.append((self._current_href, text))
            self._current_href = None
            self._current_text_parts = []


class _TextExtractor(HTMLParser):
    """Extracts visible text content from a job detail page, used as the
    verbatim JD snapshot (PRD §6.2)."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return "\n".join(self._parts)


class CareerPageConnector(ConnectorBase):
    """Fetches job postings by scraping a company's career page.

    Used only for companies without a supported ATS JSON API (Greenhouse,
    Lever, Ashby). Because career-page HTML structure is company-specific
    and prone to change (PRD §12 — scraping fragility), this connector is
    intentionally the least preferred discovery source and should be
    registered only when no ATS connector is available for a given company.
    """

    def __init__(self, config: CareerPageConfig) -> None:
        self._config = config

    @property
    def source_name(self) -> str:
        return "career_page"

    def fetch_postings(self) -> list[RawJobPosting]:
        listing_html = self._fetch_html(self._config.careers_url)

        link_extractor = _JobLinkExtractor()
        link_extractor.feed(listing_html)

        job_links = [
            (urljoin(self._config.careers_url, href), text)
            for href, text in link_extractor.links
            if self._config.job_link_path_marker in href
        ]

        postings: list[RawJobPosting] = []
        seen_urls: set[str] = set()

        for job_url, link_text in job_links:
            if job_url in seen_urls:
                continue
            seen_urls.add(job_url)

            posting = self._fetch_job_detail(job_url, fallback_title=link_text)
            if posting is not None:
                postings.append(posting)

        return postings

    def _fetch_job_detail(self, job_url: str, fallback_title: str) -> RawJobPosting | None:
        try:
            detail_html = self._fetch_html(job_url)
        except ConnectorFetchError:
            # Skip a single unreachable job detail page rather than failing
            # the whole career-page fetch (PRD §12 — fragility mitigation).
            return None

        text_extractor = _TextExtractor()
        text_extractor.feed(detail_html)
        snapshot_text = text_extractor.get_text()

        title = fallback_title.strip() if fallback_title else None
        if not title or not snapshot_text:
            return None

        return RawJobPosting(
            source_connector=self.source_name,
            external_id=job_url,
            company_name=self._config.company_name,
            role_title=title,
            jd_url=job_url,
            jd_snapshot_text=snapshot_text,
            posted_at=None,
            company_domain=None,
        )

    def _fetch_html(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ConnectorFetchError(
                f"Failed to fetch career page content from '{url}'."
            ) from exc
        return response.text