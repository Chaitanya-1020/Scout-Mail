"""
Public source page discovery.

Implements: PRD §6.3 (Contact Finder Agent — discovers publicly available
company pages such as About, Team, Leadership, Careers, and Contact as
candidate sources for hiring-contact information), §6a.2 (Job Discovery
Source Policy principle extended to contact discovery: only public,
non-login-walled pages are used).
Roadmap: Epic 5 - Contact Finder Agent, Story 2 - Public Source Discovery,
Task 1.

Discovers candidate public page URLs on a company's own domain (About, Team,
Leadership, Careers, Contact) by fetching the domain's homepage and following
same-domain links whose path or link text matches known page-type markers.
This module is responsible for URL discovery only — it does not extract
people, names, or titles from the discovered pages; that is a separate,
later responsibility (Single Responsibility, per docs/coding_guidelines.md).
Uses only the standard library `html.parser`, consistent with
app/connectors/career_page.py, to avoid adding a new third-party dependency
without first updating requirements.txt/pyproject.toml (per
docs/architecture.md §5, Deviation Process).
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import requests

_REQUEST_TIMEOUT_SECONDS = 15

# Page type -> substrings that indicate a link's href or visible text refers
# to that page type. Order matters only for readability; matching is
# independent per page type.
_PAGE_TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "about": ("about",),
    "team": ("team", "our-team", "our people", "people"),
    "leadership": ("leadership", "executives", "management", "founders"),
    "careers": ("careers", "jobs", "join-us", "join us", "work-with-us"),
    "contact": ("contact",),
}


class PublicSourceDiscoveryError(Exception):
    """Raised when a company's public pages cannot be discovered."""


@dataclass(frozen=True)
class DiscoveredPage:
    """Metadata for a candidate public page discovered on a company's domain."""

    url: str
    page_type: str
    """One of: 'about', 'team', 'leadership', 'careers', 'contact'."""
    link_text: str
    """Visible anchor text for the link that led to this page, for auditability."""


class _LinkExtractor(HTMLParser):
    """Extracts anchor tags (href + visible text) from an HTML page."""

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


class PublicSourcePageDiscoverer:
    """Discovers candidate public pages (About/Team/Leadership/Careers/
    Contact) on a company's own domain, starting from its homepage.

    Only same-domain links are considered, so discovery never follows
    off-site links (e.g. social media, aggregators) — consistent with the
    public, non-login-walled source policy applied elsewhere in the system
    (PRD §6a.2).
    """

    def discover(self, domain: str) -> list[DiscoveredPage]:
        """Fetch the domain's homepage and identify candidate public pages.

        Args:
            domain: bare domain (e.g. "acme.com"), as resolved by
                `app.agents.contact_finder.domain_resolver`.

        Returns:
            One `DiscoveredPage` per matched page type (at most one URL per
            type — the first matching link found), possibly empty if no
            matching links were found.

        Raises:
            PublicSourceDiscoveryError: if the homepage cannot be fetched.
        """
        if not domain or not domain.strip():
            raise PublicSourceDiscoveryError("Cannot discover pages for an empty domain.")

        homepage_url = self._fetch_homepage_url(domain)
        homepage_html = self._fetch_html(homepage_url)

        link_extractor = _LinkExtractor()
        link_extractor.feed(homepage_html)

        discovered: dict[str, DiscoveredPage] = {}

        for href, link_text in link_extractor.links:
            absolute_url = urljoin(homepage_url, href)

            if not self._is_same_domain(absolute_url, domain):
                continue

            page_type = self._classify_link(absolute_url, link_text)
            if page_type is None:
                continue

            # Keep only the first match per page type to avoid duplicate
            # candidates for the same page (e.g. both a nav link and a
            # footer link to "About").
            if page_type not in discovered:
                discovered[page_type] = DiscoveredPage(
                    url=absolute_url,
                    page_type=page_type,
                    link_text=link_text,
                )

        return list(discovered.values())

    def _classify_link(self, url: str, link_text: str) -> str | None:
        path = urlparse(url).path.lower()
        text = link_text.lower()

        for page_type, markers in _PAGE_TYPE_MARKERS.items():
            for marker in markers:
                if marker in path or marker in text:
                    return page_type

        return None

    def _is_same_domain(self, url: str, domain: str) -> bool:
        hostname = urlparse(url).hostname
        if not hostname:
            return False
        hostname = hostname.lower().removeprefix("www.")
        return hostname == domain.lower().removeprefix("www.")

    def _fetch_homepage_url(self, domain: str) -> str:
        for scheme in ("https", "http"):
            candidate_url = f"{scheme}://{domain}"
            try:
                response = requests.head(
                    candidate_url,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                    allow_redirects=True,
                )
                if response.status_code < 500:
                    return response.url
            except requests.RequestException:
                continue

        raise PublicSourceDiscoveryError(
            f"Could not reach homepage for domain '{domain}' over https or http."
        )

    def _fetch_html(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise PublicSourceDiscoveryError(
                f"Failed to fetch page content from '{url}'."
            ) from exc
        return response.text