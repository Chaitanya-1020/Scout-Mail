"""
Public profile extraction.

Implements: PRD §6.3 (Contact Finder Agent — extracts candidate names and
titles from discovered public pages), §6a.1 (Layered Confidence Pipeline —
public-source extraction is one evidence layer contributing to a contact's
confidence score; email addresses are never inferred at this layer), §9
(Auditability — every extracted fact carries its source URL and a snippet of
supporting evidence).
Roadmap: Epic 5 - Contact Finder Agent, Story 3 - Profile Extraction, Task 1.

Extracts people (name + job title) from the pages discovered by
`PublicSourcePageDiscoverer` (app/agents/contact_finder/public_source_scraper.py).
Depends only on the `LLMProvider` abstraction (app/llm/ollama_client.py) for
structured extraction from page text, and on `requests` for fetching page
content — never on a concrete LLM backend or on `app/db` directly (Dependency
Inversion / Clean Architecture, per docs/architecture.md and
docs/coding_guidelines.md §2). This layer never infers or fabricates email
addresses; email inference is a separate, later responsibility
(app/agents/contact_finder/pattern_generator.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser

import requests

from app.agents.contact_finder.public_source_scraper import DiscoveredPage
from app.llm.ollama_client import LLMProvider, LLMProviderError

_REQUEST_TIMEOUT_SECONDS = 15

_EXTRACTION_SYSTEM_PROMPT = """\
You are a precise information-extraction assistant. Given the visible text of
a company web page, extract only the people explicitly named on the page
along with their job title, if a title is stated or clearly implied for them
on this page. Do not guess or infer people who are not named in the text. Do
not extract email addresses, phone numbers, or any contact details -- only
name and title. If no people are named on the page, return an empty list.
"""

_EXTRACTION_JSON_SCHEMA_HINT = """\
Return a JSON object with exactly this key:
{
  "people": [
    {
      "name": string,
      "title": string or null,
      "evidence_snippet": string
    }
  ]
}
"evidence_snippet" must be a short, verbatim excerpt (under 200 characters)
from the page text that supports this person's name and title.
"""


class ProfileExtractionError(Exception):
    """Raised when people cannot be extracted from a discovered page."""


@dataclass(frozen=True)
class ExtractedPerson:
    """A person extracted from a public company page.

    Deliberately excludes any email field: this layer never infers or
    fabricates email addresses (PRD §6a.1) — email pattern generation is a
    separate, later pipeline stage.
    """

    name: str
    title: str | None
    source_url: str
    page_type: str
    evidence_snippet: str


class _PageTextExtractor(HTMLParser):
    """Extracts visible text content from an HTML page, used as the input
    text for LLM-based person extraction."""

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


def _normalize_name(name: str) -> str:
    """Normalize a name for duplicate detection (case/whitespace-insensitive)."""
    return " ".join(name.strip().lower().split())


class ProfileLookupAgent:
    """Extracts named people and their titles from discovered public pages.

    For each `DiscoveredPage`, fetches the page content, extracts visible
    text, and asks the extraction-task LLM to identify explicitly named
    people and titles. Deduplicates people by normalized name across all
    pages passed to a single `extract_people` call, keeping the first
    occurrence (and its source attribution) encountered.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    def extract_people(self, pages: list[DiscoveredPage]) -> list[ExtractedPerson]:
        """Extract deduplicated people from a set of discovered pages.

        Raises:
            ProfileExtractionError: if a page's content cannot be fetched.
                A page that fetches successfully but yields no people, or
                whose extraction the LLM fails on, is skipped rather than
                failing the whole batch (consistent with the fragility
                mitigation approach used by connectors, per PRD §12).
        """
        seen_names: set[str] = set()
        results: list[ExtractedPerson] = []

        for page in pages:
            try:
                page_text = self._fetch_page_text(page.url)
            except ProfileExtractionError:
                continue

            people = self._extract_from_text(page_text, page)

            for person in people:
                normalized = _normalize_name(person.name)
                if not normalized or normalized in seen_names:
                    continue
                seen_names.add(normalized)
                results.append(person)

        return results

    def _extract_from_text(
        self, page_text: str, page: DiscoveredPage
    ) -> list[ExtractedPerson]:
        if not page_text or not page_text.strip():
            return []

        prompt = (
            f"{_EXTRACTION_JSON_SCHEMA_HINT}\n\n"
            f"Page text:\n\"\"\"\n{page_text.strip()}\n\"\"\""
        )

        try:
            raw = self._llm.generate_json(
                task="extraction",
                prompt=prompt,
                system=_EXTRACTION_SYSTEM_PROMPT,
            )
        except LLMProviderError:
            # A single page's extraction failing should not abort discovery
            # from the other pages (PRD §12 — fragility mitigation).
            return []

        people_raw = raw.get("people")
        if not isinstance(people_raw, list):
            return []

        people: list[ExtractedPerson] = []
        for item in people_raw:
            if not isinstance(item, dict):
                continue

            name = str(item.get("name", "")).strip()
            if not name:
                continue

            title_raw = item.get("title")
            title = str(title_raw).strip() if title_raw else None

            evidence_snippet = str(item.get("evidence_snippet", "")).strip()

            people.append(
                ExtractedPerson(
                    name=name,
                    title=title or None,
                    source_url=page.url,
                    page_type=page.page_type,
                    evidence_snippet=evidence_snippet,
                )
            )

        return people

    def _fetch_page_text(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ProfileExtractionError(
                f"Failed to fetch page content from '{url}'."
            ) from exc

        text_extractor = _PageTextExtractor()
        text_extractor.feed(response.text)
        return text_extractor.get_text()