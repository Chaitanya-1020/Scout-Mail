"""
Job discovery connector interface.

Implements: PRD §5 (Job Scout Agent — scrapes/queries job boards for postings
matching resume profile), §6a.2 (Job Discovery Source Policy — modular,
swappable connectors; LinkedIn scraping prohibited outright), §9 (Non-Functional
Requirements — Extensibility: each agent/source swappable without rewriting
the graph).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 1 - Connector
Interface, Task 1.

Defines the `ConnectorBase` abstraction that every job discovery source must
implement. Per docs/architecture.md, `app/agents/job_scout/agent.py` depends
only on this interface, never on a concrete connector (e.g. Greenhouse,
Lever) directly — concrete connectors are registered in
`app/connectors/registry.py` (Epic 3, Story 4) and injected at the
composition root. This keeps the interface narrow (Interface Segregation):
implementers only need to provide `fetch_postings()`, with no obligation to
handle scheduling, caching, or persistence, which are separate collaborators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


class ConnectorFetchError(Exception):
    """Raised when a connector fails to fetch postings from its source.

    Callers (e.g. `app/agents/job_scout/agent.py`) catch this to isolate a
    single connector's failure from the rest of the job discovery run,
    per PRD §12 (Risks — scraping fragility mitigation).
    """


@dataclass(frozen=True)
class RawJobPosting:
    """A job posting as fetched from a source, prior to persistence.

    `role_title` and `jd_snapshot_text` must be the exact, verbatim strings
    from the source (PRD §6.2 — Role Title Accuracy): no paraphrasing or
    normalization is permitted between the connector and this dataclass.
    """

    source_connector: str
    external_id: str
    company_name: str
    role_title: str
    jd_url: str
    jd_snapshot_text: str
    posted_at: datetime | None = None
    company_domain: str | None = None


class ConnectorBase(ABC):
    """Abstraction over a single job discovery source.

    Every implementation (Greenhouse, Lever, Ashby, career-page scraper, RSS)
    must be substitutable for any other without special-casing by the caller
    (Liskov Substitution, per docs/architecture.md): `fetch_postings()` always
    returns a list of `RawJobPosting`, and always raises `ConnectorFetchError`
    on failure rather than returning `None` or a partially-typed result.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Stable identifier for this connector (e.g. "greenhouse", "lever").

        Used as `RawJobPosting.source_connector` and as the connector's key
        in `app/connectors/registry.py`.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_postings(self) -> list[RawJobPosting]:
        """Fetch current job postings from this source.

        Returns:
            A list of `RawJobPosting`, possibly empty if the source has no
            current postings matching this connector's configuration.

        Raises:
            ConnectorFetchError: if the source cannot be reached or its
                response cannot be parsed.
        """
        raise NotImplementedError