"""
Job Scout Agent.

Implements: PRD §5 (Job Scout Agent — scrapes/queries job boards for postings
matching resume profile), §8.3 (Job Scout Agent pulls new postings on a
schedule), §12 (Risks — scraping fragility: one connector's failure must not
block discovery from the others).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 5 - Job Scout Agent
Orchestration, Task 1.

Runs every connector registered in a `ConnectorRegistry`
(app/connectors/registry.py) and aggregates the results into a single list of
`RawJobPosting`. Depends only on `ConnectorRegistry` / `ConnectorBase`
abstractions (app/connectors/base.py, app/connectors/registry.py), never on a
concrete connector directly (Dependency Inversion, per
docs/architecture.md). Persistence of fetched postings is handled separately
by `app/agents/job_scout/repository.py` (Epic 3, Story 5, Task 3) — this
module's responsibility is fetching and aggregating only (Single
Responsibility, per docs/coding_guidelines.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.connectors.base import ConnectorFetchError, RawJobPosting
from app.connectors.registry import ConnectorRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectorFetchFailure:
    """Records a single connector's failure during a Job Scout run, so the
    run as a whole can continue and the failure can be surfaced/audited
    rather than silently swallowed."""

    source_name: str
    error_message: str


@dataclass(frozen=True)
class JobScoutRunResult:
    """Outcome of a single Job Scout Agent run across all registered connectors."""

    postings: list[RawJobPosting]
    failures: list[ConnectorFetchFailure]

    @property
    def succeeded_connector_count(self) -> int:
        return len({p.source_connector for p in self.postings}) if self.postings else 0

    @property
    def has_failures(self) -> bool:
        return len(self.failures) > 0


class JobScoutAgent:
    """Fetches job postings from every registered, policy-approved connector.

    A single failing connector (e.g. a career page whose HTML structure
    changed, per PRD §12) does not abort the run — its failure is recorded
    in `JobScoutRunResult.failures` and fetching continues with the
    remaining connectors, maximizing discovery coverage per run.
    """

    def __init__(self, registry: ConnectorRegistry) -> None:
        self._registry = registry

    def run(self) -> JobScoutRunResult:
        """Fetch postings from all registered connectors.

        Returns:
            A `JobScoutRunResult` containing all successfully fetched
            postings across connectors, plus a record of any connector
            failures encountered during this run.
        """
        all_postings: list[RawJobPosting] = []
        failures: list[ConnectorFetchFailure] = []

        for connector in self._registry.all():
            try:
                postings = connector.fetch_postings()
            except ConnectorFetchError as exc:
                logger.warning(
                    "Job Scout connector '%s' failed to fetch postings: %s",
                    connector.source_name,
                    exc,
                )
                failures.append(
                    ConnectorFetchFailure(
                        source_name=connector.source_name,
                        error_message=str(exc),
                    )
                )
                continue

            all_postings.extend(postings)

        return JobScoutRunResult(postings=all_postings, failures=failures)