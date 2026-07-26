"""
Job Scout scheduling.

Implements: PRD §8.3 (Job Scout Agent pulls new postings on a schedule, e.g.
every 6h), §7 (Tech Stack — Python APScheduler or cron).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 5 - Job Scout Agent
Orchestration, Task 2.

Wires `JobScoutAgent` (app/agents/job_scout/agent.py) into a recurring
APScheduler job, triggered every `job_scout_poll_interval_hours` (default 6,
per app/config.py). This module is the composition root for scheduled Job
Scout runs: it constructs the concrete `ConnectorRegistry` and `JobScoutAgent`
here, per docs/architecture.md (concrete wiring belongs in `main.py` /
scheduler/runner entrypoints, not inside the agent itself).

Note: persisting fetched postings (`app/agents/job_scout/repository.py`,
Epic 3, Story 5, Task 3) does not exist yet — the scheduled job currently
logs the run outcome only. Once that repository exists, `_run_job_scout`
should be updated to persist `result.postings` via it, since only this file
was requested here.
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.agents.job_scout.agent import JobScoutAgent, JobScoutRunResult
from app.config import get_settings
from app.connectors.registry import build_default_registry

logger = logging.getLogger(__name__)

_JOB_ID = "job_scout_periodic_run"


def _run_job_scout() -> None:
    """Execute a single Job Scout Agent run and log its outcome.

    Constructed fresh on each invocation so a long-lived scheduler process
    always picks up the latest connector configuration
    (`app.config.get_settings()` is cached per-process but reflects the
    settings loaded at process start).
    """
    registry = build_default_registry()
    agent = JobScoutAgent(registry=registry)

    result: JobScoutRunResult = agent.run()

    logger.info(
        "Job Scout run complete: %d posting(s) fetched from %d connector(s), "
        "%d connector failure(s).",
        len(result.postings),
        result.succeeded_connector_count,
        len(result.failures),
    )

    for failure in result.failures:
        logger.warning(
            "Job Scout connector '%s' failed: %s",
            failure.source_name,
            failure.error_message,
        )


def build_scheduler() -> BackgroundScheduler:
    """Construct a `BackgroundScheduler` with the recurring Job Scout job
    registered, using the configured poll interval (PRD §8.3, default 6h).

    Does not start the scheduler — callers (e.g. `app/main.py`) are
    responsible for calling `.start()` at application startup and
    `.shutdown()` at application teardown.
    """
    settings = get_settings()
    scheduler = BackgroundScheduler()

    scheduler.add_job(
        _run_job_scout,
        trigger=IntervalTrigger(hours=settings.job_scout_poll_interval_hours),
        id=_JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    return scheduler