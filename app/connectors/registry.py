"""
Job discovery connector registry.

Implements: PRD §6a.2 (Job Discovery Source Policy — only Greenhouse, Lever,
Ashby, company career pages, and RSS feeds are allowed connectors, each as a
modular, swappable connector; LinkedIn scraping is prohibited outright),
§12 (Risks — legal/ToS: job discovery is restricted by policy to
Greenhouse/Lever/Ashby/career-page/RSS only, no LinkedIn or other
login-walled scraping, ever), §13.2 (Non-Goals Are Enforced Constraints — no
LinkedIn connector exists in code).
Roadmap: Epic 3 - Job Discovery (Job Scout Agent), Story 4 - Connector
Registry, Task 1.

Builds and exposes the whitelist of connector instances used by
`app/agents/job_scout/agent.py` (Epic 3, Story 5), constructed from
application settings (app/config.py). This is the single place connector
instances are assembled — no other module constructs a connector directly,
so adding a new source means adding one connector class plus one branch here,
never modifying `job_scout/agent.py` internals (Open/Closed, per
docs/architecture.md). LinkedIn is not, and must never be, a registered
source: there is no LinkedIn connector class in this codebase to register.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.connectors.ashby import AshbyConnector
from app.connectors.base import ConnectorBase
from app.connectors.greenhouse import GreenhouseConnector
from app.connectors.lever import LeverConnector

# Explicit, closed allowlist of connector source names. Any connector whose
# `source_name` is not in this set is rejected by `register()` — this is the
# enforcement point for PRD §6a.2 / §13.2 (no LinkedIn connector, ever).
_ALLOWED_SOURCE_NAMES: frozenset[str] = frozenset(
    {"greenhouse", "lever", "ashby", "career_page", "rss"}
)

# Source names explicitly and permanently banned from registration, named
# here for clarity even though no implementation for them exists in this
# codebase. Attempting to register a connector under one of these names is
# always rejected, regardless of implementation.
_PROHIBITED_SOURCE_NAMES: frozenset[str] = frozenset({"linkedin"})


class ConnectorRegistryError(Exception):
    """Raised when a connector cannot be registered or is not permitted."""


class ConnectorRegistry:
    """Holds the whitelisted set of connector instances for a job discovery run.

    `app/agents/job_scout/agent.py` depends only on `ConnectorRegistry.all()`
    to obtain connectors, never on a concrete connector class (Dependency
    Inversion, per docs/architecture.md).
    """

    def __init__(self) -> None:
        self._connectors: list[ConnectorBase] = []

    def register(self, connector: ConnectorBase) -> None:
        """Add a connector to the registry, enforcing the source policy.

        Raises:
            ConnectorRegistryError: if the connector's `source_name` is
                prohibited or not on the approved allowlist.
        """
        source_name = connector.source_name.strip().lower()

        if source_name in _PROHIBITED_SOURCE_NAMES:
            raise ConnectorRegistryError(
                f"Connector source '{source_name}' is permanently prohibited "
                "(PRD §6a.2 — LinkedIn scraping is prohibited outright) and "
                "cannot be registered."
            )

        if source_name not in _ALLOWED_SOURCE_NAMES:
            raise ConnectorRegistryError(
                f"Connector source '{source_name}' is not on the approved "
                f"allowlist {sorted(_ALLOWED_SOURCE_NAMES)} (PRD §6a.2)."
            )

        self._connectors.append(connector)

    def all(self) -> list[ConnectorBase]:
        """Return all registered connectors."""
        return list(self._connectors)


def build_default_registry(settings: Settings | None = None) -> ConnectorRegistry:
    """Build a `ConnectorRegistry` populated from application settings.

    Constructs one Greenhouse connector per configured board token, one
    Lever connector per configured company slug, and one Ashby connector per
    configured org slug (PRD §7, app/config.py). Career-page and RSS
    connectors require additional per-company configuration
    (`CareerPageConfig`, feed company names) beyond simple string lists, so
    they are intentionally not auto-built here; callers register those
    manually via `ConnectorRegistry.register()` using
    `app/connectors/career_page.py` / `app/connectors/rss.py` directly.

    No LinkedIn branch exists, and none may be added (PRD §6a.2, §13.2).
    """
    settings = settings or get_settings()
    registry = ConnectorRegistry()

    for board_token in settings.greenhouse_board_tokens:
        registry.register(GreenhouseConnector(board_token=board_token))

    for company_slug in settings.lever_company_slugs:
        registry.register(LeverConnector(company_slug=company_slug))

    for org_slug in settings.ashby_org_slugs:
        registry.register(AshbyConnector(org_slug=org_slug))

    return registry