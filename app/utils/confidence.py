"""
Contact confidence scoring engine.

Implements: PRD §6.3 (Contact Finder Agent — layered confidence pipeline),
§6a.1 (Layered Confidence Pipeline — confidence is derived from agreement
across independent evidence layers: domain resolution, public-source
agreement, title relevance, SMTP handshake, and Hunter.io fallback; the
system never claims a contact is "verified", only High/Medium/Low
confidence, always shown with its evidence), §13.2 (Non-Goals Are Enforced
Constraints — no boolean "verified" flag exists anywhere in this codebase).
Roadmap: Epic 5 - Contact Finder Agent, Story 7 - Confidence Scoring,
Task 1.

Pure domain-layer scoring logic: takes a structured summary of evidence
already gathered by other Contact Finder layers (domain resolution, public
source scraper, profile lookup, SMTP validator, Hunter.io client) and
produces a `ConfidenceLevel`. This module has no I/O and no dependency on
`app/db`, `app/llm`, `app/connectors`, or any concrete evidence-gathering
module (Dependency Inversion / Single Responsibility, per
docs/architecture.md and docs/coding_guidelines.md §2) — callers (e.g.
`app/agents/contact_finder/agent.py`) translate each layer's own result type
into the generic `ConfidenceEvidence` input defined here.

Note on `ConfidenceLevel`: `app/db/models.py` independently defines an
identically-valued `ConfidenceLevel` enum for persistence purposes, since a
domain-layer module (this file) cannot be imported by infrastructure code
under the dependency rule as currently structured, nor can `app/db/models.py`
be imported here without inverting that rule. Both enums are intentionally
kept to exactly `HIGH`, `MEDIUM`, `LOW` — any future divergence between them
must be treated as a bug.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class ConfidenceLevel(str, enum.Enum):
    """The only valid confidence outputs for a contact. There is intentionally
    no "verified" or other boolean-equivalent value (PRD §13.2)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceScoringError(Exception):
    """Raised when confidence cannot be scored from the given evidence."""


@dataclass(frozen=True)
class DomainEvidence:
    """Evidence from company domain resolution
    (app/agents/contact_finder/domain_resolver.py)."""

    resolved: bool
    confirmed_reachable: bool
    source: str
    """e.g. 'known_source_url', 'name_derived_http_redirect', 'name_derived_dns'."""


@dataclass(frozen=True)
class PublicSourceEvidence:
    """Evidence from public page discovery + profile extraction
    (app/agents/contact_finder/public_source_scraper.py,
    app/agents/contact_finder/profile_lookup.py)."""

    independent_page_agreement_count: int
    """Number of independent public pages (About/Team/Leadership/Careers)
    naming this exact person with a consistent title."""
    name_found: bool
    title_found: bool


@dataclass(frozen=True)
class TitleRelevanceEvidence:
    """Evidence of whether the contact's title is plausibly relevant to
    receiving hiring outreach (e.g. recruiter, hiring manager, HR, talent
    acquisition), as opposed to an unrelated role."""

    is_relevant: bool | None
    """None means relevance could not be determined."""


@dataclass(frozen=True)
class SmtpEvidence:
    """Evidence from SMTP mailbox validation
    (app/agents/contact_finder/smtp_validator.py). Deliberately expressed as
    a generic tri-state rather than importing `SmtpValidationOutcome`
    directly, keeping this module decoupled from that concrete
    implementation (per docs/architecture.md)."""

    mailbox_confirmed_exists: bool | None
    """True = mailbox confirmed to exist, False = mailbox confirmed not to
    exist, None = inconclusive (no MX, connection failed, greylisted,
    catch-all, etc.)."""


@dataclass(frozen=True)
class HunterEvidenceSummary:
    """Evidence from the Hunter.io fallback layer
    (app/agents/contact_finder/hunter_client.py)."""

    queried: bool
    found: bool
    hunter_confidence_score: int | None
    """Hunter.io's own 0-100 score, if `found` is True."""


@dataclass(frozen=True)
class ConfidenceEvidence:
    """Aggregate evidence bundle passed into the scoring engine for a single
    candidate contact."""

    domain: DomainEvidence
    public_source: PublicSourceEvidence
    title_relevance: TitleRelevanceEvidence
    smtp: SmtpEvidence
    hunter: HunterEvidenceSummary


@dataclass(frozen=True)
class ConfidenceScoreResult:
    """Outcome of scoring a contact's evidence.

    `points` and `max_points` are retained for auditability (PRD §9) even
    though only `level` is ever surfaced as the contact's stored confidence
    value — the raw score is never itself exposed as a pseudo-"verified"
    signal.
    """

    level: ConfidenceLevel
    points: float
    max_points: float
    contributing_factors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConfidenceScoringConfig:
    """Configurable weights and thresholds for the scoring engine.

    Kept as plain data (not hardcoded constants inline in the scoring
    method) so weighting can be tuned or A/B-tested without modifying
    `ConfidenceScorer` itself (Open/Closed, per docs/architecture.md).
    """

    # --- Domain resolution weights ---
    weight_domain_resolved: float = 1.0
    weight_domain_confirmed_reachable: float = 1.0
    weight_domain_from_known_source: float = 1.0

    # --- Public source agreement weights ---
    weight_name_found: float = 1.5
    weight_title_found: float = 1.0
    weight_per_additional_page_agreement: float = 1.0
    max_page_agreement_bonus_count: int = 3
    """Caps how many additional agreeing pages continue to add points, so a
    single page type flooding results does not dominate the score."""

    # --- Title relevance weight ---
    weight_title_relevant: float = 1.0
    penalty_title_irrelevant: float = -1.5

    # --- SMTP validation weights ---
    weight_smtp_confirmed_exists: float = 2.0
    penalty_smtp_confirmed_not_exists: float = -3.0

    # --- Hunter.io weights ---
    weight_hunter_found: float = 1.5
    weight_hunter_high_score_bonus: float = 1.0
    hunter_high_score_threshold: int = 80

    # --- Level thresholds, expressed as a fraction of max attainable points ---
    high_threshold_fraction: float = 0.70
    medium_threshold_fraction: float = 0.40


class ConfidenceScorer:
    """Scores a contact's aggregate evidence into a `ConfidenceLevel`.

    The engine never outputs anything other than HIGH, MEDIUM, or LOW —
    there is no code path that produces a "verified" or otherwise
    binary-certain result, regardless of how strong the evidence is
    (PRD §13.2). A single strongly negative signal (SMTP explicitly
    confirming the mailbox does not exist, or a clearly irrelevant title)
    can floor the result at LOW even if other layers agree, since a
    confirmed-wrong contact should never be surfaced as trustworthy.
    """

    def __init__(self, config: ConfidenceScoringConfig | None = None) -> None:
        self._config = config or ConfidenceScoringConfig()

    def score(self, evidence: ConfidenceEvidence) -> ConfidenceScoreResult:
        """Compute a confidence level from the given evidence bundle.

        Raises:
            ConfidenceScoringError: if `evidence` is missing required
                sub-evidence (should not occur given the dataclass's typing,
                but guarded against malformed programmatic construction).
        """
        if evidence is None:
            raise ConfidenceScoringError("Cannot score confidence from null evidence.")

        cfg = self._config
        points = 0.0
        max_points = 0.0
        factors: list[str] = []

        # --- Domain resolution ---
        max_points += cfg.weight_domain_resolved
        if evidence.domain.resolved:
            points += cfg.weight_domain_resolved
            factors.append("company_domain_resolved")

        max_points += cfg.weight_domain_confirmed_reachable
        if evidence.domain.confirmed_reachable:
            points += cfg.weight_domain_confirmed_reachable
            factors.append("company_domain_confirmed_reachable")

        max_points += cfg.weight_domain_from_known_source
        if evidence.domain.source == "known_source_url":
            points += cfg.weight_domain_from_known_source
            factors.append("domain_from_known_source_url")

        # --- Public source agreement ---
        max_points += cfg.weight_name_found
        if evidence.public_source.name_found:
            points += cfg.weight_name_found
            factors.append("name_found_on_public_page")

        max_points += cfg.weight_title_found
        if evidence.public_source.title_found:
            points += cfg.weight_title_found
            factors.append("title_found_on_public_page")

        bonus_page_count = min(
            max(evidence.public_source.independent_page_agreement_count - 1, 0),
            cfg.max_page_agreement_bonus_count,
        )
        max_points += cfg.weight_per_additional_page_agreement * cfg.max_page_agreement_bonus_count
        if bonus_page_count > 0:
            page_bonus = cfg.weight_per_additional_page_agreement * bonus_page_count
            points += page_bonus
            factors.append(
                f"{evidence.public_source.independent_page_agreement_count}_independent_pages_agree"
            )

        # --- Title relevance ---
        if evidence.title_relevance.is_relevant is True:
            max_points += cfg.weight_title_relevant
            points += cfg.weight_title_relevant
            factors.append("title_relevant_to_hiring")
        elif evidence.title_relevance.is_relevant is False:
            points += cfg.penalty_title_irrelevant
            factors.append("title_not_relevant_to_hiring")
        # is_relevant is None (undetermined): contributes to neither points
        # nor max_points, since the engine has no basis to reward or
        # penalize an unknown relevance.

        # --- SMTP validation ---
        if evidence.smtp.mailbox_confirmed_exists is True:
            max_points += cfg.weight_smtp_confirmed_exists
            points += cfg.weight_smtp_confirmed_exists
            factors.append("smtp_mailbox_confirmed_exists")
        elif evidence.smtp.mailbox_confirmed_exists is False:
            points += cfg.penalty_smtp_confirmed_not_exists
            factors.append("smtp_mailbox_confirmed_not_exists")
        # None (inconclusive): no contribution either way.

        # --- Hunter.io fallback ---
        if evidence.hunter.queried and evidence.hunter.found:
            max_points += cfg.weight_hunter_found
            points += cfg.weight_hunter_found
            factors.append("hunter_io_match_found")

            if (
                evidence.hunter.hunter_confidence_score is not None
                and evidence.hunter.hunter_confidence_score >= cfg.hunter_high_score_threshold
            ):
                max_points += cfg.weight_hunter_high_score_bonus
                points += cfg.weight_hunter_high_score_bonus
                factors.append("hunter_io_high_confidence_score")

        if max_points <= 0:
            raise ConfidenceScoringError(
                "No scoreable evidence was provided; cannot compute a confidence level."
            )

        # Clamp negative point totals (e.g. a strong negative SMTP/title
        # signal) to zero before computing the fraction, so the result
        # floors cleanly at LOW rather than going out of range.
        clamped_points = max(points, 0.0)
        fraction = clamped_points / max_points

        level = self._level_for_fraction(fraction)

        # A confirmed-nonexistent mailbox always floors the result at LOW,
        # regardless of how strong other evidence is — a confirmed-wrong
        # contact must never be surfaced as MEDIUM or HIGH confidence.
        if evidence.smtp.mailbox_confirmed_exists is False:
            level = ConfidenceLevel.LOW

        return ConfidenceScoreResult(
            level=level,
            points=points,
            max_points=max_points,
            contributing_factors=factors,
        )

    def _level_for_fraction(self, fraction: float) -> ConfidenceLevel:
        cfg = self._config
        if fraction >= cfg.high_threshold_fraction:
            return ConfidenceLevel.HIGH
        if fraction >= cfg.medium_threshold_fraction:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW