"""
Contact Finder Agent.

Implements: PRD §5 (Contact Finder Agent — layered confidence pipeline for
resolving hiring contacts), §6.3 (full pipeline: domain resolution, public
source discovery, profile extraction, email pattern generation, SMTP
validation, Hunter.io fallback, confidence scoring), §6a.1 (Layered
Confidence Pipeline — cheap/free layers run first; Hunter.io is invoked only
when SMTP validation fails to confirm a mailbox, or confidence remains LOW).
Roadmap: Epic 5 - Contact Finder Agent, Story 8 - Contact Finder Agent
Orchestration, Task 1.

Orchestrates the full Contact Finder pipeline by composing, in order:
`DomainResolver` (Story 1), `PublicSourcePageDiscoverer` (Story 2),
`ProfileLookupAgent` (Story 3), `EmailPatternGenerator` (Story 4),
`SmtpMailboxValidator` (Story 5), `HunterClient` (Story 6),
`ConfidenceScorer` (Story 7), and `ContactFinderRepository` (Story 7). This
module contains no business logic already implemented by those
collaborators — it only sequences calls, translates each layer's result into
the next layer's input, and decides *whether* to invoke the optional
Hunter.io fallback per PRD §6a.1. Every collaborator is injected via the
constructor (Dependency Inversion, per docs/architecture.md); this agent
never constructs a concrete resolver, scraper, validator, or client itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.agents.contact_finder.domain_resolver import (
    DomainResolutionError,
    DomainResolver,
)
from app.agents.contact_finder.hunter_client import (
    HunterClient,
    HunterClientError,
    HunterFallbackTrigger,
    HunterQuotaExceededError,
)
from app.agents.contact_finder.pattern_generator import (
    EmailPatternGenerator,
    PatternGenerationError,
)
from app.agents.contact_finder.profile_lookup import ExtractedPerson, ProfileLookupAgent
from app.agents.contact_finder.public_source_scraper import PublicSourceDiscoveryError, PublicSourcePageDiscoverer
from app.agents.contact_finder.repository import (
    ContactFinderRepository,
    ContactResolutionRecord,
    EvidenceSourceRecord,
)
from app.agents.contact_finder.smtp_validator import (
    SmtpMailboxValidator,
    SmtpValidationOutcome,
)
from app.utils.confidence import (
    ConfidenceEvidence,
    ConfidenceLevel,
    ConfidenceScorer,
    DomainEvidence,
    HunterEvidenceSummary,
    PublicSourceEvidence,
    SmtpEvidence,
    TitleRelevanceEvidence,
)

_MAX_SMTP_CANDIDATES_PER_PERSON = 3

_RELEVANT_TITLE_KEYWORDS = (
    "recruit",
    "talent",
    "people",
    "human resources",
    "hr ",
    "hiring",
    "hr,",
    "hr manager",
    "hr director",
    "people operations",
)


class ContactFinderAgentError(Exception):
    """Raised when the Contact Finder pipeline cannot complete for a job posting."""


@dataclass(frozen=True)
class ContactFinderRunResult:
    """Outcome of running the Contact Finder pipeline for one job posting."""

    job_posting_id: uuid.UUID
    contacts_resolved: int


class ContactFinderAgent:
    """Resolves hiring contacts for a job posting via a layered, evidence-
    based confidence pipeline (PRD §6a.1).

    All collaborators are injected, keeping this agent free of any concrete
    infrastructure dependency (Dependency Inversion, per
    docs/architecture.md). Per docs/architecture.md §3, this agent does not
    call other agents (e.g. Resume Match, Job Scout) directly — cross-agent
    coordination belongs to the graph layer (Epic 7).
    """

    def __init__(
        self,
        domain_resolver: DomainResolver,
        page_discoverer: PublicSourcePageDiscoverer,
        profile_lookup_agent: ProfileLookupAgent,
        pattern_generator: EmailPatternGenerator,
        smtp_validator: SmtpMailboxValidator,
        hunter_client: HunterClient,
        confidence_scorer: ConfidenceScorer,
        repository: ContactFinderRepository,
    ) -> None:
        self._domain_resolver = domain_resolver
        self._page_discoverer = page_discoverer
        self._profile_lookup_agent = profile_lookup_agent
        self._pattern_generator = pattern_generator
        self._smtp_validator = smtp_validator
        self._hunter_client = hunter_client
        self._confidence_scorer = confidence_scorer
        self._repository = repository

    def find_contacts(
        self,
        job_posting_id: uuid.UUID,
        company_name: str,
        jd_url: str,
    ) -> ContactFinderRunResult:
        """Run the full Contact Finder pipeline for a single job posting.

        Raises:
            ContactFinderAgentError: if company domain resolution fails,
                which is a prerequisite for every subsequent layer.
        """
        try:
            domain_result = self._domain_resolver.resolve(
                company_name=company_name, known_source_url=jd_url
            )
        except DomainResolutionError as exc:
            raise ContactFinderAgentError(
                f"Cannot resolve contacts for job posting '{job_posting_id}': "
                "company domain resolution failed."
            ) from exc

        try:
            pages = self._page_discoverer.discover(domain_result.domain)
        except PublicSourceDiscoveryError:
            pages = []

        people = self._profile_lookup_agent.extract_people(pages)

        resolved_count = 0
        for person in people:
            self._resolve_and_persist_person(
                job_posting_id=job_posting_id,
                domain=domain_result.domain,
                domain_evidence=DomainEvidence(
                    resolved=True,
                    confirmed_reachable=domain_result.confirmed_reachable,
                    source=domain_result.source,
                ),
                person=person,
            )
            resolved_count += 1

        return ContactFinderRunResult(
            job_posting_id=job_posting_id, contacts_resolved=resolved_count
        )

    def _resolve_and_persist_person(
        self,
        job_posting_id: uuid.UUID,
        domain: str,
        domain_evidence: DomainEvidence,
        person: ExtractedPerson,
    ) -> None:
        evidence_sources: list[EvidenceSourceRecord] = [
            EvidenceSourceRecord(
                layer_name="domain_resolution",
                agreed=domain_evidence.resolved,
                source_description=f"Domain resolved via {domain_evidence.source}.",
                source_url=None,
            ),
            EvidenceSourceRecord(
                layer_name="public_source",
                agreed=True,
                source_description=(
                    f"Name and title found on {person.page_type} page: "
                    f"{person.evidence_snippet}"
                ),
                source_url=person.source_url,
            ),
        ]

        title_relevant = self._is_title_relevant(person.title)
        evidence_sources.append(
            EvidenceSourceRecord(
                layer_name="title_relevance",
                agreed=bool(title_relevant),
                source_description=f"Title '{person.title}' relevance assessed heuristically.",
                source_url=None,
            )
        )

        best_email, smtp_confirmed = self._resolve_email_via_smtp(
            person=person, domain=domain, evidence_sources=evidence_sources
        )

        preliminary_confidence = self._confidence_scorer.score(
            ConfidenceEvidence(
                domain=domain_evidence,
                public_source=PublicSourceEvidence(
                    independent_page_agreement_count=1,
                    name_found=True,
                    title_found=person.title is not None,
                ),
                title_relevance=TitleRelevanceEvidence(is_relevant=title_relevant),
                smtp=SmtpEvidence(mailbox_confirmed_exists=smtp_confirmed),
                hunter=HunterEvidenceSummary(queried=False, found=False, hunter_confidence_score=None),
            )
        )

        hunter_summary, hunter_email = self._maybe_query_hunter(
            person=person,
            domain=domain,
            smtp_confirmed=smtp_confirmed,
            preliminary_level=preliminary_confidence.level,
            evidence_sources=evidence_sources,
        )

        final_email = best_email or hunter_email

        final_confidence = self._confidence_scorer.score(
            ConfidenceEvidence(
                domain=domain_evidence,
                public_source=PublicSourceEvidence(
                    independent_page_agreement_count=1,
                    name_found=True,
                    title_found=person.title is not None,
                ),
                title_relevance=TitleRelevanceEvidence(is_relevant=title_relevant),
                smtp=SmtpEvidence(mailbox_confirmed_exists=smtp_confirmed),
                hunter=hunter_summary,
            )
        )

        self._repository.save_contact(
            ContactResolutionRecord(
                job_posting_id=job_posting_id,
                name=person.name,
                title=person.title,
                email=final_email,
                confidence_result=final_confidence,
                evidence_sources=evidence_sources,
            )
        )

    def _resolve_email_via_smtp(
        self,
        person: ExtractedPerson,
        domain: str,
        evidence_sources: list[EvidenceSourceRecord],
    ) -> tuple[str | None, bool | None]:
        try:
            candidates = self._pattern_generator.generate(person.name, domain)
        except PatternGenerationError:
            return None, None

        best_email: str | None = None
        smtp_confirmed: bool | None = None

        for candidate in candidates[:_MAX_SMTP_CANDIDATES_PER_PERSON]:
            result = self._smtp_validator.validate(candidate.email)

            evidence_sources.append(
                EvidenceSourceRecord(
                    layer_name="smtp",
                    agreed=result.outcome == SmtpValidationOutcome.MAILBOX_EXISTS,
                    source_description=(
                        f"SMTP handshake for '{candidate.email}' "
                        f"({candidate.pattern_name} pattern): {result.detail}"
                    ),
                    source_url=None,
                )
            )

            if result.outcome == SmtpValidationOutcome.MAILBOX_EXISTS:
                return candidate.email, True

            if result.outcome == SmtpValidationOutcome.MAILBOX_NOT_FOUND:
                smtp_confirmed = False

        if best_email is None and candidates:
            best_email = candidates[0].email

        return best_email, smtp_confirmed

    def _maybe_query_hunter(
        self,
        person: ExtractedPerson,
        domain: str,
        smtp_confirmed: bool | None,
        preliminary_level: ConfidenceLevel,
        evidence_sources: list[EvidenceSourceRecord],
    ) -> tuple[HunterEvidenceSummary, str | None]:
        trigger: HunterFallbackTrigger | None = None
        if smtp_confirmed is not True:
            trigger = HunterFallbackTrigger.SMTP_VALIDATION_FAILED
        elif preliminary_level == ConfidenceLevel.LOW:
            trigger = HunterFallbackTrigger.CONFIDENCE_REMAINS_LOW

        if trigger is None:
            return (
                HunterEvidenceSummary(queried=False, found=False, hunter_confidence_score=None),
                None,
            )

        first_name, _, last_name = person.name.partition(" ")
        last_name = last_name.strip() or first_name

        try:
            hunter_evidence = self._hunter_client.find_email(
                first_name=first_name,
                last_name=last_name,
                domain=domain,
                trigger_reason=trigger,
            )
        except (HunterClientError, HunterQuotaExceededError) as exc:
            evidence_sources.append(
                EvidenceSourceRecord(
                    layer_name="hunter_io",
                    agreed=False,
                    source_description=f"Hunter.io fallback not available: {exc}",
                    source_url=None,
                )
            )
            return (
                HunterEvidenceSummary(queried=False, found=False, hunter_confidence_score=None),
                None,
            )

        evidence_sources.append(
            EvidenceSourceRecord(
                layer_name="hunter_io",
                agreed=hunter_evidence.found,
                source_description=hunter_evidence.source_description,
                source_url=None,
            )
        )

        return (
            HunterEvidenceSummary(
                queried=True,
                found=hunter_evidence.found,
                hunter_confidence_score=hunter_evidence.hunter_confidence_score,
            ),
            hunter_evidence.email,
        )

    def _is_title_relevant(self, title: str | None) -> bool | None:
        if not title:
            return None
        lowered = title.lower()
        return any(keyword in lowered for keyword in _RELEVANT_TITLE_KEYWORDS)