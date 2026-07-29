"""
Contact Finder repository.

Implements: PRD §6.3 (Contact Finder Agent — persists resolved contacts),
§6a.1 (Layered Confidence Pipeline — every confidence score is stored
alongside its contributing evidence sources, never as a bare value), §9
(Auditability — every stored fact carries source and confidence, with a
timestamp).
Roadmap: Epic 5 - Contact Finder Agent, Story 7 - Confidence Scoring,
Task 2.

Persists a resolved contact (name, title, candidate email), its confidence
level, and the full evidence trail that produced that level, into the
`Contact` and `ConfidenceEvidence` ORM models (app/db/models.py). Translates
between the domain-layer `app.utils.confidence.ConfidenceLevel` (used by the
pure scoring engine) and the persistence-layer
`app.db.models.ConfidenceLevel` (used by the ORM), since the two are
intentionally kept as separate enums across the domain/infrastructure
boundary (see app/utils/confidence.py module docstring).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Contact
from app.db.models import ConfidenceLevel as DbConfidenceLevel
from app.db.models import ConfidenceEvidence as DbConfidenceEvidence
from app.utils.confidence import ConfidenceLevel, ConfidenceScoreResult

_DOMAIN_TO_DB_CONFIDENCE: dict[ConfidenceLevel, DbConfidenceLevel] = {
    ConfidenceLevel.HIGH: DbConfidenceLevel.HIGH,
    ConfidenceLevel.MEDIUM: DbConfidenceLevel.MEDIUM,
    ConfidenceLevel.LOW: DbConfidenceLevel.LOW,
}


class ContactFinderRepositoryError(Exception):
    """Raised when a contact record cannot be persisted."""


@dataclass(frozen=True)
class EvidenceSourceRecord:
    """One evidence-layer result to persist alongside a contact, mirroring
    the shape of `app.db.models.ConfidenceEvidence` without depending on the
    ORM class at the call site.
    """

    layer_name: str
    """e.g. 'domain_resolution', 'public_source', 'title_relevance', 'smtp',
    'hunter_io'."""
    agreed: bool
    source_description: str
    source_url: str | None = None


@dataclass(frozen=True)
class ContactResolutionRecord:
    """A fully-resolved contact, ready to persist: identity, candidate
    email, confidence result, and its supporting evidence."""

    job_posting_id: uuid.UUID
    name: str | None
    title: str | None
    email: str | None
    confidence_result: ConfidenceScoreResult
    evidence_sources: list[EvidenceSourceRecord]


class ContactFinderRepository:
    """Persistence for resolved contacts and their confidence evidence trail.

    Every contact is stored with its `confidence_level` (never a "verified"
    boolean, per PRD §13.2) and every evidence-layer result that contributed
    to that score, so a human reviewer can inspect exactly why a given
    confidence level was assigned (PRD §6a.1, §9).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_contact(self, record: ContactResolutionRecord) -> Contact:
        """Persist a resolved contact along with its evidence sources.

        The `Contact.created_at` timestamp is set automatically by the ORM
        model's server default (app/db/models.py) — this method does not
        set it explicitly, preserving a single source of truth for
        creation-time semantics.

        Raises:
            ContactFinderRepositoryError: if the confidence level cannot be
                translated to a persistable value.
        """
        db_confidence_level = self._to_db_confidence_level(record.confidence_result.level)

        contact = Contact(
            job_posting_id=record.job_posting_id,
            name=record.name,
            title=record.title,
            email=record.email,
            confidence_level=db_confidence_level,
        )
        self._session.add(contact)
        self._session.flush()

        for evidence_source in record.evidence_sources:
            evidence_row = DbConfidenceEvidence(
                contact_id=contact.id,
                layer_name=evidence_source.layer_name,
                agreed=evidence_source.agreed,
                source_description=evidence_source.source_description,
                source_url=evidence_source.source_url,
            )
            self._session.add(evidence_row)

        self._session.flush()
        return contact

    def get_contact(self, contact_id: uuid.UUID) -> Contact | None:
        """Fetch a persisted contact by id, including its evidence trail via
        the ORM relationship (`Contact.evidence`)."""
        return self._session.get(Contact, contact_id)

    def get_contacts_for_job_posting(self, job_posting_id: uuid.UUID) -> list[Contact]:
        """Fetch all resolved contacts for a given job posting."""
        return list(
            self._session.query(Contact)
            .filter(Contact.job_posting_id == job_posting_id)
            .all()
        )

    def _to_db_confidence_level(self, level: ConfidenceLevel) -> DbConfidenceLevel:
        try:
            return _DOMAIN_TO_DB_CONFIDENCE[level]
        except KeyError as exc:
            raise ContactFinderRepositoryError(
                f"Unrecognized confidence level '{level}'; cannot persist contact."
            ) from exc