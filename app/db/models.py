"""
ORM models — core persisted schema.

Implements: PRD §6/§6a (Accuracy & Confidence — evidence-based contact records),
§8 (Functional Requirements 1-9), §9 (Auditability — every fact stored with source
and confidence), §13.2 (Non-Goals enforced as constraints: no "verified" boolean,
only High/Medium/Low confidence; human-approval gate on outreach emails).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 2 - Database Setup, Task 2.

Single source of truth for persisted schema (per docs/project_structure.md — no
agent defines its own table). All facts (company, role, contact, email) are
stored alongside their source reference and confidence level; never as bare
values.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False
    )


class ConfidenceLevel(str, enum.Enum):
    """Confidence enum for contact records.

    Per PRD §6a.1 / §13.2: this is the ONLY valid output type for a contact's
    reliability. No boolean "verified" field exists anywhere in this schema.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OutreachStatus(str, enum.Enum):
    """Lifecycle status of a drafted outreach email."""

    DRAFT = "draft"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    SENT = "sent"


class TrackingEventType(str, enum.Enum):
    """Post-send outreach tracking events (PRD §8.9)."""

    SENT = "sent"
    REPLIED = "replied"
    NO_RESPONSE = "no_response"


class Resume(Base):
    """A parsed candidate resume (PRD §8.1)."""

    __tablename__ = "resumes"

    id: Mapped[uuid.UUID] = _uuid_pk()
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    parsed_profile: Mapped[dict] = mapped_column(
        "parsed_profile", nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    matches: Mapped[list["ResumeJobMatch"]] = relationship(
        back_populates="resume", cascade="all, delete-orphan"
    )


class JobPosting(Base):
    """A discovered job posting (PRD §6.2, §8.3).

    `role_title` and `jd_snapshot_text` are stored verbatim from the source —
    never paraphrased — so mismatches are visible during human review
    (PRD §6.2).
    """

    __tablename__ = "job_postings"
    __table_args__ = (UniqueConstraint("jd_hash", name="uq_job_postings_jd_hash"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_connector: Mapped[str] = mapped_column(String(64), nullable=False)
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)

    company_name: Mapped[str] = mapped_column(String(512), nullable=False)
    company_domain: Mapped[str | None] = mapped_column(String(256), nullable=True)

    role_title: Mapped[str] = mapped_column(
        Text, nullable=False, doc="Verbatim role title string from source JD."
    )
    jd_url: Mapped[str] = mapped_column(Text, nullable=False)
    jd_snapshot_text: Mapped[str] = mapped_column(
        Text, nullable=False, doc="Verbatim JD text snapshot, source-of-truth for review."
    )
    jd_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    matches: Mapped[list["ResumeJobMatch"]] = relationship(
        back_populates="job_posting", cascade="all, delete-orphan"
    )
    contacts: Mapped[list["Contact"]] = relationship(
        back_populates="job_posting", cascade="all, delete-orphan"
    )
    outreach_emails: Mapped[list["OutreachEmail"]] = relationship(
        back_populates="job_posting", cascade="all, delete-orphan"
    )


class ResumeJobMatch(Base):
    """Resume-to-posting similarity score (PRD §8.4, Resume Match Agent)."""

    __tablename__ = "resume_job_matches"
    __table_args__ = (
        UniqueConstraint("resume_id", "job_posting_id", name="uq_resume_job_match"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    job_posting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    resume: Mapped["Resume"] = relationship(back_populates="matches")
    job_posting: Mapped["JobPosting"] = relationship(back_populates="matches")


class Contact(Base):
    """A candidate HR/hiring-manager contact for a job posting.

    Per PRD §6a.1 / §13.2: `confidence_level` is the ONLY reliability signal
    on this record. There is intentionally no boolean "verified" column.
    """

    __tablename__ = "contacts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_posting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )

    name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)

    confidence_level: Mapped[ConfidenceLevel] = mapped_column(
        Enum(ConfidenceLevel, name="confidence_level"), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job_posting: Mapped["JobPosting"] = relationship(back_populates="contacts")
    evidence: Mapped[list["ConfidenceEvidence"]] = relationship(
        back_populates="contact", cascade="all, delete-orphan"
    )
    outreach_emails: Mapped[list["OutreachEmail"]] = relationship(
        back_populates="contact"
    )


class ConfidenceEvidence(Base):
    """One evidence-layer result contributing to a contact's confidence score.

    Per PRD §6a.1: every confidence score must be shown with its evidence
    sources, not just the number, so the human reviewer can judge for
    themselves.
    """

    __tablename__ = "confidence_evidence"

    id: Mapped[uuid.UUID] = _uuid_pk()
    contact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("contacts.id", ondelete="CASCADE"), nullable=False
    )

    layer_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        doc="e.g. 'official_source', 'public_profile', 'domain_match', "
        "'pattern_generation', 'smtp_handshake', 'hunter_io'.",
    )
    agreed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, doc="Whether this layer's result agrees with the surfaced contact."
    )
    source_description: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    contact: Mapped["Contact"] = relationship(back_populates="evidence")


class OutreachEmail(Base):
    """A drafted outreach email awaiting human approval (PRD §6.4, §8.6-8.8).

    `approved_by_human` is the single gate checked by
    `services/email_sender.py` before any SMTP call — enforced in code per
    PRD §13.2, not merely by UI convention.
    """

    __tablename__ = "outreach_emails"

    id: Mapped[uuid.UUID] = _uuid_pk()
    job_posting_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False
    )
    contact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("contacts.id", ondelete="RESTRICT"), nullable=False
    )

    subject: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)

    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validation_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[OutreachStatus] = mapped_column(
        Enum(OutreachStatus, name="outreach_status"),
        nullable=False,
        default=OutreachStatus.DRAFT,
    )
    approved_by_human: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    job_posting: Mapped["JobPosting"] = relationship(back_populates="outreach_emails")
    contact: Mapped["Contact"] = relationship(back_populates="outreach_emails")
    tracking_events: Mapped[list["EmailTrackingEvent"]] = relationship(
        back_populates="outreach_email", cascade="all, delete-orphan"
    )


class EmailTrackingEvent(Base):
    """Post-send follow-up tracking for an outreach email (PRD §8.9)."""

    __tablename__ = "email_tracking_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    outreach_email_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("outreach_emails.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[TrackingEventType] = mapped_column(
        Enum(TrackingEventType, name="tracking_event_type"), nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    outreach_email: Mapped["OutreachEmail"] = relationship(back_populates="tracking_events")