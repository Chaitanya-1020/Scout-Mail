"""
Shared LangGraph state schema.

Implements: PRD §5 (Multi-Agent Orchestration — Job Scout, Resume Match,
Contact Finder, and Outreach Composer + Validator agents share a single
pipeline state as a posting moves through the workflow), §9 (Auditability —
every stage's output, and every failure, is retained on the state for the
full run rather than discarded between nodes).
Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 1 -
Shared State Schema, Task 1.

Defines the single state object threaded through every node in
`app/graph/state_graph.py` (a later task). Modeled as Pydantic `BaseModel`
subclasses (rather than a `TypedDict`) so the state is self-validating and
trivially JSON-serializable for the Postgres checkpointer
(`app/graph/checkpointer.py`, Epic 7, Story 2) via `model_dump_json` /
`model_validate_json` — a `TypedDict` would require hand-written
serialization/validation logic to get the same guarantees. Per
docs/architecture.md §3, this module is the only place aware of every
agent's output shape simultaneously; individual agents (`app/agents/*`)
remain unaware of `GraphState` and continue to accept/return their own
plain dataclasses — translation between the two happens in
`app/graph/state_graph.py`, not here.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class PipelineStage(str, Enum):
    """Named stages of the resume-to-outreach pipeline, used to record where
    a run currently is and where an error occurred (PRD §9 — auditability).
    """

    RESUME_PARSING = "resume_parsing"
    JOB_DISCOVERY = "job_discovery"
    RESUME_MATCH = "resume_match"
    CONTACT_FINDING = "contact_finding"
    OUTREACH_COMPOSITION = "outreach_composition"
    OUTREACH_VALIDATION = "outreach_validation"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"


class RunStatus(str, Enum):
    """Overall status of a single pipeline run."""

    RUNNING = "running"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    COMPLETED = "completed"
    FAILED = "failed"


class ResumeProfileState(BaseModel):
    """Serializable mirror of
    `app.agents.resume_parser.parser_agent.ResumeProfile`, carried on the
    graph state. Kept as a separate Pydantic model (rather than reusing the
    dataclass directly) so `GraphState` has no dependency on agent-internal
    types, per docs/architecture.md §3 (the graph layer composes agents; it
    does not require agents to know about graph state types, and vice
    versa).
    """

    model_config = ConfigDict(frozen=True)

    resume_id: str
    file_hash: str
    full_name: str | None = None
    email: str | None = None
    skills: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    years_of_experience: float | None = None


class JobPostingState(BaseModel):
    """Serializable summary of a `app.db.models.JobPosting` row relevant to
    the pipeline. `role_title` and `jd_snapshot_text` are carried verbatim
    (PRD §6.2 — never paraphrased between stages).
    """

    model_config = ConfigDict(frozen=True)

    job_posting_id: str
    company_name: str
    role_title: str
    jd_url: str
    jd_snapshot_text: str
    jd_hash: str


class MatchResultState(BaseModel):
    """Serializable mirror of a resume-to-job similarity result
    (app.agents.resume_match.scorer.ScoredJobPosting)."""

    model_config = ConfigDict(frozen=True)

    similarity_score: float = Field(ge=0.0, le=1.0)
    meets_threshold: bool


class ContactState(BaseModel):
    """Serializable mirror of a resolved `app.db.models.Contact` row.

    `confidence_level` is a plain string restricted to
    'high' / 'medium' / 'low' (validated below) — consistent with
    `app.utils.confidence.ConfidenceLevel` / `app.db.models.ConfidenceLevel`
    (PRD §13.2: no "verified" value is ever valid here either).
    """

    model_config = ConfigDict(frozen=True)

    contact_id: str
    name: str | None = None
    title: str | None = None
    email: str | None = None
    confidence_level: str = Field(pattern="^(high|medium|low)$")


class OutreachDraftState(BaseModel):
    """Serializable mirror of a composed outreach email
    (app.agents.outreach.composer_agent.ComposedEmail) plus its persisted
    identity and approval status."""

    model_config = ConfigDict(frozen=True)

    outreach_email_id: str | None = None
    subject: str
    greeting: str
    body: str
    closing: str
    approved_by_human: bool = False


class ValidationIssueState(BaseModel):
    """Serializable mirror of
    `app.agents.outreach.validator_agent.ValidationIssue`."""

    model_config = ConfigDict(frozen=True)

    category: str
    severity: str
    description: str


class ValidationResultState(BaseModel):
    """Serializable mirror of
    `app.agents.outreach.validator_agent.ValidationResult`."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    issues: list[ValidationIssueState] = Field(default_factory=list)


class RunMetadata(BaseModel):
    """Metadata describing a single pipeline run, independent of any one
    stage's output. Supports checkpoint resumability (Epic 7, Story 2): a
    run can be identified and re-loaded by `run_id`.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    current_stage: PipelineStage = PipelineStage.RESUME_PARSING
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorInfo(BaseModel):
    """A single error encountered during a pipeline run.

    Errors are appended, not overwritten, so a run's full failure history
    remains inspectable even if a later stage recovers or retries
    (PRD §9 — auditability; PRD §12 — a single stage's failure should be
    diagnosable without losing prior context).
    """

    model_config = ConfigDict(frozen=True)

    stage: PipelineStage
    message: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphState(BaseModel):
    """The single state object threaded through every LangGraph node.

    Every field beyond `run_metadata` is optional/defaulted because a run
    populates them incrementally as it progresses through
    `PipelineStage` values — a node reads only the fields relevant to the
    stages that have already run, and writes only the field(s) corresponding
    to the stage it implements (Single Responsibility per node, per
    docs/architecture.md §3).
    """

    model_config = ConfigDict(frozen=False)

    run_metadata: RunMetadata = Field(default_factory=RunMetadata)
    resume_profile: ResumeProfileState | None = None
    job_posting: JobPostingState | None = None
    match_result: MatchResultState | None = None
    contact: ContactState | None = None
    outreach_draft: OutreachDraftState | None = None
    validation_result: ValidationResultState | None = None
    errors: list[ErrorInfo] = Field(default_factory=list)

    def advance_to(self, stage: PipelineStage) -> None:
        """Move the run to a new pipeline stage, updating `updated_at`.

        Logs the transition for observability (PRD §9), since a long-running
        graph execution otherwise has no visible progress trail between
        node invocations.
        """
        previous_stage = self.run_metadata.current_stage
        self.run_metadata = self.run_metadata.model_copy(
            update={
                "current_stage": stage,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        logger.info(
            "Pipeline run advanced stage",
            extra={
                "run_id": self.run_metadata.run_id,
                "previous_stage": previous_stage.value,
                "new_stage": stage.value,
            },
        )

    def record_error(self, stage: PipelineStage, message: str) -> None:
        """Append an error to the run's error history without discarding
        prior errors or state already accumulated on other fields.
        """
        self.errors.append(ErrorInfo(stage=stage, message=message))
        self.run_metadata = self.run_metadata.model_copy(
            update={"updated_at": datetime.now(timezone.utc)}
        )
        logger.warning(
            "Pipeline run recorded an error",
            extra={
                "run_id": self.run_metadata.run_id,
                "stage": stage.value,
                "message": message,
            },
        )

    def mark_failed(self) -> None:
        """Mark the run as failed, e.g. after an unrecoverable error."""
        self.run_metadata = self.run_metadata.model_copy(
            update={"status": RunStatus.FAILED, "updated_at": datetime.now(timezone.utc)}
        )
        logger.error(
            "Pipeline run marked failed",
            extra={"run_id": self.run_metadata.run_id, "error_count": len(self.errors)},
        )

    def mark_awaiting_human_review(self) -> None:
        """Mark the run as paused pending human approval of an outreach
        draft (PRD §6.4, §13.2 — no email is sent without approval)."""
        self.run_metadata = self.run_metadata.model_copy(
            update={
                "status": RunStatus.AWAITING_HUMAN_REVIEW,
                "current_stage": PipelineStage.AWAITING_APPROVAL,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        logger.info(
            "Pipeline run awaiting human review",
            extra={"run_id": self.run_metadata.run_id},
        )

    def mark_completed(self) -> None:
        """Mark the run as fully completed."""
        self.run_metadata = self.run_metadata.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "current_stage": PipelineStage.COMPLETED,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        logger.info(
            "Pipeline run completed",
            extra={"run_id": self.run_metadata.run_id},
        )

    def to_checkpoint_json(self) -> str:
        """Serialize this state to a JSON string suitable for persistence by
        `app.graph.checkpointer` (Epic 7, Story 2)."""
        return self.model_dump_json()

    @classmethod
    def from_checkpoint_json(cls, raw_json: str) -> GraphState:
        """Deserialize a previously checkpointed state.

        Raises:
            pydantic.ValidationError: if `raw_json` does not match this
                schema (e.g. after a breaking schema change).
        """
        return cls.model_validate_json(raw_json)