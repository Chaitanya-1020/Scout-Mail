"""
Postgres-backed LangGraph checkpointer.

Implements: PRD §9 (Auditability — pipeline run state is durably persisted,
not held only in process memory), §12 (Risks — a long-running or failed
pipeline run must be resumable rather than restarted from scratch), §7
(Tech Stack — PostgreSQL via Supabase free tier, reused for checkpoint
storage rather than a separate persistence backend).
Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 3 -
Checkpoint Persistence, Task 1.

Persists `GraphState` (app/graph/state_schema.py) snapshots keyed by
`run_id`, using the state's own `to_checkpoint_json` /
`from_checkpoint_json` methods for serialization so this module has no
knowledge of the state schema's internal shape (Single Responsibility, per
docs/coding_guidelines.md). Follows the repository pattern established by
`app/agents/*/repository.py`: a plain class wrapping a SQLAlchemy `Session`,
with no dependency on `app/graph/state_graph.py` or any concrete agent.
Transient database errors are retried with backoff via `tenacity`,
consistent with `app/llm/ollama_client.py`'s retry approach.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import DateTime, String, Text, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Mapped, Session, mapped_column
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.db.base import Base
from app.graph.state_schema import GraphState, RunStatus

logger = logging.getLogger(__name__)


class CheckpointError(Exception):
    """Raised when a checkpoint cannot be saved, loaded, or resumed."""


class CheckpointNotFoundError(CheckpointError):
    """Raised when a requested checkpoint does not exist."""


class CheckpointNotResumableError(CheckpointError):
    """Raised when a checkpoint exists but is not in a resumable status
    (e.g. already completed)."""


class CheckpointStatus(str, Enum):
    """Lifecycle status of a persisted checkpoint row.

    Mirrors `app.graph.state_schema.RunStatus` values but is kept as its own
    enum so this persistence-layer module does not require importing
    `RunStatus` for column typing (only for translating an incoming
    `GraphState`'s status at save time).
    """

    RUNNING = "running"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    COMPLETED = "completed"
    FAILED = "failed"


_RUN_STATUS_TO_CHECKPOINT_STATUS: dict[RunStatus, CheckpointStatus] = {
    RunStatus.RUNNING: CheckpointStatus.RUNNING,
    RunStatus.AWAITING_HUMAN_REVIEW: CheckpointStatus.AWAITING_HUMAN_REVIEW,
    RunStatus.COMPLETED: CheckpointStatus.COMPLETED,
    RunStatus.FAILED: CheckpointStatus.FAILED,
}

_RESUMABLE_STATUSES: frozenset[CheckpointStatus] = frozenset(
    {CheckpointStatus.RUNNING, CheckpointStatus.AWAITING_HUMAN_REVIEW}
)

_TERMINAL_STATUSES: frozenset[CheckpointStatus] = frozenset(
    {CheckpointStatus.COMPLETED, CheckpointStatus.FAILED}
)


class GraphCheckpoint(Base):
    """Persisted snapshot of a `GraphState`, keyed by its `run_id`."""

    __tablename__ = "graph_checkpoints"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class GraphCheckpointer:
    """Repository for saving, loading, resuming, and cleaning up pipeline
    run checkpoints in Postgres.

    Database operations that may hit a transient connectivity issue (e.g. a
    brief Supabase free-tier connection drop) are retried with exponential
    backoff, consistent with the retry approach used elsewhere for external
    dependencies (`app/llm/ollama_client.py`,
    `app/agents/contact_finder/smtp_validator.py`).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(OperationalError),
        reraise=True,
    )
    def save_checkpoint(self, state: GraphState) -> None:
        """Persist a `GraphState` snapshot, creating or updating the row for
        its `run_id`.

        Raises:
            CheckpointError: if the state cannot be serialized or the
                database write fails after retries.
        """
        run_id = state.run_metadata.run_id

        try:
            serialized = state.to_checkpoint_json()
        except Exception as exc:  # noqa: BLE001 - normalize serialization failure
            raise CheckpointError(
                f"Failed to serialize state for run '{run_id}'."
            ) from exc

        checkpoint_status = _RUN_STATUS_TO_CHECKPOINT_STATUS.get(
            state.run_metadata.status, CheckpointStatus.RUNNING
        )

        try:
            existing = self._session.get(GraphCheckpoint, run_id)

            if existing is None:
                self._session.add(
                    GraphCheckpoint(
                        run_id=run_id,
                        status=checkpoint_status.value,
                        state_json=serialized,
                    )
                )
            else:
                existing.status = checkpoint_status.value
                existing.state_json = serialized

            self._session.flush()
        except OperationalError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize unexpected DB failure
            raise CheckpointError(
                f"Failed to save checkpoint for run '{run_id}'."
            ) from exc

        logger.info(
            "Checkpoint saved",
            extra={"run_id": run_id, "status": checkpoint_status.value},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(OperationalError),
        reraise=True,
    )
    def load_checkpoint(self, run_id: str) -> GraphState | None:
        """Load a previously saved `GraphState` by `run_id`.

        Returns:
            The deserialized `GraphState`, or None if no checkpoint exists
            for `run_id`.

        Raises:
            CheckpointError: if a checkpoint row exists but cannot be
                deserialized (e.g. corrupted or schema-incompatible JSON).
        """
        try:
            row = self._session.get(GraphCheckpoint, run_id)
        except OperationalError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize unexpected DB failure
            raise CheckpointError(
                f"Failed to load checkpoint for run '{run_id}'."
            ) from exc

        if row is None:
            logger.debug("No checkpoint found", extra={"run_id": run_id})
            return None

        try:
            state = GraphState.from_checkpoint_json(row.state_json)
        except Exception as exc:  # noqa: BLE001 - normalize deserialization failure
            raise CheckpointError(
                f"Failed to deserialize checkpoint for run '{run_id}'."
            ) from exc

        logger.info("Checkpoint loaded", extra={"run_id": run_id, "status": row.status})
        return state

    def resume(self, run_id: str) -> GraphState:
        """Load a checkpoint and confirm it is in a resumable status.

        A checkpoint is resumable if its persisted status is `running` or
        `awaiting_human_review`; a `completed` or `failed` run must not be
        silently re-entered into the pipeline (PRD §12 — a terminal run's
        outcome must not be overwritten by an accidental resume).

        Raises:
            CheckpointNotFoundError: if no checkpoint exists for `run_id`.
            CheckpointNotResumableError: if the checkpoint is in a terminal
                status.
            CheckpointError: if the checkpoint cannot be loaded.
        """
        row = self._session.get(GraphCheckpoint, run_id)
        if row is None:
            raise CheckpointNotFoundError(f"No checkpoint found for run '{run_id}'.")

        status = CheckpointStatus(row.status)
        if status not in _RESUMABLE_STATUSES:
            raise CheckpointNotResumableError(
                f"Checkpoint for run '{run_id}' has status '{status.value}' and is "
                "not resumable."
            )

        state = self.load_checkpoint(run_id)
        if state is None:
            # Should not occur given the row lookup above, but guarded
            # explicitly rather than allowing a None to propagate silently.
            raise CheckpointNotFoundError(f"No checkpoint found for run '{run_id}'.")

        logger.info("Checkpoint resumed", extra={"run_id": run_id, "status": status.value})
        return state

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(OperationalError),
        reraise=True,
    )
    def cleanup_completed_runs(self, older_than_days: int = 30) -> int:
        """Delete checkpoints for terminal (`completed`/`failed`) runs older
        than `older_than_days`, based on `updated_at`.

        Running or awaiting-human-review checkpoints are never deleted by
        this method, regardless of age, since deleting them would make an
        in-progress or pending-approval run unresumable.

        Args:
            older_than_days: minimum age, in days, since the checkpoint was
                last updated, for it to be eligible for cleanup.

        Returns:
            The number of checkpoint rows deleted.

        Raises:
            CheckpointError: if the cleanup query fails.
        """
        if older_than_days < 0:
            raise CheckpointError("older_than_days must be non-negative.")

        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        try:
            rows_to_delete = list(
                self._session.execute(
                    select(GraphCheckpoint).where(
                        GraphCheckpoint.status.in_(
                            [status.value for status in _TERMINAL_STATUSES]
                        ),
                        GraphCheckpoint.updated_at < cutoff,
                    )
                ).scalars()
            )

            for row in rows_to_delete:
                self._session.delete(row)

            self._session.flush()
        except OperationalError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize unexpected DB failure
            raise CheckpointError("Failed to clean up completed checkpoints.") from exc

        deleted_count = len(rows_to_delete)
        logger.info(
            "Completed checkpoints cleaned up",
            extra={"deleted_count": deleted_count, "older_than_days": older_than_days},
        )
        return deleted_count