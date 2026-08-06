"""
Unit tests for the Postgres-backed LangGraph checkpointer.

Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 3 -
Checkpoint Persistence, Task 1.

Exercises `GraphCheckpointer` against an in-memory SQLite database built
from the application's real `Base.metadata` (app/db/base.py,
app/graph/checkpointer.py's `GraphCheckpoint` model), per
docs/coding_guidelines.md §6 — no test hits a real Postgres instance.
Covers: save/load round-tripping, update-in-place on re-save, resume-status
enforcement, and age-based cleanup of terminal runs only.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.graph.checkpointer import (
    CheckpointNotFoundError,
    CheckpointNotResumableError,
    GraphCheckpoint,
    GraphCheckpointer,
)
from app.graph.state_schema import GraphState, JobPostingState, PipelineStage, RunStatus


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, future=True)
    db_session = session_factory()
    yield db_session
    db_session.close()


@pytest.fixture
def checkpointer(session: Session) -> GraphCheckpointer:
    return GraphCheckpointer(session)


@pytest.fixture
def running_state() -> GraphState:
    return GraphState()


@pytest.fixture
def awaiting_review_state() -> GraphState:
    state = GraphState()
    state.job_posting = JobPostingState(
        job_posting_id="job-1",
        company_name="Acme Inc",
        role_title="Senior Backend Engineer",
        jd_url="https://boards.greenhouse.io/acme/jobs/1",
        jd_snapshot_text="We are hiring a Senior Backend Engineer.",
        jd_hash="a" * 64,
    )
    state.mark_awaiting_human_review()
    return state


@pytest.fixture
def completed_state() -> GraphState:
    state = GraphState()
    state.mark_completed()
    return state


@pytest.fixture
def failed_state() -> GraphState:
    state = GraphState()
    state.record_error(PipelineStage.CONTACT_FINDING, "Domain resolution failed")
    state.mark_failed()
    return state


class TestSaveAndLoadCheckpoint:
    def test_save_then_load_round_trips_state(
        self, checkpointer: GraphCheckpointer, running_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(running_state)

        loaded = checkpointer.load_checkpoint(running_state.run_metadata.run_id)

        assert loaded is not None
        assert loaded.run_metadata.run_id == running_state.run_metadata.run_id
        assert loaded.run_metadata.status == RunStatus.RUNNING

    def test_load_returns_none_for_unknown_run_id(
        self, checkpointer: GraphCheckpointer
    ) -> None:
        assert checkpointer.load_checkpoint("nonexistent-run-id") is None

    def test_save_persists_job_posting_field(
        self, checkpointer: GraphCheckpointer, awaiting_review_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(awaiting_review_state)

        loaded = checkpointer.load_checkpoint(awaiting_review_state.run_metadata.run_id)

        assert loaded is not None
        assert loaded.job_posting is not None
        assert loaded.job_posting.company_name == "Acme Inc"

    def test_resaving_same_run_id_updates_row_in_place(
        self, checkpointer: GraphCheckpointer, session: Session, running_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(running_state)

        running_state.advance_to(PipelineStage.RESUME_MATCH)
        checkpointer.save_checkpoint(running_state)

        row_count = session.query(GraphCheckpoint).filter(
            GraphCheckpoint.run_id == running_state.run_metadata.run_id
        ).count()
        assert row_count == 1

        loaded = checkpointer.load_checkpoint(running_state.run_metadata.run_id)
        assert loaded.run_metadata.current_stage == PipelineStage.RESUME_MATCH

    def test_save_stores_status_matching_run_metadata(
        self, checkpointer: GraphCheckpointer, session: Session, completed_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(completed_state)

        row = session.get(GraphCheckpoint, completed_state.run_metadata.run_id)
        assert row is not None
        assert row.status == "completed"


class TestResume:
    def test_resume_returns_state_for_running_checkpoint(
        self, checkpointer: GraphCheckpointer, running_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(running_state)

        resumed = checkpointer.resume(running_state.run_metadata.run_id)

        assert resumed.run_metadata.run_id == running_state.run_metadata.run_id

    def test_resume_returns_state_for_awaiting_review_checkpoint(
        self, checkpointer: GraphCheckpointer, awaiting_review_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(awaiting_review_state)

        resumed = checkpointer.resume(awaiting_review_state.run_metadata.run_id)

        assert resumed.run_metadata.status == RunStatus.AWAITING_HUMAN_REVIEW

    def test_resume_raises_not_found_for_unknown_run_id(
        self, checkpointer: GraphCheckpointer
    ) -> None:
        with pytest.raises(CheckpointNotFoundError):
            checkpointer.resume("nonexistent-run-id")

    def test_resume_raises_not_resumable_for_completed_run(
        self, checkpointer: GraphCheckpointer, completed_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(completed_state)

        with pytest.raises(CheckpointNotResumableError):
            checkpointer.resume(completed_state.run_metadata.run_id)

    def test_resume_raises_not_resumable_for_failed_run(
        self, checkpointer: GraphCheckpointer, failed_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(failed_state)

        with pytest.raises(CheckpointNotResumableError):
            checkpointer.resume(failed_state.run_metadata.run_id)


class TestCleanupCompletedRuns:
    def test_cleanup_deletes_old_completed_checkpoint(
        self, checkpointer: GraphCheckpointer, session: Session, completed_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(completed_state)
        row = session.get(GraphCheckpoint, completed_state.run_metadata.run_id)
        row.updated_at = datetime.now(timezone.utc) - timedelta(days=60)
        session.flush()

        deleted_count = checkpointer.cleanup_completed_runs(older_than_days=30)

        assert deleted_count == 1
        assert session.get(GraphCheckpoint, completed_state.run_metadata.run_id) is None

    def test_cleanup_deletes_old_failed_checkpoint(
        self, checkpointer: GraphCheckpointer, session: Session, failed_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(failed_state)
        row = session.get(GraphCheckpoint, failed_state.run_metadata.run_id)
        row.updated_at = datetime.now(timezone.utc) - timedelta(days=60)
        session.flush()

        deleted_count = checkpointer.cleanup_completed_runs(older_than_days=30)

        assert deleted_count == 1

    def test_cleanup_does_not_delete_recent_completed_checkpoint(
        self, checkpointer: GraphCheckpointer, session: Session, completed_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(completed_state)

        deleted_count = checkpointer.cleanup_completed_runs(older_than_days=30)

        assert deleted_count == 0
        assert session.get(GraphCheckpoint, completed_state.run_metadata.run_id) is not None

    def test_cleanup_never_deletes_running_checkpoint_regardless_of_age(
        self, checkpointer: GraphCheckpointer, session: Session, running_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(running_state)
        row = session.get(GraphCheckpoint, running_state.run_metadata.run_id)
        row.updated_at = datetime.now(timezone.utc) - timedelta(days=365)
        session.flush()

        deleted_count = checkpointer.cleanup_completed_runs(older_than_days=30)

        assert deleted_count == 0
        assert session.get(GraphCheckpoint, running_state.run_metadata.run_id) is not None

    def test_cleanup_never_deletes_awaiting_review_checkpoint_regardless_of_age(
        self, checkpointer: GraphCheckpointer, session: Session, awaiting_review_state: GraphState
    ) -> None:
        checkpointer.save_checkpoint(awaiting_review_state)
        row = session.get(GraphCheckpoint, awaiting_review_state.run_metadata.run_id)
        row.updated_at = datetime.now(timezone.utc) - timedelta(days=365)
        session.flush()

        deleted_count = checkpointer.cleanup_completed_runs(older_than_days=30)

        assert deleted_count == 0
        assert session.get(GraphCheckpoint, awaiting_review_state.run_metadata.run_id) is not None
    def test_cleanup_rejects_negative_older_than_days(
        self, checkpointer: GraphCheckpointer
    ) -> None:
        from app.graph.checkpointer import CheckpointError

        with pytest.raises(CheckpointError, match="non-negative"):
            checkpointer.cleanup_completed_runs(older_than_days=-1)