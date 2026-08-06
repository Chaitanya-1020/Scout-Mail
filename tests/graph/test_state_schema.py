"""
Unit tests for the shared LangGraph state schema.

Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 1 -
Shared State Schema, Task 1.

Covers: default construction, incremental field population, stage/status
transition helpers, error accumulation, and checkpoint JSON round-tripping
(the property the Postgres checkpointer, Epic 7 Story 2, will depend on).
No external dependencies are exercised — `GraphState` and its nested models
are pure Pydantic data, per docs/coding_guidelines.md §6.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.graph.state_schema import (
    ContactState,
    ErrorInfo,
    GraphState,
    JobPostingState,
    MatchResultState,
    OutreachDraftState,
    PipelineStage,
    ResumeProfileState,
    RunMetadata,
    RunStatus,
    ValidationIssueState,
    ValidationResultState,
)


class TestGraphStateDefaults:
    def test_default_state_has_no_stage_data(self) -> None:
        state = GraphState()

        assert state.resume_profile is None
        assert state.job_posting is None
        assert state.match_result is None
        assert state.contact is None
        assert state.outreach_draft is None
        assert state.validation_result is None
        assert state.errors == []

    def test_default_run_metadata_starts_at_resume_parsing_and_running(self) -> None:
        state = GraphState()

        assert state.run_metadata.current_stage == PipelineStage.RESUME_PARSING
        assert state.run_metadata.status == RunStatus.RUNNING
        assert state.run_metadata.run_id  # non-empty

    def test_each_state_gets_a_unique_run_id(self) -> None:
        state_a = GraphState()
        state_b = GraphState()

        assert state_a.run_metadata.run_id != state_b.run_metadata.run_id


class TestIncrementalFieldPopulation:
    def test_can_populate_resume_profile(self) -> None:
        state = GraphState()
        state.resume_profile = ResumeProfileState(
            resume_id="resume-1",
            file_hash="abc123",
            full_name="Jane Doe",
            skills=["Python", "SQL"],
            target_roles=["Backend Engineer"],
        )

        assert state.resume_profile.full_name == "Jane Doe"
        assert "Python" in state.resume_profile.skills

    def test_can_populate_job_posting(self) -> None:
        state = GraphState()
        state.job_posting = JobPostingState(
            job_posting_id="job-1",
            company_name="Acme Inc",
            role_title="Senior Backend Engineer",
            jd_url="https://boards.greenhouse.io/acme/jobs/1",
            jd_snapshot_text="We are hiring a Senior Backend Engineer.",
            jd_hash="b" * 64,
        )

        assert state.job_posting.company_name == "Acme Inc"

    def test_can_populate_match_result(self) -> None:
        state = GraphState()
        state.match_result = MatchResultState(similarity_score=0.82, meets_threshold=True)

        assert state.match_result.similarity_score == 0.82
        assert state.match_result.meets_threshold is True

    def test_match_result_rejects_out_of_range_score(self) -> None:
        with pytest.raises(ValidationError):
            MatchResultState(similarity_score=1.5, meets_threshold=True)

    def test_can_populate_contact_with_valid_confidence_level(self) -> None:
        state = GraphState()
        state.contact = ContactState(
            contact_id="contact-1",
            name="John Smith",
            title="Recruiter",
            email="john@acme.com",
            confidence_level="high",
        )

        assert state.contact.confidence_level == "high"

    def test_contact_rejects_verified_as_confidence_level(self) -> None:
        # Per PRD §13.2, "verified" must never be an accepted confidence
        # value anywhere in the codebase, including on the graph state.
        with pytest.raises(ValidationError):
            ContactState(
                contact_id="contact-1",
                name="John Smith",
                title="Recruiter",
                email="john@acme.com",
                confidence_level="verified",
            )

    def test_can_populate_outreach_draft(self) -> None:
        state = GraphState()
        state.outreach_draft = OutreachDraftState(
            subject="Interest in the role",
            greeting="Hi John,",
            body="Reaching out about the Senior Backend Engineer role.",
            closing="Best regards,",
        )

        assert state.outreach_draft.approved_by_human is False

    def test_can_populate_validation_result_with_issues(self) -> None:
        state = GraphState()
        state.validation_result = ValidationResultState(
            passed=False,
            issues=[
                ValidationIssueState(
                    category="fabricated_claim",
                    severity="error",
                    description="Unsupported claim about experience",
                )
            ],
        )

        assert state.validation_result.passed is False
        assert len(state.validation_result.issues) == 1


class TestStageTransitions:
    def test_advance_to_updates_current_stage(self) -> None:
        state = GraphState()

        state.advance_to(PipelineStage.JOB_DISCOVERY)

        assert state.run_metadata.current_stage == PipelineStage.JOB_DISCOVERY

    def test_advance_to_updates_timestamp(self) -> None:
        state = GraphState()
        original_updated_at = state.run_metadata.updated_at

        state.advance_to(PipelineStage.RESUME_MATCH)

        assert state.run_metadata.updated_at >= original_updated_at

    def test_mark_awaiting_human_review_sets_status_and_stage(self) -> None:
        state = GraphState()

        state.mark_awaiting_human_review()

        assert state.run_metadata.status == RunStatus.AWAITING_HUMAN_REVIEW
        assert state.run_metadata.current_stage == PipelineStage.AWAITING_APPROVAL

    def test_mark_completed_sets_status_and_stage(self) -> None:
        state = GraphState()

        state.mark_completed()

        assert state.run_metadata.status == RunStatus.COMPLETED
        assert state.run_metadata.current_stage == PipelineStage.COMPLETED

    def test_mark_failed_sets_status(self) -> None:
        state = GraphState()

        state.mark_failed()

        assert state.run_metadata.status == RunStatus.FAILED


class TestErrorAccumulation:
    def test_record_error_appends_without_clearing_existing_errors(self) -> None:
        state = GraphState()

        state.record_error(PipelineStage.CONTACT_FINDING, "Domain resolution failed")
        state.record_error(PipelineStage.OUTREACH_COMPOSITION, "LLM timeout")

        assert len(state.errors) == 2
        assert state.errors[0].stage == PipelineStage.CONTACT_FINDING
        assert state.errors[0].message == "Domain resolution failed"
        assert state.errors[1].stage == PipelineStage.OUTREACH_COMPOSITION

    def test_record_error_does_not_clear_populated_fields(self) -> None:
        state = GraphState()
        state.job_posting = JobPostingState(
            job_posting_id="job-1",
            company_name="Acme Inc",
            role_title="Engineer",
            jd_url="https://example.com/jobs/1",
            jd_snapshot_text="JD text",
            jd_hash="c" * 64,
        )

        state.record_error(PipelineStage.CONTACT_FINDING, "Domain resolution failed")

        assert state.job_posting is not None
        assert state.job_posting.company_name == "Acme Inc"

    def test_error_info_has_stage_message_and_timestamp(self) -> None:
        error = ErrorInfo(stage=PipelineStage.RESUME_PARSING, message="Extraction failed")

        assert error.stage == PipelineStage.RESUME_PARSING
        assert error.message == "Extraction failed"
        assert error.occurred_at is not None


class TestCheckpointSerialization:
    def test_round_trips_default_state(self) -> None:
        state = GraphState()

        serialized = state.to_checkpoint_json()
        restored = GraphState.from_checkpoint_json(serialized)

        assert restored.run_metadata.run_id == state.run_metadata.run_id
        assert restored.run_metadata.current_stage == state.run_metadata.current_stage

    def test_round_trips_fully_populated_state(self) -> None:
        state = GraphState()
        state.resume_profile = ResumeProfileState(
            resume_id="resume-1", file_hash="abc123", full_name="Jane Doe"
        )
        state.job_posting = JobPostingState(
            job_posting_id="job-1",
            company_name="Acme Inc",
            role_title="Engineer",
            jd_url="https://example.com/jobs/1",
            jd_snapshot_text="JD text",
            jd_hash="c" * 64,
        )
        state.match_result = MatchResultState(similarity_score=0.75, meets_threshold=True)
        state.contact = ContactState(
            contact_id="contact-1",
            name="John Smith",
            email="john@acme.com",
            confidence_level="medium",
        )
        state.outreach_draft = OutreachDraftState(
            subject="Subject", greeting="Hi,", body="Body", closing="Bye,"
        )
        state.validation_result = ValidationResultState(passed=True, issues=[])
        state.record_error(PipelineStage.CONTACT_FINDING, "Minor issue")
        state.advance_to(PipelineStage.OUTREACH_VALIDATION)

        serialized = state.to_checkpoint_json()
        restored = GraphState.from_checkpoint_json(serialized)

        assert restored.resume_profile.full_name == "Jane Doe"
        assert restored.job_posting.company_name == "Acme Inc"
        assert restored.match_result.similarity_score == 0.75
        assert restored.contact.confidence_level == "medium"
        assert restored.outreach_draft.subject == "Subject"
        assert restored.validation_result.passed is True
        assert len(restored.errors) == 1
        assert restored.errors[0].message == "Minor issue"
        assert restored.run_metadata.current_stage == PipelineStage.OUTREACH_VALIDATION

    def test_round_trip_preserves_run_metadata_object_shape(self) -> None:
        state = GraphState()

        serialized = state.to_checkpoint_json()
        restored = GraphState.from_checkpoint_json(serialized)

        assert isinstance(restored.run_metadata, RunMetadata)

    def test_from_checkpoint_json_rejects_invalid_confidence_level(self) -> None:
        state = GraphState()
        state.contact = ContactState(
            contact_id="contact-1", email="john@acme.com", confidence_level="high"
        )
        serialized = state.to_checkpoint_json()

        # Tamper with the serialized JSON to simulate a corrupted/invalid
        # checkpoint, verifying deserialization still enforces the schema.
        tampered = serialized.replace('"high"', '"verified"')

        with pytest.raises(ValidationError):
            GraphState.from_checkpoint_json(tampered)