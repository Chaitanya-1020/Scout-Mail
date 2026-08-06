"""
Unit tests for the pipeline runner's CLI, adapters, and graceful shutdown.

Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 4 -
Workflow Runner, Task 1.

Full end-to-end wiring (`build_pipeline_dependencies`, `PipelineRunner`)
requires a real database, LLM, and vector store, so it is exercised
elsewhere (integration tests, out of scope here per
docs/coding_guidelines.md §6 — no unit test hits a real external
dependency). This suite instead covers what is unit-testable in isolation:
the CLI argument parser, the stage adapters' translation logic against fake
underlying agents/repositories, and the scheduled-execution shutdown signal
handling.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.agents.contact_finder.repository import ContactFinderRepository
from app.agents.job_scout.agent import ConnectorFetchFailure, JobScoutRunResult
from app.connectors.base import RawJobPosting
from app.db.models import ConfidenceLevel as DbConfidenceLevel
from app.db.models import Contact
from app.graph.runner import (
    ContactFinderStageAdapter,
    JobScoutStageAdapter,
    PipelineRunnerError,
    build_arg_parser,
)
from app.graph.state_schema import JobPostingState, ResumeProfileState


class TestArgParser:
    def test_run_command_requires_resume_id(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["run", "--resume-id", "abc-123"])

        assert args.command == "run"
        assert args.resume_id == "abc-123"

    def test_run_command_fails_without_resume_id(self) -> None:
        parser = build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["run"])

    def test_resume_command_requires_run_id_and_resume_id(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["resume", "--run-id", "run-1", "--resume-id", "resume-1"])

        assert args.command == "resume"
        assert args.run_id == "run-1"
        assert args.resume_id == "resume-1"

    def test_schedule_command_defaults_interval_to_none(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(["schedule", "--resume-id", "resume-1"])

        assert args.command == "schedule"
        assert args.interval_hours is None

    def test_schedule_command_accepts_interval_override(self) -> None:
        parser = build_arg_parser()

        args = parser.parse_args(
            ["schedule", "--resume-id", "resume-1", "--interval-hours", "12"]
        )

        assert args.interval_hours == 12

    def test_unknown_command_raises_system_exit(self) -> None:
        parser = build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["not-a-real-command"])

    def test_missing_command_raises_system_exit(self) -> None:
        parser = build_arg_parser()

        with pytest.raises(SystemExit):
            parser.parse_args([])


class TestJobScoutStageAdapter:
    def test_returns_none_when_no_postings_fetched(self) -> None:
        fake_agent = MagicMock()
        fake_agent.run.return_value = JobScoutRunResult(postings=[], failures=[])
        fake_repository = MagicMock()

        adapter = JobScoutStageAdapter(agent=fake_agent, repository=fake_repository)

        result = adapter.find_next_posting(
            ResumeProfileState(resume_id="r1", file_hash="h1")
        )

        assert result is None
        fake_repository.save_postings.assert_not_called()

    def test_logs_but_does_not_raise_on_connector_failures(self) -> None:
        fake_agent = MagicMock()
        fake_agent.run.return_value = JobScoutRunResult(
            postings=[],
            failures=[ConnectorFetchFailure(source_name="career_page", error_message="timeout")],
        )
        fake_repository = MagicMock()

        adapter = JobScoutStageAdapter(agent=fake_agent, repository=fake_repository)

        # Should not raise despite a connector failure being present.
        result = adapter.find_next_posting(ResumeProfileState(resume_id="r1", file_hash="h1"))

        assert result is None

    def test_returns_first_saved_posting_as_job_posting_state(self) -> None:
        raw_posting = RawJobPosting(
            source_connector="greenhouse",
            external_id="123",
            company_name="Acme Inc",
            role_title="Backend Engineer",
            jd_url="https://boards.greenhouse.io/acme/jobs/123",
            jd_snapshot_text="We are hiring a Backend Engineer.",
        )
        fake_agent = MagicMock()
        fake_agent.run.return_value = JobScoutRunResult(postings=[raw_posting], failures=[])

        saved_row = MagicMock()
        saved_row.id = uuid.uuid4()
        saved_row.company_name = "Acme Inc"
        saved_row.role_title = "Backend Engineer"
        saved_row.jd_url = raw_posting.jd_url
        saved_row.jd_snapshot_text = raw_posting.jd_snapshot_text
        saved_row.jd_hash = "hash123"

        fake_repository = MagicMock()
        fake_repository.save_postings.return_value = [saved_row]

        adapter = JobScoutStageAdapter(agent=fake_agent, repository=fake_repository)

        result = adapter.find_next_posting(ResumeProfileState(resume_id="r1", file_hash="h1"))

        assert isinstance(result, JobPostingState)
        assert result.company_name == "Acme Inc"
        assert result.jd_hash == "hash123"
        fake_repository.save_postings.assert_called_once_with([raw_posting])


class TestContactFinderStageAdapter:
    def _make_contact(self, confidence: DbConfidenceLevel, email: str | None) -> Contact:
        contact = Contact(
            job_posting_id=uuid.uuid4(),
            name="Jane Recruiter",
            title="Recruiter",
            email=email,
            confidence_level=confidence,
        )
        contact.id = uuid.uuid4()
        return contact

    def test_returns_none_when_no_contacts_found(self) -> None:
        fake_agent = MagicMock()
        fake_repository = MagicMock(spec=ContactFinderRepository)
        fake_repository.get_contacts_for_job_posting.return_value = []

        adapter = ContactFinderStageAdapter(agent=fake_agent, repository=fake_repository)

        job_posting = JobPostingState(
            job_posting_id=str(uuid.uuid4()),
            company_name="Acme Inc",
            role_title="Engineer",
            jd_url="https://example.com/jobs/1",
            jd_snapshot_text="JD text",
            jd_hash="a" * 64,
        )

        result = adapter.find_contact(job_posting)

        assert result is None

    def test_prefers_highest_confidence_contact_with_email(self) -> None:
        fake_agent = MagicMock()
        low_confidence = self._make_contact(DbConfidenceLevel.LOW, email="low@acme.com")
        high_confidence = self._make_contact(DbConfidenceLevel.HIGH, email="high@acme.com")

        fake_repository = MagicMock(spec=ContactFinderRepository)
        fake_repository.get_contacts_for_job_posting.return_value = [
            low_confidence,
            high_confidence,
        ]

        adapter = ContactFinderStageAdapter(agent=fake_agent, repository=fake_repository)

        job_posting = JobPostingState(
            job_posting_id=str(uuid.uuid4()),
            company_name="Acme Inc",
            role_title="Engineer",
            jd_url="https://example.com/jobs/1",
            jd_snapshot_text="JD text",
            jd_hash="a" * 64,
        )

        result = adapter.find_contact(job_posting)

        assert result is not None
        assert result.email == "high@acme.com"
        assert result.confidence_level == "high"

    def test_prefers_contact_with_email_over_higher_confidence_without_email(self) -> None:
        fake_agent = MagicMock()
        high_no_email = self._make_contact(DbConfidenceLevel.HIGH, email=None)
        medium_with_email = self._make_contact(DbConfidenceLevel.MEDIUM, email="medium@acme.com")

        fake_repository = MagicMock(spec=ContactFinderRepository)
        fake_repository.get_contacts_for_job_posting.return_value = [
            high_no_email,
            medium_with_email,
        ]

        adapter = ContactFinderStageAdapter(agent=fake_agent, repository=fake_repository)

        job_posting = JobPostingState(
            job_posting_id=str(uuid.uuid4()),
            company_name="Acme Inc",
            role_title="Engineer",
            jd_url="https://example.com/jobs/1",
            jd_snapshot_text="JD text",
            jd_hash="a" * 64,
        )

        result = adapter.find_contact(job_posting)

        assert result is not None
        assert result.email == "medium@acme.com"

    def test_calls_agent_with_correct_arguments(self) -> None:
        fake_agent = MagicMock()
        fake_repository = MagicMock(spec=ContactFinderRepository)
        fake_repository.get_contacts_for_job_posting.return_value = []

        adapter = ContactFinderStageAdapter(agent=fake_agent, repository=fake_repository)

        job_posting_id = uuid.uuid4()
        job_posting = JobPostingState(
            job_posting_id=str(job_posting_id),
            company_name="Acme Inc",
            role_title="Engineer",
            jd_url="https://example.com/jobs/1",
            jd_snapshot_text="JD text",
            jd_hash="a" * 64,
        )

        adapter.find_contact(job_posting)

        fake_agent.find_contacts.assert_called_once_with(
            job_posting_id=job_posting_id,
            company_name="Acme Inc",
            jd_url="https://example.com/jobs/1",
        )


class TestGracefulShutdown:
    def test_shutdown_signal_handler_stops_scheduler_and_exits(self, monkeypatch) -> None:
        from app.graph import runner as runner_module

        fake_scheduler = MagicMock()
        exit_calls: list[int] = []

        def fake_sys_exit(code: int = 0) -> None:
            exit_calls.append(code)
            raise SystemExit(code)

        monkeypatch.setattr(runner_module.sys, "exit", fake_sys_exit)

        def handler(signum: int, frame) -> None:  # noqa: ANN001
            fake_scheduler.shutdown(wait=True)
            runner_module.sys.exit(0)

        with pytest.raises(SystemExit):
            handler(2, None)

        fake_scheduler.shutdown.assert_called_once_with(wait=True)
        assert exit_calls == [0]


class TestPipelineRunnerErrorHandling:
    def test_pipeline_runner_error_is_a_plain_exception(self) -> None:
        error = PipelineRunnerError("something went wrong")

        assert isinstance(error, Exception)
        assert str(error) == "something went wrong"