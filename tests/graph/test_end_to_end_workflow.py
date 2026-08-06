"""
End-to-end integration test for the resume-to-outreach pipeline workflow.

Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 5 -
End-to-End Workflow Integration, Task 1.

Exercises the real, compiled LangGraph pipeline (`app.graph.state_graph.
build_state_graph`) together with the real Postgres-backed checkpointer
(`app.graph.checkpointer.GraphCheckpointer`, run against an in-memory SQLite
database built from the application's actual `Base.metadata`, consistent
with `tests/outreach/test_repository.py` and
`tests/graph/test_checkpointer.py`). External services (Job Scout
connectors, Resume Match embeddings, Contact Finder's domain/SMTP/Hunter.io
lookups, and the Outreach Composer/Validator LLM calls) are mocked via fake
stage implementations satisfying the `Protocol`s declared in
`app.graph.state_graph` — per docs/coding_guidelines.md §6, no test hits a
real external API, LLM, or production database. This test verifies the
integration of the graph, the state schema, and the checkpointer together,
which no single unit test suite covers in combination.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.base import Base
from app.graph.checkpointer import GraphCheckpointer
from app.graph.state_graph import GraphAgents, build_state_graph
from app.graph.state_schema import (
    ContactState,
    GraphState,
    JobPostingState,
    MatchResultState,
    OutreachDraftState,
    PipelineStage,
    ResumeProfileState,
    RunStatus,
    ValidationResultState,
)


# --- Fakes for external-service-backed stages -------------------------------
#
# Each fake satisfies the corresponding Protocol in app.graph.state_graph
# and records the order in which it was invoked via a shared call-order
# list, so this test can assert the pipeline executed its stages in the
# correct sequence.


class RecordingJobScoutStage:
    def __init__(self, call_order: list[str], posting: JobPostingState) -> None:
        self._call_order = call_order
        self._posting = posting

    def find_next_posting(self, resume_profile: ResumeProfileState) -> JobPostingState | None:
        self._call_order.append("job_scout")
        return self._posting


class RecordingResumeMatchStage:
    def __init__(self, call_order: list[str], result: MatchResultState) -> None:
        self._call_order = call_order
        self._result = result

    def match(self, resume_profile, job_posting) -> MatchResultState:
        self._call_order.append("resume_match")
        assert job_posting is not None
        return self._result


class RecordingContactFinderStage:
    def __init__(self, call_order: list[str], contact: ContactState) -> None:
        self._call_order = call_order
        self._contact = contact

    def find_contact(self, job_posting) -> ContactState | None:
        self._call_order.append("contact_finder")
        assert job_posting is not None
        return self._contact


class RecordingOutreachComposerStage:
    def __init__(self, call_order: list[str], draft: OutreachDraftState) -> None:
        self._call_order = call_order
        self._draft = draft

    def compose(self, resume_profile, job_posting, contact, match_result) -> OutreachDraftState:
        self._call_order.append("outreach_composer")
        assert contact is not None
        assert match_result is not None
        return self._draft


class RecordingOutreachValidatorStage:
    def __init__(self, call_order: list[str], result: ValidationResultState) -> None:
        self._call_order = call_order
        self._result = result

    def validate(self, resume_profile, job_posting, contact, draft) -> ValidationResultState:
        self._call_order.append("outreach_validator")
        assert draft is not None
        return self._result


# --- Fixtures ----------------------------------------------------------


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
def resume_profile() -> ResumeProfileState:
    return ResumeProfileState(
        resume_id="resume-1",
        file_hash="a" * 64,
        full_name="Jane Doe",
        skills=["Python", "SQL"],
        target_roles=["Backend Engineer"],
        years_of_experience=4.0,
    )


@pytest.fixture
def job_posting() -> JobPostingState:
    return JobPostingState(
        job_posting_id="job-1",
        company_name="Acme Inc",
        role_title="Senior Backend Engineer",
        jd_url="https://boards.greenhouse.io/acme/jobs/1",
        jd_snapshot_text="We are hiring a Senior Backend Engineer to join our team.",
        jd_hash="b" * 64,
    )


@pytest.fixture
def matching_result() -> MatchResultState:
    return MatchResultState(similarity_score=0.88, meets_threshold=True)


@pytest.fixture
def resolved_contact() -> ContactState:
    return ContactState(
        contact_id="contact-1",
        name="John Smith",
        title="Engineering Recruiter",
        email="john.smith@acme.com",
        confidence_level="high",
    )


@pytest.fixture
def composed_draft() -> OutreachDraftState:
    return OutreachDraftState(
        subject="Interest in the Senior Backend Engineer role",
        greeting="Hi John,",
        body="I'm reaching out about the Senior Backend Engineer role at Acme Inc.",
        closing="Best regards,",
    )


@pytest.fixture
def passing_validation() -> ValidationResultState:
    return ValidationResultState(passed=True, issues=[])


@pytest.fixture
def call_order() -> list[str]:
    return []


@pytest.fixture
def wired_agents(
    call_order,
    job_posting,
    matching_result,
    resolved_contact,
    composed_draft,
    passing_validation,
) -> GraphAgents:
    return GraphAgents(
        job_scout=RecordingJobScoutStage(call_order, job_posting),
        resume_match=RecordingResumeMatchStage(call_order, matching_result),
        contact_finder=RecordingContactFinderStage(call_order, resolved_contact),
        outreach_composer=RecordingOutreachComposerStage(call_order, composed_draft),
        outreach_validator=RecordingOutreachValidatorStage(call_order, passing_validation),
    )


def _initial_state(resume_profile: ResumeProfileState) -> GraphState:
    state = GraphState()
    state.resume_profile = resume_profile
    return state


# --- Tests ---------------------------------------------------------------


class TestFullWorkflowExecutionOrder:
    def test_stages_execute_in_the_correct_order(
        self, wired_agents: GraphAgents, resume_profile: ResumeProfileState, call_order: list[str]
    ) -> None:
        graph = build_state_graph(wired_agents)

        graph.invoke(_initial_state(resume_profile))

        assert call_order == [
            "job_scout",
            "resume_match",
            "contact_finder",
            "outreach_composer",
            "outreach_validator",
        ]

    def test_all_five_stages_run_exactly_once(
        self, wired_agents: GraphAgents, resume_profile: ResumeProfileState, call_order: list[str]
    ) -> None:
        graph = build_state_graph(wired_agents)

        graph.invoke(_initial_state(resume_profile))

        assert len(call_order) == 5
        assert len(set(call_order)) == 5


class TestFullWorkflowStateTransitions:
    def test_final_state_accumulates_output_from_every_stage(
        self,
        wired_agents: GraphAgents,
        resume_profile: ResumeProfileState,
        job_posting: JobPostingState,
        matching_result: MatchResultState,
        resolved_contact: ContactState,
        composed_draft: OutreachDraftState,
    ) -> None:
        graph = build_state_graph(wired_agents)

        result_dict = graph.invoke(_initial_state(resume_profile))
        final_state = GraphState.model_validate(result_dict)

        assert final_state.resume_profile.full_name == resume_profile.full_name
        assert final_state.job_posting.company_name == job_posting.company_name
        assert final_state.match_result.similarity_score == matching_result.similarity_score
        assert final_state.contact.email == resolved_contact.email
        assert final_state.outreach_draft.subject == composed_draft.subject
        assert final_state.validation_result.passed is True

    def test_run_ends_in_awaiting_human_review_never_sent(
        self, wired_agents: GraphAgents, resume_profile: ResumeProfileState
    ) -> None:
        # Per PRD §6.4 / §13.2: the workflow must never reach a "sent"
        # state on its own -- it always stops at human review.
        graph = build_state_graph(wired_agents)

        result_dict = graph.invoke(_initial_state(resume_profile))
        final_state = GraphState.model_validate(result_dict)

        assert final_state.run_metadata.status == RunStatus.AWAITING_HUMAN_REVIEW
        assert final_state.run_metadata.current_stage == PipelineStage.AWAITING_APPROVAL

    def test_no_errors_recorded_on_successful_run(
        self, wired_agents: GraphAgents, resume_profile: ResumeProfileState
    ) -> None:
        graph = build_state_graph(wired_agents)

        result_dict = graph.invoke(_initial_state(resume_profile))
        final_state = GraphState.model_validate(result_dict)

        assert final_state.errors == []

    def test_run_id_is_stable_across_the_whole_run(
        self, wired_agents: GraphAgents, resume_profile: ResumeProfileState
    ) -> None:
        graph = build_state_graph(wired_agents)
        initial_state = _initial_state(resume_profile)
        original_run_id = initial_state.run_metadata.run_id

        result_dict = graph.invoke(initial_state)
        final_state = GraphState.model_validate(result_dict)

        assert final_state.run_metadata.run_id == original_run_id


class TestFullWorkflowCheckpointing:
    def test_final_state_can_be_checkpointed_and_reloaded(
        self,
        wired_agents: GraphAgents,
        resume_profile: ResumeProfileState,
        checkpointer: GraphCheckpointer,
    ) -> None:
        graph = build_state_graph(wired_agents)

        result_dict = graph.invoke(_initial_state(resume_profile))
        final_state = GraphState.model_validate(result_dict)

        checkpointer.save_checkpoint(final_state)
        reloaded = checkpointer.load_checkpoint(final_state.run_metadata.run_id)

        assert reloaded is not None
        assert reloaded.run_metadata.run_id == final_state.run_metadata.run_id
        assert reloaded.run_metadata.status == RunStatus.AWAITING_HUMAN_REVIEW
        assert reloaded.job_posting.company_name == final_state.job_posting.company_name
        assert reloaded.contact.email == final_state.contact.email
        assert reloaded.outreach_draft.subject == final_state.outreach_draft.subject

    def test_intermediate_checkpoint_can_be_saved_mid_run_and_survives_reload(
        self, resume_profile: ResumeProfileState, checkpointer: GraphCheckpointer
    ) -> None:
        # Simulates checkpointing after only the Job Scout + Resume Match
        # stages have run (e.g. a run interrupted before Contact Finder),
        # verifying partial state round-trips correctly.
        state = _initial_state(resume_profile)
        state.advance_to(PipelineStage.JOB_DISCOVERY)
        state.job_posting = JobPostingState(
            job_posting_id="job-1",
            company_name="Acme Inc",
            role_title="Senior Backend Engineer",
            jd_url="https://boards.greenhouse.io/acme/jobs/1",
            jd_snapshot_text="We are hiring a Senior Backend Engineer.",
            jd_hash="b" * 64,
        )
        state.advance_to(PipelineStage.RESUME_MATCH)
        state.match_result = MatchResultState(similarity_score=0.9, meets_threshold=True)

        checkpointer.save_checkpoint(state)
        reloaded = checkpointer.load_checkpoint(state.run_metadata.run_id)

        assert reloaded is not None
        assert reloaded.run_metadata.current_stage == PipelineStage.RESUME_MATCH
        assert reloaded.contact is None
        assert reloaded.outreach_draft is None


class TestFullWorkflowResume:
    def test_resumed_checkpoint_can_be_fed_back_into_the_graph(
        self,
        wired_agents: GraphAgents,
        resume_profile: ResumeProfileState,
        checkpointer: GraphCheckpointer,
    ) -> None:
        # Simulates a run that was checkpointed while still "running"
        # (e.g. process restarted between Job Scout and Resume Match),
        # then resumed and driven through the remaining stages by
        # re-invoking the graph with the reloaded state.
        partial_state = _initial_state(resume_profile)
        partial_state.job_posting = JobPostingState(
            job_posting_id="job-1",
            company_name="Acme Inc",
            role_title="Senior Backend Engineer",
            jd_url="https://boards.greenhouse.io/acme/jobs/1",
            jd_snapshot_text="We are hiring a Senior Backend Engineer.",
            jd_hash="b" * 64,
        )
        checkpointer.save_checkpoint(partial_state)

        resumed_state = checkpointer.resume(partial_state.run_metadata.run_id)
        assert resumed_state.run_metadata.status == RunStatus.RUNNING

        graph = build_state_graph(wired_agents)
        result_dict = graph.invoke(resumed_state)
        final_state = GraphState.model_validate(result_dict)

        assert final_state.run_metadata.run_id == partial_state.run_metadata.run_id
        assert final_state.run_metadata.status == RunStatus.AWAITING_HUMAN_REVIEW
        assert final_state.contact is not None
        assert final_state.outreach_draft is not None

        checkpointer.save_checkpoint(final_state)
        reloaded_final = checkpointer.load_checkpoint(final_state.run_metadata.run_id)
        assert reloaded_final.run_metadata.status == RunStatus.AWAITING_HUMAN_REVIEW

    def test_resume_raises_not_resumable_after_workflow_completes(
        self,
        wired_agents: GraphAgents,
        resume_profile: ResumeProfileState,
        checkpointer: GraphCheckpointer,
    ) -> None:
        from app.graph.checkpointer import CheckpointNotResumableError

        graph = build_state_graph(wired_agents)
        result_dict = graph.invoke(_initial_state(resume_profile))
        final_state = GraphState.model_validate(result_dict)
        final_state.mark_completed()
        checkpointer.save_checkpoint(final_state)

        with pytest.raises(CheckpointNotResumableError):
            checkpointer.resume(final_state.run_metadata.run_id)


class TestFullWorkflowFailureDuringIntegration:
    def test_stage_failure_stops_execution_and_is_recorded_on_state(
        self, resume_profile: ResumeProfileState, job_posting: JobPostingState, call_order: list[str]
    ) -> None:
        class FailingContactFinderStage:
            def find_contact(self, job_posting):
                call_order.append("contact_finder")
                raise RuntimeError("domain resolution failed")

        agents = GraphAgents(
            job_scout=RecordingJobScoutStage(call_order, job_posting),
            resume_match=RecordingResumeMatchStage(
                call_order, MatchResultState(similarity_score=0.8, meets_threshold=True)
            ),
            contact_finder=FailingContactFinderStage(),
            outreach_composer=RecordingOutreachComposerStage(
                call_order,
                OutreachDraftState(subject="s", greeting="g", body="b", closing="c"),
            ),
            outreach_validator=RecordingOutreachValidatorStage(
                call_order, ValidationResultState(passed=True, issues=[])
            ),
        )
        graph = build_state_graph(agents)

        result_dict = graph.invoke(_initial_state(resume_profile))
        final_state = GraphState.model_validate(result_dict)

        assert call_order == ["job_scout", "resume_match", "contact_finder"]
        assert final_state.run_metadata.status == RunStatus.FAILED
        assert len(final_state.errors) == 1
        assert final_state.errors[0].stage == PipelineStage.CONTACT_FINDING
        assert final_state.outreach_draft is None