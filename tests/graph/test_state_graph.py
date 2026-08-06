"""
Unit tests for the LangGraph pipeline state graph.

Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 2 -
State Graph Definition, Task 1.

Uses fake stage implementations satisfying `JobScoutStage`,
`ResumeMatchStage`, `ContactFinderStage`, `OutreachComposerStage`, and
`OutreachValidatorStage` (per docs/coding_guidelines.md §6 — no test hits a
real agent, DB, or LLM). Covers: the full happy path to awaiting human
review, early termination when no posting is found, early termination below
the match threshold, early termination when no contact is resolved, and
failure isolation when a stage raises.
"""

from __future__ import annotations

import pytest

from app.graph.state_graph import GraphAgents, GraphOrchestrationError, build_state_graph
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


class FakeJobScoutStage:
    def __init__(self, posting: JobPostingState | None, raise_error: Exception | None = None) -> None:
        self._posting = posting
        self._raise_error = raise_error
        self.called = False

    def find_next_posting(self, resume_profile: ResumeProfileState) -> JobPostingState | None:
        self.called = True
        if self._raise_error:
            raise self._raise_error
        return self._posting


class FakeResumeMatchStage:
    def __init__(self, result: MatchResultState | None, raise_error: Exception | None = None) -> None:
        self._result = result
        self._raise_error = raise_error
        self.called = False

    def match(self, resume_profile, job_posting) -> MatchResultState:
        self.called = True
        if self._raise_error:
            raise self._raise_error
        return self._result


class FakeContactFinderStage:
    def __init__(self, contact: ContactState | None, raise_error: Exception | None = None) -> None:
        self._contact = contact
        self._raise_error = raise_error
        self.called = False

    def find_contact(self, job_posting) -> ContactState | None:
        self.called = True
        if self._raise_error:
            raise self._raise_error
        return self._contact


class FakeOutreachComposerStage:
    def __init__(self, draft: OutreachDraftState | None, raise_error: Exception | None = None) -> None:
        self._draft = draft
        self._raise_error = raise_error
        self.called = False

    def compose(self, resume_profile, job_posting, contact, match_result) -> OutreachDraftState:
        self.called = True
        if self._raise_error:
            raise self._raise_error
        return self._draft


class FakeOutreachValidatorStage:
    def __init__(self, result: ValidationResultState | None, raise_error: Exception | None = None) -> None:
        self._result = result
        self._raise_error = raise_error
        self.called = False

    def validate(self, resume_profile, job_posting, contact, draft) -> ValidationResultState:
        self.called = True
        if self._raise_error:
            raise self._raise_error
        return self._result


@pytest.fixture
def resume_profile() -> ResumeProfileState:
    return ResumeProfileState(
        resume_id="resume-1",
        file_hash="a" * 64,
        full_name="Jane Doe",
        skills=["Python"],
        target_roles=["Backend Engineer"],
    )


@pytest.fixture
def job_posting() -> JobPostingState:
    return JobPostingState(
        job_posting_id="job-1",
        company_name="Acme Inc",
        role_title="Senior Backend Engineer",
        jd_url="https://boards.greenhouse.io/acme/jobs/1",
        jd_snapshot_text="We are hiring a Senior Backend Engineer.",
        jd_hash="b" * 64,
    )


@pytest.fixture
def matching_result() -> MatchResultState:
    return MatchResultState(similarity_score=0.85, meets_threshold=True)


@pytest.fixture
def below_threshold_result() -> MatchResultState:
    return MatchResultState(similarity_score=0.20, meets_threshold=False)


@pytest.fixture
def resolved_contact() -> ContactState:
    return ContactState(
        contact_id="contact-1",
        name="John Smith",
        title="Recruiter",
        email="john@acme.com",
        confidence_level="high",
    )


@pytest.fixture
def composed_draft() -> OutreachDraftState:
    return OutreachDraftState(
        subject="Interest in the role",
        greeting="Hi John,",
        body="Reaching out about the Senior Backend Engineer role.",
        closing="Best regards,",
    )


@pytest.fixture
def passing_validation() -> ValidationResultState:
    return ValidationResultState(passed=True, issues=[])


def _initial_state(resume_profile: ResumeProfileState) -> GraphState:
    state = GraphState()
    state.resume_profile = resume_profile
    return state


class TestHappyPath:
    def test_full_run_reaches_awaiting_human_review(
        self,
        resume_profile,
        job_posting,
        matching_result,
        resolved_contact,
        composed_draft,
        passing_validation,
    ) -> None:
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(matching_result),
            contact_finder=FakeContactFinderStage(resolved_contact),
            outreach_composer=FakeOutreachComposerStage(composed_draft),
            outreach_validator=FakeOutreachValidatorStage(passing_validation),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.AWAITING_HUMAN_REVIEW
        assert result["job_posting"].company_name == "Acme Inc"
        assert result["match_result"].meets_threshold is True
        assert result["contact"].email == "john@acme.com"
        assert result["outreach_draft"].subject == "Interest in the role"
        assert result["validation_result"].passed is True
        assert result["errors"] == []

    def test_all_stages_invoked_on_happy_path(
        self,
        resume_profile,
        job_posting,
        matching_result,
        resolved_contact,
        composed_draft,
        passing_validation,
    ) -> None:
        job_scout = FakeJobScoutStage(job_posting)
        resume_match = FakeResumeMatchStage(matching_result)
        contact_finder = FakeContactFinderStage(resolved_contact)
        composer = FakeOutreachComposerStage(composed_draft)
        validator = FakeOutreachValidatorStage(passing_validation)

        agents = GraphAgents(
            job_scout=job_scout,
            resume_match=resume_match,
            contact_finder=contact_finder,
            outreach_composer=composer,
            outreach_validator=validator,
        )
        graph = build_state_graph(agents)

        graph.invoke(_initial_state(resume_profile))

        assert job_scout.called is True
        assert resume_match.called is True
        assert contact_finder.called is True
        assert composer.called is True
        assert validator.called is True


class TestEarlyTermination:
    def test_ends_when_no_posting_found(self, resume_profile) -> None:
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(None),
            resume_match=FakeResumeMatchStage(None),
            contact_finder=FakeContactFinderStage(None),
            outreach_composer=FakeOutreachComposerStage(None),
            outreach_validator=FakeOutreachValidatorStage(None),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.COMPLETED
        assert result["job_posting"] is None
        assert agents.resume_match.called is False

    def test_ends_when_match_below_threshold(
        self, resume_profile, job_posting, below_threshold_result
    ) -> None:
        contact_finder = FakeContactFinderStage(None)
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(below_threshold_result),
            contact_finder=contact_finder,
            outreach_composer=FakeOutreachComposerStage(None),
            outreach_validator=FakeOutreachValidatorStage(None),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.COMPLETED
        assert result["match_result"].meets_threshold is False
        assert contact_finder.called is False

    def test_ends_when_no_contact_resolved(
        self, resume_profile, job_posting, matching_result
    ) -> None:
        composer = FakeOutreachComposerStage(None)
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(matching_result),
            contact_finder=FakeContactFinderStage(None),
            outreach_composer=composer,
            outreach_validator=FakeOutreachValidatorStage(None),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.COMPLETED
        assert result["contact"] is None
        assert composer.called is False

    def test_ends_when_contact_has_no_email(
        self, resume_profile, job_posting, matching_result
    ) -> None:
        contact_without_email = ContactState(
            contact_id="contact-1", name="John Smith", email=None, confidence_level="low"
        )
        composer = FakeOutreachComposerStage(None)
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(matching_result),
            contact_finder=FakeContactFinderStage(contact_without_email),
            outreach_composer=composer,
            outreach_validator=FakeOutreachValidatorStage(None),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.COMPLETED
        assert composer.called is False


class TestFailureIsolation:
    def test_job_scout_failure_marks_run_failed_and_stops(self, resume_profile) -> None:
        resume_match = FakeResumeMatchStage(None)
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(None, raise_error=RuntimeError("connector down")),
            resume_match=resume_match,
            contact_finder=FakeContactFinderStage(None),
            outreach_composer=FakeOutreachComposerStage(None),
            outreach_validator=FakeOutreachValidatorStage(None),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.FAILED
        assert len(result["errors"]) == 1
        assert result["errors"][0].stage == PipelineStage.JOB_DISCOVERY
        assert "connector down" in result["errors"][0].message
        assert resume_match.called is False

    def test_contact_finder_failure_marks_run_failed_and_stops(
        self, resume_profile, job_posting, matching_result
    ) -> None:
        composer = FakeOutreachComposerStage(None)
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(matching_result),
            contact_finder=FakeContactFinderStage(
                None, raise_error=RuntimeError("domain resolution failed")
            ),
            outreach_composer=composer,
            outreach_validator=FakeOutreachValidatorStage(None),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.FAILED
        assert result["errors"][0].stage == PipelineStage.CONTACT_FINDING
        assert composer.called is False

    def test_outreach_validator_failure_marks_run_failed(
        self,
        resume_profile,
        job_posting,
        matching_result,
        resolved_contact,
        composed_draft,
    ) -> None:
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(matching_result),
            contact_finder=FakeContactFinderStage(resolved_contact),
            outreach_composer=FakeOutreachComposerStage(composed_draft),
            outreach_validator=FakeOutreachValidatorStage(
                None, raise_error=RuntimeError("LLM timeout")
            ),
        )
        graph = build_state_graph(agents)

        result = graph.invoke(_initial_state(resume_profile))

        assert result["run_metadata"].status == RunStatus.FAILED
        assert result["errors"][0].stage == PipelineStage.OUTREACH_VALIDATION
        assert result["validation_result"] is None


class TestGraphConstruction:
    def test_build_state_graph_returns_invocable_graph(
        self, resume_profile, job_posting, matching_result, resolved_contact, composed_draft, passing_validation
    ) -> None:
        agents = GraphAgents(
            job_scout=FakeJobScoutStage(job_posting),
            resume_match=FakeResumeMatchStage(matching_result),
            contact_finder=FakeContactFinderStage(resolved_contact),
            outreach_composer=FakeOutreachComposerStage(composed_draft),
            outreach_validator=FakeOutreachValidatorStage(passing_validation),
        )

        graph = build_state_graph(agents)

        assert hasattr(graph, "invoke")
        