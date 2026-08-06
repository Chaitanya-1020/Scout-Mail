"""
LangGraph state graph — pipeline orchestration.

Implements: PRD §5 (Multi-Agent Orchestration — Job Scout, Resume Match,
Contact Finder, and Outreach Composer + Validator agents composed into a
single resume-to-outreach workflow), §6.4 / §13.2 (no outreach email is ever
sent without human approval — the graph always terminates at an
awaiting-approval state, never at "sent"), §12 (Risks — a single stage's
failure must not silently corrupt or hide the run; it is recorded and the
run stops cleanly).
Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 2 -
State Graph Definition, Task 1.

Defines the LangGraph `StateGraph` over `GraphState`
(app/graph/state_schema.py), with one node per pipeline stage: Job Scout,
Resume Match, Contact Finder, Outreach Composer, Outreach Validator. Per
docs/architecture.md §3, this is the only module aware of all these stages
simultaneously; each node is a thin adapter that extracts plain inputs from
`GraphState`, delegates to an injected stage interface (business logic lives
in `app/agents/*`, not here), and writes the result back onto the state.
Stage interfaces are defined as `Protocol`s here rather than importing
concrete agent classes directly, so this module has no dependency on
agent-internal wiring (LLM providers, DB sessions, vector stores) — concrete
adapters implementing these protocols are constructed at the composition
root (`app/graph/runner.py`, a later task) and injected via `GraphAgents`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from langgraph.graph import END, StateGraph

from app.graph.state_schema import (
    ContactState,
    GraphState,
    JobPostingState,
    MatchResultState,
    OutreachDraftState,
    PipelineStage,
    ResumeProfileState,
    ValidationResultState,
)

logger = logging.getLogger(__name__)

_NODE_JOB_SCOUT = "job_scout"
_NODE_RESUME_MATCH = "resume_match"
_NODE_CONTACT_FINDER = "contact_finder"
_NODE_OUTREACH_COMPOSER = "outreach_composer"
_NODE_OUTREACH_VALIDATOR = "outreach_validator"


class GraphOrchestrationError(Exception):
    """Raised when the graph itself cannot be constructed or invoked.

    Per-stage failures during a run are captured on `GraphState.errors`
    instead of raised, so a run can terminate cleanly and be inspected —
    this exception is reserved for graph construction/wiring problems.
    """


# --- Stage interfaces (business logic lives in the concrete implementations) ---
#
# Each Protocol exposes exactly the operation this graph needs from that
# agent, translated to plain inputs/outputs rather than agent-internal
# types (Interface Segregation, per docs/architecture.md). Concrete
# adapters wrapping app/agents/job_scout/agent.py,
# app/agents/resume_match/agent.py, app/agents/contact_finder/agent.py,
# app/agents/outreach/composer_agent.py, and
# app/agents/outreach/validator_agent.py implement these protocols and are
# supplied via `GraphAgents` at the composition root.


class JobScoutStage(Protocol):
    """Finds the next candidate job posting for a resume to be evaluated
    against."""

    def find_next_posting(
        self, resume_profile: ResumeProfileState
    ) -> JobPostingState | None:
        """Return the next candidate posting, or None if none are available."""
        ...


class ResumeMatchStage(Protocol):
    """Scores a resume against a job posting."""

    def match(
        self, resume_profile: ResumeProfileState, job_posting: JobPostingState
    ) -> MatchResultState:
        """Return the similarity result for this resume/posting pair."""
        ...


class ContactFinderStage(Protocol):
    """Resolves a hiring contact for a job posting."""

    def find_contact(self, job_posting: JobPostingState) -> ContactState | None:
        """Return the best-resolved contact, or None if none could be found."""
        ...


class OutreachComposerStage(Protocol):
    """Drafts a personalized outreach email."""

    def compose(
        self,
        resume_profile: ResumeProfileState,
        job_posting: JobPostingState,
        contact: ContactState,
        match_result: MatchResultState,
    ) -> OutreachDraftState:
        """Return a drafted outreach email for the given context."""
        ...


class OutreachValidatorStage(Protocol):
    """Validates a drafted outreach email against its source context."""

    def validate(
        self,
        resume_profile: ResumeProfileState,
        job_posting: JobPostingState,
        contact: ContactState,
        draft: OutreachDraftState,
    ) -> ValidationResultState:
        """Return the validation outcome for the given draft."""
        ...


@dataclass(frozen=True)
class GraphAgents:
    """Container of injected stage implementations.

    The graph never constructs a concrete agent itself — every stage
    implementation is supplied by the caller (Dependency Inversion, per
    docs/architecture.md), keeping this module free of any dependency on
    `app/db`, `app/llm`, `app/connectors`, or `app/vectorstore`.
    """

    job_scout: JobScoutStage
    resume_match: ResumeMatchStage
    contact_finder: ContactFinderStage
    outreach_composer: OutreachComposerStage
    outreach_validator: OutreachValidatorStage


def _job_scout_node(agents: GraphAgents):
    def node(state: GraphState) -> GraphState:
        state.advance_to(PipelineStage.JOB_DISCOVERY)

        if state.resume_profile is None:
            state.record_error(
                PipelineStage.JOB_DISCOVERY,
                "No resume profile present on state; cannot search for postings.",
            )
            state.mark_failed()
            return state

        try:
            posting = agents.job_scout.find_next_posting(state.resume_profile)
        except Exception as exc:  # noqa: BLE001 - stage boundary, per PRD §12
            state.record_error(PipelineStage.JOB_DISCOVERY, str(exc))
            state.mark_failed()
            return state

        if posting is None:
            logger.info(
                "Job Scout stage found no candidate posting",
                extra={"run_id": state.run_metadata.run_id},
            )
            state.mark_completed()
            return state

        state.job_posting = posting
        return state

    return node


def _resume_match_node(agents: GraphAgents):
    def node(state: GraphState) -> GraphState:
        state.advance_to(PipelineStage.RESUME_MATCH)

        try:
            match_result = agents.resume_match.match(
                state.resume_profile, state.job_posting
            )
        except Exception as exc:  # noqa: BLE001 - stage boundary, per PRD §12
            state.record_error(PipelineStage.RESUME_MATCH, str(exc))
            state.mark_failed()
            return state

        state.match_result = match_result
        return state

    return node


def _contact_finder_node(agents: GraphAgents):
    def node(state: GraphState) -> GraphState:
        state.advance_to(PipelineStage.CONTACT_FINDING)

        try:
            contact = agents.contact_finder.find_contact(state.job_posting)
        except Exception as exc:  # noqa: BLE001 - stage boundary, per PRD §12
            state.record_error(PipelineStage.CONTACT_FINDING, str(exc))
            state.mark_failed()
            return state

        if contact is None or not contact.email:
            logger.info(
                "Contact Finder stage resolved no usable contact",
                extra={"run_id": state.run_metadata.run_id},
            )
            state.mark_completed()
            return state

        state.contact = contact
        return state

    return node


def _outreach_composer_node(agents: GraphAgents):
    def node(state: GraphState) -> GraphState:
        state.advance_to(PipelineStage.OUTREACH_COMPOSITION)

        try:
            draft = agents.outreach_composer.compose(
                resume_profile=state.resume_profile,
                job_posting=state.job_posting,
                contact=state.contact,
                match_result=state.match_result,
            )
        except Exception as exc:  # noqa: BLE001 - stage boundary, per PRD §12
            state.record_error(PipelineStage.OUTREACH_COMPOSITION, str(exc))
            state.mark_failed()
            return state

        state.outreach_draft = draft
        return state

    return node


def _outreach_validator_node(agents: GraphAgents):
    def node(state: GraphState) -> GraphState:
        state.advance_to(PipelineStage.OUTREACH_VALIDATION)

        try:
            validation_result = agents.outreach_validator.validate(
                resume_profile=state.resume_profile,
                job_posting=state.job_posting,
                contact=state.contact,
                draft=state.outreach_draft,
            )
        except Exception as exc:  # noqa: BLE001 - stage boundary, per PRD §12
            state.record_error(PipelineStage.OUTREACH_VALIDATION, str(exc))
            state.mark_failed()
            return state

        state.validation_result = validation_result

        # Every draft — whether validation passed or flagged issues — stops
        # here awaiting human review. Validation outcome informs the
        # reviewer; it never authorizes sending on its own (PRD §6.4, §13.2).
        state.mark_awaiting_human_review()
        return state

    return node


def _route_after_job_scout(state: GraphState) -> str:
    if state.run_metadata.status.value == "failed" or state.job_posting is None:
        return END
    return _NODE_RESUME_MATCH


def _route_after_resume_match(state: GraphState) -> str:
    if state.run_metadata.status.value == "failed":
        return END
    if state.match_result is None or not state.match_result.meets_threshold:
        logger.info(
            "Resume match below threshold; ending run without contact search",
            extra={"run_id": state.run_metadata.run_id},
        )
        state.mark_completed()
        return END
    return _NODE_CONTACT_FINDER


def _route_after_contact_finder(state: GraphState) -> str:
    if state.run_metadata.status.value == "failed" or state.contact is None:
        return END
    return _NODE_OUTREACH_COMPOSER


def _route_after_outreach_composer(state: GraphState) -> str:
    if state.run_metadata.status.value == "failed" or state.outreach_draft is None:
        return END
    return _NODE_OUTREACH_VALIDATOR


def build_state_graph(agents: GraphAgents) -> StateGraph:
    """Construct the compiled LangGraph pipeline.

    Wiring:
        START -> job_scout -> [posting found?] -> resume_match
        resume_match -> [meets threshold?] -> contact_finder
        contact_finder -> [contact resolved?] -> outreach_composer
        outreach_composer -> [draft composed?] -> outreach_validator
        outreach_validator -> END (always ends awaiting human review)

    Any node raising during its stage records the error on `GraphState` and
    marks the run failed; every conditional edge checks for failure first
    and routes straight to END, so a failed run never continues to a later
    stage (PRD §12 — failure isolation).

    Args:
        agents: injected stage implementations (Dependency Inversion).

    Returns:
        A compiled LangGraph graph ready to `.invoke(GraphState(...))`.

    Raises:
        GraphOrchestrationError: if the graph cannot be compiled.
    """
    try:
        builder = StateGraph(GraphState)

        builder.add_node(_NODE_JOB_SCOUT, _job_scout_node(agents))
        builder.add_node(_NODE_RESUME_MATCH, _resume_match_node(agents))
        builder.add_node(_NODE_CONTACT_FINDER, _contact_finder_node(agents))
        builder.add_node(_NODE_OUTREACH_COMPOSER, _outreach_composer_node(agents))
        builder.add_node(_NODE_OUTREACH_VALIDATOR, _outreach_validator_node(agents))

        builder.set_entry_point(_NODE_JOB_SCOUT)

        builder.add_conditional_edges(
            _NODE_JOB_SCOUT,
            _route_after_job_scout,
            {_NODE_RESUME_MATCH: _NODE_RESUME_MATCH, END: END},
        )
        builder.add_conditional_edges(
            _NODE_RESUME_MATCH,
            _route_after_resume_match,
            {_NODE_CONTACT_FINDER: _NODE_CONTACT_FINDER, END: END},
        )
        builder.add_conditional_edges(
            _NODE_CONTACT_FINDER,
            _route_after_contact_finder,
            {_NODE_OUTREACH_COMPOSER: _NODE_OUTREACH_COMPOSER, END: END},
        )
        builder.add_conditional_edges(
            _NODE_OUTREACH_COMPOSER,
            _route_after_outreach_composer,
            {_NODE_OUTREACH_VALIDATOR: _NODE_OUTREACH_VALIDATOR, END: END},
        )
        builder.add_edge(_NODE_OUTREACH_VALIDATOR, END)

        return builder.compile()
    except Exception as exc:  # noqa: BLE001 - normalize graph construction failure
        raise GraphOrchestrationError("Failed to construct the pipeline state graph.") from exc