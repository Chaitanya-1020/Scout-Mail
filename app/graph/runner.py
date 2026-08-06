"""
Pipeline runner — application composition root.

Implements: PRD §5 (Multi-Agent Orchestration — end-to-end resume-to-outreach
pipeline), §8.3 (Job Scout Agent pulls new postings on a schedule), §9
(Auditability — pipeline runs are checkpointed and resumable), §12 (Risks —
graceful degradation: the process must shut down cleanly without corrupting
in-flight state).
Roadmap: Epic 7 - LangGraph Orchestration & State Persistence, Story 4 -
Workflow Runner, Task 1.

This is the single module that constructs every concrete infrastructure
class (`OllamaClient`, `ChromaVectorStore`, `PostgresCache`, connector
instances, SQLAlchemy sessions) and wires them into the agents and the
compiled LangGraph pipeline (`app/graph/state_graph.py`). Per
docs/architecture.md, no other module — not `app/agents/*`, not
`app/graph/state_graph.py` itself — constructs concrete infrastructure
directly; this file, alongside `app/main.py`, is the composition root.

Adapters in this module translate between the plain, agent-facing types used
by `app/agents/*` and the graph-facing `*State` types defined in
`app/graph/state_schema.py`, satisfying the `Protocol`s declared in
`app/graph/state_graph.py` (`JobScoutStage`, `ResumeMatchStage`,
`ContactFinderStage`, `OutreachComposerStage`, `OutreachValidatorStage`).

Known limitation: `ResumeProfileState` (app/graph/state_schema.py) carries a
reduced summary of a resume (name, skills, target roles, years of
experience) without per-entry work/education history, since that schema was
defined in an earlier task and is not regenerated here. The
`OutreachComposerStageAdapter` below reconstructs a best-effort
`ResumeProfile` from the fields available on `ResumeProfileState`, with
empty `experience`/`education` lists. A future task extending
`ResumeProfileState` with full experience/education entries would let this
adapter produce fuller composer prompts without further changes to this
runner's overall structure.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType

from sqlalchemy.orm import Session

from app.agents.contact_finder.agent import ContactFinderAgent
from app.agents.contact_finder.domain_resolver import PublicDomainResolver
from app.agents.contact_finder.hunter_client import HunterClient, HunterQuotaTracker
from app.agents.contact_finder.pattern_generator import EmailPatternGenerator
from app.agents.contact_finder.profile_lookup import ProfileLookupAgent
from app.agents.contact_finder.public_source_scraper import PublicSourcePageDiscoverer
from app.agents.contact_finder.repository import ContactFinderRepository
from app.agents.contact_finder.smtp_validator import SmtpMailboxValidator
from app.agents.job_scout.agent import JobScoutAgent
from app.agents.job_scout.repository import JobScoutRepository, compute_jd_hash
from app.agents.outreach.composer_agent import OutreachComposerAgent, OutreachComposerError
from app.agents.outreach.context_builder import (
    ContactContext,
    JobContext,
    OutreachContextBuilder,
    OutreachContextBuildError,
)
from app.agents.outreach.repository import OutreachRepository
from app.agents.outreach.validator_agent import (
    OutreachValidationError,
    OutreachValidatorAgent,
)
from app.agents.resume_match.agent import CandidateJobPosting, ResumeMatchAgent
from app.agents.resume_match.jd_embedder import JdEmbedder
from app.agents.resume_match.repository import ResumeMatchRepository
from app.agents.resume_match.scorer import ResumeMatchScorer
from app.agents.resume_parser.embedder import OllamaEmbeddingProvider
from app.agents.resume_parser.parser_agent import EducationEntry, ExperienceEntry, ResumeProfile
from app.config import Settings, get_settings
from app.connectors.registry import build_default_registry
from app.db.models import Resume
from app.db.session import session_scope
from app.graph.checkpointer import GraphCheckpointer
from app.graph.state_graph import GraphAgents, build_state_graph
from app.graph.state_schema import (
    ContactState,
    GraphState,
    JobPostingState,
    MatchResultState,
    OutreachDraftState,
    ResumeProfileState,
    ValidationResultState,
)
from app.llm.ollama_client import OllamaClient
from app.services.cache import PostgresCache
from app.utils.confidence import ConfidenceScorer
from app.vectorstore.chroma_client import ChromaVectorStore

logger = logging.getLogger(__name__)


class PipelineRunnerError(Exception):
    """Raised when the pipeline cannot be run or resumed."""


# --- Stage adapters -----------------------------------------------------
#
# Each adapter implements exactly one `Protocol` from
# `app/graph/state_graph.py`, delegating to the real agent built in earlier
# tasks and translating between plain agent types and graph `*State` types.
# Business logic remains entirely in the wrapped agent; adapters only
# translate and, where an agent's API requires it, perform the minimal
# persistence lookup needed to bridge between calls (e.g. re-fetching a
# persisted `Contact` after `ContactFinderAgent.find_contacts` persists it
# internally).


class JobScoutStageAdapter:
    """Adapts `JobScoutAgent` to the `JobScoutStage` protocol."""

    def __init__(self, agent: JobScoutAgent, repository: JobScoutRepository) -> None:
        self._agent = agent
        self._repository = repository

    def find_next_posting(self, resume_profile: ResumeProfileState) -> JobPostingState | None:
        result = self._agent.run()

        for failure in result.failures:
            logger.warning(
                "Job Scout connector failed during pipeline run",
                extra={"source_name": failure.source_name, "error": failure.error_message},
            )

        if not result.postings:
            return None

        saved_postings = self._repository.save_postings(result.postings)
        if not saved_postings:
            return None

        first = saved_postings[0]
        return JobPostingState(
            job_posting_id=str(first.id),
            company_name=first.company_name,
            role_title=first.role_title,
            jd_url=first.jd_url,
            jd_snapshot_text=first.jd_snapshot_text,
            jd_hash=first.jd_hash,
        )


class ResumeMatchStageAdapter:
    """Adapts `ResumeMatchAgent` to the `ResumeMatchStage` protocol."""

    def __init__(self, agent: ResumeMatchAgent, resume_id: uuid.UUID, resume_file_hash: str) -> None:
        self._agent = agent
        self._resume_id = resume_id
        self._resume_file_hash = resume_file_hash

    def match(
        self, resume_profile: ResumeProfileState, job_posting: JobPostingState
    ) -> MatchResultState:
        candidate = CandidateJobPosting(
            job_posting_id=job_posting.job_posting_id,
            jd_hash=job_posting.jd_hash,
            jd_snapshot_text=job_posting.jd_snapshot_text,
        )

        scored = self._agent.match(
            resume_id=self._resume_id,
            resume_file_hash=self._resume_file_hash,
            candidate_postings=[candidate],
        )

        if scored:
            return MatchResultState(similarity_score=scored[0].similarity_score, meets_threshold=True)

        # The candidate scored below the configured threshold and was
        # filtered out by ResumeMatchScorer; its exact score is not
        # returned by ResumeMatchAgent.match in that case (see module
        # docstring). 0.0 is used as a floor value rather than
        # misrepresenting an unknown score as a specific non-zero number.
        return MatchResultState(similarity_score=0.0, meets_threshold=False)


class ContactFinderStageAdapter:
    """Adapts `ContactFinderAgent` to the `ContactFinderStage` protocol."""

    def __init__(self, agent: ContactFinderAgent, repository: ContactFinderRepository) -> None:
        self._agent = agent
        self._repository = repository

    def find_contact(self, job_posting: JobPostingState) -> ContactState | None:
        job_posting_id = uuid.UUID(job_posting.job_posting_id)

        self._agent.find_contacts(
            job_posting_id=job_posting_id,
            company_name=job_posting.company_name,
            jd_url=job_posting.jd_url,
        )

        contacts = self._repository.get_contacts_for_job_posting(job_posting_id)
        best_contact = self._select_best_contact(contacts)
        if best_contact is None:
            return None

        return ContactState(
            contact_id=str(best_contact.id),
            name=best_contact.name,
            title=best_contact.title,
            email=best_contact.email,
            confidence_level=best_contact.confidence_level.value,
        )

    def _select_best_contact(self, contacts: list) -> object | None:  # noqa: ANN401
        if not contacts:
            return None

        confidence_rank = {"high": 3, "medium": 2, "low": 1}
        contacts_with_email = [c for c in contacts if c.email]
        candidates = contacts_with_email or contacts

        return max(
            candidates,
            key=lambda c: confidence_rank.get(c.confidence_level.value, 0),
        )


class OutreachComposerStageAdapter:
    """Adapts `OutreachContextBuilder` + `OutreachComposerAgent` to the
    `OutreachComposerStage` protocol."""

    def __init__(
        self,
        context_builder: OutreachContextBuilder,
        composer_agent: OutreachComposerAgent,
    ) -> None:
        self._context_builder = context_builder
        self._composer_agent = composer_agent

    def compose(
        self,
        resume_profile: ResumeProfileState,
        job_posting: JobPostingState,
        contact: ContactState,
        match_result: MatchResultState,
    ) -> OutreachDraftState:
        full_resume_profile = ResumeProfile(
            full_name=resume_profile.full_name,
            email=resume_profile.email,
            phone=None,
            skills=list(resume_profile.skills),
            target_roles=list(resume_profile.target_roles),
            experience=list[ExperienceEntry](),
            education=list[EducationEntry](),
            years_of_experience=resume_profile.years_of_experience,
        )

        job_context = JobContext(
            company_name=job_posting.company_name,
            role_title=job_posting.role_title,
            jd_url=job_posting.jd_url,
            jd_snapshot_text=job_posting.jd_snapshot_text,
        )
        contact_context = ContactContext(
            name=contact.name or "Hiring Team",
            title=contact.title,
            email=contact.email or "",
            confidence_level=contact.confidence_level,
        )

        try:
            outreach_context = self._context_builder.build_context(
                resume_profile=full_resume_profile,
                job=job_context,
                contact=contact_context,
                match_score=match_result.similarity_score,
            )
        except OutreachContextBuildError as exc:
            raise PipelineRunnerError(f"Failed to build outreach context: {exc}") from exc

        try:
            composed = self._composer_agent.compose(outreach_context)
        except OutreachComposerError as exc:
            raise PipelineRunnerError(f"Failed to compose outreach email: {exc}") from exc

        return OutreachDraftState(
            subject=composed.subject,
            greeting=composed.greeting,
            body=composed.body,
            closing=composed.closing,
        )


class OutreachValidatorStageAdapter:
    """Adapts `OutreachContextBuilder` + `OutreachValidatorAgent` to the
    `OutreachValidatorStage` protocol."""

    def __init__(
        self,
        context_builder: OutreachContextBuilder,
        validator_agent: OutreachValidatorAgent,
    ) -> None:
        self._context_builder = context_builder
        self._validator_agent = validator_agent

    def validate(
        self,
        resume_profile: ResumeProfileState,
        job_posting: JobPostingState,
        contact: ContactState,
        draft: OutreachDraftState,
    ) -> ValidationResultState:
        from app.agents.outreach.composer_agent import ComposedEmail

        full_resume_profile = ResumeProfile(
            full_name=resume_profile.full_name,
            email=resume_profile.email,
            phone=None,
            skills=list(resume_profile.skills),
            target_roles=list(resume_profile.target_roles),
            experience=list[ExperienceEntry](),
            education=list[EducationEntry](),
            years_of_experience=resume_profile.years_of_experience,
        )

        job_context = JobContext(
            company_name=job_posting.company_name,
            role_title=job_posting.role_title,
            jd_url=job_posting.jd_url,
            jd_snapshot_text=job_posting.jd_snapshot_text,
        )
        contact_context = ContactContext(
            name=contact.name or "Hiring Team",
            title=contact.title,
            email=contact.email or "",
            confidence_level=contact.confidence_level,
        )

        try:
            outreach_context = self._context_builder.build_context(
                resume_profile=full_resume_profile,
                job=job_context,
                contact=contact_context,
                match_score=0.0,
            )
        except OutreachContextBuildError as exc:
            raise PipelineRunnerError(f"Failed to build outreach context: {exc}") from exc

        composed_email = ComposedEmail(
            subject=draft.subject, greeting=draft.greeting, body=draft.body, closing=draft.closing
        )

        try:
            result = self._validator_agent.validate(outreach_context, composed_email)
        except OutreachValidationError as exc:
            raise PipelineRunnerError(f"Failed to validate outreach email: {exc}") from exc

        from app.graph.state_schema import ValidationIssueState

        return ValidationResultState(
            passed=result.passed,
            issues=[
                ValidationIssueState(
                    category=issue.category.value,
                    severity=issue.severity.value,
                    description=issue.description,
                )
                for issue in result.issues
            ],
        )


# --- Composition root -----------------------------------------------------


@dataclass
class PipelineDependencies:
    """Every constructed dependency needed to run the pipeline for a single
    database session."""

    session: Session
    checkpointer: GraphCheckpointer
    graph: object
    job_scout_repository: JobScoutRepository


def build_pipeline_dependencies(
    session: Session, settings: Settings | None = None
) -> PipelineDependencies:
    """Construct every concrete infrastructure class and agent, and compose
    them into a runnable LangGraph pipeline.

    This is the only function in the codebase that constructs
    `OllamaClient`, `ChromaVectorStore`, `PostgresCache`, connector
    instances, and every concrete agent together — per docs/architecture.md,
    no agent or graph node constructs its own infrastructure.
    """
    settings = settings or get_settings()

    llm_provider = OllamaClient()
    vector_store = ChromaVectorStore()
    cache = PostgresCache(session)
    embedding_provider = OllamaEmbeddingProvider()

    # --- Job Scout ---
    connector_registry = build_default_registry(settings)
    job_scout_agent = JobScoutAgent(registry=connector_registry)
    job_scout_repository = JobScoutRepository(session)

    # --- Resume Match ---
    jd_embedder = JdEmbedder(
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        collection_name=settings.chroma_collection_jobs,
    )
    resume_match_scorer = ResumeMatchScorer(min_score_threshold=settings.resume_match_min_score)
    resume_match_repository = ResumeMatchRepository(session)
    resume_match_agent = ResumeMatchAgent(
        vector_store=vector_store,
        resume_collection_name=settings.chroma_collection_resumes,
        jd_embedder=jd_embedder,
        scorer=resume_match_scorer,
        repository=resume_match_repository,
    )

    # --- Contact Finder ---
    domain_resolver = PublicDomainResolver()
    page_discoverer = PublicSourcePageDiscoverer()
    profile_lookup_agent = ProfileLookupAgent(llm_provider=llm_provider)
    pattern_generator = EmailPatternGenerator()
    smtp_validator = SmtpMailboxValidator()
    hunter_quota_tracker = HunterQuotaTracker(
        cache=cache, monthly_quota=settings.hunter_io_monthly_quota
    )
    hunter_client = HunterClient(
        api_key=settings.hunter_io_api_key, quota_tracker=hunter_quota_tracker
    )
    confidence_scorer = ConfidenceScorer()
    contact_finder_repository = ContactFinderRepository(session)
    contact_finder_agent = ContactFinderAgent(
        domain_resolver=domain_resolver,
        page_discoverer=page_discoverer,
        profile_lookup_agent=profile_lookup_agent,
        pattern_generator=pattern_generator,
        smtp_validator=smtp_validator,
        hunter_client=hunter_client,
        confidence_scorer=confidence_scorer,
        repository=contact_finder_repository,
    )

    # --- Outreach ---
    context_builder = OutreachContextBuilder()
    composer_agent = OutreachComposerAgent(llm_provider=llm_provider)
    validator_agent = OutreachValidatorAgent(llm_provider=llm_provider)
    OutreachRepository(session)  # constructed for parity; persistence of the
    # final approved draft is handled by the review/approval flow (Epic 8),
    # not by this runner, which only drives the graph through to the
    # awaiting-human-review checkpoint.

    return PipelineDependencies(
        session=session,
        checkpointer=GraphCheckpointer(session),
        graph=None,  # set per-run once resume identity is known; see PipelineRunner
        job_scout_repository=job_scout_repository,
    )


class PipelineRunner:
    """Runs or resumes the resume-to-outreach pipeline for a single resume,
    within a single database session's lifetime.

    A fresh `PipelineRunner` (and its underlying session/dependencies)
    should be constructed per invocation (`run_for_resume` / `resume_run`)
    rather than reused across runs, keeping each run's database session
    lifecycle explicit and short-lived.
    """

    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._checkpointer = GraphCheckpointer(session)

    def run_for_resume(self, resume_id: uuid.UUID) -> GraphState:
        """Run the full pipeline for a resume, starting fresh.

        Raises:
            PipelineRunnerError: if the resume cannot be found or has not
                yet been parsed.
        """
        resume = self._session.get(Resume, resume_id)
        if resume is None:
            raise PipelineRunnerError(f"No resume found with id '{resume_id}'.")
        if not resume.parsed_profile:
            raise PipelineRunnerError(
                f"Resume '{resume_id}' has not been parsed yet; cannot run pipeline."
            )

        resume_profile_state = ResumeProfileState(
            resume_id=str(resume.id),
            file_hash=resume.file_hash,
            full_name=resume.parsed_profile.get("full_name"),
            email=resume.parsed_profile.get("email"),
            skills=resume.parsed_profile.get("skills", []),
            target_roles=resume.parsed_profile.get("target_roles", []),
            years_of_experience=resume.parsed_profile.get("years_of_experience"),
        )

        initial_state = GraphState()
        initial_state.resume_profile = resume_profile_state

        graph = self._build_graph(resume_id=resume.id, resume_file_hash=resume.file_hash)

        logger.info(
            "Starting pipeline run",
            extra={"run_id": initial_state.run_metadata.run_id, "resume_id": str(resume_id)},
        )

        result_dict = graph.invoke(initial_state)
        final_state = GraphState.model_validate(result_dict)

        self._checkpointer.save_checkpoint(final_state)

        logger.info(
            "Pipeline run finished",
            extra={
                "run_id": final_state.run_metadata.run_id,
                "status": final_state.run_metadata.status.value,
            },
        )

        return final_state

    def resume_run(self, run_id: str, resume_id: uuid.UUID) -> GraphState:
        """Resume a previously checkpointed run.

        The resume's identity (`resume_id`) must still be supplied because
        the compiled graph and its agents are rebuilt fresh for this
        process invocation; only the `GraphState` itself is restored from
        the checkpoint.

        Raises:
            PipelineRunnerError: if the checkpoint cannot be resumed.
        """
        try:
            state = self._checkpointer.resume(run_id)
        except Exception as exc:  # noqa: BLE001 - normalize checkpoint failure
            raise PipelineRunnerError(f"Failed to resume run '{run_id}': {exc}") from exc

        resume = self._session.get(Resume, resume_id)
        if resume is None:
            raise PipelineRunnerError(f"No resume found with id '{resume_id}'.")

        graph = self._build_graph(resume_id=resume.id, resume_file_hash=resume.file_hash)

        logger.info("Resuming pipeline run", extra={"run_id": run_id})

        result_dict = graph.invoke(state)
        final_state = GraphState.model_validate(result_dict)

        self._checkpointer.save_checkpoint(final_state)

        logger.info(
            "Resumed pipeline run finished",
            extra={
                "run_id": final_state.run_metadata.run_id,
                "status": final_state.run_metadata.status.value,
            },
        )

        return final_state

    def _build_graph(self, resume_id: uuid.UUID, resume_file_hash: str):
        deps = build_pipeline_dependencies(self._session, self._settings)

        llm_provider = OllamaClient()
        vector_store = ChromaVectorStore()
        embedding_provider = OllamaEmbeddingProvider()

        connector_registry = build_default_registry(self._settings)
        job_scout_stage = JobScoutStageAdapter(
            agent=JobScoutAgent(registry=connector_registry),
            repository=deps.job_scout_repository,
        )

        jd_embedder = JdEmbedder(
            vector_store=vector_store,
            embedding_provider=embedding_provider,
            collection_name=self._settings.chroma_collection_jobs,
        )
        resume_match_stage = ResumeMatchStageAdapter(
            agent=ResumeMatchAgent(
                vector_store=vector_store,
                resume_collection_name=self._settings.chroma_collection_resumes,
                jd_embedder=jd_embedder,
                scorer=ResumeMatchScorer(min_score_threshold=self._settings.resume_match_min_score),
                repository=ResumeMatchRepository(self._session),
            ),
            resume_id=resume_id,
            resume_file_hash=resume_file_hash,
        )

        contact_finder_repository = ContactFinderRepository(self._session)
        cache = PostgresCache(self._session)
        contact_finder_stage = ContactFinderStageAdapter(
            agent=ContactFinderAgent(
                domain_resolver=PublicDomainResolver(),
                page_discoverer=PublicSourcePageDiscoverer(),
                profile_lookup_agent=ProfileLookupAgent(llm_provider=llm_provider),
                pattern_generator=EmailPatternGenerator(),
                smtp_validator=SmtpMailboxValidator(),
                hunter_client=HunterClient(
                    api_key=self._settings.hunter_io_api_key,
                    quota_tracker=HunterQuotaTracker(
                        cache=cache, monthly_quota=self._settings.hunter_io_monthly_quota
                    ),
                ),
                confidence_scorer=ConfidenceScorer(),
                repository=contact_finder_repository,
            ),
            repository=contact_finder_repository,
        )

        context_builder = OutreachContextBuilder()
        outreach_composer_stage = OutreachComposerStageAdapter(
            context_builder=context_builder,
            composer_agent=OutreachComposerAgent(llm_provider=llm_provider),
        )
        outreach_validator_stage = OutreachValidatorStageAdapter(
            context_builder=context_builder,
            validator_agent=OutreachValidatorAgent(llm_provider=llm_provider),
        )

        agents = GraphAgents(
            job_scout=job_scout_stage,
            resume_match=resume_match_stage,
            contact_finder=contact_finder_stage,
            outreach_composer=outreach_composer_stage,
            outreach_validator=outreach_validator_stage,
        )

        return build_state_graph(agents)


@contextmanager
def _runner_session():
    with session_scope() as session:
        yield PipelineRunner(session)


# --- CLI and scheduled execution -------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `python -m app.graph.runner`."""
    parser = argparse.ArgumentParser(
        prog="scout-mail-pipeline",
        description="Run or resume the Scout Mail resume-to-outreach pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline for a resume.")
    run_parser.add_argument("--resume-id", required=True, help="UUID of the resume to run.")

    resume_parser = subparsers.add_parser(
        "resume", help="Resume a previously checkpointed run."
    )
    resume_parser.add_argument("--run-id", required=True, help="run_id of the checkpoint.")
    resume_parser.add_argument(
        "--resume-id", required=True, help="UUID of the resume the run belongs to."
    )

    schedule_parser = subparsers.add_parser(
        "schedule", help="Run the pipeline on a recurring schedule."
    )
    schedule_parser.add_argument("--resume-id", required=True, help="UUID of the resume to run.")
    schedule_parser.add_argument(
        "--interval-hours",
        type=int,
        default=None,
        help="Override the poll interval (defaults to job_scout_poll_interval_hours).",
    )

    return parser


def run_once(resume_id: uuid.UUID) -> GraphState:
    """Run the pipeline once for `resume_id`, using a fresh session."""
    with _runner_session() as runner:
        return runner.run_for_resume(resume_id)


def resume_once(run_id: str, resume_id: uuid.UUID) -> GraphState:
    """Resume a checkpointed run once, using a fresh session."""
    with _runner_session() as runner:
        return runner.resume_run(run_id, resume_id)


def run_scheduled(resume_id: uuid.UUID, interval_hours: int | None = None) -> None:
    """Run the pipeline on a recurring schedule until interrupted.

    Registers SIGINT/SIGTERM handlers so the scheduler shuts down cleanly
    (waiting for any in-flight run to finish its current `graph.invoke`
    call and checkpoint before exiting) rather than being killed mid-run
    (PRD §12 — graceful degradation).
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    settings = get_settings()
    hours = interval_hours or settings.job_scout_poll_interval_hours

    scheduler = BackgroundScheduler()

    def _scheduled_job() -> None:
        try:
            run_once(resume_id)
        except PipelineRunnerError as exc:
            logger.error("Scheduled pipeline run failed", extra={"error": str(exc)})

    scheduler.add_job(
        _scheduled_job,
        trigger=IntervalTrigger(hours=hours),
        id="pipeline_scheduled_run",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    def _handle_shutdown_signal(signum: int, frame: FrameType | None) -> None:
        logger.info("Received shutdown signal; stopping scheduler", extra={"signal": signum})
        scheduler.shutdown(wait=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    logger.info(
        "Starting scheduled pipeline execution",
        extra={"resume_id": str(resume_id), "interval_hours": hours},
    )
    scheduler.start()

    signal.pause()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    logging.basicConfig(level=get_settings().log_level)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            final_state = run_once(uuid.UUID(args.resume_id))
            logger.info(
                "Run complete", extra={"status": final_state.run_metadata.status.value}
            )
            return 0

        if args.command == "resume":
            final_state = resume_once(args.run_id, uuid.UUID(args.resume_id))
            logger.info(
                "Resume complete", extra={"status": final_state.run_metadata.status.value}
            )
            return 0

        if args.command == "schedule":
            run_scheduled(uuid.UUID(args.resume_id), args.interval_hours)
            return 0

    except PipelineRunnerError as exc:
        logger.error("Pipeline run failed", extra={"error": str(exc)})
        return 1

    parser.error(f"Unknown command '{args.command}'.")
    return 2


if __name__ == "__main__":
    sys.exit(main())