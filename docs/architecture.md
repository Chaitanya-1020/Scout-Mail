# Architecture — Scout Mail

## 1. Architectural Style

Scout Mail follows **Clean Architecture**, adapted for a multi-agent LangGraph system.
Dependencies point inward: infrastructure and interface code depend on domain logic,
never the reverse.

### Layers

1. **Domain layer** — pure business rules with no framework dependency.
   - Confidence scoring rules (`app/utils/confidence.py`)
   - Contract interfaces (abstract base classes) for connectors, LLM clients, cache,
     email sender.
2. **Application layer** — orchestrates domain rules to fulfill a use case.
   - Agent modules (`app/agents/*`): each agent is a use-case coordinator (e.g.
     "resolve a contact with confidence scoring", "draft and validate an outreach
     email").
   - `app/graph/state_graph.py`: composes the four agents into the end-to-end
     resume-to-outreach workflow.
3. **Interface adapters** — translate between the outside world and the application
   layer.
   - `app/api/*`: FastAPI routes, request/response mapping only.
   - `app/dashboard/*`: Streamlit UI, calls application layer through the API.
   - `app/connectors/*`: adapters implementing the connector interface for each job
     board/ATS.
4. **Infrastructure layer** — concrete technical implementations.
   - `app/db/*`: SQLAlchemy models and session management (Postgres/Supabase).
   - `app/llm/ollama_client.py`: concrete Ollama-backed LLM provider.
   - `app/vectorstore/chroma_client.py`: concrete ChromaDB provider.
   - `app/services/*`: cache, SMTP, file storage, metrics implementations.

### Dependency Rule

- `agents/*` depend on abstract interfaces (`connectors/base.py`, an `LLMProvider`
  interface, a `CacheProvider` interface), never on a concrete infrastructure class
  directly. Concrete implementations are injected at the composition root
  (`app/main.py` / `app/graph/runner.py`).
- `api/*` and `dashboard/*` depend on `agents/*` and `graph/*`, never the reverse.
- `db/models.py` has no knowledge of agents or API routes.

## 2. SOLID Application

- **Single Responsibility**: each agent folder owns exactly one PRD agent's
  responsibility. Each connector file owns exactly one job source. `confidence.py`
  owns only scoring math — no I/O.
- **Open/Closed**: new job sources or new confidence-evidence layers are added by
  creating a new class implementing an existing interface (`ConnectorBase`,
  an `EvidenceLayer` protocol) and registering it — no existing agent code is
  modified.
- **Liskov Substitution**: any class implementing `ConnectorBase` must be usable
  interchangeably by `job_scout/agent.py` without special-casing by source. Any
  `LLMProvider` implementation must be swappable (e.g. Ollama model change) without
  touching agent logic.
- **Interface Segregation**: connector interface exposes only `fetch_postings()`;
  it does not force implementers to support scheduling, caching, or persistence —
  those are separate collaborators.
- **Dependency Inversion**: `agents/*` depend on `LLMProvider`, `CacheProvider`,
  `ConnectorBase` abstractions defined at the domain/application boundary, not on
  `ollama_client.py`, `cache.py`, or `greenhouse.py` concretely. Concrete wiring
  happens only in `graph/runner.py` and `main.py`.

## 3. Multi-Agent / LangGraph Composition

The four PRD agents (§5 of the PRD) are independent application-layer units:

1. Job Scout Agent
2. Resume Match Agent (RAG)
3. Contact Finder Agent (layered confidence pipeline)
4. Outreach Composer + Validator Agent

`app/graph/state_graph.py` is the only module aware of all four agents simultaneously.
It defines nodes and edges; each node calls exactly one agent's public entrypoint
function. Agents never call each other directly — cross-agent coordination and state
handoff belongs to the graph layer only. This preserves single responsibility and
keeps every agent independently testable.

State is persisted via `app/graph/checkpointer.py` against Postgres, so a run can
resume after failure without re-doing completed steps (supports PRD §9 auditability
and cost constraints).

## 4. Non-Goal Enforcement as Architecture

Per PRD §13.2, non-goals are enforced in code, not convention:

- `services/email_sender.py` is the single choke point for outbound email. It
  hard-checks `approved_by_human = true` before any SMTP call — this check cannot
  be bypassed by calling a backend endpoint directly, because no other code path
  constructs an SMTP call.
- The domain layer only allows a `ConfidenceLevel` enum (`High/Medium/Low`) on
  contact records. No "verified" boolean type exists anywhere in `db/models.py` or
  `utils/confidence.py`.
- `email_sender.py` enforces a per-run send cap and one-to-one sends server-side;
  no batch-send function exists in the codebase.

## 5. Deviation Process

Per PRD §13.3, any architectural change (new top-level folder, new dependency
direction, new agent) must:

1. Be documented with the forcing constraint.
2. Update this file and `project_structure.md` first.
3. Only then be implemented.

Code review must check new changes against this document and the PRD section it
maps to before merge.