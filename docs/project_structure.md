# Project Structure — Scout Mail

This document defines the canonical folder layout for the Scout Mail codebase. All
future work must place new files according to this structure. Do not introduce
parallel or ad-hoc folders for functionality that already has a home below.

scout-mail/
├── app/
│ ├── main.py # FastAPI app entrypoint
│ ├── config.py # Env-based settings (Pydantic Settings)
│ │
│ ├── api/ # Interface layer — HTTP routes only
│ │ ├── resume_routes.py
│ │ ├── dashboard_routes.py
│ │ ├── review_routes.py
│ │ └── tracking_routes.py
│ │
│ ├── schemas/ # Pydantic request/response DTOs
│ │ ├── resume.py
│ │ └── dashboard.py
│ │
│ ├── db/ # Infrastructure — persistence
│ │ ├── base.py
│ │ ├── session.py
│ │ └── models.py
│ │
│ ├── agents/ # Domain + application logic, per agent
│ │ ├── resume_parser/
│ │ │ ├── extractor.py
│ │ │ ├── parser_agent.py
│ │ │ ├── embedder.py
│ │ │ └── repository.py
│ │ ├── job_scout/
│ │ │ ├── agent.py
│ │ │ └── repository.py
│ │ ├── resume_match/
│ │ │ ├── jd_embedder.py
│ │ │ ├── scorer.py
│ │ │ ├── agent.py
│ │ │ └── repository.py
│ │ ├── contact_finder/
│ │ │ ├── domain_resolver.py
│ │ │ ├── public_source_scraper.py
│ │ │ ├── profile_lookup.py
│ │ │ ├── pattern_generator.py
│ │ │ ├── smtp_validator.py
│ │ │ ├── hunter_client.py
│ │ │ ├── agent.py
│ │ │ └── repository.py
│ │ └── outreach/
│ │ ├── composer_agent.py
│ │ ├── validator_agent.py
│ │ └── repository.py
│ │
│ ├── connectors/ # Job discovery connectors (swappable)
│ │ ├── base.py
│ │ ├── registry.py
│ │ ├── greenhouse.py
│ │ ├── lever.py
│ │ ├── ashby.py
│ │ ├── career_page.py
│ │ └── rss.py
│ │
│ ├── graph/ # LangGraph orchestration
│ │ ├── state_schema.py
│ │ ├── state_graph.py
│ │ ├── checkpointer.py
│ │ └── runner.py
│ │
│ ├── llm/ # LLM provider wrappers
│ │ └── ollama_client.py
│ │
│ ├── vectorstore/
│ │ └── chroma_client.py
│ │
│ ├── services/ # Cross-cutting infrastructure services
│ │ ├── cache.py
│ │ ├── file_storage.py
│ │ ├── smtp_client.py
│ │ ├── email_sender.py
│ │ └── metrics.py
│ │
│ ├── scheduler/
│ │ └── jobs.py
│ │
│ ├── dashboard/ # Streamlit review UI
│ │ ├── app.py
│ │ └── components/
│ │ └── evidence_view.py
│ │
│ └── utils/
│ └── confidence.py
│
├── migrations/ # Alembic migration scripts
│ └── env.py
├── alembic.ini
│
├── tests/
│ ├── test_send_gate.py
│ ├── test_confidence_enum.py
│ └── test_no_batch_send.py
│
├── docs/
│ ├── project_structure.md # This file
│ ├── architecture.md
│ └── coding_guidelines.md
│
├── pyproject.toml
├── requirements.txt
└── .env.example

## Placement Rules

- **`api/`** contains only route definitions and request/response wiring. No business
  logic, no direct DB queries, no LLM calls.
- **`agents/<agent_name>/`** contains everything specific to one of the four PRD agents
  (Job Scout, Resume Match, Contact Finder, Outreach Composer + Validator). Each agent
  folder is self-contained: its own repository, its own agent entrypoint.
- **`connectors/`** holds only job-discovery source integrations. Adding a new source
  means adding one file here and registering it in `registry.py` — never modifying
  `job_scout/agent.py` internals.
- **`graph/`** is the only place LangGraph wiring lives. Agents themselves must not
  import LangGraph directly — the graph layer composes them.
- **`services/`** holds infrastructure concerns shared across agents (cache, SMTP,
  file storage, metrics). Agents depend on these through interfaces, not on each
  other's internals.
- **`db/models.py`** is the single source of truth for persisted schema. No agent
  defines its own table.
- New files always go into the existing folder matching their responsibility. A new
  top-level folder requires updating this document first (see `architecture.md`,
  Deviation Process).