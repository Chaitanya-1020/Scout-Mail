# Coding Guidelines — Scout Mail

## 1. Language & Style

- Python 3.11+. Type hints mandatory on every function signature (params + return).
- Follow PEP 8. Line length 100 chars max.
- Formatter: `black`. Linter: `ruff`. Both run in CI; no merge on failure.
- No bare `except:`. Catch specific exceptions; re-raise with context when crossing
  a layer boundary (e.g. wrap a `requests` exception into a domain-level
  `ConnectorFetchError`).
- No `print()` in application/domain code — use the configured logger.

## 2. Clean Architecture Rules (enforced, not advisory)

- Files under `app/agents/*` and `app/utils/*` must not import from `app/db`,
  `app/llm`, `app/vectorstore`, or `app/connectors` directly. They depend on
  interfaces passed in via constructor/function parameters (dependency injection).
- Concrete infrastructure classes (`OllamaClient`, `ChromaClient`, SQLAlchemy
  sessions, connector classes) are instantiated only in `app/main.py` or
  `app/graph/runner.py` and passed down.
- `app/api/*` route handlers contain no business logic — they parse/validate input,
  call an application-layer function, and map the result to a response schema.
- Every interface (`ConnectorBase`, `LLMProvider`, `CacheProvider`) is defined as an
  `abc.ABC` with `@abstractmethod`, living next to its first consumer's layer
  boundary, not inside a concrete implementation file.

## 3. SOLID Enforcement Checklist (apply before every PR)

- **S**: does this file/class do exactly one job named in `project_structure.md`?
  If a file is doing extraction AND persistence, split it.
- **O**: can a new job source / evidence layer / LLM model be added without
  editing this file? If not, extract the varying part behind an interface.
- **L**: does every implementation of an interface honor the same method
  signature and behavioral contract (no implementation silently returning `None`
  where others return a list)?
- **I**: does this interface expose only what its narrowest consumer needs?
  Split fat interfaces.
- **D**: does this module import concrete infrastructure directly? If yes, and
  it's not `main.py` / `runner.py`, refactor to accept an injected dependency.

## 4. Data & Confidence Rules (PRD §6a, §13.2 — non-negotiable)

- Contact confidence is always one of `ConfidenceLevel.HIGH / MEDIUM / LOW`. No
  boolean "verified" field is ever added to any schema, migration, or DTO.
- Every stored fact (email, company, role) carries its source reference and the
  confidence level alongside the value — never store a bare value without
  provenance.
- Role title fields store the verbatim string from the source JD. No paraphrasing,
  truncation, or normalization that changes the original wording.

## 5. Send-Path Rules (PRD §3, §13.2 — non-negotiable)

- Any function capable of transmitting an email must check
  `approved_by_human is True` as its first guard clause and raise/return before
  doing anything else if false.
- No function, endpoint, or script may send more than one email per approved
  record. No "send all" / batch iteration over unapproved or multiple records is
  permitted anywhere in the codebase.
- Rate limiting/send caps are implemented as server-side logic in
  `services/email_sender.py`, never left to the UI to enforce alone.

## 6. Testing

- Every new interface implementation ships with a unit test using a fake/stub of
  its dependencies — no test hits a real external API, real SMTP server, or real
  LLM by default.
- Guardrail behaviors (send gate, confidence enum, no-batch-send) each have a
  dedicated test in `tests/`, referenced by the PRD section they enforce.
- Test names reference behavior, not implementation: `test_send_blocked_when_not_approved`,
  not `test_email_sender_function_1`.

## 7. Documentation & Traceability

- Every new module gets a module-level docstring stating which PRD section and
  roadmap Epic/Story/Task it implements.
- Any deviation from the PRD or this architecture must be documented per
  `architecture.md` §5 before the deviating code is merged.

## 8. Commit Discipline

- Conventional commit prefixes: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.
- One logical change per commit. No commit that mixes a new agent feature with an
  unrelated refactor.