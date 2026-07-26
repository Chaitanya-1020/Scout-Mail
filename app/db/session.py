"""
Database session management.

Implements: PRD §7 (Tech Stack — PostgreSQL via Supabase free tier), §9
(Non-Functional Requirements — Auditability, via a single consistent
persistence entrypoint).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 2 - Database Setup, Task 1.

Provides the SQLAlchemy engine, session factory, and a FastAPI-compatible
dependency for obtaining a scoped session. This is the only module that
constructs a database engine or session factory — per docs/architecture.md,
`app/agents/*` never import this module directly; sessions are injected at
the composition root (`app/main.py` / `app/graph/runner.py`).
"""

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

engine = create_engine(
    str(_settings.database_url),
    pool_size=_settings.database_pool_size,
    max_overflow=_settings.database_max_overflow,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    future=True,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session per request.

    Commits on clean exit, rolls back on exception, always closes the
    session. Route handlers depend on this via `Depends(get_db)` rather than
    constructing sessions themselves (per docs/coding_guidelines.md §2).
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context-manager session for non-request contexts (agents, scheduler jobs,
    graph runner). Commits on clean exit, rolls back on exception, always closes.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()