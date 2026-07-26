"""
SQLAlchemy declarative base.

Implements: PRD §7 (Tech Stack — PostgreSQL via Supabase free tier).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 2 - Database Setup, Task 1.

Single declarative base shared by every ORM model in the codebase. Per
docs/project_structure.md, `app/db/models.py` is the only module that defines
table classes — this module only provides the base class they inherit from,
keeping table metadata centralized (Single Responsibility, per
docs/coding_guidelines.md).
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base class for all ORM models in the application."""