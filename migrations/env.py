"""
Alembic environment script.

Implements: PRD §7 (Tech Stack — PostgreSQL via Supabase free tier), §9
(Auditability — schema changes tracked and reproducible via migrations).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 2 - Database Setup, Task 3.

Resolves the database URL and target metadata from the application's own
config and ORM base (app.config, app.db.base, app.db.models) rather than
hardcoding a connection string here, keeping a single source of truth per
docs/architecture.md.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db.base import Base

# Import models so their tables are registered on Base.metadata before
# autogenerate/upgrade runs. This import is required even though the
# names are not referenced directly below.
from app.db import models  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

_settings = get_settings()
config.set_main_option("sqlalchemy.url", str(_settings.database_url))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL without a live DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (executes against a live DB connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()