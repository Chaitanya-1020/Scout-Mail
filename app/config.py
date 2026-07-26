"""
Application configuration.

Implements: PRD §7 (Tech Stack), §9 (Non-Functional Requirements — Extensibility).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 1 - Repository Setup, Task 2.

Loads all environment-driven settings for the application through a single,
typed Settings object. No module elsewhere in the codebase reads `os.environ`
directly — every configurable value is exposed here so infrastructure choices
(DB, LLM models, quotas) can change without touching agent or domain code
(Dependency Inversion, per docs/architecture.md and docs/coding_guidelines.md).
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str | list[str]) -> list[str]:
    """Parse a comma-separated env string into a clean list of strings.

    Allows `.env` files to declare list-valued settings (e.g. board tokens,
    feed URLs) as plain comma-separated values instead of JSON arrays, which
    is friendlier for manual editing of `.env.example` / `.env`.
    """
    if isinstance(value, list):
        return value
    if not value or not value.strip():
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    """Typed, validated application settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = Field(default="scout-mail")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # --- Database (PRD §7: PostgreSQL via Supabase free tier) ---
    database_url: PostgresDsn = Field(
        ...,
        description="Postgres connection string (Supabase free-tier project or local Postgres fallback).",
    )
    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=5, ge=0)

    # --- Vector store (PRD §7: ChromaDB, local embedded mode) ---
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection_resumes: str = Field(default="resumes")
    chroma_collection_jobs: str = Field(default="job_descriptions")

    # --- LLM inference (PRD §7, §6a.3: task-specialized local models via Ollama) ---
    ollama_base_url: str = Field(default="http://localhost:11434")
    llm_model_extraction: str = Field(
        default="phi3:mini",
        description="Model used for parsing/extraction tasks (resume/JD structured output).",
    )
    llm_model_generation: str = Field(
        default="llama3.1:8b",
        description="Model used for outreach email drafting.",
    )
    llm_model_validation: str = Field(
        default="mistral:7b",
        description="Model used for cross-checking/validation of drafted outreach.",
    )
    llm_request_timeout_seconds: int = Field(default=120, ge=1)

    # --- Contact Finder / email verification (PRD §6.3, §6a.1) ---
    hunter_io_api_key: str | None = Field(default=None)
    hunter_io_monthly_quota: int = Field(default=25, ge=0)
    smtp_handshake_timeout_seconds: int = Field(default=10, ge=1)

    # --- Job discovery connectors (PRD §6a.2: allowed sources only) ---
    job_scout_poll_interval_hours: int = Field(default=6, ge=1)
    greenhouse_board_tokens: list[str] = Field(default_factory=list)
    lever_company_slugs: list[str] = Field(default_factory=list)
    ashby_org_slugs: list[str] = Field(default_factory=list)
    rss_feed_urls: list[str] = Field(default_factory=list)

    @field_validator(
        "greenhouse_board_tokens",
        "lever_company_slugs",
        "ashby_org_slugs",
        "rss_feed_urls",
        mode="before",
    )
    @classmethod
    def _parse_csv_lists(cls, value: str | list[str]) -> list[str]:
        return _split_csv(value)

    # --- Outbound email sending (PRD §8.8: user's own SMTP/Gmail API credentials) ---
    smtp_host: str | None = Field(default=None)
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str | None = Field(default=None)
    smtp_password: str | None = Field(default=None)
    smtp_use_tls: bool = Field(default=True)
    max_sends_per_run: int = Field(
        default=1,
        ge=1,
        description="Server-side send cap enforced in services/email_sender.py "
        "(PRD §13.2 — no bulk/spam outreach).",
    )

    # --- Resume match thresholds (PRD §8.4) ---
    resume_match_min_score: float = Field(default=0.55, ge=0.0, le=1.0)

    # --- File storage ---
    resume_upload_dir: str = Field(default="./data/resumes")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()