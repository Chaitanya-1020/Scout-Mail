"""
Resume API schemas.

Implements: PRD §8.1 (Upload/parse resume (PDF/DOCX) → structured profile).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 1 - Resume Upload,
Task 1.

Pydantic request/response DTOs for the resume upload endpoint. Per
docs/coding_guidelines.md, these carry no business logic — they only shape
data crossing the API boundary.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ResumeUploadResponse(BaseModel):
    """Response returned after a resume file has been accepted and stored."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str = Field(..., description="Name of the uploaded file as received.")
    file_hash: str = Field(..., description="SHA-256 hash of the file content, used for caching/dedup.")
    created_at: datetime
    already_existed: bool = Field(
        ...,
        description="True if a resume with this exact file content was already stored "
        "(upload was a no-op cache hit per PRD §6a.3).",
    )