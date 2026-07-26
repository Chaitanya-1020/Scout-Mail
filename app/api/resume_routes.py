"""
Resume upload API routes.

Implements: PRD §8.1 (Upload/parse resume (PDF/DOCX) → structured profile).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 1 - Resume Upload,
Task 1.

Route handlers parse/validate input, persist the raw upload, and map the
result to a response schema. Per docs/architecture.md, this module contains
no parsing/extraction logic — structured-profile extraction is implemented
separately in `app/agents/resume_parser/*` (Epic 2, Story 2) and will populate
`Resume.parsed_profile` in a later step. Persistent storage of the uploaded
file bytes is handled inline here pending `app/services/file_storage.py`
(Epic 2, Story 1, Task 2); once that module exists, this route should be
updated to delegate to it instead of writing to disk directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Resume
from app.db.session import get_db
from app.schemas.resume import ResumeUploadResponse

router = APIRouter(prefix="/resumes", tags=["resumes"])

_ALLOWED_EXTENSIONS = {".pdf", ".docx"}
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post(
    "",
    response_model=ResumeUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_resume(
    file: UploadFile,
    db: Session = Depends(get_db),
) -> ResumeUploadResponse:
    """Accept a PDF or DOCX resume upload and persist it (PRD §8.1).

    Structured profile extraction (skills, experience, target roles) is
    performed by a downstream agent, not here — this endpoint only accepts
    and durably stores the raw file, keyed by content hash for idempotency
    and caching (PRD §6a.3).
    """
    original_filename = file.filename or "resume"
    extension = Path(original_filename).suffix.lower()

    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{extension}'. Allowed types: "
            f"{', '.join(sorted(_ALLOWED_EXTENSIONS))}.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {_MAX_UPLOAD_BYTES} bytes.",
        )

    file_hash = hashlib.sha256(content).hexdigest()

    existing = db.execute(
        select(Resume).where(Resume.file_hash == file_hash)
    ).scalar_one_or_none()
    if existing is not None:
        return ResumeUploadResponse(
            id=existing.id,
            original_filename=existing.original_filename,
            file_hash=existing.file_hash,
            created_at=existing.created_at,
            already_existed=True,
        )

    settings = get_settings()
    upload_dir = Path(settings.resume_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    storage_path = upload_dir / f"{file_hash}{extension}"
    try:
        storage_path.write_bytes(content)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist uploaded resume file.",
        ) from exc

    resume = Resume(
        original_filename=original_filename,
        storage_path=str(storage_path),
        file_hash=file_hash,
        parsed_profile={},
    )
    db.add(resume)
    db.flush()
    db.refresh(resume)

    return ResumeUploadResponse(
        id=resume.id,
        original_filename=resume.original_filename,
        file_hash=resume.file_hash,
        created_at=resume.created_at,
        already_existed=False,
    )