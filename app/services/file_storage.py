"""
File storage service.

Implements: PRD §8.1 (Upload/parse resume (PDF/DOCX) → structured profile),
§9 (Non-Functional Requirements — Privacy: resume data stored only in the
user's own environment).
Roadmap: Epic 2 - Resume Ingestion & Profile Extraction, Story 1 - Resume Upload,
Task 2.

Defines the `FileStorageProvider` interface (domain-facing abstraction) and
its concrete local-disk implementation. Per docs/architecture.md, callers
(e.g. `app/api/resume_routes.py`) should depend on `FileStorageProvider`
rather than writing to disk directly — `app/api/resume_routes.py` currently
persists files inline and should be updated to delegate to this service in a
follow-up task, since only this file was requested here.

Note: if a dedicated `app/services/base.py` (or similar) is later preferred to
host the `FileStorageProvider` interface separately from this concrete
implementation, that split should be its own task — this file keeps both
together for now since only this file was requested.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class FileStorageError(Exception):
    """Raised when a file storage operation fails."""


@dataclass(frozen=True)
class StoredFile:
    """Result of a successful store operation."""

    file_hash: str
    storage_path: str
    already_existed: bool


class FileStorageProvider(ABC):
    """Abstraction over durable storage for uploaded files (e.g. resumes).

    Implementations are responsible for content-addressed storage (keyed by
    a hash of the file content) so repeated uploads of identical content are
    idempotent and cheap to detect (PRD §6a.3 — aggressive caching).
    """

    @abstractmethod
    def save(self, content: bytes, extension: str) -> StoredFile:
        """Persist file content and return its storage location and hash.

        Args:
            content: Raw file bytes.
            extension: File extension including the leading dot (e.g. ".pdf").

        Returns:
            StoredFile with the content hash, storage path, and whether an
            identical file was already stored.

        Raises:
            FileStorageError: if the content cannot be persisted.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, file_hash: str, extension: str) -> bool:
        """Return True if a file with this hash/extension is already stored."""
        raise NotImplementedError

    @abstractmethod
    def read(self, storage_path: str) -> bytes:
        """Read back previously stored file content.

        Raises:
            FileStorageError: if the file cannot be read.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_path(self, file_hash: str, extension: str) -> str:
        """Return the storage path a file with this hash/extension would use,
        without requiring it to already exist.
        """
        raise NotImplementedError


class LocalFileStorage(FileStorageProvider):
    """Concrete `FileStorageProvider` implementation backed by local disk.

    Files are stored under a configured root directory, named by their
    SHA-256 content hash plus original extension, keeping storage
    content-addressed and privacy-scoped to the user's own machine
    (PRD §9 — Privacy).
    """

    def __init__(self, root_dir: str) -> None:
        self._root_dir = Path(root_dir)

    def _ensure_root(self) -> None:
        try:
            self._root_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise FileStorageError(
                f"Failed to create storage root directory '{self._root_dir}'."
            ) from exc

    def resolve_path(self, file_hash: str, extension: str) -> str:
        return str(self._root_dir / f"{file_hash}{extension}")

    def exists(self, file_hash: str, extension: str) -> bool:
        return Path(self.resolve_path(file_hash, extension)).exists()

    def save(self, content: bytes, extension: str) -> StoredFile:
        if not content:
            raise FileStorageError("Cannot store empty file content.")

        self._ensure_root()
        file_hash = hashlib.sha256(content).hexdigest()
        path = Path(self.resolve_path(file_hash, extension))

        if path.exists():
            return StoredFile(
                file_hash=file_hash,
                storage_path=str(path),
                already_existed=True,
            )

        try:
            path.write_bytes(content)
        except OSError as exc:
            raise FileStorageError(
                f"Failed to write file to '{path}'."
            ) from exc

        return StoredFile(
            file_hash=file_hash,
            storage_path=str(path),
            already_existed=False,
        )

    def read(self, storage_path: str) -> bytes:
        path = Path(storage_path)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise FileStorageError(
                f"Failed to read file from '{storage_path}'."
            ) from exc