"""
Ollama-backed LLM provider.

Implements: PRD §7 (Tech Stack — Ollama running Llama 3.1 8B / Mistral 7B
locally), §6a.3 (Modular LLM Pipeline — task-specialized local models rather
than one model for everything).
Roadmap: Epic 1 - Project Foundation & Infra Setup, Story 3 - Local LLM + Vector
Store Bootstrap, Task 1.

Defines the `LLMProvider` interface (domain-facing abstraction) and its
concrete Ollama implementation. Per docs/architecture.md, `app/agents/*` must
depend on `LLMProvider`, never on `OllamaClient` directly — concrete wiring
happens only in `app/main.py` / `app/graph/runner.py` (Dependency Inversion).

Note: if a dedicated `app/llm/base.py` is later preferred to host the
`LLMProvider` interface separately from this concrete implementation (per
docs/project_structure.md conventions for other interfaces such as
`connectors/base.py`), that split should be done as its own task — this file
keeps both together for now since only this file was requested.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Literal

import ollama
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

LLMTask = Literal["extraction", "generation", "validation"]


class LLMProviderError(Exception):
    """Raised when an LLM provider fails to produce a usable response."""


class LLMProvider(ABC):
    """Abstraction over an LLM backend, keyed by task type.

    Per PRD §6a.3, different tasks (extraction, generation, validation) are
    routed to different, task-specialized models. Implementations decide how
    a given `task` maps to a concrete model; callers never reference model
    names directly.
    """

    @abstractmethod
    def generate(self, task: LLMTask, prompt: str, system: str | None = None) -> str:
        """Generate free-form text for the given task."""
        raise NotImplementedError

    @abstractmethod
    def generate_json(
        self, task: LLMTask, prompt: str, system: str | None = None
    ) -> dict:
        """Generate a response and parse it as a JSON object.

        Raises:
            LLMProviderError: if the model output is not valid JSON.
        """
        raise NotImplementedError


class OllamaClient(LLMProvider):
    """Concrete `LLMProvider` implementation backed by a local Ollama server."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = ollama.Client(host=settings.ollama_base_url)
        self._timeout = settings.llm_request_timeout_seconds
        self._model_by_task: dict[LLMTask, str] = {
            "extraction": settings.llm_model_extraction,
            "generation": settings.llm_model_generation,
            "validation": settings.llm_model_validation,
        }

    def _model_for(self, task: LLMTask) -> str:
        try:
            return self._model_by_task[task]
        except KeyError as exc:
            raise LLMProviderError(f"No model configured for task '{task}'.") from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ollama.ResponseError, ConnectionError)),
        reraise=True,
    )
    def generate(self, task: LLMTask, prompt: str, system: str | None = None) -> str:
        model = self._model_for(task)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=model,
                messages=messages,
                options={"timeout": self._timeout},
            )
        except (ollama.ResponseError, ConnectionError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any client failure
            raise LLMProviderError(
                f"Ollama generation failed for task '{task}' with model '{model}'."
            ) from exc

        content = response.get("message", {}).get("content", "")
        if not content:
            raise LLMProviderError(
                f"Ollama returned an empty response for task '{task}' with model '{model}'."
            )
        return content

    def generate_json(
        self, task: LLMTask, prompt: str, system: str | None = None
    ) -> dict:
        json_system = (
            (system + "\n\n" if system else "")
            + "Respond with a single valid JSON object only. "
            "No prose, no markdown code fences, no explanation."
        )
        raw = self.generate(task=task, prompt=prompt, system=json_system)

        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"Model did not return valid JSON for task '{task}': {raw[:200]!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMProviderError(
                f"Expected a JSON object for task '{task}', got {type(parsed).__name__}."
            )
        return parsed