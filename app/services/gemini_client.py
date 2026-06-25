"""Gemini client wrapper.

Isolates the two upstream interactions — ``count_tokens`` and
``generate_content`` — behind a small async-friendly interface. The blocking SDK
calls are run in a threadpool so they never stall the event loop.

Two explicit flows are exposed:

* ``count_input_tokens`` — used for the pre-flight input count. Raises
  ``GeminiError`` on failure (the guardrail needs a real input count).
* ``generate`` — the expensive generation call. Returns a ``GenerationResult``
  on success; raises ``GeminiUnavailable`` on failure so the caller can degrade
  to a prediction-only response instead of erroring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import google.generativeai as genai
from fastapi.concurrency import run_in_threadpool

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiError(Exception):
    """Raised when a required Gemini call (token count) fails."""


class GeminiUnavailable(Exception):
    """Raised when generation fails — caller may degrade gracefully."""


@dataclass(frozen=True)
class GenerationResult:
    """Successful generation output."""

    text: str
    actual_output: int


class GeminiClient:
    """Thin wrapper around the Gemini SDK model."""

    def __init__(self) -> None:
        self._model: genai.GenerativeModel | None = None

    def configure(self) -> bool:
        """Configure the SDK from settings. Returns whether a key was present."""
        if not settings.gemini_api_key:
            logger.warning("[gemini] GEMINI_API_KEY not set — upstream calls will fail.")
            return False
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_model)
        logger.info("[gemini] Configured model=%s.", settings.gemini_model)
        return True

    @property
    def is_configured(self) -> bool:
        return self._model is not None

    async def count_input_tokens(self, prompt: str) -> int:
        """Return the real input-token count for ``prompt``.

        Raises:
            GeminiError: If the count call fails or the client is unconfigured.
        """
        if self._model is None:
            raise GeminiError("Gemini client is not configured.")
        try:
            response = await run_in_threadpool(self._model.count_tokens, prompt)
            return response.total_tokens
        except Exception as exc:  # noqa: BLE001
            logger.error("[gemini] count_tokens failed: %s: %s", type(exc).__name__, exc)
            raise GeminiError(str(exc)) from exc

    async def generate(self, prompt: str) -> GenerationResult:
        """Generate a completion for ``prompt``.

        Raises:
            GeminiUnavailable: If generation fails or the client is unconfigured.
        """
        if self._model is None:
            raise GeminiUnavailable("Gemini client is not configured.")
        try:
            response = await run_in_threadpool(self._model.generate_content, prompt)
            actual = response.usage_metadata.candidates_token_count
            return GenerationResult(text=response.text, actual_output=actual)
        except Exception as exc:  # noqa: BLE001
            logger.error("[gemini] generate_content failed: %s: %s", type(exc).__name__, exc)
            raise GeminiUnavailable(str(exc)) from exc


# Process-wide singleton.
gemini_client = GeminiClient()
