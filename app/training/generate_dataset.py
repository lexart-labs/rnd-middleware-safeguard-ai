"""Dataset generation CLI / background task.

Iterates over every prompt in ``settings.prompts_path``, calls Gemini to obtain
the real output-token count, and appends each result to ``settings.dataset_path``.

Records are appended incrementally to an in-memory list that is flushed to disk
after each successful call, so partial progress survives an early stop without
re-reading the whole file on every record.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import google.generativeai as genai

from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_existing() -> list[dict[str, Any]]:
    if not settings.dataset_path.exists():
        return []
    try:
        with settings.dataset_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        logger.warning("Could not parse %s, starting fresh.", settings.dataset_path)
        return []


def _persist(dataset: list[dict[str, Any]]) -> None:
    settings.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.dataset_path.open("w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)


def _extract_usage(usage_metadata: Any) -> dict[str, Any]:
    return {
        "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
        "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
        "total_token_count": getattr(usage_metadata, "total_token_count", None),
        "cached_content_token_count": getattr(usage_metadata, "cached_content_token_count", None),
    }


def generate_dataset() -> None:
    """Generate (or extend) the training dataset by querying Gemini."""
    if not settings.gemini_api_key:
        raise OSError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.dataset_model)

    if not settings.prompts_path.exists():
        raise FileNotFoundError(
            f"{settings.prompts_path} not found. Download prompts.json from Kaggle (see README)."
        )
    with settings.prompts_path.open("r", encoding="utf-8") as fh:
        prompts_by_subject: dict[str, list[str]] = json.load(fh)

    dataset = _load_existing()
    total = sum(len(v) for v in prompts_by_subject.values())
    processed = 0

    for subject, prompts in prompts_by_subject.items():
        for prompt in prompts:
            try:
                logger.info(
                    "[%d/%d] subject=%s | %s...", processed + 1, total, subject, prompt[:70]
                )
                response = model.generate_content(prompt)
                usage = _extract_usage(response.usage_metadata)

                dataset.append(
                    {
                        "prompt": prompt,
                        "subject": subject,
                        "output_token": usage["candidates_token_count"],
                        "usage_metadata": usage,
                    }
                )
                _persist(dataset)
                processed += 1
                logger.info(
                    "  -> Saved. output_tokens=%s total_tokens=%s",
                    usage["candidates_token_count"],
                    usage["total_token_count"],
                )

                if processed < total:
                    time.sleep(settings.request_delay_seconds)

            except Exception as exc:  # noqa: BLE001 — log-and-stop is intentional
                logger.error("Request failed | subject=%s | prompt=%s", subject, prompt[:70])
                logger.error("  -> %s: %s", type(exc).__name__, exc)
                logger.info(
                    "Stopped early. %d record(s) saved to %s.", processed, settings.dataset_path
                )
                return

    logger.info("Completed. %d/%d record(s) saved to %s.", processed, total, settings.dataset_path)


if __name__ == "__main__":
    generate_dataset()
