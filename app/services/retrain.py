"""Asynchronous MoPE retraining.

A thread-safe ``FeedbackBuffer`` collects ``(prompt, actual_output)`` pairs from
completed requests. When it reaches ``settings.retrain_threshold`` it schedules a
background retrain on a single-worker ``ThreadPoolExecutor`` (CPU-bound training
must not block the event loop). Retraining appends the new records to
``dataset.json``, fits fresh artifacts via the shared pipeline, atomically swaps
the ``.pkl`` files on disk, and hot-reloads them through the model store.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import joblib

from app.config import settings
from app.services.model_store import model_store
from app.training.pipeline import load_dataset, train

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrain")


class FeedbackBuffer:
    """Accumulates feedback records and triggers retraining on threshold."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._is_training = False

    def add(self, prompt: str, input_tokens: int, actual_tokens: int) -> None:
        """Record a completed request; schedule a retrain if the threshold is hit."""
        record = {
            "prompt": prompt,
            "subject": "general_chat",
            "output_token": actual_tokens,
            "usage_metadata": {
                "prompt_token_count": input_tokens,
                "candidates_token_count": actual_tokens,
                "total_token_count": input_tokens + actual_tokens,
                "cached_content_token_count": 0,
            },
        }
        with self._lock:
            self._records.append(record)
            count = len(self._records)
            should_trigger = count >= settings.retrain_threshold and not self._is_training
            if should_trigger:
                self._is_training = True
                snapshot = self._records.copy()
                self._records.clear()

        if should_trigger:
            logger.info("[retrain] Threshold reached (%d samples). Scheduling retrain.", count)
            _executor.submit(_run_retrain, snapshot, self._mark_done)

    def flush(self) -> bool:
        """Drain the buffer and retrain immediately. ``False`` if already running."""
        with self._lock:
            if self._is_training:
                return False
            self._is_training = True
            snapshot = self._records.copy()
            self._records.clear()

        _executor.submit(_run_retrain, snapshot, self._mark_done)
        return True

    def _mark_done(self) -> None:
        with self._lock:
            self._is_training = False
            should_trigger = len(self._records) >= settings.retrain_threshold
            if should_trigger:
                self._is_training = True
                snapshot = self._records.copy()
                self._records.clear()

        if should_trigger:
            logger.info(
                "[retrain] Buffer overflow post-retrain (%d samples). Scheduling immediately.",
                len(snapshot),
            )
            _executor.submit(_run_retrain, snapshot, self._mark_done)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def is_training(self) -> bool:
        with self._lock:
            return self._is_training


def _run_retrain(new_records: list[dict[str, Any]], on_done: Callable[[], None]) -> None:
    logger.info("[retrain] Started. New records: %d", len(new_records))
    try:
        dataset = load_dataset(settings.dataset_path)
        dataset.extend(new_records)
        total = len(dataset)

        if total < settings.min_samples_to_train:
            logger.warning(
                "[retrain] Only %d samples total — need at least %d. Skipping.",
                total,
                settings.min_samples_to_train,
            )
            return

        settings.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with settings.dataset_path.open("w", encoding="utf-8") as fh:
            json.dump(dataset, fh, indent=2, ensure_ascii=False)
        logger.info("[retrain] dataset.json updated: %d total records.", total)

        vectorizer, regressor = train(dataset)

        # Write temp files, then atomically swap (os.replace is atomic on Unix).
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        tmp_vec = settings.model_dir / "vectorizer_new.pkl"
        tmp_reg = settings.model_dir / "regressor_new.pkl"
        joblib.dump(vectorizer, tmp_vec)
        joblib.dump(regressor, tmp_reg)
        os.replace(tmp_vec, settings.vectorizer_path)
        os.replace(tmp_reg, settings.regressor_path)

        model_store.reload()
        logger.info("[retrain] Done. Model hot-reloaded with %d samples.", total)

    except Exception:  # noqa: BLE001 — log and keep serving the old model
        logger.exception("[retrain] Training failed.")
    finally:
        on_done()


# Process-wide singleton.
feedback_buffer = FeedbackBuffer()
