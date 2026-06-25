"""Centralized, thread-safe holder for the MoPE artifacts.

Replaces the old module-level globals in ``model_state.py``. Artifacts are
loaded once at startup and can be hot-swapped by the retraining service via
``reload()`` under a lock. If artifacts are missing or corrupt the store stays
in a not-ready state instead of crashing — callers check ``is_ready()`` and fall
back to a conservative heuristic (see ``predictor``).
"""

from __future__ import annotations

import logging
import threading

import joblib

from app.config import settings

logger = logging.getLogger(__name__)


class ModelStore:
    """Owns the vectorizer/regressor pair and serializes access to them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vectorizer = None
        self._regressor = None

    def load(self) -> bool:
        """Load artifacts from disk. Returns ``True`` on success.

        A failure leaves any previously-loaded artifacts untouched so the
        service keeps serving the last known-good model.
        """
        try:
            vectorizer = joblib.load(settings.vectorizer_path)
            regressor = joblib.load(settings.regressor_path)
        except Exception as exc:  # noqa: BLE001 — missing/corrupt artifacts are expected
            logger.warning(
                "[model_store] Artifact load failed: %s. Serving in fallback mode "
                "(run `python -m app.training.train_proxy` or download the .pkl files).",
                exc,
            )
            return False

        with self._lock:
            self._vectorizer = vectorizer
            self._regressor = regressor
        logger.info("[model_store] Artifacts loaded from %s.", settings.model_dir)
        return True

    # ``reload`` is an alias kept for the retraining service's vocabulary.
    reload = load

    def is_ready(self) -> bool:
        """Whether both artifacts are loaded and usable."""
        with self._lock:
            return self._vectorizer is not None and self._regressor is not None

    @property
    def artifacts(self):
        """Return the current ``(vectorizer, regressor)`` pair (may be ``None``)."""
        with self._lock:
            return self._vectorizer, self._regressor


# Process-wide singleton.
model_store = ModelStore()
