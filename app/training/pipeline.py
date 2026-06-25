"""Shared MoPE training pipeline.

This is the single source of truth for how the vectorizer and regressor are
built. Both the offline CLI (``train_proxy``) and the async retrainer
(``services.retrain``) call ``train`` so the two can never drift apart.

The regressor uses **quantile regression** (pinball loss at q=0.85) rather than
mean regression: underestimating output tokens exhausts user budgets, so the
model is deliberately trained to predict an upper bound.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

QUANTILE_ALPHA = 0.85
MAX_FEATURES = 1000
N_ESTIMATORS = 100


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Project raw dataset records onto the ``(prompt, actual_tokens)`` columns."""
    return pd.DataFrame(
        [{"prompt": r["prompt"], "actual_tokens": r["output_token"]} for r in records]
    )


def train(records: list[dict[str, Any]]) -> tuple[TfidfVectorizer, GradientBoostingRegressor]:
    """Train a vectorizer + quantile regressor on ``records``.

    Args:
        records: Dataset entries, each with at least ``prompt`` and ``output_token``.

    Returns:
        The fitted ``(vectorizer, regressor)`` pair.

    Raises:
        ValueError: If ``records`` is empty.
    """
    if not records:
        raise ValueError("Cannot train on an empty dataset.")

    df = records_to_frame(records)
    logger.info("[train] Fitting on %d samples.", len(df))

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    features = vectorizer.fit_transform(df["prompt"])
    targets = df["actual_tokens"]

    regressor = GradientBoostingRegressor(
        loss="quantile", alpha=QUANTILE_ALPHA, n_estimators=N_ESTIMATORS
    )
    regressor.fit(features, targets)

    return vectorizer, regressor


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a dataset JSON file, returning ``[]`` when missing or malformed."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        logger.warning("[train] %s is malformed — treating as empty.", path)
        return []
