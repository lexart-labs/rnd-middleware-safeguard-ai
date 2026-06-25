"""Offline training CLI: build the initial ``.pkl`` artifacts from a dataset.

Run before serving for the first time (or to regenerate committed artifacts):

    python -m app.training.train_proxy

Reads ``settings.dataset_path`` and writes the vectorizer/regressor into
``settings.model_dir``.
"""

from __future__ import annotations

import logging

import joblib

from app.config import settings
from app.training.pipeline import load_dataset, train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load the dataset, train artifacts, and persist them to ``model_dir``."""
    records = load_dataset(settings.dataset_path)
    if not records:
        raise SystemExit(
            f"No training data found at {settings.dataset_path}. "
            "Download dataset.json from Kaggle (see README) or run generate_dataset first."
        )

    logger.info("[train] Loaded %d samples from %s.", len(records), settings.dataset_path)
    vectorizer, regressor = train(records)

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, settings.vectorizer_path)
    joblib.dump(regressor, settings.regressor_path)
    logger.info(
        "[train] Artifacts saved: %s, %s",
        settings.vectorizer_path,
        settings.regressor_path,
    )


if __name__ == "__main__":
    main()
