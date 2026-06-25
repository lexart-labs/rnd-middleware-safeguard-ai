"""Centralized application configuration.

Every environment-driven value lives here as a single typed ``Settings`` object,
replacing the scattered ``os.environ.get`` reads that used to sit in ``main.py``,
``retrain_service.py`` and ``generate_dataset.py``. Import the shared ``settings``
instance rather than reading the environment directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: <root>/app/config.py -> parents[1] == <root>
ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed application settings, loaded from environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow fields like ``model_dir`` without clashing with Pydantic's
        # reserved ``model_`` namespace.
        protected_namespaces=(),
    )

    # --- Gemini ---------------------------------------------------------------
    gemini_api_key: str = ""
    """API key for Google Gemini. Empty means the service runs in a degraded,
    prediction-only mode (no upstream calls succeed)."""

    gemini_model: str = "gemini-3-flash-preview"
    """Model used to serve chat completions in the request path."""

    dataset_model: str = "gemini-3.1-flash-lite-preview"
    """Cheaper model used to build the training dataset from ``prompts.json``."""

    # --- Paths ----------------------------------------------------------------
    model_dir: Path = ROOT_DIR / "data" / "models"
    """Directory holding the committed pre-trained artifacts."""

    dataset_path: Path = ROOT_DIR / "data" / "dataset.json"
    """Training dataset (downloaded from Kaggle / generated). Not committed."""

    prompts_path: Path = ROOT_DIR / "data" / "prompts.json"
    """Seed prompts for dataset generation (downloaded from Kaggle). Not committed."""

    # --- Retraining -----------------------------------------------------------
    retrain_threshold: int = 50
    """New feedback records that trigger an automatic background retrain."""

    min_samples_to_train: int = 5
    """Minimum total dataset size required before training will proceed."""

    request_delay_seconds: float = 6.0
    """Delay between Gemini calls during dataset generation (rate limiting)."""

    # --- Budget ---------------------------------------------------------------
    default_user_limit: int = 10000
    """Token budget assigned to seeded demo users."""

    @property
    def vectorizer_path(self) -> Path:
        return self.model_dir / "vectorizer_mope.pkl"

    @property
    def regressor_path(self) -> Path:
        return self.model_dir / "regressor_mope.pkl"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached ``Settings`` instance."""
    return Settings()


settings = get_settings()
