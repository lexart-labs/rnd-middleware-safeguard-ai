"""FastAPI application factory and entry point.

Loads the MoPE artifacts and configures the Gemini client at startup via the
lifespan handler, then mounts the health, chat, and admin routers. The service
starts even if artifacts or the API key are missing — it reports the degraded
state through ``/health`` and falls back at request time.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.routers import admin, chat, health
from app.services.gemini_client import gemini_client
from app.services.model_store import model_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts and configure the Gemini client on startup."""
    if not model_store.load():
        logger.warning("[startup] Model not loaded — serving in fallback mode.")
    if not gemini_client.configure():
        logger.warning("[startup] Gemini not configured — upstream calls will fail.")
    yield


app = FastAPI(
    title="MoPE Predictive Middleware",
    description="Marginal-latency token-budget safeguard for Gemini.",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(admin.router)
