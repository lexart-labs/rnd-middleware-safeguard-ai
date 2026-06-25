"""Per-user token budget store.

Currently an in-memory implementation seeded with a demo user. The interface is
intentionally small so it can be swapped for a Redis/DB-backed store later
without touching the routers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.config import settings


@dataclass
class Quota:
    """A single user's token budget state."""

    limit: int
    consumed: int = 0

    @property
    def available(self) -> int:
        return self.limit - self.consumed


class BudgetStore:
    """Thread-safe in-memory quota store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quotas: dict[str, Quota] = {
            "user_123": Quota(limit=settings.default_user_limit),
        }

    def get(self, user_id: str) -> Quota | None:
        """Return the user's quota, or ``None`` if unknown."""
        with self._lock:
            return self._quotas.get(user_id)

    def charge(self, user_id: str, tokens: int) -> int:
        """Debit ``tokens`` from the user and return the remaining budget.

        Raises:
            KeyError: If the user is unknown.
        """
        with self._lock:
            quota = self._quotas[user_id]
            quota.consumed += tokens
            return quota.available


# Process-wide singleton.
budget_store = BudgetStore()
