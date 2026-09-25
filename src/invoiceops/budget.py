"""Cost cap: a daily token budget shared by every run.

Before each model call the Supervisor asks "can we afford this?". If not, the invoice is parked
for a human instead of silently burning money (or hitting the provider's daily limit).
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

DAILY_TOKEN_BUDGET = int(os.environ.get("DAILY_TOKEN_BUDGET", "150000"))
TOKENS_PER_PAGE_ESTIMATE = 3500  # ~3k in + reply, measured on the sample invoices


class Budget:
    def __init__(self, path: str | Path, daily_limit: int = DAILY_TOKEN_BUDGET) -> None:
        self.path = Path(path)
        self.daily_limit = daily_limit

    def _load(self) -> dict[str, int]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def used_today(self) -> int:
        return self._load().get(date.today().isoformat(), 0)

    def remaining(self) -> int:
        return max(self.daily_limit - self.used_today(), 0)

    def can_afford(self, pages: int) -> bool:
        return self.remaining() >= pages * TOKENS_PER_PAGE_ESTIMATE

    def charge(self, tokens: int) -> None:
        data = self._load()
        today = date.today().isoformat()
        data[today] = data.get(today, 0) + tokens
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")