"""Lightweight usage/cost tracking for the model router."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _ModelUsage:
    """Accumulated stats for a single model."""

    calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    errors: int = 0
    total_latency_s: float = 0.0


class RouterStats:
    """Tracks per-model usage counts, token estimates, and latency."""

    def __init__(self) -> None:
        self._stats: dict[str, _ModelUsage] = {}

    def _ensure(self, model: str) -> _ModelUsage:
        if model not in self._stats:
            self._stats[model] = _ModelUsage()
        return self._stats[model]

    def record_call(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_s: float = 0.0,
    ) -> None:
        """Record a successful call to *model*."""
        usage = self._ensure(model)
        usage.calls += 1
        usage.total_input_tokens += input_tokens
        usage.total_output_tokens += output_tokens
        usage.total_latency_s += latency_s

    def record_error(self, model: str) -> None:
        """Record a failed call to *model*."""
        self._ensure(model).errors += 1

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of all recorded stats."""
        return {
            model: {
                "calls": u.calls,
                "errors": u.errors,
                "total_input_tokens": u.total_input_tokens,
                "total_output_tokens": u.total_output_tokens,
                "total_latency_s": round(u.total_latency_s, 3),
            }
            for model, u in sorted(self._stats.items())
        }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars per token for English text)."""
        return max(1, len(text) // 4)

    @staticmethod
    def timer() -> "_Timer":
        """Context-manager that measures elapsed wall-clock seconds."""
        return _Timer()


@dataclass
class _Timer:
    """Simple wall-clock timer usable as a context manager."""

    elapsed: float = 0.0
    _start: float = field(default=0.0, repr=False)

    def __enter__(self) -> "_Timer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed = time.monotonic() - self._start
