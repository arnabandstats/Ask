"""Token usage and cost of the current session.

Every model call (agent, LLM judge, vision, guardrail check) reports its token usage to
the Meter of the session that made it. The meter is found through a ContextVar set by
the UI around each turn; worker threads get it via in_context().
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field

# USD per 1M tokens: (input, cached input, output). OpenAI list prices after the
# 30 July 2026 change; editable in Settings → Cost (saved in preferences.json).
DEFAULT_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.6-sol": (5.00, 0.50, 30.00),
}
PRICES_NOTE = "Defaults: OpenAI list prices after the 30 July 2026 change (cached input = 10% of input)."


@dataclass
class Meter:
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, input_tokens: int, cached: int, output: int) -> None:
        with self._lock:
            m = self.by_model.setdefault(model, {"calls": 0, "input": 0, "cached": 0, "output": 0})
            m["calls"] += 1
            m["input"] += int(input_tokens or 0)
            m["cached"] += int(cached or 0)
            m["output"] += int(output or 0)

    def reset(self) -> None:
        with self._lock:
            self.by_model.clear()

    def rows(self, prices: dict[str, tuple[float, float, float]]) -> list[dict]:
        """One row per model with its cost in USD (None when the model has no price)."""
        out = []
        with self._lock:
            items = [(k, dict(v)) for k, v in self.by_model.items()]
        for model, m in items:
            p = price_for(model, prices)
            cost = None
            if p is not None:
                fresh = max(m["input"] - m["cached"], 0)
                cost = (fresh * p[0] + m["cached"] * p[1] + m["output"] * p[2]) / 1e6
            out.append({"model": model, **m, "cost": cost})
        return out

    def total(self, prices) -> tuple[float, list[str]]:
        """Total USD over priced models, and the models without a price."""
        rows = self.rows(prices)
        return (sum(r["cost"] for r in rows if r["cost"] is not None),
                [r["model"] for r in rows if r["cost"] is None])


def price_for(model: str, prices: dict) -> tuple[float, float, float] | None:
    """Exact name first, then the longest price key the model name starts with
    (e.g. a dated snapshot 'gpt-5.6-luna-2026-07-30' uses 'gpt-5.6-luna')."""
    if model in prices:
        return tuple(prices[model])
    keys = sorted((k for k in prices if model.startswith(k)), key=len, reverse=True)
    return tuple(prices[keys[0]]) if keys else None


_current: contextvars.ContextVar[Meter | None] = contextvars.ContextVar("usage_meter", default=None)


@contextmanager
def metering(meter: Meter | None):
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def in_context(fn):
    """Wrap fn so it runs with the caller's meter (and other context) in a worker thread."""
    ctx = contextvars.copy_context()
    return lambda *a, **k: ctx.copy().run(fn, *a, **k)


def record(model: str, usage) -> None:
    """Add a response's usage (Responses or Chat Completions shape) to the current meter."""
    meter = _current.get()
    if meter is None or usage is None:
        return
    inp = getattr(usage, "input_tokens", None)
    if inp is not None:                                  # Responses API
        details = getattr(usage, "input_tokens_details", None)
        cached, out = getattr(details, "cached_tokens", 0) or 0, getattr(usage, "output_tokens", 0)
    else:                                                # Chat Completions
        inp = getattr(usage, "prompt_tokens", 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached, out = getattr(details, "cached_tokens", 0) or 0, getattr(usage, "completion_tokens", 0)
    meter.add(model, inp or 0, cached, out or 0)
