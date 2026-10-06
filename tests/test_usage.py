"""Token usage metering and session cost."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ask import preferences, usage
from ask.agent.llm_call import Conversation
from tests.conftest import say

PRICES = {"gpt-5.6-luna": (0.20, 0.02, 1.20), "gpt-5.6-sol": (5.00, 0.50, 30.00)}


def test_cost_uses_fresh_cached_and_output_prices():
    m = usage.Meter()
    m.add("gpt-5.6-luna", 1_000_000, 200_000, 100_000)       # 800k fresh, 200k cached, 100k out
    m.add("gpt-5.6-sol", 10_000, 0, 1_000)
    total, unpriced = m.total(PRICES)
    luna = (800_000 * 0.20 + 200_000 * 0.02 + 100_000 * 1.20) / 1e6
    sol = (10_000 * 5.00 + 1_000 * 30.00) / 1e6
    assert total == pytest.approx(luna + sol) and unpriced == []
    m.add("my-azure-deployment", 50, 0, 5)
    total2, unpriced2 = m.total(PRICES)
    assert total2 == pytest.approx(total) and unpriced2 == ["my-azure-deployment"]


def test_dated_model_names_use_the_base_price():
    assert usage.price_for("gpt-5.6-luna-2026-07-30", PRICES) == PRICES["gpt-5.6-luna"]
    assert usage.price_for("gpt-4.1", PRICES) is None


def test_record_reads_both_api_shapes():
    m = usage.Meter()
    with usage.metering(m):
        usage.record("a", SimpleNamespace(input_tokens=100, output_tokens=7,
                                          input_tokens_details=SimpleNamespace(cached_tokens=40)))
        usage.record("b", SimpleNamespace(prompt_tokens=30, completion_tokens=3, prompt_tokens_details=None))
    assert m.by_model == {"a": {"calls": 1, "input": 100, "cached": 40, "output": 7},
                          "b": {"calls": 1, "input": 30, "cached": 0, "output": 3}}
    usage.record("a", SimpleNamespace(input_tokens=1, output_tokens=1))    # no meter: ignored
    assert m.by_model["a"]["calls"] == 1


def test_worker_threads_report_to_the_session_meter():
    m = usage.Meter()

    def work(i):
        usage.record("x", SimpleNamespace(input_tokens=10, output_tokens=1, input_tokens_details=None))
        return i
    with usage.metering(m), ThreadPoolExecutor(4) as pool:
        list(pool.map(usage.in_context(work), range(8)))
    assert m.by_model["x"] == {"calls": 8, "input": 80, "cached": 0, "output": 8}


def test_agent_calls_are_metered(fake_llm):
    r = say("hi")
    r.usage = SimpleNamespace(input_tokens=1200, output_tokens=80, input_tokens_details=SimpleNamespace(cached_tokens=0))
    fake_llm.script = [r]
    m = usage.Meter()
    with usage.metering(m):
        Conversation("S", [], "q", "gpt-5.6-luna").step(None)
    assert m.by_model["gpt-5.6-luna"]["input"] == 1200


def test_limits_per_user_and_saved_prices():
    preferences.set_cost_limit("alice", 2.5)
    assert preferences.cost_limit("alice") == 2.5 and preferences.cost_limit("bob") == 0.0
    preferences.set_prices({"gpt-5.6-luna": (0.3, 0.03, 1.5), "custom": (1, 0.1, 2)})
    p = preferences.prices()
    assert p["gpt-5.6-luna"] == (0.3, 0.03, 1.5) and p["custom"] == (1.0, 0.1, 2.0)
    assert p["gpt-5.6-sol"] == usage.DEFAULT_PRICES["gpt-5.6-sol"]        # defaults kept
