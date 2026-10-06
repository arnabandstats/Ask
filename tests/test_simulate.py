"""Simulated data: the generator, the simulate_data tool, and restoring a chat that used it."""
from __future__ import annotations

import json

import numpy as np
import pytest

from ask.agent.tools import ToolContext, dispatch
from ask.analysis import simulate
from ask.sources.registry import SourceRegistry

NORMAL = [{"name": "x", "distribution": "normal", "params": {"mean": 0, "var": 1}}]


def test_normal_sample_matches_its_parameters():
    df = simulate.generate(100_000, [{"name": "x", "distribution": "normal", "params": {"mean": 5, "var": 4}}], 1)
    assert abs(df["x"].mean() - 5) < 0.03 and abs(df["x"].std() - 2) < 0.03


def test_same_seed_same_table_and_columns_are_independent_streams():
    a = simulate.generate(500, NORMAL, 7)
    assert a.equals(simulate.generate(500, NORMAL, 7)) and not a.equals(simulate.generate(500, NORMAL, 8))
    more = simulate.generate(500, NORMAL + [{"name": "y", "distribution": "uniform"}], 7)
    assert np.array_equal(more["x"], a["x"])                  # adding a column leaves x unchanged


def test_every_distribution_runs():
    cols = [{"name": "normal", "distribution": "normal"}, {"name": "uniform", "distribution": "uniform"},
            {"name": "lognormal", "distribution": "lognormal"},
            {"name": "exponential", "distribution": "exponential", "params": {"rate": 2}},
            {"name": "gamma", "distribution": "gamma", "params": {"shape": 2}},
            {"name": "beta", "distribution": "beta", "params": {"a": 2, "b": 5}},
            {"name": "t", "distribution": "t", "params": {"df": 5}},
            {"name": "chi2", "distribution": "chi2", "params": {"df": 3}},
            {"name": "poisson", "distribution": "poisson", "params": {"lam": 3}},
            {"name": "binomial", "distribution": "binomial", "params": {"trials": 10, "p": 0.3}},
            {"name": "default", "distribution": "bernoulli", "params": {"p": 0.02}},
            {"name": "grade", "distribution": "categorical", "params": {"categories": ["A", "B"], "probs": [0.7, 0.3]}}]
    df = simulate.generate(2000, cols, 3)
    assert list(df.columns) == [c["name"] for c in cols] and len(df) == 2000
    assert set(df["default"].unique()) <= {0, 1} and set(df["grade"]) == {"A", "B"}


@pytest.mark.parametrize("cols,msg", [
    ([{"name": "x", "distribution": "cauchy"}], "unknown distribution"),
    ([{"name": "x", "distribution": "normal", "params": {"mu": 0}}], "takes: mean"),
    ([{"name": "x", "distribution": "normal", "params": {"sd": 1, "var": 1}}], "sd or var"),
    ([{"name": "g", "distribution": "categorical", "params": {"categories": ["A"], "probs": [0.5]}}], "sum to 1"),
    (NORMAL * 2, "duplicate"),
])
def test_bad_specs_are_readable_errors(cols, msg):
    with pytest.raises(ValueError, match=msg):
        simulate.generate(10, cols, 1)


def test_tool_loads_the_table_and_a_histogram_follows(tmp_path):
    ctx = ToolContext(registry=SourceRegistry(), output_dir=tmp_path / "out")
    out = dispatch(ctx, "simulate_data", json.dumps({"n": 1000, "columns": NORMAL}))
    assert "SIMULATED, 1,000 rows" in out and "seed" in out and ctx.sources_changed
    assert len(ctx.registry.get("simulated", "data").df) == 1000
    chart = dispatch(ctx, "make_chart", json.dumps({"kind": "histogram", "x": "x"}))
    assert "created" in chart and ctx.artifacts[-1]["type"] == "plotly"


def test_bad_tool_arguments_come_back_as_an_error(tmp_path):
    ctx = ToolContext(registry=SourceRegistry(), output_dir=tmp_path / "out")
    out = dispatch(ctx, "simulate_data", json.dumps({"n": 10, "columns": [{"name": "x", "distribution": "zipf"}]}))
    assert out.startswith("Error:") and "unknown distribution" in out


def test_reopening_a_chat_regenerates_the_same_rows(tmp_path):
    ctx = ToolContext(registry=SourceRegistry(), output_dir=tmp_path / "out")
    dispatch(ctx, "simulate_data", json.dumps({"n": 300, "columns": NORMAL, "seed": 11, "name": "demo"}))
    records = json.loads(json.dumps(ctx.registry.records()))          # as saved with the chat
    fresh = SourceRegistry()
    assert fresh.restore(records) == []
    assert fresh.get("demo", "data").df.equals(ctx.registry.get("demo", "data").df)
