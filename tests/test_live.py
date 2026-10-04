"""Optional end-to-end checks against the real model API.

Skipped by default (they cost tokens). Run with:
    ASK_LIVE_TESTS=1 pytest -m live
"""
from __future__ import annotations

import os

import pytest

from ask import config
from ask.agent import router

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.getenv("ASK_LIVE_TESTS") != "1",
                                 reason="set ASK_LIVE_TESTS=1 to call the real API")]


@pytest.mark.parametrize("model,effort", [(config.DEFAULT_MODEL, None),
                                          (config.DEEP_MODEL, config.DEEP_REASONING_EFFORT)])
def test_grounded_repo_answer(repo_registry, tmp_path, model, effort):
    turn = router.answer("What accuracy threshold does the quality gate use, and where is it defined?",
                         repo_registry, [], model=model, verify=True, output_dir=tmp_path,
                         status=lambda m: None, reasoning_effort=effort)
    v = turn.meta["verification"]
    assert "0.85" in turn.content
    assert v["citations"] and v["problems"] == [], v["problems"]
    assert any(c["file"] == "src/train.py" for c in v["citations"])


def test_data_question_computes_the_number(portfolio_csv, portfolio_df, tmp_path):
    from ask.sources.registry import SourceRegistry
    reg = SourceRegistry()
    reg.load(portfolio_csv)
    turn = router.answer("What is the default rate in segment B? Give it to 3 decimals.", reg, [],
                         model=config.DEFAULT_MODEL, verify=True, output_dir=tmp_path,
                         status=lambda m: None)
    expected = portfolio_df.loc[portfolio_df["segment"] == "B", "default_flag"].mean()
    assert f"{expected:.3f}" in turn.content or f"{expected * 100:.1f}%" in turn.content
    assert "query_data" in turn.meta["tools"]
