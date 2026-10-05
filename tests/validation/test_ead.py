"""EAD / CCF tests (t_ead.py): known-answer checks on small hand-computed portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ask.validation import t_lgd
from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="ead", **kw)


def _ok(res):
    assert res.status == "ok", res.error
    return res


@pytest.fixture
def hand():
    # CCFs: 0.5, 1.4, -0.125; the 4th is overdrawn (no undrawn amount) and excluded
    return pd.DataFrame({"ead": [60.0, 120.0, 10.0, 80.0], "limit": [100.0, 100.0, 100.0, 50.0],
                         "drawn": [20.0, 50.0, 20.0, 60.0], "est": [0.6, 0.9, 0.2, 0.5],
                         "pool": ["a", "b", "a", "b"]})


@pytest.fixture
def port():
    rng = np.random.default_rng(11)
    n = 300
    lim = rng.uniform(1000, 5000, n)
    drawn = lim * rng.uniform(0, 0.95, n)
    est = rng.uniform(0.2, 0.9, n).round(2)
    ccf = np.clip(est + rng.normal(0, 0.3, n), -0.2, 1.3)
    return pd.DataFrame({"ead": drawn + ccf * (lim - drawn), "limit": lim, "drawn": drawn, "est": est,
                         "pool": np.where(est < 0.5, "low", "high"), "pred_ead": drawn + est * (lim - drawn)})


def test_realised_ccf_hand(hand):
    r = _ok(run_test("ead.realised_ccf", _ctx(hand), {"ead": "ead", "limit": "limit", "drawn": "drawn"}))
    s = r.summary
    assert s["n_ccf"] == 3 and s["n_excluded_no_undrawn"] == 1 and s["n_overdrawn"] == 1
    assert s["mean_ccf"] == pytest.approx((0.5 + 1.4 - 0.125) / 3)
    assert s["undrawn_weighted_ccf"] == pytest.approx(100 / 210)
    assert s["n_ccf_below_0"] == 1 and s["n_ccf_above_1"] == 1
    assert r.tables["Realised CCF by segment"]["mean_ccf_floored_capped"].iloc[0] == pytest.approx(1.5 / 3)
    assert r.tables["Facilities without undrawn amount"]["n"].iloc[0] == 1


def test_ccf_backtest_matches_scipy(hand, port):
    r = _ok(run_test("ead.ccf_backtest", _ctx(hand), {"ead": "ead", "limit": "limit", "drawn": "drawn",
                                                      "predicted": "est"}))
    real = np.array([0.5, 1.4, -0.125])
    ref = stats.ttest_rel(real, [0.6, 0.9, 0.2], alternative="greater")
    assert r.summary["t_statistic"] == pytest.approx(ref.statistic) and r.summary["p_value"] == pytest.approx(ref.pvalue)
    port = port.assign(ccf=(port["ead"] - port["drawn"]) / (port["limit"] - port["drawn"]))
    a = _ok(run_test("ead.ccf_backtest", _ctx(port), {"actual": "ccf", "predicted": "est", "grade": "pool"}))
    b = _ok(run_test("ead.ccf_backtest", _ctx(port), {"ead": "ead", "limit": "limit", "drawn": "drawn",
                                                      "predicted": "est", "grade": "pool"}))
    assert a.summary["p_value"] == pytest.approx(b.summary["p_value"])
    assert run_test("ead.ccf_backtest", _ctx(port), {"predicted": "est"}).status == "error"


def test_coverage_ratio_hand():
    df = pd.DataFrame({"real": [100.0, 200.0, 300.0], "pred": [120.0, 150.0, 300.0]})
    r = _ok(run_test("ead.coverage_ratio", _ctx(df), {"ead": "real", "predicted": "pred"}))
    assert r.summary["coverage_ratio"] == pytest.approx(570 / 600)
    assert r.summary["share_under_predicted"] == pytest.approx(1 / 3)
    assert r.summary["median_ratio"] == pytest.approx(1.0)
    ref = stats.ttest_1samp(df["real"] - df["pred"], 0, alternative="greater")
    assert r.summary["p_value_underestimation"] == pytest.approx(ref.pvalue)


def test_ccf_distribution(hand):
    r = _ok(run_test("ead.ccf_distribution", _ctx(hand), {"ead": "ead", "limit": "limit", "drawn": "drawn",
                                                          "predicted": "est"}))
    assert r.summary["n"] == 3 and r.summary["share_above_1"] == pytest.approx(1 / 3)
    assert r.summary["share_below_0"] == pytest.approx(1 / 3)
    assert list(r.tables["Distribution summary"]["variable"]) == ["realised", "estimated"]


def test_ccf_ranking(port):
    r = _ok(run_test("ead.ccf_ranking", _ctx(port), {"ead": "ead", "limit": "limit", "drawn": "drawn",
                                                     "predicted": "est", "segment": "pool"}))
    ccf = (port["ead"] - port["drawn"]) / (port["limit"] - port["drawn"])
    assert r.summary["gAUC"] == pytest.approx(t_lgd.gauc(port["est"], ccf)["gAUC"])
    assert r.summary["gAUC"] == pytest.approx((1 + stats.somersd(ccf, port["est"]).statistic) / 2)
    assert r.summary["spearman_rho"] == pytest.approx(stats.spearmanr(port["est"], ccf).statistic)
    assert r.summary["gAUC_p_vs_0.5"] < 1e-6


def test_not_applicable_and_determinism(hand, port):
    full = hand.assign(drawn=hand["limit"])
    assert run_test("ead.realised_ccf", _ctx(full), {"ead": "ead", "limit": "limit", "drawn": "drawn"}).status \
        == "not_applicable"
    p = {"ead": "ead", "limit": "limit", "drawn": "drawn", "predicted": "est"}
    a, b = run_test("ead.ccf_ranking", _ctx(port), p), run_test("ead.ccf_ranking", _ctx(port.copy()), p)
    assert a.run_id == b.run_id and a.summary == b.summary
