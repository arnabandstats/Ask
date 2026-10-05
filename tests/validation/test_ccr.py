"""Counterparty credit risk tests (ask/validation/t_ccr.py): deterministic toy profiles, hand calculations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="ccr", **kw)


def _ok(res):
    assert res.status == "ok", res.error
    return res


@pytest.fixture
def toy_wide():
    # EE = [1, 3, 2, 4, 50] at t = 0.25 … 1, 2; ENE = [-1, -2, 0, -3, 0]
    return pd.DataFrame({"0.25": [2.0, -2.0], "0.5": [6.0, -4.0], "0.75": [4.0, 0.0], "1.0": [8.0, -6.0],
                         "2.0": [100.0, 0.0]})


def test_exposure_profile_deterministic_toy(toy_wide):
    res = _ok(run_test("ccr.exposure_profile", _ctx(toy_wide), {"time_columns": list(toy_wide.columns)}))
    s = res.summary
    assert s["EPE"] == pytest.approx((1 + 3 + 2 + 4) / 4)              # 2.5
    assert s["effective_EPE"] == pytest.approx((1 + 3 + 3 + 4) / 4)    # 2.75
    assert s["EAD_alpha_EEPE"] == pytest.approx(1.4 * 2.75)
    prof = res.tables["Exposure profile"]
    assert prof["EE"].tolist() == [1, 3, 2, 4, 50]
    assert prof["effective_EE"].tolist() == [1, 3, 3, 4, 50]
    assert prof["ENE"].tolist() == [-1, -2, 0, -3, 0]
    assert prof["weight_in_EPE"].tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25, 0.0])
    assert res.figures
    # the same paths in the long layout give the same numbers
    long = toy_wide.reset_index(names="path").melt(id_vars="path", var_name="t", value_name="v")
    long["t"] = long["t"].astype(float)
    r2 = _ok(run_test("ccr.exposure_profile", _ctx(long), {"mtm": "v", "time": "t", "path": "path"}))
    assert r2.summary == s


def test_exposure_profile_short_horizon_and_truncation(toy_wide):
    res = _ok(run_test("ccr.exposure_profile", _ctx(toy_wide), {"time_columns": ["0.25", "0.5"]}))
    assert res.summary["EPE"] == pytest.approx(2.0) and any("before the" in n for n in res.notes)
    # grid that straddles the horizon: weight of the straddling point is truncated at 1y
    w = pd.DataFrame({"a": [1.0], "b": [3.0]}).pipe(lambda d: pd.concat([d, d * 0], ignore_index=True))
    r = _ok(run_test("ccr.exposure_profile", _ctx(w), {"time_columns": ["a", "b"], "times": [0.6, 1.5]}))
    assert r.summary["EPE"] == pytest.approx((0.5 * 0.6 + 1.5 * 0.4) / 1.0)
    bad = run_test("ccr.exposure_profile", _ctx(w), {"time_columns": ["a", "b"]})
    assert bad.status == "error" and "times" in bad.error


def test_cva_flat_profile_closed_form():
    prof = pd.DataFrame({"t": [0.0, 1, 2, 3, 4, 5], "ee": 10.0})
    exp = 0.6 * 10 * (1 - math.exp(-0.1))
    for p in ({"hazard_rate": 0.02}, {"cds_spread": 0.012}, {"spreads": [0.012, 0.012], "spread_tenors": [1, 5]}):
        res = _ok(run_test("ccr.cva", _ctx(prof), {"ee": "ee", "time": "t", "lgd": 0.6} | p))
        assert res.summary["CVA"] == pytest.approx(exp, rel=1e-12)
    # discounted, right-point rule, time 0 not on the grid
    prof2 = pd.DataFrame({"t": [1.0, 2.0], "ee": [5.0, 8.0]})
    r = _ok(run_test("ccr.cva", _ctx(prof2), {"ee": "ee", "time": "t", "hazard_rate": 0.03, "discount_rate": 0.02,
                                              "integration": "right_point", "lgd": 0.4}))
    S = lambda t: math.exp(-0.03 * t)                   # noqa: E731
    D = lambda t: math.exp(-0.02 * t)                   # noqa: E731
    hand = 0.4 * ((1 - S(1)) * 5 * D(1) + (S(1) - S(2)) * 8 * D(2))
    assert r.summary["CVA"] == pytest.approx(hand, rel=1e-12)
    assert run_test("ccr.cva", _ctx(prof), {"ee": "ee", "time": "t"}).status == "error"   # no credit input


def test_cva_from_paths_with_dva(toy_wide):
    res = _ok(run_test("ccr.cva", _ctx(toy_wide), {"time_columns": list(toy_wide.columns), "hazard_rate": 0.05,
                                                   "own_hazard_rate": 0.01}))
    assert res.summary["CVA"] > 0 and res.summary["DVA"] > 0
    assert res.summary["BCVA"] == pytest.approx(res.summary["CVA"] - res.summary["DVA"])


def test_mpor_effect_hand_example():
    rows = []
    for path, V, C in ((0, [1, 3, 2], [1, 3, 2]), (1, [-1, 2, 5], [0, 2, 5])):
        for t, v, c in zip((0.04, 0.08, 0.12), V, C):
            rows.append({"path": path, "t": t, "v": float(v), "c": float(c)})
    res = _ok(run_test("ccr.mpor_effect", _ctx(pd.DataFrame(rows)),
                       {"mtm": "v", "time": "t", "path": "path", "collateral": "c", "mpor": 0.04}))
    prof = res.tables["EE profiles"]
    assert prof["EE_uncollateralised"].tolist() == pytest.approx([0.5, 2.5, 3.5])
    assert prof["EE_collateral_no_lag"].tolist() == pytest.approx([0, 0, 0])
    assert prof["EE_collateral_MPOR_lag"].tolist() == pytest.approx([0.5, 2.0, 1.5])
    assert res.summary["EEPE_collateral_MPOR_lag"] == pytest.approx((0.5 + 2 + 2) / 3)


def test_exposure_backtest_from_forecasts_and_pits():
    ids = np.arange(12)
    actual = np.tile([2.5, 2.0, 0.5, 4.5], 3)
    sims = pd.DataFrame({"id": np.repeat(ids, 4), "x": np.tile([1.0, 2.0, 3.0, 4.0], 12)})
    df = pd.DataFrame({"id": ids, "a": actual})
    res = _ok(run_test("ccr.exposure_backtest", RunContext(df=df, tables={"fc": sims}),
                       {"actual": "a", "id": "id", "other": "fc", "forecast_column": "x", "n_sims": 200}))
    u = np.tile([0.5, 0.375, 0.0, 1.0], 3)            # mid-rank PITs, clipped at 0/1
    assert res.summary["mean_pit"] == pytest.approx(np.clip(u, 1e-6, 1 - 1e-6).mean())
    assert "Berkowitz LR_3" in res.tables["PIT tests"]["test"].tolist()
    pit = pd.DataFrame({"u": np.random.default_rng(3).random(300)})
    r = _ok(run_test("ccr.exposure_backtest", _ctx(pit), {"pit": "u", "n_sims": 300}))
    assert (r.tables["PIT tests"]["p_value"] > 0.001).all()
    again = _ok(run_test("ccr.exposure_backtest", _ctx(pit.copy()), {"pit": "u", "n_sims": 300}))
    assert again.summary == r.summary
    assert run_test("ccr.exposure_backtest", _ctx(pit), {}).status == "error"


def test_wrong_way_risk_detects_dependence():
    rng = np.random.default_rng(5)
    n = 2000
    credit = rng.standard_normal(n)
    df = pd.DataFrame({"mtm": 1 + credit + 0.5 * rng.standard_normal(n), "hz": credit,
                       "t": np.repeat([0.5, 1.0], n // 2)})
    res = _ok(run_test("ccr.wrong_way_risk", _ctx(df), {"mtm": "mtm", "credit": "hz", "time": "t"}))
    assert res.summary["spearman"] > 0.5 and res.summary["spearman_p"] < 1e-10
    assert res.summary["conditional_EE_ratio"] > 1.5
    assert len(res.tables["By time point"]) == 2
    flat = df.assign(hz=1.0)
    assert run_test("ccr.wrong_way_risk", _ctx(flat), {"mtm": "mtm", "credit": "hz"}).status == "not_applicable"


def test_netting_check_hand_example():
    df = pd.DataFrame({"set": ["A", "A", "B", "B", "C"], "v": [5.0, -3.0, 4.0, 1.0, np.nan],
                       "rep": [2.0, 2.0, 6.0, 6.0, 1.0]})
    res = _ok(run_test("ccr.netting_check", _ctx(df), {"mtm": "v", "netting_set": "set", "reported": "rep"}))
    s = res.summary
    assert s["net_exposure"] == pytest.approx(7.0) and s["gross_exposure"] == pytest.approx(10.0)
    assert s["net_to_gross_ratio"] == pytest.approx(0.7)
    assert s["breaks"] == 1 and s["max_abs_difference"] == pytest.approx(1.0)
    assert res.rows_used == 4 and res.notes


def test_sa_ccr_bcbs_example_inputs():
    # trades of the BCBS 279 interest-rate example (netting set 1)
    df = pd.DataFrame({"N": [10000.0, 10000.0, 5000.0], "S": [0.0, 0.0, 1.0], "E": [10.0, 4.0, 11.0],
                       "M": [10.0, 4.0, 1.0], "delta": [-1.0, 1.0, -0.27], "v": [30.0, -20.0, 50.0]})
    res = _ok(run_test("ccr.sa_ccr_ir", _ctx(df), {"notional": "N", "start": "S", "end": "E", "maturity": "M",
                                                   "delta": "delta", "mtm": "v"}))
    sd = lambda s, e: (math.exp(-0.05 * s) - math.exp(-0.05 * e)) / 0.05    # noqa: E731
    D2 = 10000 * sd(0, 4)
    D3 = -10000 * sd(0, 10) - 0.27 * 5000 * sd(1, 11)
    addon = 0.005 * math.sqrt(D2 ** 2 + D3 ** 2 + 1.4 * D2 * D3)
    assert res.summary["AddOn_IR"] == pytest.approx(addon, rel=1e-12)
    assert res.summary["AddOn_IR"] == pytest.approx(342.5, abs=0.1)
    assert res.summary["RC"] == pytest.approx(60.0) and res.summary["multiplier"] == 1.0
    assert res.summary["EAD"] == pytest.approx(1.4 * (60 + addon))
    # out-of-the-money netting set: multiplier below 1
    r = _ok(run_test("ccr.sa_ccr_ir", _ctx(df.assign(v=[-300.0, -100.0, 0.0])),
                     {"notional": "N", "start": "S", "end": "E", "delta": "delta", "mtm": "v"}))
    assert r.summary["multiplier"] == pytest.approx(min(1, 0.05 + 0.95 * math.exp(-400 / (2 * 0.95 * r.summary["AddOn_IR"]))))
    assert r.summary["RC"] == 0.0
