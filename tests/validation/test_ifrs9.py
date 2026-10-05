"""IFRS 9 tests (t_ifrs9.py): known-answer checks on small hand-computed portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="ifrs9", **kw)


def _ok(res):
    assert res.status == "ok", res.error
    return res


E12 = 0.1 * 0.5 * 100 / 1.1                  # 4.5454...
ELT = E12 + 0.2 * 0.5 * 100 / 1.1 ** 2       # 12.8099...


@pytest.fixture
def ecl_df():
    return pd.DataFrame({"acc": ["A", "B", "C"], "stage": [1, "Stage 2", 3], "m1": [0.1] * 3, "m2": [0.2] * 3,
                         "c1": [0.1] * 3, "c2": [0.3] * 3, "h1": [0.1] * 3, "h2": [0.2 / 0.9] * 3,
                         "lgd": [0.5] * 3, "ead": [100.0] * 3, "eir": [0.1] * 3,
                         "reported": [E12, 13.0, 50.0]})


def test_ecl_recompute_hand_all_pd_types(ecl_df):
    for cols, kind in ((["m1", "m2"], "marginal"), (["c1", "c2"], "cumulative"), (["h1", "h2"], "conditional")):
        r = _ok(run_test("ifrs9.ecl_recompute", _ctx(ecl_df),
                         {"id": "acc", "ecl": "reported", "stage": "stage", "pd_columns": cols, "pd_type": kind,
                          "lgd": "lgd", "ead": "ead", "eir": "eir"}))
        t = r.tables["ECL per account"].set_index("id")
        assert t.loc["A", "ecl_12m"] == pytest.approx(E12)
        assert t.loc["B", "ecl_lifetime"] == pytest.approx(ELT)
        assert t.loc["C", "ecl_recomputed"] == pytest.approx(50.0)
        assert r.summary["difference"] == pytest.approx(ELT - 13.0)
        assert r.summary["n_mismatch"] == 1
    rec = r.tables["Reconciliation"]
    assert list(rec["stage"]) == ["1", "2", "3", "ALL"]


def test_ecl_recompute_long_table_and_horizon(ecl_df):
    ts = pd.DataFrame({"acc": ["A", "A", "B", "B", "C", "C"], "period": [1, 2] * 3, "pd": [0.1, 0.2] * 3})
    r = _ok(run_test("ifrs9.ecl_recompute", _ctx(ecl_df, tables={"ts": ts}),
                     {"id": "acc", "ecl": "reported", "horizon": "lifetime", "term_structure": "ts",
                      "lgd": "lgd", "ead": "ead", "eir_rate": 0.1}))
    assert r.summary["total_recomputed"] == pytest.approx(3 * ELT)
    # quarterly periods: 12-month ECL = first 4 periods
    q = pd.DataFrame({"acc": ["A"], "p1": [0.01], "p2": [0.01], "p3": [0.01], "p4": [0.01], "p5": [0.5],
                      "lgd": [1.0], "ead": [100.0], "rep": [4.0]})
    r = _ok(run_test("ifrs9.ecl_recompute", _ctx(q), {"id": "acc", "ecl": "rep", "horizon": "12m",
                                                      "pd_columns": ["p1", "p2", "p3", "p4", "p5"], "lgd": "lgd",
                                                      "ead": "ead", "period_length": 0.25}))
    assert r.summary["total_recomputed"] == pytest.approx(4.0)
    bad = run_test("ifrs9.ecl_recompute", _ctx(q), {"id": "acc", "ecl": "rep", "pd_columns": ["p1"], "lgd": "lgd",
                                                    "ead": "ead"})
    assert bad.status == "error" and "stage" in bad.error


@pytest.fixture
def scen():
    return pd.DataFrame({"acc": ["A", "A", "A", "B", "B", "B"], "sc": ["base", "up", "down"] * 2,
                         "ecl": [10.0, 8.0, 20.0, 5.0, 4.0, 9.0], "rep": [11.6, 11.6, 11.6, 6.0, 6.0, 6.0]})


def test_scenario_weighted_ecl_hand(scen):
    w = {"base": 0.5, "up": 0.2, "down": 0.3}
    r = _ok(run_test("ifrs9.scenario_weighted_ecl", _ctx(scen), {"id": "acc", "scenario": "sc", "ecl": "ecl",
                                                                 "weights": w, "reported": "rep",
                                                                 "base_scenario": "base"}))
    # A: 5 + 1.6 + 6 = 12.6 ; B: 2.5 + 0.8 + 2.7 = 6.0
    assert r.summary["total_weighted"] == pytest.approx(18.6)
    assert r.summary["difference"] == pytest.approx(1.0) and r.summary["n_mismatch"] == 1
    assert r.summary["weighted_to_base_ratio"] == pytest.approx(18.6 / 15)
    bad = run_test("ifrs9.scenario_weighted_ecl", _ctx(scen), {"id": "acc", "scenario": "sc", "ecl": "ecl",
                                                               "weights": {"base": 0.5, "up": 0.2, "down": 0.2}})
    assert bad.status == "error" and "sum to 1" in bad.error


def test_scenario_weight_sensitivity_hand(scen):
    r = _ok(run_test("ifrs9.scenario_weight_sensitivity", _ctx(scen),
                     {"id": "acc", "scenario": "sc", "ecl": "ecl", "weights": {"base": 0.5, "up": 0.2, "down": 0.3},
                      "alternative_weights": {"severe": {"base": 0.4, "up": 0.1, "down": 0.5}},
                      "base_scenario": "base", "shift": 0.1}))
    t = r.tables["ECL by weight set"].set_index("weight_set")
    assert t.loc["severe", "total_ecl"] == pytest.approx(0.4 * 15 + 0.1 * 12 + 0.5 * 29)
    assert t.loc["100% down", "total_ecl"] == pytest.approx(29)
    sh = r.tables["Weight shift from base"].set_index("to_scenario")
    assert sh.loc["down", "change_in_total_ecl"] == pytest.approx(0.1 * (29 - 15))
    assert r.summary["max_single_scenario"] == pytest.approx(29)


@pytest.fixture
def tm():
    return pd.DataFrame({"from": ["A", "B"], "A": [0.9, 0.1], "B": [0.08, 0.8], "D": [0.02, 0.1]})


def test_markov_lifetime_pd(tm):
    acc = pd.DataFrame({"g": ["A", "A", "B"], "pd12": [0.04, 0.06, 0.1]})
    r = _ok(run_test("ifrs9.lifetime_pd_markov", _ctx(acc, tables={"tm": tm}), {"transition": "tm", "horizon": 3,
                                                                                "grade": "g"}))
    M = np.array([[0.9, 0.08, 0.02], [0.1, 0.8, 0.1], [0, 0, 1]])
    w = r.tables["Cumulative PD by grade"].set_index("grade")
    assert w.loc["A", "cpd_2"] == pytest.approx(0.046)
    assert w.loc["B", "cpd_3"] == pytest.approx(np.linalg.matrix_power(M, 3)[1, 2])
    port = r.tables["Portfolio cumulative PD"]
    assert port["cumulative_pd"].iloc[0] == pytest.approx((2 * 0.02 + 0.1) / 3)
    # anchored on the 12-month PD (mean 0.05 in grade A)
    r = _ok(run_test("ifrs9.lifetime_pd_markov", _ctx(acc, tables={"tm": tm}),
                     {"transition": "tm", "horizon": 2, "grade": "g", "pd": "pd12"}))
    w = r.tables["Cumulative PD by grade"].set_index("grade")
    v1 = np.array([0.95 * 0.9 / 0.98, 0.95 * 0.08 / 0.98, 0.05])
    assert w.loc["A", "cpd_1"] == pytest.approx(0.05)
    assert w.loc["A", "cpd_2"] == pytest.approx((v1 @ M)[2])
    ts = r.tables["Term structure (long)"]
    row = ts[(ts["grade"] == "A") & (ts["horizon"] == 2)].iloc[0]
    assert row["conditional_pd"] == pytest.approx(row["marginal_pd"] / 0.95)
    bad = tm.assign(D=[0.5, 0.1])
    assert run_test("ifrs9.lifetime_pd_markov", _ctx(acc, tables={"tm": bad}), {"transition": "tm"}).status == "error"


def test_lifetime_pd_term_structure():
    df = pd.DataFrame({"c1": [0.1, 0.3], "c2": [0.3, 0.2], "seg": ["x", "y"]})
    r = _ok(run_test("ifrs9.lifetime_pd_term_structure", _ctx(df), {"pd_columns": ["c1", "c2"],
                                                                    "pd_type": "cumulative", "segment": "seg"}))
    t = r.tables["Average term structure"]
    x = t[t["segment"] == "x"].set_index("horizon")
    assert x.loc[2, "mean_marginal_pd"] == pytest.approx(0.2)
    assert x.loc[2, "conditional_pd_of_mean_curve"] == pytest.approx(0.2 / 0.9)
    assert r.summary["n_negative_marginal"] == 1


def test_km_backtest_hand():
    df = pd.DataFrame({"t": [1, 2, 2, 3, 4], "e": [1, 1, 0, 1, 0], "p2": [0.5] * 5, "p5": [0.9] * 5})
    r = _ok(run_test("ifrs9.km_lifetime_pd_backtest", _ctx(df), {"duration": "t", "target": "e",
                                                                 "pd_columns": ["p2", "p5"], "horizons": [2, 5]}))
    t = r.tables["Cumulative default by vintage"].set_index("horizon")
    assert t.loc[2, "km_cumulative_pd"] == pytest.approx(0.4)
    se = 0.6 * np.sqrt(1 / 20 + 1 / 12)
    assert t.loc[2, "se"] == pytest.approx(se)
    assert t.loc[2, "z"] == pytest.approx(0.1 / se)
    assert np.isnan(t.loc[5, "km_cumulative_pd"])          # beyond follow-up
    assert r.summary["km_cpd_2"] == pytest.approx(0.4)


@pytest.fixture
def staging():
    return pd.DataFrame({
        "acc": list("ABCDEFG"),
        "rep": [1, 2, 2, 1, 3, 1, 2],
        "pd": [0.01, 0.06, 0.004, 0.02, 0.5, 0.002, 0.03],
        "pd0": [0.01, 0.02, 0.001, 0.02, 0.1, 0.0005, 0.03],
        "dpd": [0, 0, 0, 45, 120, 0, 0],
        "wl": [0, 0, 0, 0, 0, 0, 1]})


def test_staging_replication_hand(staging):
    r = _ok(run_test("ifrs9.staging_replication", _ctx(staging),
                     {"id": "acc", "stage": "rep", "pd": "pd", "pd_origination": "pd0", "relative_multiple": 2.0,
                      "low_credit_risk_pd": 0.003, "dpd": "dpd", "watchlist": "wl"}))
    # A1 B2(pd x3) C2(x4, above LCR) D2(dpd 45, reported 1 -> mismatch) E3 F1(x4 but LCR) G2(watchlist)
    s = r.summary
    assert s["n_mismatch"] == 1 and s["agreement_rate"] == pytest.approx(6 / 7)
    m = r.tables["Mismatches"]
    assert list(m["id"]) == ["D"] and m["triggers"].iloc[0] == "dpd_backstop"
    cm = r.tables["Reported vs replicated stage"].set_index("reported")
    assert cm.loc[1, 2] == 1 and cm.loc[3, 3] == 1
    po, n = 6 / 7, 7
    rows, cols = np.array([3, 3, 1]), np.array([2, 4, 1])
    pe = (rows * cols).sum() / n ** 2
    assert s["cohen_kappa"] == pytest.approx((po - pe) / (1 - pe))
    trig = r.tables["Triggers"].set_index("trigger")
    assert trig.loc["low_credit_risk_exempt", "n_fired"] == 1
    # with the 'and' rule and an absolute threshold, C no longer qualifies
    r2 = _ok(run_test("ifrs9.staging_replication", _ctx(staging),
                      {"id": "acc", "stage": "rep", "pd": "pd", "pd_origination": "pd0", "relative_multiple": 2.0,
                       "absolute_change": 0.01, "combine": "and", "dpd": "dpd", "watchlist": "wl"}))
    assert set(r2.tables["Mismatches"]["id"]) == {"C", "D"}


def test_stage_migration_long_and_wide():
    df = pd.DataFrame({"acc": ["A", "B", "C", "D", "A", "B", "C", "E"], "dt": ["2023"] * 4 + ["2024"] * 4,
                       "st": [1, 1, 2, 1, 2, 1, 1, 1], "x": [100, 100, 50, 10, 0, 0, 0, 0]})
    r = _ok(run_test("ifrs9.stage_migration", _ctx(df), {"id": "acc", "stage": "st", "period": "dt", "ead": "x"}))
    s = r.summary
    assert s["n_matched"] == 3 and s["n_exited"] == 1 and s["n_new"] == 1
    assert s["share_stage1_to_2"] == pytest.approx(0.5) and s["share_stage2_to_1"] == pytest.approx(1.0)
    assert s["share_worse_stage"] == pytest.approx(1 / 3)
    wide = pd.DataFrame({"s0": [1, 1, 2], "s1": [2, 1, 1], "acc": ["A", "B", "C"]})
    r2 = _ok(run_test("ifrs9.stage_migration", _ctx(wide), {"id": "acc", "stage": "s0", "stage_to": "s1"}))
    assert r2.summary["share_same_stage"] == pytest.approx(1 / 3)


def test_coverage_by_stage_hand():
    df = pd.DataFrame({"st": [1, 1, 2, 3, 1, 2], "x": [100, 100, 50, 50, 200, 100], "ecl": [1, 1, 5, 25, 2, 8],
                       "q": ["2023Q4"] * 4 + ["2024Q1"] * 2})
    r = _ok(run_test("ifrs9.coverage_by_stage", _ctx(df), {"stage": "st", "ead": "x", "ecl": "ecl", "period": "q"}))
    s = r.summary
    assert s["period"] == "2024Q1"
    assert s["stage2_share_of_exposure"] == pytest.approx(1 / 3) and s["stage2_coverage"] == pytest.approx(0.08)
    t = r.tables["Coverage by stage"]
    q4 = t[(t["period"] == "2023Q4") & (t["stage"] == "3")].iloc[0]
    assert q4["coverage_ratio"] == pytest.approx(0.5) and q4["share_of_exposure"] == pytest.approx(50 / 300)


@pytest.fixture
def panel():
    rng = np.random.default_rng(5)
    macro = np.array([1.0, 2.0, -1.0, -2.0, 0.5, 3.0])
    rows = []
    for t, m in enumerate(macro):
        p = 0.02 * np.exp(-0.3 * m) * np.ones(500)
        rows.append(pd.DataFrame({"per": 2015 + t, "gdp": m, "pd": p, "y": (rng.random(500) < p).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def test_pit_macro_correlation(panel):
    r = _ok(run_test("ifrs9.pit_macro_correlation", _ctx(panel), {"pd": "pd", "period": "per", "feature": "gdp",
                                                                  "target": "y", "max_lag": 1}))
    agg = panel.groupby("per").agg(pd=("pd", "mean"), gdp=("gdp", "mean"))
    assert r.summary["corr_pd_macro"] == pytest.approx(stats.pearsonr(agg["pd"], agg["gdp"]).statistic)
    assert r.summary["corr_pd_macro"] < -0.9
    assert len(r.tables["Lagged correlations"]) == 3
    short = panel[panel["per"] < 2018]
    assert run_test("ifrs9.pit_macro_correlation", _ctx(short), {"pd": "pd", "period": "per",
                                                                 "feature": "gdp"}).status == "not_applicable"


def test_pit_calibration_backtest_hand():
    df = pd.DataFrame({"y": [1, 0, 0, 0, 1, 1, 0, 0], "pd": [0.25] * 8, "per": [1] * 4 + [2] * 4})
    r = _ok(run_test("ifrs9.pit_calibration_backtest", _ctx(df), {"target": "y", "pd": "pd", "period": "per"}))
    t = r.tables["Calibration by period"].set_index("period")
    assert t.loc["2", "binomial_p_two_sided"] == pytest.approx(stats.binomtest(2, 4, 0.25).pvalue)
    assert t.loc["2", "binomial_p_underestimation"] == pytest.approx(stats.binom.sf(1, 4, 0.25))
    z2 = (2 - 1) / np.sqrt(4 * 0.25 * 0.75)
    assert r.summary["chi2"] == pytest.approx(z2 ** 2)
    assert r.summary["chi2_p"] == pytest.approx(stats.chi2.sf(z2 ** 2, 2))


def test_determinism(ecl_df):
    p = {"id": "acc", "ecl": "reported", "stage": "stage", "pd_columns": ["m1", "m2"], "lgd": "lgd", "ead": "ead",
         "eir": "eir"}
    a = run_test("ifrs9.ecl_recompute", _ctx(ecl_df), p)
    b = run_test("ifrs9.ecl_recompute", _ctx(ecl_df.copy()), p)
    assert a.run_id == b.run_id and a.summary == b.summary
