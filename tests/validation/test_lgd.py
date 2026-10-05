"""LGD tests (t_lgd.py): known-answer checks on small hand-computed portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from ask.validation import t_lgd
from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="lgd", **kw)


def _ok(res):
    assert res.status == "ok", res.error
    return res


@pytest.fixture
def port():
    rng = np.random.default_rng(7)
    n = 400
    pred = rng.uniform(0.05, 0.8, n).round(2)
    real = np.clip(pred + rng.normal(0, 0.25, n), 0, 1)
    real[rng.random(n) < 0.15] = 0.0
    return pd.DataFrame({"real": real, "pred": pred, "ead": rng.uniform(10, 1000, n),
                         "pool": np.where(pred < 0.3, "P1", np.where(pred < 0.55, "P2", "P3")),
                         "seg": rng.choice(["retail", "sme"], n), "year": rng.choice([2018, 2019, 2020, 2021], n)})


# ── gAUC ──

def test_gauc_hand_example():
    df = pd.DataFrame({"real": [0.0, 0.5, 0.4], "pred": [0.1, 0.2, 0.3]})
    r = _ok(run_test("lgd.gauc", _ctx(df), {"actual": "real", "predicted": "pred"}))
    assert r.summary["gAUC"] == pytest.approx(2 / 3)
    assert r.summary["pairs_compared"] == 3


def test_gauc_binary_equals_auc_and_delong_variance():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300).astype(float)
    x = np.round(y * 0.5 + rng.normal(0, 1, 300), 1)          # rounding creates ties
    g = t_lgd.gauc(x, y)
    assert g["gAUC"] == pytest.approx(roc_auc_score(y, x))
    pos, neg = x[y == 1], x[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    a = psi.mean()
    var = ((v10 - a) ** 2).sum() / len(pos) ** 2 + ((v01 - a) ** 2).sum() / len(neg) ** 2
    assert g["se"] == pytest.approx(np.sqrt(var))


def test_gauc_continuous_equals_somers_d_and_tree_path(port, monkeypatch):
    g = t_lgd.gauc(port["pred"], port["real"])
    d = stats.somersd(port["real"], port["pred"]).statistic
    assert g["gAUC"] == pytest.approx((1 + d) / 2)
    monkeypatch.setattr(t_lgd, "_GAUC_TABLE_MAX_CELLS", 0)
    g2 = t_lgd.gauc(port["pred"], port["real"])
    assert g2["gAUC"] == pytest.approx(g["gAUC"]) and g2["se"] == pytest.approx(g["se"])


def test_gauc_with_bins_initial_and_not_applicable(port):
    r = _ok(run_test("lgd.gauc", _ctx(port), {"actual": "real", "predicted": "pred", "realised_bins": [0.2, 0.5],
                                              "initial_gauc": 0.9}))
    assert 0.5 < r.summary["gAUC"] < 1 and r.summary["p_value_vs_initial"] < 0.05
    flat = pd.DataFrame({"real": [0.3] * 5, "pred": [0.1, 0.2, 0.3, 0.4, 0.5]})
    assert run_test("lgd.gauc", _ctx(flat), {"actual": "real", "predicted": "pred"}).status == "not_applicable"


def test_rank_correlation_matches_scipy(port):
    r = _ok(run_test("lgd.rank_correlation", _ctx(port), {"actual": "real", "predicted": "pred", "segment": "seg"}))
    assert r.summary["spearman_rho"] == pytest.approx(stats.spearmanr(port["pred"], port["real"]).statistic)
    assert r.summary["kendall_tau_b"] == pytest.approx(stats.kendalltau(port["pred"], port["real"]).statistic)
    assert list(r.tables["Correlations"]["group"]) == ["retail", "sme", "ALL"]


# ── calibration ──

def test_loss_shortfall_hand():
    df = pd.DataFrame({"real": [0.5, 0.2, np.nan], "pred": [0.4, 0.2, 0.1], "ead": [100, 200, 50]})
    r = _ok(run_test("lgd.loss_shortfall", _ctx(df), {"actual": "real", "predicted": "pred", "ead": "ead"}))
    assert r.summary["loss_shortfall"] == pytest.approx(1 - 80 / 90)
    assert r.summary["MAD"] == pytest.approx(10 / 300)
    assert r.rows_used == 2 and any("1 rows" in n for n in r.notes)


def test_backtest_ttest_matches_scipy(port):
    r = _ok(run_test("lgd.backtest_ttest", _ctx(port), {"actual": "real", "predicted": "pred", "grade": "pool"}))
    ref = stats.ttest_rel(port["real"], port["pred"], alternative="greater")
    assert r.summary["t_statistic"] == pytest.approx(ref.statistic)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue)
    t = r.tables["LGD back-test by pool"]
    sub = port[port["pool"] == "P2"]
    assert t.loc[t["group"] == "P2", "p_value"].iloc[0] == pytest.approx(
        stats.ttest_rel(sub["real"], sub["pred"], alternative="greater").pvalue)


def test_wilcoxon_matches_scipy(port):
    r = _ok(run_test("lgd.wilcoxon", _ctx(port), {"actual": "real", "predicted": "pred",
                                                  "alternative": "two-sided"}))
    d = (port["real"] - port["pred"]).to_numpy()
    assert r.summary["wilcoxon_p"] == pytest.approx(stats.wilcoxon(d[d != 0], alternative="two-sided").pvalue)


def test_clar_hand_example_and_perfect():
    df = pd.DataFrame({"real": [0.9, 0.1, 0.5, 0.0], "pred": [0.8, 0.8, 0.2, 0.2]})
    r = _ok(run_test("lgd.clar", _ctx(df), {"actual": "real", "predicted": "pred"}))
    assert r.summary["CLAR"] == pytest.approx(0.75)
    perfect = pd.DataFrame({"real": [0.9, 0.7, 0.3, 0.1], "pred": [0.8, 0.8, 0.2, 0.2]})
    assert run_test("lgd.clar", _ctx(perfect), {"actual": "real", "predicted": "pred"}).summary["CLAR"] == \
        pytest.approx(1.0)
    # ties straddling the boundary are split proportionally, independent of row order
    tied = pd.DataFrame({"real": [0.5, 0.5, 0.5, 0.5], "pred": [0.8, 0.8, 0.2, 0.2]})
    a = run_test("lgd.clar", _ctx(tied), {"actual": "real", "predicted": "pred"}).summary["CLAR"]
    b = run_test("lgd.clar", _ctx(tied.iloc[::-1]), {"actual": "real", "predicted": "pred"}).summary["CLAR"]
    assert a == pytest.approx(b) == pytest.approx(2 * (0.5 * 0.25 / 2 + 0.5 * 1.25 / 2))


def test_error_by_segment_hand():
    df = pd.DataFrame({"real": [0.0, 1.0], "pred": [0.5, 0.5], "w": [1.0, 3.0]})
    r = _ok(run_test("lgd.error_by_segment", _ctx(df), {"actual": "real", "predicted": "pred", "ead": "w"}))
    assert r.summary["bias"] == pytest.approx(0) and r.summary["MAE"] == pytest.approx(0.5)
    assert r.summary["RMSE"] == pytest.approx(0.5) and r.summary["R2"] == pytest.approx(0)
    assert r.tables["Accuracy by segment"]["weighted_bias"].iloc[0] == pytest.approx((0.5 - 1.5) / 4)


def test_distribution_mass_at_bounds():
    x = np.array([0, 0, 1, 1, 0.5, 1.2, -0.1])
    df = pd.DataFrame({"real": x})
    r = _ok(run_test("lgd.distribution", _ctx(df), {"actual": "real"}))
    assert r.summary["share_at_0"] == pytest.approx(2 / 7) and r.summary["share_at_1"] == pytest.approx(2 / 7)
    assert r.summary["share_above_1"] == pytest.approx(1 / 7) and r.summary["share_below_0"] == pytest.approx(1 / 7)
    n = 7
    g1, g2 = stats.skew(x, bias=False), stats.kurtosis(x, bias=False)
    assert r.summary["bimodality_coefficient"] == pytest.approx((g1 ** 2 + 1) / (g2 + 3 * 36 / (5 * 4)))
    h = r.tables["Histogram"]
    assert h["count"].sum() == n


# ── downturn / ELBE / workout ──

def test_downturn_hand():
    df = pd.DataFrame({"real": [0.2, 0.4, 0.6, 0.8, 0.3, 0.3], "year": [2019, 2019, 2020, 2020, 2021, 2021],
                       "dt": [0, 0, 1, 1, 0, 0]})
    r = _ok(run_test("lgd.downturn_comparison", _ctx(df), {"actual": "real", "period": "year",
                                                           "downturn_periods": ["2020"]}))
    s = r.summary
    assert s["LRA_default_weighted"] == pytest.approx(2.6 / 6)
    assert s["downturn_mean"] == pytest.approx(0.7) and s["non_downturn_mean"] == pytest.approx(0.3)
    assert s["welch_p"] == pytest.approx(stats.ttest_ind([0.6, 0.8], [0.2, 0.4, 0.3, 0.3], equal_var=False).pvalue)
    assert s["worst_period"] == "2020"
    r2 = _ok(run_test("lgd.downturn_comparison", _ctx(df), {"actual": "real", "period": "year",
                                                            "downturn_flag": "dt"}))
    assert r2.summary["difference"] == pytest.approx(0.4)
    r3 = _ok(run_test("lgd.downturn_comparison", _ctx(df), {"actual": "real", "period": "year"}))
    assert "downturn_mean" not in r3.summary
    bad = run_test("lgd.downturn_comparison", _ctx(df), {"actual": "real", "period": "year",
                                                         "downturn_periods": ["1999"]})
    assert bad.status == "error"


def test_elbe_backtest_hand():
    df = pd.DataFrame({"real": [0.5, 0.7, 0.2], "elbe": [0.4, 0.6, 0.3], "lid": [0.45, 0.55, 0.35]})
    r = _ok(run_test("lgd.elbe_backtest", _ctx(df), {"actual": "real", "predicted": "elbe",
                                                     "lgd_in_default": "lid"}))
    assert r.summary["mean_difference"] == pytest.approx(0.1 / 3)
    assert r.summary["p_value"] == pytest.approx(stats.ttest_rel(df["real"], df["elbe"]).pvalue)
    assert r.summary["n_lgd_in_default_below_elbe"] == 1
    assert r.summary["mean_add_on"] == pytest.approx((0.05 - 0.05 + 0.05) / 3)


def test_workout_length_hand():
    df = pd.DataFrame({"d0": ["2020-01-01"] * 4, "d1": ["2020-01-31", "2020-03-01", "2020-03-31", None]})
    r = _ok(run_test("lgd.workout_length", _ctx(df), {"date": "d0", "end_date": "d1", "unit": "days",
                                                      "as_of": "2020-02-15"}))
    # with as_of 2020-02-15 the 60 and 90-day closures are still open at as_of
    assert r.summary["n_closed"] == 1 and r.summary["n_open"] == 3
    r = _ok(run_test("lgd.workout_length", _ctx(df), {"date": "d0", "end_date": "d1", "unit": "days",
                                                      "as_of": "2020-12-31"}))
    s = r.summary
    assert s["n_closed"] == 3 and s["n_open"] == 1
    assert s["mean_closed"] == pytest.approx(60) and s["median_closed"] == pytest.approx(60)
    assert s["km_median"] == pytest.approx(60)            # S(30)=.75, S(60)=.5


def test_cure_rate_categories_and_binomial():
    df = pd.DataFrame({"out": ["cure", "liquidation", "cure", "restructure", "cure", "liquidation"],
                       "seg": ["a", "a", "a", "b", "b", "b"], "p": [0.5] * 6,
                       "lgd": [0.0, 0.6, 0.02, 0.4, 0.0, 0.8]})
    r = _ok(run_test("lgd.cure_rate", _ctx(df), {"outcome": "out", "segment": "seg", "predicted": "p",
                                                 "actual": "lgd"}))
    assert r.summary["cure_rate"] == pytest.approx(0.5)
    assert r.summary["binomial_p"] == pytest.approx(stats.binomtest(3, 6, 0.5).pvalue)
    assert r.summary["mean_lgd_not_cured"] == pytest.approx(0.6)
    t = r.tables["Cure rate"]
    assert t.loc[t["segment"] == "a", "cure_rate"].iloc[0] == pytest.approx(2 / 3)
    r2 = _ok(run_test("lgd.cure_rate", _ctx(df.assign(f=[1, 0, 1, 0, 1, 0])), {"outcome": "f"}))
    assert r2.summary["cures"] == 3


def test_realised_lgd_from_cashflows_hand():
    defaults = pd.DataFrame({"fid": ["A", "B"], "ead": [100.0, 200.0], "ddate": ["2021-01-01", "2021-06-30"],
                             "rep": [0.55, 0.9]})
    cf = pd.DataFrame({"fid": ["A", "A", "A", "X"], "date": ["2022-01-01", "2021-01-01", "2020-12-01", "2021-01-01"],
                       "amount": [55.0, -5.0, 10.0, 1.0]})
    ctx = _ctx(defaults, tables={"cf": cf})
    r = _ok(run_test("lgd.realised_from_cashflows", ctx, {"id": "fid", "ead": "ead", "date": "ddate",
                                                         "cashflows": "cf", "discount_rate": 0.1, "actual": "rep"}))
    t = r.tables["Realised LGD per default"].set_index("id")
    assert t.loc["A", "realised_lgd"] == pytest.approx(0.55)          # 1 - (55/1.1 - 5)/100
    assert t.loc["B", "realised_lgd"] == pytest.approx(1.0)
    assert r.summary["ead_weighted_lgd"] == pytest.approx(1 - 45 / 300)
    assert r.summary["n_without_cashflows"] == 1 and r.summary["n_mismatch"] == 1
    assert any("before the default date" in n for n in r.notes)
    # typed cash flows give the same answer
    cf2 = pd.DataFrame({"fid": ["A", "A"], "date": ["2022-01-01", "2021-01-01"], "amount": [55.0, 5.0],
                        "kind": ["recovery", "Cost"]})
    r2 = _ok(run_test("lgd.realised_from_cashflows", _ctx(defaults, tables={"cf2": cf2}),
                      {"id": "fid", "ead": "ead", "date": "ddate", "cashflows": "cf2", "discount_rate": 0.1,
                       "cf_type": "kind"}))
    assert r2.tables["Realised LGD per default"].set_index("id").loc["A", "realised_lgd"] == pytest.approx(0.55)


def test_pool_homogeneity_matches_scipy(port):
    r = _ok(run_test("lgd.pool_homogeneity", _ctx(port), {"actual": "real", "grade": "pool", "predicted": "pred"}))
    groups = [port.loc[port["pool"] == p, "real"] for p in ["P1", "P2", "P3"]]
    assert r.summary["kruskal_p"] == pytest.approx(stats.kruskal(*groups).pvalue)
    adj = r.tables["Adjacent pools"]
    assert list(adj["lower_pool"]) == ["P1", "P2"]
    assert adj["mann_whitney_p_two_sided"].iloc[0] == pytest.approx(
        stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided").pvalue)


def test_incomplete_workouts_hand():
    df = pd.DataFrame({"real": [0.2, 0.4, 0.9, 0.7], "open": ["N", "N", "Y", "Y"], "est": [np.nan, np.nan, 0.5, 0.3],
                       "yr": [2020, 2021, 2021, 2021]})
    r = _ok(run_test("lgd.incomplete_workouts", _ctx(df), {"actual": "real", "open_flag": "open", "predicted": "est",
                                                           "period": "yr"}))
    s = r.summary
    assert s["n_open"] == 2 and s["share_open"] == pytest.approx(0.5)
    assert s["lra_closed_only"] == pytest.approx(0.3)
    assert s["lra_open_at_loss_to_date"] == pytest.approx(0.55)
    assert s["lra_open_at_estimate"] == pytest.approx((0.2 + 0.4 + 0.5 + 0.3) / 4)
    assert len(r.tables["Open vs closed workouts"]) == 3


def test_determinism_and_nan_handling(port):
    p = port.copy()
    p.loc[:4, "real"] = np.nan
    a = run_test("lgd.backtest_ttest", _ctx(p), {"actual": "real", "predicted": "pred", "grade": "pool"})
    b = run_test("lgd.backtest_ttest", _ctx(p.copy()), {"actual": "real", "predicted": "pred", "grade": "pool"})
    assert a.run_id == b.run_id and a.summary == b.summary
    assert a.rows_used == len(p) - 5 and any("5 rows" in n for n in a.notes)
