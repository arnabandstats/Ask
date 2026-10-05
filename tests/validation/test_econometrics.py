"""Econometric diagnostics (ask/validation/t_econometrics.py): known answers and edge cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from ask.validation.core import RunContext, run_test
from ask.validation.t_econometrics import _bde_pvalue, diebold_mariano_stat, phillips_perron


def _ctx(df):
    return RunContext(df=df, source_name="econ")


def _ok(test_id, df, params):
    res = run_test(test_id, _ctx(df), params)
    assert res.status == "ok", f"{test_id}: {res.error}"
    return res


@pytest.fixture(scope="module")
def ts():
    rng = np.random.default_rng(3)
    n = 120
    x1, x2 = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        x1[t] = 0.6 * x1[t - 1] + rng.normal()
        x2[t] = 0.3 * x2[t - 1] + rng.normal()
    y = 1 + 0.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)
    dates = pd.period_range("1995Q1", periods=n, freq="Q").astype(str)
    df = pd.DataFrame({"date": dates, "y": y, "x1": x1, "x2": x2})
    return df.sample(frac=1, random_state=1).reset_index(drop=True)      # shuffled: date must restore order


@pytest.fixture(scope="module")
def logit_df():
    rng = np.random.default_rng(5)
    n = 3000
    a, b = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 0.8 * a - 0.5 * b)))).astype(int)
    return pd.DataFrame({"default": y, "a": a, "b": b, "noise": rng.normal(size=n)})


SPEC = {"target": "y", "features": ["x1", "x2"], "date": "date"}


def _sorted(ts):
    return ts.sort_values("date").reset_index(drop=True)


# ── coefficients, SEs, fit ─────────────────────────────────────────────────

def test_ols_coefficients_replicate_lstsq_and_documented(ts):
    s = _sorted(ts)
    X = sm.add_constant(s[["x1", "x2"]])
    beta = np.linalg.lstsq(X.to_numpy(), s["y"].to_numpy(), rcond=None)[0]
    exp = {"intercept": beta[0], "x1": beta[1], "x2": beta[2]}
    res = _ok("econ.coefficients", ts, SPEC | {"expected_coefficients": exp})
    t = res.tables["Coefficients"]
    assert np.allclose(t["estimate"], beta)
    assert res.summary["wald_equal_documented"] == pytest.approx(0, abs=1e-12)
    assert res.summary["sign_mismatches"] == 0
    wrong = _ok("econ.coefficients", ts, SPEC | {"expected_coefficients": {"x1": -0.5},
                                                 "expected_signs": {"x2": "+"}})
    comp = wrong.tables["Comparison with documented coefficients"].set_index("variable")
    assert comp.loc["x1", "sign_holds"] is False or comp.loc["x1", "sign_holds"] == False  # noqa: E712
    assert comp.loc["x2", "sign_holds"] == False  # noqa: E712
    assert wrong.summary["wald_p_value"] < 1e-10
    bad = run_test("econ.coefficients", _ctx(ts), SPEC | {"expected_coefficients": {"zzz": 1}})
    assert bad.status == "error" and "zzz" in bad.error


def test_robust_se_matches_statsmodels(ts):
    s = _sorted(ts)
    X = sm.add_constant(s[["x1", "x2"]])
    ref = sm.OLS(s["y"], X).fit(cov_type="HC3")
    hac = sm.OLS(s["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    res = _ok("econ.robust_se", ts, SPEC | {"hac_lags": 3})
    w = res.tables["Standard errors"].set_index("variable")
    assert np.allclose(w.loc[["const", "x1", "x2"], "SE_HC3"], ref.bse)
    assert np.allclose(w.loc[["const", "x1", "x2"], "SE_HAC"], hac.bse)
    r2 = _ok("econ.coefficients", ts, SPEC | {"cov_type": "HAC"})
    assert any("Newey–West rule" in n for n in r2.notes)


def test_goodness_of_fit_ols_and_logit(ts, logit_df):
    s = _sorted(ts)
    ref = sm.OLS(s["y"], sm.add_constant(s[["x1", "x2"]])).fit()
    res = _ok("econ.goodness_of_fit", ts, SPEC)
    assert res.summary["R2"] == pytest.approx(ref.rsquared)
    assert res.summary["AIC"] == pytest.approx(ref.aic)
    lg = sm.Logit(logit_df["default"], sm.add_constant(logit_df[["a", "b"]])).fit(disp=0)
    res = _ok("econ.goodness_of_fit", logit_df, {"target": "default", "features": ["a", "b"]})
    assert res.summary["model_kind"] == "logit"
    assert res.summary["McFadden_R2"] == pytest.approx(lg.prsquared)
    assert res.summary["LR_stat"] == pytest.approx(2 * (lg.llf - lg.llnull))
    assert res.summary["LR_p_value"] < 1e-20
    frac = logit_df.assign(rate=1 / (1 + np.exp(-(0.3 * logit_df["a"]))))
    res = _ok("econ.goodness_of_fit", frac, {"target": "rate", "features": ["a"], "model_kind": "fractional_logit"})
    assert res.summary["corr2_fitted_actual"] == pytest.approx(1, abs=1e-6)


def test_logit_coefficients_recover_truth(logit_df):
    res = _ok("econ.coefficients", logit_df, {"target": "default", "features": ["a", "b"],
                                              "expected_coefficients": {"const": -1.5, "a": 0.8, "b": -0.5}})
    assert res.summary["model_kind"] == "logit"
    comp = res.tables["Comparison with documented coefficients"]
    assert comp["documented_inside_CI"].all() and comp["sign_holds"].all()


# ── residual diagnostics ───────────────────────────────────────────────────

def test_durbin_watson_known_series():
    n = 40
    df = pd.DataFrame({"e": np.tile([1.0, -1.0], n // 2)})
    res = _ok("econ.autocorrelation", df, {"residuals": "e", "lags": 2})
    assert res.summary["durbin_watson"] == pytest.approx(4 * (n - 1) / n)
    assert res.summary["rho1"] < -0.9


def test_autocorrelation_refit_matches_statsmodels(ts):
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson
    s = _sorted(ts)
    ref = sm.OLS(s["y"], sm.add_constant(s[["x1", "x2"]])).fit()
    res = _ok("econ.autocorrelation", ts, SPEC | {"lags": 4})
    assert res.summary["durbin_watson"] == pytest.approx(durbin_watson(ref.resid))
    assert res.summary["breusch_godfrey_LM"] == pytest.approx(acorr_breusch_godfrey(ref, nlags=4, result_object=True).lm)
    # row order matters: without the date the shuffled order gives a different DW
    shuffled = _ok("econ.autocorrelation", ts, {"target": "y", "features": ["x1", "x2"]})
    assert shuffled.summary["durbin_watson"] != pytest.approx(res.summary["durbin_watson"])


def test_normality_matches_scipy(ts):
    s = _sorted(ts)
    e = sm.OLS(s["y"], sm.add_constant(s[["x1", "x2"]])).fit().resid
    res = _ok("econ.residual_normality", ts, SPEC)
    assert res.summary["jarque_bera"] == pytest.approx(stats.jarque_bera(e).statistic)
    assert res.summary["shapiro_wilk_p"] == pytest.approx(stats.shapiro(e).pvalue)
    skewed = pd.DataFrame({"r": np.random.default_rng(0).exponential(size=500)})
    assert _ok("econ.residual_normality", skewed, {"residuals": "r"}).summary["jarque_bera_p"] < 1e-10


def test_heteroskedasticity_detected():
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 10, 400)
    y = 2 + x + rng.normal(0, 1, 400) * x
    res = _ok("econ.heteroskedasticity", pd.DataFrame({"y": y, "x": x}), {"target": "y", "features": ["x"],
                                                                          "sort_by": "x"})
    assert res.summary["breusch_pagan_p"] < 1e-6 and res.summary["goldfeld_quandt_p"] < 1e-6
    assert res.summary["white_p"] < 1e-6


def test_specification_reset_and_link(logit_df):
    rng = np.random.default_rng(4)
    x = rng.uniform(-3, 3, 300)
    df = pd.DataFrame({"y": 1 + x + 0.8 * x ** 2 + rng.normal(0, 0.5, 300), "x": x})
    res = _ok("econ.specification", df, {"target": "y", "features": ["x"]})
    t = res.tables["Specification tests"]
    assert t.loc[t["test"].str.startswith("Ramsey"), "p_value"].iloc[0] < 1e-10
    res = _ok("econ.specification", logit_df, {"target": "default", "features": ["a", "b"]})
    assert res.tables["Specification tests"]["test"].str.contains("Pregibon").all()


# ── structural breaks ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def broken():
    rng = np.random.default_rng(9)
    n = 100
    x = rng.normal(size=n)
    y = np.where(np.arange(n) < 60, 1 + 2 * x, 3 + 0.5 * x) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"t": np.arange(n), "y": y, "x": x})


def _ssr(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(((y - X @ b) ** 2).sum())


def test_chow_hand_computation(broken):
    y, X = broken["y"].to_numpy(), sm.add_constant(broken[["x"]]).to_numpy()
    sp, s1, s2 = _ssr(y, X), _ssr(y[:60], X[:60]), _ssr(y[60:], X[60:])
    F = ((sp - s1 - s2) / 2) / ((s1 + s2) / (100 - 4))
    res = _ok("econ.chow_test", broken, {"target": "y", "features": ["x"], "date": "t", "break_at": "60"})
    assert res.summary["F_stat"] == pytest.approx(F)
    assert res.summary["p_value"] == pytest.approx(stats.f.sf(F, 2, 96))
    assert res.summary["p_value"] < 1e-10 and res.summary["n_regime1"] == 60
    same = _ok("econ.chow_test", broken, {"target": "y", "features": ["x"], "break_index": 60})
    assert same.summary["F_stat"] == pytest.approx(F)
    pred = _ok("econ.chow_test", broken, {"target": "y", "features": ["x"], "break_index": 99})
    assert pred.summary["test"] == "Chow predictive F"


def test_chow_no_break_and_logit_lr(logit_df):
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    df = pd.DataFrame({"y": 1 + x + rng.normal(size=200), "x": x})
    assert _ok("econ.chow_test", df, {"target": "y", "features": ["x"], "break_index": 100}).summary["p_value"] > 0.01
    res = _ok("econ.chow_test", logit_df, {"target": "default", "features": ["a", "b"], "break_index": 1500})
    assert res.summary["test"] == "Likelihood-ratio Chow" and res.summary["df"] == 3


def test_sup_f_finds_break_and_matches_chow(broken):
    res = _ok("econ.sup_f_break", broken, {"target": "y", "features": ["x"], "date": "t", "n_boot": 99})
    assert abs(res.summary["break_position"] - 60) <= 2
    assert res.summary["bootstrap_p_value"] == pytest.approx(1 / 100)
    path = res.tables["F statistic by candidate break"].set_index("break_position")
    chow = _ok("econ.chow_test", broken, {"target": "y", "features": ["x"], "break_index": 60})
    assert path.loc[60, "F_stat"] == pytest.approx(chow.summary["F_stat"])
    again = _ok("econ.sup_f_break", broken, {"target": "y", "features": ["x"], "date": "t", "n_boot": 99})
    assert again.summary == res.summary


def test_cusum_pvalue_formula_and_break(broken):
    assert _bde_pvalue(0.948) == pytest.approx(0.05, abs=0.001)
    assert _bde_pvalue(1.143) == pytest.approx(0.01, abs=0.001)
    res = _ok("econ.cusum", broken, {"target": "y", "features": ["x"], "date": "t"})
    assert res.summary["cusum_sq_max_deviation"] > 0
    assert res.summary["ols_cusum_p_value"] < 0.01
    assert len(res.tables["CUSUM path"]) == 100 - 2 and res.figures


def test_recursive_coefficients(broken):
    res = _ok("econ.recursive_coefficients", broken, {"target": "y", "features": ["x"], "date": "t", "step": 5})
    st = res.tables["Coefficient stability"].set_index("variable")
    assert st.loc["x", "max"] > 1.5 and st.loc["x", "share_inside_full_CI"] < 1
    path = res.tables["Recursive estimates"]
    last = path[(path["variable"] == "x") & (path["n_obs"] == 100)]["estimate"].iloc[0]
    assert last == pytest.approx(st.loc["x", "full_sample"])          # last window = full sample


# ── stationarity, cointegration, causality ─────────────────────────────────

def test_phillips_perron_lag0_equals_dickey_fuller():
    from statsmodels.tsa.stattools import adfuller
    x = np.random.default_rng(0).normal(size=150).cumsum()
    for reg in ("c", "ct", "n"):
        pp = phillips_perron(x, reg, lags=0)
        df_t = adfuller(x, maxlag=0, autolag=None, regression=reg, result_object=True).statistic
        assert pp["z_tau"] == pytest.approx(df_t)


def test_unit_root_tests_distinguish():
    rng = np.random.default_rng(8)
    n = 250
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.3 * ar[t - 1] + rng.normal()
    df = pd.DataFrame({"rw": rng.normal(size=n).cumsum(), "ar": ar, "t": range(n)})
    res = _ok("econ.unit_root", df, {"columns": ["rw", "ar"], "date": "t", "differences": True})
    t = res.tables["Unit-root and stationarity tests"].set_index(["series", "form"])
    assert t.loc[("rw", "level"), "ADF_p"] > 0.1 and t.loc[("rw", "level"), "PP_p"] > 0.1
    assert t.loc[("ar", "level"), "ADF_p"] < 0.01 and t.loc[("ar", "level"), "PP_p"] < 0.01
    assert t.loc[("rw", "first difference"), "PP_p"] < 0.01
    za = _ok("econ.zivot_andrews", df, {"columns": ["ar"], "date": "t"})
    assert za.summary["min_p_value"] < 0.05


def test_cointegration_and_granger():
    rng = np.random.default_rng(12)
    n = 300
    x = rng.normal(size=n).cumsum()
    y = 2 + 0.7 * x + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"x": x, "y": y, "z": rng.normal(size=n).cumsum(), "t": range(n)})
    eg = _ok("econ.engle_granger", df, {"target": "y", "features": ["x"], "date": "t"})
    assert eg.summary["p_value"] < 0.01
    assert eg.tables["Cointegrating regression"].set_index("variable").loc["x", "long_run_coefficient"] == \
        pytest.approx(0.7, abs=0.05)
    jo = _ok("econ.johansen", df, {"columns": ["x", "y", "z"], "date": "t"})
    assert jo.summary["rank_trace"] >= 1
    lead = rng.normal(size=n)
    out = np.r_[0, 0.8 * lead[:-1]] + rng.normal(0, 0.3, n)
    g = pd.DataFrame({"y": out, "x": lead, "t": range(n)})
    res = _ok("econ.granger_causality", g, {"target": "y", "features": ["x"], "date": "t", "max_lag": 2,
                                            "both_directions": True})
    t = res.tables["Granger causality"]
    assert t.loc[(t["cause"] == "x") & (t["lag"] == 1), "F_p"].iloc[0] < 1e-10


# ── influence, VIF ─────────────────────────────────────────────────────────

def test_influence_flags_planted_outlier():
    rng = np.random.default_rng(6)
    x = rng.normal(size=80)
    y = 1 + x + rng.normal(0, 0.3, 80)
    x[17], y[17] = 6.0, -10.0
    df = pd.DataFrame({"y": y, "x": x})
    res = _ok("econ.influence", df, {"target": "y", "features": ["x"]})
    assert res.summary["max_cooks_row"] == "17"
    ref = sm.OLS(y, sm.add_constant(x)).fit().get_influence()
    assert res.summary["max_cooks_distance"] == pytest.approx(ref.cooks_distance[0].max())


def test_influence_logit(logit_df):
    res = _ok("econ.influence", logit_df, {"target": "default", "features": ["a", "b"], "top_n": 5})
    assert len(res.tables["Top observations by Cook's distance"]) == 5


def test_vif_matches_auxiliary_regression():
    rng = np.random.default_rng(7)
    a = rng.normal(size=500)
    b = 0.9 * a + 0.3 * rng.normal(size=500)
    c = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    r2 = sm.OLS(a, sm.add_constant(np.column_stack([b, c]))).fit().rsquared
    res = _ok("econ.vif", df, {"features": ["a", "b", "c"]})
    v = res.tables["VIF"].set_index("variable")
    assert v.loc["a", "VIF"] == pytest.approx(1 / (1 - r2))
    assert res.summary["condition_number"] > 1
    dup = df.assign(d=df["a"] * 2)
    assert run_test("econ.vif", _ctx(dup), {"features": ["a", "d"]}).status == "not_applicable"


# ── forecasting ────────────────────────────────────────────────────────────

def test_forecast_accuracy_hand_values():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.array([1.5, 2.0, 2.5, 4.5])
    df = pd.DataFrame({"a": a, "f": f, "b": a + 1, "s": ["oot"] * 4})
    res = _ok("econ.forecast_accuracy", df, {"actual": "a", "predicted": "f", "benchmark": "b",
                                             "sample": "s", "current_value": "oot"})
    rmse = math.sqrt(np.mean((a - f) ** 2))
    assert res.summary["RMSE"] == pytest.approx(rmse)
    assert res.summary["MAE"] == pytest.approx(0.375)
    assert res.summary["MAPE"] == pytest.approx(np.mean(np.abs((a - f) / a)))
    assert res.summary["theil_U1"] == pytest.approx(rmse / (math.sqrt(np.mean(a ** 2)) + math.sqrt(np.mean(f ** 2))))
    assert res.summary["theil_U2_vs_no_change"] == pytest.approx(math.sqrt(np.mean((a[1:] - f[1:]) ** 2)) / 1.0)
    row = res.tables["Forecast accuracy"].iloc[0]
    assert row["bias_proportion_UM"] + row["variance_proportion_US"] + row["covariance_proportion_UC"] == \
        pytest.approx(1)
    assert res.summary["RMSE_ratio_to_benchmark"] == pytest.approx(rmse / 1.0)


def test_diebold_mariano_hand_computation():
    rng = np.random.default_rng(10)
    a = rng.normal(size=60)
    f = a + rng.normal(0, 0.5, 60)
    b = a + rng.normal(0, 1.0, 60)
    d = (a - f) ** 2 - (a - b) ** 2
    n = len(d)
    dm = d.mean() / math.sqrt(d.var(ddof=0) / n)
    hln = dm * math.sqrt((n - 1) / n)
    r = diebold_mariano_stat(a - f, a - b, 1, "squared")
    assert r["DM_stat"] == pytest.approx(dm) and r["HLN_stat"] == pytest.approx(hln)
    assert r["HLN_p_value"] == pytest.approx(2 * stats.t.sf(abs(hln), n - 1))
    df = pd.DataFrame({"a": a, "f": f, "b": b})
    res = _ok("econ.diebold_mariano", df, {"actual": "a", "predicted": "f", "benchmark": "b", "horizon": 2})
    assert res.summary["DM_stat"] < 0 and res.summary["HLN_p_value_model_better"] < 0.05


# ── sensitivity, bootstrap ─────────────────────────────────────────────────

def test_macro_sensitivity_ols_is_beta_times_sd(ts):
    res = _ok("econ.macro_sensitivity", ts, SPEC | {"shock_sd": 2})
    t = res.tables["Sensitivity by driver"].set_index("variable")
    for v in ("x1", "x2"):
        assert t.loc[v, "change_up"] == pytest.approx(t.loc[v, "coefficient"] * 2 * ts[v].std())
        assert t.loc[v, "change_down"] == pytest.approx(-t.loc[v, "change_up"])


def test_bootstrap_stability_deterministic(logit_df):
    p = {"target": "default", "features": ["a", "b", "noise"], "n_boot": 40}
    r1 = _ok("econ.bootstrap_stability", logit_df, p)
    r2 = _ok("econ.bootstrap_stability", logit_df.copy(), p)
    assert r1.summary == r2.summary and r1.run_id == r2.run_id
    t = r1.tables["Bootstrap stability"].set_index("variable")
    assert t.loc["a", "share_significant"] == 1.0 and t.loc["a", "share_expected_sign"] == 1.0
    assert t.loc["noise", "share_significant"] < 0.5
    blk = _ok("econ.bootstrap_stability", logit_df, p | {"block_length": 20, "expected_signs": {"b": "-"}})
    assert blk.tables["Bootstrap stability"].set_index("variable").loc["b", "share_expected_sign"] == 1.0


# ── scorecard specifics ────────────────────────────────────────────────────

def test_woe_iv_hand_computation():
    # bin A: 40 good 10 bad; bin B: 60 good 40 bad (categorical)
    df = pd.DataFrame({"g": ["A"] * 50 + ["B"] * 100,
                       "bad": [0] * 40 + [1] * 10 + [0] * 60 + [1] * 40})
    res = _ok("econ.woe_iv", df, {"target": "bad", "features": ["g"]})
    w = res.tables["WoE by bin"].set_index("bin")
    woe_a, woe_b = math.log(0.4 / 0.2), math.log(0.6 / 0.8)
    assert w.loc["A", "WoE"] == pytest.approx(woe_a) and w.loc["B", "WoE"] == pytest.approx(woe_b)
    assert res.summary["max_IV"] == pytest.approx((0.4 - 0.2) * woe_a + (0.6 - 0.8) * woe_b)


def test_woe_monotonicity(logit_df):
    df = logit_df.copy()
    df.loc[:9, "a"] = np.nan
    res = _ok("econ.woe_iv", df, {"target": "default", "features": ["a", "noise"], "bins": 5,
                                  "bin_edges": {"noise": [-1, 0, 1]}})
    s = res.tables["IV by variable"].set_index("variable")
    assert bool(s.loc["a", "WoE_monotone"]) and s.loc["a", "spearman_bin_order_vs_WoE"] == pytest.approx(-1)
    assert s.loc["a", "IV"] > s.loc["noise", "IV"]
    assert "<missing>" in set(res.tables["WoE by bin"]["bin"])
    assert s.loc["noise", "bins"] == 4


def test_scorecard_points_known_values():
    pdv = np.array([1 / 51, 1 / 101, 0.5])
    df = pd.DataFrame({"pd": pdv})
    factor = 20 / math.log(2)
    expected = 600 + factor * (np.log((1 - pdv) / pdv) - math.log(50))
    df["score"] = expected
    res = _ok("econ.scorecard_points", df, {"pd": "pd", "score": "score"})
    assert res.summary["factor"] == pytest.approx(factor)
    assert res.summary["max_abs_difference"] == pytest.approx(0, abs=1e-9)
    assert expected[0] == pytest.approx(600) and expected[1] == pytest.approx(600 + factor * math.log(100 / 50))
    # same via logit coefficients on a WoE-coded feature: ln(odds_bad) = b0 + b1 x
    x = np.array([0.2, -0.4, 0.1])
    b0, b1 = -2.0, -0.9
    eta = b0 + b1 * x
    df2 = pd.DataFrame({"x": x, "score": 600 + factor * (-eta - math.log(50))})
    res = _ok("econ.scorecard_points", df2, {"features": ["x"], "coefficients": {"const": b0, "x": b1},
                                             "score": "score"})
    assert res.summary["share_within_tolerance"] == 1.0
    assert "Points by attribute" in res.tables
    rounded = _ok("econ.scorecard_points", df2, {"features": ["x"], "coefficients": {"const": b0, "x": b1},
                                                 "score": "score", "round_points": True})
    assert rounded.summary["max_abs_difference"] <= 0.5 + 1e-9
    assert run_test("econ.scorecard_points", _ctx(df2), {}).status == "error"


# ── edge cases ─────────────────────────────────────────────────────────────

def test_edge_cases(ts, logit_df):
    one = logit_df.assign(default=0)
    r = run_test("econ.coefficients", _ctx(one), {"target": "default", "features": ["a"], "model_kind": "logit"})
    assert r.status in {"not_applicable", "error"}
    assert run_test("econ.woe_iv", _ctx(one), {"target": "default", "features": ["a"]}).status == "not_applicable"
    tiny = ts.head(3)
    assert run_test("econ.coefficients", _ctx(tiny), SPEC).status == "not_applicable"
    holes = ts.copy()
    holes.loc[:4, "x1"] = np.nan
    r = _ok("econ.coefficients", holes, SPEC)
    assert r.rows_used == len(ts) - 5 and any("5 rows" in n for n in r.notes)
    r = run_test("econ.heteroskedasticity", _ctx(logit_df), {"target": "default", "features": ["a"]})
    assert r.status == "not_applicable"
    const = ts.assign(k=1.0)
    assert run_test("econ.coefficients", _ctx(const), {"target": "y", "features": ["x1", "k"]}).status == \
        "not_applicable"
