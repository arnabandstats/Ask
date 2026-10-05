"""Market risk VaR/ES tests (ask/validation/t_market.py): known answers, edge cases, determinism."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="bt", **kw)


def _hits_frame(hits, var=1.0):
    """P&L series whose exceptions are exactly `hits` (loss 2 > VaR 1 on hit days)."""
    h = np.asarray(hits)
    return pd.DataFrame({"pnl": np.where(h == 1, -2.0, 0.5), "var": var})


def _ok(res):
    assert res.status == "ok", res.error
    return res


BT = {"pnl": "pnl", "var": "var"}


@pytest.fixture
def garch_like():
    rng = np.random.default_rng(22)
    n = 750
    sig = np.where(np.arange(n) % 250 < 60, 2.0, 1.0)
    pnl = rng.standard_normal(n) * sig
    var = 2.326 * 1.2 * np.ones(n)
    es = 2.665 * 1.2 * np.ones(n)
    pit = stats.norm.cdf(pnl / sig)
    date = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": date, "pnl": pnl, "var": var, "es": es, "pit": pit, "sig": sig})


# ── coverage tests ────────────────────────────────────────────────────────

def test_kupiec_pof_matches_hand_calculation():
    hits = np.zeros(250, int); hits[[10, 50, 90, 130, 170]] = 1
    res = _ok(run_test("var.kupiec_pof", _ctx(_hits_frame(hits)), BT))
    ll0 = 245 * math.log(0.99) + 5 * math.log(0.01)
    ll1 = 245 * math.log(0.98) + 5 * math.log(0.02)
    assert res.summary["exceptions"] == 5
    assert res.summary["LR_POF"] == pytest.approx(-2 * (ll0 - ll1), rel=1e-12)
    assert res.summary["LR_POF"] == pytest.approx(1.9568, abs=1e-4)
    assert res.summary["p_value"] == pytest.approx(stats.chi2.sf(-2 * (ll0 - ll1), 1))
    assert res.summary["binomial_p_value_too_many"] == pytest.approx(stats.binom.sf(4, 250, 0.01))


def test_kupiec_tuff_hand_calculation_and_no_failure():
    hits = np.zeros(300, int); hits[49] = 1
    res = _ok(run_test("var.kupiec_tuff", _ctx(_hits_frame(hits)), BT))
    v, p = 50, 0.01
    lr = -2 * (math.log(p) + (v - 1) * math.log(1 - p)) + 2 * (math.log(1 / v) + (v - 1) * math.log(1 - 1 / v))
    assert res.summary["time_until_first_failure"] == 50
    assert res.summary["LR_TUFF"] == pytest.approx(lr, rel=1e-12)
    res = run_test("var.kupiec_tuff", _ctx(_hits_frame(np.zeros(100, int))), BT)
    assert res.status == "not_applicable"


@pytest.mark.parametrize("x,zone,plus", [(0, "green", 0.0), (4, "green", 0.0), (5, "yellow", 0.40),
                                         (9, "yellow", 0.85), (10, "red", 1.0), (12, "red", 1.0)])
def test_basel_traffic_light_boundaries(x, zone, plus):
    hits = np.zeros(250, int); hits[:x] = 1
    res = _ok(run_test("var.traffic_light", _ctx(_hits_frame(hits)), BT))
    assert res.summary["zone"] == zone
    assert res.summary["plus_factor"] == pytest.approx(plus)
    assert res.summary["cumulative_probability"] == pytest.approx(stats.binom.cdf(x, 250, 0.01))


def test_traffic_light_other_window_has_no_plus_factor():
    hits = np.zeros(500, int); hits[:8] = 1
    res = _ok(run_test("var.traffic_light", _ctx(_hits_frame(hits)), BT))
    assert "plus_factor" not in res.summary
    assert res.summary["zone"] == ("green" if stats.binom.cdf(8, 500, 0.01) < 0.95 else "yellow")
    assert any("generalised" in n for n in res.notes)


def test_christoffersen_hand_built_sequence():
    seq = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    res = _ok(run_test("var.christoffersen", _ctx(_hits_frame(seq)), BT | {"confidence": 0.9}))
    t = res.tables["Transition counts"]["count"].tolist()
    assert t == [8, 2, 2, 2]          # n00, n01, n10, n11
    pi01, pi11, pi = 2 / 10, 2 / 4, 4 / 14
    ll0 = 10 * math.log(1 - pi) + 4 * math.log(pi)
    ll1 = 8 * math.log(1 - pi01) + 2 * math.log(pi01) + 2 * math.log(1 - pi11) + 2 * math.log(pi11)
    lr_ind = -2 * (ll0 - ll1)
    assert res.summary["LR_ind"] == pytest.approx(lr_ind, rel=1e-12)
    n, x, p = 15, 4, 0.1
    pof = -2 * ((n - x) * math.log(1 - p) + x * math.log(p) - (n - x) * math.log(1 - x / n) - x * math.log(x / n))
    assert res.summary["LR_cc"] == pytest.approx(lr_ind + pof, rel=1e-12)
    assert res.summary["p_value_cc"] == pytest.approx(stats.chi2.sf(lr_ind + pof, 2))


def test_haas_mixed_kupiec_is_sum_of_tuffs_plus_pof():
    hits = np.zeros(250, int); hits[[19, 29, 199]] = 1
    res = _ok(run_test("var.haas_mixed_kupiec", _ctx(_hits_frame(hits)), BT))
    p = 0.01

    def tuff(v):
        return -2 * (math.log(p) + (v - 1) * math.log(1 - p)) + 2 * (math.log(1 / v) + (v - 1) * math.log(1 - 1 / v))
    ind = tuff(20) + tuff(10) + tuff(170)
    assert res.summary["LR_ind"] == pytest.approx(ind, rel=1e-12)
    pof = -2 * (247 * math.log(.99) + 3 * math.log(.01) - 247 * math.log(247 / 250) - 3 * math.log(3 / 250))
    assert res.summary["LR_mix"] == pytest.approx(ind + pof, rel=1e-12)
    assert res.summary["p_value_mix"] == pytest.approx(stats.chi2.sf(ind + pof, 4))


def test_duration_test_matches_direct_weibull_mle():
    rng = np.random.default_rng(3)
    hits = (rng.random(1000) < 0.03).astype(int)
    res = _ok(run_test("var.duration_weibull", _ctx(_hits_frame(hits)), BT | {"confidence": 0.97}))
    # independent implementation: full two-parameter censored Weibull likelihood
    t = np.flatnonzero(hits) + 1
    D = list(np.diff(t)); C = [0] * len(D)
    if t[0] > 1:
        D = [t[0]] + D; C = [1] + C
    if t[-1] < len(hits):
        D.append(len(hits) - t[-1]); C.append(1)
    D, C = np.array(D, float), np.array(C)

    def nll(th):
        a, b = np.exp(th)
        lf = b * np.log(a) + np.log(b) + (b - 1) * np.log(D) - (a * D) ** b
        ls = -(a * D) ** b
        return -np.where(C == 1, ls, lf).sum()
    u = optimize.minimize(nll, [np.log(0.03), 0.0], method="Nelder-Mead",
                          options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
    r = optimize.minimize_scalar(lambda la: nll([la, 0.0]), bounds=(-10, 2), method="bounded",
                                 options={"xatol": 1e-12})
    assert res.summary["weibull_b"] == pytest.approx(np.exp(u.x[1]), rel=1e-4)
    assert res.summary["LR"] == pytest.approx(2 * (r.fun - u.fun), abs=1e-5)
    few = np.zeros(100, int); few[[5, 50]] = 1
    assert run_test("var.duration_weibull", _ctx(_hits_frame(few)), BT).status == "not_applicable"


def test_dq_matches_statsmodels_ols():
    import statsmodels.api as sm
    rng = np.random.default_rng(5)
    n, lags, p = 500, 4, 0.05
    var = 1.645 * (1 + 0.3 * rng.random(n))
    pnl = rng.standard_normal(n) * 1.1
    df = pd.DataFrame({"pnl": pnl, "var": var})
    res = _ok(run_test("var.dynamic_quantile", _ctx(df), BT | {"confidence": 0.95, "lags": lags}))
    hit = ((-pnl) > var).astype(float) - p
    X = np.column_stack([np.ones(n - lags)] + [hit[lags - j:n - j] for j in range(1, lags + 1)] + [var[lags:]])
    fit = sm.OLS(hit[lags:], X).fit()
    expected = fit.fittedvalues @ fit.fittedvalues / (p * (1 - p))
    assert res.summary["DQ"] == pytest.approx(expected, rel=1e-9)
    assert res.summary["df"] == 6
    assert res.summary["p_value"] == pytest.approx(stats.chi2.sf(expected, 6))


# ── exceptions, signs, ordering, NaN ──────────────────────────────────────

def test_exceptions_signs_and_ordering(garch_like):
    base = _ok(run_test("var.exceptions", _ctx(garch_like), BT | {"date": "date"}))
    x = base.summary["exceptions"]
    assert x == int((-garch_like["pnl"] > garch_like["var"]).sum())
    assert base.figures
    # VaR given as negative returns is auto-detected
    neg = garch_like.assign(var=-garch_like["var"])
    r = _ok(run_test("var.exceptions", _ctx(neg), BT | {"date": "date"}))
    assert r.summary["exceptions"] == x and any("auto-detected" in n for n in r.notes)
    # losses given positive
    lp = garch_like.assign(pnl=-garch_like["pnl"])
    r = _ok(run_test("var.exceptions", _ctx(lp), BT | {"date": "date", "loss_positive": True}))
    assert r.summary["exceptions"] == x
    # shuffled rows with a date column give the same answer as sorted rows
    sh = garch_like.sample(frac=1, random_state=1)
    a = _ok(run_test("var.christoffersen", _ctx(sh), BT | {"date": "date"}))
    b = _ok(run_test("var.christoffersen", _ctx(garch_like), BT | {"date": "date"}))
    assert a.summary == b.summary


def test_mixed_sign_var_is_an_error_and_nans_are_dropped(garch_like):
    bad = garch_like.copy()
    bad.loc[0, "var"] = -1.0
    res = run_test("var.kupiec_pof", _ctx(bad), BT)
    assert res.status == "error" and "var_sign" in res.error
    res = _ok(run_test("var.kupiec_pof", _ctx(bad), BT | {"var_sign": "positive_loss"}))
    nan = garch_like.copy()
    nan.loc[[3, 4], "pnl"] = np.nan
    res = _ok(run_test("var.kupiec_pof", _ctx(nan), BT))
    assert res.rows_used == len(nan) - 2 and any("2 rows" in n for n in res.notes)


def test_rolling_exceptions_hand_count():
    hits = [1, 0, 0, 1, 1, 0, 0, 0]
    res = _ok(run_test("var.rolling_exceptions", _ctx(_hits_frame(hits)), BT | {"window": 4}))
    assert res.tables["Rolling exception count"]["exceptions"].tolist() == [2, 2, 2, 2, 1]
    assert res.summary["latest_exceptions"] == 1 and res.summary["max_exceptions"] == 2


# ── ES tests ───────────────────────────────────────────────────────────────

def test_acerbi_szekely_statistics_and_power(garch_like):
    df = garch_like.copy()
    params = BT | {"es": "es", "confidence": 0.975, "n_sims": 2000}
    res = _ok(run_test("var.es_acerbi_szekely", _ctx(df), params))
    L, V, E = -df["pnl"].to_numpy(), df["var"].to_numpy(), df["es"].to_numpy()
    I = L > V
    z2 = 1 - (I * L / E).sum() / (len(L) * 0.025)
    z1 = 1 - (L[I] / E[I]).mean()
    assert res.summary["Z2"] == pytest.approx(z2, rel=1e-12)
    assert res.summary["Z1"] == pytest.approx(z1, rel=1e-12)
    assert 0 <= res.summary["p_value_Z2"] <= 1
    # risk underestimated by a factor 2 -> Z2 strongly negative, tiny p-value
    under = df.assign(pnl=df["pnl"] * 2.5)
    r = _ok(run_test("var.es_acerbi_szekely", _ctx(under), params))
    assert r.summary["Z2"] < -1 and r.summary["p_value_Z2"] < 0.01
    # determinism
    again = _ok(run_test("var.es_acerbi_szekely", _ctx(df.copy()), params))
    assert again.summary == res.summary and again.run_id == res.run_id


def test_acerbi_szekely_correct_model_is_not_rejected():
    rng = np.random.default_rng(11)
    n, c = 1000, 0.975
    z = stats.norm.ppf(c)
    sig = 1 + rng.random(n)
    df = pd.DataFrame({"pnl": -sig * rng.standard_normal(n), "var": sig * z,
                       "es": sig * stats.norm.pdf(z) / (1 - c)})
    res = _ok(run_test("var.es_acerbi_szekely", _ctx(df), BT | {"es": "es", "confidence": c,
                                                              "distribution": "normal", "n_sims": 2000}))
    assert res.summary["p_value_Z2"] > 0.01


def test_es_requires_es_above_var(garch_like):
    bad = garch_like.assign(es=garch_like["var"] * 0.5)
    res = run_test("var.es_acerbi_szekely", _ctx(bad), BT | {"es": "es"})
    assert res.status == "error" and "ES" in res.error


def test_mcneil_frey(garch_like):
    params = BT | {"es": "es", "confidence": 0.975, "n_boot": 2000}
    res = _ok(run_test("var.es_mcneil_frey", _ctx(garch_like), params | {"volatility": "sig"}))
    L, E, V, s = -garch_like["pnl"], garch_like["es"], garch_like["var"], garch_like["sig"]
    r = ((L - E) / s)[L > V]
    assert res.summary["exceptions"] == len(r)
    assert res.summary["mean_residual"] == pytest.approx(r.mean())
    assert res.summary["t_statistic"] == pytest.approx(r.mean() / (r.std(ddof=1) / math.sqrt(len(r))))
    assert 0 <= res.summary["p_value_one_sided"] <= 1
    under = garch_like.assign(pnl=garch_like["pnl"] * 3)
    r2 = _ok(run_test("var.es_mcneil_frey", _ctx(under), params))
    assert r2.summary["p_value_one_sided"] < 0.01


# ── PIT tests ──────────────────────────────────────────────────────────────

def test_berkowitz_loglik_matches_statsmodels_arima():
    from statsmodels.tsa.arima.model import ARIMA
    rng = np.random.default_rng(2)
    n = 400
    e = rng.standard_normal(n)
    z = np.empty(n); z[0] = e[0]
    for t in range(1, n):
        z[t] = 0.2 + 0.4 * (z[t - 1] - 0.2) + 0.9 * e[t]
    df = pd.DataFrame({"pit": stats.norm.cdf(z)})
    res = _ok(run_test("var.berkowitz", _ctx(df), {"pit": "pit"}))
    fit = ARIMA(z, order=(1, 0, 0), trend="c").fit(method="innovations_mle")
    ll0 = stats.norm.logpdf(z).sum()
    assert res.summary["LR_3"] / 2 + ll0 == pytest.approx(fit.llf, abs=1e-3)
    assert res.summary["rho"] == pytest.approx(fit.params[1], abs=1e-3)
    assert res.summary["p_value_LR_3"] < 1e-6 and res.summary["p_value_LR_ind"] < 1e-6


def test_berkowitz_and_uniformity_accept_correct_pits(garch_like):
    res = _ok(run_test("var.berkowitz", _ctx(garch_like), {"pit": "pit", "date": "date"}))
    assert res.summary["p_value_LR_3"] > 0.01
    res = _ok(run_test("var.pit_uniformity", _ctx(garch_like), {"pit": "pit", "n_sims": 500}))
    t = res.tables["Uniformity tests"].set_index("test")
    u = garch_like["pit"].to_numpy()
    assert t.loc["Kolmogorov–Smirnov", "statistic"] == pytest.approx(stats.kstest(u, "uniform").statistic)
    assert t.loc["Cramér–von Mises", "statistic"] == pytest.approx(stats.cramervonmises(u, "uniform").statistic)
    gof = stats.goodness_of_fit(stats.uniform, u, known_params={"loc": 0, "scale": 1}, statistic="ad",
                                n_mc_samples=99, rng=np.random.default_rng(0))
    assert t.loc["Anderson–Darling", "statistic"] == pytest.approx(gof.statistic, rel=1e-9)
    assert (t["p_value"] > 0.001).all()


def test_pit_out_of_range_is_error():
    res = run_test("var.pit_uniformity", _ctx(pd.DataFrame({"pit": [0.1, 1.2] * 10})), {"pit": "pit"})
    assert res.status == "error"


# ── loss functions and PLA ────────────────────────────────────────────────

def test_loss_functions_hand_values():
    df = pd.DataFrame({"pnl": [-3.0, 1.0, -0.5], "var": [2.0, 2.0, 1.0], "es": [2.5, 2.5, 1.5],
                       "bvar": [3.5, 2.5, 1.5]})
    res = _ok(run_test("var.loss_functions", _ctx(df), BT | {"confidence": 0.9, "es": "es",
                                                         "benchmark_var": "bvar"}))
    L = np.array([3.0, -1.0, 0.5]); V = np.array([2.0, 2.0, 1.0])
    u = L - V
    ql = (u * (0.9 - (u < 0))).mean()
    assert res.summary["mean_quantile_loss"] == pytest.approx(ql)
    assert res.summary["lopez_magnitude_total"] == pytest.approx(1 + 1.0 ** 2)
    Y, v, e = -L, -V, -np.array([2.5, 2.5, 1.5])
    fz = (-((Y <= v) * (v - Y)) / (0.1 * e) + v / e + np.log(-e) - 1).mean()
    assert res.summary["mean_FZ0_loss"] == pytest.approx(fz)
    assert "DM_statistic" in res.summary


def test_pla_zones_and_metrics():
    rng = np.random.default_rng(4)
    h = rng.standard_normal(250)
    good = pd.DataFrame({"hpl": h, "rtpl": h + 0.05 * rng.standard_normal(250)})
    res = _ok(run_test("var.pla_test", _ctx(good), {"hpl": "hpl", "rtpl": "rtpl"}))
    assert res.summary["pla_zone"] == "green"
    assert res.summary["spearman"] == pytest.approx(stats.spearmanr(good["hpl"], good["rtpl"]).statistic)
    assert res.summary["ks_statistic"] == pytest.approx(stats.ks_2samp(good["hpl"], good["rtpl"]).statistic)
    bad = pd.DataFrame({"hpl": h, "rtpl": rng.standard_normal(250) * 2})
    assert _ok(run_test("var.pla_test", _ctx(bad), {"hpl": "hpl", "rtpl": "rtpl"})).summary["pla_zone"] == "red"


# ── replication ───────────────────────────────────────────────────────────

def test_hs_var_known_answer_and_layouts():
    pnl = -np.arange(1, 251, dtype=float)          # losses 1..250
    df = pd.DataFrame({"pnl": pnl})
    res = _ok(run_test("var.hs_var", _ctx(df), {"pnl": "pnl", "reported_var": 250.0}))
    assert res.summary["VaR"] == 248.0            # 3rd-largest loss of 250 at 99%
    assert res.summary["ES"] == pytest.approx((250 + 249 + 0.5 * 248) / 2.5)
    assert res.summary["VaR_difference"] == pytest.approx(2.0)
    # positions x scenarios: two positions that sum to the same strip
    tab = pd.DataFrame({"pos_a": pnl * 0.25, "pos_b": pnl * 0.75, "label": "x"})
    r = _ok(run_test("var.hs_var", RunContext(df=None, tables={"strip": tab}), {"scenarios": "strip"}))
    assert r.summary["VaR"] == pytest.approx(248.0) and r.summary["positions"] == 2
    r2 = _ok(run_test("var.hs_var", RunContext(df=None, tables={"strip": tab.drop(columns="label").T}),
                      {"scenarios": "strip", "layout": "columns_are_scenarios"}))
    assert r2.summary["VaR"] == pytest.approx(248.0)


def test_hs_rolling_replication_is_exact_for_hs_model():
    rng = np.random.default_rng(8)
    pnl = rng.standard_normal(400)
    loss = -pnl
    w = 100
    var = np.full(400, np.nan)
    for t in range(w, 400):
        var[t] = np.quantile(loss[t - w:t], 0.99, method="inverted_cdf")
    var[:w] = 1.0                                  # days without a full window are not compared
    df = pd.DataFrame({"pnl": pnl, "var": var})
    r = _ok(run_test("var.hs_rolling_replication", _ctx(df), BT | {"window": w}))
    assert r.summary["days_compared"] == 300
    assert r.tables["Replication by day"]["difference"].abs().max() == pytest.approx(0.0, abs=1e-12)


def test_parametric_var_matches_hand_formula():
    rng = np.random.default_rng(9)
    R = pd.DataFrame(rng.multivariate_normal([0, 0], [[1e-4, 3e-5], [3e-5, 4e-4]], size=500), columns=["eq", "fx"])
    w = {"eq": 1e6, "fx": -5e5}
    res = _ok(run_test("var.parametric_var", RunContext(tables={"ret": R}),
                       {"returns": "ret", "weights": w, "horizon_days": 10, "reported_var": 1e5}))
    S = np.cov(R.to_numpy(), rowvar=False)
    wv = np.array([1e6, -5e5])
    sp = math.sqrt(wv @ S @ wv)
    assert res.summary["VaR"] == pytest.approx(math.sqrt(10) * stats.norm.ppf(0.99) * sp)
    assert res.tables["Component VaR"]["component_VaR"].sum() == pytest.approx(res.summary["VaR"])
    bad = run_test("var.parametric_var", RunContext(tables={"ret": R}), {"returns": "ret", "weights": {"zz": 1}})
    assert bad.status == "error"


def test_scaling_and_stressed_period():
    rng = np.random.default_rng(12)
    pnl = rng.standard_normal(2000)
    pnl[1000:1250] *= 4                            # stressed year
    df = pd.DataFrame({"pnl": pnl, "date": pd.bdate_range("2010-01-01", periods=2000)})
    res = _ok(run_test("var.scaling_check", _ctx(df), {"pnl": "pnl", "date": "date"}))
    assert res.summary["VaR_scaled"] == pytest.approx(math.sqrt(10) * res.summary["VaR_1d"])
    res = _ok(run_test("var.stressed_period", _ctx(df), {"pnl": "pnl", "date": "date"}))
    start = pd.Timestamp(res.summary["stressed_start"])
    assert df["date"].iloc[900] <= start <= df["date"].iloc[1250]
    assert res.summary["stressed_VaR"] > 2 * res.summary["latest_VaR"]
