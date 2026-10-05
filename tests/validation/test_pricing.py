"""Pricing / valuation tests (ask/validation/t_pricing.py): textbook values, cross-checks, edge cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ask.validation.core import RunContext, run_test
from ask.validation.t_pricing import bs


def _ctx(df, **kw):
    return RunContext(df=df, source_name="px", **kw)


def _ok(res):
    assert res.status == "ok", res.error
    return res


def _chain(S=100.0, r=0.03, q=0.01, sig=0.25, Ks=(80, 90, 100, 110, 120), Ts=(0.5, 1.0)):
    rows = []
    for T in Ts:
        for K in Ks:
            c = float(bs(S, K, T, r, q, sig, True)["price"])
            p = float(bs(S, K, T, r, q, sig, False)["price"])
            rows.append({"S": S, "K": K, "T": T, "r": r, "q": q, "sig": sig, "call": c, "put": p,
                         "F": S * math.exp((r - q) * T)})
    return pd.DataFrame(rows)


# ── benchmark pricers ─────────────────────────────────────────────────────

def test_black_scholes_hull_textbook_values():
    df = pd.DataFrame({"S": [42.0, 42.0, 49.0], "K": [40.0, 40.0, 50.0], "T": [0.5, 0.5, 20 / 52],
                       "sig": [0.2, 0.2, 0.2], "r": [0.1, 0.1, 0.05], "type": ["call", "put", "c"]})
    res = _ok(run_test("pricing.black_scholes", _ctx(df), {"spot": "S", "strike": "K", "maturity": "T",
                                                           "vol": "sig", "rate": "r", "option_type": "type"}))
    t = res.tables["Benchmark prices and Greeks"]
    assert round(t["price"][0], 2) == 4.76 and round(t["price"][1], 2) == 0.81     # Hull Example 15.6
    g = t.iloc[2]                                                                  # Hull ch. 19 example
    assert round(g["price"], 2) == 2.40 and round(g["delta"], 3) == 0.522
    assert round(g["gamma"], 3) == 0.066 and round(g["vega"], 1) == 12.1
    assert round(g["theta"], 2) == -4.31 and round(g["rho"], 2) == 8.91


def test_black_scholes_dividend_parity_and_model_comparison():
    ch = _chain()
    ch["model"] = ch["call"] + 0.01
    res = _ok(run_test("pricing.black_scholes", _ctx(ch), {"spot": "S", "strike": "K", "maturity": "T", "vol": "sig",
                                                           "rate": "r", "dividend": "q", "model_price": "model"}))
    assert res.summary["max_abs_price_diff"] == pytest.approx(0.01, abs=1e-12)
    lhs = ch["call"] - ch["put"]
    rhs = ch["S"] * np.exp(-ch["q"] * ch["T"]) - ch["K"] * np.exp(-ch["r"] * ch["T"])
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_invalid_rows_are_excluded():
    df = pd.DataFrame({"S": [100.0, -1.0, np.nan], "K": [100.0] * 3, "T": [1.0] * 3, "sig": [0.2] * 3})
    res = _ok(run_test("pricing.black_scholes", _ctx(df), {"spot": "S", "strike": "K", "maturity": "T", "vol": "sig"}))
    assert res.rows_used == 1 and len(res.notes) == 2
    res = run_test("pricing.black_scholes", _ctx(df.iloc[1:]), {"spot": "S", "strike": "K", "maturity": "T", "vol": "sig"})
    assert res.status == "not_applicable"


def test_black76_hull_futures_put():
    df = pd.DataFrame({"F": [20.0], "K": [20.0], "T": [4 / 12], "sig": [0.25]})
    res = _ok(run_test("pricing.black76", _ctx(df), {"forward": "F", "strike": "K", "maturity": "T", "vol": "sig",
                                                      "rate_value": 0.09, "default_type": "put"}))
    assert round(res.tables["Benchmark prices and Greeks"]["price"][0], 2) == 1.12   # Hull Example 18.7


def test_bachelier_atm_closed_form_and_parity():
    df = pd.DataFrame({"F": [0.01, 0.01, -0.002], "K": [0.01, 0.01, 0.0], "T": [2.0, 2.0, 1.0],
                       "sig": [0.008, 0.008, 0.006], "type": ["call", "put", "call"]})
    res = _ok(run_test("pricing.bachelier", _ctx(df), {"forward": "F", "strike": "K", "maturity": "T", "vol": "sig",
                                                        "option_type": "type", "rate_value": 0.02}))
    t = res.tables["Benchmark prices and Greeks"]
    atm = math.exp(-0.04) * 0.008 * math.sqrt(2) / math.sqrt(2 * math.pi)
    assert t["price"][0] == pytest.approx(atm) and t["price"][1] == pytest.approx(atm)
    assert res.rows_used == 3          # negative forward allowed in the normal model


# ── parity and arbitrage ──────────────────────────────────────────────────

def test_put_call_parity_and_implied_forward():
    ch = _chain()
    res = _ok(run_test("pricing.put_call_parity", _ctx(ch), {"strike": "K", "maturity": "T", "call_price": "call",
                                                             "put_price": "put", "spot": "S", "rate": "r",
                                                             "dividend": "q"}))
    assert res.summary["max_abs_deviation"] < 1e-10
    imp = res.tables["Implied discount factor and forward"]
    assert np.allclose(imp["implied_DF"], imp["input_DF"]) and np.allclose(imp["implied_forward"], imp["input_forward"])
    ch.loc[2, "put"] += 0.5
    res = _ok(run_test("pricing.put_call_parity", _ctx(ch), {"strike": "K", "maturity": "T", "call_price": "call",
                                                             "put_price": "put", "forward": "F", "rate": "r"}))
    assert res.summary["max_abs_deviation"] == pytest.approx(0.5)


def test_strike_arbitrage_detects_violations():
    ch = _chain()
    p = {"strike": "K", "maturity": "T", "price": "call", "forward": "F", "rate": "r"}
    clean = _ok(run_test("pricing.strike_arbitrage", _ctx(ch), p))
    assert clean.summary["convexity_violations"] == 0 and clean.summary["monotonicity_violations"] == 0
    assert clean.summary["price_bounds_violations"] == 0
    bad = ch.copy()
    bad.loc[2, "call"] += 3.0            # ATM call too expensive: butterfly negative
    r = _ok(run_test("pricing.strike_arbitrage", _ctx(bad), p))
    assert r.summary["convexity_violations"] >= 1
    bad.loc[3, "call"] = bad.loc[2, "call"] + 1      # higher strike more expensive
    r = _ok(run_test("pricing.strike_arbitrage", _ctx(bad), p))
    assert r.summary["monotonicity_violations"] >= 1
    puts = ch.assign(type="put")
    r = _ok(run_test("pricing.strike_arbitrage", _ctx(puts), {"strike": "K", "maturity": "T", "price": "put",
                                                              "option_type": "type", "rate": "r"}))
    assert r.summary["monotonicity_violations"] == 0 and r.summary["convexity_violations"] == 0
    r = _ok(run_test("pricing.strike_arbitrage", _ctx(ch), {"strike": "K", "maturity": "T", "vol": "sig",
                                                            "forward": "F"}))
    assert r.summary["max_violation"] == 0.0


def test_calendar_arbitrage():
    rows = [{"K": k, "T": T, "sig": s, "S": 100.0} for T, s in ((0.5, 0.2), (1.0, 0.2), (2.0, 0.2))
            for k in (80, 100, 120)]
    df = pd.DataFrame(rows)
    p = {"strike": "K", "maturity": "T", "vol": "sig", "spot": "S"}
    assert _ok(run_test("pricing.calendar_arbitrage", _ctx(df), p)).summary["violations"] == 0
    df.loc[df["T"] == 1.0, "sig"] = 0.1    # w(1y) = 0.01 < w(0.5y) = 0.02
    r = _ok(run_test("pricing.calendar_arbitrage", _ctx(df), p))
    assert r.summary["violations"] >= 3
    assert r.summary["max_violation"] == pytest.approx(0.02 - 0.01)


# ── Greeks and Monte Carlo ────────────────────────────────────────────────

def test_fd_greeks_against_bs_delta_gamma():
    ch = _chain()
    h = 0.01
    args = (ch["K"], ch["T"], ch["r"], ch["q"], ch["sig"], True)
    ch["V"] = bs(ch["S"], *args)["price"]
    ch["Vu"] = bs(ch["S"] * (1 + h), *args)["price"]
    ch["Vd"] = bs(ch["S"] * (1 - h), *args)["price"]
    g = bs(ch["S"], *args)
    ch["delta"], ch["gamma"] = g["delta"], g["gamma"]
    res = _ok(run_test("pricing.fd_greeks", _ctx(ch), {"price": "V", "price_up": "Vu", "price_down": "Vd", "bump": h,
                                                       "level": "S", "reported_first": "delta",
                                                       "reported_second": "gamma"}))
    assert res.summary["max_abs_first_diff"] < 5e-4          # O(h^2) truncation with a 1% bump
    assert res.summary["max_abs_second_diff"] < 1e-4


def test_mc_convergence_hand_values():
    x = np.random.default_rng(1).exponential(2.0, 4096)
    res = _ok(run_test("pricing.mc_convergence", _ctx(pd.DataFrame({"pay": x})), {"payoff": "pay", "benchmark": 2.0}))
    assert res.summary["estimate"] == pytest.approx(x.mean())
    assert res.summary["std_error"] == pytest.approx(x.std(ddof=1) / 64)
    assert -0.65 < res.summary["log_se_slope"] < -0.35
    assert res.summary["z_score"] == pytest.approx((x.mean() - 2.0) / (x.std(ddof=1) / 64))
    assert run_test("pricing.mc_convergence", _ctx(pd.DataFrame({"pay": [1.0] * 5})),
                    {"payoff": "pay"}).status == "not_applicable"


def test_mc_gbm_check_agrees_with_closed_form_and_is_deterministic():
    p = {"spot": 100, "strike": 105, "maturity": 1.0, "vol": 0.2, "rate": 0.03, "n_paths": 20000}
    a = _ok(run_test("pricing.mc_gbm_check", RunContext(), p))
    assert a.summary["benchmark"] == pytest.approx(float(bs(100, 105, 1, 0.03, 0, 0.2, True)["price"]))
    assert abs(a.summary["z_score"]) < 4
    b = _ok(run_test("pricing.mc_gbm_check", RunContext(), p))
    assert a.summary == b.summary and a.run_id == b.run_id


def test_implied_vol_round_trip_and_bounds():
    ch = _chain()
    ch = pd.concat([ch, ch.iloc[[0]].assign(call=0.001)], ignore_index=True)   # below intrinsic
    res = _ok(run_test("pricing.implied_vol", _ctx(ch), {"price": "call", "strike": "K", "maturity": "T",
                                                         "underlying": "S", "rate": "r", "dividend": "q",
                                                         "reported_vol": "sig"}))
    t = res.tables["Implied volatilities"]
    assert res.summary["outside_bounds"] == 1
    assert np.allclose(t.loc[t["status"] == "ok", "implied_vol"], 0.25, atol=1e-8)
    assert res.summary["max_abs_roundtrip_error"] < 1e-9
    df = pd.DataFrame({"F": [0.01], "K": [0.012], "T": [1.0], "p": [0.002]})
    r = _ok(run_test("pricing.implied_vol", _ctx(df), {"price": "p", "strike": "K", "maturity": "T",
                                                       "underlying": "F", "model": "bachelier"}))
    iv = r.tables["Implied volatilities"]["implied_vol"][0]
    from ask.validation.t_pricing import bachelier
    assert float(bachelier(0.01, 0.012, 1.0, 0.0, iv, True)["price"]) == pytest.approx(0.002, abs=1e-12)


# ── curves and bonds ──────────────────────────────────────────────────────

def test_curve_diagnostics_flat_curve_and_consistency():
    t = np.array([0.5, 1, 2, 5, 10.0])
    df = pd.DataFrame({"t": t, "z": 0.03, "df": np.exp(-0.03 * t)})
    res = _ok(run_test("pricing.curve_diagnostics", _ctx(df), {"maturity": "t", "zero_rate": "z",
                                                               "discount_factor": "df"}))
    assert np.allclose(res.tables["Curve nodes"]["forward_rate"], 0.03)
    assert res.summary["max_abs_df_difference"] < 1e-15 and res.summary["negative_forwards"] == 0
    kinked = pd.DataFrame({"t": t, "df": [0.99, 0.98, 0.985, 0.9, 0.8]})
    r = _ok(run_test("pricing.curve_diagnostics", _ctx(kinked), {"maturity": "t", "discount_factor": "df"}))
    assert r.summary["df_increases"] == 1 and r.summary["negative_forwards"] == 1
    ann = _ok(run_test("pricing.curve_diagnostics", _ctx(pd.DataFrame({"t": [1.0, 2.0], "z": [0.02, 0.02]})),
                       {"maturity": "t", "zero_rate": "z", "compounding": "annual"}))
    assert ann.tables["Curve nodes"]["discount_factor"].tolist() == pytest.approx([1 / 1.02, 1 / 1.02 ** 2])


def test_curve_repricing_flat_curve():
    curve = pd.DataFrame({"maturity": [0.5, 1, 3, 5, 10.0], "zero": 0.03})
    T, c = 4.5, 0.05
    times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    dirty = (5 * np.exp(-0.03 * times)).sum() + 100 * math.exp(-0.03 * 4.5)
    bonds = pd.DataFrame({"c": [c, c], "T": [T, T], "px": [dirty - 2.5, dirty - 2.4]})
    res = _ok(run_test("pricing.curve_repricing", RunContext(df=bonds, tables={"curve": curve}),
                       {"coupon": "c", "maturity": "T", "market_price": "px", "curve": "curve",
                        "curve_value": "zero", "curve_value_type": "zero_rate", "price_type": "clean"}))
    out = res.tables["Repricing"]
    assert out["curve_price"][0] == pytest.approx(dirty - 2.5)       # accrued = half a 5% annual coupon
    assert out["difference"].tolist() == pytest.approx([0.0, 0.1], abs=1e-10)


def test_bond_analytics_hull_and_par_bond():
    df = pd.DataFrame({"c": [0.10], "T": [3.0], "y": [0.12]})
    res = _ok(run_test("pricing.bond_analytics", _ctx(df), {"coupon": "c", "maturity": "T", "ytm": "y",
                                                            "compounding": "continuous"}))
    assert round(res.summary["dirty_price"], 3) == 94.213                     # Hull Table 4.6
    assert round(res.tables["Bond analytics"]["macaulay_duration"][0], 3) == 2.653
    par = pd.DataFrame({"c": [0.06, 0.0], "T": [10.0, 7.0], "y": [0.06, 0.04]})
    r = _ok(run_test("pricing.bond_analytics", _ctx(par), {"coupon": "c", "maturity": "T", "ytm": "y"}))
    t = r.tables["Bond analytics"]
    assert t["dirty_price"][0] == pytest.approx(100.0)
    assert t["macaulay_duration"][1] == pytest.approx(7.0)
    assert t["modified_duration"][1] == pytest.approx(7.0 / 1.02)
    # convexity agrees with a finite difference of the price in yield
    from ask.validation.t_pricing import _bond_cfs, _bond_from_yield
    tt, cf, _ = _bond_cfs(0.06, 10.0, 2, 100.0)
    h = 1e-4
    pu, p0, pd_ = (_bond_from_yield(tt, cf, 0.06 + s, 2, "periodic")[0] for s in (h, 0, -h))
    assert t["convexity"][0] == pytest.approx((pu - 2 * p0 + pd_) / (h * h * p0), rel=1e-4)
    # yield solved from price round-trips
    y = _ok(run_test("pricing.bond_analytics", _ctx(t.assign(c=par["c"], T=par["T"])),
                     {"coupon": "c", "maturity": "T", "price": "dirty_price"}))
    assert y.tables["Bond analytics"]["ytm"].tolist() == pytest.approx([0.06, 0.04], abs=1e-10)


# ── P&L explain and stress ────────────────────────────────────────────────

def test_pnl_explain_hand_values():
    df = pd.DataFrame({"full": [1.0, -2.0, 3.0, 0.5], "risk": [0.8, -2.1, 2.5, 0.6]})
    res = _ok(run_test("pricing.pnl_explain", _ctx(df), {"actual": "full", "predicted": "risk"}))
    U = df["full"] - df["risk"]
    assert res.summary["variance_ratio"] == pytest.approx(U.var() / df["full"].var())
    assert res.summary["mean_ratio"] == pytest.approx(U.mean() / df["full"].std())
    assert res.summary["abs_unexplained_ratio"] == pytest.approx(U.abs().sum() / df["full"].abs().sum())


def test_stress_repricing():
    ch = _chain()
    ch["qty"] = [1, -1] * 5
    res = _ok(run_test("pricing.stress_repricing", _ctx(ch), {"spot": "S", "strike": "K", "maturity": "T",
                                                              "vol": "sig", "rate": "r", "dividend": "q",
                                                              "quantity": "qty", "spot_shocks": [-0.01, 0, 0.01],
                                                              "vol_shocks": [0.0]}))
    t = res.tables["Scenario P&L"].set_index("spot_shock")
    assert t.loc[0.0, "full_reval_pnl"] == pytest.approx(0.0, abs=1e-12)
    assert abs(t.loc[0.01, "approximation_error"]) < 1e-3
    S1 = ch["S"] * 1.01
    exp = (ch["qty"] * (bs(S1, ch["K"], ch["T"], ch["r"], ch["q"], ch["sig"], True)["price"]
                        - bs(ch["S"], ch["K"], ch["T"], ch["r"], ch["q"], ch["sig"], True)["price"])).sum()
    assert t.loc[0.01, "full_reval_pnl"] == pytest.approx(exp)
