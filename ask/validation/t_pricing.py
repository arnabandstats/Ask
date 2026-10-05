"""Pricing and valuation models: independent benchmark pricers and consistency checks.

Benchmarks: Black–Scholes–Merton (continuous dividend yield), Black-76 and Bachelier (normal) with Greeks;
bond pricing from a yield or a discount curve with duration and convexity; implied volatility by Brent.
Checks: model-vs-benchmark price differences, put–call parity, static no-arbitrage on option chains and
volatility surfaces, finite-difference vs reported Greeks, Monte Carlo convergence, yield-curve
diagnostics and instrument repricing, P&L explain, and stress repricing of an option book.

Conventions: maturities and times in years; rates and dividend yields continuously compounded decimals;
volatilities as decimals (0.20 = 20%). Theta is per year (divide by 365 or 252 for per day); vega and
rho are per 1.00 change (divide by 100 for per vol/rate point). Rows with invalid inputs (non-positive
spot/strike/maturity/vol; forward and strike may be <= 0 in the Bachelier model) are excluded and counted
in the notes.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, dropped_note, num, register

_MT = ("pricing",)
_N, _n = stats.norm.cdf, stats.norm.pdf

_HULL = "Hull, J. C., Options, Futures, and Other Derivatives (10th ed., 2018)"


# ── closed-form pricers (vectorised; is_call boolean array) ────────────────

def bs(S, K, T, r, q, sig, is_call):
    """Black–Scholes–Merton price and Greeks (delta, gamma, vega, theta per year, rho)."""
    S, K, T, r, q, sig = (np.asarray(a, float) for a in (S, K, T, r, q, sig))
    c = np.asarray(is_call, bool)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sq)
    d2 = d1 - sig * sq
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    call = S * dq * _N(d1) - K * dr * _N(d2)
    put = K * dr * _N(-d2) - S * dq * _N(-d1)
    common_theta = -S * dq * _n(d1) * sig / (2 * sq)
    return {
        "price": np.where(c, call, put),
        "delta": np.where(c, dq * _N(d1), dq * (_N(d1) - 1)),
        "gamma": dq * _n(d1) / (S * sig * sq),
        "vega": S * dq * _n(d1) * sq,
        "theta": np.where(c, common_theta - r * K * dr * _N(d2) + q * S * dq * _N(d1),
                          common_theta + r * K * dr * _N(-d2) - q * S * dq * _N(-d1)),
        "rho": np.where(c, K * T * dr * _N(d2), -K * T * dr * _N(-d2)),
    }


def black76(F, K, T, r, sig, is_call):
    """Black (1976) on a forward F, discounted at r. Delta/gamma w.r.t. F; theta with F fixed; rho = dV/dr."""
    F, K, T, r, sig = (np.asarray(a, float) for a in (F, K, T, r, sig))
    c = np.asarray(is_call, bool)
    sq = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig ** 2 * T) / (sig * sq)
    d2 = d1 - sig * sq
    D = np.exp(-r * T)
    price = np.where(c, D * (F * _N(d1) - K * _N(d2)), D * (K * _N(-d2) - F * _N(-d1)))
    return {"price": price, "delta": np.where(c, D * _N(d1), D * (_N(d1) - 1)),
            "gamma": D * _n(d1) / (F * sig * sq), "vega": D * F * _n(d1) * sq,
            "theta": r * price - D * F * _n(d1) * sig / (2 * sq), "rho": -T * price}


def bachelier(F, K, T, r, sig, is_call):
    """Bachelier (normal) model on a forward with absolute (normal) volatility sig."""
    F, K, T, r, sig = (np.asarray(a, float) for a in (F, K, T, r, sig))
    c = np.asarray(is_call, bool)
    s = sig * np.sqrt(T)
    d = (F - K) / s
    D = np.exp(-r * T)
    price = np.where(c, D * ((F - K) * _N(d) + s * _n(d)), D * ((K - F) * _N(-d) + s * _n(d)))
    return {"price": price, "delta": np.where(c, D * _N(d), D * (_N(d) - 1)), "gamma": D * _n(d) / s,
            "vega": D * np.sqrt(T) * _n(d), "theta": r * price - D * sig * _n(d) / (2 * np.sqrt(T)),
            "rho": -T * price}


# ── input helpers ──────────────────────────────────────────────────────────

def _is_call(ctx, option_type, default_type, n):
    if not option_type:
        return np.full(n, default_type == "call")
    s = ctx.df[option_type].astype(str).str.strip().str.lower()
    m = s.map({"call": True, "c": True, "put": False, "p": False, "1": True, "-1": False})
    if m.isna().any():
        bad = sorted(set(s[m.isna()]))[:6]
        raise ValueError(f"option_type values must be call/put (or c/p); found {bad}")
    return m.to_numpy(bool)


def _frame(ctx, cols: dict, consts: dict) -> pd.DataFrame:
    """Numeric frame from column params (None = absent) and constant fallbacks for absent ones."""
    d = pd.DataFrame(index=ctx.df.index)
    for k, c in cols.items():
        if c:
            d[k] = num(ctx.df, c)
        elif k in consts:
            d[k] = float(consts[k])
    return d


def _valid(d: pd.DataFrame, positive: list[str], notes: list[str]) -> pd.Series:
    ok = d.notna().all(axis=1)
    miss = int((~ok).sum())
    for c in positive:
        ok &= d[c] > 0
    bad = int((~ok).sum()) - miss
    notes += dropped_note(miss)
    if bad:
        notes.append(f"{bad} rows with non-positive {'/'.join(positive)} excluded.")
    if not ok.any():
        raise NotApplicable("No rows with valid inputs.")
    return ok


def _compare(out: pd.DataFrame, d: pd.DataFrame, pairs: list[tuple[str, str]], summ: dict) -> None:
    for bench, model in pairs:
        if model in d:
            diff = d[model].to_numpy() - out[bench].to_numpy()
            out[f"model_{bench}"] = d[model].to_numpy()
            out[f"{bench}_diff"] = diff
            out[f"{bench}_rel_diff"] = diff / np.where(out[bench] != 0, out[bench], np.nan)
            summ[f"max_abs_{bench}_diff"] = float(np.nanmax(np.abs(diff)))
            summ[f"mean_abs_{bench}_diff"] = float(np.nanmean(np.abs(diff)))
            summ[f"max_abs_{bench}_rel_diff"] = float(np.nanmax(np.abs(out[f"{bench}_rel_diff"])))


_TYPE = (P("option_type", required=False, help="Column with call/put (c/p); default `default_type`"),
         P("default_type", "string", default="call", choices=("call", "put")))
_RATE = (P("rate", required=False, help="Risk-free rate column (continuous)"),
         P("rate_value", "number", default=0.0, help="Constant rate when no rate column"))
_DIV = (P("dividend", required=False, help="Dividend-yield column (continuous)"),
        P("dividend_value", "number", default=0.0, help="Constant dividend yield when no column"))
_MODELCMP = (P("model_price", required=False, help="Model (bank) price column to compare"),
             P("model_delta", required=False), P("model_gamma", required=False), P("model_vega", required=False))
_GREEKS = ["price", "delta", "gamma", "vega", "theta", "rho"]


def _benchmark(ctx, kind, under, strike, maturity, vol, option_type, default_type, rate, rate_value,
               dividend, dividend_value, model_price, model_delta, model_gamma, model_vega):
    notes = []
    d = _frame(ctx, {"U": under, "K": strike, "T": maturity, "sig": vol, "r": rate, "q": dividend,
                     "mp": model_price, "md": model_delta, "mg": model_gamma, "mv": model_vega},
               {"r": rate_value, "q": dividend_value})
    call = _is_call(ctx, option_type, default_type, len(d))
    pos = ["T", "sig"] + (["U", "K"] if kind != "bachelier" else [])
    ok = _valid(d[["U", "K", "T", "sig", "r", "q"]], pos, notes)
    d, call = d[ok], call[ok.to_numpy()]
    if kind == "bs":
        g = bs(d["U"], d["K"], d["T"], d["r"], d["q"], d["sig"], call)
    elif kind == "black76":
        g = black76(d["U"], d["K"], d["T"], d["r"], d["sig"], call)
    else:
        g = bachelier(d["U"], d["K"], d["T"], d["r"], d["sig"], call)
    out = pd.DataFrame({"row": d.index.to_numpy(), "type": np.where(call, "call", "put"),
                        **{k: np.asarray(g[k], float) for k in _GREEKS}})
    summ = {"options": len(out)}
    pairs = [("price", "mp"), ("delta", "md"), ("gamma", "mg"), ("vega", "mv")]
    _compare(out, d, pairs, summ)
    return Outcome(summ, {"Benchmark prices and Greeks": out}, notes=notes, rows_used=len(out))


@register("pricing.black_scholes", "Black–Scholes–Merton benchmark price and Greeks", "Benchmarking", _MT,
          params=(P("spot"), P("strike"), P("maturity", help="Time to expiry in years"),
                  P("vol", help="Lognormal volatility (decimal)"), *_TYPE, *_RATE, *_DIV, *_MODELCMP),
          description="""European option on an asset with continuous dividend yield q:
C = S e^(−qT) N(d1) − K e^(−rT) N(d2), P = K e^(−rT) N(−d2) − S e^(−qT) N(−d1),
d1 = [ln(S/K) + (r − q + σ²/2)T]/(σ√T), d2 = d1 − σ√T; analytic delta, gamma, vega, theta (per year), rho.
When model columns are given, reports model − benchmark differences (absolute and relative) per option and
their maximum/mean. Assumes constant r, q, σ and European exercise.""",
          references=("Black, F. and Scholes, M. (1973), The pricing of options and corporate liabilities, JPE 81(3)",
                      "Merton, R. C. (1973), Theory of rational option pricing, Bell Journal of Economics 4(1)",
                      _HULL))
def black_scholes(ctx: RunContext, spot, strike, maturity, vol, option_type=None, default_type="call",
                  rate=None, rate_value=0.0, dividend=None, dividend_value=0.0, model_price=None,
                  model_delta=None, model_gamma=None, model_vega=None) -> Outcome:
    return _benchmark(ctx, "bs", spot, strike, maturity, vol, option_type, default_type, rate, rate_value,
                      dividend, dividend_value, model_price, model_delta, model_gamma, model_vega)


@register("pricing.black76", "Black-76 benchmark price and Greeks (options on forwards/futures)", "Benchmarking",
          _MT, params=(P("forward"), P("strike"), P("maturity"), P("vol"), *_TYPE, *_RATE, *_MODELCMP),
          description="""C = e^(−rT)[F N(d1) − K N(d2)], P = e^(−rT)[K N(−d2) − F N(−d1)],
d1 = [ln(F/K) + σ²T/2]/(σ√T). Delta and gamma are with respect to the forward; theta holds F fixed; rho is
the sensitivity to the discount rate. Used for caps/floors, swaptions (with annuity as discount factor) and
futures options. Model columns give model − benchmark differences.""",
          references=("Black, F. (1976), The pricing of commodity contracts, Journal of Financial Economics 3, 167–179",
                      _HULL))
def black76_test(ctx: RunContext, forward, strike, maturity, vol, option_type=None, default_type="call",
                 rate=None, rate_value=0.0, model_price=None, model_delta=None, model_gamma=None,
                 model_vega=None) -> Outcome:
    return _benchmark(ctx, "black76", forward, strike, maturity, vol, option_type, default_type, rate,
                      rate_value, None, 0.0, model_price, model_delta, model_gamma, model_vega)


@register("pricing.bachelier", "Bachelier (normal) model benchmark price and Greeks", "Benchmarking", _MT,
          params=(P("forward"), P("strike"), P("maturity"), P("vol", help="Normal (absolute) volatility"),
                  *_TYPE, *_RATE, *_MODELCMP),
          description="""Normal model, appropriate for rates that can be negative:
C = e^(−rT)[(F − K) N(d) + σ√T φ(d)], P = e^(−rT)[(K − F) N(−d) + σ√T φ(d)], d = (F − K)/(σ√T),
σ the absolute (normal / basis-point) volatility. Greeks with respect to F and σ; theta holds F fixed.
Model columns give model − benchmark differences.""",
          references=("Bachelier, L. (1900), Théorie de la spéculation, Annales scientifiques de l'É.N.S. 17",
                      "Schachermayer, W. and Teichmann, J. (2008), How close are the option pricing formulas of "
                      "Bachelier and Black–Merton–Scholes?, Mathematical Finance 18(1), 155–170"))
def bachelier_test(ctx: RunContext, forward, strike, maturity, vol, option_type=None, default_type="call",
                   rate=None, rate_value=0.0, model_price=None, model_delta=None, model_gamma=None,
                   model_vega=None) -> Outcome:
    return _benchmark(ctx, "bachelier", forward, strike, maturity, vol, option_type, default_type, rate,
                      rate_value, None, 0.0, model_price, model_delta, model_gamma, model_vega)


# ── parity and static arbitrage ───────────────────────────────────────────

@register("pricing.put_call_parity", "Put–call parity check across an option chain", "No-arbitrage", _MT,
          params=(P("strike"), P("maturity"), P("call_price"), P("put_price"),
                  P("spot", required=False, help="Spot (with rate/dividend), or give `forward`"),
                  P("forward", required=False), *_RATE, *_DIV),
          description="""European put–call parity: C − P = e^(−rT)(F − K) with F = S e^((r−q)T), i.e.
C − P = S e^(−qT) − K e^(−rT). Reports the parity deviation per strike/maturity. For every maturity with at
least two strikes it also regresses C − P on K: slope = −DF and intercept = DF·F, giving the implied discount
factor/rate and implied forward, compared with the inputs. Deviations reflect bid/ask, American early
exercise, discrete dividends, funding or data errors.""",
          references=("Stoll, H. R. (1969), The relationship between put and call option prices, Journal of "
                      "Finance 24(5), 801–824", _HULL))
def put_call_parity(ctx: RunContext, strike, maturity, call_price, put_price, spot=None, forward=None,
                    rate=None, rate_value=0.0, dividend=None, dividend_value=0.0) -> Outcome:
    if (spot is None) == (forward is None):
        raise ValueError("Give exactly one of `spot` or `forward`.")
    notes = []
    d = _frame(ctx, {"K": strike, "T": maturity, "C": call_price, "P": put_price, "S": spot, "F": forward,
                     "r": rate, "q": dividend}, {"r": rate_value, "q": dividend_value})
    ok = _valid(d, ["K", "T"], notes)
    d = d[ok]
    DF = np.exp(-d["r"] * d["T"])
    F = d["F"] if forward else d["S"] * np.exp((d["r"] - d["q"]) * d["T"])
    theo = DF * (F - d["K"])
    dev = (d["C"] - d["P"]) - theo
    out = pd.DataFrame({"row": d.index, "maturity": d["T"], "strike": d["K"], "call_minus_put": d["C"] - d["P"],
                        "parity_value": theo, "deviation": dev,
                        "deviation_pct_of_strike": dev / d["K"]}).sort_values(["maturity", "strike"])
    imp = []
    for T, g in d.groupby("T", sort=True):
        if g["K"].nunique() >= 2:
            b, a = np.polyfit(g["K"], g["C"] - g["P"], 1)
            dfi = -b
            imp.append({"maturity": T, "strikes": len(g), "implied_DF": dfi,
                        "implied_rate": -math.log(dfi) / T if dfi > 0 else float("nan"),
                        "input_DF": float(np.exp(-g["r"].iloc[0] * T)),
                        "implied_forward": a / dfi if dfi else float("nan"),
                        "input_forward": float(F[g.index].iloc[0])})
    tabs = {"Parity deviations": out}
    if imp:
        tabs["Implied discount factor and forward"] = pd.DataFrame(imp)
    return Outcome({"options": len(out), "max_abs_deviation": float(dev.abs().max()),
                    "mean_abs_deviation": float(dev.abs().mean()), "mean_deviation": float(dev.mean())},
                   tabs, notes=notes, rows_used=len(out))


@register("pricing.strike_arbitrage", "Strike no-arbitrage checks: monotonicity, slope, convexity (butterfly)",
          "No-arbitrage", _MT,
          params=(P("strike"), P("maturity"),
                  P("price", required=False, help="Option price column; or give `vol` and `forward`"),
                  P("vol", required=False, help="Implied vol column (prices computed with Black-76)"),
                  P("forward", required=False, help="Forward column (needed with `vol`, and for price bounds)"),
                  *_TYPE, *_RATE,
                  P("tolerance", "number", default=0.0, help="Violations smaller than this are not counted")),
          description="""Per maturity, sorted by strike, for calls (puts analogously with reversed signs):
(1) monotonicity: C(K_i) − C(K_{i+1}) >= 0; (2) vertical-spread slope bound: (C(K_i) − C(K_{i+1}))/(K_{i+1} − K_i)
<= DF = e^(−rT) (DF = 1 when no rate is given, a weaker bound for positive rates);
(3) convexity / butterfly: C(K_{i−1})(K_{i+1} − K_i) − C(K_i)(K_{i+1} − K_{i−1}) + C(K_{i+1})(K_i − K_{i−1}) >= 0;
(4) with a forward: price bounds DF·max(F − K, 0) <= C <= DF·F (puts: DF·max(K − F, 0) <= P <= DF·K).
Lists every violation with its size. With `vol` instead of `price`, Black-76 prices are checked
(the same as checking the implied-vol smile for butterfly arbitrage).""",
          references=("Carr, P. and Madan, D. B. (2005), A note on sufficient conditions for no arbitrage, "
                      "Finance Research Letters 2(3), 125–130",
                      "Gatheral, J. (2006), The Volatility Surface: A Practitioner's Guide, Wiley"))
def strike_arbitrage(ctx: RunContext, strike, maturity, price=None, vol=None, forward=None, option_type=None,
                     default_type="call", rate=None, rate_value=0.0, tolerance=0.0) -> Outcome:
    if price is None and (vol is None or forward is None):
        raise ValueError("Give `price`, or `vol` together with `forward`.")
    notes = []
    d = _frame(ctx, {"K": strike, "T": maturity, "V": price, "sig": vol, "F": forward, "r": rate},
               {"r": rate_value})
    d["call"] = _is_call(ctx, option_type, default_type, len(d))
    ok = _valid(d, ["K", "T"] + (["sig", "F"] if price is None else []), notes)
    d = d[ok].copy()
    if price is None:
        d["V"] = black76(d["F"], d["K"], d["T"], d["r"], d["sig"], d["call"])["price"]
    rows = []
    checks = {"monotonicity": 0, "slope_bound": 0, "convexity": 0, "price_bounds": 0}
    if rate is None and rate_value == 0.0:
        notes.append("No rate given: DF = 1 in the slope bound and price bounds.")
    for (T, call), g in d.groupby(["T", "call"], sort=True):
        if g["K"].duplicated().any():
            raise ValueError(f"Duplicate strikes at maturity {T} ({'calls' if call else 'puts'}).")
        g = g.sort_values("K")
        K, V = g["K"].to_numpy(), g["V"].to_numpy()
        DF = float(np.exp(-g["r"].iloc[0] * T))
        sgn = 1 if call else -1         # calls decrease in K, puts increase
        typ = "call" if call else "put"
        for i in range(len(K) - 1):
            drop = sgn * (V[i] - V[i + 1])
            if -drop > tolerance:
                rows.append({"check": "monotonicity", "type": typ, "maturity": T, "strikes": f"{K[i]:g}/{K[i + 1]:g}",
                             "violation": -drop})
            slope = drop / (K[i + 1] - K[i])
            if (slope - DF) * (K[i + 1] - K[i]) > tolerance:
                rows.append({"check": "slope_bound", "type": typ, "maturity": T,
                             "strikes": f"{K[i]:g}/{K[i + 1]:g}", "violation": (slope - DF) * (K[i + 1] - K[i])})
        for i in range(1, len(K) - 1):
            bf = V[i - 1] * (K[i + 1] - K[i]) - V[i] * (K[i + 1] - K[i - 1]) + V[i + 1] * (K[i] - K[i - 1])
            bf /= (K[i + 1] - K[i])           # butterfly in units of the right wing
            if -bf > tolerance:
                rows.append({"check": "convexity", "type": typ, "maturity": T,
                             "strikes": f"{K[i - 1]:g}/{K[i]:g}/{K[i + 1]:g}", "violation": -bf})
        if forward:
            F = g["F"].to_numpy()
            lo = DF * np.maximum(sgn * (F - K), 0)
            hi = DF * (F if call else K)
            for k_, v_, l_, h_ in zip(K, V, lo, hi):
                viol = max(l_ - v_, v_ - h_)
                if viol > tolerance:
                    rows.append({"check": "price_bounds", "type": typ, "maturity": T, "strikes": f"{k_:g}",
                                 "violation": viol})
    out = pd.DataFrame(rows, columns=["check", "type", "maturity", "strikes", "violation"])
    for c in checks:
        checks[c] = int((out["check"] == c).sum())
    return Outcome({"options": len(d), "maturities": int(d["T"].nunique()),
                    **{f"{k}_violations": v for k, v in checks.items()},
                    "max_violation": float(out["violation"].max()) if len(out) else 0.0},
                   {"Violations": out}, notes=notes, rows_used=len(d))


@register("pricing.calendar_arbitrage", "Calendar no-arbitrage: total implied variance increasing in maturity",
          "No-arbitrage", _MT,
          params=(P("strike"), P("maturity"), P("vol"),
                  P("forward", required=False, help="Forward column; or spot with rate/dividend"),
                  P("spot", required=False), *_RATE, *_DIV,
                  P("tolerance", "number", default=0.0, help="Violations smaller than this are not counted")),
          description="""Absence of calendar-spread arbitrage requires total implied variance w(k, T) = σ²(k, T)·T to be
non-decreasing in T at fixed forward log-moneyness k = ln(K/F_T). For each pair of consecutive maturities the
smiles are linearly interpolated in k on the union of their strikes within the common k range, and every
point where w(T1) > w(T2) + tolerance is listed. No extrapolation is done.""",
          references=("Gatheral, J. and Jacquier, A. (2014), Arbitrage-free SVI volatility surfaces, "
                      "Quantitative Finance 14(1), 59–71",
                      "Gatheral, J. (2006), The Volatility Surface: A Practitioner's Guide, Wiley"))
def calendar_arbitrage(ctx: RunContext, strike, maturity, vol, forward=None, spot=None, rate=None, rate_value=0.0,
                       dividend=None, dividend_value=0.0, tolerance=0.0) -> Outcome:
    if (forward is None) == (spot is None):
        raise ValueError("Give exactly one of `forward` or `spot`.")
    notes = []
    d = _frame(ctx, {"K": strike, "T": maturity, "sig": vol, "F": forward, "S": spot, "r": rate, "q": dividend},
               {"r": rate_value, "q": dividend_value})
    ok = _valid(d, ["K", "T", "sig"], notes)
    d = d[ok].copy()
    if spot:
        d["F"] = d["S"] * np.exp((d["r"] - d["q"]) * d["T"])
    d["k"] = np.log(d["K"] / d["F"])
    d["w"] = d["sig"] ** 2 * d["T"]
    Ts = sorted(d["T"].unique())
    if len(Ts) < 2:
        raise NotApplicable("Need at least two maturities.")
    rows, pairs = [], 0
    for T1, T2 in zip(Ts[:-1], Ts[1:]):
        a = d[d["T"] == T1].sort_values("k")
        b = d[d["T"] == T2].sort_values("k")
        lo, hi = max(a["k"].min(), b["k"].min()), min(a["k"].max(), b["k"].max())
        grid = np.unique(np.concatenate([a["k"], b["k"]]))
        grid = grid[(grid >= lo) & (grid <= hi)]
        if not len(grid):
            continue
        pairs += 1
        w1, w2 = np.interp(grid, a["k"], a["w"]), np.interp(grid, b["k"], b["w"])
        for k_, x, y in zip(grid, w1, w2):
            if x - y > tolerance:
                rows.append({"maturity_1": T1, "maturity_2": T2, "log_moneyness": k_, "total_var_1": x,
                             "total_var_2": y, "violation": x - y})
    out = pd.DataFrame(rows, columns=["maturity_1", "maturity_2", "log_moneyness", "total_var_1", "total_var_2",
                                      "violation"])
    return Outcome({"quotes": len(d), "maturities": len(Ts), "pairs_compared": pairs, "violations": len(out),
                    "max_violation": float(out["violation"].max()) if len(out) else 0.0},
                   {"Calendar violations": out}, notes=notes, rows_used=len(d))


# ── Greeks and Monte Carlo ────────────────────────────────────────────────

@register("pricing.fd_greeks", "Finite-difference sensitivities vs reported Greeks", "Sensitivities", _MT,
          params=(P("price", help="Model price at the base point"),
                  P("price_up", help="Model price with the input bumped up by h"),
                  P("price_down", help="Model price with the input bumped down by h"),
                  P("bump", "number", help="Bump size h (absolute, or relative when `level` is given)"),
                  P("level", required=False, help="Input level column; h = bump × level (relative bump)"),
                  P("reported_first", required=False, help="Reported first-order Greek (e.g. delta)"),
                  P("reported_second", required=False, help="Reported second-order Greek (e.g. gamma)")),
          description="""From the bank's own revaluations at x ± h: central first derivative (V+ − V−)/(2h), second
derivative (V+ − 2V + V−)/h², and forward/backward one-sided first derivatives (their gap measures
curvature/noise relative to h). Compared with the reported Greeks (difference and relative difference).
Central differences have O(h²) truncation error; very small h amplifies pricing noise (e.g. Monte Carlo).""",
          references=("Glasserman, P. (2003), Monte Carlo Methods in Financial Engineering, ch. 7", _HULL))
def fd_greeks(ctx: RunContext, price, price_up, price_down, bump, level=None, reported_first=None,
              reported_second=None) -> Outcome:
    if bump <= 0:
        raise ValueError("bump must be > 0")
    notes = []
    d = _frame(ctx, {"V": price, "Vu": price_up, "Vd": price_down, "x": level, "g1": reported_first,
                     "g2": reported_second}, {})
    ok = _valid(d, ["x"] if level else [], notes)
    d = d[ok]
    h = bump * d["x"] if level else pd.Series(bump, index=d.index)
    fd1 = (d["Vu"] - d["Vd"]) / (2 * h)
    fd2 = (d["Vu"] - 2 * d["V"] + d["Vd"]) / h ** 2
    out = pd.DataFrame({"row": d.index, "bump": h, "fd_first": fd1, "fd_second": fd2,
                        "forward_first": (d["Vu"] - d["V"]) / h, "backward_first": (d["V"] - d["Vd"]) / h})
    summ = {"rows": len(out)}
    for lab, fd, col in (("first", fd1, "g1"), ("second", fd2, "g2")):
        if col in d:
            diff = d[col] - fd
            out[f"reported_{lab}"] = d[col]
            out[f"{lab}_diff"] = diff
            out[f"{lab}_rel_diff"] = diff / fd.where(fd != 0)
            summ[f"max_abs_{lab}_diff"] = float(diff.abs().max())
            summ[f"max_abs_{lab}_rel_diff"] = float(out[f"{lab}_rel_diff"].abs().max())
    return Outcome(summ, {"Finite-difference Greeks": out}, notes=notes, rows_used=len(out))


def _convergence(x: np.ndarray) -> pd.DataFrame:
    n = len(x)
    ks = sorted(set([2 ** k for k in range(4, int(math.log2(n)) + 1)] + [n]))
    cs, cs2 = np.cumsum(x), np.cumsum(x ** 2)
    rows = []
    for k in ks:
        m = cs[k - 1] / k
        var = max((cs2[k - 1] - k * m * m) / (k - 1), 0.0)
        rows.append({"paths": k, "estimate": m, "std_error": math.sqrt(var / k)})
    return pd.DataFrame(rows)


def _conv_summary(t: pd.DataFrame, benchmark) -> tuple[dict, list[str]]:
    est, se = float(t["estimate"].iloc[-1]), float(t["std_error"].iloc[-1])
    ok = t["std_error"] > 0
    slope = float(np.polyfit(np.log(t.loc[ok, "paths"]), np.log(t.loc[ok, "std_error"]), 1)[0]) \
        if ok.sum() >= 3 else float("nan")
    s = {"paths": int(t["paths"].iloc[-1]), "estimate": est, "std_error": se,
         "ci95_low": est - 1.959964 * se, "ci95_high": est + 1.959964 * se, "log_se_slope": slope}
    if benchmark is not None:
        z = (est - benchmark) / se if se > 0 else float("nan")
        s |= {"benchmark": benchmark, "difference": est - benchmark, "z_score": z,
              "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")}
    return s, []


@register("pricing.mc_convergence", "Monte Carlo convergence diagnostics from simulated payoffs", "Monte Carlo",
          _MT, params=(P("payoff", help="Discounted simulated payoff per path (rows in simulation order)"),
                       P("benchmark", "number", required=False, help="Reference price (e.g. closed form)")),
          description="""Running estimate and standard error s/√n at n = 16, 32, …, N paths, and the slope of
log(SE) on log(n), which is −0.5 for i.i.d. sampling (other values flag correlated paths or a heavy-tailed
payoff). Final estimate with 95% CI; with `benchmark`, z = (estimate − benchmark)/SE and its two-sided
p-value (H0: the MC estimator is unbiased for the benchmark).""",
          references=("Glasserman, P. (2003), Monte Carlo Methods in Financial Engineering, Springer, ch. 1",))
def mc_convergence(ctx: RunContext, payoff, benchmark=None) -> Outcome:
    s = num(ctx.df, payoff)
    notes = dropped_note(int(s.isna().sum()))
    x = s.dropna().to_numpy(float)
    if len(x) < 16:
        raise NotApplicable(f"{len(x)} paths; at least 16 needed.")
    t = _convergence(x)
    summ, _ = _conv_summary(t, benchmark)
    return Outcome(summ, {"Convergence": t}, notes=notes, rows_used=len(x))


@register("pricing.mc_gbm_check", "Monte Carlo GBM European option vs Black–Scholes", "Monte Carlo", _MT,
          params=(P("spot", "number"), P("strike", "number"), P("maturity", "number"), P("vol", "number"),
                  P("rate", "number", default=0.0), P("dividend", "number", default=0.0),
                  P("option_type", "string", default="call", choices=("call", "put")),
                  P("n_paths", "integer", default=100000), P("antithetic", "boolean", default=True),
                  P("model_price", "number", required=False, help="Bank's (MC) price to compare")),
          description="""Simulates S_T = S exp((r − q − σ²/2)T + σ√T Z) with a seeded generator (antithetic pairs
optional), prices the European payoff and reports the convergence table (estimate and SE by path count)
against the closed-form Black–Scholes price: z-score of the MC error and the log-SE slope (−0.5 expected).
A reference check of MC engine settings (path counts needed for a target SE). With antithetic pairs the SE
uses pair averages. With `model_price`, its distance from the closed form is reported in MC-SE units.""",
          references=("Glasserman, P. (2003), Monte Carlo Methods in Financial Engineering, Springer, ch. 3–4",
                      _HULL))
def mc_gbm_check(ctx: RunContext, spot, strike, maturity, vol, rate=0.0, dividend=0.0, option_type="call",
                 n_paths=100000, antithetic=True, model_price=None) -> Outcome:
    if min(spot, strike, maturity, vol) <= 0 or n_paths < 32:
        raise ValueError("spot, strike, maturity, vol must be > 0 and n_paths >= 32")
    rng = ctx.rng(23)
    call = option_type == "call"
    m = n_paths // 2 if antithetic else n_paths
    Z = rng.standard_normal(m)
    drift, sd = (rate - dividend - 0.5 * vol ** 2) * maturity, vol * math.sqrt(maturity)

    def pay(z):
        ST = spot * np.exp(drift + sd * z)
        return math.exp(-rate * maturity) * np.maximum((ST - strike) if call else (strike - ST), 0)

    x = 0.5 * (pay(Z) + pay(-Z)) if antithetic else pay(Z)
    bench = float(bs(spot, strike, maturity, rate, dividend, vol, call)["price"])
    t = _convergence(x)
    if antithetic:
        t["paths"] = t["paths"] * 2
    summ, _ = _conv_summary(t, bench)
    if model_price is not None:
        summ |= {"model_price": model_price, "model_minus_closed_form": model_price - bench}
    notes = ["Antithetic variates: path counts include both paths of each pair; SE from pair averages."] \
        if antithetic else []
    return Outcome(summ, {"Convergence": t}, notes=notes, rows_used=None)


@register("pricing.implied_vol", "Implied volatility solver (Brent) and round-trip check", "Benchmarking", _MT,
          params=(P("price"), P("strike"), P("maturity"),
                  P("underlying", help="Spot (black_scholes) or forward (black76/bachelier)"),
                  P("model", "string", default="black_scholes", choices=("black_scholes", "black76", "bachelier")),
                  *_TYPE, *_RATE, *_DIV,
                  P("reported_vol", required=False, help="Bank's implied vol column to compare")),
          description="""Solves price = model(σ) for σ by Brent's method on [1e−6, 10] (lognormal) or
[1e−10, 10·max(|F|, |K|, 1)] (normal), after checking the price lies strictly inside the no-arbitrage bounds
(intrinsic value < price < upper bound). Round-trip: re-prices at the solved σ and reports the price error.
Rows outside the bounds get status 'outside bounds' and no vol. Optionally compares to reported vols.""",
          references=("Brent, R. P. (1973), Algorithms for Minimization without Derivatives, Prentice-Hall",
                      _HULL))
def implied_vol(ctx: RunContext, price, strike, maturity, underlying, model="black_scholes", option_type=None,
                default_type="call", rate=None, rate_value=0.0, dividend=None, dividend_value=0.0,
                reported_vol=None) -> Outcome:
    notes = []
    d = _frame(ctx, {"V": price, "K": strike, "T": maturity, "U": underlying, "r": rate, "q": dividend,
                     "rv": reported_vol}, {"r": rate_value, "q": dividend_value})
    call = _is_call(ctx, option_type, default_type, len(d))
    pos = ["T"] + (["U", "K"] if model != "bachelier" else [])
    ok = _valid(d[["V", "K", "T", "U", "r", "q"]], pos, notes)
    d, call = d[ok], call[ok.to_numpy()]

    def f(sig, row, c):
        if model == "black_scholes":
            return float(bs(row.U, row.K, row.T, row.r, row.q, sig, c)["price"])
        if model == "black76":
            return float(black76(row.U, row.K, row.T, row.r, sig, c)["price"])
        return float(bachelier(row.U, row.K, row.T, row.r, sig, c)["price"])

    rows = []
    for row, c in zip(d.itertuples(), call):
        DF = math.exp(-row.r * row.T)
        fwdv = row.U * math.exp(-row.q * row.T) if model == "black_scholes" else row.U * DF
        kv = row.K * DF
        intrinsic = max(fwdv - kv, 0) if c else max(kv - fwdv, 0)
        upper = (fwdv if c else kv) if model != "bachelier" else float("inf")
        lo, hi = (1e-6, 10.0) if model != "bachelier" else (1e-10, 10 * max(abs(row.U), abs(row.K), 1.0))
        status, iv, err = "ok", float("nan"), float("nan")
        if not (intrinsic < row.V < upper):
            status = "outside bounds"
        else:
            g = lambda s: f(s, row, c) - row.V          # noqa: E731
            if g(lo) > 0 or g(hi) < 0:
                status = "not bracketed"
            else:
                iv = optimize.brentq(g, lo, hi, xtol=1e-14, rtol=1e-12, maxiter=500)
                err = f(iv, row, c) - row.V
        rows.append({"row": row.Index, "type": "call" if c else "put", "price": row.V, "implied_vol": iv,
                     "repriced_error": err, "status": status,
                     **({"reported_vol": row.rv, "vol_diff": row.rv - iv} if reported_vol else {})})
    out = pd.DataFrame(rows)
    solved = out["status"] == "ok"
    summ = {"options": len(out), "solved": int(solved.sum()), "outside_bounds": int((~solved).sum()),
            "max_abs_roundtrip_error": float(out.loc[solved, "repriced_error"].abs().max()) if solved.any()
            else float("nan")}
    if reported_vol and solved.any():
        summ["max_abs_vol_diff"] = float(out.loc[solved, "vol_diff"].abs().max())
    return Outcome(summ, {"Implied volatilities": out}, notes=notes, rows_used=len(out))


# ── curves and bonds ──────────────────────────────────────────────────────

_COMP = ("continuous", "annual", "semi_annual", "quarterly", "monthly", "simple")
_FREQ = {"annual": 1, "semi_annual": 2, "quarterly": 4, "monthly": 12}


def _df_from_zero(z, t, comp):
    if comp == "continuous":
        return np.exp(-z * t)
    if comp == "simple":
        return 1 / (1 + z * t)
    f = _FREQ[comp]
    return (1 + z / f) ** (-f * t)


@register("pricing.curve_diagnostics", "Yield-curve diagnostics: discount factors, forwards, consistency",
          "Curves", _MT,
          params=(P("maturity", help="Node time in years"),
                  P("discount_factor", required=False), P("zero_rate", required=False),
                  P("compounding", "string", default="continuous", choices=_COMP,
                    help="Compounding of zero_rate")),
          description="""On the curve nodes: discount factors DF(t) (from the DF column, else from zero rates), continuously
compounded zero rates, and forward rates between nodes f_i = −ln(DF_i/DF_{i−1})/(t_i − t_{i−1}) (with DF(0) = 1).
Reports: DF > 1 (negative zero rates — legitimate for EUR), DF increases between nodes (equivalently negative
forwards), the forward-curve jumps Δf and roughness Σ(Δf)², and — when both DF and zero columns are given —
the maximum difference between the DF column and the DF implied by the zero rates (zero↔DF consistency).""",
          references=("Hagan, P. S. and West, G. (2006), Interpolation methods for curve construction, "
                      "Applied Mathematical Finance 13(2), 89–129",))
def curve_diagnostics(ctx: RunContext, maturity, discount_factor=None, zero_rate=None,
                      compounding="continuous") -> Outcome:
    if not discount_factor and not zero_rate:
        raise ValueError("Give discount_factor and/or zero_rate.")
    notes = []
    d = _frame(ctx, {"t": maturity, "df": discount_factor, "z": zero_rate}, {})
    ok = _valid(d, ["t"] + (["df"] if discount_factor else []), notes)
    d = d[ok].sort_values("t")
    if d["t"].duplicated().any():
        raise ValueError("Duplicate maturities in the curve.")
    if len(d) < 2:
        raise NotApplicable("Need at least two curve nodes.")
    t = d["t"].to_numpy()
    dfz = _df_from_zero(d["z"].to_numpy(), t, compounding) if zero_rate else None
    DF = d["df"].to_numpy() if discount_factor else dfz
    zc = -np.log(DF) / t
    tp, dp = np.concatenate([[0.0], t[:-1]]), np.concatenate([[1.0], DF[:-1]])
    fwd = -np.log(DF / dp) / (t - tp)
    dfwd = np.diff(fwd, prepend=np.nan)
    out = pd.DataFrame({"maturity": t, "discount_factor": DF, "zero_rate_cont": zc, "forward_rate": fwd,
                        "forward_change": dfwd})
    summ = {"nodes": len(d), "df_above_one": int((DF > 1).sum()), "df_increases": int((np.diff(DF) > 0).sum()),
            "negative_forwards": int((fwd < 0).sum()), "min_forward": float(fwd.min()),
            "max_abs_forward_jump": float(np.nanmax(np.abs(dfwd))), "forward_roughness": float(np.nansum(dfwd ** 2))}
    if discount_factor and zero_rate:
        out["df_from_zero"] = dfz
        out["df_difference"] = DF - dfz
        summ["max_abs_df_difference"] = float(np.abs(DF - dfz).max())
    if (DF > 1).any():
        notes.append("Discount factors above 1 imply negative zero rates; this is legitimate in negative-rate regimes.")
    return Outcome(summ, {"Curve nodes": out}, notes=notes, rows_used=len(d))


def _schedule(T: float, freq: int) -> np.ndarray:
    k = np.arange(0, int(math.floor(T * freq + 1e-9)) + 1)
    t = T - k / freq
    return np.sort(t[t > 1e-9])


def _bond_cfs(coupon: float, T: float, freq: int, face: float):
    t = _schedule(T, freq)
    cf = np.full(len(t), face * coupon / freq)
    cf[-1] += face
    accrued = face * coupon / freq * (1 - t[0] * freq) if len(t) else 0.0
    return t, cf, max(accrued, 0.0)


def _curve_df(ctab: pd.DataFrame, tcol: str, vcol: str, vtype: str):
    c = ctab[[tcol, vcol]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(tcol)
    c = c[c[tcol] > 0]
    if len(c) < 1:
        raise ValueError("Curve table has no usable nodes.")
    ct = c[tcol].to_numpy()
    z = (-np.log(c[vcol].to_numpy()) / ct) if vtype == "discount_factor" else c[vcol].to_numpy()

    def df(t):
        # log-linear DF between nodes (piecewise-flat forwards); flat zero rate outside the node range
        t = np.asarray(t, float)
        lnd = np.interp(t, ct, -z * ct)
        lnd = np.where(t < ct[0], -z[0] * t, lnd)
        lnd = np.where(t > ct[-1], -z[-1] * t, lnd)
        return np.exp(lnd)
    return df


@register("pricing.curve_repricing", "Repricing of input bonds from a discount curve", "Curves", _MT,
          params=(P("coupon", help="Annual coupon rate (decimal)"), P("maturity", help="Time to maturity (years)"),
                  P("market_price", help="Market price per `face`"),
                  P("curve", "table", help="Curve table with node times and DFs or zero rates"),
                  P("curve_time", "string", default="maturity", help="Time column in the curve table"),
                  P("curve_value", "string", default="discount_factor", help="Value column in the curve table"),
                  P("curve_value_type", "string", default="discount_factor",
                    choices=("discount_factor", "zero_rate"), help="zero_rate = continuously compounded"),
                  P("frequency", "integer", default=1, help="Coupons per year"),
                  P("face", "number", default=100.0),
                  P("price_type", "string", default="dirty", choices=("dirty", "clean"))),
          description="""Prices each fixed-coupon bond from the curve, PV = Σ CF_i DF(t_i), with log-linear interpolation
of discount factors (piecewise-flat forwards) and flat zero-rate extrapolation, and compares with the market
price: difference, relative difference and the implied z-spread-like error in basis points
(difference / (duration × price)). Coupon dates run backwards from maturity at the given frequency; accrued
interest is linear (clean = dirty − accrued). A curve fitted to these instruments should reprice them
to within its fitting tolerance.""",
          references=("Hagan, P. S. and West, G. (2006), Interpolation methods for curve construction, "
                      "Applied Mathematical Finance 13(2), 89–129",
                      "Fabozzi, F. J. (2007), Fixed Income Analysis, 2nd ed., CFA Institute"))
def curve_repricing(ctx: RunContext, coupon, maturity, market_price, curve, curve_time="maturity",
                    curve_value="discount_factor", curve_value_type="discount_factor", frequency=1, face=100.0,
                    price_type="dirty") -> Outcome:
    ctab = ctx.tables[curve]
    for c in (curve_time, curve_value):
        if c not in ctab.columns:
            raise ValueError(f"Curve table has no column '{c}'. Columns: {list(ctab.columns)}")
    dfn = _curve_df(ctab, curve_time, curve_value, curve_value_type)
    notes = []
    d = _frame(ctx, {"c": coupon, "T": maturity, "P": market_price}, {})
    ok = _valid(d, ["T"], notes)
    d = d[ok]
    rows = []
    for r in d.itertuples():
        t, cf, acc = _bond_cfs(r.c, r.T, frequency, face)
        dfs = dfn(t)
        dirty = float(cf @ dfs)
        model = dirty - acc if price_type == "clean" else dirty
        dur = float((t * cf * dfs).sum() / dirty)
        diff = r.P - model
        rows.append({"row": r.Index, "coupon": r.c, "maturity": r.T, "market_price": r.P, "curve_price": model,
                     "difference": diff, "relative_difference": diff / model,
                     "error_bp": -diff / (dur * dirty) * 1e4 if dur > 0 else float("nan")})
    out = pd.DataFrame(rows)
    return Outcome({"bonds": len(out), "max_abs_difference": float(out["difference"].abs().max()),
                    "rmse": float(np.sqrt((out["difference"] ** 2).mean())),
                    "max_abs_error_bp": float(out["error_bp"].abs().max())},
                   {"Repricing": out}, notes=notes, rows_used=len(out))


def _bond_from_yield(t, cf, y, freq, comp):
    disc = np.exp(-y * t) if comp == "continuous" else (1 + y / freq) ** (-freq * t)
    P = float(cf @ disc)
    mac = float((t * cf * disc).sum() / P)
    if comp == "continuous":
        mod, cx = mac, float((t ** 2 * cf * disc).sum() / P)
    else:
        mod = mac / (1 + y / freq)
        cx = float((cf * t * (t + 1 / freq) * disc).sum() / P / (1 + y / freq) ** 2)
    return P, mac, mod, cx


@register("pricing.bond_analytics", "Fixed-coupon bond price, yield, duration and convexity", "Curves", _MT,
          params=(P("coupon"), P("maturity"),
                  P("ytm", required=False, help="Yield to maturity (decimal); or give `price`"),
                  P("price", required=False, help="Dirty price per `face` (yield is solved)"),
                  P("frequency", "integer", default=2), P("face", "number", default=100.0),
                  P("compounding", "string", default="periodic", choices=("periodic", "continuous")),
                  P("model_price", required=False, help="Bank's dirty price to compare"),
                  P("model_duration", required=False, help="Bank's modified duration to compare")),
          description="""Dirty price P = Σ CF_i (1 + y/f)^(−f t_i) (or e^(−y t_i) continuous), clean price, accrued,
Macaulay duration Σ t_i CF_i DF_i / P, modified duration (Macaulay/(1 + y/f); equal for continuous),
convexity and DV01 = modified duration × P × 1bp. With `price` the yield is solved by Brent. Coupon dates run
backwards from maturity; fractional first period uses fractional exponents (street convention).""",
          references=("Fabozzi, F. J. (2007), Fixed Income Analysis, 2nd ed., CFA Institute", _HULL))
def bond_analytics(ctx: RunContext, coupon, maturity, ytm=None, price=None, frequency=2, face=100.0,
                   compounding="periodic", model_price=None, model_duration=None) -> Outcome:
    if (ytm is None) == (price is None):
        raise ValueError("Give exactly one of `ytm` or `price`.")
    notes = []
    d = _frame(ctx, {"c": coupon, "T": maturity, "y": ytm, "P": price, "mp": model_price, "md": model_duration}, {})
    ok = _valid(d[[c for c in ("c", "T", "y", "P") if c in d]], ["T"], notes)
    d = d[ok]
    rows = []
    for r in d.itertuples():
        t, cf, acc = _bond_cfs(r.c, r.T, frequency, face)
        if price is not None:
            g = lambda y: _bond_from_yield(t, cf, y, frequency, compounding)[0] - r.P   # noqa: E731
            y = optimize.brentq(g, -0.99 * frequency if compounding == "periodic" else -1.0, 5.0, xtol=1e-14)
        else:
            y = r.y
        P_, mac, mod, cx = _bond_from_yield(t, cf, y, frequency, compounding)
        row = {"row": r.Index, "coupon": r.c, "maturity": r.T, "ytm": y, "dirty_price": P_, "accrued": acc,
               "clean_price": P_ - acc, "macaulay_duration": mac, "modified_duration": mod, "convexity": cx,
               "DV01": mod * P_ * 1e-4}
        if model_price:
            row["price_diff"] = r.mp - P_
        if model_duration:
            row["duration_diff"] = r.md - mod
        rows.append(row)
    out = pd.DataFrame(rows)
    summ = {"bonds": len(out)}
    if len(out) == 1:
        summ |= {k: float(out[k].iloc[0]) for k in ("dirty_price", "ytm", "modified_duration", "convexity")}
    for k in ("price_diff", "duration_diff"):
        if k in out:
            summ[f"max_abs_{k}"] = float(out[k].abs().max())
    return Outcome(summ, {"Bond analytics": out}, notes=notes, rows_used=len(out))


# ── P&L explain and stress ────────────────────────────────────────────────

@register("pricing.pnl_explain", "P&L explain: risk-based vs full-revaluation P&L", "P&L explain", _MT,
          params=(P("actual", help="Full-revaluation P&L"), P("predicted", help="Sensitivity (risk-based) P&L")),
          description="""Unexplained P&L U = full-revaluation P&L − sensitivity-based P&L. Reports
mean ratio mean(U)/sd(full), variance ratio var(U)/var(full) (the 2016 FRTB PLA metrics), the ratio
Σ|U| / Σ|full|, correlation and R² of full on risk-based P&L, and the days with the largest |U|.
Describes how well the Greeks explain revaluation; no thresholds applied.""",
          references=("Basel Committee on Banking Supervision (2016), Minimum capital requirements for market risk "
                      "(P&L attribution, superseded by MAR32 in 2019)",))
def pnl_explain(ctx: RunContext, actual, predicted) -> Outcome:
    a, p = num(ctx.df, actual), num(ctx.df, predicted)
    ok = a.notna() & p.notna()
    notes = dropped_note(int((~ok).sum()))
    a, p = a[ok].to_numpy(), p[ok].to_numpy()
    if len(a) < 3 or a.std() == 0:
        raise NotApplicable("Need at least 3 observations with non-constant full-revaluation P&L.")
    U = a - p
    r = float(np.corrcoef(a, p)[0, 1]) if p.std() > 0 else float("nan")
    idx = np.argsort(-np.abs(U))[:10]
    worst = pd.DataFrame({"row": ctx.df.index[ok.to_numpy()][idx], "full_pnl": a[idx], "risk_pnl": p[idx],
                          "unexplained": U[idx]})
    return Outcome({"n": len(a), "mean_unexplained": float(U.mean()),
                    "mean_ratio": float(U.mean() / a.std(ddof=1)),
                    "variance_ratio": float(U.var(ddof=1) / a.var(ddof=1)),
                    "abs_unexplained_ratio": float(np.abs(U).sum() / np.abs(a).sum()),
                    "correlation": r, "r_squared": r ** 2 if np.isfinite(r) else float("nan")},
                   {"Largest unexplained P&L": worst}, notes=notes, rows_used=len(a))


@register("pricing.stress_repricing", "Stress / scenario repricing of a Black–Scholes option book", "Stress testing",
          _MT, params=(P("spot"), P("strike"), P("maturity"), P("vol"), *_TYPE, *_RATE, *_DIV,
                       P("quantity", required=False, help="Position size column (default 1)"),
                       P("spot_shocks", "list", default=[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
                         help="Relative spot shocks"),
                       P("vol_shocks", "list", default=[-0.05, 0.0, 0.05], help="Absolute vol shocks")),
          description="""Full revaluation of the book under a grid of spot (relative) × volatility (absolute) shocks:
P&L = Σ q·[BS(S(1+s), σ+v) − BS(S, σ)]. Also the delta–gamma–vega approximation
Σ q·[Δ dS + ½ Γ dS² + vega dσ] and the approximation error, showing where the Greeks stop explaining P&L.
Shocked vols are floored at 1e−4.""",
          references=(_HULL, "Basel Committee on Banking Supervision (2009), Principles for sound stress testing "
                             "practices and supervision"))
def stress_repricing(ctx: RunContext, spot, strike, maturity, vol, option_type=None, default_type="call",
                     rate=None, rate_value=0.0, dividend=None, dividend_value=0.0, quantity=None,
                     spot_shocks=(-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2), vol_shocks=(-0.05, 0.0, 0.05)) -> Outcome:
    notes = []
    d = _frame(ctx, {"S": spot, "K": strike, "T": maturity, "sig": vol, "r": rate, "q": dividend, "n": quantity},
               {"r": rate_value, "q": dividend_value, "n": 1.0})
    call = _is_call(ctx, option_type, default_type, len(d))
    ok = _valid(d, ["S", "K", "T", "sig"], notes)
    d, call = d[ok], call[ok.to_numpy()]
    base = bs(d["S"], d["K"], d["T"], d["r"], d["q"], d["sig"], call)
    q = d["n"].to_numpy()
    rows = []
    for s in sorted(float(x) for x in spot_shocks):
        for v in sorted(float(x) for x in vol_shocks):
            S1 = d["S"] * (1 + s)
            sg = np.maximum(d["sig"] + v, 1e-4)
            full = float(q @ (bs(S1, d["K"], d["T"], d["r"], d["q"], sg, call)["price"] - base["price"]))
            dS = (S1 - d["S"]).to_numpy()
            approx = float(q @ (base["delta"] * dS + 0.5 * base["gamma"] * dS ** 2 + base["vega"] * (sg - d["sig"])))
            rows.append({"spot_shock": s, "vol_shock": v, "full_reval_pnl": full, "greeks_pnl": approx,
                         "approximation_error": full - approx})
    out = pd.DataFrame(rows)
    i = int(out["full_reval_pnl"].to_numpy().argmin())
    return Outcome({"positions": len(d), "book_value": float(q @ base["price"]),
                    "worst_pnl": float(out["full_reval_pnl"].iloc[i]),
                    "worst_spot_shock": float(out["spot_shock"].iloc[i]),
                    "worst_vol_shock": float(out["vol_shock"].iloc[i]),
                    "max_abs_approximation_error": float(out["approximation_error"].abs().max())},
                   {"Scenario P&L": out}, notes=notes, rows_used=len(d))
