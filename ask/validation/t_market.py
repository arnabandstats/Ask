"""Market risk VaR / ES: back-testing, ES tests, PIT tests, P&L attribution, independent replication.

Conventions used by every test in this module:
  * Losses are POSITIVE numbers internally. `pnl` is read as P&L (losses negative) unless
    `loss_positive=True`, in which case the column already holds losses.
  * VaR and ES are loss amounts (positive). `var_sign` says how the columns are given:
    "positive_loss", "negative_return" (a quantile of the P&L, so negative), or "auto"
    (all values <= 0 -> negative_return, all >= 0 -> positive_loss, mixed signs -> error).
  * An exception (hit) is a day with loss > VaR (strictly greater).
  * Rows are put in time order by `date` when given, otherwise the table order is used.
  * `confidence` is the VaR confidence level c (0.99); the exception probability is p = 1 - c.

Helpers `pit_uniformity_table` and `berkowitz_lr` are also used by the CCR module.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, dropped_note, num, register

_MT = ("var",)
_SIGNS = ("auto", "positive_loss", "negative_return")
_QMETHODS = ("inverted_cdf", "linear", "lower", "higher", "nearest", "midpoint", "hazen", "weibull",
             "median_unbiased")

_CONF = P("confidence", "number", default=0.99, help="VaR confidence level c, e.g. 0.99 (p = 1 - c)")
_LOSSPOS = P("loss_positive", "boolean", default=False,
             help="True when the pnl column already holds losses as positive numbers")
_VSIGN = P("var_sign", "string", default="auto", choices=_SIGNS,
           help="How VaR/ES are given: positive loss amounts, negative returns, or auto-detect")
_BT = (P("pnl"), P("var"), P("date", required=False), _CONF, _LOSSPOS, _VSIGN)

_BASEL_REFS = (
    "Basel Committee on Banking Supervision (1996), Supervisory framework for the use of 'backtesting' "
    "in conjunction with the internal models approach to market risk capital requirements",
    "Regulation (EU) No 575/2013 (CRR), Article 366 (regulatory back-testing and plus-factor/addend)",
)


# ── shared input handling ──────────────────────────────────────────────────

def _check_conf(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return 1.0 - confidence


def _sign_fix(x: np.ndarray, how: str, label: str) -> tuple[np.ndarray, list[str]]:
    if how == "negative_return":
        return -x, [f"{label} given as negative returns; sign flipped to loss amounts."]
    if how == "positive_loss":
        neg = int((x < 0).sum())
        return x, ([f"{neg} {label} values are negative (a forecast gain) and were kept as given."]
                   if neg else [])
    pos, neg = int((x > 0).sum()), int((x < 0).sum())
    if pos and neg:
        raise ValueError(f"{label} has {pos} positive and {neg} negative values; cannot auto-detect its "
                         f"sign. Set var_sign to 'positive_loss' or 'negative_return'.")
    if neg:
        return -x, [f"{label} values are all <= 0: read as negative returns (a P&L quantile) and "
                    f"sign-flipped to loss amounts (auto-detected)."]
    return x, []


def _order(ctx: RunContext, date) -> tuple[pd.Series | None, list[str]]:
    if not date:
        return None, ["No date given: the table's row order is taken as time order."]
    d = ctx.df[date]
    dt = pd.to_datetime(d, errors="coerce") if not pd.api.types.is_numeric_dtype(d) else d
    if pd.isna(dt).sum() > pd.isna(d).sum():
        dt = d.astype(str)          # not parseable as dates: order by the raw values
    return dt, []


def load_series(ctx: RunContext, cols: dict[str, str | None], date=None, loss_positive=False,
                var_sign="auto", pnl_key="pnl", sign_keys=("var", "es")
                ) -> tuple[pd.DataFrame, list[str], int]:
    """Numeric columns (by role key) in time order, NaN rows dropped, losses positive,
    VaR/ES as positive loss amounts. Returns (frame with 'loss' and role keys, notes, dropped)."""
    df = ctx.df
    d = pd.DataFrame({k: num(df, c) for k, c in cols.items() if c}, index=df.index)
    when, notes = _order(ctx, date)
    if when is not None:
        d["date"] = when
    n0 = len(d)
    d = d.dropna()
    dropped = n0 - len(d)
    if when is not None:
        if d["date"].duplicated().any():
            notes.append(f"{int(d['date'].duplicated().sum())} duplicate dates; ties kept in table order.")
        d = d.sort_values("date", kind="mergesort")
    d = d.reset_index(drop=True)
    if pnl_key in d:
        d["loss"] = d[pnl_key] if loss_positive else -d[pnl_key]
    for k in sign_keys:
        if k in d:
            v, nt = _sign_fix(d[k].to_numpy(float), var_sign, k.upper())
            d[k] = v
            notes += nt
    return d, notes, dropped


def _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, es=None, min_n=2):
    p = _check_conf(confidence)
    d, notes, dropped = load_series(ctx, {"pnl": pnl, "var": var, "es": es}, date, loss_positive, var_sign)
    if len(d) < min_n:
        raise NotApplicable(f"Only {len(d)} complete observations; at least {min_n} needed.")
    d["hit"] = (d["loss"] > d["var"]).astype(int)
    return d, p, notes + dropped_note(dropped), dropped


def _label(d: pd.DataFrame) -> pd.Series:
    if "date" in d:
        return d["date"].astype(str)
    return pd.Series(np.arange(1, len(d) + 1), index=d.index).astype(str)


# ── core statistics (pure functions, also used by tests) ───────────────────

def kupiec_pof(x: int, n: int, p: float) -> float:
    """Kupiec (1995) proportion-of-failures LR statistic."""
    ph = x / n
    ll0 = special.xlogy(n - x, 1 - p) + special.xlogy(x, p)
    ll1 = special.xlogy(n - x, 1 - ph) + special.xlogy(x, ph)
    return float(-2 * (ll0 - ll1))


def _lr_tuff(v: int, p: float) -> float:
    """-2 ln[p(1-p)^(v-1)] + 2 ln[(1/v)(1-1/v)^(v-1)]."""
    return float(-2 * (math.log(p) + (v - 1) * math.log1p(-p))
                 + 2 * (-math.log(v) + special.xlogy(v - 1, 1 - 1 / v)))


def christoffersen(hit: np.ndarray) -> dict:
    """Transition counts and the Markov independence LR (Christoffersen 1998)."""
    a, b = hit[:-1], hit[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll0 = special.xlogy(n00 + n10, 1 - pi) + special.xlogy(n01 + n11, pi)
    ll1 = (special.xlogy(n00, 1 - pi01) + special.xlogy(n01, pi01)
           + special.xlogy(n10, 1 - pi11) + special.xlogy(n11, pi11))
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "pi01": pi01, "pi11": pi11, "pi": pi,
            "LR_ind": float(-2 * (ll0 - ll1))}


def basel_zone(x: int, n: int, p: float) -> tuple[str, float]:
    """Zone from the cumulative binomial probability P(X <= x): green < 95%, yellow < 99.99%, red."""
    cp = float(stats.binom.cdf(x, n, p))
    return ("green" if cp < 0.95 else "yellow" if cp < 0.9999 else "red"), cp


# Basel (1996) / CRR Art. 366 plus factor (addend) for 250 observations at 99%.
_PLUS = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def berkowitz_lr(z: np.ndarray) -> dict:
    """Berkowitz (2001) LR tests on z = Phi^-1(PIT) with the exact Gaussian AR(1) likelihood."""
    z = np.asarray(z, float)
    T = len(z)

    def nll(th):
        mu, ls, ar = th
        s2, rho = math.exp(2 * ls), math.tanh(ar)
        v1 = s2 / (1 - rho ** 2)
        e = z[1:] - mu - rho * (z[:-1] - mu)
        return -(-0.5 * (math.log(2 * math.pi * v1) + (z[0] - mu) ** 2 / v1)
                 - 0.5 * (T - 1) * math.log(2 * math.pi * s2) - (e @ e) / (2 * s2))

    # start from the conditional least-squares estimates
    X = np.column_stack([np.ones(T - 1), z[:-1]])
    beta = np.linalg.lstsq(X, z[1:], rcond=None)[0]
    rho0 = float(np.clip(beta[1], -0.95, 0.95))
    mu0 = float(beta[0] / (1 - rho0))
    res = z[1:] - X @ beta
    th0 = np.array([mu0, 0.5 * math.log(max(res @ res / (T - 1), 1e-12)), math.atanh(rho0)])
    fit = optimize.minimize(nll, th0, method="BFGS", options={"gtol": 1e-8, "maxiter": 2000})
    th = fit.x if fit.fun <= nll(th0) else th0
    ll1 = -float(nll(th))
    ll0 = float(stats.norm.logpdf(z).sum())
    mu_r, s2_r = float(z.mean()), float(z.var())
    ll_r = float(stats.norm.logpdf(z, mu_r, math.sqrt(s2_r)).sum()) if s2_r > 0 else -np.inf
    lr3, lri = max(2 * (ll1 - ll0), 0.0), max(2 * (ll1 - ll_r), 0.0)
    return {"mu": float(th[0]), "sigma": math.exp(th[1]), "rho": math.tanh(th[2]),
            "LR_3": lr3, "p_value_LR_3": float(stats.chi2.sf(lr3, 3)),
            "LR_ind": lri, "p_value_LR_ind": float(stats.chi2.sf(lri, 1)),
            "converged": bool(fit.success or np.max(np.abs(fit.jac)) < 1e-3)}


def _ad_uniform(u_sorted: np.ndarray) -> np.ndarray:
    """Anderson–Darling A^2 against U(0,1); works row-wise on a 2-D array of sorted samples."""
    u = np.clip(u_sorted, 1e-12, 1 - 1e-12)
    n = u.shape[-1]
    i = np.arange(1, n + 1)
    return -n - ((2 * i - 1) * (np.log(u) + np.log1p(-u[..., ::-1]))).sum(axis=-1) / n


def pit_uniformity_table(u: np.ndarray, rng: np.random.Generator, n_sims: int = 2000,
                         bins: int = 10) -> pd.DataFrame:
    """KS, Anderson–Darling (Monte Carlo p-value), Cramér–von Mises and chi-square tests of U(0,1)."""
    u = np.asarray(u, float)
    n = len(u)
    rows = []
    ks = stats.kstest(u, "uniform")
    rows.append({"test": "Kolmogorov–Smirnov", "statistic": float(ks.statistic), "p_value": float(ks.pvalue),
                 "p_value_method": "exact/asymptotic (scipy)"})
    a2 = float(_ad_uniform(np.sort(u)))
    sims = []
    for start in range(0, n_sims, 500):
        m = min(500, n_sims - start)
        sims.append(_ad_uniform(np.sort(rng.random((m, n)), axis=1)))
    sims = np.concatenate(sims)
    rows.append({"test": "Anderson–Darling", "statistic": a2,
                 "p_value": float((1 + (sims >= a2).sum()) / (n_sims + 1)),
                 "p_value_method": f"Monte Carlo, {n_sims} uniform samples"})
    cvm = stats.cramervonmises(u, "uniform")
    rows.append({"test": "Cramér–von Mises", "statistic": float(cvm.statistic), "p_value": float(cvm.pvalue),
                 "p_value_method": "asymptotic (scipy)"})
    obs = np.histogram(u, bins=np.linspace(0, 1, bins + 1))[0]
    chi = stats.chisquare(obs)
    rows.append({"test": f"Chi-square ({bins} equal bins)", "statistic": float(chi.statistic),
                 "p_value": float(chi.pvalue), "p_value_method": f"chi-square({bins - 1})"})
    return pd.DataFrame(rows)


def empirical_es(losses: np.ndarray, confidence: float) -> float:
    """ES of the empirical distribution (Acerbi–Tasche 2002): mean of the worst n·alpha losses,
    the boundary loss weighted by the fractional part."""
    L = np.sort(np.asarray(losses, float))[::-1]
    na = len(L) * (1 - confidence)
    k = int(math.floor(na + 1e-12))
    tot = L[:k].sum() + ((na - k) * L[k] if k < len(L) else 0.0)
    return float(tot / na)


def _newey_west_var(x: np.ndarray) -> float:
    x = x - x.mean()
    n = len(x)
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    v = x @ x / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / n
    return float(v)


# ── exception counting ────────────────────────────────────────────────────

@register("var.exceptions", "VaR exception series and count", "Back-testing", _MT, params=_BT,
          description="""Exception (hit) indicator I_t = 1{loss_t > VaR_t}. Reports the number of exceptions x,
the observed rate x/n against the expected rate p = 1 − c and the expected count n·p, the list of exception
days with the excess loss (loss − VaR) and its ratio to VaR, and a chart of losses against VaR.
Descriptive: the formal tests are var.kupiec_pof, var.christoffersen, var.traffic_light and others.""",
          references=_BASEL_REFS)
def exceptions(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
               var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, min_n=1)
    n, x = len(d), int(d["hit"].sum())
    lab = _label(d)
    ex = d[d["hit"] == 1]
    table = pd.DataFrame({"observation": lab[ex.index].to_numpy(), "position": ex.index + 1,
                          "loss": ex["loss"].to_numpy(), "VaR": ex["var"].to_numpy(),
                          "excess_loss": (ex["loss"] - ex["var"]).to_numpy(),
                          "loss_to_VaR": (ex["loss"] / ex["var"]).to_numpy()})
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=lab, y=d["loss"], mode="lines", name="Loss (positive = loss)")
    fig.add_scatter(x=lab, y=d["var"], mode="lines", name=f"VaR {confidence:.1%}")
    fig.add_scatter(x=lab[ex.index], y=ex["loss"], mode="markers", name="Exception",
                    marker=dict(size=9, symbol="x"))
    fig.update_layout(title="Losses against VaR", xaxis_title="observation", yaxis_title="loss")
    return Outcome({"n": n, "exceptions": x, "exception_rate": x / n, "expected_rate": p,
                    "expected_exceptions": n * p, "max_excess_loss": float(table["excess_loss"].max())
                    if x else 0.0},
                   {"Exceptions": table}, [fig], notes, rows_used=n)


@register("var.rolling_exceptions", "Rolling exception count (e.g. 250-day window)", "Back-testing", _MT,
          params=(*_BT, P("window", "integer", default=250, help="Window length in observations")),
          description="""Number of exceptions in every rolling window of `window` consecutive observations
(the regulatory back-test counts exceptions over the most recent 250 business days). For each window the
cumulative binomial probability P(X <= x | window, p) and the corresponding zone under the Basel cut-offs
(green < 95%, yellow < 99.99%, red >= 99.99%; for window 250 at 99% this is 0–4 / 5–9 / 10+) are shown.""",
          references=_BASEL_REFS)
def rolling_exceptions(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
                       var_sign="auto", window=250) -> Outcome:
    if window < 1:
        raise ValueError("window must be >= 1")
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, min_n=window)
    cnt = d["hit"].rolling(window).sum().iloc[window - 1:].astype(int)
    cp = stats.binom.cdf(cnt.to_numpy(), window, p)
    zone = np.where(cp < 0.95, "green", np.where(cp < 0.9999, "yellow", "red"))
    lab = _label(d)
    t = pd.DataFrame({"window_start": lab.iloc[:len(cnt)].to_numpy(), "window_end": lab[cnt.index].to_numpy(),
                      "exceptions": cnt.to_numpy(), "cumulative_probability": cp, "zone": zone})
    i = int(t["exceptions"].to_numpy().argmax())
    return Outcome({"windows": len(t), "window": window, "latest_exceptions": int(t["exceptions"].iloc[-1]),
                    "latest_zone": t["zone"].iloc[-1], "max_exceptions": int(t["exceptions"].iloc[i]),
                    "max_window_end": t["window_end"].iloc[i], "expected_per_window": window * p},
                   {"Rolling exception count": t}, notes=notes, rows_used=len(d))


# ── unconditional coverage ────────────────────────────────────────────────

@register("var.kupiec_pof", "Kupiec proportion-of-failures (POF) test", "Back-testing", _MT, params=_BT,
          description="""Unconditional coverage. H0: P(exception) = p = 1 − c.
LR_POF = −2 ln[(1−p)^(n−x) p^x] + 2 ln[(1−x/n)^(n−x) (x/n)^x] ~ χ²(1) asymptotically.
Also reports the exact binomial test (two-sided p-value and the one-sided P(X >= x) for too many
exceptions). Assumes independent exceptions; low power in samples of 250.""",
          references=("Kupiec, P. (1995), Techniques for verifying the accuracy of risk measurement models, "
                      "Journal of Derivatives 3(2), 73–84",))
def kupiec(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
           var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign)
    n, x = len(d), int(d["hit"].sum())
    lr = kupiec_pof(x, n, p)
    bt = stats.binomtest(x, n, p)
    if n * p < 5:
        notes.append(f"Expected exceptions n·p = {n * p:.2f} < 5: the χ² approximation is poor; "
                     f"prefer the exact binomial p-values.")
    return Outcome({"n": n, "exceptions": x, "expected_exceptions": n * p, "exception_rate": x / n,
                    "LR_POF": lr, "p_value": float(stats.chi2.sf(lr, 1)),
                    "binomial_p_value_two_sided": float(bt.pvalue),
                    "binomial_p_value_too_many": float(stats.binom.sf(x - 1, n, p))},
                   notes=notes, rows_used=n)


@register("var.kupiec_tuff", "Kupiec time-until-first-failure (TUFF) test", "Back-testing", _MT, params=_BT,
          description="""H0: P(exception) = p. Uses only v = the position of the first exception (1-based).
LR_TUFF = −2 ln[p (1−p)^(v−1)] + 2 ln[(1/v)(1 − 1/v)^(v−1)] ~ χ²(1). Very low power; reported because
some validation standards require it. Not applicable when no exception occurred.""",
          references=("Kupiec, P. (1995), Techniques for verifying the accuracy of risk measurement models, "
                      "Journal of Derivatives 3(2), 73–84",))
def kupiec_tuff(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
                var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign)
    hits = np.flatnonzero(d["hit"].to_numpy())
    if not len(hits):
        raise NotApplicable("No exception observed: the time until first failure is undefined (censored).")
    v = int(hits[0] + 1)
    lr = _lr_tuff(v, p)
    return Outcome({"n": len(d), "time_until_first_failure": v, "expected_time": 1 / p,
                    "first_failure_at": _label(d).iloc[hits[0]], "LR_TUFF": lr,
                    "p_value": float(stats.chi2.sf(lr, 1))}, notes=notes, rows_used=len(d))


@register("var.traffic_light", "Basel traffic-light test", "Back-testing", _MT, params=_BT,
          description="""Cumulative binomial probability P(X <= x | n, p) of the observed exception count x.
Zones as defined by the Basel back-testing framework: green while that probability is below 95%, yellow
from 95% up to 99.99%, red from 99.99% (for 250 observations at 99%: green 0–4, yellow 5–9, red 10+).
For 250 observations at 99% the Basel plus factor (CRR Art. 366 addend) table is reported and the
applicable plus factor shown. For other window lengths or confidence levels the zone is the
generalisation by the same cumulative-probability cut-offs; no plus factor is defined then.""",
          references=_BASEL_REFS)
def traffic_light(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
                  var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign)
    n, x = len(d), int(d["hit"].sum())
    zone, cp = basel_zone(x, n, p)
    ks = np.arange(0, max(x, int(stats.binom.ppf(0.99999, n, p))) + 2)
    cps = stats.binom.cdf(ks, n, p)
    t = pd.DataFrame({"exceptions": ks, "probability": stats.binom.pmf(ks, n, p), "cumulative_probability": cps,
                      "zone": np.where(cps < 0.95, "green", np.where(cps < 0.9999, "yellow", "red"))})
    summary = {"n": n, "exceptions": x, "cumulative_probability": cp, "zone": zone}
    basel = n == 250 and abs(confidence - 0.99) < 1e-12
    if basel:
        t["plus_factor"] = [_PLUS.get(int(k), 1.0) for k in ks]
        summary["plus_factor"] = _PLUS.get(x, 1.0)
    else:
        notes.append(f"The Basel zones and plus factors are defined for 250 observations at 99%; here n = {n} "
                     f"at {confidence:.4g}, so the zone uses the generalised cut-offs (cumulative binomial "
                     f"probability 95% / 99.99%) and no plus factor is reported.")
    t["observed"] = t["exceptions"] == x
    return Outcome(summary, {"Traffic-light table": t}, notes=notes, rows_used=n)


# ── independence / conditional coverage ───────────────────────────────────

@register("var.christoffersen", "Christoffersen independence and conditional coverage tests",
          "Back-testing", _MT, params=_BT,
          description="""First-order Markov test on the hit sequence. Transition counts n_ij (state i on day t−1,
state j on day t), π01 = n01/(n00+n01), π11 = n11/(n10+n11), π = (n01+n11)/(n−1).
LR_ind = −2 ln[(1−π)^(n00+n10) π^(n01+n11)] + 2 ln[(1−π01)^n00 π01^n01 (1−π11)^n10 π11^n11] ~ χ²(1),
H0: exceptions independent (π01 = π11). LR_cc = LR_POF + LR_ind ~ χ²(2), H0: correct coverage AND
independence. LR_POF uses all n observations (common practice); LR_ind uses the n−1 transitions.
Only first-order dependence is detected.""",
          references=("Christoffersen, P. (1998), Evaluating interval forecasts, International Economic "
                      "Review 39(4), 841–862",))
def christoffersen_test(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
                        var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, min_n=3)
    h = d["hit"].to_numpy()
    n, x = len(h), int(h.sum())
    c = christoffersen(h)
    pof = kupiec_pof(x, n, p)
    lrcc = pof + c["LR_ind"]
    if c["n11"] == 0:
        notes.append("No consecutive exceptions (n11 = 0): π11 = 0, so LR_ind can only detect too few clusters.")
    if x == 0:
        notes.append("No exceptions: LR_ind is 0 by construction.")
    tr = pd.DataFrame({"from_state": ["no exception", "no exception", "exception", "exception"],
                       "to_state": ["no exception", "exception", "no exception", "exception"],
                       "count": [c["n00"], c["n01"], c["n10"], c["n11"]]})
    return Outcome({"n": n, "exceptions": x, "pi01": c["pi01"], "pi11": c["pi11"],
                    "LR_ind": c["LR_ind"], "p_value_ind": float(stats.chi2.sf(c["LR_ind"], 1)),
                    "LR_POF": pof, "LR_cc": lrcc, "p_value_cc": float(stats.chi2.sf(lrcc, 2))},
                   {"Transition counts": tr}, notes=notes, rows_used=n)


def _spells(hit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Durations between hits with censoring flags (1 = censored) as in Christoffersen–Pelletier (2004)."""
    t = np.flatnonzero(hit) + 1
    n = len(hit)
    d, c = [], []
    if t[0] > 1:
        d.append(t[0]); c.append(1)        # left-censored: previous hit unknown
    d += list(np.diff(t)); c += [0] * (len(t) - 1)
    if t[-1] < n:
        d.append(n - t[-1]); c.append(1)   # right-censored: next hit not yet observed
    return np.asarray(d, float), np.asarray(c, int)


@register("var.duration_weibull", "Christoffersen–Pelletier Weibull duration test", "Back-testing", _MT,
          params=_BT,
          description="""Duration-based independence test. Durations D between exceptions are modelled as Weibull with
hazard λ(d) = a^b b d^(b−1); the first and last spells are censored when the sample does not start/end with
an exception. H0: b = 1 (exponential = memoryless durations, i.e. no clustering).
Maximum likelihood with a profiled out (a^b = N_uncensored / Σ D^b), b maximised numerically on [0.01, 100].
LR = 2[ln L(â, b̂) − ln L(ã, 1)] ~ χ²(1). b < 1 indicates clustering (decreasing hazard).
Needs at least 3 exceptions; low power below ~10.""",
          references=("Christoffersen, P. and Pelletier, D. (2004), Backtesting value-at-risk: a duration-based "
                      "approach, Journal of Financial Econometrics 2(1), 84–108",))
def duration_weibull(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
                     var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign)
    h = d["hit"].to_numpy()
    if h.sum() < 3:
        raise NotApplicable(f"{int(h.sum())} exceptions; the duration test needs at least 3.")
    D, C = _spells(h)
    u = C == 0
    nu, slog = int(u.sum()), float(np.log(D[u]).sum())

    def ll(b):
        return nu * math.log(nu / float((D ** b).sum())) + nu * math.log(b) + (b - 1) * slog - nu

    fit = optimize.minimize_scalar(lambda lb: -ll(math.exp(lb)), bounds=(math.log(0.01), math.log(100)),
                                   method="bounded", options={"xatol": 1e-10})
    b = math.exp(fit.x)
    ll1, ll0 = max(ll(b), ll(1.0)), ll(1.0)
    lr = max(2 * (ll1 - ll0), 0.0)
    a = (nu / float((D ** b).sum())) ** (1 / b)
    if abs(fit.x - math.log(0.01)) < 1e-4 or abs(fit.x - math.log(100)) < 1e-4:
        notes.append("Weibull shape b at the search bound: the likelihood is flat/degenerate; interpret with care.")
    if h.sum() < 10:
        notes.append(f"Only {int(h.sum())} exceptions: the duration test has little power.")
    t = pd.DataFrame({"duration": D.astype(int), "censored": C.astype(bool)})
    return Outcome({"n": len(d), "exceptions": int(h.sum()), "uncensored_durations": nu,
                    "mean_duration": float(D[u].mean()), "expected_duration": 1 / p,
                    "weibull_b": b, "weibull_a": a, "LR": lr, "p_value": float(stats.chi2.sf(lr, 1))},
                   {"Durations": t}, notes=notes, rows_used=len(d))


@register("var.haas_mixed_kupiec", "Haas mixed Kupiec test", "Back-testing", _MT, params=_BT,
          description="""Combines coverage and independence through the time between exceptions. For each exception i
with v_i days since the previous exception (v_1 = days since the start),
LR_i = −2 ln[p (1−p)^(v_i−1)] + 2 ln[(1/v_i)(1 − 1/v_i)^(v_i−1)].
LR_ind = Σ LR_i ~ χ²(x); LR_mix = LR_ind + LR_POF ~ χ²(x+1). H0: exceptions independent with
probability p. Not applicable without exceptions.""",
          references=("Haas, M. (2001), New methods in backtesting, Financial Engineering Research Center "
                      "caesar, Bonn",
                      "Kupiec, P. (1995), Journal of Derivatives 3(2), 73–84"))
def haas(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
         var_sign="auto") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign)
    h = d["hit"].to_numpy()
    n, x = len(h), int(h.sum())
    if x == 0:
        raise NotApplicable("No exceptions: the mixed Kupiec test needs at least one.")
    t = np.flatnonzero(h) + 1
    v = np.diff(np.concatenate([[0], t]))
    lri = np.array([_lr_tuff(int(k), p) for k in v])
    pof = kupiec_pof(x, n, p)
    lr_ind = float(lri.sum())
    tab = pd.DataFrame({"exception": np.arange(1, x + 1), "position": t, "days_since_previous": v,
                        "LR_i": lri})
    return Outcome({"n": n, "exceptions": x, "LR_ind": lr_ind, "p_value_ind": float(stats.chi2.sf(lr_ind, x)),
                    "LR_POF": pof, "LR_mix": lr_ind + pof,
                    "p_value_mix": float(stats.chi2.sf(lr_ind + pof, x + 1))},
                   {"Exception spacing": tab}, notes=notes, rows_used=n)


@register("var.dynamic_quantile", "Engle–Manganelli Dynamic Quantile (DQ) test", "Back-testing", _MT,
          params=(*_BT, P("lags", "integer", default=4, help="Number of lagged hits in the regression"),
                  P("include_var", "boolean", default=True, help="Include VaR_t as a regressor")),
          description="""Regress the demeaned hit Hit_t = I_t − p on X_t = [1, Hit_{t−1}, …, Hit_{t−lags}, VaR_t] by OLS.
DQ = Hit'X(X'X)^(−1)X'Hit / (p(1−p)) ~ χ²(q), q = number of regressors (the rank of X when collinear).
H0: hits are unpredictable and have mean p (correct conditional coverage). Table shows the OLS coefficients.""",
          references=("Engle, R. F. and Manganelli, S. (2004), CAViaR: conditional autoregressive value at risk "
                      "by regression quantiles, Journal of Business & Economic Statistics 22(4), 367–381",))
def dq(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False, var_sign="auto",
       lags=4, include_var=True) -> Outcome:
    if lags < 0:
        raise ValueError("lags must be >= 0")
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, min_n=lags + 10)
    hit = d["hit"].to_numpy(float) - p
    y = hit[lags:]
    cols, names = [np.ones_like(y)], ["constant"]
    for j in range(1, lags + 1):
        cols.append(hit[lags - j:len(hit) - j]); names.append(f"Hit(t-{j})")
    if include_var:
        cols.append(d["var"].to_numpy(float)[lags:]); names.append("VaR(t)")
    X = np.column_stack(cols)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    stat = float(y @ fitted / (p * (1 - p)))
    if rank < X.shape[1]:
        notes.append(f"Regressors are collinear (rank {rank} of {X.shape[1]}, e.g. no exceptions in the lags); "
                     f"degrees of freedom set to the rank.")
    tab = pd.DataFrame({"regressor": names, "coefficient": beta})
    return Outcome({"n_regression": len(y), "regressors": X.shape[1], "df": int(rank), "DQ": stat,
                    "p_value": float(stats.chi2.sf(stat, rank))},
                   {"DQ regression": tab}, notes=notes, rows_used=len(d))


# ── expected shortfall ────────────────────────────────────────────────────

_ES_PARAMS = (P("pnl"), P("var"), P("es"), P("date", required=False),
              P("confidence", "number", default=0.975, help="VaR/ES confidence level, e.g. 0.975"),
              _LOSSPOS, _VSIGN)


def _std_tail(dist: str, df: float, c: float) -> tuple[float, float, object]:
    """Quantile z_c and tail mean e_c = E[Z | Z > z_c] of the standard distribution."""
    if dist == "normal":
        z = stats.norm.ppf(c)
        return z, stats.norm.pdf(z) / (1 - c), stats.norm
    if df <= 1:
        raise ValueError("t degrees of freedom must exceed 1 for ES to exist")
    z = stats.t.ppf(c, df)
    return z, (df + z ** 2) / (df - 1) * stats.t.pdf(z, df) / (1 - c), stats.t(df)


@register("var.es_acerbi_szekely", "Acerbi–Szekely ES back-tests Z1 and Z2", "Back-testing", _MT,
          params=(*_ES_PARAMS,
                  P("distribution", "string", default="t", choices=("normal", "t"),
                    help="Null distribution family used to simulate P&L for p-values"),
                  P("df", "number", default=5.0, help="Degrees of freedom when distribution='t'"),
                  P("n_sims", "integer", default=10000, help="Monte Carlo scenarios under H0")),
          description="""With losses L_t, VaR_t, ES_t (positive), I_t = 1{L_t > VaR_t}, α = 1 − c, N = Σ I_t:
Z1 = 1 − (1/N) Σ I_t L_t / ES_t  (conditional on exceptions; tests ES given that VaR is right),
Z2 = 1 − Σ I_t L_t / (T α ES_t)  (unconditional; tests VaR and ES jointly).
E[Z] = 0 under H0; negative values mean risk is underestimated. One-sided p-value = P(Z_sim <= Z_obs) by
Monte Carlo under H0: each day's loss is simulated from a location–scale distribution (normal or Student-t
with `df`) whose location and scale are solved so that its VaR and ES at c equal the reported VaR_t and
ES_t exactly. The p-values therefore depend on that distributional assumption (the statistics do not).
Acerbi–Szekely report that Z2 critical values are nearly distribution-independent (about −0.70 at 5% for
c = 97.5%, T = 250), so the simulated p-value is insensitive to the family. Requires ES_t > 0 and ES_t >= VaR_t.""",
          references=("Acerbi, C. and Szekely, B. (2014), Back-testing expected shortfall, Risk, December",
                      "Acerbi, C. and Szekely, B. (2017), General properties of backtestable statistics, "
                      "SSRN working paper"))
def acerbi_szekely(ctx: RunContext, pnl, var, es, date=None, confidence=0.975, loss_positive=False,
                   var_sign="auto", distribution="t", df=5.0, n_sims=10000) -> Outcome:
    d, a, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, es=es, min_n=10)
    L, V, E = d["loss"].to_numpy(), d["var"].to_numpy(), d["es"].to_numpy()
    if (E <= 0).any():
        raise ValueError(f"{int((E <= 0).sum())} ES values are <= 0; the Z tests need positive ES.")
    if (E < V - 1e-12 * np.abs(V)).any():
        raise ValueError(f"ES < VaR on {int((E < V).sum())} rows; ES must be >= VaR.")
    if n_sims < 100:
        raise ValueError("n_sims must be >= 100")
    T = len(L)
    I = L > V

    def z12(Ls, Is):
        N = Is.sum(axis=-1)
        s = (Is * Ls / E).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            z1 = np.where(N > 0, 1 - s / np.maximum(N, 1), np.nan)
        return z1, 1 - s / (T * a)

    z1, z2 = z12(L, I)
    zc, ec, law = _std_tail(distribution, df, confidence)
    sig = (E - V) / (ec - zc)
    mu = V - sig * zc
    rng = ctx.rng(11)
    s1, s2 = [], []
    for start in range(0, n_sims, 1000):
        m = min(1000, n_sims - start)
        Z = rng.standard_normal((m, T)) if distribution == "normal" else rng.standard_t(df, (m, T))
        Ls = mu + sig * Z
        r1, r2 = z12(Ls, Ls > V)
        s1.append(r1); s2.append(r2)
    s1, s2 = np.concatenate(s1), np.concatenate(s2)
    ok1 = np.isfinite(s1)
    p1 = float((s1[ok1] <= z1).mean()) if np.isfinite(z1) and ok1.any() else float("nan")
    p2 = float((s2 <= z2).mean())
    if not I.any():
        notes.append("No exceptions: Z1 is undefined.")
    notes.append(f"p-values by {n_sims} simulations under H0 with a location–scale "
                 f"{'normal' if distribution == 'normal' else f'Student-t (df={df:g})'} matching each day's VaR and ES; "
                 f"{int((~ok1).sum())} simulated samples without exceptions were excluded from the Z1 p-value.")
    tab = pd.DataFrame({"statistic": ["Z1", "Z2"], "value": [float(z1), float(z2)], "p_value": [p1, p2],
                        "simulated_mean": [float(np.nanmean(s1)), float(s2.mean())],
                        "simulated_5pct_quantile": [float(np.nanquantile(s1, 0.05)), float(np.quantile(s2, 0.05))]})
    return Outcome({"n": T, "exceptions": int(I.sum()), "expected_exceptions": T * a, "Z1": float(z1),
                    "p_value_Z1": p1, "Z2": float(z2), "p_value_Z2": p2},
                   {"Acerbi–Szekely tests": tab}, notes=notes, rows_used=T)


@register("var.es_mcneil_frey", "McNeil–Frey exceedance-residual test for ES", "Back-testing", _MT,
          params=(*_ES_PARAMS,
                  P("volatility", required=False, help="Conditional volatility forecast to standardise residuals"),
                  P("n_boot", "integer", default=10000, help="Bootstrap replications")),
          description="""On exception days (L_t > VaR_t) the exceedance residuals r_t = (L_t − ES_t)/σ_t (σ_t = the
`volatility` column; unscaled L_t − ES_t when omitted) have mean zero if ES is correct.
H0: E[r] = 0 vs H1: E[r] > 0 (ES underestimated). t = mean(r)/(sd(r)/√N); the p-value is from the
bootstrap distribution of t computed on the mean-centred residuals (Efron–Tibshirani). A two-sided
p-value is also given. Needs at least 2 exceptions; meaningful from about 10.""",
          references=("McNeil, A. J. and Frey, R. (2000), Estimation of tail-related risk measures for "
                      "heteroscedastic financial time series: an extreme value approach, Journal of Empirical "
                      "Finance 7, 271–300",
                      "Efron, B. and Tibshirani, R. (1993), An Introduction to the Bootstrap, ch. 16"))
def mcneil_frey(ctx: RunContext, pnl, var, es, date=None, confidence=0.975, loss_positive=False,
                var_sign="auto", volatility=None, n_boot=10000) -> Outcome:
    p = _check_conf(confidence)
    d, notes, dropped = load_series(ctx, {"pnl": pnl, "var": var, "es": es, "vol": volatility}, date,
                                    loss_positive, var_sign)
    notes += dropped_note(dropped)
    hit = d["loss"] > d["var"]
    r = (d["loss"] - d["es"])[hit].to_numpy()
    if volatility:
        s = d["vol"][hit].to_numpy()
        if (s <= 0).any():
            raise ValueError("volatility must be positive")
        r = r / s
    else:
        notes.append("No volatility column: residuals are unscaled (L − ES); heteroscedasticity can distort the test.")
    N = len(r)
    if N < 2:
        raise NotApplicable(f"{N} exceptions; the test needs at least 2.")
    sd = r.std(ddof=1)
    if sd == 0:
        raise NotApplicable("All exceedance residuals are equal; the t statistic is undefined.")
    t_obs = r.mean() / (sd / math.sqrt(N))
    rc = r - r.mean()
    idx = ctx.rng(13).integers(0, N, size=(n_boot, N))
    b = rc[idx]
    bs = b.std(axis=1, ddof=1)
    tb = np.where(bs > 0, b.mean(axis=1) / (np.where(bs > 0, bs, 1) / math.sqrt(N)), 0.0)
    if N < 10:
        notes.append(f"Only {N} exceptions: bootstrap p-values are unreliable.")
    return Outcome({"n": len(d), "exceptions": N, "mean_residual": float(r.mean()), "t_statistic": float(t_obs),
                    "p_value_one_sided": float((tb >= t_obs).mean()),
                    "p_value_two_sided": float((np.abs(tb) >= abs(t_obs)).mean())},
                   {"Exceedance residuals": pd.DataFrame({"position": np.flatnonzero(hit.to_numpy()) + 1,
                                                          "residual": r})},
                   notes=notes, rows_used=len(d))


# ── distribution forecasts (PIT) ──────────────────────────────────────────

def _pit(ctx, pit, date):
    d, notes, dropped = load_series(ctx, {"pit": pit}, date, sign_keys=())
    u = d["pit"].to_numpy(float)
    if ((u < 0) | (u > 1)).any():
        raise ValueError("PIT values must lie in [0, 1].")
    k = int(((u <= 0) | (u >= 1)).sum())
    if k:
        notes.append(f"{k} PIT values at 0 or 1 clipped to [1e-6, 1 − 1e-6].")
        u = np.clip(u, 1e-6, 1 - 1e-6)
    return u, notes + dropped_note(dropped)


@register("var.berkowitz", "Berkowitz likelihood-ratio test on PIT values", "Back-testing", _MT,
          params=(P("pit", help="Probability integral transform of realised P&L under the forecast distribution"),
                  P("date", required=False)),
          description="""z_t = Φ^(−1)(PIT_t) is N(0,1) i.i.d. if the forecast distributions are correct.
Fits z_t − μ = ρ(z_{t−1} − μ) + ε_t, ε ~ N(0, σ²) by exact Gaussian AR(1) maximum likelihood.
LR_3 = 2[ln L(μ̂, σ̂, ρ̂) − ln L(0, 1, 0)] ~ χ²(3) (H0: μ=0, σ=1, ρ=0: correct distribution and
independence); LR_ind = 2[ln L(μ̂, σ̂, ρ̂) − ln L(μ̃, σ̃, 0)] ~ χ²(1) (H0: ρ = 0).
Tests normality only through the first two moments.""",
          references=("Berkowitz, J. (2001), Testing density forecasts, with applications to risk management, "
                      "Journal of Business & Economic Statistics 19(4), 465–474",))
def berkowitz(ctx: RunContext, pit, date=None) -> Outcome:
    u, notes = _pit(ctx, pit, date)
    if len(u) < 10:
        raise NotApplicable(f"{len(u)} PIT values; at least 10 needed.")
    r = berkowitz_lr(stats.norm.ppf(u))
    if not r.pop("converged"):
        notes.append("Optimiser reported non-convergence; estimates are the best point found.")
    return Outcome({"n": len(u), **r}, notes=notes, rows_used=len(u))


@register("var.pit_uniformity", "Uniformity tests of PIT values (KS, Anderson–Darling, CvM, chi-square)",
          "Back-testing", _MT,
          params=(P("pit"), P("date", required=False),
                  P("n_sims", "integer", default=2000, help="Simulations for the Anderson–Darling p-value"),
                  P("bins", "integer", default=10, help="Bins for the chi-square test")),
          description="""H0: PIT values are U(0,1) (the forecast distributions are correctly calibrated).
Kolmogorov–Smirnov, Anderson–Darling (tail-weighted; p-value by Monte Carlo with uniform samples of the same
size), Cramér–von Mises and Pearson chi-square on equal-width bins. All assume independent PIT values;
overlapping horizons make them over-reject. Also reports the lag-1 autocorrelation of the PITs.""",
          references=("Diebold, F. X., Gunther, T. A. and Tay, A. S. (1998), Evaluating density forecasts with "
                      "applications to financial risk management, International Economic Review 39(4), 863–883",
                      "Anderson, T. W. and Darling, D. A. (1954), A test of goodness of fit, JASA 49, 765–769"))
def pit_uniformity(ctx: RunContext, pit, date=None, n_sims=2000, bins=10) -> Outcome:
    u, notes = _pit(ctx, pit, date)
    if len(u) < 5:
        raise NotApplicable(f"{len(u)} PIT values; at least 5 needed.")
    t = pit_uniformity_table(u, ctx.rng(17), n_sims, bins)
    ac = float(np.corrcoef(u[:-1], u[1:])[0, 1]) if len(u) > 2 and u.std() > 0 else float("nan")
    return Outcome({"n": len(u), **{f"{r.test} statistic": r.statistic for r in t.itertuples()},
                    **{f"{r.test} p_value": r.p_value for r in t.itertuples()}, "lag1_autocorrelation": ac},
                   {"Uniformity tests": t}, notes=notes, rows_used=len(u))


# ── loss functions ────────────────────────────────────────────────────────

@register("var.loss_functions", "VaR / ES scoring functions (quantile loss, Lopez, FZ0)", "Back-testing", _MT,
          params=(*_BT, P("es", required=False),
                  P("benchmark_var", required=False, help="A second VaR series to compare (same sign convention)")),
          description="""Scoring functions for comparing forecasts (lower = better):
quantile (pinball) loss ρ_c(L − VaR) = (L − VaR)(c − 1{L < VaR}) (strictly consistent for VaR);
Lopez (1999) magnitude loss 1{L > VaR}·[1 + (L − VaR)²]; mean exceedance (L − VaR)/VaR on exception days;
with ES: the FZ0 joint VaR–ES loss of Patton, Ziegel and Chen (2019), written for returns Y = −L,
v = −VaR, e = −ES: −1{Y <= v}(v − Y)/(α e) + v/e + ln(−e) − 1. With `benchmark_var` the mean quantile-loss
difference is tested with the Diebold–Mariano statistic (Newey–West variance), H0: equal predictive accuracy.""",
          references=("Lopez, J. A. (1999), Methods for evaluating value-at-risk estimates, FRBSF Economic Review 2, 3–17",
                      "Koenker, R. and Bassett, G. (1978), Regression quantiles, Econometrica 46(1), 33–50",
                      "Patton, A. J., Ziegel, J. F. and Chen, R. (2019), Dynamic semiparametric models for expected "
                      "shortfall (and value-at-risk), Journal of Econometrics 211(2), 388–413",
                      "Diebold, F. X. and Mariano, R. S. (1995), Comparing predictive accuracy, JBES 13(3), 253–263"))
def loss_functions(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False,
                   var_sign="auto", es=None, benchmark_var=None) -> Outcome:
    p = _check_conf(confidence)
    d, notes, dropped = load_series(ctx, {"pnl": pnl, "var": var, "es": es, "bvar": benchmark_var}, date,
                                    loss_positive, var_sign, sign_keys=("var", "es", "bvar"))
    notes += dropped_note(dropped)
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 complete observations.")
    L = d["loss"].to_numpy()

    def scores(V):
        u = L - V
        hit = u > 0
        return {"quantile_loss": u * (confidence - (u < 0)), "lopez_magnitude": hit * (1 + u ** 2), "hit": hit,
                "rel": np.where(hit, u / np.where(V != 0, V, np.nan), np.nan)}

    s = scores(d["var"].to_numpy())
    rows = [{"model": "model", "mean_quantile_loss": s["quantile_loss"].mean(),
             "lopez_magnitude_total": s["lopez_magnitude"].sum(), "exceptions": int(s["hit"].sum()),
             "mean_relative_exceedance": float(np.nanmean(s["rel"])) if s["hit"].any() else float("nan")}]
    summ = {"n": len(d), "mean_quantile_loss": float(rows[0]["mean_quantile_loss"]),
            "lopez_magnitude_total": float(rows[0]["lopez_magnitude_total"])}
    if es:
        Ev = d["es"].to_numpy()
        if (Ev <= 0).any():
            raise ValueError("FZ0 loss needs ES > 0 (as a loss amount).")
        Y, v, e = -L, -d["var"].to_numpy(), -Ev
        fz = -((Y <= v) * (v - Y)) / (p * e) + v / e + np.log(-e) - 1
        rows[0]["mean_FZ0_loss"] = float(fz.mean())
        summ["mean_FZ0_loss"] = float(fz.mean())
    if benchmark_var:
        sb = scores(d["bvar"].to_numpy())
        rows.append({"model": "benchmark", "mean_quantile_loss": sb["quantile_loss"].mean(),
                     "lopez_magnitude_total": sb["lopez_magnitude"].sum(), "exceptions": int(sb["hit"].sum()),
                     "mean_relative_exceedance": float(np.nanmean(sb["rel"])) if sb["hit"].any() else float("nan")})
        diff = s["quantile_loss"] - sb["quantile_loss"]
        v = _newey_west_var(diff)
        dm = float(diff.mean() / math.sqrt(v / len(diff))) if v > 0 else float("nan")
        summ |= {"benchmark_mean_quantile_loss": float(sb["quantile_loss"].mean()), "DM_statistic": dm,
                 "DM_p_value": float(2 * stats.norm.sf(abs(dm))) if np.isfinite(dm) else float("nan")}
        notes.append("DM statistic > 0 means the model has higher (worse) quantile loss than the benchmark.")
    return Outcome(summ, {"Scores": pd.DataFrame(rows)}, notes=notes, rows_used=len(d))


# ── FRTB P&L attribution ──────────────────────────────────────────────────

@register("var.pla_test", "FRTB P&L attribution test (Spearman correlation and KS distance)", "P&L attribution",
          _MT, params=(P("hpl", help="Hypothetical P&L column"), P("rtpl", help="Risk-theoretical P&L column"),
                       P("date", required=False)),
          description="""Basel MAR32 P&L attribution test metrics between hypothetical P&L (HPL) and risk-theoretical
P&L (RTPL) of a trading desk: Spearman rank correlation, and the Kolmogorov–Smirnov distance between the two
empirical distributions (max |F_HPL − F_RTPL|). Zones as defined in MAR32 (for 250 observations): green if
Spearman > 0.80 and KS < 0.09; red if Spearman < 0.70 or KS > 0.12; amber otherwise. Also reports the
Pearson correlation, KS p-value and the mean/variance of the difference for diagnosis.""",
          references=("Basel Committee on Banking Supervision, Minimum capital requirements for market risk "
                      "(FRTB, January 2019), MAR32 Backtesting and P&L attribution test requirements",))
def pla(ctx: RunContext, hpl, rtpl, date=None) -> Outcome:
    d, notes, dropped = load_series(ctx, {"hpl": hpl, "rtpl": rtpl}, date, sign_keys=())
    notes += dropped_note(dropped)
    if len(d) < 10:
        raise NotApplicable(f"{len(d)} observations; at least 10 needed.")
    h, r = d["hpl"].to_numpy(), d["rtpl"].to_numpy()
    if h.std() == 0 or r.std() == 0:
        raise NotApplicable("HPL or RTPL is constant; correlation undefined.")
    rho = float(stats.spearmanr(h, r).statistic)
    ks = stats.ks_2samp(h, r)
    zs = "green" if rho > 0.80 else "amber" if rho >= 0.70 else "red"
    zk = "green" if ks.statistic < 0.09 else "amber" if ks.statistic <= 0.12 else "red"
    zone = "red" if "red" in (zs, zk) else "green" if zs == zk == "green" else "amber"
    if len(d) != 250:
        notes.append(f"MAR32 zones are calibrated for 250 observations; here n = {len(d)}.")
    diff = h - r
    tab = pd.DataFrame({"metric": ["Spearman correlation", "KS distance"], "value": [rho, float(ks.statistic)],
                        "zone": [zs, zk]})
    return Outcome({"n": len(d), "spearman": rho, "ks_statistic": float(ks.statistic),
                    "ks_p_value": float(ks.pvalue), "pla_zone": zone,
                    "pearson": float(np.corrcoef(h, r)[0, 1]), "mean_difference": float(diff.mean()),
                    "variance_ratio_unexplained": float(diff.var(ddof=1) / h.var(ddof=1))},
                   {"PLA metrics": tab}, notes=notes, rows_used=len(d))


# ── independent replication ───────────────────────────────────────────────

@register("var.hs_var", "Historical-simulation VaR/ES replication from a P&L vector", "Replication", _MT,
          params=(P("scenarios", "table", required=False,
                    help="Table of scenario P&L (positions × scenarios or scenarios × positions)"),
                  P("pnl", required=False, help="Alternatively: a P&L strip column in the active table"),
                  P("layout", "string", default="rows_are_scenarios",
                    choices=("rows_are_scenarios", "columns_are_scenarios"),
                    help="Orientation of the scenarios table"),
                  _CONF, _LOSSPOS,
                  P("window", "integer", default=0, help="Use only the last `window` scenarios (0 = all)"),
                  P("method", "string", default="inverted_cdf", choices=_QMETHODS,
                    help="Quantile convention (numpy); inverted_cdf = the empirical-distribution VaR"),
                  P("reported_var", "number", required=False, help="Bank's reported VaR (positive loss)"),
                  P("reported_es", "number", required=False, help="Bank's reported ES (positive loss)")),
          description="""Independent HS VaR and ES. Portfolio scenario P&L = sum over positions; losses = −P&L
(unless loss_positive). VaR = quantile of losses at c with the chosen convention (default: empirical
distribution, e.g. the 3rd-largest of 250 losses at 99%); ES = ES of the empirical distribution
(Acerbi–Tasche: mean of the worst n·α losses with the boundary loss fractionally weighted). A table shows VaR
under all common quantile conventions (they differ in small samples). With reported figures the absolute and
relative differences are given. Scenario rows with missing values are dropped.""",
          references=("Acerbi, C. and Tasche, D. (2002), On the coherence of expected shortfall, Journal of "
                      "Banking & Finance 26(7), 1487–1503",
                      "Hyndman, R. J. and Fan, Y. (1996), Sample quantiles in statistical packages, "
                      "The American Statistician 50(4), 361–365"))
def hs_var(ctx: RunContext, scenarios=None, pnl=None, layout="rows_are_scenarios", confidence=0.99,
           loss_positive=False, window=0, method="inverted_cdf", reported_var=None, reported_es=None) -> Outcome:
    _check_conf(confidence)
    notes = []
    if (scenarios is None) == (pnl is None):
        raise ValueError("Give exactly one of `scenarios` (a table) or `pnl` (a column).")
    if scenarios is not None:
        t = ctx.tables[scenarios]
        if layout == "columns_are_scenarios":
            t = t.T
        numt = t.apply(pd.to_numeric, errors="coerce")
        keep = [c for c in numt.columns if numt[c].notna().any()]
        if len(keep) < t.shape[1]:
            notes.append(f"{t.shape[1] - len(keep)} non-numeric position column(s) ignored.")
        numt = numt[keep]
        n0 = len(numt)
        numt = numt.dropna()
        notes += dropped_note(n0 - len(numt), "scenarios with missing P&L")
        x = numt.sum(axis=1).to_numpy(float)
        npos = len(keep)
    else:
        s = num(ctx.df, pnl)
        notes += dropped_note(int(s.isna().sum()), "scenarios with missing P&L")
        x = s.dropna().to_numpy(float)
        npos = 1
    if window:
        x = x[-window:]
    L = x if loss_positive else -x
    n = len(L)
    if n < 2 or n * (1 - confidence) < 1:
        raise NotApplicable(f"{n} scenarios cannot support a {confidence:.4g} quantile (need n·(1−c) >= 1).")
    v = float(np.quantile(L, confidence, method=method))
    e = empirical_es(L, confidence)
    conv = pd.DataFrame({"method": list(_QMETHODS),
                         "VaR": [float(np.quantile(L, confidence, method=m)) for m in _QMETHODS]})
    summ = {"scenarios": n, "positions": npos, "VaR": v, "ES": e, "method": method}
    tabs = {"VaR by quantile convention": conv,
            "Worst losses": pd.DataFrame({"rank": np.arange(1, min(n, 20) + 1),
                                          "loss": np.sort(L)[::-1][:min(n, 20)]})}
    rows = []
    for lab, mine, rep in (("VaR", v, reported_var), ("ES", e, reported_es)):
        if rep is not None:
            rows.append({"measure": lab, "replicated": mine, "reported": rep, "difference": rep - mine,
                         "relative_difference": (rep - mine) / mine if mine else float("nan")})
            summ[f"{lab}_difference"] = rep - mine
    if rows:
        tabs["Comparison with reported"] = pd.DataFrame(rows)
    return Outcome(summ, tabs, notes=notes, rows_used=n)


@register("var.hs_rolling_replication", "Rolling historical-simulation VaR vs reported VaR", "Replication", _MT,
          params=(*_BT, P("window", "integer", default=250, help="Look-back window of past P&L"),
                  P("method", "string", default="inverted_cdf", choices=_QMETHODS)),
          description="""For each day t, VaR_HS(t) = the c-quantile of losses over the previous `window` days
(t−window … t−1) of the P&L series, compared with the reported VaR_t: difference (reported − replicated),
relative difference, and summary statistics. Appropriate when the P&L column is the scenario P&L of the
portfolio being measured (e.g. a plain HS model on a static book); for a changing book it is only a proxy.""",
          references=("Jorion, P. (2007), Value at Risk, 3rd ed., ch. 10 (historical simulation)",))
def hs_rolling(ctx: RunContext, pnl, var, date=None, confidence=0.99, loss_positive=False, var_sign="auto",
               window=250, method="inverted_cdf") -> Outcome:
    d, p, notes, _ = _bt(ctx, pnl, var, date, confidence, loss_positive, var_sign, min_n=window + 1)
    L = d["loss"].to_numpy()
    w = np.lib.stride_tricks.sliding_window_view(L, window)[:-1]
    rep = np.quantile(w, confidence, axis=1, method=method)
    rv = d["var"].to_numpy()[window:]
    diff = rv - rep
    t = pd.DataFrame({"observation": _label(d).iloc[window:].to_numpy(), "reported_VaR": rv,
                      "replicated_VaR": rep, "difference": diff,
                      "relative_difference": diff / np.where(rep != 0, rep, np.nan)})
    return Outcome({"days_compared": len(t), "mean_difference": float(diff.mean()),
                    "mean_abs_difference": float(np.abs(diff).mean()), "max_abs_difference": float(np.abs(diff).max()),
                    "mean_relative_difference": float(np.nanmean(t["relative_difference"])),
                    "correlation": float(np.corrcoef(rv, rep)[0, 1]) if rv.std() > 0 and rep.std() > 0
                    else float("nan")},
                   {"Replication by day": t}, notes=notes, rows_used=len(d))


@register("var.parametric_var", "Parametric (variance–covariance) VaR replication", "Replication", _MT,
          params=(P("returns", "table", help="Table of risk-factor returns, rows = dates, one column per factor"),
                  P("weights", "dict", help="Exposure per factor column, e.g. {\"EQ\": 1e6, \"FX\": -5e5}"),
                  _CONF, P("horizon_days", "integer", default=1, help="Horizon; scaled by √h"),
                  P("ewma_lambda", "number", default=0.0, help="EWMA decay (e.g. 0.94); 0 = equal weights"),
                  P("include_mean", "boolean", default=False, help="Subtract the mean P&L from VaR"),
                  P("reported_var", "number", required=False, help="Bank's reported VaR (positive loss)")),
          description="""Delta-normal VaR: σ_p = √(wᵀΣw) with Σ the sample covariance of returns (or the
RiskMetrics EWMA covariance about zero mean with decay λ). VaR = √h (z_c σ_p) − h μ_p (μ only when
include_mean), ES = √h σ_p φ(z_c)/(1−c) − h μ_p. Component VaR_i = w_i (Σw)_i / σ_p · z_c √h (sums to the
zero-mean VaR). Assumes jointly normal, linear P&L and i.i.d. returns for the √h scaling.""",
          references=("J.P. Morgan/Reuters (1996), RiskMetrics — Technical Document, 4th ed.",
                      "Jorion, P. (2007), Value at Risk, 3rd ed., ch. 7"))
def parametric_var(ctx: RunContext, returns, weights, confidence=0.99, horizon_days=1, ewma_lambda=0.0,
                   include_mean=False, reported_var=None) -> Outcome:
    _check_conf(confidence)
    R = ctx.tables[returns]
    missing = [k for k in weights if k not in R.columns]
    if missing:
        raise ValueError(f"weights refer to columns not in '{returns}': {missing}")
    cols = sorted(weights, key=list(R.columns).index)
    X = R[cols].apply(pd.to_numeric, errors="coerce")
    n0 = len(X)
    X = X.dropna().to_numpy(float)
    notes = dropped_note(n0 - len(X), "return rows with missing values")
    if len(X) < len(cols) + 2:
        raise NotApplicable(f"{len(X)} return observations for {len(cols)} factors.")
    w = np.array([float(weights[c]) for c in cols])
    if ewma_lambda:
        if not 0 < ewma_lambda < 1:
            raise ValueError("ewma_lambda must be in (0, 1)")
        k = np.arange(len(X))[::-1]
        a = (1 - ewma_lambda) * ewma_lambda ** k
        a = a / a.sum()
        S = (X * a[:, None]).T @ X
        mu = np.zeros(len(cols))
        notes.append(f"EWMA covariance with λ = {ewma_lambda}, zero mean (RiskMetrics).")
    else:
        S = np.cov(X, rowvar=False, ddof=1).reshape(len(cols), len(cols))
        mu = X.mean(axis=0)
    sp = float(math.sqrt(w @ S @ w))
    mp = float(w @ mu) if include_mean else 0.0
    z = stats.norm.ppf(confidence)
    h = horizon_days
    v = math.sqrt(h) * z * sp - h * mp
    e = math.sqrt(h) * sp * stats.norm.pdf(z) / (1 - confidence) - h * mp
    comp = w * (S @ w) / sp * z * math.sqrt(h) if sp > 0 else np.zeros_like(w)
    t = pd.DataFrame({"factor": cols, "exposure": w, "volatility": np.sqrt(np.diag(S)), "component_VaR": comp,
                      "share": comp / comp.sum() if comp.sum() else np.nan})
    summ = {"observations": len(X), "portfolio_sigma": sp, "VaR": v, "ES": e, "horizon_days": h}
    if reported_var is not None:
        summ |= {"reported_VaR": reported_var, "difference": reported_var - v,
                 "relative_difference": (reported_var - v) / v if v else float("nan")}
    return Outcome(summ, {"Component VaR": t}, notes=notes, rows_used=len(X))


@register("var.scaling_check", "Square-root-of-time scaling check (1-day vs h-day)", "Replication", _MT,
          params=(P("pnl"), P("date", required=False), _CONF, _LOSSPOS,
                  P("horizon", "integer", default=10, help="Holding period in days"),
                  P("method", "string", default="inverted_cdf", choices=_QMETHODS)),
          description="""Compares √h × (1-day HS VaR) with the HS VaR of overlapping h-day P&L (rolling sums) and of
non-overlapping h-day P&L. Ratio = h-day VaR / scaled 1-day VaR; 1 under i.i.d. P&L with stable
distribution. Also reports the variance ratio Var(h-day)/(h·Var(1-day)) and the lag-1 autocorrelation of
daily P&L (positive autocorrelation makes √h scaling understate risk). Overlapping sums are serially
dependent, so their quantile is noisy.""",
          references=("Basel Committee on Banking Supervision (2011), Messages from the academic literature on "
                      "risk measurement for the trading book, Working Paper 19",
                      "Lo, A. W. and MacKinlay, A. C. (1988), Stock market prices do not follow random walks, "
                      "Review of Financial Studies 1(1), 41–66"))
def scaling_check(ctx: RunContext, pnl, date=None, confidence=0.99, loss_positive=False, horizon=10,
                  method="inverted_cdf") -> Outcome:
    _check_conf(confidence)
    d, notes, dropped = load_series(ctx, {"pnl": pnl}, date, loss_positive, sign_keys=())
    notes += dropped_note(dropped)
    L = d["loss"].to_numpy()
    h = horizon
    if h < 2 or len(L) < 5 * h:
        raise NotApplicable(f"{len(L)} observations for a {h}-day horizon (need at least {5 * h}, h >= 2).")
    v1 = float(np.quantile(L, confidence, method=method))
    ov = np.convolve(L, np.ones(h), "valid")
    nov = L[: len(L) // h * h].reshape(-1, h).sum(axis=1)
    vo = float(np.quantile(ov, confidence, method=method))
    vn = float(np.quantile(nov, confidence, method=method)) if len(nov) * (1 - confidence) >= 1 else float("nan")
    if len(nov) * (1 - confidence) < 1:
        notes.append("Too few non-overlapping periods for this quantile; non-overlapping VaR not computed.")
    sc = math.sqrt(h) * v1
    t = pd.DataFrame({"measure": [f"1-day VaR × √{h}", f"{h}-day VaR (overlapping)", f"{h}-day VaR (non-overlapping)"],
                      "VaR": [sc, vo, vn], "observations": [len(L), len(ov), len(nov)]})
    return Outcome({"n": len(L), "VaR_1d": v1, "VaR_scaled": sc, "VaR_overlapping": vo,
                    "VaR_non_overlapping": vn, "ratio_overlapping_to_scaled": vo / sc if sc else float("nan"),
                    "variance_ratio": float(ov.var(ddof=1) / (h * L.var(ddof=1))),
                    "lag1_autocorrelation": float(np.corrcoef(L[:-1], L[1:])[0, 1])},
                   {"Scaling comparison": t}, notes=notes, rows_used=len(L))


@register("var.stressed_period", "Stressed-period identification (window with maximum VaR/ES)", "Replication",
          _MT, params=(P("pnl", help="P&L of the current portfolio over a long history (full revaluation)"),
                       P("date", required=False), _CONF, _LOSSPOS,
                       P("window", "integer", default=250, help="Window length (250 = one year)"),
                       P("measure", "string", default="var", choices=("var", "es"))),
          description="""Rolls a `window`-observation window over the P&L history, computes HS VaR (empirical
quantile) and ES (empirical, Acerbi–Tasche) in each, and reports the window with the largest chosen measure
— the stressed period used for stressed VaR (CRR Art. 365) or the reduced-set stress period of FRTB ES.
Validates the bank's choice when the P&L is the current portfolio revalued over history.""",
          references=("Regulation (EU) No 575/2013 (CRR), Article 365 (stressed VaR)",
                      "EBA Guidelines on Stressed Value at Risk (EBA/GL/2012/2)",
                      "Basel Committee on Banking Supervision, MAR33 Internal models approach: capital requirements "
                      "calculation"))
def stressed_period(ctx: RunContext, pnl, date=None, confidence=0.99, loss_positive=False, window=250,
                    measure="var") -> Outcome:
    _check_conf(confidence)
    d, notes, dropped = load_series(ctx, {"pnl": pnl}, date, loss_positive, sign_keys=())
    notes += dropped_note(dropped)
    L = d["loss"].to_numpy()
    if window < 2 or len(L) < window or window * (1 - confidence) < 1:
        raise NotApplicable(f"{len(L)} observations cannot support windows of {window} at {confidence}.")
    W = np.lib.stride_tricks.sliding_window_view(L, window)
    v = np.quantile(W, confidence, axis=1, method="inverted_cdf")
    na = window * (1 - confidence)
    k = int(math.floor(na + 1e-12))
    srt = -np.sort(-W, axis=1)
    e = (srt[:, :k].sum(axis=1) + ((na - k) * srt[:, k] if k < window else 0)) / na
    lab = _label(d)
    t = pd.DataFrame({"window_start": lab.iloc[:len(v)].to_numpy(), "window_end": lab.iloc[window - 1:].to_numpy(),
                      "VaR": v, "ES": e})
    i = int((t["VaR"] if measure == "var" else t["ES"]).to_numpy().argmax())
    return Outcome({"windows": len(t), "stressed_start": t["window_start"].iloc[i],
                    "stressed_end": t["window_end"].iloc[i], "stressed_VaR": float(t["VaR"].iloc[i]),
                    "stressed_ES": float(t["ES"].iloc[i]), "latest_VaR": float(t["VaR"].iloc[-1]),
                    "latest_ES": float(t["ES"].iloc[-1])},
                   {"Rolling window VaR/ES": t}, notes=notes, rows_used=len(L))
