"""Counterparty credit risk: exposure profiles, CVA, collateral (MPOR), exposure back-testing,
wrong-way risk, netting aggregation and SA-CCR for an interest-rate netting set.

Simulated exposure input comes in one of two layouts:
  * wide: one row per simulation path, one column per time point (`time_columns`); times in years are
    parsed from the column names or given in `times`;
  * long: one row per (path, time) with columns `mtm`, `time` and `path`.
MTM is the netting-set (or trade) value from the bank's perspective; exposure = max(MTM, 0).
Time-weighted averages use right-point weights Δt_k = t_k − t_{k−1} (t_0 = 0), as in the Basel
definition of (Effective) EPE, truncated at the horizon (1 year by default).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, dropped_note, num, register
from ask.validation.t_market import berkowitz_lr, pit_uniformity_table

_MT = ("ccr",)
_BASEL_CCR = ("Basel Committee on Banking Supervision, CRE53 Internal models method for counterparty credit risk",
              "Regulation (EU) No 575/2013 (CRR), Articles 284 (IMM exposure value) and 272 (definitions)")

_LAYOUT = (P("time_columns", "columns", required=False,
             help="Wide layout: one column per time point (rows = paths)"),
           P("times", "list", required=False, help="Times in years for time_columns (default: parsed from names)"),
           P("mtm", required=False, help="Long layout: simulated MTM per path and time"),
           P("time", required=False, help="Long layout: time point in years"),
           P("path", required=False, help="Long layout: simulation path id"))


def _paths(ctx: RunContext, time_columns=None, times=None, mtm=None, time=None, path=None, extra=None):
    """Times (sorted) and matrices paths × times for `mtm` (and extra long-format value columns)."""
    notes: list[str] = []
    df = ctx.df
    if time_columns:
        if mtm or time:
            raise ValueError("Use either the wide layout (time_columns) or the long layout (mtm, time, path).")
        if times:
            if len(times) != len(time_columns):
                raise ValueError("`times` must have one entry per time column.")
            t = np.array([float(x) for x in times])
        else:
            try:
                t = np.array([float(str(c).strip().lower().rstrip("y")) for c in time_columns])
            except ValueError:
                raise ValueError("Cannot read times from the column names; pass `times` (years).") from None
        V = np.column_stack([num(df, c).to_numpy() for c in time_columns])
        o = np.argsort(t, kind="mergesort")
        t, V = t[o], V[:, o]
        mats = {"mtm": V}
    else:
        if not (mtm and time):
            raise ValueError("Give time_columns (wide) or mtm and time (long layout).")
        cols = {"mtm": mtm, **(extra or {})}
        d = pd.DataFrame({"time": num(df, time)}, index=df.index)
        d["path"] = df[path] if path else d.groupby("time").cumcount()
        if not path:
            notes.append("No path column: values are matched to paths by order within each time point.")
        for k, c in cols.items():
            d[k] = num(df, c)
        n0 = len(d)
        d = d.dropna()
        notes += dropped_note(n0 - len(d))
        if d.duplicated(["path", "time"]).any():
            raise ValueError("Duplicate (path, time) rows in the long layout.")
        mats = {k: d.pivot(index="path", columns="time", values=k).sort_index(axis=1) for k in cols}
        t = mats["mtm"].columns.to_numpy(float)
        mats = {k: m.to_numpy(float) for k, m in mats.items()}
    if np.unique(t).size != t.size:
        raise ValueError("Duplicate time points.")
    if (t < 0).any():
        raise ValueError("Times must be >= 0 (years from today).")
    miss = int(np.isnan(mats["mtm"]).sum())
    if miss:
        notes.append(f"{miss} missing path/time cells ignored in the statistics.")
    if mats["mtm"].shape[0] < 2:
        raise NotApplicable("Fewer than 2 simulation paths.")
    return t, mats, notes


def _weights(t: np.ndarray, horizon: float) -> np.ndarray:
    tp = np.concatenate([[0.0], t[:-1]])
    return np.clip(np.minimum(t, horizon) - np.minimum(tp, horizon), 0, None)


def epe_eepe(t: np.ndarray, ee: np.ndarray, horizon: float = 1.0) -> tuple[float, float, np.ndarray]:
    """EPE and Effective EPE over min(horizon, last time) with Basel right-point weights."""
    w = _weights(t, horizon)
    if w.sum() <= 0:
        raise NotApplicable("No time points within the horizon.")
    eff = np.maximum.accumulate(ee)
    return float(w @ ee / w.sum()), float(w @ eff / w.sum()), eff


@register("ccr.exposure_profile", "Exposure profile: EE, PFE, ENE, EPE, Effective EE/EPE, EAD", "Exposure", _MT,
          params=(*_LAYOUT,
                  P("quantile", "number", default=0.95, help="PFE quantile, e.g. 0.95 or 0.99"),
                  P("horizon", "number", default=1.0, help="EPE averaging horizon in years (Basel: 1)"),
                  P("alpha", "number", default=1.4, help="Alpha multiplier for EAD = alpha × EEPE")),
          description="""From simulated MTM V(t) per path: EE(t) = E[max(V, 0)], ENE(t) = E[min(V, 0)] (<= 0),
PFE_q(t) = q-quantile of max(V, 0), Effective EE(t_k) = max(Effective EE(t_{k−1}), EE(t_k)) (non-decreasing).
EPE = Σ EE_k Δt_k / Σ Δt_k and Effective EPE = Σ EffEE_k Δt_k / Σ Δt_k over the first min(horizon, last
time point) years (Basel definitions, right-point weights). EAD = alpha × Effective EPE (alpha = 1.4
supervisory default; own-estimate alpha needs an economic-capital model and is not computed here).""",
          references=_BASEL_CCR + ("Gregory, J. (2015), The xVA Challenge, 3rd ed., Wiley, ch. 11",))
def exposure_profile(ctx: RunContext, time_columns=None, times=None, mtm=None, time=None, path=None,
                     quantile=0.95, horizon=1.0, alpha=1.4) -> Outcome:
    if not 0 < quantile < 1 or horizon <= 0:
        raise ValueError("quantile must be in (0, 1) and horizon > 0")
    t, mats, notes = _paths(ctx, time_columns, times, mtm, time, path)
    V = mats["mtm"]
    E = np.where(np.isnan(V), np.nan, np.maximum(V, 0))
    ee, ene = np.nanmean(E, axis=0), np.nanmean(np.minimum(V, 0), axis=0)
    pfe = np.nanquantile(E, quantile, axis=0)
    epe, eepe, eff = epe_eepe(t, ee, horizon)
    prof = pd.DataFrame({"time": t, "EE": ee, "effective_EE": eff, f"PFE_{quantile:g}": pfe, "ENE": ene,
                         "mean_MTM": np.nanmean(V, axis=0), "weight_in_EPE": _weights(t, horizon),
                         "paths": (~np.isnan(V)).sum(axis=0)})
    if t[-1] < horizon:
        notes.append(f"Last time point {t[-1]:g}y is before the {horizon:g}y horizon: EPE averages up to {t[-1]:g}y "
                     f"(Basel: the shorter of one year and the netting set's maturity).")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=t, y=ee, name="EE")
    fig.add_scatter(x=t, y=eff, name="Effective EE", line=dict(dash="dash"))
    fig.add_scatter(x=t, y=pfe, name=f"PFE {quantile:.0%}")
    fig.add_scatter(x=t, y=ene, name="ENE")
    fig.update_layout(title="Exposure profile", xaxis_title="time (years)", yaxis_title="exposure")
    k = int(np.nanargmax(pfe))
    return Outcome({"paths": V.shape[0], "time_points": len(t), "EPE": epe, "effective_EPE": eepe,
                    "EAD_alpha_EEPE": alpha * eepe, "peak_EE": float(np.nanmax(ee)),
                    "peak_PFE": float(pfe[k]), "peak_PFE_time": float(t[k])},
                   {"Exposure profile": prof}, [fig], notes, rows_used=V.shape[0])


def _survival(t, hazard_rate, cds_spread, spread_tenors, spreads, lgd, who="counterparty"):
    """S(t) from a flat hazard rate, a flat CDS spread or a spread term structure (credit triangle)."""
    given = [x is not None for x in (hazard_rate, cds_spread)] + [bool(spreads)]
    if sum(given) != 1:
        raise ValueError(f"Give exactly one of hazard_rate, cds_spread or spreads (+ spread_tenors) for the {who}.")
    if hazard_rate is not None:
        return np.exp(-hazard_rate * t), f"flat hazard rate {hazard_rate:g}"
    if not 0 < lgd <= 1:
        raise ValueError("lgd must be in (0, 1]")
    if cds_spread is not None:
        return np.exp(-cds_spread * t / lgd), f"flat CDS spread {cds_spread:g}, λ = s/LGD"
    if not spread_tenors or len(spread_tenors) != len(spreads):
        raise ValueError("spread_tenors and spreads must have the same length.")
    ten, sp = np.array([float(x) for x in spread_tenors]), np.array([float(x) for x in spreads])
    o = np.argsort(ten)
    s_t = np.interp(t, ten[o], sp[o])
    return np.exp(-s_t * t / lgd), "CDS spread term structure, S(t) = exp(−s(t)·t/LGD)"


@register("ccr.cva", "Unilateral CVA (and DVA/BCVA) from an exposure profile", "CVA", _MT,
          params=(*_LAYOUT,
                  P("ee", required=False, help="Alternatively: an EE profile column (with `time`), one row per time"),
                  P("nee", required=False, help="Negative-exposure profile column E[max(−V,0)] for DVA (with `ee`)"),
                  P("lgd", "number", default=0.6, help="Counterparty loss given default (market LGD)"),
                  P("hazard_rate", "number", required=False), P("cds_spread", "number", required=False,
                                                                help="Flat CDS spread (decimal, 0.01 = 100bp)"),
                  P("spread_tenors", "list", required=False), P("spreads", "list", required=False),
                  P("discount_rate", "number", default=0.0, help="Flat continuously compounded discount rate"),
                  P("own_hazard_rate", "number", required=False), P("own_cds_spread", "number", required=False),
                  P("own_lgd", "number", default=0.6),
                  P("integration", "string", default="basel", choices=("basel", "right_point"))),
          description="""CVA = LGD Σ_i ΔPD_i × (EE_{i−1} D_{i−1} + EE_i D_i)/2 (Basel advanced-CVA formula; or
EE_i D_i at the right point), ΔPD_i = max(S(t_{i−1}) − S(t_i), 0), t_0 = 0 with EE_0 the exposure at time 0
(the first grid value when 0 is not on the grid). Survival S(t) from a flat hazard rate λ, or from CDS spreads
via the credit-triangle approximation λ ≈ s/LGD: S(t) = exp(−s(t)·t/LGD) with s(t) linearly interpolated
(flat outside the tenors), as in the Basel formula. D(t) = exp(−r t). With own credit inputs, DVA uses the
negative exposure profile and own survival; BCVA = CVA − DVA (ignores first-to-default and
exposure–default dependence, i.e. no wrong-way risk).""",
          references=("Basel Committee on Banking Supervision (2011), Basel III: A global regulatory framework "
                      "for more resilient banks and banking systems (rev.), CVA risk capital charge",
                      "Gregory, J. (2015), The xVA Challenge, 3rd ed., Wiley, ch. 14",
                      "Brigo, D., Morini, M. and Pallavicini, A. (2013), Counterparty Credit Risk, Collateral and "
                      "Funding, Wiley"))
def cva(ctx: RunContext, time_columns=None, times=None, mtm=None, time=None, path=None, ee=None, nee=None,
        lgd=0.6, hazard_rate=None, cds_spread=None, spread_tenors=None, spreads=None, discount_rate=0.0,
        own_hazard_rate=None, own_cds_spread=None, own_lgd=0.6, integration="basel") -> Outcome:
    if ee:
        if not time:
            raise ValueError("`ee` needs the `time` column.")
        d = pd.DataFrame({"t": num(ctx.df, time), "ee": num(ctx.df, ee)})
        if nee:
            d["nee"] = num(ctx.df, nee)
        n0 = len(d)
        d = d.dropna().sort_values("t")
        notes = dropped_note(n0 - len(d))
        if d["t"].duplicated().any():
            raise ValueError("Duplicate times in the EE profile.")
        t, eev = d["t"].to_numpy(), d["ee"].to_numpy()
        neev = d["nee"].to_numpy() if nee else None
        rows = len(d)
    else:
        t, mats, notes = _paths(ctx, time_columns, times, mtm, time, path)
        V = mats["mtm"]
        eev = np.nanmean(np.maximum(V, 0), axis=0)
        neev = np.nanmean(np.maximum(-V, 0), axis=0)
        rows = V.shape[0]
    if not len(t):
        raise NotApplicable("Empty exposure profile.")
    if t[0] > 0:
        t, eev = np.concatenate([[0.0], t]), np.concatenate([[eev[0]], eev])
        neev = np.concatenate([[neev[0]], neev]) if neev is not None else None
        notes.append("Time 0 not on the grid: EE at t = 0 set to the first grid value.")
    S, how = _survival(t, hazard_rate, cds_spread, spread_tenors, spreads, lgd)
    D = np.exp(-discount_rate * t)

    def charge(prof, surv, L):
        dpd = np.maximum(surv[:-1] - surv[1:], 0)
        x = prof * D
        e = 0.5 * (x[:-1] + x[1:]) if integration == "basel" else x[1:]
        return L * dpd * e, dpd

    contrib, dpd = charge(eev, S, lgd)
    out = pd.DataFrame({"time": t[1:], "EE": eev[1:], "discount_factor": D[1:], "survival": S[1:],
                        "marginal_PD": dpd, "CVA_contribution": contrib})
    summ = {"CVA": float(contrib.sum()), "PD_to_horizon": float(1 - S[-1]), "horizon": float(t[-1]),
            "time_points": len(t) - 1}
    notes.append(f"Counterparty survival from {how}; LGD = {lgd:g}.")
    if own_hazard_rate is not None or own_cds_spread is not None:
        if neev is None:
            raise ValueError("DVA needs the negative exposure profile (`nee` with `ee`, or simulated paths).")
        So, how_o = _survival(t, own_hazard_rate, own_cds_spread, None, None, own_lgd, "bank")
        dcontrib, _ = charge(neev, So, own_lgd)
        out["NEE"] = neev[1:]
        out["DVA_contribution"] = dcontrib
        summ |= {"DVA": float(dcontrib.sum()), "BCVA": float(contrib.sum() - dcontrib.sum())}
        notes.append(f"Own survival from {how_o}. BCVA = CVA − DVA ignores first-to-default effects.")
    return Outcome(summ, {"CVA by time bucket": out}, notes=notes, rows_used=rows)


@register("ccr.mpor_effect", "Collateral and margin-period-of-risk effect on exposure", "Exposure", _MT,
          params=(P("mtm"), P("time"), P("path"),
                  P("collateral", help="Collateral balance held at each path/time (positive = held by the bank)"),
                  P("mpor", "number", default=0.04, help="Margin period of risk in years (10 business days ≈ 0.04)"),
                  P("horizon", "number", default=1.0)),
          description="""Long layout per path and time. Compares EE profiles and (Effective) EPE of
(a) uncollateralised exposure max(V(t), 0); (b) exposure net of contemporaneous collateral max(V(t) − C(t), 0);
(c) exposure with collateral lagged by the MPOR, max(V(t) − C(t − MPOR), 0), where C(t − MPOR) is taken from the
same path at the latest grid time <= t − MPOR (0 before the first grid point). The gap between (b) and (c)
is the exposure that builds up during the close-out period. The grid must resolve the MPOR for (c) to be
accurate.""",
          references=_BASEL_CCR + ("Andersen, L., Pykhtin, M. and Sokol, A. (2017), Rethinking the margin period "
                                   "of risk, Journal of Credit Risk 13(1), 1–45",))
def mpor_effect(ctx: RunContext, mtm, time, path, collateral, mpor=0.04, horizon=1.0) -> Outcome:
    if mpor < 0:
        raise ValueError("mpor must be >= 0")
    t, mats, notes = _paths(ctx, None, None, mtm, time, path, extra={"col": collateral})
    V, C = mats["mtm"], mats["col"]
    j = np.searchsorted(t, t - mpor + 1e-12, side="right") - 1
    Cl = np.where(j[None, :] >= 0, C[:, np.clip(j, 0, None)], 0.0)
    gap = (t - mpor) - np.where(j >= 0, t[np.clip(j, 0, None)], 0.0)
    if (gap[j >= 0] > 1e-9).any() and mpor > 0:
        notes.append("Some t − MPOR fall between grid points: the collateral of the previous grid point is used "
                     "(a longer effective lag).")
    prof = {"uncollateralised": np.nanmean(np.maximum(V, 0), axis=0),
            "collateral_no_lag": np.nanmean(np.maximum(V - C, 0), axis=0),
            "collateral_MPOR_lag": np.nanmean(np.maximum(V - Cl, 0), axis=0)}
    out = pd.DataFrame({"time": t, **{f"EE_{k}": v for k, v in prof.items()}})
    summ = {"paths": V.shape[0], "mpor_years": mpor}
    rows = []
    for k, v in prof.items():
        epe, eepe, _ = epe_eepe(t, v, horizon)
        rows.append({"exposure": k, "EPE": epe, "effective_EPE": eepe, "peak_EE": float(np.nanmax(v))})
        summ[f"EEPE_{k}"] = eepe
    if summ["EEPE_uncollateralised"] > 0:
        summ["EEPE_ratio_MPOR_to_uncollateralised"] = summ["EEPE_collateral_MPOR_lag"] / summ["EEPE_uncollateralised"]
    return Outcome(summ, {"EE profiles": out, "EPE comparison": pd.DataFrame(rows)}, notes=notes,
                   rows_used=V.shape[0])


@register("ccr.exposure_backtest", "Exposure back-testing on PIT values (KS, AD, CvM, Berkowitz)", "Back-testing",
          _MT, params=(P("pit", required=False, help="PIT of realised MTM under the forecast distribution"),
                       P("actual", required=False, help="Realised MTM (PITs computed from `other`)"),
                       P("id", required=False, help="Key column linking each realised value to its forecasts"),
                       P("other", "table", required=False, help="Forecast table: key column + simulated values"),
                       P("forecast_column", "string", required=False, help="Simulated-value column in `other`"),
                       P("n_sims", "integer", default=2000), P("bins", "integer", default=10)),
          description="""Probability integral transforms u = F_forecast(realised MTM) — given, or computed from the
simulated forecast values in `other` matched on `id` (u = [#{sim < x} + ½#{sim = x}] / N). H0: u ~ U(0,1)
i.i.d. (forecast distributions correct). Kolmogorov–Smirnov, Anderson–Darling (Monte Carlo p-value),
Cramér–von Mises, chi-square, and the Berkowitz LR_3 test on Φ^(−1)(u). PITs from overlapping forecast
horizons or many trades of one counterparty are dependent, which inflates rejection rates; aggregate
statistics then need a simulated null (not done here).""",
          references=("Kenyon, C. and Stamm, R. (2012), Discounting, LIBOR, CVA and Funding, Palgrave, ch. on "
                      "back-testing",
                      "Ruiz, I. (2014), Backtesting counterparty risk: how good is your model?, Journal of Credit "
                      "Risk 10(1)",
                      "Berkowitz, J. (2001), Testing density forecasts, JBES 19(4), 465–474",
                      "Basel Committee on Banking Supervision (2010), Sound practices for backtesting counterparty "
                      "credit risk models"))
def exposure_backtest(ctx: RunContext, pit=None, actual=None, id=None, other=None, forecast_column=None,
                      n_sims=2000, bins=10) -> Outcome:
    notes: list[str] = []
    if (pit is None) == (actual is None):
        raise ValueError("Give exactly one of `pit` or `actual` (with id, other, forecast_column).")
    if pit:
        u = num(ctx.df, pit)
        notes += dropped_note(int(u.isna().sum()))
        u = u.dropna().to_numpy()
    else:
        if not (id and other and forecast_column):
            raise ValueError("`actual` needs `id`, `other` and `forecast_column`.")
        F = ctx.tables[other]
        if id not in F.columns or forecast_column not in F.columns:
            raise ValueError(f"`other` must contain '{id}' and '{forecast_column}'.")
        sims = pd.DataFrame({"k": F[id], "x": pd.to_numeric(F[forecast_column], errors="coerce")}).dropna()
        groups = {k: np.sort(g.to_numpy()) for k, g in sims.groupby("k")["x"]}
        x = num(ctx.df, actual)
        us, miss = [], 0
        for k, v in zip(ctx.df[id], x):
            s = groups.get(k)
            if s is None or not len(s) or pd.isna(v):
                miss += 1
                continue
            lo, hi = np.searchsorted(s, v, "left"), np.searchsorted(s, v, "right")
            us.append((lo + 0.5 * (hi - lo)) / len(s))
        u = np.array(us)
        notes += dropped_note(miss, "realised values without forecasts or with missing values")
    if ((u < 0) | (u > 1)).any():
        raise ValueError("PIT values must be in [0, 1].")
    if len(u) < 10:
        raise NotApplicable(f"{len(u)} PIT values; at least 10 needed.")
    u = np.clip(u, 1e-6, 1 - 1e-6)
    t = pit_uniformity_table(u, ctx.rng(31), n_sims, bins)
    b = berkowitz_lr(stats.norm.ppf(u))
    t = pd.concat([t, pd.DataFrame([{"test": "Berkowitz LR_3", "statistic": b["LR_3"],
                                     "p_value": b["p_value_LR_3"], "p_value_method": "chi-square(3)"}])],
                  ignore_index=True)
    return Outcome({"n": len(u), **{f"{r.test} p_value": r.p_value for r in t.itertuples()},
                    "mean_pit": float(u.mean())},
                   {"PIT tests": t}, notes=notes, rows_used=len(u))


@register("ccr.wrong_way_risk", "Wrong-way risk: dependence between exposure and counterparty credit quality",
          "Wrong-way risk", _MT,
          params=(P("mtm", help="Simulated or realised MTM / exposure"),
                  P("credit", help="Counterparty credit-quality proxy, higher = worse (hazard, spread, PD)"),
                  P("time", required=False, help="Time point; correlations are also given per time point"),
                  P("use_exposure", "boolean", default=True, help="Correlate max(MTM,0) instead of MTM"),
                  P("tail", "number", default=0.9, help="Credit quantile defining the 'distressed' scenarios")),
          description="""Pearson, Spearman and Kendall correlations (with p-values, H0: no association) between
exposure and a credit-quality proxy, pooled and by time point. Positive dependence = wrong-way risk.
Also the conditional EE ratio E[exposure | credit >= its `tail` quantile] / E[exposure] per time point
(> 1 indicates exposure is higher when the counterparty is distressed). Pooled correlations mix horizons;
read the per-time table.""",
          references=("Basel Committee on Banking Supervision, CRE53 (specific and general wrong-way risk)",
                      "Hull, J. and White, A. (2012), CVA and wrong-way risk, Financial Analysts Journal 68(5), 58–69"))
def wrong_way_risk(ctx: RunContext, mtm, credit, time=None, use_exposure=True, tail=0.9) -> Outcome:
    d = pd.DataFrame({"x": num(ctx.df, mtm), "c": num(ctx.df, credit)})
    if time:
        d["t"] = num(ctx.df, time)
    n0 = len(d)
    d = d.dropna()
    notes = dropped_note(n0 - len(d))
    if use_exposure:
        d["x"] = d["x"].clip(lower=0)
    if len(d) < 5 or d["x"].std() == 0 or d["c"].std() == 0:
        raise NotApplicable("Need at least 5 observations with variation in both exposure and credit proxy.")

    def corr(g):
        r = {"n": len(g)}
        if len(g) < 3 or g["x"].std() == 0 or g["c"].std() == 0:
            return r | {"pearson": np.nan, "pearson_p": np.nan, "spearman": np.nan, "spearman_p": np.nan,
                        "kendall": np.nan, "kendall_p": np.nan, "conditional_EE_ratio": np.nan}
        pe, sp, kt = stats.pearsonr(g["x"], g["c"]), stats.spearmanr(g["x"], g["c"]), stats.kendalltau(g["x"], g["c"])
        hi = g["c"] >= g["c"].quantile(tail)
        ee = g["x"].mean()
        return r | {"pearson": float(pe.statistic), "pearson_p": float(pe.pvalue), "spearman": float(sp.statistic),
                    "spearman_p": float(sp.pvalue), "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
                    "conditional_EE_ratio": float(g.loc[hi, "x"].mean() / ee) if ee > 0 else np.nan}

    pooled = corr(d)
    tabs = {}
    if time:
        tabs["By time point"] = pd.DataFrame([{"time": k} | corr(g) for k, g in d.groupby("t", sort=True)])
        notes.append("Pooled correlations mix time points with different exposure scales.")
    return Outcome({k: v for k, v in pooled.items()}, tabs, notes=notes, rows_used=len(d))


@register("ccr.netting_check", "Netting-set aggregation check and netting benefit", "Exposure", _MT,
          params=(P("mtm", help="Trade-level MTM"), P("netting_set", help="Netting-set identifier column"),
                  P("reported", required=False, help="Reported netting-set MTM (repeated on each trade row)"),
                  P("path", required=False), P("time", required=False),
                  P("tolerance", "number", default=1e-6, help="Absolute difference counted as a break")),
          description="""Per netting set (and per path/time when given): Σ trade MTM vs the reported netting-set MTM
(difference; sets whose reported value is not constant across their trades are flagged). Exposure with and
without netting: net = max(Σ V_i, 0), gross = Σ max(V_i, 0); net-to-gross ratio NGR = Σ net / Σ gross
(1 − NGR = netting benefit).""",
          references=_BASEL_CCR + ("Basel Committee on Banking Supervision (1995), Basel Capital Accord: treatment of "
                                   "potential exposure for off-balance-sheet items (net-to-gross ratio)",))
def netting_check(ctx: RunContext, mtm, netting_set, reported=None, path=None, time=None,
                  tolerance=1e-6) -> Outcome:
    keys = ["set"] + (["path"] if path else []) + (["time"] if time else [])
    d = pd.DataFrame({"set": ctx.df[netting_set], "v": num(ctx.df, mtm)})
    if path:
        d["path"] = ctx.df[path]
    if time:
        d["time"] = num(ctx.df, time)
    if reported:
        d["rep"] = num(ctx.df, reported)
    n0 = len(d)
    d = d.dropna()
    notes = dropped_note(n0 - len(d))
    if d.empty:
        raise NotApplicable("No complete rows.")
    agg = {"trades": ("v", "size"), "sum_trade_mtm": ("v", "sum"),
           "gross_exposure": ("v", lambda s: s.clip(lower=0).sum())}
    if reported:
        agg |= {"reported_mtm": ("rep", "first"), "reported_values": ("rep", "nunique")}
    g = d.groupby(keys, sort=True).agg(**agg).reset_index()
    g["net_exposure"] = g["sum_trade_mtm"].clip(lower=0)
    summ = {"groups": len(g), "netting_sets": int(d["set"].nunique()), "trades_rows": len(d),
            "net_exposure": float(g["net_exposure"].sum()), "gross_exposure": float(g["gross_exposure"].sum())}
    summ["net_to_gross_ratio"] = summ["net_exposure"] / summ["gross_exposure"] if summ["gross_exposure"] else np.nan
    if reported:
        g["difference"] = g["reported_mtm"] - g["sum_trade_mtm"]
        summ |= {"max_abs_difference": float(g["difference"].abs().max()),
                 "breaks": int((g["difference"].abs() > tolerance).sum()),
                 "inconsistent_reported": int((g["reported_values"] > 1).sum())}
    return Outcome(summ, {"Netting-set aggregation": g}, notes=notes, rows_used=len(d))


@register("ccr.sa_ccr_ir", "SA-CCR exposure for an unmargined interest-rate netting set", "Exposure", _MT,
          params=(P("notional", help="Trade notional (positive)"), P("start", help="Start date S in years (0 if started)"),
                  P("end", help="End date E of the referenced period in years"),
                  P("delta", help="Supervisory delta: +1 long (pay fixed), −1 short (receive fixed), or option delta"),
                  P("mtm", help="Trade MTM"),
                  P("maturity", required=False, help="Maturity M in years (default = end)"),
                  P("currency", required=False, help="Hedging set (currency); default one hedging set"),
                  P("collateral", "number", default=0.0, help="Net haircut collateral held C"),
                  P("alpha", "number", default=1.4)),
          description="""Basel SA-CCR for an unmargined netting set of interest-rate trades.
Supervisory duration SD = [exp(−0.05 S) − exp(−0.05 E)]/0.05 (S floored at 0); adjusted notional d = N·SD;
maturity factor MF = √(min(max(M, 10/250), 1)); time buckets by E: < 1y, 1–5y, > 5y;
D_k = Σ δ·d·MF per bucket; effective notional EN = √(D1² + D2² + D3² + 1.4 D1D2 + 1.4 D2D3 + 0.6 D1D3)
per currency; AddOn = Σ 0.5%·EN; multiplier = min(1, 0.05 + 0.95 exp((V − C)/(2·0.95·AddOn)));
RC = max(V − C, 0); PFE = multiplier × AddOn; EAD = alpha(RC + PFE). Option deltas must be supplied
(supervisory delta with σ = 50%); margined netting sets are out of scope.""",
          references=("Basel Committee on Banking Supervision (2014), The standardised approach for measuring "
                      "counterparty credit risk exposures (BCBS 279)",
                      "Regulation (EU) No 575/2013 as amended by Regulation (EU) 2019/876 (CRR2), Articles 274–280a"))
def sa_ccr_ir(ctx: RunContext, notional, start, end, delta, mtm, maturity=None, currency=None, collateral=0.0,
              alpha=1.4) -> Outcome:
    d = pd.DataFrame({"N": num(ctx.df, notional), "S": num(ctx.df, start), "E": num(ctx.df, end),
                      "delta": num(ctx.df, delta), "V": num(ctx.df, mtm)})
    d["M"] = num(ctx.df, maturity) if maturity else d["E"]
    d["ccy"] = ctx.df[currency].astype(str) if currency else "all"
    n0 = len(d)
    d = d.dropna()
    notes = dropped_note(n0 - len(d))
    if d.empty:
        raise NotApplicable("No complete trades.")
    if (d["N"] < 0).any() or (d["E"] < d["S"].clip(lower=0)).any():
        raise ValueError("Notionals must be >= 0 and end >= start.")
    S = d["S"].clip(lower=0)
    d["SD"] = (np.exp(-0.05 * S) - np.exp(-0.05 * d["E"])) / 0.05
    d["adjusted_notional"] = d["N"] * d["SD"]
    d["MF"] = np.sqrt(np.minimum(np.maximum(d["M"], 10 / 250), 1.0))
    d["bucket"] = np.where(d["E"] < 1, 1, np.where(d["E"] <= 5, 2, 3))
    d["effective_contribution"] = d["delta"] * d["adjusted_notional"] * d["MF"]
    rows = []
    for ccy, g in d.groupby("ccy", sort=True):
        D = [float(g.loc[g["bucket"] == k, "effective_contribution"].sum()) for k in (1, 2, 3)]
        en = math.sqrt(max(D[0] ** 2 + D[1] ** 2 + D[2] ** 2 + 1.4 * D[0] * D[1] + 1.4 * D[1] * D[2]
                           + 0.6 * D[0] * D[2], 0.0))
        rows.append({"hedging_set": ccy, "D1": D[0], "D2": D[1], "D3": D[2], "effective_notional": en,
                     "add_on": 0.005 * en})
    hs = pd.DataFrame(rows)
    addon = float(hs["add_on"].sum())
    V = float(d["V"].sum())
    mult = min(1.0, 0.05 + 0.95 * math.exp((V - collateral) / (2 * 0.95 * addon))) if addon > 0 else 1.0
    rc = max(V - collateral, 0.0)
    pfe = mult * addon
    trades = d[["N", "S", "E", "M", "delta", "SD", "adjusted_notional", "MF", "bucket", "effective_contribution"]]
    return Outcome({"trades": len(d), "V": V, "RC": rc, "AddOn_IR": addon, "multiplier": mult, "PFE": pfe,
                    "EAD": alpha * (rc + pfe)},
                   {"Trades": trades.reset_index(drop=True), "Hedging sets": hs},
                   notes=notes + ["Unmargined netting set; IR asset class only."], rows_used=len(d))
