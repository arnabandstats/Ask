"""IFRS 9 ECL validation: ECL recomputation, scenarios, lifetime PD, staging, coverage and PIT calibration.

Term-structure conventions:
  - Periods are numbered 1..T; each period lasts `period_length` years (1 = annual, 1/12 = monthly).
  - A PD term structure can be given as unconditional MARGINAL PDs (probability of defaulting in
    period t seen from today), CUMULATIVE PDs (default by end of period t) or CONDITIONAL PDs / hazard
    rates (default in t given survival to t-1); all are converted to marginal PDs.
  - ECL = Σ_t marginal PD_t · LGD_t · EAD_t · (1 + EIR)^(-time_t), the 12-month ECL uses the periods
    covering the first year. Stage 3 (credit-impaired) is recomputed as LGD_1 · EAD_1 (PD = 1, no
    further discounting), which is stated in the notes.
The Kaplan–Meier helpers here are also used by the LGD workout-length test.
"""
from __future__ import annotations

import re

import numpy as np
import pandas
import pandas as pd
from scipy import stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, dropped_note, num, register
from ask.validation.t_lgd import flag, sorted_levels

_IFRS9 = ("ifrs9",)


# ── survival helpers ───────────────────────────────────────────────────────

def km_curve(t, e) -> pd.DataFrame:
    """Kaplan–Meier survival at each event time, with the Greenwood sum Σ d/(n(n-d))."""
    t, e = np.asarray(t, float), np.asarray(e, float)
    times = np.unique(t[e == 1])
    st = np.sort(t)
    at_risk = len(t) - np.searchsorted(st, times, side="left")
    ev_sorted = np.sort(t[e == 1])
    d = np.searchsorted(ev_sorted, times, side="right") - np.searchsorted(ev_sorted, times, side="left")
    surv = np.cumprod(1 - d / at_risk)
    with np.errstate(divide="ignore"):
        gw = np.cumsum(np.where(at_risk > d, d / (at_risk * (at_risk - d)), np.inf))
    return pd.DataFrame({"time": times, "at_risk": at_risk, "events": d, "survival": surv, "greenwood": gw})


def km_at(curve: pd.DataFrame, h: float) -> tuple[float, float]:
    """S(h) and its Greenwood standard error."""
    m = curve["time"] <= h
    if not m.any():
        return 1.0, 0.0
    r = curve[m].iloc[-1]
    return float(r["survival"]), float(r["survival"] * np.sqrt(r["greenwood"]))


def km_quantile(curve: pd.DataFrame, q: float) -> float:
    """Smallest time with S(t) <= 1 - q (NaN if the curve never gets there)."""
    m = curve["survival"] <= 1 - q + 1e-12
    return float(curve.loc[m, "time"].iloc[0]) if m.any() else float("nan")


# ── term-structure helpers ─────────────────────────────────────────────────

def to_marginal(m: np.ndarray, kind: str) -> np.ndarray:
    m = np.asarray(m, float)
    if kind == "marginal":
        return m
    if kind == "cumulative":
        return np.diff(np.c_[np.zeros(len(m)), m], axis=1)
    if kind == "conditional":
        surv = np.cumprod(np.c_[np.ones(len(m)), 1 - m[:, :-1]], axis=1)
        return m * surv
    raise ValueError(f"Unknown PD type {kind}")


def _stage(s: pd.Series, label: str) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) if float(v) in (1.0, 2.0, 3.0) else np.nan
        mm = re.search(r"[123]", str(v))
        return float(mm.group()) if mm else np.nan
    out = s.map(one)
    bad = int((s.notna() & out.isna()).sum())
    if bad:
        raise ValueError(f"{bad} values of '{label}' are not stages 1/2/3 (e.g. {s[s.notna() & out.isna()].iloc[0]!r}).")
    return out


def _matrix(ctx: RunContext, cols, single, label: str, T: int, index) -> np.ndarray:
    if cols:
        if len(cols) != T:
            raise ValueError(f"{label}: {len(cols)} columns given but the PD term structure has {T} periods.")
        return np.column_stack([num(ctx.df, c).loc[index].to_numpy() for c in cols])
    if single:
        return np.repeat(num(ctx.df, single).loc[index].to_numpy()[:, None], T, axis=1)
    raise ValueError(f"Give the {label} as a column or a list of per-period columns (or in the term-structure table).")


# ── ECL recomputation ──────────────────────────────────────────────────────

@register("ifrs9.ecl_recompute", "ECL recomputation (12-month / lifetime) and reconciliation", "ECL measurement",
          _IFRS9,
          params=(P("id"), P("ecl", help="Reported ECL per account"),
                  P("stage", required=False, help="Reported stage (1/2/3); selects 12-month vs lifetime ECL"),
                  P("horizon", "string", default="stage", choices=("stage", "12m", "lifetime"),
                    help="Which recomputed ECL to compare: per stage, or 12m / lifetime for all accounts"),
                  P("pd_columns", "columns", required=False, help="Per-period PD columns, period 1..T (wide format)"),
                  P("pd_type", "string", default="marginal", choices=("marginal", "cumulative", "conditional")),
                  P("lgd", required=False, help="LGD (one column for all periods)"),
                  P("lgd_columns", "columns", required=False, help="Per-period LGD columns"),
                  P("ead", required=False, help="EAD (one column for all periods)"),
                  P("ead_columns", "columns", required=False, help="Per-period EAD columns"),
                  P("eir", required=False, help="Effective interest rate per account (annual)"),
                  P("eir_rate", "number", required=False, help="One EIR for all accounts (annual)"),
                  P("term_structure", "table", required=False,
                    help="Long table id × period with PD (and optionally LGD, EAD) instead of wide columns"),
                  P("ts_id", "string", required=False, help="Id column in term_structure (default: same as id)"),
                  P("ts_period", "string", default="period", help="Period index column (1..T) in term_structure"),
                  P("ts_pd", "string", default="pd"), P("ts_lgd", "string", required=False),
                  P("ts_ead", "string", required=False),
                  P("period_length", "number", default=1.0, help="Years per period (1 annual, 0.25 quarterly ...)"),
                  P("discount_timing", "string", default="end", choices=("end", "mid")),
                  P("tolerance", "number", default=0.01, help="Absolute difference counted as a mismatch"),
                  P("top_n", "integer", default=20)),
          description="""Independent recomputation of ECL per account from its components:
ECL_lifetime = Σ_{t=1..T} mPD_t · LGD_t · EAD_t · (1 + EIR)^(−τ_t), ECL_12m = the same sum over the periods in
the first year, with mPD the unconditional marginal PD (converted from cumulative or conditional input),
τ_t = t·period_length (end-of-period) or (t − ½)·period_length (mid-period). Stage 1 → 12-month ECL,
stage 2 → lifetime ECL, stage 3 → LGD_1 · EAD_1 (PD = 1). Missing PDs beyond maturity count as 0.
Compared with the reported ECL: per-account differences (recomputed − reported), the largest differences,
the number beyond `tolerance`, and an aggregate reconciliation by stage. Term structures come either as wide
per-period columns or as a long table (`term_structure`).""",
          references=("IFRS 9 Financial Instruments, section 5.5 and B5.5.28–B5.5.48 (measurement of ECL)",
                      "EBA/GL/2017/06, Guidelines on credit institutions' credit risk management practices and "
                      "accounting for expected credit losses"))
def ecl_recompute(ctx: RunContext, id, ecl, stage=None, horizon="stage", pd_columns=None, pd_type="marginal",
                  lgd=None, lgd_columns=None, ead=None, ead_columns=None, eir=None, eir_rate=None,
                  term_structure=None, ts_id=None, ts_period="period", ts_pd="pd", ts_lgd=None, ts_ead=None,
                  period_length=1.0, discount_timing="end", tolerance=0.01, top_n=20) -> Outcome:
    df = ctx.df
    notes = []
    if not 0 < period_length <= 1:
        raise ValueError("period_length must be in (0, 1] years.")
    if horizon == "stage" and not stage:
        raise ValueError("horizon='stage' needs the `stage` column (or choose horizon '12m' / 'lifetime').")
    if df[id].duplicated().any():
        raise ValueError(f"'{id}' is not unique.")
    base = pd.DataFrame({"id": df[id], "reported": num(df, ecl)}, index=df.index)
    if stage:
        base["stage"] = _stage(df[stage], stage)
    if eir:
        base["eir"] = num(df, eir)
    elif eir_rate is not None:
        base["eir"] = float(eir_rate)
    else:
        base["eir"] = 0.0
        notes.append("No EIR given: cash shortfalls are not discounted.")
    n0 = len(base)
    base = base.dropna()
    notes += dropped_note(n0 - len(base), "accounts with missing id / reported ECL / stage / EIR")
    idx = base.index
    if term_structure:
        ts = ctx.tables[term_structure]
        tid = ts_id or id
        for c in [tid, ts_period, ts_pd] + [c for c in (ts_lgd, ts_ead) if c]:
            if c not in ts.columns:
                raise ValueError(f"Column '{c}' not in term_structure. Columns: {', '.join(map(str, ts.columns))}")
        t = pd.DataFrame({"key": ts[tid].astype(str), "per": pd.to_numeric(ts[ts_period], errors="coerce")})
        if t["per"].isna().any() or (t["per"] < 1).any() or (t["per"] % 1 != 0).any():
            raise ValueError(f"'{ts_period}' must hold integer periods 1..T.")
        if t.duplicated().any():
            raise ValueError("term_structure has duplicate (id, period) rows.")
        T = int(t["per"].max())
        keys = base["id"].astype(str)

        def wide(col):
            w = t.assign(v=pd.to_numeric(ts[col], errors="coerce").to_numpy()).pivot(
                index="key", columns="per", values="v").reindex(columns=range(1, T + 1))
            return w.reindex(keys).to_numpy()
        pdm = wide(ts_pd)
        missing_ts = int(np.isnan(pdm).all(axis=1).sum())
        lgdm = wide(ts_lgd) if ts_lgd else _matrix(ctx, lgd_columns, lgd, "LGD", T, idx)
        eadm = wide(ts_ead) if ts_ead else _matrix(ctx, ead_columns, ead, "EAD", T, idx)
    else:
        if not pd_columns:
            raise ValueError("Give pd_columns (wide format) or term_structure (long table).")
        T = len(pd_columns)
        pdm = np.column_stack([num(df, c).loc[idx].to_numpy() for c in pd_columns])
        missing_ts = int(np.isnan(pdm).all(axis=1).sum())
        lgdm = _matrix(ctx, lgd_columns, lgd, "LGD", T, idx)
        eadm = _matrix(ctx, ead_columns, ead, "EAD", T, idx)
    nan_pd = int((np.isnan(pdm).any(axis=1) & ~np.isnan(pdm).all(axis=1)).sum())
    if pd_type == "cumulative":      # carry the last cumulative PD forward past maturity
        pdm = pd.DataFrame(pdm).ffill(axis=1).fillna(0).to_numpy()
    else:
        pdm = np.nan_to_num(pdm, nan=0.0)
    if nan_pd:
        notes.append(f"{nan_pd} accounts have missing PDs in some periods (treated as beyond maturity: no default).")
    mpd = to_marginal(pdm, pd_type)
    if (mpd < -1e-12).any():
        notes.append(f"{int((mpd < -1e-12).any(axis=1).sum())} accounts have negative marginal PDs "
                     "(non-monotone cumulative PD).")
    need = mpd != 0
    gaps = (np.isnan(lgdm) | np.isnan(eadm)) & need
    incomplete = gaps.any(axis=1)
    if missing_ts:
        notes.append(f"{missing_ts} accounts have no PD term structure (ECL recomputed as 0 for stages 1-2).")
    lgdm0, eadm0 = np.nan_to_num(lgdm), np.nan_to_num(eadm)
    k12 = max(1, int(round(1 / period_length)))
    tau = (np.arange(1, T + 1) - (0.5 if discount_timing == "mid" else 0.0)) * period_length
    dfac = (1 + base["eir"].to_numpy()[:, None]) ** (-tau[None, :])
    contrib = mpd * lgdm0 * eadm0 * dfac
    e12, elt = contrib[:, :k12].sum(axis=1), contrib.sum(axis=1)
    e12[incomplete], elt[incomplete] = np.nan, np.nan
    out = pd.DataFrame({"id": base["id"].to_numpy(), "ecl_12m": e12, "ecl_lifetime": elt})
    if stage:
        st = base["stage"].to_numpy()
        out.insert(1, "stage", st.astype(int))
        e3 = lgdm[:, 0] * eadm[:, 0]
    if horizon == "stage":
        rec = np.where(st == 1, e12, np.where(st == 2, elt, e3))
        if (st == 3).any():
            notes.append("Stage 3 recomputed as LGD_1 × EAD_1 (PD = 1, no further discounting).")
    else:
        rec = e12 if horizon == "12m" else elt
    out["ecl_recomputed"] = rec
    out["ecl_reported"] = base["reported"].to_numpy()
    out["difference"] = out["ecl_recomputed"] - out["ecl_reported"]
    out["abs_difference"] = out["difference"].abs()
    out["relative_difference"] = out["difference"] / out["ecl_reported"].where(out["ecl_reported"] != 0)
    bad = out["ecl_recomputed"].isna()
    if bad.any():
        notes.append(f"{int(bad.sum())} accounts with missing LGD/EAD where the recomputation needs them "
                     "excluded from the reconciliation.")
    ok = out[~bad]
    if ok.empty:
        raise NotApplicable("No account could be recomputed.")
    rec_rows = []
    groups = [(g, ok[ok["stage"] == g]) for g in sorted(ok["stage"].unique())] if stage else []
    for g, s in groups + [("ALL", ok)]:
        rep, rcp = float(s["ecl_reported"].sum()), float(s["ecl_recomputed"].sum())
        rec_rows.append({"stage": str(g), "n": len(s), "reported_ecl": rep, "recomputed_ecl": rcp,
                         "difference": rcp - rep, "relative_difference": (rcp - rep) / rep if rep else np.nan,
                         "n_mismatch": int((s["abs_difference"] > tolerance).sum())})
    recon = pd.DataFrame(rec_rows)
    a = recon.iloc[-1]
    top = ok.sort_values(["abs_difference", "id"], ascending=[False, True], kind="mergesort").head(top_n)
    notes.append(f"Periods: {T} of {period_length:g} years; 12-month ECL uses the first {k12} period(s); "
                 f"discounting at {discount_timing} of period; PD input type '{pd_type}'.")
    return Outcome({"total_reported": a["reported_ecl"], "total_recomputed": a["recomputed_ecl"],
                    "difference": a["difference"], "relative_difference": a["relative_difference"],
                    "n_accounts": int(a["n"]), "n_mismatch": int(a["n_mismatch"]),
                    "max_abs_difference": float(ok["abs_difference"].max())},
                   {"Reconciliation": recon, "Largest differences": top.reset_index(drop=True),
                    "ECL per account": out},
                   notes=notes, rows_used=len(ok))


# ── scenarios ──────────────────────────────────────────────────────────────

def _scenarios(ctx: RunContext, id, scenario, ecl, weights) -> tuple[pd.DataFrame, dict, list[str]]:
    df = ctx.df
    d = pd.DataFrame({"id": df[id], "scen": df[scenario].astype(str).str.strip(), "ecl": num(df, ecl)})
    n0 = len(d)
    d = d.dropna()
    notes = dropped_note(n0 - len(d))
    if d.duplicated(["id", "scen"]).any():
        raise ValueError("Duplicate (id, scenario) rows.")
    w = {str(k).strip(): float(v) for k, v in weights.items()}
    if any(v < 0 for v in w.values()) or abs(sum(w.values()) - 1) > 1e-6:
        raise ValueError(f"Scenario weights must be non-negative and sum to 1 (sum = {sum(w.values()):.6g}).")
    have = set(d["scen"])
    if set(w) - have:
        raise ValueError(f"Scenario(s) {sorted(set(w) - have)} not in '{scenario}' (values: {sorted(have)}).")
    if have - set(w):
        raise ValueError(f"Scenario(s) {sorted(have - set(w))} have no weight; give a weight (0 to ignore).")
    wide = d.pivot(index="id", columns="scen", values="ecl")[list(w)]
    incomplete = wide.isna().any(axis=1)
    if incomplete.any():
        notes.append(f"{int(incomplete.sum())} accounts missing at least one scenario excluded.")
    wide = wide[~incomplete]
    wide = wide.loc[sorted(wide.index, key=lambda v: (str(type(v)), v))]
    if wide.empty:
        raise NotApplicable("No account has ECL for every scenario.")
    return wide, w, notes


@register("ifrs9.scenario_weighted_ecl", "Probability-weighted ECL over macro scenarios vs reported",
          "ECL measurement", _IFRS9,
          params=(P("id"), P("scenario", help="Scenario name column (long format: one row per account × scenario)"),
                  P("ecl", help="ECL of the account under the scenario"),
                  P("weights", "dict", help='Scenario weights, e.g. {"base": 0.5, "up": 0.2, "down": 0.3}'),
                  P("reported", required=False, help="Reported probability-weighted ECL per account (repeated per row)"),
                  P("base_scenario", "string", required=False, help="Name of the baseline scenario"),
                  P("tolerance", "number", default=0.01)),
          description="""Recomputes the probability-weighted ECL per account, ECL_w = Σ_s w_s · ECL_s, and reconciles it
with the reported ECL (per-account and total differences, number beyond `tolerance`). Shows the total ECL
under each scenario and, with `base_scenario`, the non-linearity ratio ECL_w / ECL_base (IFRS 9 requires an
unbiased probability-weighted amount, not the single most-likely scenario). Weights must sum to 1.""",
          references=("IFRS 9 Financial Instruments, paragraphs 5.5.17–5.5.18 and B5.5.41–B5.5.43",
                      "EBA/GL/2017/06, Guidelines on credit institutions' credit risk management practices and "
                      "accounting for expected credit losses"))
def scenario_weighted_ecl(ctx: RunContext, id, scenario, ecl, weights, reported=None, base_scenario=None,
                          tolerance=0.01) -> Outcome:
    wide, w, notes = _scenarios(ctx, id, scenario, ecl, weights)
    out = wide.copy()
    out["ecl_weighted"] = wide.to_numpy() @ np.array([w[s] for s in wide.columns])
    summary = {"total_weighted": float(out["ecl_weighted"].sum()), "n_accounts": len(out)}
    if reported:
        rp = pd.DataFrame({"id": ctx.df[id], "r": num(ctx.df, reported)}).dropna()
        if (rp.groupby("id")["r"].nunique() > 1).any():
            raise ValueError(f"'{reported}' varies within an account; it must be the account's weighted ECL.")
        r = rp.groupby("id")["r"].first()
        out["ecl_reported"] = r.reindex(out.index)
        out["difference"] = out["ecl_weighted"] - out["ecl_reported"]
        has = out["ecl_reported"].notna()
        if (~has).any():
            notes.append(f"{int((~has).sum())} accounts without reported ECL not reconciled.")
        tr = float(out.loc[has, "ecl_reported"].sum())
        summary |= {"total_reported": tr, "difference": float(out.loc[has, "difference"].sum()),
                    "relative_difference": float(out.loc[has, "difference"].sum() / tr) if tr else np.nan,
                    "n_mismatch": int((out.loc[has, "difference"].abs() > tolerance).sum()),
                    "max_abs_difference": float(out.loc[has, "difference"].abs().max())}
    tot = pd.DataFrame({"scenario": list(wide.columns), "weight": [w[s] for s in wide.columns],
                        "total_ecl": wide.sum().to_numpy()})
    tot = pd.concat([tot, pd.DataFrame([{"scenario": "probability-weighted", "weight": 1.0,
                                          "total_ecl": summary["total_weighted"]}])], ignore_index=True)
    if base_scenario:
        b = str(base_scenario).strip()
        if b not in wide.columns:
            raise ValueError(f"base_scenario '{b}' not among {list(wide.columns)}.")
        bt = float(wide[b].sum())
        summary |= {"total_base": bt, "weighted_to_base_ratio": summary["total_weighted"] / bt if bt else np.nan}
    return Outcome(summary, {"ECL by scenario": tot, "ECL per account": out.reset_index()}, notes=notes,
                   rows_used=len(out) * len(w))


@register("ifrs9.scenario_weight_sensitivity", "Sensitivity of ECL to scenario weights", "ECL measurement", _IFRS9,
          params=(P("id"), P("scenario"), P("ecl"), P("weights", "dict", help="Current scenario weights"),
                  P("alternative_weights", "dict", required=False,
                    help='Named alternative weight sets, e.g. {"severe": {"base": 0.4, "up": 0.1, "down": 0.5}}'),
                  P("base_scenario", "string", required=False),
                  P("shift", "number", default=0.1, help="Weight moved from base_scenario to each other scenario")),
          description="""Total probability-weighted ECL under the current weights, under each alternative weight set and
under 100% weight on each single scenario, with the change versus the current weights (absolute and %).
Because ECL is linear in the weights, ∂ECL/∂w_s = total ECL_s; with `base_scenario` the table also shows the
effect of moving `shift` of weight from the base to each other scenario: shift · (ECL_s − ECL_base).""",
          references=("IFRS 9 Financial Instruments, paragraphs 5.5.17–5.5.18 and B5.5.42",
                      "IFRS 7 Financial Instruments: Disclosures, paragraph 35G (inputs and assumptions)"))
def scenario_weight_sensitivity(ctx: RunContext, id, scenario, ecl, weights, alternative_weights=None,
                                base_scenario=None, shift=0.1) -> Outcome:
    wide, w, notes = _scenarios(ctx, id, scenario, ecl, weights)
    totals = wide.sum()
    cur = float(sum(w[s] * totals[s] for s in wide.columns))
    sets = [("current", w)]
    for name in sorted(alternative_weights or {}):
        alt = {str(k).strip(): float(v) for k, v in alternative_weights[name].items()}
        if set(alt) - set(wide.columns) or abs(sum(alt.values()) - 1) > 1e-6 or any(v < 0 for v in alt.values()):
            raise ValueError(f"Alternative weights '{name}' must use scenarios {list(wide.columns)}, be "
                             "non-negative and sum to 1.")
        sets.append((str(name), alt))
    sets += [(f"100% {s}", {s: 1.0}) for s in wide.columns]
    rows = []
    for name, ws in sets:
        tot = float(sum(ws.get(s, 0.0) * totals[s] for s in wide.columns))
        rows.append({"weight_set": name, "weights": ", ".join(f"{s}={ws.get(s, 0.0):g}" for s in wide.columns),
                     "total_ecl": tot, "change_vs_current": tot - cur,
                     "change_pct": (tot - cur) / cur if cur else np.nan})
    tabs = {"ECL by weight set": pd.DataFrame(rows)}
    if base_scenario:
        b = str(base_scenario).strip()
        if b not in wide.columns:
            raise ValueError(f"base_scenario '{b}' not among {list(wide.columns)}.")
        if shift > w[b]:
            notes.append(f"shift {shift:g} exceeds the base weight {w[b]:g}: the shifted weights are hypothetical.")
        tabs["Weight shift from base"] = pd.DataFrame(
            [{"to_scenario": s, "shift": shift, "change_in_total_ecl": shift * (totals[s] - totals[b]),
              "change_pct": shift * (totals[s] - totals[b]) / cur if cur else np.nan}
             for s in wide.columns if s != b])
    t = tabs["ECL by weight set"]
    single = t[t["weight_set"].str.startswith("100% ")]
    return Outcome({"total_current_weights": cur, "min_single_scenario": float(single["total_ecl"].min()),
                    "max_single_scenario": float(single["total_ecl"].max()), "weight_sets": len(t),
                    "n_accounts": len(wide)}, tabs, notes=notes, rows_used=len(wide) * len(w))


# ── lifetime PD ────────────────────────────────────────────────────────────

@register("ifrs9.lifetime_pd_markov", "Lifetime PD term structure from a rating transition matrix (Markov chain)",
          "Lifetime PD", ("ifrs9", "pd"),
          params=(P("transition", "table", help="Loaded transition matrix: rows = from grade, columns = to grade "
                                                "(incl. default), probabilities per period"),
                  P("from_column", "string", required=False,
                    help="Column of the matrix table holding the from-grade labels (default: first non-numeric column)"),
                  P("default_state", "string", required=False, help="Default state label (default: last column)"),
                  P("horizon", "integer", default=10, help="Number of periods"),
                  P("grade", required=False, help="Current grade per account (portfolio average curve)"),
                  P("pd", required=False, help="12-month PD per account; anchors period 1 (needs grade)")),
          description="""Cumulative PD by horizon from a time-homogeneous Markov chain: CPD_g(t) = (e_g · M^t)[D], with
the default state D absorbing. Marginal PD_t = CPD(t) − CPD(t−1) and conditional PD_t = marginal_t /
(1 − CPD(t−1)). With a 12-month `pd` per account the first period is anchored on it: the start vector is
pd·e_D + (1 − pd)·(non-default part of row g, renormalised), then propagated with M; because this is linear in
pd, grade averages use the mean pd of the grade. Also checks the matrix (row sums, negative entries,
absorbing default) and, with `pd`, compares the mean 12-month PD per grade with the matrix one-period PD.""",
          references=("Jarrow, Lando & Turnbull (1997), A Markov model for the term structure of credit risk "
                      "spreads, Review of Financial Studies 10(2)",
                      "IFRS 9 Financial Instruments, B5.5.13 and B5.5.42 (lifetime PD)"))
def lifetime_pd_markov(ctx: RunContext, transition, from_column=None, default_state=None, horizon=10,
                       grade=None, pd=None) -> Outcome:
    pd_col, pd = pd, pandas          # the parameter uses the ROLES name; restore the module name
    tm = ctx.tables[transition].copy()
    notes = []
    if from_column is None:
        nonnum = [c for c in tm.columns if not pd.to_numeric(tm[c], errors="coerce").notna().all()]
        from_column = nonnum[0] if nonnum else None
    if from_column is not None:
        if from_column not in tm.columns:
            raise ValueError(f"'{from_column}' not in the transition table.")
        rows_lab = tm[from_column].astype(str).str.strip().tolist()
        tm = tm.drop(columns=[from_column])
    else:
        rows_lab = [str(c).strip() for c in tm.columns][:len(tm)]
    cols_lab = [str(c).strip() for c in tm.columns]
    M = tm.apply(lambda c: pd.to_numeric(c, errors="coerce")).to_numpy(float)
    if np.isnan(M).any() or (M < 0).any():
        raise ValueError("Transition matrix has missing or negative entries.")
    D = str(default_state).strip() if default_state is not None else cols_lab[-1]
    if D not in cols_lab:
        raise ValueError(f"Default state '{D}' not among the columns {cols_lab}.")
    if any(r not in cols_lab for r in rows_lab):
        raise ValueError(f"Row labels {rows_lab} must all appear among the column labels {cols_lab}.")
    order = cols_lab
    full = np.zeros((len(order), len(order)))
    for i, r in enumerate(rows_lab):
        full[order.index(r)] = M[i]
    di = order.index(D)
    if D not in rows_lab:
        full[di, di] = 1.0
    elif abs(full[di, di] - 1) > 1e-9:
        notes.append("Default row was not absorbing; set to absorbing.")
        full[di] = 0
        full[di, di] = 1.0
    missing_rows = [s for s in order if s not in rows_lab and s != D]
    if missing_rows:
        raise ValueError(f"No row for state(s) {missing_rows}.")
    rs = full.sum(axis=1)
    if (np.abs(rs - 1) > 1e-3).any():
        raise ValueError(f"Row sums differ from 1 (min {rs.min():.4f}, max {rs.max():.4f}).")
    if (np.abs(rs - 1) > 1e-9).any():
        notes.append(f"Row sums renormalised (max deviation {np.abs(rs - 1).max():.2e}).")
        full = full / rs[:, None]
    nd = [s for s in order if s != D]
    anchor = {}
    counts = None
    if pd_col and not grade:
        raise ValueError("`pd` anchoring needs the `grade` column.")
    if grade:
        g = ctx.df[grade].astype(str).str.strip()
        sub = pd.DataFrame({"g": g.where(ctx.df[grade].notna())})
        if pd_col:
            sub["p"] = num(ctx.df, pd_col)
        sub = sub.dropna()
        n0 = len(g)
        notes += dropped_note(n0 - len(sub), "accounts with missing grade / PD")
        unknown = sorted(set(sub["g"]) - set(nd))
        if unknown:
            raise ValueError(f"Grades {unknown[:8]} are not non-default states of the matrix {nd}.")
        counts = sub["g"].value_counts().reindex(nd, fill_value=0)
        if pd_col:
            anchor = sub.groupby("g")["p"].mean().to_dict()
    powers = [np.eye(len(order))]
    for _ in range(horizon):
        powers.append(powers[-1] @ full)
    rows, check = [], []
    for s in nd:
        i = order.index(s)
        if s in anchor:
            p = anchor[s]
            rest = full[i].copy()
            rest[di] = 0
            rest = rest / rest.sum() if rest.sum() > 0 else rest
            v1 = (1 - p) * rest
            v1[di] = p
            cpd = np.r_[0.0, [(v1 @ powers[t - 1])[di] for t in range(1, horizon + 1)]]
            check.append({"grade": s, "n_accounts": int(counts[s]), "mean_12m_pd": p,
                          "matrix_one_period_pd": full[i, di], "difference": p - full[i, di]})
        else:
            cpd = np.r_[0.0, [powers[t][i, di] for t in range(1, horizon + 1)]]
        marg = np.diff(cpd)
        cond = marg / np.where(1 - cpd[:-1] > 0, 1 - cpd[:-1], np.nan)
        for t in range(1, horizon + 1):
            rows.append({"grade": s, "horizon": t, "cumulative_pd": cpd[t], "marginal_pd": marg[t - 1],
                         "conditional_pd": cond[t - 1]})
    ts = pd.DataFrame(rows)
    wide = ts.pivot(index="grade", columns="horizon", values="cumulative_pd").reindex(nd)
    wide.columns = [f"cpd_{c}" for c in wide.columns]
    tabs = {"Cumulative PD by grade": wide.reset_index(), "Term structure (long)": ts}
    summary = {"states": len(order), "horizon": horizon, "default_state": D}
    if counts is not None and counts.sum() > 0:
        wts = counts / counts.sum()
        port = (wide.mul(wts, axis=0)).sum()
        tabs["Portfolio cumulative PD"] = pd.DataFrame({"horizon": range(1, horizon + 1),
                                                        "cumulative_pd": port.to_numpy()})
        summary |= {"portfolio_cpd_1": float(port.iloc[0]), f"portfolio_cpd_{horizon}": float(port.iloc[-1]),
                    "n_accounts": int(counts.sum())}
    if check:
        tabs["12m PD vs matrix"] = pd.DataFrame(check)
        notes.append("Period 1 anchored on the mean 12-month PD per grade; later periods follow the matrix.")
    notes.append("Assumes a time-homogeneous first-order Markov chain; the matrix period is one model period.")
    return Outcome(summary, tabs, notes=notes,
                   rows_used=int(counts.sum()) if counts is not None else len(nd))


@register("ifrs9.lifetime_pd_term_structure", "Lifetime PD term structure from per-period PDs (consistency)",
          "Lifetime PD", ("ifrs9", "pd"),
          params=(P("pd_columns", "columns", help="PD per period, period 1..T"),
                  P("pd_type", "string", default="marginal", choices=("marginal", "cumulative", "conditional")),
                  P("segment", required=False)),
          description="""Converts a per-account PD term structure between its three forms — unconditional marginal
PD_t, cumulative PD CPD_t = Σ_{s≤t} marginal_s, conditional PD (hazard) h_t = marginal_t / (1 − CPD_{t−1}) — and
reports the average curves per segment. Internal-consistency counts: accounts with PDs outside [0, 1],
decreasing cumulative PD (negative marginal PD), cumulative PD above 1, and conditional PDs outside [0, 1].""",
          references=("IFRS 9 Financial Instruments, B5.5.13 and B5.5.42 (lifetime PD)",
                      "EBA/GL/2017/06, Guidelines on credit institutions' credit risk management practices and "
                      "accounting for expected credit losses"))
def lifetime_pd_term_structure(ctx: RunContext, pd_columns, pd_type="marginal", segment=None) -> Outcome:
    df = ctx.df
    m = np.column_stack([num(df, c).to_numpy() for c in pd_columns])
    ok = ~np.isnan(m).any(axis=1)
    seg = df[segment] if segment else pd.Series("ALL", index=df.index)
    ok &= seg.notna().to_numpy()
    m, seg = m[ok], seg[ok]
    if len(m) == 0:
        raise NotApplicable("No account with a complete term structure.")
    T = m.shape[1]
    out_of_range = int(((m < 0) | (m > 1)).any(axis=1).sum())
    marg = to_marginal(m, pd_type)
    cum = np.cumsum(marg, axis=1)
    prev = np.c_[np.zeros(len(cum)), cum[:, :-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = marg / (1 - prev)
    viol = {"n_input_outside_0_1": out_of_range,
            "n_negative_marginal": int((marg < -1e-12).any(axis=1).sum()),
            "n_cumulative_above_1": int((cum > 1 + 1e-12).any(axis=1).sum()),
            "n_conditional_outside_0_1": int(((cond < -1e-12) | (cond > 1 + 1e-12)).any(axis=1).sum())}
    rows = []
    groups = sorted_levels(seg) if segment else []
    for g in [*groups, "ALL"]:
        mask = np.ones(len(m), bool) if g == "ALL" else (seg == g).to_numpy()
        mm = marg[mask].mean(axis=0)
        cc = np.cumsum(mm)
        pc = np.r_[0.0, cc[:-1]]
        for t in range(T):
            rows.append({"segment": str(g), "horizon": t + 1, "n": int(mask.sum()), "mean_marginal_pd": mm[t],
                         "mean_cumulative_pd": cc[t],
                         "conditional_pd_of_mean_curve": mm[t] / (1 - pc[t]) if pc[t] < 1 else np.nan})
    t = pd.DataFrame(rows)
    allc = t[t["segment"] == "ALL"]
    return Outcome({"n": len(m), "periods": T, "mean_cpd_1": float(allc["mean_cumulative_pd"].iloc[0]),
                    f"mean_cpd_{T}": float(allc["mean_cumulative_pd"].iloc[-1])} | viol,
                   {"Average term structure": t, "Consistency checks": pd.DataFrame([viol])},
                   notes=dropped_note(int((~ok).sum()), "accounts with incomplete term structure / segment"),
                   rows_used=len(m))


@register("ifrs9.km_lifetime_pd_backtest", "Kaplan–Meier cumulative default curves vs predicted lifetime PD",
          "Lifetime PD", ("ifrs9", "pd"),
          params=(P("duration", help="Time from cohort start to default or censoring (same unit as horizons)"),
                  P("target", help="1 = default observed at `duration`, 0 = censored (repaid, sold, still performing)"),
                  P("segment", required=False, help="Vintage / cohort"),
                  P("pd_columns", "columns", required=False,
                    help="Predicted cumulative PD at each horizon (one column per horizon, same order)"),
                  P("horizons", "list", default=[1, 2, 3, 4, 5]),
                  P("confidence", "number", default=0.95)),
          description="""Empirical cumulative default rate by horizon per vintage with the Kaplan–Meier estimator
(censored accounts leave the risk set; F(h) = 1 − Π_{t_j ≤ h}(1 − d_j/n_j)), Greenwood standard errors and
normal CIs clipped to [0, 1]. With predicted cumulative PDs, the mean prediction per horizon is compared with
F(h): z = (predicted − F)/se_F, two-sided p (H0: predicted curve equals the empirical one; prediction treated
as fixed). Horizons beyond the longest follow-up of a vintage are left empty. Assumes non-informative
censoring; prepayments are treated as censoring (no competing-risk adjustment).""",
          references=("Kaplan & Meier (1958), JASA 53(282)", "Greenwood (1926), Reports on Public Health and "
                      "Medical Subjects 33",
                      "IFRS 9 Financial Instruments, B5.5.52 (back-testing of ECL inputs)"))
def km_lifetime_pd_backtest(ctx: RunContext, duration, target, segment=None, pd_columns=None,
                            horizons=(1, 2, 3, 4, 5), confidence=0.95) -> Outcome:
    hs = sorted(float(h) for h in horizons)
    if pd_columns and len(pd_columns) != len(hs):
        raise ValueError(f"{len(pd_columns)} pd_columns for {len(hs)} horizons.")
    df = ctx.df
    d = pd.DataFrame({"t": num(df, duration), "e": flag(df[target], target)})
    if segment:
        d["seg"] = df[segment]
    for i, c in enumerate(pd_columns or []):
        d[f"p{i}"] = num(df, c)
    n0 = len(d)
    d = d.dropna()
    if (d["t"] < 0).any():
        raise ValueError("Negative durations.")
    if d["e"].sum() == 0:
        raise NotApplicable("No defaults observed.")
    z = stats.norm.ppf(0.5 + confidence / 2)
    rows = []
    groups = sorted_levels(d["seg"]) if segment else []
    for g in [*groups, "ALL"]:
        s = d if g == "ALL" else d[d["seg"] == g]
        curve = km_curve(s["t"], s["e"])
        tmax = s["t"].max()
        for i, h in enumerate(hs):
            row = {"vintage": str(g), "horizon": h, "n": len(s), "defaults_to_h": int(((s["t"] <= h) & (s["e"] == 1)).sum()),
                   "at_risk_at_h": int((s["t"] >= h).sum())}
            if h <= tmax:
                S, se = km_at(curve, h)
                F = 1 - S
                row |= {"km_cumulative_pd": F, "se": se, "ci_low": max(0.0, F - z * se), "ci_high": min(1.0, F + z * se)}
                if pd_columns:
                    pm = float(s[f"p{i}"].mean())
                    zz = (pm - F) / se if se > 0 else np.nan
                    row |= {"predicted_cumulative_pd": pm, "difference": pm - F, "z": zz,
                            "p_value": float(2 * stats.norm.sf(abs(zz))) if se > 0 else np.nan}
            rows.append(row)
    t = pd.DataFrame(rows)
    a = t[t["vintage"] == "ALL"]
    summary = {"n": len(d), "defaults": int(d["e"].sum())}
    for _, r in a.iterrows():
        summary[f"km_cpd_{r['horizon']:g}"] = r.get("km_cumulative_pd", np.nan)
        if pd_columns:
            summary[f"predicted_cpd_{r['horizon']:g}"] = r.get("predicted_cumulative_pd", np.nan)
    return Outcome(summary, {"Cumulative default by vintage": t},
                   notes=dropped_note(n0 - len(d)) + ["Normal CI on F(h) with Greenwood variance; prepayment and "
                                                      "other exits treated as non-informative censoring."],
                   rows_used=len(d))


# ── staging ────────────────────────────────────────────────────────────────

@register("ifrs9.staging_replication", "Stage allocation replication from SICR rules", "Staging", _IFRS9,
          params=(P("id"), P("stage", help="Reported stage 1/2/3"),
                  P("pd", required=False, help="PD at the reporting date (same basis as at origination)"),
                  P("pd_origination", required=False, help="PD at initial recognition"),
                  P("relative_multiple", "number", required=False, help="SICR if pd / pd_origination >= multiple"),
                  P("absolute_change", "number", required=False, help="SICR if pd − pd_origination >= change"),
                  P("combine", "string", default="or", choices=("or", "and"),
                    help="Combine relative and absolute PD criteria with or / and"),
                  P("low_credit_risk_pd", "number", required=False,
                    help="Low-credit-risk exemption: PD criteria ignored when pd <= this value"),
                  P("dpd", required=False, help="Days past due"),
                  P("dpd_backstop", "integer", default=30, help="Stage 2 if dpd > backstop"),
                  P("stage3_dpd", "integer", default=90, help="Stage 3 if dpd > this"),
                  P("watchlist", required=False, help="0/1 watchlist flag (stage 2 trigger)"),
                  P("forbearance", required=False, help="0/1 forbearance flag (stage 2 trigger)"),
                  P("default_flag", required=False, help="0/1 default / credit-impaired flag (stage 3)")),
          description="""Replicates the stage of each account from the stated rules and compares with the reported stage.
Stage 3 if default_flag = 1 or dpd > stage3_dpd. Otherwise stage 2 if any of: quantitative SICR (relative
pd/pd_origination ≥ multiple and/or absolute pd − pd_origination ≥ change, combined by `combine`, switched off
when pd ≤ low_credit_risk_pd), dpd > dpd_backstop (30-days-past-due rebuttable presumption), watchlist = 1,
forbearance = 1. Else stage 1. Reports the reported × replicated confusion matrix, agreement rate, Cohen's
kappa, the mismatching accounts with the triggers that fired, and per-trigger counts (including accounts where
it was the only trigger). Missing optional inputs are treated as 'trigger not fired'.""",
          references=("IFRS 9 Financial Instruments, paragraphs 5.5.3, 5.5.9–5.5.11 and B5.5.15–B5.5.24 "
                      "(significant increase in credit risk, low credit risk, 30 days past due)",
                      "EBA/GL/2017/06, Guidelines on credit institutions' credit risk management practices and "
                      "accounting for expected credit losses",
                      "Cohen (1960), Educational and Psychological Measurement 20(1)"))
def staging_replication(ctx: RunContext, id, stage, pd=None, pd_origination=None, relative_multiple=None,
                        absolute_change=None, combine="or", low_credit_risk_pd=None, dpd=None, dpd_backstop=30,
                        stage3_dpd=90, watchlist=None, forbearance=None, default_flag=None) -> Outcome:
    pd_col, pd = pd, pandas          # the parameter uses the ROLES name; restore the module name
    df = ctx.df
    if (relative_multiple is not None or absolute_change is not None) and not (pd_col and pd_origination):
        raise ValueError("PD-based SICR criteria need both `pd` and `pd_origination`.")
    d = pd.DataFrame({"id": df[id], "reported": _stage(df[stage], stage)})
    n0 = len(d)
    keep = d.notna().all(axis=1)
    if low_credit_risk_pd is not None and not pd_col:
        raise ValueError("The low-credit-risk exemption needs `pd`.")
    if pd_col and (relative_multiple is not None or absolute_change is not None or low_credit_risk_pd is not None):
        cur, orig = num(df, pd_col), num(df, pd_origination) if pd_origination else None
        keep &= cur.notna() & (orig.notna() if orig is not None else True)
    d = d[keep]
    notes = dropped_note(n0 - len(d), "accounts with missing stage / PD inputs")
    if d.empty:
        raise NotApplicable("No complete accounts.")
    idx = d.index
    trig = {}
    quant = []
    if relative_multiple is not None:
        c, o = num(df, pd_col).loc[idx], num(df, pd_origination).loc[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(o > 0, c / o, np.where(c > 0, np.inf, 1.0))
        quant.append(pd.Series(ratio >= relative_multiple, idx))
    if absolute_change is not None:
        c, o = num(df, pd_col).loc[idx], num(df, pd_origination).loc[idx]
        quant.append((c - o) >= absolute_change)
    if quant:
        q = quant[0]
        for x in quant[1:]:
            q = (q | x) if combine == "or" else (q & x)
        if low_credit_risk_pd is not None:
            lcr = num(df, pd_col).loc[idx] <= low_credit_risk_pd
            trig["low_credit_risk_exempt"] = lcr & q
            q = q & ~lcr
        trig["pd_sicr"] = q
    if dpd:
        dd = num(df, dpd).loc[idx]
        trig["dpd_backstop"] = (dd > dpd_backstop).fillna(False)
        trig["dpd_stage3"] = (dd > stage3_dpd).fillna(False)
    if watchlist:
        trig["watchlist"] = flag(df[watchlist], watchlist).loc[idx].fillna(0) == 1
    if forbearance:
        trig["forbearance"] = flag(df[forbearance], forbearance).loc[idx].fillna(0) == 1
    if default_flag:
        trig["default"] = flag(df[default_flag], default_flag).loc[idx].fillna(0) == 1
    false = pd.Series(np.zeros(len(idx), bool), idx)
    s3 = trig.get("default", false) | trig.get("dpd_stage3", false)
    s2_names = [k for k in ("pd_sicr", "dpd_backstop", "watchlist", "forbearance") if k in trig]
    s2 = false.copy()
    for k in s2_names:
        s2 = s2 | trig[k]
    rep = np.where(s3, 3, np.where(s2, 2, 1))
    d["replicated"] = rep
    names = [k for k in ("default", "dpd_stage3", "pd_sicr", "dpd_backstop", "watchlist", "forbearance") if k in trig]
    fired = pd.DataFrame({k: trig[k].astype(bool) for k in names}, index=idx) if names else pd.DataFrame(index=idx)
    d["triggers"] = fired.apply(lambda r: ", ".join(k for k in names if r[k]), axis=1) if names else ""
    cm = pd.crosstab(d["reported"].astype(int), d["replicated"].astype(int)).reindex(
        index=[1, 2, 3], columns=[1, 2, 3], fill_value=0)
    cm.index.name, cm.columns.name = "reported", "replicated"
    n = len(d)
    po = np.trace(cm.to_numpy()) / n
    pe = float((cm.sum(axis=1).to_numpy() * cm.sum(axis=0).to_numpy()).sum()) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    mism = d[d["reported"] != d["replicated"]].copy()
    extra = {"pd": pd_col, "pd_origination": pd_origination, "dpd": dpd}
    for k, c in extra.items():
        if c:
            mism[k] = num(df, c).loc[mism.index]
    mism["reported"] = mism["reported"].astype(int)
    mism = mism.sort_values("id", key=lambda s: s.astype(str), kind="mergesort")
    trows = []
    for k in names + (["low_credit_risk_exempt"] if "low_credit_risk_exempt" in trig else []):
        f = trig[k].astype(bool)
        only = f & (fired.drop(columns=[k]).sum(axis=1) == 0) if k in names else f & False
        trows.append({"trigger": k, "n_fired": int(f.sum()), "n_fired_reported_stage2": int((f & (d["reported"] == 2)).sum()),
                      "n_fired_reported_stage3": int((f & (d["reported"] == 3)).sum()),
                      "n_only_trigger": int(only.sum())})
    if not names:
        notes.append("No staging rule inputs given: every account replicates to stage 1.")
    return Outcome({"n": n, "agreement_rate": po, "cohen_kappa": kappa, "n_mismatch": len(mism),
                    "n_reported_stage2": int((d["reported"] == 2).sum()), "n_replicated_stage2": int((rep == 2).sum()),
                    "n_reported_stage3": int((d["reported"] == 3).sum()), "n_replicated_stage3": int((rep == 3).sum())},
                   {"Reported vs replicated stage": cm.reset_index(), "Mismatches": mism.reset_index(drop=True),
                    "Triggers": pd.DataFrame(trows)},
                   notes=notes, rows_used=n)




@register("ifrs9.stage_migration", "Stage migration matrix between two dates", "Staging", _IFRS9,
          params=(P("id"), P("stage", help="Stage (long format: one row per account and date)"),
                  P("period", required=False, help="Reporting date column (long format)"),
                  P("reference_value", "string", required=False, help="From date (default: first)"),
                  P("current_value", "string", required=False, help="To date (default: last)"),
                  P("stage_to", required=False, help="Stage at the second date (wide format, instead of period)"),
                  P("ead", required=False, help="Exposure at the first date, for an exposure-weighted matrix")),
          description="""Transition of accounts between stages from a first to a second reporting date: counts, row
percentages (share of each from-stage moving to each to-stage) and, with `ead`, exposure-weighted shares. Also
the share staying in the same stage, moving to a worse stage and moving to a better stage, plus accounts present
at only one date (exits / new originations) in long format. Descriptive.""",
          references=("IFRS 7 Financial Instruments: Disclosures, paragraph 35H (reconciliation of loss allowance by stage)",
                      "EBA/GL/2017/06, Guidelines on credit institutions' credit risk management practices and "
                      "accounting for expected credit losses"))
def stage_migration(ctx: RunContext, id, stage, period=None, reference_value=None, current_value=None,
                    stage_to=None, ead=None) -> Outcome:
    df = ctx.df
    notes = []
    n_exit = n_new = 0
    if stage_to:
        d = pd.DataFrame({"id": df[id], "s0": _stage(df[stage], stage), "s1": _stage(df[stage_to], stage_to)})
        if ead:
            d["w"] = num(df, ead)
        n0 = len(d)
        d = d.dropna()
        notes += dropped_note(n0 - len(d))
        labels = ("from", "to")
    elif period:
        per = df[period]
        vals = sorted_levels(per)
        if len(vals) < 2:
            raise NotApplicable("Need at least two reporting dates.")
        def pick(v, default):
            if v is None:
                return default
            m = [x for x in vals if str(x) == str(v)]
            if not m:
                raise ValueError(f"'{v}' not among the dates of '{period}': {[str(x) for x in vals[:12]]}")
            return m[0]
        p0, p1 = pick(reference_value, vals[0]), pick(current_value, vals[-1])
        a = pd.DataFrame({"id": df[id], "s": _stage(df[stage], stage)})
        if ead:
            a["w"] = num(df, ead)
        a0, a1 = a[per == p0].dropna(), a[per == p1].dropna(subset=["id", "s"])
        if a0["id"].duplicated().any() or a1["id"].duplicated().any():
            raise ValueError("Duplicate ids within a reporting date.")
        d = a0.rename(columns={"s": "s0"}).merge(a1[["id", "s"]].rename(columns={"s": "s1"}), on="id", how="inner")
        n_exit, n_new = int((~a0["id"].isin(a1["id"])).sum()), int((~a1["id"].isin(a0["id"])).sum())
        labels = (str(p0), str(p1))
    else:
        raise ValueError("Give `period` (long format) or `stage_to` (wide format).")
    if d.empty:
        raise NotApplicable("No account observed at both dates.")
    cm = pd.crosstab(d["s0"].astype(int), d["s1"].astype(int)).reindex(index=[1, 2, 3], columns=[1, 2, 3], fill_value=0)
    pct = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    for t in (cm, pct):
        t.index.name, t.columns.name = f"stage_{labels[0]}", f"stage_{labels[1]}"
    tabs = {"Migration counts": cm.reset_index(), "Migration row %": pct.reset_index()}
    if ead:
        wm = d.pivot_table(index="s0", columns="s1", values="w", aggfunc="sum").reindex(
            index=[1, 2, 3], columns=[1, 2, 3]).fillna(0)
        wm.index, wm.columns = wm.index.astype(int), wm.columns.astype(int)
        wp = wm.div(wm.sum(axis=1).replace(0, np.nan), axis=0)
        wp.index.name, wp.columns.name = f"stage_{labels[0]}", f"stage_{labels[1]}"
        tabs["Exposure-weighted row %"] = wp.reset_index()
    n = len(d)
    summary = {"n_matched": n, "share_same_stage": float((d["s0"] == d["s1"]).mean()),
               "share_worse_stage": float((d["s1"] > d["s0"]).mean()),
               "share_better_stage": float((d["s1"] < d["s0"]).mean()),
               "share_stage1_to_2": float(pct.loc[1, 2]) if cm.loc[1].sum() else np.nan,
               "share_stage2_to_1": float(pct.loc[2, 1]) if cm.loc[2].sum() else np.nan,
               "from": labels[0], "to": labels[1]}
    if period:
        summary |= {"n_exited": n_exit, "n_new": n_new}
    return Outcome(summary, tabs, notes=notes, rows_used=n)


@register("ifrs9.coverage_by_stage", "Stage mix and ECL coverage ratios by stage / segment over time",
          "Staging", _IFRS9,
          params=(P("stage"), P("ead", help="Exposure / gross carrying amount"), P("ecl", help="ECL / loss allowance"),
                  P("period", required=False), P("segment", required=False)),
          description="""Per reporting period × segment × stage: number of accounts, exposure, ECL, coverage ratio
(ECL / exposure) and the stage's share of accounts and of exposure; totals per period × segment. Shows the
stage-2 share and the coverage of each stage over time — the usual IFRS 9 monitoring KPIs. Descriptive.""",
          references=("IFRS 7 Financial Instruments: Disclosures, paragraphs 35H–35M",
                      "EBA/GL/2017/06, Guidelines on credit institutions' credit risk management practices and "
                      "accounting for expected credit losses"))
def coverage_by_stage(ctx: RunContext, stage, ead, ecl, period=None, segment=None) -> Outcome:
    df = ctx.df
    d = pd.DataFrame({"stage": _stage(df[stage], stage), "x": num(df, ead), "ecl": num(df, ecl)})
    keys = []
    if period:
        d["period"], keys = df[period], keys + ["period"]
    if segment:
        d["segment"], keys = df[segment], keys + ["segment"]
    n0 = len(d)
    d = d.dropna()
    if d.empty:
        raise NotApplicable("No complete rows.")
    d["stage"] = d["stage"].astype(int)
    combos = sorted(d[keys].drop_duplicates().itertuples(index=False, name=None),
                    key=lambda r: tuple((str(type(v)), v) for v in r)) if keys else [()]
    rows = []
    for combo in combos:
        sub = d[(d[keys] == pd.Series(combo, index=keys)).all(axis=1)] if keys else d
        n, x = len(sub), float(sub["x"].sum())
        for st in [1, 2, 3, "ALL"]:
            s = sub if st == "ALL" else sub[sub["stage"] == st]
            row = {k: str(v) for k, v in zip(keys, combo)}
            sx, se = float(s["x"].sum()), float(s["ecl"].sum())
            row |= {"stage": str(st), "n": len(s), "exposure": sx, "ecl": se,
                    "coverage_ratio": se / sx if sx else np.nan, "share_of_accounts": len(s) / n if n else np.nan,
                    "share_of_exposure": sx / x if x else np.nan}
            rows.append(row)
    t = pd.DataFrame(rows)
    latest = d
    if period:
        lp = sorted_levels(d["period"])[-1]
        latest = d[d["period"] == lp]
    tot_x = float(latest["x"].sum())
    summary = {}
    for st in (1, 2, 3):
        s = latest[latest["stage"] == st]
        x, e = float(s["x"].sum()), float(s["ecl"].sum())
        summary[f"stage{st}_share_of_exposure"] = x / tot_x if tot_x else np.nan
        summary[f"stage{st}_coverage"] = e / x if x else np.nan
    summary["total_coverage"] = float(latest["ecl"].sum()) / tot_x if tot_x else np.nan
    if period:
        summary["period"] = str(lp)
    return Outcome(summary, {"Coverage by stage": t},
                   notes=dropped_note(n0 - len(d)) + (["Headline figures are for the latest period."] if period else []),
                   rows_used=len(d))


# ── PIT / TTC and point-in-time calibration ────────────────────────────────

def _period_series(ctx: RunContext, period, cols: dict) -> tuple[pd.DataFrame, int]:
    df = ctx.df
    d = pd.DataFrame({"period": df[period]} | {k: (flag(df[c], c) if k == "dr" else num(df, c))
                                               for k, c in cols.items() if c})
    n0 = len(d)
    d = d.dropna()
    return d, n0 - len(d)


@register("ifrs9.pit_macro_correlation", "PIT-ness: correlation of the PD time series with a macro variable",
          "PIT / TTC", ("ifrs9", "pd"),
          params=(P("pd"), P("period"), P("feature", help="Macro-economic variable (one value per period)"),
                  P("target", required=False, help="Default flag, to add the observed default rate series"),
                  P("max_lag", "integer", default=2, help="Lags (in periods) for the lagged correlations")),
          description="""Degree of point-in-time behaviour of a PD model. Builds per-period series of mean PD, the macro
variable (period mean) and, with `target`, the observed default rate (DR). Reports Pearson and Spearman
correlations of mean PD with the macro variable in levels and first differences, lagged Pearson correlations
corr(PD_t, macro_{t−k}) for k = −max_lag..max_lag, and with DR: corr(PD, DR), corr(DR, macro) and the
variability ratio sd(mean PD)/sd(DR) (near 1 = PD moves as much as defaults, i.e. PIT; near 0 = TTC-like).
Very few periods give unstable correlations — p-values are reported, not judged.""",
          references=("IFRS 9 Financial Instruments, paragraph 5.5.17 (forward-looking, point-in-time measurement)",
                      "Aguais et al. (2004), Point-in-time versus through-the-cycle ratings, in Ong (ed.), "
                      "The Basel Handbook, Risk Books",
                      "BCBS (2005), Working Paper No. 14, Studies on the Validation of Internal Rating Systems"))
def pit_macro_correlation(ctx: RunContext, pd, period, feature, target=None, max_lag=2) -> Outcome:
    pd_col, pd = pd, pandas          # the parameter uses the ROLES name; restore the module name
    d, dropped = _period_series(ctx, period, {"pd": pd_col, "macro": feature, "dr": target})
    notes = dropped_note(dropped)
    if (d.groupby("period")["macro"].nunique() > 1).any():
        notes.append(f"'{feature}' varies within a period; the period mean is used.")
    agg = {"n": ("pd", "size"), "mean_pd": ("pd", "mean"), "macro": ("macro", "mean")}
    if target:
        agg["default_rate"] = ("dr", "mean")
    s = d.groupby("period").agg(**agg)
    s = s.loc[sorted(s.index, key=lambda v: (str(type(v)), v))]
    if len(s) < 4:
        raise NotApplicable(f"Only {len(s)} periods; need at least 4 for a correlation.")

    def corr(a, b):
        ok = a.notna() & b.notna()
        if ok.sum() < 3 or a[ok].nunique() < 2 or b[ok].nunique() < 2:
            return np.nan, np.nan
        r = stats.pearsonr(a[ok], b[ok])
        return float(r.statistic), float(r.pvalue)

    rows = []
    pairs = [("mean PD vs macro (levels)", s["mean_pd"], s["macro"]),
             ("mean PD vs macro (first differences)", s["mean_pd"].diff(), s["macro"].diff())]
    if target:
        pairs += [("mean PD vs default rate", s["mean_pd"], s["default_rate"]),
                  ("default rate vs macro", s["default_rate"], s["macro"])]
    for name, a, b in pairs:
        r, p = corr(a, b)
        ok = a.notna() & b.notna()
        sp = stats.spearmanr(a[ok], b[ok]) if ok.sum() >= 3 else None
        rows.append({"pair": name, "periods": int(ok.sum()), "pearson_r": r, "pearson_p": p,
                     "spearman_rho": float(sp.statistic) if sp is not None else np.nan,
                     "spearman_p": float(sp.pvalue) if sp is not None else np.nan})
    lags = []
    for k in range(-max_lag, max_lag + 1):
        r, p = corr(s["mean_pd"], s["macro"].shift(k))
        lags.append({"lag": k, "meaning": f"PD_t vs macro_(t{-k:+d})" if k else "PD_t vs macro_t",
                     "periods": int((s["mean_pd"].notna() & s["macro"].shift(k).notna()).sum()),
                     "pearson_r": r, "pearson_p": p})
    t = pd.DataFrame(rows)
    summary = {"periods": len(s), "corr_pd_macro": t.iloc[0]["pearson_r"], "p_pd_macro": t.iloc[0]["pearson_p"],
               "corr_pd_macro_diff": t.iloc[1]["pearson_r"]}
    if target:
        sd_dr = float(s["default_rate"].std(ddof=1))
        summary |= {"corr_pd_dr": t.iloc[2]["pearson_r"],
                    "variability_ratio": float(s["mean_pd"].std(ddof=1)) / sd_dr if sd_dr else np.nan}
    return Outcome(summary, {"Period series": s.reset_index().assign(period=lambda x: x["period"].astype(str)),
                             "Correlations": t, "Lagged correlations": pd.DataFrame(lags)},
                   notes=notes, rows_used=len(d))


@register("ifrs9.pit_calibration_backtest", "Point-in-time calibration of 12-month PD per period",
          "Calibration", ("ifrs9", "pd"),
          params=(P("target", help="Default within 12 months of the observation date"), P("pd", help="12-month PIT PD"),
                  P("period", help="Observation date / cohort")),
          description="""For each observation period: observed default rate vs mean 12-month PD, exact binomial test with
the mean PD (two-sided, and one-sided H1: PD underestimates defaults, p = P(X ≥ d)), and the normal z-score
(d − Σp_i)/√Σp_i(1−p_i) that uses the account-level PDs. Across periods: χ² = Σ_t z_t² ~ χ²(T) (H0: every period
correctly calibrated, independent periods and defaults), mean absolute gap |DR − mean PD| and the Spearman
correlation of mean PD and DR over time (PIT models should track the cycle).""",
          references=("IFRS 9 Financial Instruments, B5.5.52 (regular review of methodology against actual losses)",
                      "BCBS (2005), Working Paper No. 14, Studies on the Validation of Internal Rating Systems",
                      "Tasche (2008), Validation of internal rating systems and PD estimates, in The Analytics of "
                      "Risk Model Validation, Elsevier"))
def pit_calibration_backtest(ctx: RunContext, target, pd, period) -> Outcome:
    pd_col, pd = pd, pandas          # the parameter uses the ROLES name; restore the module name
    d, dropped = _period_series(ctx, period, {"dr": target, "pd": pd_col})
    if d.empty:
        raise NotApplicable("No complete rows.")
    if ((d["pd"] < 0) | (d["pd"] > 1)).any():
        raise ValueError("PD values must be in [0, 1].")
    rows = []
    for p in sorted_levels(d["period"]):
        s = d[d["period"] == p]
        n, k = len(s), int(s["dr"].sum())
        mp = float(s["pd"].mean())
        var = float((s["pd"] * (1 - s["pd"])).sum())
        z = (k - s["pd"].sum()) / np.sqrt(var) if var > 0 else np.nan
        rows.append({"period": str(p), "n": n, "defaults": k, "default_rate": k / n, "mean_pd": mp,
                     "difference": k / n - mp,
                     "binomial_p_two_sided": float(stats.binomtest(k, n, mp).pvalue) if 0 < mp < 1 else np.nan,
                     "binomial_p_underestimation": float(stats.binom.sf(k - 1, n, mp)) if 0 < mp < 1 else np.nan,
                     "z": z})
    t = pd.DataFrame(rows)
    zz = t["z"].dropna()
    chi2 = float((zz ** 2).sum())
    summary = {"periods": len(t), "chi2": chi2, "chi2_df": len(zz), "chi2_p": float(stats.chi2.sf(chi2, len(zz))) if len(zz) else np.nan,
               "mean_abs_gap": float(t["difference"].abs().mean()), "n": len(d), "defaults": int(d["dr"].sum())}
    if len(t) >= 3 and t["mean_pd"].nunique() > 1 and t["default_rate"].nunique() > 1:
        sp = stats.spearmanr(t["mean_pd"], t["default_rate"])
        summary |= {"spearman_pd_dr": float(sp.statistic), "spearman_p": float(sp.pvalue)}
    return Outcome(summary, {"Calibration by period": t},
                   notes=dropped_note(dropped) + ["Binomial tests use the period mean PD and assume independent "
                                                  "defaults; the z-score uses account-level PDs."],
                   rows_used=len(d))
