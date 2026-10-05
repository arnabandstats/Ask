"""LGD model validation: ranking power, back-testing, calibration, distribution and workout analytics.

Conventions used throughout:
  - `actual` is the realised LGD per defaulted facility, `predicted` the model estimate
    (facility LGD, pool LGD, ELBE ... as stated per test). Both are fractions of EAD.
  - Differences are always realised - estimated, so a positive mean difference means the
    model UNDER-estimates loss (the direction ECB back-tests look at).
  - `ead` (optional) weights statistics by exposure; without it every default counts once
    (default-weighted, as for the long-run average LGD).
  - Groups (pools, segments, periods) are always processed in sorted order and the whole
    portfolio is reported as group "ALL".

The generalised AUC, the back-test t-test table and the flag/frame helpers are also used by
the EAD/CCF module (t_ead.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, dropped_note, num, register

_LGD = ("lgd",)
_GAUC_TABLE_MAX_CELLS = 4_000_000          # above this the O(n log n) tree path is used


# ── shared helpers ─────────────────────────────────────────────────────────

def _frame(ctx: RunContext, numeric: dict, cats: dict | None = None) -> tuple[pd.DataFrame, int]:
    """Columns renamed to their role (numeric ones coerced to float; a Series is used as is),
    complete rows only. The original index is kept so other columns can be aligned."""
    df = ctx.df
    out = pd.DataFrame(index=df.index)
    for k, c in numeric.items():
        if isinstance(c, pd.Series):
            out[k] = c.astype(float)
        elif c:
            out[k] = num(df, c)
    for k, c in (cats or {}).items():
        if c:
            out[k] = df[c]
    n0 = len(out)
    out = out.dropna()
    return out, n0 - len(out)


def sorted_levels(s: pd.Series) -> list:
    return sorted(pd.unique(s.dropna()), key=lambda v: (str(type(v)), v))


def flag(s: pd.Series, label: str, true_values=None) -> pd.Series:
    """A 0/1 flag from bool, 0/1 numbers or yes/no/true/false/y/n text (NaN stays NaN)."""
    if true_values:
        tv = {str(v).strip().lower() for v in true_values}
        return s.map(lambda v: np.nan if pd.isna(v) else float(str(v).strip().lower() in tv))
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float)
        vals = set(out.dropna().unique())
        if not vals <= {0.0, 1.0}:
            raise ValueError(f"Column '{label}' is not a 0/1 flag (values like {sorted(vals)[:6]})")
        return out
    m = s.map(lambda v: np.nan if pd.isna(v) else str(v).strip().lower()).map(
        {"1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0, "y": 1.0, "n": 0.0, "yes": 1.0, "no": 0.0,
         "true": 1.0, "false": 0.0})
    if m.isna().sum() > s.isna().sum():
        bad = sorted({str(v) for v, mm in zip(s, m) if pd.notna(v) and pd.isna(mm)})[:6]
        raise ValueError(f"Column '{label}' is not a 0/1 flag (values like {bad})")
    return m


def _wavg(x, w) -> float:
    w = np.asarray(w, float)
    return float(np.sum(np.asarray(x, float) * w) / np.sum(w)) if np.sum(w) else float("nan")


def _counts_lower(xr: np.ndarray, yr: np.ndarray, kx: int) -> tuple[np.ndarray, np.ndarray]:
    """For each k: #{j: y_j < y_k, x_j < x_k} and #{j: y_j < y_k, x_j == x_k} (Fenwick tree)."""
    n = len(xr)
    order = np.lexsort((xr, yr))
    tree = np.zeros(kx + 1, dtype=np.int64)
    per_level = np.zeros(kx, dtype=np.int64)
    lt, eq = np.zeros(n), np.zeros(n)
    ys = yr[order]
    i = 0
    while i < n:
        j = i
        while j < n and ys[j] == ys[i]:
            j += 1
        for k in order[i:j]:
            pos, s = int(xr[k]), 0           # prefix sum over levels 0..x_k-1
            while pos > 0:
                s += tree[pos]
                pos -= pos & -pos
            lt[k], eq[k] = s, per_level[xr[k]]
        for k in order[i:j]:
            pos = int(xr[k]) + 1
            while pos <= kx:
                tree[pos] += 1
                pos += pos & -pos
            per_level[xr[k]] += 1
        i = j
    return lt, eq


def gauc(predicted, realised) -> dict:
    """Generalised AUC: P(pred_i > pred_j | real_i > real_j), ties in prediction count 1/2.

    Variance from the Hoeffding (DeLong-type) decomposition of the U-statistic:
    var = sum_k h_k^2 / M^2 with h_k the centred concordance of observation k against all
    others and M the number of pairs with different realised values. Reduces to DeLong (1988)
    for a binary outcome."""
    x, y = np.asarray(predicted, float), np.asarray(realised, float)
    n = len(x)
    xr = np.unique(x, return_inverse=True)[1].astype(np.int64)
    ylev, yr = np.unique(y, return_inverse=True)
    kx, ky = int(xr.max()) + 1, len(ylev)
    if ky < 2:
        raise NotApplicable("Realised values do not vary; the generalised AUC is undefined.")
    ycnt = np.bincount(yr, minlength=ky)
    ycum = np.cumsum(ycnt)
    below, above = (ycum - ycnt)[yr].astype(float), (n - ycum)[yr].astype(float)
    if kx * ky <= _GAUC_TABLE_MAX_CELLS:
        T = np.zeros((kx, ky))
        np.add.at(T, (xr, yr), 1.0)
        C = T.cumsum(0).cumsum(1)
        A = np.zeros((kx + 1, ky + 1))
        A[1:, 1:] = C
        A = A[:-1, :-1]                                   # sum T[:a, :b]
        rowc = T.cumsum(1)
        B = rowc - T                                      # sum T[a, :b]
        D = T.sum(1)[:, None] - rowc                      # sum T[a, b+1:]
        C2 = n - T.sum(1).cumsum()[:, None] - T.sum(0).cumsum()[None, :] + C   # sum T[a+1:, b+1:]
        low_c, low_t = A[xr, yr], B[xr, yr]
        up_c, up_t = C2[xr, yr], D[xr, yr]
    else:
        low_c, low_t = _counts_lower(xr, yr, kx)
        up_c, up_t = _counts_lower(kx - 1 - xr, ky - 1 - yr, kx)
    M = below.sum()
    theta = float((low_c + 0.5 * low_t).sum() / M)
    h = (low_c + 0.5 * low_t - theta * below) + (up_c + 0.5 * up_t - theta * above)
    se = float(np.sqrt(np.sum(h ** 2)) / M)
    return {"gAUC": theta, "se": se, "pairs": int(M), "n": n}


def gauc_outcome(pred, real, confidence: float, initial_gauc, notes: list[str], rows_used: int,
                 label: str = "LGD") -> Outcome:
    g = gauc(pred, real)
    zc = stats.norm.ppf(0.5 + confidence / 2)
    se = g["se"]
    z0 = (g["gAUC"] - 0.5) / se if se > 0 else float("nan")
    summary = {"gAUC": g["gAUC"], "se": se, f"ci_lower_{confidence:g}": max(0.0, g["gAUC"] - zc * se),
               f"ci_upper_{confidence:g}": min(1.0, g["gAUC"] + zc * se),
               "z_vs_0.5": z0, "p_value_vs_0.5": float(2 * stats.norm.sf(abs(z0))) if se > 0 else float("nan"),
               "n": g["n"], "pairs_compared": g["pairs"]}
    if initial_gauc is not None:
        s = (initial_gauc - g["gAUC"]) / se if se > 0 else float("nan")
        summary |= {"initial_gAUC": float(initial_gauc), "z_vs_initial": s,
                    "p_value_vs_initial": float(stats.norm.sf(s)) if se > 0 else float("nan")}
        notes.append("Comparison with the initial-validation gAUC: S = (gAUC_init - gAUC_curr)/se, "
                     "p = 1 - Phi(S); H0: the current gAUC is not lower than the initial one "
                     "(se of the current sample only, the initial value is treated as fixed).")
    notes.append(f"Pairs with equal realised {label} are not compared; ties in the estimate count 1/2. "
                 "se from the Hoeffding/DeLong-type placement decomposition; normal-approximation CI.")
    tab = pd.DataFrame([{k: v for k, v in summary.items()}])
    return Outcome(summary, {"Generalised AUC": tab}, notes=notes, rows_used=rows_used)


def ttest_rows(d: pd.DataFrame, by: str | None, alternative: str) -> pd.DataFrame:
    """One-sample t-test of diff = realised - estimated per group and for ALL."""
    groups = [(g, d[d[by] == g]) for g in sorted_levels(d[by])] if by else []
    rows = []
    for g, sub in groups + [("ALL", d)]:
        diff = sub["real"] - sub["pred"]
        n = len(diff)
        sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
        t, p = float("nan"), float("nan")
        if n > 1 and sd > 0:
            r = stats.ttest_1samp(diff, 0.0, alternative=alternative)
            t, p = float(r.statistic), float(r.pvalue)
        rows.append({"group": str(g), "n": n, "mean_realised": float(sub["real"].mean()),
                     "mean_estimated": float(sub["pred"].mean()), "mean_difference": float(diff.mean()),
                     "sd_difference": sd, "t_statistic": t, "df": n - 1, "p_value": p})
    return pd.DataFrame(rows)


def _cut_bins(x: pd.Series, edges) -> pd.Series:
    e = sorted(float(v) for v in edges)
    return pd.Series(pd.cut(x, [-np.inf, *e, np.inf], right=True, labels=False), index=x.index).astype(float)


def bimodality_coefficient(x: np.ndarray) -> float:
    n = len(x)
    if n < 4 or np.std(x) == 0:
        return float("nan")
    g1 = stats.skew(x, bias=False)
    g2 = stats.kurtosis(x, bias=False)
    return float((g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def distribution_tables(values: dict[str, pd.Series], lo: float, hi: float, tol: float, bins: int
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, hist = [], []
    edges = np.linspace(lo, hi, bins + 1)
    for name, s in values.items():
        x = s.to_numpy(float)
        q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append({"variable": name, "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1) if len(x) > 1 else np.nan,
                     "min": x.min(), "p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
                     "max": x.max(), f"share_below_{lo:g}": np.mean(x < lo - tol),
                     f"share_at_{lo:g}": np.mean(np.abs(x - lo) <= tol),
                     f"share_at_{hi:g}": np.mean(np.abs(x - hi) <= tol),
                     f"share_above_{hi:g}": np.mean(x > hi + tol),
                     "share_near_lower_10pct": np.mean((x >= lo - tol) & (x <= lo + 0.1 * (hi - lo))),
                     "share_near_upper_10pct": np.mean((x >= hi - 0.1 * (hi - lo)) & (x <= hi + tol)),
                     "skewness": stats.skew(x, bias=False) if len(x) > 2 else np.nan,
                     "excess_kurtosis": stats.kurtosis(x, bias=False) if len(x) > 3 else np.nan,
                     "bimodality_coefficient": bimodality_coefficient(x)})
        inside = x[(x >= lo) & (x <= hi)]
        cnt = np.histogram(inside, edges)[0]
        labels = [f"< {lo:g}"] + [f"[{a:.3g}, {b:.3g}{']' if i == bins - 1 else ')'}"
                                  for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))] + [f"> {hi:g}"]
        counts = [int(np.sum(x < lo)), *cnt.tolist(), int(np.sum(x > hi))]
        for lab, c in zip(labels, counts):
            hist.append({"variable": name, "bin": lab, "count": c, "share": c / len(x)})
    return pd.DataFrame(rows), pd.DataFrame(hist)


def _wilson(k: float, n: float, conf: float) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(max(0.0, c - h)), float(min(1.0, c + h))


_REAL_PRED = (P("actual", help="Realised LGD per defaulted facility (fraction of EAD)"),
              P("predicted", help="Estimated LGD per facility (or the LGD of its pool)"))


# ── ranking power ──────────────────────────────────────────────────────────

@register("lgd.gauc", "Generalised AUC (gAUC) of LGD estimates", "Discrimination", _LGD,
          params=(*_REAL_PRED,
                  P("realised_bins", "list", required=False,
                    help="Optional edges to bucket realised LGD into ordinal classes before the gAUC "
                         "(e.g. the LGD grade boundaries); default uses realised values as they are"),
                  P("initial_gauc", "number", required=False,
                    help="gAUC at initial validation / development, for the ECB comparison test"),
                  P("confidence", "number", default=0.95)),
          description="""Ranking power of LGD estimates for an ordinal/continuous outcome.
gAUC = P(LGD_est_i > LGD_est_j | LGD_real_i > LGD_real_j) + 1/2 P(LGD_est_i = LGD_est_j | LGD_real_i > LGD_real_j),
computed over all pairs of facilities with different realised LGD (or realised LGD class when
`realised_bins` is given). 0.5 = no ranking power, 1 = perfect ordering. Equivalent to (1 + Somers' D)/2
with D the Somers' D of the estimate given the realised value; reduces to the ordinary AUC for a binary
outcome. Standard error from the U-statistic (Hoeffding/DeLong-type) decomposition; normal CI and z-test of
H0: gAUC = 0.5. If `initial_gauc` is given, the ECB comparison S = (gAUC_init − gAUC_curr)/se with
one-sided p = 1 − Φ(S) (H0: no deterioration).""",
          references=("ECB (2019), Instructions for reporting the validation results of internal models — "
                      "IRB Pillar I models for credit risk (generalised AUC for LGD)",
                      "DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44(3)",
                      "Somers (1962), American Sociological Review 27(6)"))
def lgd_gauc(ctx: RunContext, actual, predicted, realised_bins=None, initial_gauc=None,
             confidence=0.95) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted})
    if len(d) < 3:
        raise NotApplicable("Fewer than 3 complete observations.")
    real = _cut_bins(d["real"], realised_bins) if realised_bins else d["real"]
    notes = dropped_note(dropped)
    if realised_bins:
        notes.append(f"Realised LGD bucketed with edges {sorted(float(v) for v in realised_bins)} (right-closed).")
    return gauc_outcome(d["pred"], real, confidence, initial_gauc, notes, len(d))


@register("lgd.rank_correlation", "Pearson, Spearman and Kendall correlation of realised vs estimated LGD",
          "Discrimination", _LGD,
          params=(*_REAL_PRED, P("segment", required=False), P("confidence", "number", default=0.95)),
          description="""Association between realised and estimated LGD, overall and per segment:
Pearson r (linear, with Fisher-z CI), Spearman ρ (rank, Fisher-z CI with the Bonett–Wright variance
1.06/(n−3)) and Kendall τ-b (rank, tie-corrected). H0 for each p-value: no association (two-sided).
Higher values = estimates move with realised losses; rank measures are robust to the bimodal LGD shape.""",
          references=("BCBS (2005), Working Paper No. 14, Studies on the Validation of Internal Rating Systems",
                      "Bonett & Wright (2000), Psychometrika 65(1)",
                      "scipy.stats pearsonr / spearmanr / kendalltau"))
def lgd_rank_correlation(ctx: RunContext, actual, predicted, segment=None, confidence=0.95) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted}, {"seg": segment})
    return correlation_outcome(d, "seg" if segment else None, confidence, dropped)


def correlation_outcome(d: pd.DataFrame, by: str | None, confidence: float, dropped: int) -> Outcome:
    groups = [(g, d[d[by] == g]) for g in sorted_levels(d[by])] if by else []
    z = stats.norm.ppf(0.5 + confidence / 2)
    rows = []
    for g, sub in groups + [("ALL", d)]:
        n = len(sub)
        row = {"group": str(g), "n": n}
        if n >= 4 and sub["real"].nunique() > 1 and sub["pred"].nunique() > 1:
            pr = stats.pearsonr(sub["pred"], sub["real"])
            sp = stats.spearmanr(sub["pred"], sub["real"])
            kt = stats.kendalltau(sub["pred"], sub["real"])
            fz = np.arctanh(np.clip(pr.statistic, -0.999999, 0.999999))
            sz = np.arctanh(np.clip(sp.statistic, -0.999999, 0.999999))
            row |= {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                    "pearson_ci_low": float(np.tanh(fz - z / np.sqrt(n - 3))),
                    "pearson_ci_high": float(np.tanh(fz + z / np.sqrt(n - 3))),
                    "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
                    "spearman_ci_low": float(np.tanh(sz - z * np.sqrt(1.06 / (n - 3)))),
                    "spearman_ci_high": float(np.tanh(sz + z * np.sqrt(1.06 / (n - 3)))),
                    "kendall_tau_b": float(kt.statistic), "kendall_p": float(kt.pvalue)}
        rows.append(row)
    t = pd.DataFrame(rows)
    allrow = t.iloc[-1]
    if "pearson_r" not in t.columns or pd.isna(allrow.get("pearson_r")):
        raise NotApplicable("Need at least 4 observations with variation in both realised and estimated values.")
    summary = {k: allrow[k] for k in ("pearson_r", "pearson_p", "spearman_rho", "spearman_p",
                                       "kendall_tau_b", "kendall_p")} | {"n": int(allrow["n"])}
    return Outcome(summary, {"Correlations": t}, notes=dropped_note(dropped), rows_used=len(d))


# ── calibration / back-testing ─────────────────────────────────────────────

@register("lgd.loss_shortfall", "Loss shortfall and mean absolute deviation", "Calibration", _LGD,
          params=(*_REAL_PRED, P("ead", required=False, help="Exposure at default (weights); default 1 per facility"),
                  P("segment", required=False)),
          description="""Loss shortfall LS = 1 − Σ(LGD_est·EAD) / Σ(LGD_real·EAD): the share of realised loss
not covered by the estimates (LS > 0 = estimated losses below realised losses, LS < 0 = estimates exceed
realised). Also reported: coverage Σ(LGD_est·EAD)/Σ(LGD_real·EAD) = 1 − LS and the EAD-weighted mean absolute
deviation MAD = Σ|LGD_real − LGD_est|·EAD / ΣEAD. Overall and per segment. Descriptive, no H0.""",
          references=("BCBS (2005), Working Paper No. 14, Studies on the Validation of Internal Rating Systems "
                      "(loss shortfall, mean absolute deviation)",))
def lgd_loss_shortfall(ctx: RunContext, actual, predicted, ead=None, segment=None) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted, "w": ead}, {"seg": segment})
    if d.empty:
        raise NotApplicable("No complete observations.")
    if "w" not in d:
        d["w"] = 1.0
    if (d["w"] < 0).any():
        raise ValueError("EAD weights must be non-negative.")
    groups = [(g, d[d["seg"] == g]) for g in sorted_levels(d["seg"])] if segment else []
    rows = []
    for g, s in groups + [("ALL", d)]:
        real_loss, est_loss = float((s["real"] * s["w"]).sum()), float((s["pred"] * s["w"]).sum())
        cov = est_loss / real_loss if real_loss else float("nan")
        rows.append({"group": str(g), "n": len(s), "total_ead": float(s["w"].sum()),
                     "realised_loss": real_loss, "estimated_loss": est_loss,
                     "loss_shortfall": 1 - cov, "coverage_ratio": cov,
                     "MAD": _wavg((s["real"] - s["pred"]).abs(), s["w"])})
    t = pd.DataFrame(rows)
    a = t.iloc[-1]
    notes = dropped_note(dropped)
    if not ead:
        notes.append("No EAD given: every facility weighted 1.")
    return Outcome({"loss_shortfall": a["loss_shortfall"], "coverage_ratio": a["coverage_ratio"],
                    "MAD": a["MAD"], "n": int(a["n"])}, {"Loss shortfall": t}, notes=notes, rows_used=len(d))


_ALT = P("alternative", "string", default="greater", choices=("greater", "two-sided", "less"),
         help="H1 on realised − estimated: 'greater' = estimates too low (ECB direction)")


@register("lgd.backtest_ttest", "LGD back-test t-test (realised vs estimated, per pool and portfolio)",
          "Calibration", _LGD,
          params=(*_REAL_PRED, P("grade", required=False, help="LGD grade / pool"), _ALT),
          description="""ECB-style LGD back-test. For each pool and the portfolio, with d_i = realised LGD_i −
estimated LGD_i: T = √N · mean(d) / s_d, s_d² = Σ(d_i − mean d)²/(N−1), compared with Student t(N−1).
Default H0: estimated LGD ≥ realised LGD (mean d ≤ 0) vs H1: realised exceeds estimated (one-sided,
p = 1 − t_{N−1}(T)). When the estimate is constant within a pool this is the one-sample t-test of
realised LGDs against the pool LGD; otherwise the paired t-test. Assumes independent facilities and an
approximately normal mean (CLT); with few defaults per pool the p-values are approximate.""",
          references=("ECB (2019), Instructions for reporting the validation results of internal models — "
                      "IRB Pillar I models for credit risk (LGD back-testing t-test)",
                      "EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures"))
def lgd_backtest_ttest(ctx: RunContext, actual, predicted, grade=None, alternative="greater") -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted}, {"grade": grade})
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 complete observations.")
    t = ttest_rows(d, "grade" if grade else None, alternative)
    a = t.iloc[-1]
    notes = dropped_note(dropped)
    small = t[(t["group"] != "ALL") & (t["n"] < 30)]
    if len(small):
        notes.append(f"{len(small)} pool(s) with fewer than 30 observations: t-test relies on approximate normality.")
    if t["p_value"].isna().any():
        notes.append("p-value not computed where n < 2 or the differences have zero variance.")
    return Outcome({"mean_realised": a["mean_realised"], "mean_estimated": a["mean_estimated"],
                    "mean_difference": a["mean_difference"], "t_statistic": a["t_statistic"],
                    "p_value": a["p_value"], "n": int(a["n"]), "alternative": alternative},
                   {"LGD back-test by pool": t}, notes=notes, rows_used=len(d))


@register("lgd.wilcoxon", "Wilcoxon signed-rank and sign test of realised vs estimated LGD", "Calibration", _LGD,
          params=(*_REAL_PRED, P("grade", required=False), _ALT),
          description="""Non-parametric alternative to the LGD t-test, robust to the bimodal LGD distribution.
On d_i = realised − estimated, per pool and portfolio: Wilcoxon signed-rank test (H0: d symmetric about 0;
zero differences discarded, 'wilcox' method; exact distribution for small samples, normal approximation
otherwise as chosen by scipy) and the sign test (H0: P(d > 0) = 1/2, exact binomial on non-zero d).
Default alternative 'greater' = realised tends to exceed the estimate.""",
          references=("Wilcoxon (1945), Biometrics Bulletin 1(6)", "scipy.stats wilcoxon / binomtest"))
def lgd_wilcoxon(ctx: RunContext, actual, predicted, grade=None, alternative="greater") -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted}, {"grade": grade})
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 complete observations.")
    groups = [(g, d[d["grade"] == g]) for g in sorted_levels(d["grade"])] if grade else []
    rows = []
    for g, s in groups + [("ALL", d)]:
        diff = (s["real"] - s["pred"]).to_numpy()
        nz = diff[diff != 0]
        row = {"group": str(g), "n": len(diff), "n_nonzero": len(nz), "median_difference": float(np.median(diff)),
               "n_positive": int((nz > 0).sum()), "n_negative": int((nz < 0).sum()),
               "wilcoxon_statistic": np.nan, "wilcoxon_p": np.nan, "sign_test_p": np.nan}
        if len(nz) >= 1:
            row["sign_test_p"] = float(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5,
                                                       alternative=alternative).pvalue)
            w = stats.wilcoxon(nz, alternative=alternative, zero_method="wilcox")
            row["wilcoxon_statistic"], row["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
        rows.append(row)
    t = pd.DataFrame(rows)
    a = t.iloc[-1]
    return Outcome({"wilcoxon_statistic": a["wilcoxon_statistic"], "wilcoxon_p": a["wilcoxon_p"],
                    "sign_test_p": a["sign_test_p"], "median_difference": a["median_difference"],
                    "n": int(a["n"]), "alternative": alternative},
                   {"Signed-rank tests": t}, notes=dropped_note(dropped), rows_used=len(d))


@register("lgd.clar", "Cumulative LGD Accuracy Ratio (CLAR)", "Discrimination", _LGD,
          params=(*_REAL_PRED,),
          description="""CLAR (Ozdemir & Miu): treats each distinct estimated LGD as a grade. Facilities are ranked
by realised LGD (highest first) and cut into buckets with the same sizes as the estimated grades (highest
grade first). For the k worst grades (cumulative share c_k of facilities) the curve plots the share of all
facilities that are both in those k grades and among the c_k·N highest realised LGDs. CLAR = 2 × area under
this curve (trapezoid): 1 = perfect ordering, lower = misranking. Ties in realised LGD that straddle a bucket
boundary are split proportionally (expected value over random tie-breaking), so the result does not depend
on row order.""",
          references=("Ozdemir & Miu (2009), Basel II Implementation: A Guide to Developing and Validating a "
                      "Compliant, Internal Risk Rating System, McGraw-Hill",))
def lgd_clar(ctx: RunContext, actual, predicted) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted})
    n = len(d)
    if n < 2:
        raise NotApplicable("Fewer than 2 complete observations.")
    levels = np.sort(d["pred"].unique())[::-1]               # worst grade first
    if len(levels) < 2:
        raise NotApplicable("The estimate takes a single value; CLAR needs at least two grades.")
    sizes = d["pred"].value_counts().reindex(levels).to_numpy()
    cum = np.cumsum(sizes)
    grade_rank = pd.Series(np.arange(len(levels)), index=levels)[d["pred"]].to_numpy()
    real = d["real"].to_numpy()
    srt = np.sort(real)
    hi_pos = np.searchsorted(srt, real, side="right")
    start = n - hi_pos                                     # facilities with strictly higher realised LGD
    width = hi_pos - np.searchsorted(srt, real, side="left")
    rows = []
    for k, c in enumerate(cum):
        p_in = np.clip((c - start) / width, 0, 1)
        correct = float(p_in[grade_rank <= k].sum())
        rows.append({"grade": float(levels[k]), "n_grade": int(sizes[k]), "cum_share_obs": c / n,
                     "cum_correct": correct, "cum_share_correct": correct / n})
    t = pd.DataFrame(rows)
    xs = np.r_[0.0, t["cum_share_obs"].to_numpy()]
    ys = np.r_[0.0, t["cum_share_correct"].to_numpy()]
    area = float(np.sum(np.diff(xs) * (ys[1:] + ys[:-1]) / 2))
    return Outcome({"CLAR": 2 * area, "grades": len(levels), "n": n}, {"CLAR curve": t},
                   notes=dropped_note(dropped), rows_used=n)


@register("lgd.error_by_segment", "LGD accuracy: bias, MAE, RMSE, R² by segment", "Calibration", _LGD,
          params=(*_REAL_PRED, P("segment", required=False), P("ead", required=False, help="Optional EAD weights")),
          description="""Point-accuracy of facility-level LGD estimates, overall and per segment:
bias = mean(estimated − realised) (positive = conservative), MAE = mean|estimated − realised|,
RMSE = √mean((estimated − realised)²), R² = 1 − SSE/SST (around the segment mean of realised LGD).
With `ead`, EAD-weighted bias / MAE / RMSE are added. Descriptive, no H0.""",
          references=("BCBS (2005), Working Paper No. 14, Studies on the Validation of Internal Rating Systems",))
def lgd_error_by_segment(ctx: RunContext, actual, predicted, segment=None, ead=None) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted, "w": ead}, {"seg": segment})
    return error_outcome(d, "seg" if segment else None, dropped)


def error_outcome(d: pd.DataFrame, by: str | None, dropped: int) -> Outcome:
    if d.empty:
        raise NotApplicable("No complete observations.")
    groups = [(g, d[d[by] == g]) for g in sorted_levels(d[by])] if by else []
    rows = []
    for g, s in groups + [("ALL", d)]:
        e = s["pred"] - s["real"]
        sst = float(((s["real"] - s["real"].mean()) ** 2).sum())
        row = {"group": str(g), "n": len(s), "mean_realised": float(s["real"].mean()),
               "mean_estimated": float(s["pred"].mean()), "bias": float(e.mean()), "MAE": float(e.abs().mean()),
               "RMSE": float(np.sqrt((e ** 2).mean())),
               "R2": 1 - float((e ** 2).sum()) / sst if sst > 0 else float("nan")}
        if "w" in s:
            row |= {"weighted_bias": _wavg(e, s["w"]), "weighted_MAE": _wavg(e.abs(), s["w"]),
                    "weighted_RMSE": float(np.sqrt(_wavg(e ** 2, s["w"])))}
        rows.append(row)
    t = pd.DataFrame(rows)
    a = t.iloc[-1]
    return Outcome({"bias": a["bias"], "MAE": a["MAE"], "RMSE": a["RMSE"], "R2": a["R2"], "n": int(a["n"])},
                   {"Accuracy by segment": t}, notes=dropped_note(dropped), rows_used=len(d))


# ── distribution ───────────────────────────────────────────────────────────

@register("lgd.distribution", "Realised / estimated LGD distribution, mass at 0 and 1, bimodality",
          "Distribution", _LGD,
          params=(P("actual", help="Realised LGD"), P("predicted", required=False, help="Optional estimated LGD"),
                  P("tol", "number", default=1e-6, help="Tolerance for 'exactly 0' / 'exactly 1'"),
                  P("bins", "integer", default=20)),
          description="""Shape of the LGD distribution: moments, quantiles, share exactly at 0 and 1 (within `tol`),
share below 0 and above 1 (out of bounds, e.g. over-recoveries or costs exceeding recoveries), share within
10% of each bound (the typical U-shape), and the sample bimodality coefficient
BC = (g1² + 1) / (g2 + 3(n−1)²/((n−2)(n−3))) with bias-corrected skewness g1 and excess kurtosis g2
(BC of a uniform distribution is 5/9; larger values point to bimodality). Histogram on [0, 1] with
out-of-range counts. Descriptive.""",
          references=("Pfister et al. (2013), Good things peak in pairs: a note on the bimodality coefficient, "
                      "Frontiers in Psychology 4",
                      "Schuermann (2004), What do we know about loss given default?, Wharton FIC WP 04-01"))
def lgd_distribution(ctx: RunContext, actual, predicted=None, tol=1e-6, bins=20) -> Outcome:
    vals, notes = {}, []
    for name, col in (("realised", actual), ("estimated", predicted)):
        if col:
            s = num(ctx.df, col)
            notes += dropped_note(int(s.isna().sum()), f"missing {name} values")
            vals[name] = s.dropna()
    if len(vals["realised"]) < 2:
        raise NotApplicable("Fewer than 2 realised values.")
    t, h = distribution_tables(vals, 0.0, 1.0, tol, bins)
    r = t.iloc[0]
    return Outcome({"n": int(r["n"]), "mean": r["mean"], "median": r["median"], "share_at_0": r["share_at_0"],
                    "share_at_1": r["share_at_1"], "share_below_0": r["share_below_0"],
                    "share_above_1": r["share_above_1"], "bimodality_coefficient": r["bimodality_coefficient"]},
                   {"Distribution summary": t, "Histogram": h}, notes=notes,
                   rows_used=int(len(vals["realised"])))


# ── downturn, ELBE, in-default ─────────────────────────────────────────────

@register("lgd.downturn_comparison", "Downturn vs long-run realised LGD by period", "Calibration", _LGD,
          params=(P("actual", help="Realised LGD"), P("period", help="Default year / period"),
                  P("downturn_periods", "list", required=False, help="Periods identified as downturn"),
                  P("downturn_flag", required=False, help="0/1 column marking downturn defaults (alternative)"),
                  P("ead", required=False),
                  P("predicted", required=False, help="Estimated (downturn) LGD to compare with the realised levels")),
          description="""Realised LGD per default period, the long-run average (LRA) LGD and the realised LGD in the
downturn period(s). LRA = default-weighted mean of realised LGD over all periods (period-average and
EAD-weighted versions also shown). Downturn vs non-downturn: difference and ratio of means, Welch t-test
(H0: equal means) and Mann–Whitney U (H0: same distribution), both two-sided. With `predicted`, the
mean estimate is compared with the downturn realised mean and the LRA. Without downturn periods only the
per-period profile and the worst period are reported.""",
          references=("EBA/GL/2019/03, Guidelines for the estimation of LGD appropriate for an economic downturn",
                      "EBA/RTS/2018/04, RTS on the specification of the nature, severity and duration of an "
                      "economic downturn",
                      "EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures"))
def lgd_downturn_comparison(ctx: RunContext, actual, period, downturn_periods=None, downturn_flag=None,
                            ead=None, predicted=None) -> Outcome:
    dt = flag(ctx.df[downturn_flag], downturn_flag) if downturn_flag else None
    d, dropped = _frame(ctx, {"real": actual, "w": ead, "pred": predicted, "dt": dt}, {"period": period})
    if d.empty:
        raise NotApplicable("No complete observations.")
    if "w" not in d:
        d["w"] = 1.0
    if downturn_periods and not downturn_flag:
        dts = {str(v) for v in downturn_periods}
        d["dt"] = d["period"].map(lambda v: float(str(v) in dts))
        missing = dts - {str(v) for v in d["period"].unique()}
        if missing:
            raise ValueError(f"Downturn period(s) {sorted(missing)} not found in '{period}'.")
    rows = []
    for p in sorted_levels(d["period"]):
        s = d[d["period"] == p]
        row = {"period": str(p), "n_defaults": len(s), "mean_realised": float(s["real"].mean()),
               "ead_weighted_realised": _wavg(s["real"], s["w"])}
        if "pred" in s:
            row["mean_estimated"] = float(s["pred"].mean())
        if "dt" in s:
            row["downturn_share"] = float(s["dt"].mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    worst = t.loc[t["mean_realised"].idxmax()]
    summary = {"LRA_default_weighted": float(d["real"].mean()), "LRA_period_average": float(t["mean_realised"].mean()),
               "LRA_ead_weighted": _wavg(d["real"], d["w"]), "worst_period": worst["period"],
               "worst_period_mean": float(worst["mean_realised"]), "periods": len(t), "n": len(d)}
    notes = dropped_note(dropped)
    if "dt" in d:
        a, b = d.loc[d["dt"] == 1, "real"], d.loc[d["dt"] == 0, "real"]
        if len(a) < 2 or len(b) < 2:
            raise NotApplicable("Need at least 2 downturn and 2 non-downturn defaults.")
        wt = stats.ttest_ind(a, b, equal_var=False)
        mw = stats.mannwhitneyu(a, b, alternative="two-sided")
        summary |= {"downturn_mean": float(a.mean()), "non_downturn_mean": float(b.mean()),
                    "difference": float(a.mean() - b.mean()), "ratio": float(a.mean() / b.mean()) if b.mean() else np.nan,
                    "welch_t": float(wt.statistic), "welch_p": float(wt.pvalue), "mann_whitney_p": float(mw.pvalue),
                    "n_downturn": len(a), "n_non_downturn": len(b)}
        if "pred" in d:
            summary |= {"mean_estimated": float(d["pred"].mean()),
                        "estimated_minus_downturn_realised": float(d["pred"].mean() - a.mean())}
    else:
        notes.append("No downturn periods given: only the per-period profile and the worst period are reported.")
        if "pred" in d:
            summary["mean_estimated"] = float(d["pred"].mean())
    if len(t) < 5:
        notes.append(f"Only {len(t)} periods: the long-run average covers a short history.")
    return Outcome(summary, {"Realised LGD by period": t}, notes=notes, rows_used=len(d))


@register("lgd.elbe_backtest", "ELBE and LGD in-default vs realised LGD (defaulted exposures)", "Calibration", _LGD,
          params=(P("actual", help="Realised LGD of defaulted exposures (closed workouts)"),
                  P("predicted", help="ELBE (expected loss best estimate) at the reference date"),
                  P("lgd_in_default", required=False, help="LGD in-default (ELBE + add-on), optional"),
                  P("grade", required=False, help="Time-in-default band or pool"),
                  P("alternative", "string", default="two-sided", choices=("two-sided", "greater", "less"))),
          description="""Back-test of ELBE on defaulted exposures: per time-in-default band and overall, paired t-test
of d = realised LGD − ELBE (default two-sided, since ELBE is a best estimate; H0: mean d = 0), with MAE and
RMSE. With `lgd_in_default`: mean add-on (LGD in-default − ELBE), the number of facilities where
LGD in-default < ELBE (should not occur), and the share of realised LGDs exceeding LGD in-default.""",
          references=("EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted "
                      "exposures (ELBE and LGD in-default)",
                      "ECB (2024), ECB guide to internal models, credit risk chapter"))
def lgd_elbe_backtest(ctx: RunContext, actual, predicted, lgd_in_default=None, grade=None,
                      alternative="two-sided") -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted, "lid": lgd_in_default}, {"grade": grade})
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 complete observations.")
    t = ttest_rows(d, "grade" if grade else None, alternative)
    groups = [(g, d[d["grade"] == g]) for g in sorted_levels(d["grade"])] if grade else []
    extra = []
    for g, s in groups + [("ALL", d)]:
        e = s["real"] - s["pred"]
        row = {"MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}
        if "lid" in s:
            row |= {"mean_lgd_in_default": float(s["lid"].mean()), "mean_add_on": float((s["lid"] - s["pred"]).mean()),
                    "n_lgd_in_default_below_elbe": int((s["lid"] < s["pred"]).sum()),
                    "share_realised_above_lgd_in_default": float((s["real"] > s["lid"]).mean())}
        extra.append(row)
    t = pd.concat([t, pd.DataFrame(extra)], axis=1)
    a = t.iloc[-1]
    summary = {"mean_realised": a["mean_realised"], "mean_ELBE": a["mean_estimated"],
               "mean_difference": a["mean_difference"], "t_statistic": a["t_statistic"], "p_value": a["p_value"],
               "MAE": a["MAE"], "n": int(a["n"])}
    if "lid" in d:
        summary |= {"mean_add_on": a["mean_add_on"], "n_lgd_in_default_below_elbe": int(a["n_lgd_in_default_below_elbe"])}
    return Outcome(summary, {"ELBE back-test": t.rename(columns={"mean_estimated": "mean_ELBE"})},
                   notes=dropped_note(dropped), rows_used=len(d))


# ── workout analytics ──────────────────────────────────────────────────────

_UNIT_DAYS = {"days": 1.0, "months": 365.25 / 12, "years": 365.25}


@register("lgd.workout_length", "Recovery time / workout length statistics", "Workout", _LGD,
          params=(P("date", help="Default date"), P("end_date", help="Workout closure date (empty = open case)"),
                  P("as_of", "string", required=False,
                    help="Reference date (YYYY-MM-DD) for open cases: elapsed time and Kaplan–Meier censoring"),
                  P("segment", required=False),
                  P("unit", "string", default="months", choices=("days", "months", "years"))),
          description="""Workout length = closure date − default date for closed cases (mean, median, quartiles, P90,
max), the number of open cases and their time in default at `as_of`. With `as_of`, a Kaplan–Meier estimate of
the workout-length distribution treats open cases as censored at `as_of`, giving a median and quartiles that
are not biased towards short (already closed) workouts. Per segment and overall. Descriptive.""",
          references=("Kaplan & Meier (1958), JASA 53(282)",
                      "EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted "
                      "exposures (incomplete recovery processes)"))
def lgd_workout_length(ctx: RunContext, date, end_date, as_of=None, segment=None, unit="months") -> Outcome:
    from ask.validation.t_ifrs9 import km_curve, km_quantile
    df = ctx.df
    start = pd.to_datetime(df[date], errors="coerce")
    end = pd.to_datetime(df[end_date], errors="coerce")
    bad_end = int((df[end_date].notna() & end.isna()).sum())
    if bad_end:
        raise ValueError(f"{bad_end} non-empty '{end_date}' values are not dates.")
    seg = df[segment] if segment else pd.Series("ALL", index=df.index)
    ok = start.notna() & seg.notna()
    notes = dropped_note(int((~ok).sum()), "rows without a valid default date / segment")
    ref = pd.Timestamp(as_of) if as_of else None
    dur = (end - start) / pd.Timedelta(days=1) / _UNIT_DAYS[unit]
    neg = ok & end.notna() & (dur < 0)
    if neg.any():
        notes.append(f"{int(neg.sum())} cases with closure before default excluded.")
    ok &= ~neg
    closed = ok & end.notna()
    opened = ok & end.isna()
    elapsed = ((ref - start) / pd.Timedelta(days=1) / _UNIT_DAYS[unit]) if ref is not None else None
    if ref is not None:
        late = closed & (end > ref)
        if late.any():
            notes.append(f"{int(late.sum())} closure dates after as_of treated as open at as_of.")
            opened, closed = opened | late, closed & ~late
    groups = sorted_levels(seg[ok]) if segment else []
    rows = []
    for g in [*groups, "ALL"]:
        m = ok if g == "ALL" else ok & (seg == g)
        c = dur[m & closed].to_numpy()
        row = {"group": str(g), "n_closed": len(c), "n_open": int((m & opened).sum())}
        if len(c):
            q = np.quantile(c, [0.25, 0.5, 0.75, 0.9])
            row |= {"mean_closed": c.mean(), "p25_closed": q[0], "median_closed": q[1], "p75_closed": q[2],
                    "p90_closed": q[3], "max_closed": c.max()}
        if ref is not None:
            o = elapsed[m & opened].to_numpy()
            row["mean_elapsed_open"] = o.mean() if len(o) else np.nan
            tt = np.r_[c, o]
            ee = np.r_[np.ones(len(c)), np.zeros(len(o))]
            if len(c):
                curve = km_curve(tt, ee)
                row |= {"km_p25": km_quantile(curve, 0.25), "km_median": km_quantile(curve, 0.5),
                        "km_p75": km_quantile(curve, 0.75)}
        rows.append(row)
    t = pd.DataFrame(rows)
    a = t.iloc[-1]
    if a["n_closed"] == 0:
        raise NotApplicable("No closed workouts.")
    summary = {"n_closed": int(a["n_closed"]), "n_open": int(a["n_open"]), "mean_closed": a["mean_closed"],
               "median_closed": a["median_closed"], "unit": unit}
    if ref is not None:
        summary["km_median"] = a.get("km_median", np.nan)
        notes.append("Kaplan–Meier quantiles are NaN when the survival curve does not fall to that level "
                     "(too many open cases).")
    return Outcome(summary, {"Workout length": t}, notes=notes, rows_used=int(ok.sum()))


@register("lgd.cure_rate", "Cure rate by segment / period", "Workout", _LGD,
          params=(P("outcome", help="Workout outcome per default (cure flag 0/1, or category such as cure/liquidation)"),
                  P("cure_value", "string", required=False, help="Value of `outcome` meaning cured (for categories)"),
                  P("segment", required=False), P("period", required=False),
                  P("predicted", required=False, help="Estimated cure probability per default"),
                  P("actual", required=False, help="Realised LGD, to compare cured vs non-cured losses"),
                  P("confidence", "number", default=0.95)),
          description="""Observed cure rate = cured defaults / resolved defaults per segment × period and overall, with
Wilson confidence intervals. With `predicted` (cure-probability model), the mean predicted cure rate per group
and an exact two-sided binomial test (H0: observed cures ~ Binomial(n, mean predicted)). With `actual`, mean
realised LGD for cured vs non-cured defaults (the two-component structure of many LGD models). Open
(unresolved) cases should be excluded beforehand or given a missing outcome — they are dropped.""",
          references=("EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted "
                      "exposures (cure rates, return to non-defaulted status)",
                      "Wilson (1927), JASA 22(158)"))
def lgd_cure_rate(ctx: RunContext, outcome, cure_value=None, segment=None, period=None, predicted=None,
                  actual=None, confidence=0.95) -> Outcome:
    df = ctx.df
    s = df[outcome]
    if cure_value is not None:
        cured = flag(s, outcome, [cure_value])
    else:
        try:
            cured = flag(s, outcome)
        except ValueError:
            low = {str(v).strip().lower() for v in s.dropna().unique()}
            hit = [v for v in ("cure", "cured") if v in low]
            if not hit:
                raise ValueError(f"'{outcome}' is not a 0/1 flag; pass cure_value (values: {sorted(low)[:8]}).")
            cured = flag(s, outcome, hit)
    d = pd.DataFrame({"cured": cured})
    if segment:
        d["seg"] = df[segment]
    if period:
        d["per"] = df[period]
    if predicted:
        d["pred"] = num(df, predicted)
    if actual:
        d["real"] = num(df, actual)
    n0 = len(d)
    d = d.dropna(subset=[c for c in d.columns if c != "real"])
    dropped = n0 - len(d)
    if d.empty:
        raise NotApplicable("No resolved defaults.")
    keys = [c for c in ("seg", "per") if c in d]
    combos = sorted(d[keys].drop_duplicates().itertuples(index=False, name=None),
                    key=lambda r: tuple((str(type(v)), v) for v in r)) if keys else []
    rows = []
    for combo in [*combos, None]:
        sub = d if combo is None else d[(d[keys] == pd.Series(combo, index=keys)).all(axis=1)]
        n, k = len(sub), float(sub["cured"].sum())
        lo, hi = _wilson(k, n, confidence)
        row = {}
        for key, name in (("seg", "segment"), ("per", "period")):
            if key in keys:
                row[name] = "ALL" if combo is None else str(combo[keys.index(key)])
        row |= {"n": n, "cures": int(k), "cure_rate": k / n, "ci_low": lo, "ci_high": hi}
        if "pred" in sub:
            pm = float(sub["pred"].mean())
            row |= {"mean_predicted": pm,
                    "binomial_p": float(stats.binomtest(int(k), n, min(max(pm, 0.0), 1.0)).pvalue)}
        if "real" in sub:
            row |= {"mean_lgd_cured": float(sub.loc[sub["cured"] == 1, "real"].mean()),
                    "mean_lgd_not_cured": float(sub.loc[sub["cured"] == 0, "real"].mean())}
        rows.append(row)
    t = pd.DataFrame(rows)
    a = t.iloc[-1]
    summary = {"cure_rate": a["cure_rate"], "ci_low": a["ci_low"], "ci_high": a["ci_high"], "n": int(a["n"]),
               "cures": int(a["cures"])}
    if "pred" in d:
        summary |= {"mean_predicted": a["mean_predicted"], "binomial_p": a["binomial_p"]}
    if "real" in d:
        summary |= {"mean_lgd_cured": a["mean_lgd_cured"], "mean_lgd_not_cured": a["mean_lgd_not_cured"]}
    return Outcome(summary, {"Cure rate": t}, notes=dropped_note(dropped, "rows with missing outcome/inputs"),
                   rows_used=len(d))


@register("lgd.realised_from_cashflows", "Realised LGD replicated from workout cash flows", "Replication", _LGD,
          params=(P("id", help="Default / facility id in the defaults table (active table)"),
                  P("ead", help="EAD at default"), P("date", help="Default date"),
                  P("cashflows", "table", help="Loaded table of workout cash flows"),
                  P("discount_rate", "number", help="Annual discount rate, e.g. 0.05 (effective, compounding annually)"),
                  P("cf_id", "string", required=False, help="Id column in the cash-flow table (default: same as id)"),
                  P("cf_date", "string", default="date", help="Cash-flow date column"),
                  P("cf_amount", "string", default="amount", help="Cash-flow amount column"),
                  P("cf_type", "string", required=False,
                    help="Type column; without it positive amounts = recoveries, negative = costs"),
                  P("cost_values", "list", default=["cost", "costs", "expense", "expenses"],
                    help="Values of cf_type that are costs"),
                  P("day_count", "number", default=365.0, help="Days per year for discounting"),
                  P("actual", required=False, help="Reported realised LGD, to reconcile against"),
                  P("tolerance", "number", default=1e-4, help="Absolute LGD difference counted as a mismatch")),
          description="""Independent recomputation of realised (economic) LGD per default:
LGD = 1 − (PV(recoveries) − PV(costs)) / EAD, with PV(cf) = amount / (1 + r)^(days since default / day_count),
r = `discount_rate`. Cost amounts are taken in absolute value. Cash flows dated before the default date are
excluded and counted; defaults without cash flows get LGD = 1 (no recovery) and are counted. LGD is NOT
floored or capped, so values outside [0, 1] are visible. With `actual`, the reported LGD is reconciled:
difference, max absolute difference and the number of defaults differing by more than `tolerance`.
Portfolio LGD shown default-weighted and EAD-weighted (1 − ΣPV net recoveries / ΣEAD).""",
          references=("EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted "
                      "exposures (realised LGD, discounting, direct and indirect costs)",
                      "Regulation (EU) No 575/2013 (CRR), Article 5 (definition of economic loss)"))
def lgd_realised_from_cashflows(ctx: RunContext, id, ead, date, cashflows, discount_rate, cf_id=None,
                                cf_date="date", cf_amount="amount", cf_type=None, cost_values=None,
                                day_count=365.0, actual=None, tolerance=1e-4) -> Outcome:
    df, cf = ctx.df, ctx.tables[cashflows]
    cf_id = cf_id or id
    for c in [cf_id, cf_date, cf_amount] + ([cf_type] if cf_type else []):
        if c not in cf.columns:
            raise ValueError(f"Column '{c}' not in cash-flow table. Columns: {', '.join(map(str, cf.columns))}")
    if discount_rate <= -1:
        raise ValueError("discount_rate must be > -1.")
    d = pd.DataFrame({"id": df[id], "ead": num(df, ead), "ddate": pd.to_datetime(df[date], errors="coerce")})
    if actual:
        d["reported"] = num(df, actual)
    n0 = len(d)
    d = d.dropna(subset=["id", "ead", "ddate"])
    notes = dropped_note(n0 - len(d), "defaults with missing id / EAD / default date")
    if d["id"].duplicated().any():
        raise ValueError(f"'{id}' is not unique in the defaults table.")
    nonpos = d["ead"] <= 0
    if nonpos.any():
        notes.append(f"{int(nonpos.sum())} defaults with EAD <= 0 excluded.")
        d = d[~nonpos]
    if d.empty:
        raise NotApplicable("No usable defaults.")
    c = pd.DataFrame({"key": cf[cf_id].astype(str), "cdate": pd.to_datetime(cf[cf_date], errors="coerce"),
                      "amt": pd.to_numeric(cf[cf_amount], errors="coerce")})
    bad = c["cdate"].isna() | c["amt"].isna()
    if bad.any():
        notes.append(f"{int(bad.sum())} cash flows with missing/invalid date or amount excluded.")
    if cf_type:
        costs = {str(v).strip().lower() for v in (cost_values or [])}
        is_cost = cf[cf_type].map(lambda v: str(v).strip().lower() in costs).to_numpy()
        c["rec"] = np.where(is_cost, 0.0, c["amt"])
        c["cost"] = np.where(is_cost, c["amt"].abs(), 0.0)
    else:
        c["rec"] = c["amt"].clip(lower=0)
        c["cost"] = (-c["amt"]).clip(lower=0)
    c = c[~bad]
    d["key"] = d["id"].astype(str)
    unknown = int((~c["key"].isin(d["key"])).sum())
    if unknown:
        notes.append(f"{unknown} cash flows whose id is not among the usable defaults ignored.")
    c = c.merge(d[["key", "ddate"]], on="key", how="inner")
    days = (c["cdate"] - c["ddate"]) / pd.Timedelta(days=1)
    early = days < 0
    if early.any():
        notes.append(f"{int(early.sum())} cash flows dated before the default date excluded.")
    c = c[~early]
    df_ = (1 + discount_rate) ** (-(days[~early]) / day_count)
    c = c.assign(pv_rec=c["rec"] * df_, pv_cost=c["cost"] * df_)
    agg = c.groupby("key", sort=True).agg(pv_recoveries=("pv_rec", "sum"), pv_costs=("pv_cost", "sum"),
                                          n_cashflows=("pv_rec", "size"),
                                          undiscounted_recoveries=("rec", "sum"), undiscounted_costs=("cost", "sum"))
    out = d.merge(agg, left_on="key", right_index=True, how="left")
    none = out["n_cashflows"].isna()
    out[["pv_recoveries", "pv_costs", "n_cashflows", "undiscounted_recoveries", "undiscounted_costs"]] = \
        out[["pv_recoveries", "pv_costs", "n_cashflows", "undiscounted_recoveries", "undiscounted_costs"]].fillna(0)
    out["realised_lgd"] = 1 - (out["pv_recoveries"] - out["pv_costs"]) / out["ead"]
    out = out.sort_values("key", kind="mergesort")
    cols = ["id", "ead", "ddate", "n_cashflows", "undiscounted_recoveries", "undiscounted_costs",
            "pv_recoveries", "pv_costs", "realised_lgd"]
    summary = {"n_defaults": len(out), "mean_lgd": float(out["realised_lgd"].mean()),
               "ead_weighted_lgd": float(1 - (out["pv_recoveries"] - out["pv_costs"]).sum() / out["ead"].sum()),
               "n_without_cashflows": int(none.sum()), "n_lgd_below_0": int((out["realised_lgd"] < 0).sum()),
               "n_lgd_above_1": int((out["realised_lgd"] > 1).sum()), "discount_rate": discount_rate}
    if actual:
        out["reported_lgd"] = out["reported"]
        out["difference"] = out["realised_lgd"] - out["reported_lgd"]
        cols += ["reported_lgd", "difference"]
        has = out["reported_lgd"].notna()
        summary |= {"mean_difference": float(out.loc[has, "difference"].mean()),
                    "max_abs_difference": float(out.loc[has, "difference"].abs().max()),
                    "n_mismatch": int((out.loc[has, "difference"].abs() > tolerance).sum()),
                    "n_reconciled": int(has.sum())}
    tabs = {"Realised LGD per default": out[cols].rename(columns={"ddate": "default_date"}).reset_index(drop=True)}
    if actual:
        tabs["Largest differences"] = tabs["Realised LGD per default"].assign(
            abs_difference=lambda t: t["difference"].abs()).sort_values(
            ["abs_difference", "id"], ascending=[False, True], kind="mergesort").head(20).reset_index(drop=True)
    notes.append(f"Discounting: annual rate {discount_rate:g}, (1+r)^(-days/{day_count:g}) from the default date.")
    return Outcome(summary, tabs, notes=notes, rows_used=len(out))


@register("lgd.pool_homogeneity", "Heterogeneity across and homogeneity within LGD pools", "Discrimination", _LGD,
          params=(P("actual", help="Realised LGD"), P("grade", help="LGD pool / grade"),
                  P("predicted", required=False, help="Estimated LGD, used to order pools (else label order)")),
          description="""Do pools separate realised losses? Across all pools: Kruskal–Wallis H (H0: same distribution
in all pools), one-way ANOVA F (H0: equal means) and Levene/Brown–Forsythe (H0: equal variances). For each
pair of ADJACENT pools (ordered by mean estimated LGD, else by label): Mann–Whitney U and Welch t-test, two-sided
(H0: no difference) and one-sided (H1: the riskier pool has higher realised LGD), plus whether realised means
are in the same order as the pools. Per pool: n, mean estimate, mean / median / sd and coefficient of variation
of realised LGD (within-pool homogeneity).""",
          references=("ECB (2024), ECB guide to internal models, credit risk chapter (homogeneity of pools)",
                      "Kruskal & Wallis (1952), JASA 47(260)", "Brown & Forsythe (1974), JASA 69(346)"))
def lgd_pool_homogeneity(ctx: RunContext, actual, grade, predicted=None) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "pred": predicted}, {"grade": grade})
    return homogeneity_outcome(d, dropped)


def homogeneity_outcome(d: pd.DataFrame, dropped: int) -> Outcome:
    levels = sorted_levels(d["grade"])
    if "pred" in d:
        mp = d.groupby("grade")["pred"].mean()
        levels = sorted(levels, key=lambda g: (mp[g], str(g)))
    if len(levels) < 2:
        raise NotApplicable("Need at least two pools.")
    samples = [d.loc[d["grade"] == g, "real"].to_numpy() for g in levels]
    if any(len(s) < 2 for s in samples):
        raise NotApplicable("Every pool needs at least 2 observations.")
    pools = []
    for g, s in zip(levels, samples):
        row = {"pool": str(g), "n": len(s)}
        if "pred" in d:
            row["mean_estimated"] = float(d.loc[d["grade"] == g, "pred"].mean())
        m = s.mean()
        row |= {"mean_realised": m, "median_realised": float(np.median(s)), "sd_realised": s.std(ddof=1),
                "cv_realised": s.std(ddof=1) / m if m else np.nan}
        pools.append(row)
    adj = []
    for i in range(len(levels) - 1):
        a, b = samples[i], samples[i + 1]
        mw2 = stats.mannwhitneyu(a, b, alternative="two-sided")
        mw1 = stats.mannwhitneyu(a, b, alternative="less")
        t2 = stats.ttest_ind(a, b, equal_var=False)
        t1 = stats.ttest_ind(a, b, equal_var=False, alternative="less")
        adj.append({"lower_pool": str(levels[i]), "higher_pool": str(levels[i + 1]),
                    "mean_lower": a.mean(), "mean_higher": b.mean(), "means_ordered": bool(b.mean() > a.mean()),
                    "mann_whitney_p_two_sided": float(mw2.pvalue), "mann_whitney_p_one_sided": float(mw1.pvalue),
                    "welch_t": float(t2.statistic), "welch_p_two_sided": float(t2.pvalue),
                    "welch_p_one_sided": float(t1.pvalue)})
    kw = stats.kruskal(*samples)
    an = stats.f_oneway(*samples)
    lv = stats.levene(*samples, center="median")
    adj_t = pd.DataFrame(adj)
    return Outcome({"kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue), "anova_F": float(an.statistic),
                    "anova_p": float(an.pvalue), "levene_W": float(lv.statistic), "levene_p": float(lv.pvalue),
                    "pools": len(levels), "adjacent_pairs_not_ordered": int((~adj_t["means_ordered"]).sum()),
                    "n": len(d)},
                   {"Pools": pd.DataFrame(pools), "Adjacent pools": adj_t}, notes=dropped_note(dropped),
                   rows_used=len(d))


@register("lgd.incomplete_workouts", "Incomplete workouts: open cases and their effect on realised LGD",
          "Workout", _LGD,
          params=(P("actual", help="Realised LGD (final for closed cases, loss-to-date for open cases)"),
                  P("open_flag", help="1 = workout still open (incomplete), 0 = closed"),
                  P("period", required=False, help="Default period / vintage"),
                  P("ead", required=False),
                  P("predicted", required=False, help="Estimated final LGD of open cases (expected future recoveries)")),
          description="""MoC-relevant view of incomplete recovery processes: counts and share of open cases (overall and per
default period), mean realised LGD of closed cases vs loss-to-date of open cases, and the long-run average LGD
(i) on closed cases only, (ii) including open cases at their loss-to-date, (iii) including open cases at
their estimated final LGD (`predicted`), with the differences between them. Default-weighted, and
EAD-weighted when `ead` is given. Descriptive; quantifies the potential bias from excluding or including
incomplete workouts.""",
          references=("EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted "
                      "exposures (incomplete recovery processes, margin of conservatism)",))
def lgd_incomplete_workouts(ctx: RunContext, actual, open_flag, period=None, ead=None, predicted=None) -> Outcome:
    d, dropped = _frame(ctx, {"real": actual, "w": ead, "open": flag(ctx.df[open_flag], open_flag)},
                        {"period": period})
    if predicted:
        d["pred"] = num(ctx.df, predicted).loc[d.index]
    if "w" not in d:
        d["w"] = 1.0
    if d.empty:
        raise NotApplicable("No complete observations.")
    if "pred" in d:
        miss = int((d["open"].eq(1) & d["pred"].isna()).sum())
        if miss:
            raise ValueError(f"{miss} open cases have no estimated final LGD in '{predicted}'.")

    def block(s: pd.DataFrame) -> dict:
        cl, op = s[s["open"] == 0], s[s["open"] == 1]
        row = {"n": len(s), "n_closed": len(cl), "n_open": len(op), "share_open": len(op) / len(s) if len(s) else np.nan,
               "mean_lgd_closed": float(cl["real"].mean()) if len(cl) else np.nan,
               "mean_lgd_open_to_date": float(op["real"].mean()) if len(op) else np.nan,
               "lra_closed_only": float(cl["real"].mean()) if len(cl) else np.nan,
               "lra_open_at_loss_to_date": float(s["real"].mean())}
        if "pred" in s:
            final = np.where(s["open"] == 1, s["pred"], s["real"])
            row["lra_open_at_estimate"] = float(np.mean(final))
        if ead:
            row |= {"ead_lra_closed_only": _wavg(cl["real"], cl["w"]) if len(cl) else np.nan,
                    "ead_lra_open_at_loss_to_date": _wavg(s["real"], s["w"])}
            if "pred" in s:
                row["ead_lra_open_at_estimate"] = _wavg(np.where(s["open"] == 1, s["pred"], s["real"]), s["w"])
        return row

    rows = []
    if period:
        for p in sorted_levels(d["period"]):
            rows.append({"period": str(p)} | block(d[d["period"] == p]))
    rows.append(({"period": "ALL"} if period else {}) | block(d))
    t = pd.DataFrame(rows)
    a = t.iloc[-1]
    summary = {"n_closed": int(a["n_closed"]), "n_open": int(a["n_open"]), "share_open": a["share_open"],
               "lra_closed_only": a["lra_closed_only"], "lra_open_at_loss_to_date": a["lra_open_at_loss_to_date"],
               "difference_loss_to_date_vs_closed": a["lra_open_at_loss_to_date"] - a["lra_closed_only"]}
    if "pred" in d:
        summary |= {"lra_open_at_estimate": a["lra_open_at_estimate"],
                    "difference_estimate_vs_closed": a["lra_open_at_estimate"] - a["lra_closed_only"]}
    return Outcome(summary, {"Open vs closed workouts": t},
                   notes=dropped_note(dropped), rows_used=len(d))
