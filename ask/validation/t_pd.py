"""PD / rating-system validation: discrimination, calibration, rating-system structure,
default definition.

Conventions used by every test in this module
- `target` is the observed default flag (1 = default within the outcome horizon).
- `score` is oriented so that HIGHER = RISKIER. Pass higher_is_riskier=false for
  scorecards where a high score means low risk; the test then uses -score.
- `pd` is a probability in [0, 1]; values outside raise an error.
- Grades are ordered from best (lowest risk) to worst: the explicit `grade_order`
  if given, else by the mean `pd` of each grade when a pd column is available, else
  numerically when every grade is a number, else alphabetically. The order used is
  always written to the notes, because inversion counts, notches and migration
  bandwidths depend on it.
- Grade-level PD = the obligor-weighted mean of `pd` in the grade (ECB instructions:
  PD of the grade at the beginning of the observation period).
- Rows with a missing (or non-numeric) value in any input are dropped and counted.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import (NotApplicable, Outcome, P, RunContext, binary, complete,
                                 dropped_note, register, require_two_classes, split_samples)

# Several tests take a parameter named `pd` (the ROLES name for a PD column), which shadows
# pandas inside those functions; they build frames through these aliases instead.
_frame = pd.DataFrame

_DISC = ("pd", "ifrs9", "ews", "ml_classification", "aml")
_CAL = ("pd", "ifrs9", "ews", "ml_classification", "aml")
_RATING = ("pd", "ifrs9", "ews")
_DEFDEF = ("pd", "ifrs9")

_TGT = P("target", help="Observed default flag (1 = default within the horizon)")
_SCORE = (P("score", help="Score or PD; HIGHER = RISKIER unless higher_is_riskier=false"),
          P("higher_is_riskier", "boolean", default=True,
            help="Set false when a high score means LOW risk (e.g. scorecard points)"))
_PD = P("pd", help="Predicted probability of default in [0, 1]")
_GRADE_OPT = P("grade", required=False, help="Rating grade / pool; omit for portfolio level only")
_ORDER = P("grade_order", "list", required=False,
           help="Grades from best (lowest risk) to worst; default: by mean pd, else numeric, else alphabetic")
_CONF = P("confidence", "number", default=0.95, help="Confidence level of the intervals")
_SAMPLES = (
    P("sample", required=False, help="Sample column (dev / oot ...); omit when using `other`"),
    P("reference_value", "string", required=False),
    P("current_value", "string", required=False),
    P("other", "table", required=False, help="Second loaded table holding the current sample"),
)

_REF_DELONG = ("DeLong, DeLong & Clarke-Pearson (1988), Comparing the areas under two or more correlated "
               "receiver operating characteristic curves: a nonparametric approach, Biometrics 44(3)")
_REF_ECB = ("ECB Banking Supervision (2019), Instructions for reporting the validation results of internal "
            "models — IRB Pillar I models for credit risk")
_REF_BCBS14 = ("BCBS (2005), Studies on the validation of internal rating systems, Working Paper No. 14")
_REF_ENG = ("Engelmann, Hayden & Tasche (2003), Testing rating accuracy, Risk 16(1)")


# ── helpers ────────────────────────────────────────────────────────────────

def _bin(df: pd.DataFrame, col: str) -> pd.Series:
    """binary(), robust to pandas' string dtype (it is not `object` in pandas >= 3)."""
    if pd.api.types.is_string_dtype(df[col]) and df[col].dtype != object:
        df = df.assign(**{col: df[col].astype(object)})
    return binary(df, col)


def _load_frame(df: pd.DataFrame, cols, bin_cols=(), num_cols=()) -> tuple[pd.DataFrame, int]:
    """Complete cases of `cols`, with binary/numeric conversion; non-convertible values count as dropped."""
    cols = list(dict.fromkeys(c for c in cols if c))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing} are not in the table")
    sub, dropped = complete(df, cols)
    sub = sub.copy()
    for c in dict.fromkeys(bin_cols):
        sub[c] = _bin(sub, c)
    for c in dict.fromkeys(num_cols):
        sub[c] = pd.to_numeric(sub[c], errors="coerce").astype(float)
    conv = list(dict.fromkeys([*bin_cols, *num_cols]))
    if conv:
        bad = sub[conv].isna().any(axis=1) | np.isinf(sub[conv].to_numpy(float)).any(axis=1)
        dropped += int(bad.sum())
        sub = sub[~bad]
    return sub, dropped


def _load(ctx: RunContext, cols, bin_cols=(), num_cols=()):
    if ctx.df is None:
        raise ValueError("This test needs a data table")
    return _load_frame(ctx.df, cols, bin_cols, num_cols)


def _ys(sub, target, score, higher_is_riskier=True):
    y = sub[target].to_numpy(float).astype(int)
    s = sub[score].to_numpy(float)
    return y, (s if higher_is_riskier else -s)


def _check_pd(sub, pdc):
    p = sub[pdc]
    if ((p < 0) | (p > 1)).any():
        raise ValueError(f"'{pdc}' has values outside [0, 1]; pass a probability (not a percentage or score).")


def _zq(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    return float(stats.norm.ppf(0.5 + confidence / 2))


def _key(v) -> str:
    """Canonical text of a grade label, so 1, 1.0, '1' and '1.0' all match."""
    if isinstance(v, (float, np.floating)) and math.isfinite(v) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip()
    return re.sub(r"^(-?\d+)\.0+$", r"\1", s)


def _keys(s: pd.Series) -> pd.Series:
    return s.map(_key).astype(object)


def _order(present, grade_order=None, risk: pd.Series | None = None, keep_absent=False):
    """(ordered grade keys best -> worst, how the order was chosen)."""
    present = list(dict.fromkeys(present))
    if grade_order:
        order = list(dict.fromkeys(_key(g) for g in grade_order))
        missing = sorted(set(present) - set(order), key=str)
        if missing:
            raise ValueError(f"grade_order does not list grades {missing[:10]} found in the data")
        out = order if keep_absent else [g for g in order if g in set(present)]
        return out, "as given in grade_order"
    if risk is not None:
        r = risk.reindex(present)
        return sorted(present, key=lambda g: (r[g], g)), "by ascending mean PD of the grade"
    try:
        vals = {g: float(g) for g in present}
        return sorted(present, key=lambda g: vals[g]), "numerically ascending (assumed best -> worst)"
    except ValueError:
        return sorted(present), "alphabetically (assumed best -> worst; pass grade_order otherwise)"


def _order_note(order, how) -> str:
    shown = ", ".join(order[:40]) + (" ..." if len(order) > 40 else "")
    return f"Grade order (best -> worst) {how}: {shown}."


def _delong_components(y, s):
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    r_all = stats.rankdata(np.concatenate([pos, neg]))
    v10 = (r_all[:m] - stats.rankdata(pos)) / n          # share of non-defaults ranked below each default
    v01 = 1.0 - (r_all[m:] - stats.rankdata(neg)) / m    # share of defaults ranked above each non-default
    return v10, v01


def _auc_var(y, s):
    """(AUC, DeLong variance, n defaults, n non-defaults); ties count 1/2 (Mann–Whitney)."""
    v10, v01 = _delong_components(y, s)
    m, n = len(v10), len(v01)
    if m < 2 or n < 2:
        raise NotApplicable("The DeLong variance needs at least 2 defaults and 2 non-defaults.")
    return float(v10.mean()), float(v10.var(ddof=1) / m + v01.var(ddof=1) / n), m, n


def _hanley_mcneil_var(a, m, n):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float((a * (1 - a) + (m - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (m * n))


def _thin(t: pd.DataFrame, k: int = 201) -> pd.DataFrame:
    if len(t) <= k:
        return t.reset_index(drop=True)
    idx = np.unique(np.linspace(0, len(t) - 1, k).round().astype(int))
    return t.iloc[idx].reset_index(drop=True)


def _ref_cur(ctx, sample, reference_value, current_value, other):
    if other is not None:
        return ctx.df, ctx.tables[other], ctx.source_name or "reference", other
    if not sample:
        raise ValueError("Give either `sample` (a sample column) or `other` (a second table).")
    return split_samples(ctx.df, sample, reference_value, current_value)


def _grade_rows(sub, target, pdc, grade, grade_order, weight=None):
    """Per-grade N, D, DR, PD (+ exposure) table followed by a 'Portfolio' row."""
    notes = []
    agg = {"N": (target, "size"), "D": (target, "sum"), "PD": (pdc, "mean")}
    if weight:
        agg["exposure"] = (weight, "sum")
    parts = []
    if grade:
        keys = _keys(sub[grade])
        g = sub.groupby(keys, sort=True).agg(**agg)
        order, how = _order(list(g.index), grade_order, g["PD"])
        g = g.reindex(order)
        g.index.name = "grade"
        parts.append(g.reset_index())
        notes.append(_order_note(order, how))
    port = {"grade": "Portfolio", "N": len(sub), "D": sub[target].sum(), "PD": sub[pdc].mean()}
    if weight:
        port["exposure"] = sub[weight].sum()
    parts.append(pd.DataFrame([port]))
    t = pd.concat(parts, ignore_index=True)
    t["N"], t["D"] = t["N"].astype(int), t["D"].astype(int)
    t["DR"] = t["D"] / t["N"]
    cols = ["grade", "N", "D", "DR", "PD"] + (["exposure"] if weight else [])
    return t[cols], notes


def _cal_load(ctx, target, pdc, grade=None, extra=(), extra_num=()):
    sub, dropped = _load(ctx, [target, pdc, grade, *extra, *extra_num], [target], [pdc, *extra_num])
    _check_pd(sub, pdc)
    if sub.empty:
        raise NotApplicable("No complete rows.")
    return sub, dropped


def _fig_xy(traces, title, xlab, ylab):
    import plotly.graph_objects as go
    fig = go.Figure()
    for x, y, name, mode in traces:
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode=mode))
    fig.update_layout(title=title, xaxis_title=xlab, yaxis_title=ylab)
    return fig


# ── discrimination ─────────────────────────────────────────────────────────

@register("pd.auc", "AUC (ROC) with DeLong variance and confidence interval", "Discrimination", _DISC,
          params=(_TGT, *_SCORE, _CONF),
          description="""Area under the ROC curve, AUC = U / (|A|·|B|) with U the Mann–Whitney statistic over
all (default, non-default) pairs, ties counted 1/2 — the ECB definition. Variance by DeLong et al. (1988)
via placement values: s² = var(V10)/|A| + var(V01)/|B| (unbiased sample variances; identical to the ECB
annex formula). Also reports Gini = 2·AUC − 1, the Hanley–McNeil (1982) SE for comparison, the Wald test
of H0: AUC = 0.5 and the ROC curve points. Normal-approximation CI clipped to [0, 1]. Read: probability
that a random defaulter is ranked riskier than a random non-defaulter.""",
          references=(_REF_DELONG,
                      "Hanley & McNeil (1982), The meaning and use of the area under a ROC curve, Radiology 143",
                      _REF_ECB + ", Annex 3.1"))
def auc(ctx: RunContext, target, score, higher_is_riskier=True, confidence=0.95) -> Outcome:
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    a, var, m, n = _auc_var(y, s)
    z = _zq(confidence)
    se = math.sqrt(var)
    lo, hi = max(a - z * se, 0.0), min(a + z * se, 1.0)
    zs = (a - 0.5) / se if se > 0 else float("nan")
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    roc = _thin(pd.DataFrame({"threshold": thr if higher_is_riskier else -thr,
                              "false_positive_rate": fpr, "true_positive_rate": tpr}))
    notes = dropped_note(dropped)
    if se == 0:
        notes.append("Perfect (or zero) separation: DeLong variance is 0; interval and z-test not informative.")
    if m < 20:
        notes.append(f"Only {m} defaults: the normal-approximation interval may be unreliable.")
    fig = _fig_xy([(fpr, tpr, "ROC", "lines"), ([0, 1], [0, 1], "Random", "lines")],
                  f"ROC curve (AUC = {a:.4f})", "False positive rate", "True positive rate")
    return Outcome({"AUC": a, "AUC_SE_DeLong": se, "AUC_CI_lower": lo, "AUC_CI_upper": hi,
                    "Gini": 2 * a - 1, "Gini_CI_lower": 2 * lo - 1, "Gini_CI_upper": 2 * hi - 1,
                    "AUC_SE_Hanley_McNeil": math.sqrt(_hanley_mcneil_var(a, m, n)),
                    "z_vs_0.5": zs, "p_value_vs_0.5": float(2 * stats.norm.sf(abs(zs))) if se > 0 else float("nan"),
                    "confidence": confidence, "n": len(y), "defaults": m},
                   {"ROC curve": roc}, [fig], notes, rows_used=len(y))


def _cap(y, s):
    u, inv = np.unique(s, return_inverse=True)
    cnt = np.bincount(inv)[::-1].astype(float)
    dft = np.bincount(inv, weights=y)[::-1]
    pop = np.concatenate([[0.0], np.cumsum(cnt) / cnt.sum()])
    cap = np.concatenate([[0.0], np.cumsum(dft) / dft.sum()])
    return u[::-1], pop, cap


@register("pd.cap_accuracy_ratio", "CAP curve and Accuracy Ratio", "Discrimination", _DISC,
          params=(_TGT, *_SCORE),
          description="""Cumulative Accuracy Profile: obligors sorted from riskiest to safest; x = cumulative share
of the population, y = cumulative share of defaults captured (tied scores form one linear segment).
Accuracy Ratio AR = (area under model CAP − 0.5) / (area under perfect CAP − 0.5) = (A − 0.5)/(0.5·(1 − π)),
π = default rate. With this tie treatment AR = 2·AUC − 1 = Gini exactly (Engelmann et al. 2003); both are
reported. Also the share of defaults captured in the riskiest 5/10/20/30/50% of the population.""",
          references=(_REF_ENG, _REF_BCBS14,
                      "Sobehart, Keenan & Stein (2000), Benchmarking quantitative default risk models, Moody's"))
def cap_accuracy_ratio(ctx: RunContext, target, score, higher_is_riskier=True) -> Outcome:
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    thr, pop, cap = _cap(y, s)
    area = float(np.trapezoid(cap, pop))
    pi = y.mean()
    ar = (area - 0.5) / (0.5 * (1 - pi))
    a = _delong_components(y, s)[0].mean()
    cuts = [0.05, 0.10, 0.20, 0.30, 0.50]
    capt = pd.DataFrame({"population_share": cuts, "defaults_captured_share": np.interp(cuts, pop, cap)})
    pts = _thin(pd.DataFrame({"score_threshold": np.concatenate([[np.nan], thr if higher_is_riskier else -thr]),
                              "population_share": pop, "defaults_captured_share": cap}))
    fig = _fig_xy([(pop, cap, "Model", "lines"), ([0, 1], [0, 1], "Random", "lines"),
                   ([0, pi, 1], [0, 1, 1], "Perfect", "lines")],
                  f"CAP curve (AR = {ar:.4f})", "Share of population (riskiest first)", "Share of defaults")
    return Outcome({"accuracy_ratio": float(ar), "area_under_CAP": area, "AUC": float(a),
                    "gini_2AUC_minus_1": float(2 * a - 1),
                    "default_rate": float(pi), "n": len(y), "defaults": int(y.sum())},
                   {"Defaults captured": capt, "CAP curve": pts}, [fig], dropped_note(dropped), rows_used=len(y))


@register("pd.ks", "Kolmogorov–Smirnov statistic (max separation) with location", "Discrimination", _DISC,
          params=(_TGT, *_SCORE, P("bins", "integer", default=10, help="Quantile bins for the KS table")),
          description="""KS = max over thresholds t of |F_nondefault(t) − F_default(t)|, the largest vertical gap
between the empirical score CDFs of non-defaulters and defaulters, evaluated at every distinct score.
Reports the threshold where it occurs and the population share at or below it, the two-sample KS test
p-value (H0: both groups have the same score distribution; scipy ks_2samp — conservative with heavy ties),
and a quantile-bin KS table (bins on the score).""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 6", _REF_BCBS14,
                      "scipy.stats.ks_2samp"))
def ks(ctx: RunContext, target, score, higher_is_riskier=True, bins=10) -> Outcome:
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    u, inv = np.unique(s, return_inverse=True)
    d = np.bincount(inv, weights=y)
    g = np.bincount(inv) - d
    fd, fg = np.cumsum(d) / d.sum(), np.cumsum(g) / g.sum()
    gap = fg - fd
    i = int(np.argmax(np.abs(gap)))
    ksv = float(abs(gap[i]))
    p = float(stats.ks_2samp(s[y == 1], s[y == 0]).pvalue)
    thr = float(u[i] if higher_is_riskier else -u[i])
    b = pd.qcut(pd.Series(s), q=bins, duplicates="drop")
    t = (pd.DataFrame({"bin": b, "y": y}).groupby("bin", observed=True)["y"]
         .agg(n="size", defaults="sum").reset_index())
    t["non_defaults"] = t["n"] - t["defaults"]
    t["cum_defaults_pct"] = t["defaults"].cumsum() / t["defaults"].sum()
    t["cum_non_defaults_pct"] = t["non_defaults"].cumsum() / t["non_defaults"].sum()
    t["separation"] = (t["cum_non_defaults_pct"] - t["cum_defaults_pct"]).abs()
    t["bin"] = t["bin"].astype(str)
    side = "score <= threshold" if higher_is_riskier else "score >= threshold"
    share = float(np.cumsum(np.bincount(inv))[i] / len(s))
    return Outcome({"KS": ksv, "threshold": thr, "population_share_on_safe_side": share, "p_value": p,
                    "n": len(y), "defaults": int(y.sum())},
                   {"KS table (risk-ascending bins)": t}, [],
                   dropped_note(dropped) + [f"Population share counts obligors with {side}. Bins are on the "
                                            "risk-oriented score, safest first."],
                   rows_used=len(y))


@register("pd.rank_correlation", "Somers' D and Kendall tau-b between score and default", "Discrimination", _DISC,
          params=(_TGT, *_SCORE),
          description="""Somers' D(score | default) = (concordant − discordant) / pairs not tied on default; equals
the Gini / Accuracy Ratio. Also the asymmetric D(default | score) and Kendall tau-b (symmetric tie
correction), each with its p-value (H0: no association). Computed with scipy.stats.somersd / kendalltau.""",
          references=("Somers (1962), A new asymmetric measure of association for ordinal variables, ASR 27",
                      "Kendall (1945), The treatment of ties in ranking problems, Biometrika 33",
                      _REF_BCBS14))
def rank_correlation(ctx: RunContext, target, score, higher_is_riskier=True) -> Outcome:
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    d1 = stats.somersd(y, s)
    d2 = stats.somersd(s, y)
    kt = stats.kendalltau(s, y)
    t = pd.DataFrame([
        {"measure": "Somers' D(score | default)", "statistic": d1.statistic, "p_value": d1.pvalue},
        {"measure": "Somers' D(default | score)", "statistic": d2.statistic, "p_value": d2.pvalue},
        {"measure": "Kendall tau-b", "statistic": kt.statistic, "p_value": kt.pvalue}])
    return Outcome({"somers_d_score_given_default": float(d1.statistic), "somers_d_p_value": float(d1.pvalue),
                    "somers_d_default_given_score": float(d2.statistic),
                    "kendall_tau_b": float(kt.statistic), "kendall_p_value": float(kt.pvalue), "n": len(y)},
                   {"Rank correlations": t}, notes=dropped_note(dropped), rows_used=len(y))


@register("pd.information_value", "Information Value and WoE of grades / score bands", "Discrimination", _DISC,
          params=(_TGT, P("grade", required=False, help="Grades / score bands; omit to bin `score`"),
                  P("score", required=False, help="Score to bin into quantile bands when no grade is given"),
                  P("bins", "integer", default=10)),
          description="""Weight of Evidence per band WoE_k = ln(%non-defaults_k / %defaults_k) and
Information Value IV = Σ (%non-defaults_k − %defaults_k)·WoE_k (Siddiqi convention: positive WoE = safer
band). Bands are the grades, or quantile bins of the score. Bands with zero defaults or zero non-defaults
get 0.5 added to both counts (reported in the notes) so WoE stays finite.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 6",
                      "Kullback (1959), Information Theory and Statistics"))
def information_value(ctx: RunContext, target, grade=None, score=None, bins=10) -> Outcome:
    if not grade and not score:
        raise ValueError("Give `grade` (bands) or `score` (to be binned).")
    col = grade or score
    sub, dropped = _load(ctx, [target, col], [target], [] if grade else [score])
    y = sub[target].to_numpy(int)
    require_two_classes(y)
    if grade:
        band = _keys(sub[grade])
    else:
        band = pd.qcut(sub[score], q=bins, duplicates="drop").astype(str)
    t = (pd.DataFrame({"band": band.to_numpy(), "y": y}).groupby("band", sort=True)["y"]
         .agg(n="size", defaults="sum").reset_index())
    t["non_defaults"] = t["n"] - t["defaults"]
    adj = (t["defaults"] == 0) | (t["non_defaults"] == 0)
    d = t["defaults"] + 0.5 * adj
    g = t["non_defaults"] + 0.5 * adj
    t["pct_defaults"], t["pct_non_defaults"] = d / d.sum(), g / g.sum()
    t["WoE"] = np.log(t["pct_non_defaults"] / t["pct_defaults"])
    t["IV_contribution"] = (t["pct_non_defaults"] - t["pct_defaults"]) * t["WoE"]
    t["default_rate"] = t["defaults"] / t["n"]
    notes = dropped_note(dropped)
    if adj.any():
        notes.append(f"{int(adj.sum())} band(s) with an empty class: 0.5 added to both counts.")
    return Outcome({"IV": float(t["IV_contribution"].sum()), "bands": len(t), "n": len(y)},
                   {"WoE / IV by band": t}, notes=notes, rows_used=len(y))


@register("pd.divergence", "Divergence (separation of score means)", "Discrimination", _DISC,
          params=(_TGT, *_SCORE),
          description="""Divergence D = (μ_nondefault − μ_default)² / (½(σ²_nondefault + σ²_default)), the squared
standardised distance between the class means of the score; with Welch's t-test of equal means (H0: equal
means). Assumes the score is roughly normal within each class for the interpretation as a separation index.""",
          references=("Anderson (2007), The Credit Scoring Toolkit, OUP",
                      "Siddiqi (2006), Credit Risk Scorecards"))
def divergence(ctx: RunContext, target, score, higher_is_riskier=True) -> Outcome:
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    b, g = s[y == 1], s[y == 0]
    if len(b) < 2 or len(g) < 2:
        raise NotApplicable("Need at least 2 observations in each class.")
    vb, vg = b.var(ddof=1), g.var(ddof=1)
    if vb + vg == 0:
        raise NotApplicable("Score has no variance within classes.")
    dv = (g.mean() - b.mean()) ** 2 / (0.5 * (vb + vg))
    tt = stats.ttest_ind(b, g, equal_var=False)
    return Outcome({"divergence": float(dv), "mean_default": float(b.mean()), "mean_non_default": float(g.mean()),
                    "sd_default": float(np.sqrt(vb)), "sd_non_default": float(np.sqrt(vg)),
                    "welch_t": float(tt.statistic), "p_value": float(tt.pvalue), "n": len(y)},
                   notes=dropped_note(dropped) + ([] if higher_is_riskier else
                                                  ["Means are of the sign-flipped (risk-oriented) score."]),
                   rows_used=len(y))


@register("pd.pr_auc", "Precision–recall AUC and average precision", "Discrimination", _DISC,
          params=(_TGT, *_SCORE),
          description="""Average precision AP = Σ (R_k − R_{k−1})·P_k (sklearn, step-wise, no interpolation) and the
trapezoidal area under the precision–recall curve. The no-skill baseline is the default rate; lift = AP /
default rate. Also the maximum F1 over thresholds and its threshold. Informative for low-default/imbalanced
portfolios where ROC AUC is dominated by the non-defaults.""",
          references=("Davis & Goadrich (2006), The relationship between precision-recall and ROC curves, ICML",
                      "Saito & Rehmsmeier (2015), The precision-recall plot is more informative than the ROC plot "
                      "when evaluating binary classifiers on imbalanced datasets, PLoS ONE 10(3)"))
def pr_auc(ctx: RunContext, target, score, higher_is_riskier=True) -> Outcome:
    from sklearn.metrics import auc as _area, average_precision_score, precision_recall_curve
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    ap = float(average_precision_score(y, s))
    prec, rec, thr = precision_recall_curve(y, s)
    area = float(_area(rec, prec))
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0)[:-1]
    k = int(np.argmax(f1))
    base = float(y.mean())
    return Outcome({"average_precision": ap, "PR_AUC_trapezoid": area, "baseline_default_rate": base,
                    "lift_over_baseline": ap / base, "max_F1": float(f1[k]),
                    "max_F1_threshold": float(thr[k] if higher_is_riskier else -thr[k]), "n": len(y)},
                   {"PR curve": _thin(pd.DataFrame({"recall": rec, "precision": prec}))},
                   notes=dropped_note(dropped), rows_used=len(y))


@register("pd.delong_compare", "DeLong test: two scores' AUCs on the same sample", "Discrimination", _DISC,
          params=(_TGT, *_SCORE,
                  P("benchmark", help="Challenger / benchmark score column on the same rows"),
                  P("benchmark_higher_is_riskier", "boolean", default=True), _CONF),
          description="""Paired comparison of two correlated AUCs computed on the same obligors (model vs
challenger, benchmark or previous model version). DeLong et al. (1988): covariance of the two AUCs from
the placement values, var(Δ) = s²₁ + s²₂ − 2·s₁₂; z = (AUC₁ − AUC₂)/√var(Δ). H0: equal AUCs; two-sided
p-value, plus the one-sided p-value for H1: model AUC > benchmark AUC. Rows missing either score are
dropped so both AUCs use identical obligors.""",
          references=(_REF_DELONG, _REF_ECB + ", Annex 3.1 (covariance of two rating assignments)",
                      "Sun & Xu (2014), Fast implementation of DeLong's algorithm, IEEE Signal Processing Letters 21"))
def delong_compare(ctx: RunContext, target, score, benchmark, higher_is_riskier=True,
                   benchmark_higher_is_riskier=True, confidence=0.95) -> Outcome:
    sub, dropped = _load(ctx, [target, score, benchmark], [target], [score, benchmark])
    y, s1 = _ys(sub, target, score, higher_is_riskier)
    _, s2 = _ys(sub, target, benchmark, benchmark_higher_is_riskier)
    require_two_classes(y)
    a1, a2 = _delong_components(y, s1), _delong_components(y, s2)
    m, n = len(a1[0]), len(a1[1])
    if m < 2 or n < 2:
        raise NotApplicable("Need at least 2 defaults and 2 non-defaults.")
    S = np.cov(np.vstack([a1[0], a2[0]])) / m + np.cov(np.vstack([a1[1], a2[1]])) / n
    auc1, auc2 = float(a1[0].mean()), float(a2[0].mean())
    vd = float(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    diff = auc1 - auc2
    z = diff / math.sqrt(vd) if vd > 0 else float("nan")
    q = _zq(confidence)
    sd = math.sqrt(max(vd, 0))
    t = pd.DataFrame([{"score": score, "AUC": auc1, "SE": math.sqrt(S[0, 0]), "Gini": 2 * auc1 - 1},
                      {"score": benchmark, "AUC": auc2, "SE": math.sqrt(S[1, 1]), "Gini": 2 * auc2 - 1}])
    corr = S[0, 1] / math.sqrt(S[0, 0] * S[1, 1]) if S[0, 0] * S[1, 1] > 0 else float("nan")
    return Outcome({"AUC_model": auc1, "AUC_benchmark": auc2, "AUC_difference": diff, "SE_difference": sd,
                    "diff_CI_lower": diff - q * sd, "diff_CI_upper": diff + q * sd, "z": z,
                    "p_value_two_sided": float(2 * stats.norm.sf(abs(z))) if vd > 0 else float("nan"),
                    "p_value_model_better": float(stats.norm.sf(z)) if vd > 0 else float("nan"),
                    "AUC_correlation": float(corr), "n": len(y), "defaults": m},
                   {"AUCs": t}, notes=dropped_note(dropped), rows_used=len(y))


def _auc_by(ctx, target, score, by, higher_is_riskier, confidence, label):
    sub, dropped = _load(ctx, [target, score, by], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    keys = sub[by].to_numpy()
    z = _zq(confidence)
    rows = []
    for k in sorted(pd.unique(keys), key=lambda v: (str(type(v)), v)):
        msk = keys == k
        yy, ss = y[msk], s[msk]
        r = {label: str(k), "n": int(msk.sum()), "defaults": int(yy.sum())}
        if yy.sum() >= 2 and (len(yy) - yy.sum()) >= 2:
            a, v, _, _ = _auc_var(yy, ss)
            se = math.sqrt(v)
            r |= {"AUC": a, "SE": se, "AUC_CI_lower": max(a - z * se, 0), "AUC_CI_upper": min(a + z * se, 1),
                  "Gini": 2 * a - 1, "Gini_CI_lower": 2 * max(a - z * se, 0) - 1,
                  "Gini_CI_upper": 2 * min(a + z * se, 1) - 1, "comment": ""}
        else:
            r |= {"AUC": np.nan, "SE": np.nan, "comment": "fewer than 2 defaults or non-defaults"}
        rows.append(r)
    t = pd.DataFrame(rows)
    ok = t.dropna(subset=["AUC"])
    ok = ok[ok["SE"] > 0]
    if len(ok) == 0:
        raise NotApplicable(f"No {label} has at least 2 defaults and 2 non-defaults.")
    w = 1 / ok["SE"] ** 2
    pooled = float((w * ok["AUC"]).sum() / w.sum())
    q = float((w * (ok["AUC"] - pooled) ** 2).sum())
    dof = len(ok) - 1
    summ = {f"{label}s": len(t), f"{label}s_evaluated": len(ok),
            "min_AUC": float(ok["AUC"].min()), f"min_AUC_{label}": ok.loc[ok["AUC"].idxmin(), label],
            "max_AUC": float(ok["AUC"].max()), f"max_AUC_{label}": ok.loc[ok["AUC"].idxmax(), label],
            "heterogeneity_chi2": q, "heterogeneity_dof": dof,
            "heterogeneity_p_value": float(stats.chi2.sf(q, dof)) if dof > 0 else float("nan")}
    notes = dropped_note(dropped) + [
        f"Heterogeneity test: Q = Σ (AUC_g − pooled)²/SE_g², chi-square with (groups − 1) dof, H0: equal AUC in "
        f"all {label}s; assumes the {label}s are independent samples (disjoint obligors)."]
    return t, summ, notes, len(y)


_BY_DESC = """AUC with DeLong SE and normal CI (and Gini = 2·AUC − 1 with its CI) computed separately for
each {g}. Groups with fewer than 2 defaults or 2 non-defaults are listed without an AUC. Also a
heterogeneity test across groups: Q = Σ (AUC_g − AUC_pooled)²/SE_g² ~ chi²(G − 1) under H0 of a common AUC
(inverse-variance pooled AUC; groups treated as independent samples)."""


@register("pd.auc_by_segment", "AUC / Gini by segment with confidence intervals", "Discrimination", _DISC,
          params=(_TGT, *_SCORE, P("segment"), _CONF),
          description=_BY_DESC.format(g="segment"),
          references=(_REF_DELONG, "Cochran (1954), The combination of estimates from different experiments, "
                                   "Biometrics 10"))
def auc_by_segment(ctx: RunContext, target, score, segment, higher_is_riskier=True, confidence=0.95) -> Outcome:
    t, summ, notes, n = _auc_by(ctx, target, score, segment, higher_is_riskier, confidence, "segment")
    return Outcome(summ, {"AUC by segment": t}, notes=notes, rows_used=n)


@register("pd.auc_by_period", "AUC / Gini by period with confidence intervals", "Discrimination", _DISC,
          params=(_TGT, *_SCORE, P("period"), _CONF),
          description=_BY_DESC.format(g="period") + """ Obligors present in several periods make the
periods dependent; the heterogeneity p-value is then approximate.""",
          references=(_REF_DELONG, _REF_ECB))
def auc_by_period(ctx: RunContext, target, score, period, higher_is_riskier=True, confidence=0.95) -> Outcome:
    t, summ, notes, n = _auc_by(ctx, target, score, period, higher_is_riskier, confidence, "period")
    return Outcome(summ, {"AUC by period": t}, notes=notes, rows_used=n)


def _period_cdf(s, per):
    """ECB annex 3.1 multi-period aggregation: replace each score by its within-period empirical CDF."""
    f = pd.DataFrame({"s": s, "p": np.asarray(per)})
    g = f.groupby("p")["s"]
    return (g.rank(method="max") / g.transform("size")).to_numpy(float)


@register("pd.auc_change", "Change in AUC: current vs development / initial validation", "Discrimination", _DISC,
          params=(_TGT, *_SCORE, *_SAMPLES,
                  P("reference_auc", "number", required=False,
                    help="AUC at initial validation / development (ECB test treats it as deterministic)"),
                  P("reference_auc_se", "number", required=False,
                    help="SE of reference_auc; if given, an independent-samples z-test is used instead"),
                  P("period", required=False,
                    help="Period column: pool several periods with the ECB within-period rank transform"),
                  P("se_method", "string", default="delong", choices=("delong", "hanley_mcneil"))),
          description="""Tests whether discriminatory power has deteriorated.
(1) ECB test (reference_auc given, no reference_auc_se): S = (AUC_init − AUC_curr)/s with s the SE of the
current AUC, AUC_init treated as deterministic; p-value = 1 − Φ(S). H0: AUC_init ≤ AUC_curr (no
deterioration); a small p-value indicates a significant drop. The current sample is the `current_value`
rows of `sample`, else the `other` table, else the whole table.
(2) Independent samples (both samples in the data via `sample`/`other`, or reference_auc with
reference_auc_se): z = (AUC_ref − AUC_cur)/√(s²_ref + s²_cur); one-sided p = 1 − Φ(z), two-sided also given.
SEs by DeLong (default) or Hanley–McNeil. With `period`, scores are replaced by their within-period
empirical CDF before pooling periods (ECB annex 3.1 aggregation, e.g. three one-year periods).
For two different scores on the SAME obligors (paired, correlated AUCs) use pd.delong_compare.""",
          references=(_REF_ECB + ", section 2.5.4.1 and Annex 3.1", _REF_DELONG,
                      "Hanley & McNeil (1982), Radiology 143"))
def auc_change(ctx: RunContext, target, score, higher_is_riskier=True, sample=None, reference_value=None,
               current_value=None, other=None, reference_auc=None, reference_auc_se=None, period=None,
               se_method="delong") -> Outcome:
    notes = []

    def _one(frame, label):
        sub, dropped = _load_frame(frame, [target, score, period], [target], [score])
        if dropped:
            notes.append(f"{dropped} rows with missing inputs excluded from {label}.")
        y, s = _ys(sub, target, score, higher_is_riskier)
        if period:
            s = _period_cdf(s, sub[period])
        require_two_classes(y, f"target in {label}")
        a, v, m, n = _auc_var(y, s)
        if se_method == "hanley_mcneil":
            v = _hanley_mcneil_var(a, m, n)
        return {"sample": label, "n": len(y), "defaults": m, "AUC": a, "variance": v, "SE": math.sqrt(v),
                "Gini": 2 * a - 1}

    if reference_auc is not None:
        if not 0 <= reference_auc <= 1:
            raise ValueError("reference_auc must be in [0, 1]")
        if sample:
            _, frame, _, label = split_samples(ctx.df, sample, reference_value, current_value)
        elif other is not None:
            frame, label = ctx.tables[other], other
        else:
            frame, label = ctx.df, ctx.source_name or "current"
        cur = _one(frame, label)
        ref = {"sample": "reference (given)", "n": np.nan, "defaults": np.nan, "AUC": reference_auc,
               "variance": (reference_auc_se ** 2 if reference_auc_se is not None else 0.0),
               "SE": reference_auc_se if reference_auc_se is not None else 0.0, "Gini": 2 * reference_auc - 1}
        method = ("independent-samples z-test" if reference_auc_se is not None
                  else "ECB test (reference AUC deterministic)")
    else:
        rdf, cdf, rl, cl = _ref_cur(ctx, sample, reference_value, current_value, other)
        ref, cur = _one(rdf, rl), _one(cdf, cl)
        method = "independent-samples z-test"
    sd = math.sqrt(ref["variance"] + cur["variance"])
    stat = (ref["AUC"] - cur["AUC"]) / sd if sd > 0 else float("nan")
    if period:
        notes.append("Scores pooled across periods via the within-period empirical CDF (ECB annex 3.1).")
    if method.startswith("independent"):
        notes.append("Assumes reference and current samples are independent; overlapping obligors make the "
                     "test conservative-or-liberal in unknown direction.")
    return Outcome({"AUC_reference": ref["AUC"], "AUC_current": cur["AUC"], "AUC_change": cur["AUC"] - ref["AUC"],
                    "variance_current": cur["variance"], "statistic": stat,
                    "p_value_deterioration": float(stats.norm.sf(stat)) if sd > 0 else float("nan"),
                    "p_value_two_sided": float(2 * stats.norm.sf(abs(stat))) if sd > 0 else float("nan"),
                    "method": method, "se_method": se_method, "n_current": cur["n"]},
                   {"AUC by sample": pd.DataFrame([ref, cur])}, notes=notes,
                   rows_used=int(cur["n"] + (0 if pd.isna(ref["n"]) else ref["n"])))


@register("pd.gini_bootstrap", "Bootstrap confidence interval for Gini", "Discrimination", _DISC,
          params=(_TGT, *_SCORE, P("n_boot", "integer", default=1000, help="Bootstrap replicates (>= 100)"),
                  _CONF),
          description="""Stratified non-parametric bootstrap of Gini = 2·AUC − 1: defaults and non-defaults are
resampled with replacement separately (class sizes fixed), AUC recomputed by ranks on each replicate.
Reports the bootstrap SE, the percentile interval and the basic (reverse-percentile) interval, alongside
the analytical DeLong interval for comparison. Seeded via the run seed, so results are reproducible.""",
          references=("Efron & Tibshirani (1993), An Introduction to the Bootstrap, Chapman & Hall", _REF_DELONG))
def gini_bootstrap(ctx: RunContext, target, score, higher_is_riskier=True, n_boot=1000, confidence=0.95) -> Outcome:
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")
    sub, dropped = _load(ctx, [target, score], [target], [score])
    y, s = _ys(sub, target, score, higher_is_riskier)
    require_two_classes(y)
    a, v, m, n = _auc_var(y, s)
    pos, neg = s[y == 1], s[y == 0]
    rng = ctx.rng(0)
    chunk = max(1, 2_000_000 // (m + n))
    aucs = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        pi = rng.integers(0, m, size=(k, m))
        ni = rng.integers(0, n, size=(k, n))
        r = stats.rankdata(np.concatenate([pos[pi], neg[ni]], axis=1), axis=1)
        aucs[start:start + k] = (r[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    g = 2 * aucs - 1
    gini = 2 * a - 1
    alpha = 1 - confidence
    ql, qh = np.quantile(g, [alpha / 2, 1 - alpha / 2])
    z = _zq(confidence)
    se = math.sqrt(v)
    t = pd.DataFrame([
        {"interval": "bootstrap percentile", "lower": ql, "upper": qh},
        {"interval": "bootstrap basic", "lower": 2 * gini - qh, "upper": 2 * gini - ql},
        {"interval": "DeLong normal", "lower": 2 * max(a - z * se, 0) - 1, "upper": 2 * min(a + z * se, 1) - 1}])
    return Outcome({"Gini": gini, "bootstrap_SE": float(g.std(ddof=1)), "bootstrap_mean": float(g.mean()),
                    "percentile_CI_lower": float(ql), "percentile_CI_upper": float(qh),
                    "n_boot": n_boot, "confidence": confidence, "n": len(y), "defaults": m},
                   {"Gini intervals": t}, notes=dropped_note(dropped), rows_used=len(y))


@register("pd.rank_ordering", "Rank ordering of default rates across grades", "Discrimination", _RATING,
          params=(_TGT, P("grade"), P("pd", required=False, help="PD column, used to order grades"), _ORDER),
          description="""Default rate per grade in risk order (best -> worst) and checks of monotonicity:
number of adjacent inversions (DR_{k+1} < DR_k), number of pairwise inversions over all grade pairs k < l
(DR_l < DR_k) out of K(K−1)/2, Spearman rank correlation between grade position and DR (with p-value, H0: no
monotone association) and Kendall tau-b. Perfect rank ordering: 0 inversions, Spearman = 1.""",
          references=(_REF_BCBS14, "EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and the treatment "
                                    "of defaulted exposures"))
def rank_ordering(ctx: RunContext, target, grade, pd=None, grade_order=None) -> Outcome:
    sub, dropped = _load(ctx, [target, grade, pd], [target], [pd] if pd else [])
    keys = _keys(sub[grade])
    risk = sub[pd].groupby(keys).mean() if pd else None
    order, how = _order(list(keys.unique()), grade_order, risk)
    agg = {"N": (target, "size"), "D": (target, "sum")} | ({"mean_PD": (pd, "mean")} if pd else {})
    t = sub.groupby(keys).agg(**agg).reindex(order)
    t.index.name = "grade"
    t = t.reset_index()
    t["D"] = t["D"].astype(int)
    t["DR"] = t["D"] / t["N"]
    K = len(t)
    if K < 2:
        raise NotApplicable("Need at least two grades.")
    dr = t["DR"].to_numpy()
    t["inversion_vs_previous"] = np.concatenate([[False], np.diff(dr) < 0])
    i, j = np.triu_indices(K, 1)
    pair_inv = int((dr[j] < dr[i]).sum())
    pair_tie = int((dr[j] == dr[i]).sum())
    sp = stats.spearmanr(np.arange(K), dr) if K >= 3 else None
    kt = stats.kendalltau(np.arange(K), dr) if K >= 3 else None
    figs = []
    fig = _fig_xy([(t["grade"], t["DR"], "Default rate", "lines+markers")]
                  + ([(t["grade"], t["mean_PD"], "Mean PD", "lines+markers")] if pd else []),
                  "Default rate by grade", "Grade (best -> worst)", "Rate")
    figs.append(fig)
    return Outcome({"grades": K, "adjacent_inversions": int(t["inversion_vs_previous"].sum()),
                    "pairwise_inversions": pair_inv, "pairwise_ties": pair_tie, "pairs": K * (K - 1) // 2,
                    "spearman_rho": float(sp.statistic) if sp else float("nan"),
                    "spearman_p_value": float(sp.pvalue) if sp else float("nan"),
                    "kendall_tau_b": float(kt.statistic) if kt else float("nan"), "n": len(sub)},
                   {"Default rate by grade": t}, figs,
                   dropped_note(dropped) + [_order_note(order, how)], rows_used=len(sub))


# ── calibration ────────────────────────────────────────────────────────────

@register("pd.binomial_test", "Exact binomial test per grade and portfolio", "Calibration", _CAL,
          params=(_TGT, _PD, _GRADE_OPT, _ORDER),
          description="""For each grade and the portfolio: D defaults out of N obligors against the grade PD
(mean pd). Under H0 (PD correct, independent defaults) D ~ Binomial(N, PD). Reports the exact one-sided
p-value for under-estimation P(X ≥ D), for over-estimation P(X ≤ D), and the exact two-sided p-value
(scipy binomtest, minimum-likelihood method). Independence of defaults makes the test liberal (too many
rejections) when defaults are correlated — see pd.vasicek_test. At portfolio level the average PD is used
although obligor PDs differ (a heterogeneous-PD check is in pd.calibration_in_the_large).""",
          references=(_REF_BCBS14, "Tasche (2008), Validation of internal rating systems and PD estimates, in "
                                   "The Analytics of Risk Model Validation, Academic Press"))
def binomial_test(ctx: RunContext, target, pd, grade=None, grade_order=None) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, grade)
    t, notes = _grade_rows(sub, target, pd, grade, grade_order)
    t["expected_D"] = t["N"] * t["PD"]
    t["p_underestimation"] = [float(stats.binom.sf(d - 1, n, p)) for d, n, p in zip(t["D"], t["N"], t["PD"])]
    t["p_overestimation"] = [float(stats.binom.cdf(d, n, p)) for d, n, p in zip(t["D"], t["N"], t["PD"])]
    t["p_two_sided"] = [float(stats.binomtest(int(d), int(n), float(p)).pvalue)
                        for d, n, p in zip(t["D"], t["N"], t["PD"])]
    port = t.iloc[-1]
    return Outcome({"N": int(port["N"]), "D": int(port["D"]), "DR": float(port["DR"]), "PD": float(port["PD"]),
                    "p_underestimation": float(port["p_underestimation"]),
                    "p_two_sided": float(port["p_two_sided"]), "grades": len(t) - 1},
                   {"Binomial test by grade": t}, notes=dropped_note(dropped) + notes, rows_used=len(sub))


@register("pd.jeffreys_test", "Jeffreys test per grade and portfolio (ECB)", "Calibration", _CAL,
          params=(_TGT, _PD, _GRADE_OPT, _ORDER,
                  P("weight", required=False, help="Original exposure, reported per grade as the ECB requires"),
                  _CONF),
          description="""ECB back-testing test. With the Jeffreys prior Beta(½, ½), the posterior of the default
rate is Beta(D + ½, N − D + ½); the p-value is the posterior CDF evaluated at the grade PD:
p = Beta_cdf(PD; D + ½, N − D + ½). H0: the PD is greater than (or equal to) the true default rate (one-
sided); a small p-value indicates PD under-estimation. Run for every grade and the portfolio. Also the
two-sided p-value 2·min(p, 1 − p) and the equal-tailed Jeffreys interval for the default rate.""",
          references=(_REF_ECB + ", section 2.5.3.1",
                      "Brown, Cai & DasGupta (2001), Interval estimation for a binomial proportion, Statistical "
                      "Science 16(2)"))
def jeffreys_test(ctx: RunContext, target, pd, grade=None, grade_order=None, weight=None, confidence=0.95) -> Outcome:
    _zq(confidence)
    sub, dropped = _cal_load(ctx, target, pd, grade, extra_num=[weight] if weight else [])
    t, notes = _grade_rows(sub, target, pd, grade, grade_order, weight)
    a, b = t["D"] + 0.5, t["N"] - t["D"] + 0.5
    p = stats.beta.cdf(t["PD"], a, b)
    t["p_value"] = p
    t["p_two_sided"] = np.minimum(2 * np.minimum(p, 1 - p), 1.0)
    al = (1 - confidence) / 2
    t["DR_interval_lower"] = np.where(t["D"] == 0, 0.0, stats.beta.ppf(al, a, b))
    t["DR_interval_upper"] = np.where(t["D"] == t["N"], 1.0, stats.beta.ppf(1 - al, a, b))
    port = t.iloc[-1]
    return Outcome({"N": int(port["N"]), "D": int(port["D"]), "DR": float(port["DR"]), "PD": float(port["PD"]),
                    "p_value": float(port["p_value"]), "grades": len(t) - 1,
                    "min_grade_p_value": float(t["p_value"].iloc[:-1].min()) if len(t) > 1 else float("nan")},
                   {"Jeffreys test by grade": t}, notes=dropped_note(dropped) + notes +
                   ["Interval: Jeffreys equal-tailed with the endpoint convention of Brown et al. (2001)."],
                   rows_used=len(sub))


@register("pd.normal_test", "Normal-approximation (z) test of defaults per grade", "Calibration", _CAL,
          params=(_TGT, _PD, _GRADE_OPT, _ORDER),
          description="""z = (D − N·PD) / √(N·PD·(1 − PD)) per grade and portfolio — the normal approximation of
the binomial test. One-sided p = 1 − Φ(z) (H0: PD correct vs H1: PD too low) and two-sided p. The
approximation is poor when N·PD·(1 − PD) is small (column `npq`); use the exact binomial or Jeffreys test
then.""",
          references=(_REF_BCBS14,))
def normal_test(ctx: RunContext, target, pd, grade=None, grade_order=None) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, grade)
    t, notes = _grade_rows(sub, target, pd, grade, grade_order)
    t["npq"] = t["N"] * t["PD"] * (1 - t["PD"])
    t["z"] = np.where(t["npq"] > 0, (t["D"] - t["N"] * t["PD"]) / np.sqrt(t["npq"].where(t["npq"] > 0, 1)),
                      np.nan)
    t["p_underestimation"] = stats.norm.sf(t["z"])
    t["p_two_sided"] = 2 * stats.norm.sf(np.abs(t["z"]))
    port = t.iloc[-1]
    return Outcome({"z": float(port["z"]), "p_underestimation": float(port["p_underestimation"]),
                    "p_two_sided": float(port["p_two_sided"]), "N": int(port["N"]), "D": int(port["D"]),
                    "grades": len(t) - 1},
                   {"Normal test by grade": t}, notes=dropped_note(dropped) + notes, rows_used=len(sub))


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(120)
_GH_W = _GH_W / np.sqrt(2 * np.pi)


def _vasicek_p(d, n, pd_, rho):
    """P(D >= d) when defaults are conditionally independent given one N(0,1) factor (Vasicek)."""
    if pd_ <= 0:
        return 1.0 if d == 0 else 0.0
    if pd_ >= 1:
        return 1.0
    pz = stats.norm.cdf((stats.norm.ppf(pd_) + math.sqrt(rho) * _GH_X) / math.sqrt(1 - rho))
    return float(min(np.sum(_GH_W * stats.binom.sf(d - 1, n, pz)), 1.0))


def _vasicek_asym(dr, pd_, rho):
    if rho <= 0 or pd_ <= 0 or pd_ >= 1:
        return float("nan")
    if dr <= 0:
        return 1.0
    if dr >= 1:
        return 0.0
    return float(stats.norm.cdf((stats.norm.ppf(pd_) - math.sqrt(1 - rho) * stats.norm.ppf(dr)) / math.sqrt(rho)))


@register("pd.vasicek_test", "Vasicek / ASRF correlation-adjusted binomial test", "Calibration", _CAL,
          params=(_TGT, _PD, _GRADE_OPT, _ORDER,
                  P("rho", "number", default=0.12, help="Asset correlation of the one-factor (ASRF) model")),
          description="""Binomial test allowing for default correlation through the Vasicek one-factor model:
given the systematic factor Z ~ N(0,1), defaults are independent with
p(Z) = Φ((Φ⁻¹(PD) + √ρ·Z)/√(1 − ρ)). The finite-portfolio p-value P(D_obs or more defaults) =
E_Z[P(Binomial(N, p(Z)) ≥ D)] is computed by Gauss–Hermite quadrature (120 nodes). Also the infinitely-
granular (large-portfolio) p-value P(DR_∞ ≥ DR_obs) = Φ((Φ⁻¹(PD) − √(1−ρ)·Φ⁻¹(DR_obs))/√ρ). H0: PD correct;
one-sided against under-estimation. Results depend materially on ρ (stated in the notes).""",
          references=("Vasicek (2002), The distribution of loan portfolio value, Risk 15(12)", _REF_BCBS14,
                      "Tasche (2008), Validation of internal rating systems and PD estimates"))
def vasicek_test(ctx: RunContext, target, pd, grade=None, grade_order=None, rho=0.12) -> Outcome:
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    sub, dropped = _cal_load(ctx, target, pd, grade)
    t, notes = _grade_rows(sub, target, pd, grade, grade_order)
    t["p_value_finite_N"] = [_vasicek_p(int(d), int(n), float(p), rho) for d, n, p in zip(t["D"], t["N"], t["PD"])]
    t["p_value_asymptotic"] = [_vasicek_asym(float(r), float(p), rho) for r, p in zip(t["DR"], t["PD"])]
    t["p_value_binomial_independent"] = [float(stats.binom.sf(d - 1, n, p))
                                         for d, n, p in zip(t["D"], t["N"], t["PD"])]
    port = t.iloc[-1]
    return Outcome({"rho": rho, "p_value_finite_N": float(port["p_value_finite_N"]),
                    "p_value_asymptotic": float(port["p_value_asymptotic"]),
                    "p_value_independent": float(port["p_value_binomial_independent"]),
                    "N": int(port["N"]), "D": int(port["D"]), "grades": len(t) - 1},
                   {"Vasicek test by grade": t},
                   notes=dropped_note(dropped) + notes + [f"Asset correlation assumed rho = {rho}."],
                   rows_used=len(sub))


def _hl_groups(sub, pdc, grade, groups):
    if grade:
        return _keys(sub[grade]).to_numpy(), "grades"
    if groups < 2:
        raise ValueError("groups must be >= 2")
    return pd.qcut(sub[pdc], q=groups, duplicates="drop").astype(str).to_numpy(), "PD quantile groups"


@register("pd.hosmer_lemeshow", "Hosmer–Lemeshow goodness-of-fit test", "Calibration", _CAL,
          params=(_TGT, _PD, P("grade", required=False, help="Use grades as groups instead of PD deciles"),
                  P("groups", "integer", default=10, help="Number of PD quantile groups"),
                  P("dof", "string", default="g", choices=("g", "g-2"),
                    help="'g' for out-of-sample validation (default), 'g-2' for the development sample")),
          description="""HL = Σ_g (O_g − N_g·p̄_g)² / (N_g·p̄_g·(1 − p̄_g)) over groups g (PD quantile groups, or
grades), O_g observed defaults, p̄_g mean PD in the group. H0: PDs correctly calibrated in every group.
Degrees of freedom: g when the PDs were NOT fitted on this sample (validation / out-of-time — the PDs are
fixed, no parameters estimated), g − 2 when testing the logistic model on its own development sample
(Hosmer & Lemeshow's original reference distribution). Both p-values are in the table; `dof` picks the
headline. Quantile groups with tied PDs may merge (fewer groups).""",
          references=("Hosmer & Lemeshow (1980), Goodness of fit tests for the multiple logistic regression model, "
                      "Communications in Statistics A9", "Hosmer, Lemeshow & Sturdivant (2013), Applied Logistic "
                      "Regression, 3rd ed., Wiley, ch. 5", _REF_BCBS14))
def hosmer_lemeshow(ctx: RunContext, target, pd, grade=None, groups=10, dof="g") -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, grade)
    g, what = _hl_groups(sub, pd, grade, groups)
    t = sub.assign(_g=g).groupby("_g").agg(N=(target, "size"), D=(target, "sum"), PD=(pd, "mean")).reset_index()
    t = t.rename(columns={"_g": "group"}).sort_values("PD", kind="mergesort").reset_index(drop=True)
    t["expected_D"] = t["N"] * t["PD"]
    var = t["N"] * t["PD"] * (1 - t["PD"])
    bad = var <= 0
    t["contribution"] = np.where(bad, np.nan, (t["D"] - t["expected_D"]) ** 2 / var.where(~bad, 1))
    k = int((~bad).sum())
    hl = float(t["contribution"].sum())
    d_g, d_g2 = k, k - 2
    p_g = float(stats.chi2.sf(hl, d_g)) if d_g > 0 else float("nan")
    p_g2 = float(stats.chi2.sf(hl, d_g2)) if d_g2 > 0 else float("nan")
    notes = dropped_note(dropped) + [f"Groups: {what}."]
    if bad.any():
        notes.append(f"{int(bad.sum())} group(s) with mean PD of 0 or 1 excluded from the statistic.")
    if (t["expected_D"] < 5).any():
        notes.append("Some groups have fewer than 5 expected defaults; the chi-square approximation is rough.")
    head_dof, head_p = (d_g, p_g) if dof == "g" else (d_g2, p_g2)
    tests = _frame([{"dof_rule": "g (validation sample)", "dof": d_g, "p_value": p_g},
                          {"dof_rule": "g-2 (development sample)", "dof": d_g2, "p_value": p_g2}])
    return Outcome({"HL": hl, "dof": head_dof, "p_value": head_p, "groups": k, "n": len(sub)},
                   {"HL groups": t, "HL p-values": tests}, notes=notes, rows_used=len(sub))


@register("pd.chi_square_grades", "Pearson chi-square calibration test across grades", "Calibration", _CAL,
          params=(_TGT, _PD, P("grade"), _ORDER),
          description="""χ² = Σ_k [(D_k − N_k·PD_k)²/(N_k·PD_k) + ((N_k − D_k) − N_k(1 − PD_k))²/(N_k(1 − PD_k))]
= Σ_k (D_k − N_k·PD_k)²/(N_k·PD_k·(1 − PD_k)) over the K rating grades, with PD_k the grade PD. H0: all
grade PDs are correct simultaneously (independent defaults); χ² ~ chi²(K) because the grade PDs are fixed in
advance. Algebraically the Hosmer–Lemeshow statistic with grades as groups.""",
          references=(_REF_BCBS14, "Hosmer & Lemeshow (1980)"))
def chi_square_grades(ctx: RunContext, target, pd, grade, grade_order=None) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, grade)
    t, notes = _grade_rows(sub, target, pd, grade, grade_order)
    t = t.iloc[:-1].copy()
    var = t["N"] * t["PD"] * (1 - t["PD"])
    bad = var <= 0
    t["expected_D"] = t["N"] * t["PD"]
    t["contribution"] = np.where(bad, np.nan, (t["D"] - t["expected_D"]) ** 2 / var.where(~bad, 1))
    k = int((~bad).sum())
    if k == 0:
        raise NotApplicable("Every grade has PD 0 or 1.")
    chi = float(t["contribution"].sum())
    if bad.any():
        notes.append(f"{int(bad.sum())} grade(s) with PD 0 or 1 excluded.")
    if (t["expected_D"] < 5).any():
        notes.append("Some grades have fewer than 5 expected defaults; the chi-square approximation is rough.")
    return Outcome({"chi2": chi, "dof": k, "p_value": float(stats.chi2.sf(chi, k)), "n": len(sub)},
                   {"Chi-square by grade": t}, notes=dropped_note(dropped) + notes, rows_used=len(sub))


@register("pd.spiegelhalter", "Spiegelhalter z-test of calibration", "Calibration", _CAL,
          params=(_TGT, _PD),
          description="""Z = Σ (y_i − p_i)(1 − 2p_i) / √(Σ (1 − 2p_i)² p_i(1 − p_i)): standardised Brier score
under H0 that every obligor-level PD is correct (E[BS] and Var[BS] given the PDs). Two-sided p-value. Tests
obligor-level calibration without grouping.""",
          references=("Spiegelhalter (1986), Probabilistic prediction in patient management and clinical trials, "
                      "Statistics in Medicine 5", "Rauhmeier & Scheule (2005), Rating properties and their "
                      "implications for Basel II capital, Risk 18(3)"))
def spiegelhalter(ctx: RunContext, target, pd) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd)
    y, p = sub[target].to_numpy(float), sub[pd].to_numpy(float)
    den = np.sum((1 - 2 * p) ** 2 * p * (1 - p))
    if den <= 0:
        raise NotApplicable("All PDs are 0, 0.5 or 1; the statistic is undefined.")
    z = float(np.sum((y - p) * (1 - 2 * p)) / math.sqrt(den))
    return Outcome({"z": z, "p_value": float(2 * stats.norm.sf(abs(z))), "brier": float(np.mean((y - p) ** 2)),
                    "n": len(y)}, notes=dropped_note(dropped), rows_used=len(y))


@register("pd.brier", "Brier score with Murphy decomposition and skill score", "Calibration", _CAL,
          params=(_TGT, _PD, P("grade", required=False, help="Group by grade for the decomposition"),
                  P("bins", "integer", default=10, help="PD quantile bins when PDs are continuous")),
          description="""Brier score BS = mean (p_i − y_i)². Murphy (1973) decomposition over groups k (grades;
else the distinct PD values when there are ≤ 50; else PD quantile bins): BS = REL − RES + UNC (+ a within-
group residual when PDs vary inside a group), REL = Σ n_k(p̄_k − ō_k)²/N (reliability, lower is better),
RES = Σ n_k(ō_k − ō)²/N (resolution, higher is better), UNC = ō(1 − ō). Brier skill score BSS = 1 − BS/UNC
(reference: constant forecast at the observed default rate).""",
          references=("Brier (1950), Verification of forecasts expressed in terms of probability, Monthly "
                      "Weather Review 78", "Murphy (1973), A new vector partition of the probability score, "
                      "Journal of Applied Meteorology 12"))
def brier(ctx: RunContext, target, pd, grade=None, bins=10) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, grade)
    y, p = sub[target].to_numpy(float), sub[pd].to_numpy(float)
    if grade:
        g, what = _keys(sub[grade]).to_numpy(), "grades"
    elif sub[pd].nunique() <= 50:
        g, what = p, "distinct PD values"
    else:
        g, what = _qcut_labels(sub[pd], bins), "PD quantile bins"
    t = (_gyp(g, y, p).groupby("group").agg(n=("y", "size"), observed=("y", "mean"), mean_PD=("p", "mean"))
         .reset_index().sort_values("mean_PD", kind="mergesort"))
    N, ob = len(y), y.mean()
    bs = float(np.mean((p - y) ** 2))
    rel = float((t["n"] * (t["mean_PD"] - t["observed"]) ** 2).sum() / N)
    res = float((t["n"] * (t["observed"] - ob) ** 2).sum() / N)
    unc = float(ob * (1 - ob))
    t["group"] = t["group"].astype(str)
    return Outcome({"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
                    "within_group_residual": bs - (rel - res + unc),
                    "brier_skill_score": 1 - bs / unc if unc > 0 else float("nan"), "groups": len(t), "n": N},
                   {"Decomposition groups": t.reset_index(drop=True)},
                   notes=dropped_note(dropped) + [f"Decomposition groups: {what}."], rows_used=N)


def _qcut_labels(s, bins):
    return pd.qcut(s, q=bins, duplicates="drop").astype(str).to_numpy()


def _gyp(g, y, p):
    return pd.DataFrame({"group": g, "y": y, "p": p})


@register("pd.calibration_in_the_large", "Calibration in the large (observed vs expected defaults)",
          "Calibration", _CAL,
          params=(_TGT, _PD, P("weight", required=False, help="Exposure for exposure-weighted DR and PD")),
          description="""Portfolio-level comparison of the observed default rate with the average PD.
Reports O/E = D / Σ p_i and D − Σ p_i, and two tests of H0: PDs correct on average:
(i) z = (D − Σp_i)/√Σ p_i(1 − p_i), the normal approximation of the Poisson-binomial distribution of D
(respects heterogeneous obligor PDs, assumes independence); (ii) the exact binomial test at the average PD.
With `weight`, the exposure-weighted default rate and PD are also reported (descriptive, no test).""",
          references=(_REF_BCBS14, "Steyerberg (2019), Clinical Prediction Models, 2nd ed., Springer, ch. 15"))
def calibration_in_the_large(ctx: RunContext, target, pd, weight=None) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, extra_num=[weight] if weight else [])
    y, p = sub[target].to_numpy(float), sub[pd].to_numpy(float)
    n, d, e = len(y), int(y.sum()), float(p.sum())
    v = float(np.sum(p * (1 - p)))
    z = (d - e) / math.sqrt(v) if v > 0 else float("nan")
    bt = stats.binomtest(d, n, float(p.mean()))
    out = {"N": n, "D": d, "DR": d / n, "mean_PD": float(p.mean()), "expected_D": e,
           "O_over_E": d / e if e > 0 else float("nan"), "z": z,
           "p_value_z_two_sided": float(2 * stats.norm.sf(abs(z))) if v > 0 else float("nan"),
           "p_value_z_underestimation": float(stats.norm.sf(z)) if v > 0 else float("nan"),
           "p_value_binomial_two_sided": float(bt.pvalue)}
    if weight:
        w = sub[weight].to_numpy(float)
        if w.sum() <= 0:
            raise ValueError("weights sum to zero or less")
        out |= {"exposure_weighted_DR": float((w * y).sum() / w.sum()),
                "exposure_weighted_PD": float((w * p).sum() / w.sum())}
    return Outcome(out, notes=dropped_note(dropped), rows_used=n)


@register("pd.calibration_slope", "Calibration intercept and slope (logistic recalibration)", "Calibration", _CAL,
          params=(_TGT, _PD),
          description="""Logistic recalibration (Cox 1958): logit P(y=1) = a + b·logit(pd), fitted by maximum
likelihood (statsmodels Logit). Perfect calibration: a = 0, b = 1. Reports a and b with SEs, CIs and Wald
tests (H0: a = 0; H0: b = 1); calibration-in-the-large intercept a* from logit P(y=1) = a* + offset logit(pd)
(H0: a* = 0); and the likelihood-ratio test of (a, b) = (0, 1) jointly (chi² with 2 dof). b < 1: PDs too
extreme (over-fitted); b > 1: PDs not spread enough. PDs are clipped to [1e-9, 1 − 1e-9] for the logit.""",
          references=("Cox (1958), Two further applications of a model for binary regression, Biometrika 45",
                      "Van Calster et al. (2016), A calibration hierarchy for risk models was defined, Journal of "
                      "Clinical Epidemiology 74"))
def calibration_slope(ctx: RunContext, target, pd) -> Outcome:
    import statsmodels.api as sm
    sub, dropped = _cal_load(ctx, target, pd)
    y = sub[target].to_numpy(float)
    require_two_classes(y)
    raw = sub[pd].to_numpy(float)
    eps = 1e-9
    p = np.clip(raw, eps, 1 - eps)
    clipped = int((p != raw).sum())
    lp = np.log(p / (1 - p))
    if np.ptp(lp) == 0:
        raise NotApplicable("All PDs are equal; the slope is not identified.")
    try:
        full = sm.Logit(y, np.column_stack([np.ones_like(lp), lp])).fit(disp=0, maxiter=200)
        citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    except Exception as exc:
        raise NotApplicable(f"Logistic recalibration failed ({type(exc).__name__}: {exc}).")
    a, b = full.params
    sa, sb = full.bse
    ci = full.conf_int()
    ll0 = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    lr = 2 * (float(full.llf) - ll0)
    za, zb = a / sa, (b - 1) / sb
    t = _frame([
        {"parameter": "intercept a (H0: 0)", "estimate": a, "SE": sa, "CI_lower": ci[0, 0], "CI_upper": ci[0, 1],
         "z": za, "p_value": 2 * stats.norm.sf(abs(za))},
        {"parameter": "slope b (H0: 1)", "estimate": b, "SE": sb, "CI_lower": ci[1, 0], "CI_upper": ci[1, 1],
         "z": zb, "p_value": 2 * stats.norm.sf(abs(zb))},
        {"parameter": "calibration-in-the-large a* (H0: 0)", "estimate": citl.params[0], "SE": citl.bse[0],
         "CI_lower": citl.conf_int()[0, 0], "CI_upper": citl.conf_int()[0, 1],
         "z": citl.params[0] / citl.bse[0], "p_value": 2 * stats.norm.sf(abs(citl.params[0] / citl.bse[0]))}])
    notes = dropped_note(dropped) + ([f"{clipped} PDs clipped to [1e-9, 1-1e-9]."] if clipped else [])
    return Outcome({"intercept": float(a), "slope": float(b), "p_value_intercept_0": float(t.loc[0, "p_value"]),
                    "p_value_slope_1": float(t.loc[1, "p_value"]), "citl_intercept": float(citl.params[0]),
                    "LR_joint": lr, "p_value_joint": float(stats.chi2.sf(lr, 2)), "n": len(y)},
                   {"Recalibration parameters": t}, notes=notes, rows_used=len(y))


@register("pd.ece", "Expected calibration error (ECE / MCE) with reliability diagram", "Calibration", _CAL,
          params=(_TGT, _PD, P("bins", "integer", default=10),
                  P("strategy", "string", default="quantile", choices=("quantile", "uniform"))),
          description="""ECE = Σ_b (n_b/N)·|DR_b − PD̄_b| over PD bins (equal-frequency 'quantile' or equal-width
'uniform' on [0, 1]); MCE = max_b |DR_b − PD̄_b|. Descriptive (no test); bin-dependent. Includes the
reliability (calibration) diagram data.""",
          references=("Naeini, Cooper & Hauskrecht (2015), Obtaining well calibrated probabilities using Bayesian "
                      "binning, AAAI", "Guo et al. (2017), On calibration of modern neural networks, ICML"))
def ece(ctx: RunContext, target, pd, bins=10, strategy="quantile") -> Outcome:
    if bins < 1:
        raise ValueError("bins must be >= 1")
    sub, dropped = _cal_load(ctx, target, pd)
    y, p = sub[target].to_numpy(float), sub[pd].to_numpy(float)
    if strategy == "uniform":
        b = np.minimum((p * bins).astype(int), bins - 1)
        lab = np.array([f"[{i / bins:.3g}, {(i + 1) / bins:.3g})" for i in range(bins)])[b]
    else:
        lab = _qcut_labels(sub[pd], bins)
    t = (_gyp(lab, y, p).groupby("group").agg(n=("y", "size"), observed_DR=("y", "mean"),
                                                   mean_PD=("p", "mean"))
         .reset_index().sort_values("mean_PD", kind="mergesort").reset_index(drop=True))
    t["abs_gap"] = (t["observed_DR"] - t["mean_PD"]).abs()
    e = float((t["n"] * t["abs_gap"]).sum() / len(y))
    fig = _fig_xy([(t["mean_PD"], t["observed_DR"], "Observed", "lines+markers"),
                   ([0, float(t["mean_PD"].max())], [0, float(t["mean_PD"].max())], "Perfect", "lines")],
                  f"Reliability diagram (ECE = {e:.4f})", "Mean PD in bin", "Observed default rate")
    return Outcome({"ECE": e, "MCE": float(t["abs_gap"].max()), "bins": len(t), "strategy": strategy, "n": len(y)},
                   {"Reliability bins": t}, [fig], dropped_note(dropped), rows_used=len(y))


def _period_rows(sub, target, pdc, period, weight=None):
    agg = {"N": (target, "size"), "D": (target, "sum")}
    if pdc:
        agg["PD"] = (pdc, "mean")
    t = sub.groupby(period, sort=True).agg(**agg).reset_index()
    t["D"] = t["D"].astype(int)
    t["DR"] = t["D"] / t["N"]
    if weight:
        wy = (sub[weight] * sub[target]).groupby(sub[period], sort=True).sum()
        ws = sub[weight].groupby(sub[period], sort=True).sum()
        t["exposure_weighted_DR"] = (wy / ws.where(ws > 0)).to_numpy()
    t[period] = t[period].astype(str)
    return t.rename(columns={period: "period"})


@register("pd.multi_period_normal_test", "Multi-period normal test of PD calibration (Tasche)", "Calibration",
          _CAL, params=(_TGT, _PD, P("period")),
          description="""Across T periods (e.g. years) with default rate DR_t and average PD PD_t: statistic
S = Σ_t (DR_t − PD_t) / (√T·τ), τ² = (1/(T−1))·Σ_t ((DR_t − PD_t) − mean(DR − PD))². Under H0 (PDs correct,
period deviations independent and roughly normal) S is approximately N(0,1) (Student-t with T−1 dof for
small T is also reported); one-sided p = 1 − Φ(S) against PD under-estimation. Uses the empirical
dispersion of the yearly deviations, so it is robust to default correlation within a year. Also each
period's binomial z-statistic and one-sided p-value, and a sign test on the number of periods with DR > PD.
Traffic-light zones are not assigned.""",
          references=(_REF_BCBS14 + " (normal test, Tasche)",
                      "Blochwitz, Hohl, Tasche & Wehn (2004), Validating default probabilities on short time series, "
                      "Capital & Market Risk Insights, Federal Reserve Bank of Chicago"))
def multi_period_normal_test(ctx: RunContext, target, pd, period) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, extra=[period])
    t = _period_rows(sub, target, pd, period)
    T = len(t)
    if T < 2:
        raise NotApplicable("Need at least two periods.")
    dev = (t["DR"] - t["PD"]).to_numpy()
    tau = float(np.std(dev, ddof=1))
    s = float(dev.sum() / (math.sqrt(T) * tau)) if tau > 0 else float("nan")
    npq = t["N"] * t["PD"] * (1 - t["PD"])
    t["deviation"] = dev
    t["z_binomial"] = np.where(npq > 0, (t["D"] - t["N"] * t["PD"]) / np.sqrt(npq.where(npq > 0, 1)), np.nan)
    t["p_underestimation"] = stats.norm.sf(t["z_binomial"])
    above = int((dev > 0).sum())
    return Outcome({"periods": T, "statistic": s,
                    "p_value": float(stats.norm.sf(s)) if tau > 0 else float("nan"),
                    "p_value_t": float(stats.t.sf(s, T - 1)) if tau > 0 else float("nan"),
                    "mean_deviation": float(dev.mean()), "tau": tau, "periods_DR_above_PD": above,
                    "sign_test_p_value": float(stats.binomtest(above, T, 0.5, alternative="greater").pvalue)},
                   {"Periods": t}, notes=dropped_note(dropped) + (
                       ["Fewer than 5 periods: the normal approximation of S is rough."] if T < 5 else []),
                   rows_used=len(sub))


@register("pd.long_run_default_rate", "Long-run average default rate vs average PD", "Calibration", _CAL,
          params=(_TGT, _PD, P("period"), P("weight", required=False)),
          description="""Year-by-year (period-by-period) N, D, default rate and average PD, then the long-run
average default rate as the arithmetic mean of the period default rates (EBA GL on PD estimation), the
pooled default rate ΣD/ΣN, and the mean of period-average PDs; their difference and ratio. A t-test on the
period deviations DR_t − PD_t (H0: mean deviation 0; T − 1 dof) is given as a descriptive check. With
`weight`, the exposure-weighted default rate per period is added.""",
          references=("EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and the treatment of defaulted "
                      "exposures (long-run average default rate)", "Regulation (EU) No 575/2013 (CRR), Article 180"))
def long_run_default_rate(ctx: RunContext, target, pd, period, weight=None) -> Outcome:
    sub, dropped = _cal_load(ctx, target, pd, extra=[period], extra_num=[weight] if weight else [])
    t = _period_rows(sub, target, pd, period, weight)
    T = len(t)
    lra, apd = float(t["DR"].mean()), float(t["PD"].mean())
    dev = (t["DR"] - t["PD"]).to_numpy()
    tt = stats.ttest_1samp(dev, 0.0) if T >= 2 and np.std(dev) > 0 else None
    return Outcome({"periods": T, "long_run_average_DR": lra, "pooled_DR": float(t["D"].sum() / t["N"].sum()),
                    "average_PD": apd, "difference_LRA_DR_minus_PD": lra - apd,
                    "ratio_LRA_DR_to_PD": lra / apd if apd > 0 else float("nan"),
                    "t_statistic": float(tt.statistic) if tt else float("nan"),
                    "p_value_two_sided": float(tt.pvalue) if tt else float("nan")},
                   {"Default rate and PD by period": t}, notes=dropped_note(dropped), rows_used=len(sub))


# ── rating system ──────────────────────────────────────────────────────────

def _shares_stats(r: np.ndarray) -> dict:
    K = len(r)
    hhi = float(np.sum(r ** 2))
    cv = float(math.sqrt(K * np.sum((r - 1 / K) ** 2)))
    return {"HHI": hhi, "HHI_normalised": (hhi - 1 / K) / (1 - 1 / K) if K > 1 else float("nan"),
            "effective_grades": 1 / hhi if hhi > 0 else float("nan"), "CV": cv,
            "HI_ECB": 1 + math.log(hhi) / math.log(K) if K > 1 and hhi > 0 else float("nan")}


@register("pd.grade_concentration", "Grade distribution and concentration (HHI, ECB Herfindahl test)",
          "Rating system", _RATING,
          params=(P("grade"), P("weight", required=False, help="Exposure for exposure-weighted concentration"),
                  _ORDER, *_SAMPLES,
                  P("reference_cv", "number", required=False,
                    help="Coefficient of variation at initial validation (CV_init) for the ECB test")),
          description="""Share of obligors (and of exposure) per grade; Herfindahl–Hirschman index HHI = Σ R_i²,
normalised HHI* = (HHI − 1/K)/(1 − 1/K), effective number of grades 1/HHI. ECB: CV = √(K·Σ(R_i − 1/K)²) and
Herfindahl index HI = 1 + ln((CV² + 1)/K)/ln K (∈ [0, 1]). If a reference (development / initial validation)
sample or CV_init is given, the ECB test: p = 1 − Φ(√(K−1)·(CV_curr − CV_init)/√(CV_curr²·(0.5 + CV_curr²)));
H0: current concentration is not higher than at initial validation. K = number of grades (from grade_order
when given, including empty grades). The current sample is the `current_value` rows (or `other`), else the
whole table.""",
          references=(_REF_ECB + ", section 2.5.5.3", "Miller (1991), Asymptotic test statistics for coefficients "
                      "of variation, Communications in Statistics — Theory and Methods 20(10)"))
def grade_concentration(ctx: RunContext, grade, weight=None, grade_order=None, sample=None, reference_value=None,
                        current_value=None, other=None, reference_cv=None) -> Outcome:
    if sample or other is not None:
        rdf, cdf, rl, cl = _ref_cur(ctx, sample, reference_value, current_value, other)
    else:
        rdf, cdf, rl, cl = None, ctx.df, None, ctx.source_name or "current"
    cur, dropped = _load_frame(cdf, [grade, weight], num_cols=[weight] if weight else [])
    ck = _keys(cur[grade])
    ref = rk = None
    present = list(ck.unique())
    if rdf is not None:
        ref, _ = _load_frame(rdf, [grade, weight], num_cols=[weight] if weight else [])
        rk = _keys(ref[grade])
        present = list(dict.fromkeys(present + list(rk.unique())))
    order, how = _order(present, grade_order, keep_absent=True)
    K = len(order)
    if K < 2:
        raise NotApplicable("Need at least two grades.")
    cnt = ck.value_counts().reindex(order, fill_value=0).to_numpy(float)
    t = pd.DataFrame({"grade": order, "N": cnt.astype(int), "share": cnt / cnt.sum()})
    s_cur = _shares_stats(cnt / cnt.sum())
    summ = {"grades_K": K, "N": int(cnt.sum()), "HHI": s_cur["HHI"], "HHI_normalised": s_cur["HHI_normalised"],
            "effective_grades": s_cur["effective_grades"], "CV": s_cur["CV"], "HI_ECB": s_cur["HI_ECB"],
            "max_share": float(t["share"].max()), "max_share_grade": t.loc[t["share"].idxmax(), "grade"]}
    rows = [{"basis": f"obligors ({cl})"} | s_cur]
    if weight:
        e = cur.groupby(ck)[weight].sum().reindex(order, fill_value=0).to_numpy(float)
        if e.sum() <= 0:
            raise ValueError("Exposures sum to zero or less")
        t["exposure"], t["exposure_share"] = e, e / e.sum()
        s_e = _shares_stats(e / e.sum())
        rows.append({"basis": f"exposure ({cl})"} | s_e)
        summ |= {"HHI_exposure": s_e["HHI"], "HI_ECB_exposure": s_e["HI_ECB"]}
    cv_init = reference_cv
    if ref is not None:
        rc = rk.value_counts().reindex(order, fill_value=0).to_numpy(float)
        t["reference_N"], t["reference_share"] = rc.astype(int), rc / rc.sum()
        s_r = _shares_stats(rc / rc.sum())
        rows.append({"basis": f"obligors ({rl})"} | s_r)
        if cv_init is None:
            cv_init = s_r["CV"]
        summ["HI_ECB_reference"] = s_r["HI_ECB"]
    if cv_init is not None:
        cvc = s_cur["CV"]
        den = math.sqrt(cvc ** 2 * (0.5 + cvc ** 2))
        z = math.sqrt(K - 1) * (cvc - cv_init) / den if den > 0 else float("nan")
        summ |= {"CV_init": float(cv_init), "z": z,
                 "p_value": float(stats.norm.sf(z)) if den > 0 else float("nan")}
    return Outcome(summ, {"Grade distribution": t, "Concentration measures": pd.DataFrame(rows)},
                   notes=dropped_note(dropped) + [_order_note(order, how)], rows_used=len(cur))


@register("pd.migration_matrix", "Migration (transition) matrix and stability metrics", "Rating system", _RATING,
          params=(P("grade", help="Grade at the start (or the only grade column with id/period)"),
                  P("grade_to", required=False, help="Grade at the end, on the same row"),
                  P("id", required=False), P("period", required=False),
                  P("from_period", "string", required=False), P("to_period", "string", required=False), _ORDER),
          description="""Transition counts N_ij and frequencies p_ij = N_ij/N_i from start grade i to end status j,
either from two columns on the same row (grade -> grade_to) or from a panel (id, period, grade) between two
periods. End values that are not rating grades (default, exit, missing) are kept as extra columns and
excluded from the K×K metrics. Metrics on the K×K block: share on the diagonal, upgrades (to a better
grade) and downgrades, shares within ±1 and ±2 notches; ECB matrix weighted bandwidth
MWB_upper = Σ_{i<K} Σ_{j>i} |i−j|·N_i·p_ij / M_up, MWB_lower analogously for j < i, with
M_up = Σ_{i<K} max(|i−K|, |i−1|)·N_i·Σ_{j>i} p_ij (and M_low over j < i); Shorrocks mobility
(K − trace P)/(K − 1) and the singular-value mobility index M_SVD = mean singular value of (P − I) on the
row-normalised K×K matrix. Also the ECB z-tests of monotone off-diagonal frequencies,
z_ij = (p_{i,j±1} − p_ij)/√((p_ij(1−p_ij) + p_{i,j±1}(1−p_{i,j±1}) + 2·p_ij·p_{i,j±1})/N_i) with p-value
Φ(z_ij) (j+1 for j < i, j−1 for j > i). Grade order best -> worst: 'upper' (j > i) = downgrades.""",
          references=(_REF_ECB + ", sections 2.5.5.1–2.5.5.2",
                      "Jafry & Schuermann (2004), Measurement, estimation and comparison of credit migration "
                      "matrices, Journal of Banking & Finance 28", "Shorrocks (1978), The measurement of mobility, "
                      "Econometrica 46"))
def migration_matrix(ctx: RunContext, grade, grade_to=None, id=None, period=None, from_period=None,
                     to_period=None, grade_order=None) -> Outcome:
    df = ctx.df
    notes = []
    if grade_to:
        sub, dropped = complete(df, [grade])
        start = _keys(sub[grade])
        end = df.loc[sub.index, grade_to].map(lambda v: "<no end grade>" if pd.isna(v) else _key(v)).astype(object)
    else:
        if not (id and period):
            raise ValueError("Give `grade_to`, or `id` and `period` (with from_period/to_period).")
        sub, dropped = complete(df, [id, period, grade])
        pk = _keys(sub[period])
        periods = sorted(pk.unique(), key=lambda v: (str(type(v)), v))
        if from_period is None or to_period is None:
            if len(periods) != 2:
                raise ValueError(f"'{period}' has {len(periods)} values; pass from_period and to_period.")
            from_period, to_period = periods
        fp, tp = _key(from_period), _key(to_period)
        if fp not in set(pk) or tp not in set(pk):
            raise ValueError(f"Periods {from_period}/{to_period} not found; values: {periods[:12]}")
        a, b = sub[pk == fp], sub[pk == tp]
        for frame, lab in ((a, fp), (b, tp)):
            if frame[id].duplicated().any():
                raise ValueError(f"'{id}' is not unique within period {lab}.")
        m = a[[id, grade]].merge(b[[id, grade]], on=id, how="left", suffixes=("_from", "_to"))
        start = _keys(m[f"{grade}_from"])
        end = m[f"{grade}_to"].map(lambda v: "<not observed at end>" if pd.isna(v) else _key(v)).astype(object)
        new = int((~b[id].isin(a[id])).sum())
        if new:
            notes.append(f"{new} ids present only at {tp} (new obligors) are not in the matrix.")
        notes.append(f"Migration from period {fp} to {tp}.")
    order, how = _order(list(start.unique()), grade_order, keep_absent=bool(grade_order))
    K = len(order)
    if K < 2:
        raise NotApplicable("Need at least two grades.")
    statuses = sorted(set(end) - set(order), key=str)
    ct = pd.crosstab(start, end).reindex(index=order, columns=order + statuses, fill_value=0)
    C = ct.to_numpy(float)
    Ni = C.sum(axis=1)
    NK = C[:, :K]
    idx = np.arange(1, K + 1)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    dist = np.abs(I - J)
    up, lo = J > I, J < I
    mx = np.maximum(np.abs(idx - K), np.abs(idx - 1))[:, None]
    m_up, m_lo = float((mx * NK * up).sum()), float((mx * NK * lo).sum())
    mwb_up = float((dist * NK * up).sum()) / m_up if m_up > 0 else float("nan")
    mwb_lo = float((dist * NK * lo).sum()) / m_lo if m_lo > 0 else float("nan")
    tot = NK.sum()
    nonempty = NK.sum(axis=1) > 0
    Pm = np.where(nonempty[:, None], NK / np.where(nonempty, NK.sum(axis=1), 1)[:, None], np.eye(K))
    kp = int(nonempty.sum())
    shorrocks = (kp - np.trace(Pm[nonempty][:, nonempty])) / (kp - 1) if kp > 1 else float("nan")
    msvd = float(np.linalg.svd(Pm - np.eye(K), compute_uv=False).mean())
    metrics = {"obligors": int(C.sum()), "obligors_ending_in_grades": int(tot),
               "share_diagonal": float(np.trace(NK) / tot) if tot else float("nan"),
               "share_downgrades": float((NK * up).sum() / tot) if tot else float("nan"),
               "share_upgrades": float((NK * lo).sum() / tot) if tot else float("nan"),
               "share_within_1_notch": float((NK * (dist <= 1)).sum() / tot) if tot else float("nan"),
               "share_within_2_notches": float((NK * (dist <= 2)).sum() / tot) if tot else float("nan"),
               "share_beyond_2_notches": float((NK * (dist > 2)).sum() / tot) if tot else float("nan"),
               "MWB_upper_downgrades": mwb_up, "MWB_lower_upgrades": mwb_lo,
               "mobility_shorrocks": float(shorrocks), "mobility_svd": msvd,
               "share_to_non_grade_status": float(C[:, K:].sum() / C.sum()) if C.sum() else float("nan")}
    # ECB z-tests of monotone off-diagonal frequencies
    Pz = C[:, :K] / np.where(Ni > 0, Ni, np.nan)[:, None]
    zr = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            nb = j + 1 if j < i else j - 1
            pij, pn = Pz[i, j], Pz[i, nb]
            var = (pij * (1 - pij) + pn * (1 - pn) + 2 * pij * pn) / Ni[i] if Ni[i] > 0 else np.nan
            z = (pn - pij) / math.sqrt(var) if var and var > 0 else np.nan
            zr.append({"from_grade": order[i], "to_grade": order[j], "neighbour_grade": order[nb],
                       "p_ij": pij, "p_neighbour": pn, "z": z, "p_value": stats.norm.cdf(z) if z == z else np.nan})
    cols = [str(c) for c in ct.columns]
    counts = pd.DataFrame(C.astype(int), columns=cols)
    counts.insert(0, "from_grade", order)
    freq = pd.DataFrame(C / np.where(Ni > 0, Ni, np.nan)[:, None], columns=cols)
    freq.insert(0, "N_i", Ni.astype(int))
    freq.insert(0, "from_grade", order)
    if statuses:
        notes.append(f"Non-grade end statuses kept as extra columns: {', '.join(statuses)}.")
    notes += dropped_note(dropped) + [_order_note(order, how),
                                      "Shorrocks/SVD mobility use the K×K block row-normalised over obligors "
                                      "ending in a grade; empty rows are set to 'stay'."]
    return Outcome(metrics, {"Migration counts": counts, "Migration frequencies (row %)": freq,
                             "ECB z-tests": pd.DataFrame(zr)}, notes=notes, rows_used=int(C.sum()))


@register("pd.overrides", "Override analysis (model grade vs final grade)", "Rating system", _RATING,
          params=(P("grade", help="Model (proposed) grade"), P("final_grade", help="Final grade after overrides"),
                  _ORDER, P("target", required=False, help="Default flag, to compare default rates")),
          description="""Override rate (final ≠ model grade; ECB M_def/N), shares of overrides to a worse grade
(downgrade, more conservative) and to a better grade (upgrade), notch distribution (final position −
model position; positive = worse), mean absolute notches, override rate per model grade, and a two-sided
sign test of H0: upgrades and downgrades equally likely. With `target`: default rates of overridden vs
non-overridden obligors and of upgrades vs downgrades, and a DeLong comparison of the AUC of the final
grade vs the model grade (did overrides improve ranking?).""",
          references=(_REF_ECB + ", section 2.5.2.2", "Regulation (EU) No 575/2013 (CRR), Article 172(3)",
                      _REF_DELONG))
def overrides(ctx: RunContext, grade, final_grade, grade_order=None, target=None) -> Outcome:
    sub, dropped = _load(ctx, [grade, final_grade, target], [target] if target else [])
    mk, fk = _keys(sub[grade]), _keys(sub[final_grade])
    order, how = _order(list(dict.fromkeys(list(mk.unique()) + list(fk.unique()))), grade_order)
    pos = {g: i for i, g in enumerate(order)}
    notch = fk.map(pos).to_numpy(int) - mk.map(pos).to_numpy(int)
    n = len(notch)
    if n == 0:
        raise NotApplicable("No complete rows.")
    ov = notch != 0
    down, upg = int((notch > 0).sum()), int((notch < 0).sum())
    dist = (pd.Series(notch[ov]).value_counts().sort_index().rename_axis("notches").reset_index(name="count"))
    dist["share_of_overrides"] = dist["count"] / max(int(ov.sum()), 1)
    byg = (pd.DataFrame({"model_grade": mk.to_numpy(), "ov": ov, "down": notch > 0, "up": notch < 0})
           .groupby("model_grade").agg(N=("ov", "size"), overrides=("ov", "sum"), downgrades=("down", "sum"),
                                       upgrades=("up", "sum")).reindex([g for g in order if g in set(mk)])
           .reset_index())
    byg["override_rate"] = byg["overrides"] / byg["N"]
    summ = {"N": n, "overrides": int(ov.sum()), "override_rate": float(ov.mean()),
            "downgrades_worse": down, "upgrades_better": upg,
            "mean_abs_notches_when_overridden": float(np.abs(notch[ov]).mean()) if ov.any() else 0.0,
            "sign_test_p_value": float(stats.binomtest(down, down + upg, 0.5).pvalue) if down + upg else float("nan")}
    tables = {"Notch distribution": dist, "Overrides by model grade": byg,
              "Model vs final grade": pd.crosstab(pd.Categorical(mk, order), pd.Categorical(fk, order),
                                                  rownames=["model_grade"], colnames=["final_grade"],
                                                  dropna=False).reset_index()}
    tables["Model vs final grade"].columns = tables["Model vs final grade"].columns.astype(str)
    if target:
        y = sub[target].to_numpy(int)
        grp = np.where(notch > 0, "downgraded (worse)", np.where(notch < 0, "upgraded (better)", "not overridden"))
        tables["Default rate by override type"] = (pd.DataFrame({"type": grp, "y": y}).groupby("type")["y"]
                                                   .agg(N="size", D="sum", DR="mean").reset_index())
        summ |= {"DR_overridden": float(y[ov].mean()) if ov.any() else float("nan"),
                 "DR_not_overridden": float(y[~ov].mean()) if (~ov).any() else float("nan")}
        if y.sum() >= 2 and len(y) - y.sum() >= 2:
            sm_, sf_ = mk.map(pos).to_numpy(float), fk.map(pos).to_numpy(float)
            a1, a2 = _delong_components(y, sf_), _delong_components(y, sm_)
            S = (np.cov(np.vstack([a1[0], a2[0]])) / len(a1[0]) + np.cov(np.vstack([a1[1], a2[1]])) / len(a1[1]))
            vd = S[0, 0] + S[1, 1] - 2 * S[0, 1]
            z = (a1[0].mean() - a2[0].mean()) / math.sqrt(vd) if vd > 0 else float("nan")
            summ |= {"AUC_final_grade": float(a1[0].mean()), "AUC_model_grade": float(a2[0].mean()),
                     "AUC_difference_p_value": float(2 * stats.norm.sf(abs(z))) if vd > 0 else float("nan")}
    return Outcome(summ, tables, notes=dropped_note(dropped) + [_order_note(order, how)], rows_used=n)


@register("pd.adjacent_grade_dr_test", "Default-rate heterogeneity between adjacent grades", "Rating system",
          _RATING, params=(_TGT, P("grade"), P("pd", required=False, help="PD column, used to order grades"), _ORDER),
          description="""For each pair of adjacent grades (k, k+1) in risk order: H0: DR_k = DR_{k+1} against
H1: DR_{k+1} > DR_k (the riskier grade defaults more). Two-proportion z-test with pooled variance (one-sided
p = 1 − Φ(z)) and Fisher's exact test (one-sided). Also the overall chi-square test of homogeneity of default
rates across all grades (H0: all grades share one default rate). Large p-values flag grades that are not
distinguishable in default experience.""",
          references=(_REF_BCBS14, "Fisher (1935), The logic of inductive inference, JRSS 98",
                      "EBA/GL/2017/16, Guidelines on PD estimation (heterogeneity of grades)"))
def adjacent_grade_dr_test(ctx: RunContext, target, grade, pd=None, grade_order=None) -> Outcome:
    sub, dropped = _load(ctx, [target, grade, pd], [target], [pd] if pd else [])
    keys = _keys(sub[grade])
    risk = sub[pd].groupby(keys).mean() if pd else None
    order, how = _order(list(keys.unique()), grade_order, risk)
    g = sub.groupby(keys).agg(N=(target, "size"), D=(target, "sum")).reindex(order)
    if len(g) < 2:
        raise NotApplicable("Need at least two grades.")
    N, D = g["N"].to_numpy(float), g["D"].to_numpy(float)
    rows = []
    for k in range(len(order) - 1):
        n1, d1, n2, d2 = N[k], D[k], N[k + 1], D[k + 1]
        pp = (d1 + d2) / (n1 + n2)
        se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        z = (d2 / n2 - d1 / n1) / se if se > 0 else np.nan
        fe = stats.fisher_exact([[d2, n2 - d2], [d1, n1 - d1]], alternative="greater")
        rows.append({"grade": order[k], "next_grade": order[k + 1], "DR": d1 / n1, "DR_next": d2 / n2,
                     "N": int(n1), "N_next": int(n2), "z": z,
                     "p_value_z": float(stats.norm.sf(z)) if z == z else np.nan,
                     "p_value_fisher": float(fe.pvalue)})
    ct = np.column_stack([D, N - D])
    ct = ct[:, ct.sum(axis=0) > 0]
    if ct.shape[1] == 2:
        chi, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    else:
        chi, p, dof = np.nan, np.nan, 0
    t = _frame(rows)
    return Outcome({"pairs": len(t), "max_p_value_z": float(t["p_value_z"].max()),
                    "max_p_value_fisher": float(t["p_value_fisher"].max()),
                    "homogeneity_chi2": float(chi), "homogeneity_dof": int(dof), "homogeneity_p_value": float(p),
                    "n": len(sub)},
                   {"Adjacent grade tests": t}, notes=dropped_note(dropped) + [_order_note(order, how)],
                   rows_used=len(sub))


@register("pd.grade_homogeneity", "Homogeneity of default rates within grades by sub-segment", "Rating system",
          _RATING, params=(_TGT, P("grade"), P("segment"), _ORDER),
          description="""Within each grade, a chi-square test of independence between default and `segment`
(H0: the sub-segments of the grade share one default rate; Pearson, no continuity correction), with
Fisher's exact test when the grade has two segments, and the minimum expected cell count. Combined test
over grades: Σχ² with Σdof (independent strata). Default rates by grade × segment are tabulated. Grades with
one segment or no defaults are skipped (listed).""",
          references=(_REF_BCBS14, "EBA/GL/2017/16, Guidelines on PD estimation (homogeneity of grades)",
                      "Agresti (2013), Categorical Data Analysis, 3rd ed., Wiley"))
def grade_homogeneity(ctx: RunContext, target, grade, segment, grade_order=None) -> Outcome:
    sub, dropped = _load(ctx, [target, grade, segment], [target])
    keys = _keys(sub[grade])
    seg = sub[segment].astype(str)
    order, how = _order(list(keys.unique()), grade_order)
    rows, cells = [], []
    y = sub[target].to_numpy(int)
    for gk in order:
        msk = (keys == gk).to_numpy()
        tab = pd.crosstab(seg[msk], y[msk]).reindex(columns=[0, 1], fill_value=0)
        for sname, r in tab.iterrows():
            cells.append({"grade": gk, "segment": sname, "N": int(r.sum()), "D": int(r[1]),
                          "DR": r[1] / r.sum() if r.sum() else np.nan})
        row = {"grade": gk, "N": int(msk.sum()), "segments": len(tab)}
        if len(tab) < 2 or tab[1].sum() == 0 or tab[0].sum() == 0:
            rows.append(row | {"chi2": np.nan, "dof": 0, "p_value": np.nan, "fisher_p_value": np.nan,
                               "min_expected": np.nan, "comment": "skipped: one segment or one outcome"})
            continue
        chi, p, dof, exp = stats.chi2_contingency(tab.to_numpy(), correction=False)
        fp = float(stats.fisher_exact(tab.to_numpy()).pvalue) if tab.shape == (2, 2) else np.nan
        rows.append(row | {"chi2": chi, "dof": int(dof), "p_value": p, "fisher_p_value": fp,
                           "min_expected": float(exp.min()), "comment": ""})
    t = pd.DataFrame(rows)
    ok = t[t["dof"] > 0]
    if ok.empty:
        raise NotApplicable("No grade has at least two segments and both outcomes.")
    chi, dof = float(ok["chi2"].sum()), int(ok["dof"].sum())
    notes = dropped_note(dropped) + [_order_note(order, how)]
    if (ok["min_expected"] < 5).any():
        notes.append("Some grades have expected counts below 5: prefer the Fisher p-value or merge segments.")
    return Outcome({"grades_tested": len(ok), "combined_chi2": chi, "combined_dof": dof,
                    "combined_p_value": float(stats.chi2.sf(chi, dof)), "min_grade_p_value": float(ok["p_value"].min()),
                    "n": len(sub)},
                   {"Homogeneity by grade": t, "Default rate by grade and segment": pd.DataFrame(cells)},
                   notes=notes, rows_used=len(sub))


# ── default definition ─────────────────────────────────────────────────────

@register("pd.default_definition_replication", "Replicate the default flag from days past due", "Default definition",
          _DEFDEF,
          params=(P("id"), P("date"), P("dpd", help="Days-past-due counter"),
                  P("target", help="Provided default flag to check"),
                  P("past_due_amount", required=False, help="Amount past due (for materiality thresholds)"),
                  P("exposure", required=False, help="Total on-balance exposure (for the relative threshold)"),
                  P("dpd_threshold", "integer", default=90, help="Default when dpd > this"),
                  P("abs_threshold", "number", default=0.0, help="Absolute materiality: amount past due > this"),
                  P("rel_threshold", "number", default=0.0,
                    help="Relative materiality: amount past due / exposure > this (e.g. 0.01)"),
                  P("window_months", "integer", default=12,
                    help="Flag = trigger within (date, date + window]; 0 = trigger at the date itself")),
          description="""Recomputes the default flag from the days-past-due data and compares it with the
provided flag, row by row (id × date). Trigger on a record: dpd > dpd_threshold AND (if abs_threshold > 0)
past-due amount > abs_threshold AND (if rel_threshold > 0) past-due amount / exposure > rel_threshold — both
materiality thresholds must be breached, as in the EBA RTS on materiality. Replicated flag at an observation
date t: some record of the same id with date in (t, t + window_months] triggers (forward-looking default
horizon); window_months = 0 compares the default status at t itself. Outputs the confusion table
(provided × replicated), mismatch count and rate, Cohen's kappa, and mismatching ids. Simplifications:
the dpd counter is taken as given (no recount of consecutive days from the materiality breach), no
probation/cure period, no unlikeliness-to-pay triggers — mismatches can stem from those.""",
          references=("Regulation (EU) No 575/2013 (CRR), Article 178",
                      "EBA/GL/2016/07, Guidelines on the application of the definition of default",
                      "Commission Delegated Regulation (EU) 2018/171 (materiality threshold for credit obligations "
                      "past due)"))
def default_definition_replication(ctx: RunContext, id, date, dpd, target, past_due_amount=None, exposure=None,
                                   dpd_threshold=90, abs_threshold=0.0, rel_threshold=0.0,
                                   window_months=12) -> Outcome:
    if window_months < 0:
        raise ValueError("window_months must be >= 0")
    if (abs_threshold > 0 or rel_threshold > 0) and not past_due_amount:
        raise ValueError("Materiality thresholds need `past_due_amount`.")
    if rel_threshold > 0 and not exposure:
        raise ValueError("The relative threshold needs `exposure`.")
    df = ctx.df
    sub, dropped = complete(df, [id, date, dpd])
    d = pd.DataFrame({"id": sub[id].to_numpy(), "date": pd.to_datetime(sub[date], errors="coerce").to_numpy(),
                      "dpd": pd.to_numeric(sub[dpd], errors="coerce").to_numpy(float)}, index=sub.index)
    bad = d["date"].isna() | d["dpd"].isna()
    dropped += int(bad.sum())
    d = d[~bad]
    if d.empty:
        raise NotApplicable("No rows with a valid id, date and dpd.")
    if d.duplicated(["id", "date"]).any():
        raise ValueError(f"Several rows share the same {id} and {date}; aggregate to one record per id and date.")
    trig = d["dpd"] > dpd_threshold
    notes = []
    if past_due_amount:
        amt = pd.to_numeric(df.loc[d.index, past_due_amount], errors="coerce")
        if amt.isna().any():
            notes.append(f"{int(amt.isna().sum())} records without a past-due amount treated as not material.")
        if abs_threshold > 0:
            trig &= amt > abs_threshold
        if rel_threshold > 0:
            ex = pd.to_numeric(df.loc[d.index, exposure], errors="coerce")
            ratio = amt / ex.where(ex > 0)
            trig &= ratio > rel_threshold
    d["trigger"] = trig.fillna(False).astype(bool)
    d = d.sort_values(["id", "date"], kind="mergesort")
    if window_months == 0:
        d["replicated"] = d["trigger"]
    else:
        tdate = d["date"].where(d["trigger"])
        nxt = tdate.groupby(d["id"]).shift(-1)
        nxt = nxt.groupby(d["id"]).bfill()
        horizon = d["date"] + pd.DateOffset(months=window_months)
        d["replicated"] = nxt.notna() & (nxt <= horizon)
        cens = int((horizon > d["date"].max()).sum())
        if cens:
            notes.append(f"{cens} observation dates have a window extending beyond the last date in the data "
                         "(possible censoring: replicated flag may be 0 for lack of data).")
    prov_raw = df.loc[d.index, [target]]
    has = prov_raw[target].notna().to_numpy()
    if not has.any():
        raise NotApplicable("The provided flag is missing on every row.")
    cmp_ = d[has].copy()
    cmp_["provided"] = _bin(prov_raw[has], target).to_numpy()
    nb = cmp_["provided"].isna()
    if nb.any():
        notes.append(f"{int(nb.sum())} rows with a non-binary provided flag excluded.")
        cmp_ = cmp_[~nb]
    cmp_["provided"] = cmp_["provided"].astype(int)
    cmp_["replicated"] = cmp_["replicated"].astype(int)
    conf = (pd.crosstab(cmp_["provided"], cmp_["replicated"]).reindex(index=[0, 1], columns=[0, 1], fill_value=0))
    conf.index = pd.Index(["provided 0", "provided 1"], name="provided_flag")
    conf.columns = ["replicated 0", "replicated 1"]
    conf = conf.reset_index()
    mm = cmp_[cmp_["provided"] != cmp_["replicated"]]
    from sklearn.metrics import cohen_kappa_score
    kappa = (float(cohen_kappa_score(cmp_["provided"], cmp_["replicated"]))
             if cmp_["provided"].nunique() + cmp_["replicated"].nunique() > 2 else float("nan"))
    ids = sorted(pd.unique(mm["id"]), key=str)
    sample = mm.head(50)[["id", "date", "dpd", "provided", "replicated"]].reset_index(drop=True)
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    notes += [f"Trigger: dpd > {dpd_threshold}" + (f", amount past due > {abs_threshold}" if abs_threshold > 0 else "")
              + (f", amount/exposure > {rel_threshold}" if rel_threshold > 0 else "")
              + (f"; window (t, t + {window_months} months]." if window_months else "; status at t.")]
    notes += dropped_note(dropped)
    return Outcome({"rows_compared": len(cmp_), "mismatches": len(mm), "mismatch_rate": len(mm) / len(cmp_),
                    "provided_only": int(((cmp_["provided"] == 1) & (cmp_["replicated"] == 0)).sum()),
                    "replicated_only": int(((cmp_["provided"] == 0) & (cmp_["replicated"] == 1)).sum()),
                    "provided_defaults": int(cmp_["provided"].sum()),
                    "replicated_defaults": int(cmp_["replicated"].sum()), "cohen_kappa": kappa,
                    "mismatching_ids": len(ids), "sample_mismatching_ids": ", ".join(map(str, ids[:20]))},
                   {"Confusion table": conf, "Mismatches (first 50, sorted by id, date)": sample},
                   notes=notes, rows_used=len(cmp_))


@register("pd.default_rate_series", "Default rate time series by period", "Default definition", _RATING,
          params=(_TGT, P("period"), P("pd", required=False), P("weight", required=False), _CONF),
          description="""Per period: number of obligors N, defaults D, default rate D/N with exact
Clopper–Pearson interval, the mean PD (if given) and the exposure-weighted default rate (if `weight`).
Descriptive (no test); use it to spot breaks in the default definition, data gaps or cyclicality.""",
          references=("Clopper & Pearson (1934), The use of confidence or fiducial limits illustrated in the case "
                      "of the binomial, Biometrika 26", "EBA/GL/2017/16 (one-year default rate)"))
def default_rate_series(ctx: RunContext, target, period, pd=None, weight=None, confidence=0.95) -> Outcome:
    _zq(confidence)
    num_cols = [c for c in (pd, weight) if c]
    sub, dropped = _load(ctx, [target, period, *num_cols], [target], num_cols)
    if sub.empty:
        raise NotApplicable("No complete rows.")
    t = _period_rows(sub, target, pd, period, weight)
    ci = [stats.binomtest(int(d), int(n)).proportion_ci(confidence, method="exact") for d, n in zip(t["D"], t["N"])]
    t.insert(4, "DR_CI_lower", [c.low for c in ci])
    t.insert(5, "DR_CI_upper", [c.high for c in ci])
    traces = [(t["period"], t["DR"], "Default rate", "lines+markers")]
    if pd:
        t = t.rename(columns={"PD": "mean_PD"})
        traces.append((t["period"], t["mean_PD"], "Mean PD", "lines+markers"))
    fig = _fig_xy(traces, "Default rate by period", "Period", "Rate")
    return Outcome({"periods": len(t), "N": int(t["N"].sum()), "D": int(t["D"].sum()),
                    "pooled_DR": float(t["D"].sum() / t["N"].sum()), "min_DR": float(t["DR"].min()),
                    "max_DR": float(t["DR"].max()), "sd_DR": float(t["DR"].std(ddof=1)) if len(t) > 1 else float("nan")},
                   {"Default rate by period": t}, [fig], dropped_note(dropped), rows_used=len(sub))
