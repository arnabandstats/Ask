"""Fairness / non-discrimination tests for classification and scoring models.

Inputs: the observed outcome (`target`, 1 = event), the model's decision — a predicted 0/1 label
column (`predicted`) or a score with a `threshold` (predicted 1 when score >= threshold) — and a
protected attribute (`protected`). Group comparisons are made against a reference group (default:
the largest group; ties broken alphabetically).

Convention: with target 1 = default / SAR / fraud, a predicted 1 is UNfavourable for the person
(declined, flagged). `favourable_label` (default 0) says which prediction is favourable; selection
rates, statistical parity and disparate impact are computed on the favourable rate. Error rates
(TPR, FPR, PPV, ...) always use the event = 1 convention.

No fairness thresholds (e.g. the US "four-fifths rule") are applied: tests report differences,
ratios and p-values only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import (NotApplicable, Outcome, P, RunContext, binary, complete, dropped_note,
                                 num, register)
from ask.validation.models import model_features, predict_scores

_FAIR = ("ml_classification", "pd", "aml", "ews")

_COMMON = (
    P("target"), P("protected"),
    P("score", required=False, help="Model score (higher = riskier); predicted 1 when score >= threshold"),
    P("predicted", required=False, help="Predicted 0/1 label column (instead of score + threshold)"),
    P("threshold", "number", default=0.5),
    P("reference_group", "string", required=False, help="Reference group; default the largest group"),
    P("favourable_label", "integer", default=0, choices=(0, 1),
      help="Prediction that is favourable to the person (0 = not predicted as event: approved / not flagged)"),
)

_REFS_GENERAL = ("Barocas, Hardt & Narayanan (2023), Fairness and Machine Learning: Limitations and Opportunities",
                 "Verma & Rubin (2018), Fairness definitions explained, FairWare",
                 "European Banking Authority (2020), EBA/GL/2020/06 Guidelines on loan origination and "
                 "monitoring — use of automated models")


def _uniq(cols) -> list[str]:
    return list(dict.fromkeys(c for c in cols if c))


def _prep(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5, extra=(),
          need_score=False) -> tuple[pd.DataFrame, int]:
    """Frame with y, yhat, g (group as text), s (score, if given) and extra columns; rows dropped."""
    if need_score and not score:
        raise ValueError("This test needs `score`.")
    if not score and not predicted:
        raise ValueError("Give `predicted` (a 0/1 label column) or `score` (with `threshold`).")
    sub, dropped = complete(ctx.df, _uniq([target, protected, score, predicted, *extra]))
    d = pd.DataFrame({"y": binary(sub, target), "g": sub[protected].astype(str)}, index=sub.index)
    if score:
        d["s"] = num(sub, score)
    d["yhat"] = binary(sub, predicted) if predicted else (d["s"] >= threshold).astype(float).where(d["s"].notna())
    for e in extra:
        d[e] = sub[e]
    bad = d.drop(columns=["g", *extra]).isna().any(axis=1)
    d = d[~bad]
    dropped += int(bad.sum())
    d["y"], d["yhat"] = d["y"].astype(int), d["yhat"].astype(int)
    if d["g"].nunique() < 2:
        raise NotApplicable(f"'{protected}' has fewer than two groups in the complete rows.")
    return d, dropped


def _ref(d: pd.DataFrame, reference_group) -> str:
    counts = d["g"].value_counts()
    if reference_group is not None:
        hits = [g for g in counts.index if str(g).lower() == str(reference_group).lower()]
        if not hits:
            raise ValueError(f"Reference group '{reference_group}' not found; groups: {sorted(counts.index)[:20]}")
        return hits[0]
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _div(a, b):
    return a / b if b else np.nan


def _rates(d: pd.DataFrame, fav: int, by: str = "g") -> pd.DataFrame:
    rows = []
    for g in sorted(d[by].unique(), key=str):
        x = d[d[by] == g]
        y, h = x["y"].to_numpy(), x["yhat"].to_numpy()
        tp, fp = int(((y == 1) & (h == 1)).sum()), int(((y == 0) & (h == 1)).sum())
        tn, fn = int(((y == 0) & (h == 0)).sum()), int(((y == 1) & (h == 0)).sum())
        n = len(x)
        rows.append({"group": g, "n": n, "events": tp + fn, "base_rate": (tp + fn) / n,
                     "n_favourable": int((h == fav).sum()), "favourable_rate": float((h == fav).mean()),
                     "positive_prediction_rate": (tp + fp) / n, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "tpr": _div(tp, tp + fn), "fpr": _div(fp, fp + tn), "fnr": _div(fn, tp + fn),
                     "tnr": _div(tn, tn + fp), "ppv": _div(tp, tp + fp), "npv": _div(tn, tn + fn),
                     "fdr": _div(fp, tp + fp), "for": _div(fn, fn + tn), "accuracy": (tp + tn) / n,
                     "fn_fp_ratio": _div(fn, fp)})
    return pd.DataFrame(rows)


# rate: (numerator, denominator) from a _rates row
_FRAC = {
    "favourable_rate": (lambda r: r["n_favourable"], lambda r: r["n"]),
    "positive_prediction_rate": (lambda r: r["TP"] + r["FP"], lambda r: r["n"]),
    "base_rate": (lambda r: r["events"], lambda r: r["n"]),
    "tpr": (lambda r: r["TP"], lambda r: r["TP"] + r["FN"]),
    "fnr": (lambda r: r["FN"], lambda r: r["TP"] + r["FN"]),
    "fpr": (lambda r: r["FP"], lambda r: r["FP"] + r["TN"]),
    "tnr": (lambda r: r["TN"], lambda r: r["FP"] + r["TN"]),
    "ppv": (lambda r: r["TP"], lambda r: r["TP"] + r["FP"]),
    "fdr": (lambda r: r["FP"], lambda r: r["TP"] + r["FP"]),
    "npv": (lambda r: r["TN"], lambda r: r["TN"] + r["FN"]),
    "for": (lambda r: r["FN"], lambda r: r["TN"] + r["FN"]),
}


def _ztest(x1, n1, x2, n2) -> tuple[float, float]:
    """Pooled two-proportion z-test, two-sided."""
    if not n1 or not n2:
        return np.nan, np.nan
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return np.nan, np.nan
    z = (x1 / n1 - x2 / n2) / se
    return float(z), float(2 * stats.norm.sf(abs(z)))


def _vs_ref(t: pd.DataFrame, ref: str, metrics) -> pd.DataFrame:
    r0 = t.set_index("group").loc[ref]
    rows = []
    for m in metrics:
        nf, df_ = _FRAC[m]
        x0, n0 = nf(r0), df_(r0)
        v0 = _div(x0, n0)
        for _, r in t.iterrows():
            x, n = nf(r), df_(r)
            v = _div(x, n)
            z, p = _ztest(x, n, x0, n0) if r["group"] != ref else (np.nan, np.nan)
            rows.append({"metric": m, "group": r["group"], "is_reference": r["group"] == ref, "numerator": int(x),
                         "denominator": int(n), "value": v, "reference_value": v0, "difference_vs_reference": v - v0,
                         "ratio_vs_reference": _div(v, v0), "z": z, "p_value": p})
    return pd.DataFrame(rows)


def _spread(t: pd.DataFrame, m: str) -> tuple[float, float]:
    v = t[m].dropna()
    if v.empty:
        return np.nan, np.nan
    return float(v.max() - v.min()), (float(v.min() / v.max()) if v.max() > 0 else np.nan)


def _chi2(d: pd.DataFrame, mask, col: str) -> dict:
    """Chi-square test of independence of `col` (0/1) and group, on the rows in mask."""
    x = d[mask]
    ct = pd.crosstab(x["g"], x[col])
    ct = ct.loc[ct.sum(axis=1) > 0, ct.sum(axis=0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return {"chi2": np.nan, "dof": np.nan, "p_value": np.nan, "min_expected": np.nan, "cramers_v": np.nan,
                "fisher_p_value": np.nan, "n": len(x)}
    chi2, p, dof, exp = stats.chi2_contingency(ct.to_numpy(), correction=False)
    n = ct.to_numpy().sum()
    out = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "min_expected": float(exp.min()),
           "cramers_v": float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))), "fisher_p_value": np.nan, "n": int(n)}
    if ct.shape == (2, 2):
        out["fisher_p_value"] = float(stats.fisher_exact(ct.to_numpy()).pvalue)
    return out


def _common_notes(d, ref, dropped, predicted, threshold, fav) -> list[str]:
    how = "predicted label column" if predicted else f"score >= {threshold}"
    small = d["g"].value_counts()
    notes = dropped_note(dropped) + [f"Reference group: '{ref}'. Predicted 1 = {how}. Favourable prediction = {fav}."]
    if small.min() < 30:
        notes.append(f"Small groups (n < 30): {sorted(small[small < 30].index.tolist())}; rates are imprecise.")
    return notes


def _pair_test(ctx, target, protected, score, predicted, threshold, reference_group, favourable_label, metrics,
               chi_spec):
    d, dropped = _prep(ctx, target, protected, score, predicted, threshold)
    ref = _ref(d, reference_group)
    t = _rates(d, favourable_label)
    cmp = _vs_ref(t, ref, metrics)
    chis = {k: _chi2(d, (d[c] == v).to_numpy(), col) for k, (c, v, col) in chi_spec.items()}
    return d, dropped, ref, t, cmp, chis


# ── tests ──────────────────────────────────────────────────────────────────

@register("fairness.group_metrics", "Confusion-matrix metrics by protected group", "Fairness", _FAIR,
          params=_COMMON,
          description="""Per group of the protected attribute: n, base rate P(Y=1), positive-prediction rate
P(Ŷ=1), favourable rate P(Ŷ=favourable), TP/FP/TN/FN, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR, accuracy and
FN/FP ratio. Summary gives the max − min spread across groups of the key rates. The overview behind the
specific parity tests.""",
          references=_REFS_GENERAL)
def group_metrics(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                  reference_group=None, favourable_label=0) -> Outcome:
    d, dropped = _prep(ctx, target, protected, score, predicted, threshold)
    ref = _ref(d, reference_group)
    t = _rates(d, favourable_label)
    summary = {"groups": len(t), "reference_group": ref, "n": len(d)}
    for m in ("favourable_rate", "tpr", "fpr", "ppv", "accuracy", "base_rate"):
        summary[f"{m}_max_minus_min"] = _spread(t, m)[0]
    return Outcome(summary, {"Metrics by group": t}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.demographic_parity", "Statistical parity difference and disparate impact ratio", "Fairness", _FAIR,
          params=_COMMON,
          description="""Demographic (statistical) parity: the favourable-prediction rate should not depend on
the protected attribute. Per group: favourable rate and positive-prediction rate, difference and ratio vs
the reference group (the ratio of favourable rates is the disparate-impact ratio), pooled two-proportion
z-test (H0: equal rates). Summary: statistical parity difference = max − min favourable rate across groups,
disparate impact = min/max favourable rate, lowest ratio vs the reference, and the chi-square test of
independence between favourable prediction and group. Ignores differences in base rates by design.""",
          references=("Feldman et al. (2015), Certifying and removing disparate impact, KDD",
                      "Calders & Verwer (2010), Three naive Bayes approaches for discrimination-free "
                      "classification, Data Mining and Knowledge Discovery 21", *_REFS_GENERAL))
def demographic_parity(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                       reference_group=None, favourable_label=0) -> Outcome:
    d, dropped, ref, t, cmp, _ = _pair_test(ctx, target, protected, score, predicted, threshold, reference_group,
                                            favourable_label, ["favourable_rate", "positive_prediction_rate"], {})
    d = d.assign(fav=(d["yhat"] == favourable_label).astype(int))
    chi = _chi2(d, np.ones(len(d), bool), "fav")
    fav = cmp[(cmp["metric"] == "favourable_rate") & ~cmp["is_reference"]]
    worst = fav.sort_values(["ratio_vs_reference", "group"]).iloc[0]
    spd, di = _spread(t, "favourable_rate")
    return Outcome({"groups": len(t), "reference_group": ref, "statistical_parity_difference": spd,
                    "disparate_impact_min_over_max": di, "lowest_ratio_vs_reference": worst["ratio_vs_reference"],
                    "lowest_ratio_group": worst["group"], "chi2": chi["chi2"], "chi2_p_value": chi["p_value"],
                    "n": len(d)},
                   {"Parity by group": cmp}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.equal_opportunity", "Equal opportunity (true-positive-rate gap)", "Fairness", _FAIR,
          params=_COMMON,
          description="""Equal opportunity (Hardt et al. 2016): among actual events (Y = 1) the probability of
being predicted an event, TPR = P(Ŷ=1 | Y=1), should be equal across groups (equivalently FNR). Per group:
TPR and FNR, difference and ratio vs the reference, two-proportion z-test among events. Summary: TPR
max − min, min/max ratio, chi-square test of Ŷ vs group among Y = 1 rows.""",
          references=("Hardt, Price & Srebro (2016), Equality of opportunity in supervised learning, NeurIPS",
                      *_REFS_GENERAL))
def equal_opportunity(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                      reference_group=None, favourable_label=0) -> Outcome:
    d, dropped, ref, t, cmp, chis = _pair_test(ctx, target, protected, score, predicted, threshold, reference_group,
                                               favourable_label, ["tpr", "fnr"], {"pos": ("y", 1, "yhat")})
    gap, ratio = _spread(t, "tpr")
    return Outcome({"groups": len(t), "reference_group": ref, "tpr_max_minus_min": gap, "tpr_min_over_max": ratio,
                    "chi2_among_events": chis["pos"]["chi2"], "chi2_p_value": chis["pos"]["p_value"],
                    "events": int(d["y"].sum())},
                   {"TPR / FNR by group": cmp}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.equalized_odds", "Equalized odds (TPR and FPR gaps)", "Fairness", _FAIR,
          params=_COMMON,
          description="""Equalized odds (Hardt et al. 2016): Ŷ independent of the group conditional on Y — equal
TPR AND equal FPR across groups. Per group: TPR, FPR with difference, ratio and z-test vs the reference.
Summary: TPR and FPR max − min gaps, equalized-odds difference = max(TPR gap, FPR gap), equalized-odds
ratio = min(TPR min/max, FPR min/max), and chi-square tests of Ŷ vs group among Y = 1 and among Y = 0.""",
          references=("Hardt, Price & Srebro (2016), Equality of opportunity in supervised learning, NeurIPS",
                      "Bird et al. (2020), Fairlearn: a toolkit for assessing and improving fairness in AI, "
                      "Microsoft Tech. Report MSR-TR-2020-32", *_REFS_GENERAL))
def equalized_odds(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                   reference_group=None, favourable_label=0) -> Outcome:
    d, dropped, ref, t, cmp, chis = _pair_test(
        ctx, target, protected, score, predicted, threshold, reference_group, favourable_label, ["tpr", "fpr"],
        {"pos": ("y", 1, "yhat"), "neg": ("y", 0, "yhat")})
    tg, tr = _spread(t, "tpr")
    fg, fr = _spread(t, "fpr")
    return Outcome({"groups": len(t), "reference_group": ref, "tpr_max_minus_min": tg, "fpr_max_minus_min": fg,
                    "equalized_odds_difference": np.nanmax([tg, fg]), "equalized_odds_ratio": np.nanmin([tr, fr]),
                    "chi2_p_value_among_events": chis["pos"]["p_value"],
                    "chi2_p_value_among_non_events": chis["neg"]["p_value"], "n": len(d)},
                   {"TPR / FPR by group": cmp}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.predictive_parity", "Predictive parity (PPV and NPV gaps)", "Fairness", _FAIR,
          params=_COMMON,
          description="""Predictive parity (outcome test / sufficiency at the decision level): among rows
predicted as events, the event rate PPV = P(Y=1 | Ŷ=1) should be equal across groups; likewise
NPV = P(Y=0 | Ŷ=0). Per group: PPV, NPV, difference, ratio and z-test vs the reference. Summary: PPV and NPV
max − min and chi-square tests of Y vs group among Ŷ = 1 and among Ŷ = 0. With different base rates,
predictive parity and equalized odds cannot both hold (Chouldechova 2017).""",
          references=("Chouldechova (2017), Fair prediction with disparate impact, Big Data 5",
                      "Kleinberg, Mullainathan & Raghavan (2017), Inherent trade-offs in the fair determination of "
                      "risk scores, ITCS", *_REFS_GENERAL))
def predictive_parity(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                      reference_group=None, favourable_label=0) -> Outcome:
    d, dropped, ref, t, cmp, chis = _pair_test(
        ctx, target, protected, score, predicted, threshold, reference_group, favourable_label, ["ppv", "npv"],
        {"pp": ("yhat", 1, "y"), "pn": ("yhat", 0, "y")})
    return Outcome({"groups": len(t), "reference_group": ref, "ppv_max_minus_min": _spread(t, "ppv")[0],
                    "npv_max_minus_min": _spread(t, "npv")[0],
                    "chi2_p_value_among_predicted_events": chis["pp"]["p_value"],
                    "chi2_p_value_among_predicted_non_events": chis["pn"]["p_value"], "n": len(d)},
                   {"PPV / NPV by group": cmp}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.error_rate_balance", "Error-rate balance (FNR, FPR, FDR, FOR by group)", "Fairness", _FAIR,
          params=_COMMON,
          description="""Balance of the four error rates across groups: false negative rate FN/(TP+FN), false
positive rate FP/(FP+TN), false discovery rate FP/(TP+FP), false omission rate FN/(FN+TN). Per group:
value, difference and ratio vs the reference and two-proportion z-test. Summary: max − min of each rate and
the largest ratio vs the reference. Shows which group bears which kind of error.""",
          references=("Chouldechova (2017), Fair prediction with disparate impact, Big Data 5",
                      "Berk et al. (2021), Fairness in criminal justice risk assessments: the state of the art, "
                      "Sociological Methods & Research 50", *_REFS_GENERAL))
def error_rate_balance(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                       reference_group=None, favourable_label=0) -> Outcome:
    d, dropped, ref, t, cmp, _ = _pair_test(ctx, target, protected, score, predicted, threshold, reference_group,
                                            favourable_label, ["fnr", "fpr", "fdr", "for"], {})
    summary = {"groups": len(t), "reference_group": ref, "n": len(d)}
    for m in ("fnr", "fpr", "fdr", "for"):
        summary[f"{m}_max_minus_min"] = _spread(t, m)[0]
        summary[f"{m}_max_ratio_vs_reference"] = float(cmp.loc[cmp["metric"] == m, "ratio_vs_reference"].max())
    return Outcome(summary, {"Error rates by group": cmp}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.treatment_equality", "Treatment equality (FN / FP ratio by group)", "Fairness", _FAIR,
          params=_COMMON,
          description="""Treatment equality (Berk et al.): the ratio of false negatives to false positives,
FN/FP, should be equal across groups — i.e. the model trades off its two kinds of error the same way for
everyone. Per group: FN, FP, FN/FP, difference and ratio vs the reference. Summary: min and max FN/FP
across groups and their ratio (max/min). Groups with FP = 0 have an undefined ratio.""",
          references=("Berk et al. (2021), Fairness in criminal justice risk assessments: the state of the art, "
                      "Sociological Methods & Research 50", *_REFS_GENERAL))
def treatment_equality(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                       reference_group=None, favourable_label=0) -> Outcome:
    d, dropped = _prep(ctx, target, protected, score, predicted, threshold)
    ref = _ref(d, reference_group)
    t = _rates(d, favourable_label)[["group", "n", "FN", "FP", "fn_fp_ratio"]]
    r0 = float(t.set_index("group").loc[ref, "fn_fp_ratio"])
    t["difference_vs_reference"] = t["fn_fp_ratio"] - r0
    t["ratio_vs_reference"] = t["fn_fp_ratio"] / r0 if r0 else np.nan
    v = t["fn_fp_ratio"].dropna()
    return Outcome({"groups": len(t), "reference_group": ref, "reference_fn_fp_ratio": r0,
                    "min_fn_fp_ratio": float(v.min()) if len(v) else np.nan,
                    "max_fn_fp_ratio": float(v.max()) if len(v) else np.nan,
                    "max_over_min": float(v.max() / v.min()) if len(v) and v.min() > 0 else np.nan, "n": len(d)},
                   {"FN / FP by group": t}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label))


@register("fairness.calibration_by_group", "Calibration within groups (observed vs predicted by group and bin)",
          "Fairness", _FAIR,
          params=(P("target"), P("protected"), P("score", help="Predicted probability in [0, 1]"),
                  P("bins", "integer", default=10), P("reference_group", "string", required=False)),
          description="""Calibration within groups (sufficiency): for the same predicted probability the
observed event rate should be the same in every group. Score bins are quantiles of the whole population, so
bins are comparable across groups. Group × bin table: n, mean predicted, observed rate, difference. Per
group: mean predicted, observed rate, observed/expected ratio, difference vs the reference, expected
calibration error ECE = Σ (n_b/n_g)|ō − p̄|, Hosmer–Lemeshow chi-square over the group's bins (df = bins − 2)
with p-value, and a binomial test of total events against the sum of predicted probabilities
(normal approximation; H0: calibrated in the large).""",
          references=("Kleinberg, Mullainathan & Raghavan (2017), Inherent trade-offs in the fair determination "
                      "of risk scores, ITCS",
                      "Chouldechova (2017), Fair prediction with disparate impact, Big Data 5",
                      "Hosmer & Lemeshow (2000), Applied Logistic Regression, 2nd ed.", *_REFS_GENERAL))
def calibration_by_group(ctx: RunContext, target, protected, score, bins=10, reference_group=None) -> Outcome:
    d, dropped = _prep(ctx, target, protected, score, None, 0.5, need_score=True)
    if d["s"].min() < 0 or d["s"].max() > 1:
        raise ValueError("Scores must be probabilities in [0, 1].")
    ref = _ref(d, reference_group)
    edges = np.unique(np.quantile(d["s"], np.linspace(0, 1, bins + 1)))
    d["bin"] = np.clip(np.searchsorted(edges, d["s"], side="right") - 1, 0, max(len(edges) - 2, 0)) + 1
    gb = d.groupby(["g", "bin"]).agg(n=("y", "size"), events=("y", "sum"), mean_predicted=("s", "mean"),
                                     observed_rate=("y", "mean")).reset_index().rename(columns={"g": "group"})
    gb["observed_minus_predicted"] = gb["observed_rate"] - gb["mean_predicted"]
    rows = []
    for g, x in gb.groupby("group", sort=True):
        n = x["n"].sum()
        pe = x["mean_predicted"].clip(1e-12, 1 - 1e-12)
        hl = float(np.sum((x["events"] - x["n"] * pe) ** 2 / (x["n"] * pe * (1 - pe))))
        dfh = max(len(x) - 2, 1)
        sg = d.loc[d["g"] == g, "s"].to_numpy()
        exp_ev, var = sg.sum(), float(np.sum(sg * (1 - sg)))
        z = (x["events"].sum() - exp_ev) / np.sqrt(var) if var > 0 else np.nan
        rows.append({"group": g, "n": int(n), "mean_predicted": float(sg.mean()),
                     "observed_rate": float(x["events"].sum() / n), "observed_over_expected": _div(x["events"].sum(), exp_ev),
                     "ece": float(np.sum(x["n"] / n * np.abs(x["observed_minus_predicted"]))),
                     "hosmer_lemeshow_chi2": hl, "hl_df": dfh, "hl_p_value": float(stats.chi2.sf(hl, dfh)),
                     "events_vs_expected_z": float(z), "events_vs_expected_p_value": float(2 * stats.norm.sf(abs(z)))
                     if np.isfinite(z) else np.nan})
    t = pd.DataFrame(rows)
    gap0 = t.set_index("group").loc[ref, "observed_rate"] - t.set_index("group").loc[ref, "mean_predicted"]
    t["calibration_gap"] = t["observed_rate"] - t["mean_predicted"]
    t["calibration_gap_minus_reference"] = t["calibration_gap"] - gap0
    worst = t.sort_values(["ece", "group"], ascending=[False, True]).iloc[0]
    return Outcome({"groups": len(t), "bins": len(edges) - 1, "reference_group": ref, "max_group_ece": worst["ece"],
                    "max_ece_group": worst["group"],
                    "max_abs_calibration_gap": float(t["calibration_gap"].abs().max()),
                    "min_hl_p_value": float(t["hl_p_value"].min()), "n": len(d)},
                   {"Calibration by group": t, "Group x bin": gb}, rows_used=len(d),
                   notes=dropped_note(dropped) + [f"Reference group: '{ref}'. Bins are population quantiles; "
                                                  "Hosmer–Lemeshow per group uses that group's non-empty bins."])


def _delong(y, s) -> tuple[float, float]:
    """AUC and its DeLong standard error (fast midrank form)."""
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return (np.nan, np.nan) if (m == 0 or n == 0) else (_auc_only(pos, neg), np.nan)
    tz = stats.rankdata(np.concatenate([pos, neg]))
    tx, ty = stats.rankdata(pos), stats.rankdata(neg)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1 - (tz[m:] - ty) / m
    return float(auc), float(np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n))


def _auc_only(pos, neg) -> float:
    tz = stats.rankdata(np.concatenate([pos, neg]))
    m, n = len(pos), len(neg)
    return float((tz[:m].sum() - m * (m + 1) / 2) / (m * n))


@register("fairness.auc_by_group", "Discrimination (AUC) by group, with BPSN / BNSP AUC", "Fairness", _FAIR,
          params=(P("target"), P("protected"), P("score"), P("reference_group", "string", required=False)),
          description="""Ranking quality within each group: AUC with DeLong standard error and 95% CI, Gini,
and a z-test of the difference vs the reference group (independent samples, z = ΔAUC/√(se₁² + se₂²)).
Also the bias-AUCs of Borkan et al. (2019): BPSN AUC (background-positive, subgroup-negative: subgroup
non-events vs other groups' events — low values mean the group's non-events are scored too high) and
BNSP AUC (subgroup events vs other groups' non-events — low values mean the group's events are scored too
low).""",
          references=("DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44",
                      "Sun & Xu (2014), Fast implementation of DeLong's algorithm, IEEE Signal Processing Letters 21",
                      "Borkan et al. (2019), Nuanced metrics for measuring unintended bias with real data for text "
                      "classification, WWW Companion", *_REFS_GENERAL))
def auc_by_group(ctx: RunContext, target, protected, score, reference_group=None) -> Outcome:
    d, dropped = _prep(ctx, target, protected, score, None, 0.5, need_score=True)
    ref = _ref(d, reference_group)
    y, s, g = d["y"].to_numpy(), d["s"].to_numpy(), d["g"].to_numpy()
    rows = []
    for grp in sorted(np.unique(g), key=str):
        i = g == grp
        auc, se = _delong(y[i], s[i])
        bpsn = _auc_only(s[~i & (y == 1)], s[i & (y == 0)]) if (~i & (y == 1)).any() and (i & (y == 0)).any() else np.nan
        bnsp = _auc_only(s[i & (y == 1)], s[~i & (y == 0)]) if (i & (y == 1)).any() and (~i & (y == 0)).any() else np.nan
        rows.append({"group": grp, "n": int(i.sum()), "events": int(y[i].sum()), "auc": auc, "auc_se": se,
                     "ci_low_95": auc - 1.959964 * se, "ci_high_95": auc + 1.959964 * se, "gini": 2 * auc - 1,
                     "bpsn_auc": bpsn, "bnsp_auc": bnsp})
    t = pd.DataFrame(rows)
    r0 = t.set_index("group").loc[ref]
    t["difference_vs_reference"] = t["auc"] - r0["auc"]
    z = t["difference_vs_reference"] / np.sqrt(t["auc_se"] ** 2 + r0["auc_se"] ** 2)
    t["z"] = np.where(t["group"] == ref, np.nan, z)
    t["p_value"] = 2 * stats.norm.sf(np.abs(t["z"]))
    overall, _ = _delong(y, s)
    return Outcome({"groups": len(t), "reference_group": ref, "overall_auc": overall,
                    "auc_max_minus_min": _spread(t, "auc")[0], "min_auc_group": t.sort_values("auc").iloc[0]["group"],
                    "min_bpsn_auc": float(t["bpsn_auc"].min()), "min_bnsp_auc": float(t["bnsp_auc"].min()), "n": len(d)},
                   {"AUC by group": t}, rows_used=len(d),
                   notes=dropped_note(dropped) + [f"Reference group: '{ref}'. Groups with a single class have NaN AUC."])


@register("fairness.intersectional", "Intersectional groups (two protected attributes combined)", "Fairness", _FAIR,
          params=(*_COMMON, P("protected_2", help="Second protected attribute; groups are the combinations"),
                  P("alpha", "number", default=0.0,
                    help="Dirichlet smoothing count for differential fairness (0 = empirical rates)")),
          description="""Groups formed by every observed combination of two protected attributes (e.g. sex ×
age band), because disparities can hide at intersections. Per cell: n, base rate, favourable rate, TPR, FPR,
PPV, with difference and ratio of the favourable rate vs the reference cell (largest by default). Summary:
favourable-rate max − min and min/max, TPR and FPR gaps, smallest cell size, and the empirical differential
fairness ε = max over cells i, j and outcomes ŷ of |ln P(ŷ|i) − ln P(ŷ|j)| (Foulds et al.), optionally with
Dirichlet smoothing α.""",
          references=("Foulds, Islam, Keya & Pan (2020), An intersectional definition of fairness, IEEE ICDE",
                      "Kearns, Neel, Roth & Wu (2018), Preventing fairness gerrymandering, ICML",
                      "Buolamwini & Gebru (2018), Gender shades, FAT*", *_REFS_GENERAL))
def intersectional(ctx: RunContext, target, protected, protected_2, score=None, predicted=None, threshold=0.5,
                   reference_group=None, favourable_label=0, alpha=0.0) -> Outcome:
    d, dropped = _prep(ctx, target, protected, score, predicted, threshold, extra=(protected_2,))
    d["g"] = d["g"] + " | " + d[protected_2].astype(str)
    ref = _ref(d, reference_group)
    t = _rates(d, favourable_label)
    cmp = _vs_ref(t, ref, ["favourable_rate"]).drop(columns="metric")
    tab = t[["group", "n", "base_rate", "favourable_rate", "tpr", "fpr", "ppv"]].merge(
        cmp[["group", "difference_vs_reference", "ratio_vs_reference", "z", "p_value"]], on="group")
    k = 2
    pos = (t["TP"] + t["FP"]).to_numpy(float)
    nn = t["n"].to_numpy(float)
    with np.errstate(divide="ignore"):
        eps = 0.0
        for cnt in (pos, nn - pos):
            r = np.log((cnt + alpha) / (nn + k * alpha))
            eps = max(eps, float(np.max(r) - np.min(r)))
    fg, fr = _spread(t, "favourable_rate")
    return Outcome({"cells": len(t), "reference_cell": ref, "smallest_cell_n": int(t["n"].min()),
                    "favourable_rate_max_minus_min": fg, "favourable_rate_min_over_max": fr,
                    "tpr_max_minus_min": _spread(t, "tpr")[0], "fpr_max_minus_min": _spread(t, "fpr")[0],
                    "differential_fairness_epsilon": eps, "n": len(d)},
                   {"Intersectional groups": tab}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label)
                   + (["ε is infinite when some cell has a zero rate; use alpha > 0 to smooth."] if not np.isfinite(eps)
                      else []))


@register("fairness.group_difference_tests", "Omnibus tests of group differences (chi-square, Fisher)",
          "Fairness", _FAIR, params=_COMMON,
          description="""Chi-square tests of independence between the protected group and: the favourable
prediction (all rows; demographic parity), the predicted label among Y = 1 (TPR, equal opportunity) and
among Y = 0 (FPR), the outcome among Ŷ = 1 (PPV, predictive parity) and the outcome itself (base rate).
H0 for each: the rate is the same in every group. Reports chi-square, df, p-value, Cramér's V, the minimum
expected cell count (chi-square approximation is poor when small) and, for two groups, Fisher's exact
p-value. Pairwise two-proportion z-tests vs the reference group are in the specific parity tests.""",
          references=("Agresti (2013), Categorical Data Analysis, 3rd ed., Wiley",
                      "Fisher (1922), On the interpretation of χ² from contingency tables, JRSS 85", *_REFS_GENERAL))
def group_difference_tests(ctx: RunContext, target, protected, score=None, predicted=None, threshold=0.5,
                           reference_group=None, favourable_label=0) -> Outcome:
    d, dropped = _prep(ctx, target, protected, score, predicted, threshold)
    ref = _ref(d, reference_group)
    d = d.assign(fav=(d["yhat"] == favourable_label).astype(int))
    allr = np.ones(len(d), bool)
    spec = [("favourable rate (demographic parity)", allr, "fav"),
            ("TPR (among Y = 1)", (d["y"] == 1).to_numpy(), "yhat"),
            ("FPR (among Y = 0)", (d["y"] == 0).to_numpy(), "yhat"),
            ("PPV (among predicted 1)", (d["yhat"] == 1).to_numpy(), "y"),
            ("base rate (outcome)", allr, "y")]
    t = pd.DataFrame([{"quantity": q} | _chi2(d, m, c) for q, m, c in spec])
    return Outcome({"groups": int(d["g"].nunique()), "n": len(d),
                    **{f"p_value_{k}": v for k, v in zip(["favourable", "tpr", "fpr", "ppv", "base_rate"], t["p_value"])}},
                   {"Group difference tests": t}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label)
                   + ["Chi-square without continuity correction; check min_expected (< 5 = unreliable approximation)."])


@register("fairness.conditional_parity", "Conditional statistical parity within strata (CMH test)", "Fairness", _FAIR,
          params=(*_COMMON, P("segment", help="Legitimate stratifying factor (e.g. risk grade, product, income band)")),
          description="""Conditional statistical parity: favourable rates compared between groups WITHIN strata
of a legitimate factor, so that differences explained by that factor are removed. Stratum × group table of
favourable rates. For each group vs the reference: Mantel–Haenszel pooled odds ratio of a favourable
prediction (group vs reference) across strata, Cochran–Mantel–Haenszel chi-square test (H0: common odds
ratio = 1, i.e. no association within strata) and the Breslow–Day test of homogeneous odds ratios across
strata. Strata where either group is absent are skipped.""",
          references=("Mantel & Haenszel (1959), J. National Cancer Institute 22",
                      "Corbett-Davies et al. (2017), Algorithmic decision making and the cost of fairness, KDD",
                      "Kamiran, Žliobaitė & Calders (2013), Quantifying explainable discrimination and removing "
                      "illegal discrimination, Knowledge and Information Systems 35", *_REFS_GENERAL))
def conditional_parity(ctx: RunContext, target, protected, segment, score=None, predicted=None, threshold=0.5,
                       reference_group=None, favourable_label=0) -> Outcome:
    from statsmodels.stats.contingency_tables import StratifiedTable
    d, dropped = _prep(ctx, target, protected, score, predicted, threshold, extra=(segment,))
    ref = _ref(d, reference_group)
    d = d.assign(fav=(d["yhat"] == favourable_label).astype(int), stratum=d[segment].astype(str))
    st = d.groupby(["stratum", "g"]).agg(n=("fav", "size"), favourable_rate=("fav", "mean")).reset_index() \
        .rename(columns={"g": "group"})
    rows = []
    for g in sorted(d["g"].unique(), key=str):
        if g == ref:
            continue
        tables = []
        for s_, x in d.groupby("stratum", sort=True):
            a, b = x[x["g"] == g]["fav"], x[x["g"] == ref]["fav"]
            if len(a) and len(b):
                tables.append(np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]], dtype=float))
        row = {"group": g, "strata_used": len(tables)}
        if tables:
            try:
                tab = StratifiedTable(tables)
                tn = tab.test_null_odds(correction=False)
                row |= {"mh_odds_ratio": float(tab.oddsratio_pooled), "cmh_chi2": float(tn.statistic),
                        "cmh_p_value": float(tn.pvalue)}
                try:
                    te = tab.test_equal_odds()
                    row |= {"breslow_day_chi2": float(te.statistic), "breslow_day_p_value": float(te.pvalue)}
                except Exception:
                    pass
            except Exception as exc:
                row["error"] = str(exc)[:120]
        rows.append(row)
    t = pd.DataFrame(rows)
    p = t["cmh_p_value"] if "cmh_p_value" in t else pd.Series(dtype=float)
    return Outcome({"groups": int(d["g"].nunique()), "strata": int(d["stratum"].nunique()), "reference_group": ref,
                    "min_mh_odds_ratio": float(t["mh_odds_ratio"].min()) if "mh_odds_ratio" in t else np.nan,
                    "max_mh_odds_ratio": float(t["mh_odds_ratio"].max()) if "mh_odds_ratio" in t else np.nan,
                    "min_cmh_p_value": float(p.min()) if len(p) else np.nan, "n": len(d)},
                   {"Mantel–Haenszel vs reference": t, "Favourable rate by stratum and group": st}, rows_used=len(d),
                   notes=_common_notes(d, ref, dropped, predicted, threshold, favourable_label)
                   + ["Odds ratio > 1: the group is MORE likely than the reference to get the favourable prediction "
                      "within strata."])


@register("fairness.score_distribution", "Score distribution by group (SMD, KS, Mann–Whitney)", "Fairness", _FAIR,
          params=(P("target", required=False, help="Optional: also compare scores within outcome classes"),
                  P("protected"), P("score"), P("reference_group", "string", required=False)),
          description="""Compares the distribution of the model score across groups, independently of any
threshold: per group n, mean, median, SD, standardised mean difference vs the reference
SMD = (x̄_g − x̄_ref)/√((s_g² + s_ref²)/2), two-sample Kolmogorov–Smirnov and Mann–Whitney tests vs the
reference (H0: same distribution); omnibus Kruskal–Wallis test. With a target, the same comparison is
repeated within Y = 0 and within Y = 1 (score differences among people with the same outcome).""",
          references=("Austin (2009), Balance diagnostics for comparing the distribution of baseline covariates, "
                      "Statistics in Medicine 28", "Kruskal & Wallis (1952), JASA 47", *_REFS_GENERAL))
def score_distribution(ctx: RunContext, protected, score, target=None, reference_group=None) -> Outcome:
    sub, dropped = complete(ctx.df, _uniq([protected, score, target]))
    d = pd.DataFrame({"g": sub[protected].astype(str), "s": num(sub, score)}, index=sub.index)
    if target:
        d["y"] = binary(sub, target)
    bad = d.drop(columns="g").isna().any(axis=1)
    d, dropped = d[~bad], dropped + int(bad.sum())
    if d["g"].nunique() < 2:
        raise NotApplicable("Fewer than two groups.")
    ref = _ref(d, reference_group)

    def _cmp(x, label):
        r = x.loc[x["g"] == ref, "s"].to_numpy()
        rows = []
        for g in sorted(x["g"].unique(), key=str):
            v = x.loc[x["g"] == g, "s"].to_numpy()
            row = {"subset": label, "group": g, "n": len(v), "mean": v.mean(), "median": float(np.median(v)),
                   "sd": v.std(ddof=1) if len(v) > 1 else np.nan}
            if g != ref and len(v) and len(r):
                pool = np.sqrt((np.var(v, ddof=1) + np.var(r, ddof=1)) / 2) if len(v) > 1 and len(r) > 1 else np.nan
                row |= {"smd_vs_reference": (v.mean() - r.mean()) / pool if pool else np.nan,
                        "ks": stats.ks_2samp(v, r).statistic, "ks_p_value": stats.ks_2samp(v, r).pvalue,
                        "mann_whitney_p_value": stats.mannwhitneyu(v, r, alternative="two-sided").pvalue}
            rows.append(row)
        return rows
    rows = _cmp(d, "all")
    if target:
        for yv in (0, 1):
            x = d[d["y"] == yv]
            if x["g"].nunique() >= 2 and (x["g"] == ref).any():
                rows += _cmp(x, f"Y = {yv}")
    t = pd.DataFrame(rows)
    groups = [v.to_numpy() for _, v in d.groupby("g")["s"]]
    kw = stats.kruskal(*groups) if all(len(v) for v in groups) and np.ptp(d["s"]) > 0 else None
    allr = t[(t["subset"] == "all") & (t["group"] != ref)]
    return Outcome({"groups": int(d["g"].nunique()), "reference_group": ref,
                    "max_abs_smd": float(allr["smd_vs_reference"].abs().max()) if "smd_vs_reference" in allr else np.nan,
                    "kruskal_wallis_H": float(kw.statistic) if kw else np.nan,
                    "kruskal_wallis_p_value": float(kw.pvalue) if kw else np.nan, "n": len(d)},
                   {"Score distribution by group": t}, rows_used=len(d),
                   notes=dropped_note(dropped) + [f"Reference group: '{ref}'."])


@register("fairness.counterfactual_flip", "Counterfactual flip test of the protected attribute (real model)",
          "Fairness", _FAIR,
          params=(P("model", "model", help="Name of the loaded model under validation"), P("protected"),
                  P("features", "columns", required=False, help="Model input columns; default the model's own"),
                  P("threshold", "number", default=0.5)),
          description="""Direct-discrimination check on the model under validation: for every row, the protected
attribute (which must be a model input) is replaced by each other observed value, all other inputs unchanged,
and the model is re-scored. Per (original value → counterfactual value): n, mean score change, mean and
maximum absolute change, share of predicted labels that flip at the threshold. Summary: share of rows whose
label flips under at least one counterfactual, and the largest mean |change|. NotApplicable when the
attribute is not a model input (then only proxy effects are possible; use the parity tests). Proxies of the
attribute are NOT changed, so this measures direct use only.""",
          references=("Kusner, Loftus, Russell & Silva (2017), Counterfactual fairness, NeurIPS",
                      "Garg et al. (2019), Counterfactual fairness in text classification through robustness, AIES",
                      *_REFS_GENERAL))
def counterfactual_flip(ctx: RunContext, model, protected, features=None, threshold=0.5) -> Outcome:
    m = ctx.models[model]
    feats = model_features(m, ctx.df, features)
    if protected not in feats:
        raise NotApplicable(f"'{protected}' is not an input of model '{model}', so it cannot be flipped. Direct "
                            "use is excluded; check proxies with the parity tests.")
    sub, dropped = complete(ctx.df, feats)
    X = sub[feats]
    vals = sorted(X[protected].unique(), key=str)
    if len(vals) < 2:
        raise NotApplicable(f"'{protected}' takes a single value.")
    base = np.asarray(predict_scores(m, X), dtype=float)
    orig = X[protected].to_numpy()
    flipped_any = np.zeros(len(X), bool)
    rows = []
    for v in vals:
        Xc = X.copy()
        Xc[protected] = v
        s = np.asarray(predict_scores(m, Xc), dtype=float)
        diff = s - base
        flip = (s >= threshold) != (base >= threshold)
        flipped_any |= flip
        for g in vals:
            if g == v:
                continue
            i = orig == g
            if not i.any():
                continue
            rows.append({"original_value": str(g), "counterfactual_value": str(v), "n": int(i.sum()),
                         "mean_score_change": float(diff[i].mean()), "mean_abs_score_change": float(np.abs(diff[i]).mean()),
                         "max_abs_score_change": float(np.abs(diff[i]).max()), "share_label_flips": float(flip[i].mean())})
    t = pd.DataFrame(rows)
    top = t.sort_values(["mean_abs_score_change", "original_value"], ascending=[False, True]).iloc[0]
    return Outcome({"n": len(X), "values": len(vals), "share_rows_any_label_flip": float(flipped_any.mean()),
                    "max_mean_abs_score_change": top["mean_abs_score_change"],
                    "max_change_pair": f"{top['original_value']} -> {top['counterfactual_value']}",
                    "mean_abs_score_change_overall": float(np.average(t["mean_abs_score_change"], weights=t["n"]))},
                   {"Counterfactual changes": t}, rows_used=len(X),
                   notes=dropped_note(dropped) + [f"Threshold {threshold}. Proxies of '{protected}' are unchanged; "
                                                  "flipping can create combinations not seen in the data."])
