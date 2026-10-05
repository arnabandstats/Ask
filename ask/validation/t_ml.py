"""Machine-learning model validation: performance, overfitting, explainability, robustness,
data leakage, multicollinearity and unsupervised (clustering / PCA) diagnostics.

Tests that explain, refit or stress a model take `model` (a model loaded with load_model) and
score the REAL model under validation through ask.validation.models.predict_scores. Explanations
and perturbations are evaluated on the rows provided, so validators should pass a held-out or
out-of-time sample; every perturbation and resample is seeded from the run context.

Exactly one test fits a stand-in model, `ml.surrogate_explainability`, and it says so in its
name, summary and notes: a RandomForest fitted on a training split and evaluated on a held-out
split. It describes the data (or the model's score, when given), never the model's internals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import (NotApplicable, Outcome, P, RunContext, binary, complete, dropped_note,
                                 num, register, require_two_classes, split_samples)
from ask.validation.models import model_features, predict_labels, predict_scores

_CLS = ("ml_classification", "pd", "aml", "ews")
_REG = ("ml_regression", "lgd")
_SUP = ("ml_classification", "ml_regression", "pd", "aml", "ews", "lgd")
_UNS = ("ml_unsupervised",)

_MODEL = P("model", "model", help="Name of the loaded model under validation (see load_model)")
_MODEL_OPT = P("model", "model", required=False,
               help="Loaded model to score the rows with (instead of a score / predicted column)")
_FEATS = P("features", "columns", required=False,
           help="Model input columns; default: the feature names recorded in the model")
_TARGET_OPT = P("target", required=False, help="Observed binary outcome (classification metrics)")
_ACTUAL_OPT = P("actual", required=False, help="Observed continuous outcome (regression metrics)")
_SCORE_OPT = P("score", required=False, help="Model score / probability column (instead of `model`)")
_THRESHOLD = P("threshold", "number", default=0.5, help="Classification cut-off: predicted 1 when score >= threshold")
_METRIC = P("metric", "string", required=False,
            help="Metric name (auc, gini, ks, pr_auc, log_loss, brier, accuracy, balanced_accuracy, f1, "
                 "mcc, precision, recall | rmse, mae, r2); default auc (classification) or rmse (regression)")
_HELDOUT_NOTE = ("Evaluated on the rows provided: pass a held-out / out-of-time sample, because results on "
                 "the training sample describe the fit, not generalisation.")


# ── data helpers ───────────────────────────────────────────────────────────

def _uniq(cols) -> list[str]:
    return list(dict.fromkeys(c for c in cols if c))


def _rows(ctx: RunContext, cols, score=None, model=None, features=None):
    """(complete rows, prediction per row, rows dropped, source label, model features or None)."""
    feats = None
    if score:
        sub, dropped = complete(ctx.df, _uniq([*cols, score]))
        s = num(sub, score).to_numpy()
        src = f"column '{score}'"
    elif model:
        m = ctx.models[model]
        feats = model_features(m, ctx.df, features)
        sub, dropped = complete(ctx.df, _uniq([*cols, *feats]))
        if sub.empty:
            raise NotApplicable("No rows with complete inputs.")
        s = np.asarray(predict_scores(m, sub[feats]), dtype=float)
        src = f"model '{model}'"
    else:
        raise ValueError("Give a score / predicted column, or `model` (a loaded model).")
    ok = np.isfinite(s)
    if not ok.all():
        dropped += int((~ok).sum())
        sub, s = sub[ok], s[ok]
    if len(sub) == 0:
        raise NotApplicable("No rows with complete inputs.")
    return sub, s, dropped, src, feats


def _task(target, actual) -> tuple[str, str]:
    if bool(target) == bool(actual):
        raise ValueError("Give exactly one of `target` (binary outcome) or `actual` (continuous outcome).")
    return (target, "classification") if target else (actual, "regression")


def _yvals(sub: pd.DataFrame, col: str, task: str) -> np.ndarray:
    return (binary(sub, col) if task == "classification" else num(sub, col)).to_numpy()


def _model_xy(ctx: RunContext, model, features, target=None, actual=None, need_y=True):
    """Model, feature list, complete rows, X, y (or None), task, rows dropped."""
    m = ctx.models[model]
    feats = model_features(m, ctx.df, features)
    ycol, task = (None, "classification" if hasattr(m.obj, "predict_proba") else "regression")
    if need_y or target or actual:
        ycol, task = _task(target, actual)
    sub, dropped = complete(ctx.df, _uniq([*feats, ycol]))
    y = None
    if ycol:
        y = _yvals(sub, ycol, task)
        ok = np.isfinite(y)
        dropped += int((~ok).sum())
        sub, y = sub[ok], y[ok]
        if task == "classification":
            require_two_classes(y, ycol)
            y = y.astype(int)
    if len(sub) < 10:
        raise NotApplicable(f"Only {len(sub)} complete rows; at least 10 are needed.")
    return m, feats, sub, sub[feats], y, task, dropped


def _sample_rows(X: pd.DataFrame, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    return X.iloc[np.sort(rng.choice(len(X), int(max_rows), replace=False))]


def _numeric_feats(X: pd.DataFrame, feats, min_unique: int = 2) -> list[str]:
    return [f for f in feats if pd.api.types.is_numeric_dtype(X[f]) and not pd.api.types.is_bool_dtype(X[f])
            and X[f].nunique() >= min_unique]


def _numeric_matrix(ctx: RunContext, features, extra=()) -> tuple[pd.DataFrame, int]:
    bad = [f for f in features if not pd.api.types.is_numeric_dtype(ctx.df[f])]
    if bad:
        raise ValueError(f"Non-numeric feature(s) {bad}: encode them first.")
    sub, dropped = complete(ctx.df, _uniq([*features, *extra]))
    return sub, dropped


# ── metrics ────────────────────────────────────────────────────────────────

def _div(a, b):
    return a / b if b else np.nan


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < np.sum(y) < len(y) else np.nan


def _ks(y, s) -> float:
    from sklearn.metrics import roc_curve
    if not 0 < np.sum(y) < len(y):
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def _ap(y, s) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if 0 < np.sum(y) < len(y) else np.nan


def _is_prob(s) -> bool:
    return bool(len(s)) and float(np.min(s)) >= 0 and float(np.max(s)) <= 1


def _logloss(y, s) -> float:
    if not _is_prob(s):
        return np.nan
    p = np.clip(s, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y, s) -> float:
    return float(np.mean((s - y) ** 2)) if _is_prob(s) else np.nan


def _label_metrics(y, yhat, beta: float = 1.0) -> dict:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    n = tp + fp + tn + fn
    rec, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    b2 = beta ** 2
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "accuracy": po,
            "balanced_accuracy": (rec + spec) / 2, "precision": _div(tp, tp + fp), "recall": rec,
            "specificity": spec, "npv": _div(tn, tn + fn), "f1": _div(2 * tp, 2 * tp + fp + fn),
            "f_beta": _div((1 + b2) * tp, (1 + b2) * tp + b2 * fn + fp),
            "mcc": (tp * tn - fp * fn) / den if den else 0.0,
            "cohen_kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan}


def _binary_metrics(y, s, threshold: float, beta: float = 1.0) -> dict:
    out = _label_metrics(y, s >= threshold, beta)
    auc = _auc(y, s)
    out |= {"roc_auc": auc, "gini": 2 * auc - 1, "pr_auc": _ap(y, s), "ks": _ks(y, s),
            "log_loss": _logloss(y, s), "brier": _brier(y, s)}
    return out


def _reg_metrics(a, p, n_features=None) -> dict:
    e = a - p
    n = len(a)
    sst = float(np.sum((a - a.mean()) ** 2))
    r2 = 1 - float(np.sum(e ** 2)) / sst if sst > 0 else np.nan
    nz = a != 0
    den = np.abs(a) + np.abs(p)
    smape = np.where(den > 0, 2 * np.abs(e) / np.where(den > 0, den, 1), 0.0)
    adj = (1 - (1 - r2) * (n - 1) / (n - n_features - 1)) if n_features and n - n_features - 1 > 0 else np.nan
    return {"n": n, "rmse": float(np.sqrt(np.mean(e ** 2))), "mse": float(np.mean(e ** 2)),
            "mae": float(np.mean(np.abs(e))), "median_ae": float(np.median(np.abs(e))),
            "max_error": float(np.max(np.abs(e))),
            "mape": float(np.mean(np.abs(e[nz]) / np.abs(a[nz]))) if nz.any() else np.nan,
            "smape": float(np.mean(smape)), "r2": r2, "adjusted_r2": adj,
            "explained_variance": 1 - float(np.var(e)) / float(np.var(a)) if np.var(a) > 0 else np.nan,
            "bias_mean_predicted_minus_actual": float(np.mean(-e)),
            "pearson_r": float(np.corrcoef(a, p)[0, 1]) if np.std(a) > 0 and np.std(p) > 0 else np.nan,
            "spearman_rho": float(stats.spearmanr(a, p).statistic) if np.std(a) > 0 and np.std(p) > 0
            else np.nan}


# name: (task, higher_is_better, fn(y, s, threshold))
_METRICS = {
    "auc": ("classification", True, lambda y, s, t: _auc(y, s)),
    "gini": ("classification", True, lambda y, s, t: 2 * _auc(y, s) - 1),
    "ks": ("classification", True, lambda y, s, t: _ks(y, s)),
    "pr_auc": ("classification", True, lambda y, s, t: _ap(y, s)),
    "log_loss": ("classification", False, lambda y, s, t: _logloss(y, s)),
    "brier": ("classification", False, lambda y, s, t: _brier(y, s)),
    **{k: ("classification", True, (lambda k: lambda y, s, t: _label_metrics(y, s >= t)[k])(k))
       for k in ("accuracy", "balanced_accuracy", "f1", "mcc", "precision", "recall")},
    "rmse": ("regression", False, lambda y, s, t: float(np.sqrt(np.mean((y - s) ** 2)))),
    "mae": ("regression", False, lambda y, s, t: float(np.mean(np.abs(y - s)))),
    "r2": ("regression", True, lambda y, s, t: _reg_metrics(y, s)["r2"]),
    "bias": ("regression", None, lambda y, s, t: float(np.mean(s - y))),
}


def _metric_spec(name: str | None, task: str, for_importance: bool = False):
    name = (name or ("auc" if task == "classification" else "rmse")).lower()
    if name not in _METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose from {sorted(_METRICS)}.")
    mtask, hib, fn = _METRICS[name]
    if mtask != task:
        raise ValueError(f"Metric '{name}' is a {mtask} metric but the outcome is {task}.")
    if for_importance and hib is None:
        raise ValueError(f"Metric '{name}' has no better/worse direction; choose another.")
    return name, hib, fn


class _Facade:
    """Minimal estimator facade so sklearn.inspection can drive any loaded model."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(predict_scores(self.model, X), dtype=float)


def _perm_importance(model, X, y, metric_fn, hib, threshold, n_repeats, seed):
    """sklearn permutation importance; importance = metric deterioration (positive = feature matters)."""
    from sklearn.inspection import permutation_importance
    sign = 1.0 if hib else -1.0
    r = permutation_importance(_Facade(model), X, y,
                               scoring=lambda est, X_, y_: sign * metric_fn(y_, est.predict(X_), threshold),
                               n_repeats=int(n_repeats), random_state=seed)
    return r.importances      # (features, repeats)


def _refittable(ctx: RunContext, model):
    """A fresh unfitted clone of the model (sklearn-compatible only) and the params seeded."""
    from sklearn.base import clone
    m = ctx.models[model]
    obj = m.obj
    if not (hasattr(obj, "fit") and hasattr(obj, "get_params")):
        raise NotApplicable(f"Model '{model}' is a {type(obj).__name__} ({m.flavour}); refitting needs an "
                            "sklearn-compatible estimator (fit/get_params) that sklearn.base.clone can copy. "
                            "Ask the model owner for the training pipeline, or use the held-out tests instead.")
    try:
        est = clone(obj)
    except Exception as exc:
        raise NotApplicable(f"Model '{model}' cannot be cloned for refitting: {exc}") from exc
    seeded = sorted(k for k, v in est.get_params().items() if k.endswith("random_state") and v is None)
    if seeded:
        est.set_params(**{k: ctx.seed for k in seeded})
    return est, seeded


def _folds(y, k: int, task: str, seed: int):
    from sklearn.model_selection import KFold, StratifiedKFold
    if k < 2:
        raise ValueError("k must be at least 2.")
    if task == "classification":
        if np.bincount(y.astype(int)).min() < k:
            raise NotApplicable(f"The rarer class has fewer than k={k} rows; stratified {k}-fold is impossible.")
        cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    else:
        if len(y) < 2 * k:
            raise NotApplicable(f"Too few rows ({len(y)}) for {k}-fold cross-validation.")
        cv = KFold(k, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def _grid(x: pd.Series, points: int, lo: float, hi: float) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(x):
        return np.array(sorted(x.value_counts().index[:points], key=str), dtype=object)
    u = np.unique(x.to_numpy())
    if len(u) <= points:
        return u
    return np.unique(np.quantile(x.to_numpy(dtype=float), np.linspace(lo, hi, points)))


def _ice(model, X: pd.DataFrame, feature: str, grid) -> np.ndarray:
    """Individual conditional expectation curves: rows x grid points."""
    n = len(X)
    big = pd.concat([X] * len(grid), ignore_index=True)
    big[feature] = np.repeat(np.asarray(grid), n)
    return np.asarray(predict_scores(model, big), dtype=float).reshape(len(grid), n).T


def _check_quantiles(lo, hi):
    if not 0 <= lo < hi <= 1:
        raise ValueError("Grid quantiles must satisfy 0 <= lower < upper <= 1.")


# ── classification performance ─────────────────────────────────────────────

@register("ml.classification_metrics", "Binary classification performance at a threshold", "Performance", _CLS,
          params=(P("target"), _SCORE_OPT, _MODEL_OPT, _FEATS, _THRESHOLD,
                  P("beta", "number", default=1.0, help="beta of the F-beta score (beta > 1 favours recall)"),
                  P("n_boot", "integer", default=0, help="Bootstrap resamples for percentile 95% CIs (0 = none)")),
          description="""Confusion matrix and every standard binary metric for score >= threshold:
accuracy, balanced accuracy ((TPR+TNR)/2), precision (PPV), recall (TPR), specificity (TNR), NPV, F1,
F-beta = (1+β²)TP/((1+β²)TP+β²FN+FP), Matthews correlation coefficient
MCC = (TP·TN−FP·FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)), Cohen's kappa = (p_o−p_e)/(1−p_e); and the
threshold-free ROC-AUC, Gini = 2·AUC−1, PR-AUC (average precision), KS, log loss and Brier score
(the last two only for scores in [0, 1]). Scores come from a score column or from the loaded model
(P(class 1)). Optional percentile bootstrap CIs resample rows with a seeded generator. Undefined ratios
(zero denominators) are reported as NaN, not 0.""",
          references=("Fawcett (2006), An introduction to ROC analysis, Pattern Recognition Letters 27",
                      "Matthews (1975), Biochimica et Biophysica Acta 405; Chicco & Jurman (2020), BMC Genomics 21",
                      "Cohen (1960), A coefficient of agreement for nominal scales, Educ. Psychol. Meas. 20",
                      "Saito & Rehmsmeier (2015), The precision-recall plot, PLoS ONE 10",
                      "Efron & Tibshirani (1993), An Introduction to the Bootstrap"),
          suite=False)
def classification_metrics(ctx: RunContext, target, score=None, model=None, features=None, threshold=0.5,
                           beta=1.0, n_boot=0) -> Outcome:
    sub, s, dropped, src, _ = _rows(ctx, [target], score, model, features)
    y = binary(sub, target).to_numpy()
    ok = np.isfinite(y)
    dropped += int((~ok).sum())
    y, s = y[ok].astype(int), s[ok]
    require_two_classes(y, target)
    m = _binary_metrics(y, s, threshold, beta)
    names = [k for k in m if k not in {"TP", "FP", "TN", "FN"}]
    tab = pd.DataFrame({"metric": names, "value": [m[k] for k in names]})
    notes = dropped_note(dropped) + [f"Scores from {src}; predicted 1 when score >= {threshold}."]
    if not _is_prob(s):
        notes.append("Scores are not in [0, 1]: log loss and Brier score are not computed.")
    if n_boot and n_boot > 0:
        rng = ctx.rng()
        n = len(y)
        boots = []
        for _ in range(int(n_boot)):
            i = rng.integers(0, n, n)
            if 0 < y[i].sum() < n:
                bm = _binary_metrics(y[i], s[i], threshold, beta)
                boots.append([bm[k] for k in names])
        b = np.asarray(boots, dtype=float)
        tab["ci_low_95"] = np.nanpercentile(b, 2.5, axis=0)
        tab["ci_high_95"] = np.nanpercentile(b, 97.5, axis=0)
        notes.append(f"Percentile bootstrap CIs from {len(b)} resamples (seed {ctx.seed}).")
    cm = pd.DataFrame({"actual": ["1 (event)", "0 (non-event)"], "predicted_1": [m["TP"], m["FP"]],
                       "predicted_0": [m["FN"], m["TN"]]})
    summary = {"n": len(y), "events": int(y.sum()), "event_rate": float(y.mean()), "threshold": threshold}
    summary |= {k: m[k] for k in ("accuracy", "balanced_accuracy", "precision", "recall", "specificity",
                                  "f1", "f_beta", "mcc", "cohen_kappa", "roc_auc", "pr_auc", "log_loss", "brier")}
    return Outcome(summary, {"Confusion matrix": cm, "Metrics": tab}, notes=notes, rows_used=len(y))


@register("ml.threshold_sweep", "Metrics across classification thresholds (Youden, cost, F-beta optima)",
          "Performance", _CLS,
          params=(P("target"), _SCORE_OPT, _MODEL_OPT, _FEATS,
                  P("thresholds", "list", required=False,
                    help="Thresholds to evaluate; default the 0, 1, ..., 100th percentiles of the score"),
                  P("cost_fp", "number", default=1.0, help="Cost of one false positive"),
                  P("cost_fn", "number", default=1.0, help="Cost of one false negative"),
                  P("beta", "number", default=1.0)),
          description="""For each threshold t (predict 1 when score >= t): TP, FP, TN, FN, predicted-positive
rate, precision, recall (TPR), specificity, F-beta, MCC, Youden's J = TPR + TNR − 1 and the expected
misclassification cost C(t) = cost_fp·FP + cost_fn·FN. Reports the thresholds maximising J, F-beta and MCC
and minimising cost on the evaluated grid, plus the Bayes-optimal threshold cost_fp/(cost_fp+cost_fn),
which is optimal only if the scores are calibrated probabilities. Optima are chosen on the data provided,
so they are in-sample for these data.""",
          references=("Youden (1950), Index for rating diagnostic tests, Cancer 3",
                      "Elkan (2001), The foundations of cost-sensitive learning, IJCAI",
                      "Fawcett (2006), An introduction to ROC analysis, Pattern Recognition Letters 27"))
def threshold_sweep(ctx: RunContext, target, score=None, model=None, features=None, thresholds=None,
                    cost_fp=1.0, cost_fn=1.0, beta=1.0) -> Outcome:
    sub, s, dropped, src, _ = _rows(ctx, [target], score, model, features)
    y = binary(sub, target).to_numpy()
    ok = np.isfinite(y)
    dropped += int((~ok).sum())
    y, s = y[ok].astype(int), s[ok]
    require_two_classes(y, target)
    t = np.unique(np.asarray([float(v) for v in thresholds], dtype=float)) if thresholds else \
        np.unique(np.quantile(s, np.linspace(0, 1, 101)))
    order = np.argsort(s, kind="mergesort")
    ss, ys = s[order], y[order]
    cpos = np.concatenate([[0], np.cumsum(ys)])
    n, pos = len(y), int(y.sum())
    idx = np.searchsorted(ss, t, side="left")           # rows idx..n-1 have score >= t
    tp = pos - cpos[idx]
    pp = n - idx
    fp, fn = pp - tp, pos - tp
    tn = (n - pos) - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(pp > 0, tp / np.maximum(pp, 1), np.nan)
        rec, spec = tp / pos, tn / (n - pos)
        b2 = beta ** 2
        fb = np.where(tp + fp + fn > 0, (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp), np.nan)
        den = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    cost = cost_fp * fp + cost_fn * fn
    tab = pd.DataFrame({"threshold": t, "predicted_positive_rate": pp / n, "TP": tp, "FP": fp, "TN": tn,
                        "FN": fn, "precision": prec, "recall_tpr": rec, "specificity_tnr": spec,
                        "f_beta": fb, "mcc": mcc, "youden_j": rec + spec - 1, "expected_cost": cost,
                        "cost_per_obs": cost / n})
    best = {k: tab.loc[tab[k].idxmax()] for k in ("youden_j", "f_beta", "mcc")}
    cmin = tab.loc[tab["expected_cost"].idxmin()]
    summary = {"n": n, "thresholds_evaluated": len(t),
               "youden_optimal_threshold": best["youden_j"]["threshold"], "max_youden_j": best["youden_j"]["youden_j"],
               "f_beta_optimal_threshold": best["f_beta"]["threshold"], "max_f_beta": best["f_beta"]["f_beta"],
               "mcc_optimal_threshold": best["mcc"]["threshold"], "max_mcc": best["mcc"]["mcc"],
               "cost_optimal_threshold": cmin["threshold"], "min_expected_cost": cmin["expected_cost"],
               "bayes_threshold_if_calibrated": cost_fp / (cost_fp + cost_fn)}
    import plotly.graph_objects as go
    fig = go.Figure()
    for c in ("precision", "recall_tpr", "specificity_tnr", "f_beta", "youden_j"):
        fig.add_trace(go.Scatter(x=tab["threshold"], y=tab[c], mode="lines", name=c))
    fig.update_layout(title="Classification metrics by threshold", xaxis_title="threshold", yaxis_title="value")
    return Outcome(summary, {"Metrics by threshold": tab}, [fig], rows_used=n,
                   notes=dropped_note(dropped) + [f"Scores from {src}. Optimal thresholds are selected on these "
                                                  "data (in-sample for them) and on the evaluated grid only."])


def _labels(s) -> pd.Series:
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s) and len(s) and np.all(np.mod(s.astype(float), 1) == 0):
        return s.astype("int64").astype(str)
    return s.astype(str)


@register("ml.multiclass_metrics", "Multi-class classification performance (macro / micro / weighted)",
          "Performance", ("ml_classification", "pd"),
          params=(P("target", help="Observed class label (two or more classes)"),
                  P("predicted", required=False, help="Predicted class label column (instead of `model`)"),
                  _MODEL_OPT, _FEATS),
          description="""Per-class precision, recall, F1 and support; macro (unweighted mean over classes),
micro (pooled counts) and weighted (support-weighted) averages; accuracy, balanced accuracy (mean
per-class recall), Cohen's kappa and multi-class MCC (Gorodkin's R_K); the full confusion matrix. With a
model that has predict_proba: multi-class log loss and one-vs-rest macro ROC-AUC. Labels are compared as
text (integer-valued numbers are normalised, so 1 and 1.0 match).""",
          references=("Sokolova & Lapalme (2009), A systematic analysis of performance measures for "
                      "classification tasks, Information Processing & Management 45",
                      "Gorodkin (2004), Comparing two K-category assignments by a K-category correlation "
                      "coefficient, Computational Biology and Chemistry 28",
                      "Hand & Till (2001), A simple generalisation of the AUC for multiple class "
                      "classification problems, Machine Learning 45"))
def multiclass_metrics(ctx: RunContext, target, predicted=None, model=None, features=None) -> Outcome:
    from sklearn import metrics as skm
    proba, classes = None, None
    if predicted:
        sub, dropped = complete(ctx.df, _uniq([target, predicted]))
        yp = _labels(sub[predicted].to_numpy())
        src = f"column '{predicted}'"
    elif model:
        m = ctx.models[model]
        feats = model_features(m, ctx.df, features)
        sub, dropped = complete(ctx.df, _uniq([target, *feats]))
        yp = _labels(predict_labels(m, sub[feats]))
        if hasattr(m.obj, "predict_proba") and hasattr(m.obj, "classes_"):
            proba = np.asarray(m.obj.predict_proba(sub[feats]), dtype=float)
            classes = _labels(np.asarray(m.obj.classes_)).tolist()
        src = f"model '{model}'"
    else:
        raise ValueError("Give `predicted` (a label column) or `model`.")
    yt = _labels(sub[target].to_numpy())
    if yt.nunique() < 2:
        raise NotApplicable("The target has a single class.")
    labels = sorted(set(yt) | set(yp), key=str)
    p, r, f, sup = skm.precision_recall_fscore_support(yt, yp, labels=labels, zero_division=np.nan)
    per = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": sup,
                        "predicted_count": [int((yp == c).sum()) for c in labels]})
    summary = {"n": len(yt), "classes": len(labels), "accuracy": skm.accuracy_score(yt, yp),
               "balanced_accuracy": float(np.nanmean(per.loc[per["support"] > 0, "recall"])),
               "cohen_kappa": skm.cohen_kappa_score(yt, yp), "mcc": skm.matthews_corrcoef(yt, yp)}
    for avg in ("macro", "micro", "weighted"):
        pa, ra, fa, _ = skm.precision_recall_fscore_support(yt, yp, labels=labels, average=avg,
                                                            zero_division=np.nan)
        summary |= {f"precision_{avg}": pa, f"recall_{avg}": ra, f"f1_{avg}": fa}
    notes = dropped_note(dropped) + [f"Predictions from {src}."]
    if proba is not None:
        if set(yt) <= set(classes):
            summary["log_loss"] = skm.log_loss(yt, proba, labels=classes)
            try:
                summary["roc_auc_ovr_macro"] = (skm.roc_auc_score((yt == classes[1]).astype(int), proba[:, 1])
                                                if len(classes) == 2 else
                                                skm.roc_auc_score(yt, proba, multi_class="ovr", average="macro",
                                                                  labels=classes))
            except ValueError as exc:
                notes.append(f"ROC-AUC not computed: {exc}")
        else:
            notes.append("Some observed classes are unknown to the model: log loss / AUC not computed.")
    cm = pd.DataFrame(skm.confusion_matrix(yt, yp, labels=labels), columns=[f"pred_{c}" for c in labels])
    cm.insert(0, "actual", labels)
    return Outcome(summary, {"Per-class report": per, "Confusion matrix": cm}, notes=notes, rows_used=len(yt))


@register("ml.probability_calibration", "Probability calibration: reliability table, ECE, Brier decomposition",
          "Calibration", _CLS,
          params=(P("target"), _SCORE_OPT, _MODEL_OPT, _FEATS, P("bins", "integer", default=10),
                  P("strategy", "string", default="quantile", choices=("quantile", "uniform"))),
          description="""Reliability of predicted probabilities. Rows are grouped into bins of predicted
probability (equal-count quantile bins or equal-width bins); per bin: n, mean predicted, observed event
rate and a Wilson 95% interval. Summary: Brier score with Murphy decomposition
(Brier ≈ reliability − resolution + uncertainty, exact when predictions are constant within bins),
expected calibration error ECE = Σ (n_b/n)·|ō_b − p̄_b|, maximum calibration error, calibration-in-the-large
(mean predicted − observed rate), calibration intercept and slope (logistic regression of the outcome on
logit(p); ideal 0 and 1), Spiegelhalter's z-test (H0: predictions are calibrated) and the
Hosmer–Lemeshow statistic on the bins (chi-square with bins−2 df).""",
          references=("Murphy (1973), A new vector partition of the probability score, J. Applied Meteorology 12",
                      "Spiegelhalter (1986), Probabilistic prediction in patient management, Statistics in Medicine 5",
                      "Hosmer & Lemeshow (2000), Applied Logistic Regression, 2nd ed.",
                      "Cox (1958), Two further applications of a model for binary regression, Biometrika 45",
                      "Naeini, Cooper & Hauskrecht (2015), Obtaining well calibrated probabilities using "
                      "Bayesian binning, AAAI"))
def probability_calibration(ctx: RunContext, target, score=None, model=None, features=None, bins=10,
                            strategy="quantile") -> Outcome:
    sub, s, dropped, src, _ = _rows(ctx, [target], score, model, features)
    y = binary(sub, target).to_numpy()
    ok = np.isfinite(y)
    dropped += int((~ok).sum())
    y, s = y[ok].astype(int), s[ok]
    require_two_classes(y, target)
    if not _is_prob(s):
        raise ValueError("Scores must be probabilities in [0, 1].")
    n = len(y)
    if strategy == "quantile":
        edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    g = pd.DataFrame({"b": b, "y": y, "p": s}).groupby("b")
    t = g.agg(n=("y", "size"), events=("y", "sum"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    t = t.reset_index(drop=True)
    t.insert(0, "bin", [f"[{edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in sorted(set(b))])
    z = 1.959963984540054
    ph, nn = t["observed_rate"], t["n"]
    centre = (ph + z ** 2 / (2 * nn)) / (1 + z ** 2 / nn)
    half = z * np.sqrt(ph * (1 - ph) / nn + z ** 2 / (4 * nn ** 2)) / (1 + z ** 2 / nn)
    t["wilson_low_95"], t["wilson_high_95"] = centre - half, centre + half
    t["observed_minus_predicted"] = t["observed_rate"] - t["mean_predicted"]
    w = t["n"] / n
    ybar = y.mean()
    rel = float(np.sum(w * (t["mean_predicted"] - t["observed_rate"]) ** 2))
    res = float(np.sum(w * (t["observed_rate"] - ybar) ** 2))
    pe = t["mean_predicted"].clip(1e-12, 1 - 1e-12)
    hl = float(np.sum((t["events"] - t["n"] * pe) ** 2 / (t["n"] * pe * (1 - pe))))
    hl_df = max(len(t) - 2, 1)
    var = np.sum((1 - 2 * s) ** 2 * s * (1 - s))
    spz = float(np.sum((y - s) * (1 - 2 * s)) / np.sqrt(var)) if var > 0 else np.nan
    import statsmodels.api as sm
    pc = np.clip(s, 1e-10, 1 - 1e-10)
    lg = np.log(pc / (1 - pc))
    try:
        fit = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial()).fit()
        slope_i, slope = float(fit.params[0]), float(fit.params[1])
        fit0 = sm.GLM(y, np.ones((n, 1)), family=sm.families.Binomial(), offset=lg).fit()
        citl = float(fit0.params[0])
    except Exception:
        slope_i = slope = citl = np.nan
    summary = {"n": n, "bins": len(t), "brier": float(np.mean((s - y) ** 2)), "brier_reliability": rel,
               "brier_resolution": res, "brier_uncertainty": float(ybar * (1 - ybar)),
               "ece": float(np.sum(w * np.abs(t["observed_minus_predicted"]))),
               "mce": float(np.max(np.abs(t["observed_minus_predicted"]))),
               "mean_predicted": float(s.mean()), "observed_rate": float(ybar),
               "calibration_in_the_large": float(s.mean() - ybar),
               "calibration_intercept_offset_logit": citl, "calibration_slope": slope,
               "calibration_slope_model_intercept": slope_i,
               "spiegelhalter_z": spz, "spiegelhalter_p_value": float(2 * stats.norm.sf(abs(spz))),
               "hosmer_lemeshow_chi2": hl, "hosmer_lemeshow_df": hl_df,
               "hosmer_lemeshow_p_value": float(stats.chi2.sf(hl, hl_df))}
    return Outcome(summary, {"Reliability table": t}, rows_used=n,
                   notes=dropped_note(dropped) + [f"Scores from {src}; {strategy} bins.",
                                                  "Calibration intercept is fitted with logit(p) as offset "
                                                  "(slope fixed at 1)."])


@register("ml.gains_lift", "Cumulative gains, lift and KS by score band", "Discrimination", _CLS,
          params=(P("target"), _SCORE_OPT, _MODEL_OPT, _FEATS, P("bins", "integer", default=10)),
          description="""Rows sorted by descending score and cut into equal-count bands (deciles by default).
Per band: n, events, event rate, cumulative share of population, cumulative share of events captured
(gain), lift = band event rate / overall rate, cumulative lift, and the KS distance between cumulative
event and non-event distributions. Summary: top-band lift, capture rates in the top 10% / 20% of scores,
and the maximum KS. Ties in the score are kept in the same band.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 6",
                      "Provost & Fawcett (2013), Data Science for Business, ch. 8"))
def gains_lift(ctx: RunContext, target, score=None, model=None, features=None, bins=10) -> Outcome:
    sub, s, dropped, src, _ = _rows(ctx, [target], score, model, features)
    y = binary(sub, target).to_numpy()
    ok = np.isfinite(y)
    dropped += int((~ok).sum())
    y, s = y[ok].astype(int), s[ok]
    require_two_classes(y, target)
    n, pos = len(y), int(y.sum())
    rank = stats.rankdata(-s, method="max")        # ties share the worst position
    band = np.minimum(np.ceil(rank * bins / n).astype(int), bins)
    d = pd.DataFrame({"band": band, "y": y, "s": s}).groupby("band").agg(
        n=("y", "size"), events=("y", "sum"), min_score=("s", "min"), max_score=("s", "max")).reset_index()
    d["event_rate"] = d["events"] / d["n"]
    d["cum_population_share"] = d["n"].cumsum() / n
    d["cum_event_capture"] = d["events"].cumsum() / pos
    d["cum_nonevent_share"] = (d["n"] - d["events"]).cumsum() / (n - pos)
    d["lift"] = d["event_rate"] / (pos / n)
    d["cum_lift"] = d["cum_event_capture"] / d["cum_population_share"]
    d["ks"] = d["cum_event_capture"] - d["cum_nonevent_share"]
    order = np.argsort(-s, kind="mergesort")
    cap = np.cumsum(y[order]) / pos
    summary = {"n": n, "events": pos, "bands": len(d), "top_band_lift": float(d["lift"].iloc[0]),
               "capture_top_10pct": float(cap[max(int(np.ceil(0.1 * n)) - 1, 0)]),
               "capture_top_20pct": float(cap[max(int(np.ceil(0.2 * n)) - 1, 0)]),
               "ks_max_banded": float(d["ks"].max()), "ks_exact": _ks(y, s)}
    return Outcome(summary, {"Gains and lift": d}, rows_used=n,
                   notes=dropped_note(dropped) + [f"Scores from {src}. Band 1 = highest scores."])


# ── regression performance ────────────────────────────────────────────────

@register("ml.regression_metrics", "Regression performance (RMSE, MAE, MAPE, R², ...)", "Performance", _REG,
          params=(P("actual"), P("predicted", required=False, help="Prediction column (instead of `model`)"),
                  _MODEL_OPT, _FEATS,
                  P("n_features", "integer", required=False,
                    help="Number of predictors for adjusted R² (default: model feature count, if a model is given)")),
          description="""Error metrics with residual e = actual − predicted: RMSE, MSE, MAE, median absolute
error, max absolute error, MAPE (rows with actual = 0 excluded and counted), symmetric MAPE
2|e|/(|a|+|p|) (0 when both are 0), R² = 1 − SSE/SST, adjusted R² = 1 − (1−R²)(n−1)/(n−p−1),
explained variance 1 − Var(e)/Var(a), bias = mean(predicted − actual), and Pearson / Spearman
correlation of predicted with actual.""",
          references=("Hyndman & Koehler (2006), Another look at measures of forecast accuracy, IJF 22",
                      "Theil (1961), Economic Forecasts and Policy"))
def regression_metrics(ctx: RunContext, actual, predicted=None, model=None, features=None,
                       n_features=None) -> Outcome:
    sub, p, dropped, src, feats = _rows(ctx, [actual], predicted, model, features)
    a = num(sub, actual).to_numpy()
    ok = np.isfinite(a)
    dropped += int((~ok).sum())
    a, p = a[ok], p[ok]
    if len(a) < 3:
        raise NotApplicable("Fewer than 3 complete rows.")
    k = n_features or (len(feats) if feats else None)
    m = _reg_metrics(a, p, k)
    notes = dropped_note(dropped) + [f"Predictions from {src}."]
    zeros = int((a == 0).sum())
    if zeros:
        notes.append(f"MAPE excludes {zeros} rows with actual = 0.")
    if k is None:
        notes.append("Adjusted R² needs n_features.")
    tab = pd.DataFrame({"metric": list(m), "value": list(m.values())})
    return Outcome(m, {"Metrics": tab}, notes=notes, rows_used=len(a))


@register("ml.residual_diagnostics", "Residual diagnostics by decile of prediction", "Performance", _REG,
          params=(P("actual"), P("predicted", required=False), _MODEL_OPT, _FEATS, P("bins", "integer", default=10)),
          description="""Residuals e = actual − predicted, grouped into equal-count bins of the prediction:
per bin n, prediction range, mean predicted, mean actual, mean residual, residual SD, RMSE and MAE (shows
where the model is biased or noisy). Summary: Mincer–Zarnowitz regression actual = α + β·predicted with
the joint F-test H0: α = 0, β = 1 (unbiased, efficient forecast); Breusch–Pagan LM test of residual
variance against the prediction (H0: homoscedastic); Jarque–Bera normality test, skewness and excess
kurtosis of residuals; Spearman correlation of |e| with the prediction.""",
          references=("Mincer & Zarnowitz (1969), The evaluation of economic forecasts, NBER",
                      "Breusch & Pagan (1979), Econometrica 47",
                      "Jarque & Bera (1980), Economics Letters 6"))
def residual_diagnostics(ctx: RunContext, actual, predicted=None, model=None, features=None, bins=10) -> Outcome:
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    sub, p, dropped, src, _ = _rows(ctx, [actual], predicted, model, features)
    a = num(sub, actual).to_numpy()
    ok = np.isfinite(a)
    dropped += int((~ok).sum())
    a, p = a[ok], p[ok]
    if len(a) < 10 or np.std(p) == 0:
        raise NotApplicable("Need at least 10 rows and a non-constant prediction.")
    e = a - p
    b = pd.qcut(stats.rankdata(p, method="ordinal"), bins, labels=False)
    d = pd.DataFrame({"bin": b + 1, "p": p, "a": a, "e": e}).groupby("bin")
    t = d.agg(n=("e", "size"), min_predicted=("p", "min"), max_predicted=("p", "max"), mean_predicted=("p", "mean"),
              mean_actual=("a", "mean"), mean_residual=("e", "mean"), residual_sd=("e", "std")).reset_index()
    t["rmse"] = d["e"].apply(lambda x: float(np.sqrt(np.mean(x ** 2)))).to_numpy()
    t["mae"] = d["e"].apply(lambda x: float(np.mean(np.abs(x)))).to_numpy()
    X = sm.add_constant(p)
    fit = sm.OLS(a, X).fit()
    ft = fit.f_test((np.eye(2), np.array([0.0, 1.0])))
    lm, lm_p, _, _ = het_breuschpagan(e, X)
    jb = stats.jarque_bera(e)
    sp = stats.spearmanr(np.abs(e), p)
    summary = {"n": len(a), "mean_residual": float(e.mean()), "mz_intercept": float(fit.params[0]),
               "mz_slope": float(fit.params[1]), "mz_F": float(np.squeeze(ft.fvalue)),
               "mz_p_value": float(ft.pvalue), "breusch_pagan_lm": float(lm), "breusch_pagan_p_value": float(lm_p),
               "jarque_bera": float(jb.statistic), "jarque_bera_p_value": float(jb.pvalue),
               "residual_skewness": float(stats.skew(e)), "residual_excess_kurtosis": float(stats.kurtosis(e)),
               "spearman_abs_residual_vs_prediction": float(sp.statistic),
               "spearman_p_value": float(sp.pvalue)}
    return Outcome(summary, {"Residuals by prediction bin": t}, rows_used=len(a),
                   notes=dropped_note(dropped) + [f"Predictions from {src}. Tests assume independent rows; "
                                                  "OLS standard errors in the Mincer–Zarnowitz test are not "
                                                  "robust to heteroscedasticity."])


# ── overfitting ────────────────────────────────────────────────────────────

@register("ml.train_test_gap", "Overfitting: train vs test metric gap with bootstrap CI", "Overfitting", _SUP,
          params=(_TARGET_OPT, _ACTUAL_OPT, _SCORE_OPT,
                  P("predicted", required=False, help="Prediction column for regression (instead of `model`)"),
                  _MODEL_OPT, _FEATS, P("sample", help="Sample column (train / test)"),
                  P("reference_value", "string", required=False, help="Value marking the training sample"),
                  P("current_value", "string", required=False, help="Value marking the test sample"),
                  P("metrics", "list", required=False,
                    help="Metrics; default auc, gini, ks, log_loss, brier, accuracy, f1 (classification) "
                         "or rmse, mae, r2, bias (regression)"),
                  _THRESHOLD, P("n_boot", "integer", default=500, help="Bootstrap resamples for the gap CI")),
          description="""Each metric on the training and the test sample, the gap = train − test, the relative
gap = gap / |train|, and a percentile bootstrap 95% CI and standard error of the gap (rows are resampled
independently within each sample with a seeded generator). For error metrics (log_loss, brier, rmse, mae)
a NEGATIVE gap means the test sample is worse. A CI excluding 0 indicates a performance difference
beyond sampling noise.""",
          references=("Hastie, Tibshirani & Friedman (2009), The Elements of Statistical Learning, ch. 7",
                      "Efron & Tibshirani (1993), An Introduction to the Bootstrap",
                      "ECB Guide to internal models (2024), credit risk chapter — back-testing on "
                      "out-of-sample and out-of-time data"))
def train_test_gap(ctx: RunContext, sample, target=None, actual=None, score=None, predicted=None, model=None,
                   features=None, reference_value=None, current_value=None, metrics=None, threshold=0.5,
                   n_boot=500) -> Outcome:
    ycol, task = _task(target, actual)
    sc = score if task == "classification" else predicted
    if task == "classification" and predicted and not score:
        raise ValueError("For a binary target pass the score column as `score`.")
    sub, s, dropped, src, _ = _rows(ctx, [ycol, sample], sc, model, features)
    y = _yvals(sub, ycol, task)
    ok = np.isfinite(y)
    dropped += int((~ok).sum())
    sub, y, s = sub[ok], y[ok], s[ok]
    names = [str(m).strip().lower() for m in metrics] if metrics else (
        ["auc", "gini", "ks", "log_loss", "brier", "accuracy", "f1"] if task == "classification"
        else ["rmse", "mae", "r2", "bias"])
    fns = {nm: _metric_spec(nm, task)[2] for nm in names}
    ref, cur, rl, cl = split_samples(sub.assign(_i=np.arange(len(sub))), sample, reference_value, current_value)
    ir, ic = ref["_i"].to_numpy(), cur["_i"].to_numpy()
    if len(ir) < 5 or len(ic) < 5:
        raise NotApplicable("Each sample needs at least 5 rows.")
    if task == "classification":
        require_two_classes(y[ir], f"{ycol} (train)")
        require_two_classes(y[ic], f"{ycol} (test)")

    def _all(i):
        return np.array([fns[nm](y[i], s[i], threshold) for nm in names], dtype=float)
    tr, te = _all(ir), _all(ic)
    rng = ctx.rng()
    gaps = np.array([_all(ir[rng.integers(0, len(ir), len(ir))]) - _all(ic[rng.integers(0, len(ic), len(ic))])
                     for _ in range(int(n_boot))]) if n_boot > 0 else np.full((1, len(names)), np.nan)
    tab = pd.DataFrame({"metric": names, "train": tr, "test": te, "gap_train_minus_test": tr - te,
                        "relative_gap": (tr - te) / np.where(np.abs(tr) > 0, np.abs(tr), np.nan),
                        "boot_se": np.nanstd(gaps, axis=0, ddof=1) if n_boot > 1 else np.nan,
                        "gap_ci_low_95": np.nanpercentile(gaps, 2.5, axis=0),
                        "gap_ci_high_95": np.nanpercentile(gaps, 97.5, axis=0)})
    summary = {"train_sample": rl, "test_sample": cl, "n_train": len(ir), "n_test": len(ic)}
    for _, r in tab.iterrows():
        summary |= {f"{r['metric']}_train": r["train"], f"{r['metric']}_test": r["test"],
                    f"{r['metric']}_gap": r["gap_train_minus_test"]}
    notes = dropped_note(dropped) + [f"Predictions from {src}. Bootstrap: {n_boot} resamples, seed {ctx.seed}."]
    if task == "classification" and not _is_prob(s):
        notes.append("Scores are not in [0, 1]: log loss and Brier are NaN.")
    return Outcome(summary, {"Train vs test": tab}, notes=notes, rows_used=len(ir) + len(ic))


@register("ml.cross_validation", "k-fold cross-validation of the real model (refit)", "Overfitting", _SUP,
          params=(_MODEL, _TARGET_OPT, _ACTUAL_OPT, _FEATS, P("k", "integer", default=5),
                  P("metrics", "list", required=False, help="Metrics; default auc, log_loss, brier (classification) "
                                                           "or rmse, mae, r2 (regression)"),
                  _THRESHOLD),
          description="""Refits a fresh clone of the model under validation (sklearn.base.clone: same
hyper-parameters, unfitted) on k−1 folds and scores the held-out fold, for k shuffled folds (stratified for
a binary target; folds and any unset random_state seeded from the run). Per fold: train and test metrics.
Summary per metric: mean and SD of the test-fold metric, coefficient of variation SD/|mean|, min, max and
the mean train − test gap. High dispersion means performance depends on the particular sample; a large
gap means the training procedure overfits. This assesses the model-building procedure on these data,
not the delivered fitted model. NotApplicable for models that cannot be cloned and refitted.""",
          references=("Stone (1974), Cross-validatory choice and assessment of statistical predictions, JRSS B 36",
                      "Kohavi (1995), A study of cross-validation and bootstrap for accuracy estimation, IJCAI",
                      "Bengio & Grandvalet (2004), No unbiased estimator of the variance of K-fold "
                      "cross-validation, JMLR 5"))
def cross_validation(ctx: RunContext, model, target=None, actual=None, features=None, k=5, metrics=None,
                     threshold=0.5) -> Outcome:
    est, seeded = _refittable(ctx, model)
    m, feats, sub, X, y, task, dropped = _model_xy(ctx, model, features, target, actual)
    names = [str(x).strip().lower() for x in metrics] if metrics else (
        ["auc", "log_loss", "brier"] if task == "classification" else ["rmse", "mae", "r2"])
    fns = {nm: _metric_spec(nm, task)[2] for nm in names}
    from sklearn.base import clone
    rows = []
    for i, (tr, te) in enumerate(_folds(y, int(k), task, ctx.seed), start=1):
        e = clone(est).fit(X.iloc[tr], y[tr])
        str_, ste = predict_scores(e, X.iloc[tr]), predict_scores(e, X.iloc[te])
        for nm in names:
            rows.append({"fold": i, "metric": nm, "n_train": len(tr), "n_test": len(te),
                         "train": fns[nm](y[tr], str_, threshold), "test": fns[nm](y[te], ste, threshold)})
    folds = pd.DataFrame(rows)
    g = folds.groupby("metric", sort=False)
    agg = pd.DataFrame({"mean_test": g["test"].mean(), "sd_test": g["test"].std(ddof=1),
                        "min_test": g["test"].min(), "max_test": g["test"].max(),
                        "mean_train": g["train"].mean()}).reset_index()
    agg["cv_coefficient"] = agg["sd_test"] / agg["mean_test"].abs()
    agg["mean_gap_train_minus_test"] = agg["mean_train"] - agg["mean_test"]
    summary = {"k": int(k), "n": len(y)}
    for _, r in agg.iterrows():
        summary |= {f"{r['metric']}_cv_mean": r["mean_test"], f"{r['metric']}_cv_sd": r["sd_test"]}
    notes = dropped_note(dropped) + [
        "Refits a clone of the model on these data: this validates the training procedure / hyper-parameters, "
        "not the delivered fitted model. Fold SDs understate the true uncertainty (folds overlap; "
        "Bengio & Grandvalet 2004)."]
    if seeded:
        notes.append(f"Unset random_state parameter(s) {seeded} set to the run seed {ctx.seed}.")
    return Outcome(summary, {"Cross-validation summary": agg, "Per fold": folds}, notes=notes, rows_used=len(y))


@register("ml.learning_curve", "Learning curve of the real model (refit on growing training sizes)",
          "Overfitting", _SUP,
          params=(_MODEL, _TARGET_OPT, _ACTUAL_OPT, _FEATS, _METRIC, P("k", "integer", default=5),
                  P("train_sizes", "list", required=False,
                    help="Fractions of each training fold; default 0.1, 0.25, 0.5, 0.75, 1.0"),
                  _THRESHOLD),
          description="""For each of k shuffled folds (stratified for a binary target) and each training-size
fraction, a fresh clone of the model is fitted on a seeded random subset of the training folds and scored
on that subset (train) and on the held-out fold (test). Reports mean and SD of train and test metric by
training size, and the gap. A test curve still rising at full size suggests more data would help; a
persistent large gap suggests over-fitting (high variance); low, converged curves suggest under-fitting.""",
          references=("Hastie, Tibshirani & Friedman (2009), The Elements of Statistical Learning, ch. 7",
                      "Perlich, Provost & Simonoff (2003), Tree induction vs. logistic regression: a "
                      "learning-curve analysis, JMLR 4"))
def learning_curve(ctx: RunContext, model, target=None, actual=None, features=None, metric=None, k=5,
                   train_sizes=None, threshold=0.5) -> Outcome:
    est, seeded = _refittable(ctx, model)
    m, feats, sub, X, y, task, dropped = _model_xy(ctx, model, features, target, actual)
    name, _, fn = _metric_spec(metric, task)
    sizes = sorted({float(v) for v in (train_sizes or [0.1, 0.25, 0.5, 0.75, 1.0])})
    if not all(0 < v <= 1 for v in sizes):
        raise ValueError("train_sizes must be fractions in (0, 1].")
    from sklearn.base import clone
    rows, skipped = [], 0
    for i, (tr, te) in enumerate(_folds(y, int(k), task, ctx.seed), start=1):
        perm = tr[ctx.rng(i).permutation(len(tr))]
        for f in sizes:
            sel = perm[:max(int(np.floor(f * len(tr))), 2)]
            if task == "classification" and len(np.unique(y[sel])) < 2:
                skipped += 1
                continue
            e = clone(est).fit(X.iloc[sel], y[sel])
            rows.append({"fraction": f, "fold": i, "n_train": len(sel),
                         "train": fn(y[sel], predict_scores(e, X.iloc[sel]), threshold),
                         "test": fn(y[te], predict_scores(e, X.iloc[te]), threshold)})
    if not rows:
        raise NotApplicable("No training subset contained both classes.")
    per = pd.DataFrame(rows)
    g = per.groupby("fraction")
    tab = pd.DataFrame({"n_train_mean": g["n_train"].mean(), "train_mean": g["train"].mean(),
                        "train_sd": g["train"].std(ddof=1), "test_mean": g["test"].mean(),
                        "test_sd": g["test"].std(ddof=1)}).reset_index()
    tab["gap_train_minus_test"] = tab["train_mean"] - tab["test_mean"]
    import plotly.graph_objects as go
    fig = go.Figure([go.Scatter(x=tab["n_train_mean"], y=tab["train_mean"], name="train", mode="lines+markers"),
                     go.Scatter(x=tab["n_train_mean"], y=tab["test_mean"], name="test (held-out fold)",
                                mode="lines+markers")])
    fig.update_layout(title=f"Learning curve ({name}) — model '{model}' refit", xaxis_title="training rows",
                      yaxis_title=name)
    last = tab.iloc[-1]
    notes = dropped_note(dropped) + ["Refits clones of the model: assesses the training procedure on these data."]
    if seeded:
        notes.append(f"Unset random_state parameter(s) {seeded} set to the run seed {ctx.seed}.")
    if skipped:
        notes.append(f"{skipped} fold/size combinations skipped (single class in the training subset).")
    return Outcome({"metric": name, "k": int(k), "n": len(y), "test_at_full_size": last["test_mean"],
                    "train_at_full_size": last["train_mean"], "gap_at_full_size": last["gap_train_minus_test"],
                    "test_at_smallest_size": tab.iloc[0]["test_mean"]},
                   {"Learning curve": tab, "Per fold": per}, [fig], notes=notes, rows_used=len(y))


# ── explainability (real model) ────────────────────────────────────────────

@register("ml.permutation_importance", "Permutation feature importance of the real model", "Explainability", _SUP,
          params=(_MODEL, _TARGET_OPT, _ACTUAL_OPT, _FEATS, _METRIC, P("n_repeats", "integer", default=10),
                  _THRESHOLD),
          description="""Model-agnostic importance: each feature column is randomly permuted (n_repeats times,
seeded) in the rows provided, the model under validation is re-scored, and importance = deterioration of
the metric (baseline − permuted for higher-is-better metrics, permuted − baseline for error metrics).
Reports mean, SD, min and max over repeats and the rank. Computed with sklearn.inspection.
permutation_importance. Importance is measured on these rows (use held-out data); correlated features
share importance and permutation creates unrealistic feature combinations.""",
          references=("Breiman (2001), Random forests, Machine Learning 45",
                      "Fisher, Rudin & Dominici (2019), All models are wrong, but many are useful, JMLR 20",
                      "Molnar (2022), Interpretable Machine Learning, 2nd ed., ch. 8.5",
                      "Strobl et al. (2008), Conditional variable importance for random forests, BMC Bioinformatics 9"))
def permutation_importance(ctx: RunContext, model, target=None, actual=None, features=None, metric=None,
                           n_repeats=10, threshold=0.5) -> Outcome:
    m, feats, sub, X, y, task, dropped = _model_xy(ctx, model, features, target, actual)
    name, hib, fn = _metric_spec(metric, task, for_importance=True)
    base = fn(y, predict_scores(m, X), threshold)
    imp = _perm_importance(m, X, y, fn, hib, threshold, n_repeats, ctx.seed)
    tab = pd.DataFrame({"feature": feats, "importance_mean": imp.mean(axis=1),
                        "importance_sd": imp.std(axis=1, ddof=1) if imp.shape[1] > 1 else np.nan,
                        "importance_min": imp.min(axis=1), "importance_max": imp.max(axis=1)})
    tab = tab.sort_values(["importance_mean", "feature"], ascending=[False, True]).reset_index(drop=True)
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    return Outcome({"metric": name, "baseline_metric": base, "n": len(y), "n_repeats": int(n_repeats),
                    "top_feature": tab.loc[0, "feature"], "top_importance": tab.loc[0, "importance_mean"],
                    "features_with_non_positive_importance": int((tab["importance_mean"] <= 0).sum())},
                   {"Permutation importance": tab}, rows_used=len(y),
                   notes=dropped_note(dropped) + [_HELDOUT_NOTE, f"Permutations seeded with {ctx.seed}."])


def _shap_values(ctx: RunContext, m, X: pd.DataFrame, max_rows: int, background_rows: int):
    """(shap values rows x features, explained rows, base value, method, additivity max error or NaN)."""
    import shap
    Xe = _sample_rows(X, max_rows, ctx.rng(0))
    try:
        expl = shap.TreeExplainer(m.obj)
        vals = expl.shap_values(Xe)
        base = expl.expected_value
        if isinstance(vals, list):
            vals = np.asarray(vals[1] if len(vals) == 2 else np.mean(np.abs(np.stack(vals)), axis=0))
        vals = np.asarray(vals, dtype=float)
        if vals.ndim == 3:
            vals = vals[:, :, 1] if vals.shape[2] == 2 else np.mean(np.abs(vals), axis=2)
        base = np.atleast_1d(np.asarray(base, dtype=float))
        base = float(base[1] if len(base) == 2 else base[0]) if len(base) <= 2 else np.nan
        return vals, Xe, base, "shap.TreeExplainer (tree_path_dependent; model's raw output units)", np.nan
    except Exception:
        pass
    bg = _sample_rows(X, background_rows, ctx.rng(1))
    dtypes = X.dtypes.to_dict()
    feats = list(X.columns)

    def f(A):
        return np.asarray(predict_scores(m, pd.DataFrame(A, columns=feats).astype(dtypes)), dtype=float)
    p = len(feats)
    algo = "exact" if p <= 10 else "permutation"
    explainer = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=len(bg)), algorithm=algo,
                               seed=ctx.seed)
    kw = {"max_evals": max(500, 2 * p + 1)} if algo == "permutation" else {}
    ex = explainer(Xe, silent=True, **kw)
    vals = np.asarray(ex.values, dtype=float)
    base = float(np.mean(np.atleast_1d(ex.base_values)))
    add_err = float(np.max(np.abs(np.atleast_1d(ex.base_values) + vals.sum(axis=1) - f(Xe.to_numpy()))))
    return vals, Xe, base, f"shap.Explainer ({algo}, interventional; background {len(bg)} rows; output = " \
                           "model score)", add_err


@register("ml.shap_importance", "SHAP global importance and direction of the real model", "Explainability", _SUP,
          params=(_MODEL, _FEATS, P("max_rows", "integer", default=200, help="Rows explained (seeded sample)"),
                  P("background_rows", "integer", default=50,
                    help="Background rows for model-agnostic SHAP (seeded sample)")),
          description="""Shapley additive explanations of the model under validation. Tree models (sklearn
trees/forests/boosting, XGBoost, LightGBM, CatBoost) use shap.TreeExplainer (exact, path-dependent, in the
model's raw output units, e.g. log-odds for gradient boosting); other models use the model-agnostic
shap.Explainer (exact for <= 10 features, otherwise permutation) on the model score with an interventional
background drawn with the run's seeded generator. Per feature: mean |SHAP| (global importance), share of
total, mean SHAP, and direction = Spearman correlation between the feature value and its SHAP value
(+ means higher values push the score up). Model-agnostic runs also report the additivity error
max|base + ΣSHAP − f(x)|.""",
          references=("Lundberg & Lee (2017), A unified approach to interpreting model predictions, NeurIPS",
                      "Lundberg et al. (2020), From local explanations to global understanding with explainable "
                      "AI for trees, Nature Machine Intelligence 2",
                      "Shapley (1953), A value for n-person games",
                      "EBA (2021), EBA/REP/2021/12 Report on the use of machine learning in IRB models"))
def shap_importance(ctx: RunContext, model, features=None, max_rows=200, background_rows=50) -> Outcome:
    m = ctx.models[model]
    feats = model_features(m, ctx.df, features)
    sub, dropped = complete(ctx.df, feats)
    if len(sub) < 10:
        raise NotApplicable("Fewer than 10 complete rows.")
    vals, Xe, base, method, add_err = _shap_values(ctx, m, sub[feats], int(max_rows), int(background_rows))
    if vals.shape != (len(Xe), len(feats)):
        raise NotApplicable(f"Unexpected SHAP output shape {vals.shape} for {len(feats)} features.")
    mabs = np.abs(vals).mean(axis=0)
    direc = []
    for j, f_ in enumerate(feats):
        x = Xe[f_]
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 1 and np.std(vals[:, j]) > 0:
            direc.append(float(stats.spearmanr(x.to_numpy(dtype=float), vals[:, j]).statistic))
        else:
            direc.append(np.nan)
    tab = pd.DataFrame({"feature": feats, "mean_abs_shap": mabs,
                        "share_of_total": mabs / mabs.sum() if mabs.sum() > 0 else np.nan,
                        "mean_shap": vals.mean(axis=0), "direction_spearman": direc})
    tab = tab.sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    tab.insert(0, "rank", np.arange(1, len(tab) + 1))
    import plotly.graph_objects as go
    top = tab.head(20).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h"))
    fig.update_layout(title=f"Mean |SHAP| — model '{model}'", xaxis_title="mean |SHAP value|")
    summary = {"method": method, "rows_explained": len(Xe), "base_value": base,
               "top_feature": tab.loc[0, "feature"], "top_mean_abs_shap": tab.loc[0, "mean_abs_shap"]}
    if not np.isnan(add_err):
        summary["additivity_max_error"] = add_err
    return Outcome(summary, {"SHAP importance": tab}, [fig], rows_used=len(Xe),
                   notes=dropped_note(dropped) + [_HELDOUT_NOTE,
                                                  f"Rows explained: seeded sample of {len(Xe)} of {len(sub)}.",
                                                  "SHAP attributes the model output, not the truth; correlated "
                                                  "features can split or swap attributions."])


@register("ml.partial_dependence", "Partial dependence and ICE summary of the real model", "Explainability", _SUP,
          params=(_MODEL, _FEATS, P("explain", "columns", required=False,
                                    help="Features to profile; default all model features (max 20)"),
                  P("grid_points", "integer", default=20), P("max_rows", "integer", default=500),
                  P("grid_lower_quantile", "number", default=0.05), P("grid_upper_quantile", "number", default=0.95)),
          description="""For each profiled feature, a grid of values (quantiles of the feature between the
lower and upper grid quantile, every distinct value when there are few, the most frequent levels for
categorical features) is substituted into every sampled row and the model under validation is re-scored.
ICE curve = one row's score along the grid; partial dependence (PD) = mean ICE. Table: PD and ICE
percentiles (10/50/90) at each grid value. Summary per feature: PD range (max − min, effect size), Spearman
correlation of PD with the grid (shape: +1 increasing, −1 decreasing), and ICE heterogeneity = mean SD of
centred ICE curves (large values indicate interactions that the PD average hides).""",
          references=("Friedman (2001), Greedy function approximation: a gradient boosting machine, "
                      "Annals of Statistics 29",
                      "Goldstein et al. (2015), Peeking inside the black box: visualizing statistical learning "
                      "with plots of individual conditional expectation, JCGS 24",
                      "Molnar (2022), Interpretable Machine Learning, 2nd ed., ch. 8.1"))
def partial_dependence(ctx: RunContext, model, features=None, explain=None, grid_points=20, max_rows=500,
                       grid_lower_quantile=0.05, grid_upper_quantile=0.95) -> Outcome:
    _check_quantiles(grid_lower_quantile, grid_upper_quantile)
    m = ctx.models[model]
    feats = model_features(m, ctx.df, features)
    targets = list(explain) if explain else feats[:20]
    bad = [f for f in targets if f not in feats]
    if bad:
        raise ValueError(f"{bad} are not model features.")
    sub, dropped = complete(ctx.df, feats)
    if len(sub) < 10:
        raise NotApplicable("Fewer than 10 complete rows.")
    X = _sample_rows(sub[feats], int(max_rows), ctx.rng())
    rows, summ, curves = [], [], {}
    for f in targets:
        grid = _grid(X[f], int(grid_points), grid_lower_quantile, grid_upper_quantile)
        if len(grid) < 2:
            continue
        ice = _ice(m, X, f, grid)
        pdp = ice.mean(axis=0)
        q = np.percentile(ice, [10, 50, 90], axis=0)
        for j, v in enumerate(grid):
            rows.append({"feature": f, "grid_value": v, "pd": pdp[j], "ice_p10": q[0, j], "ice_p50": q[1, j],
                         "ice_p90": q[2, j]})
        numeric = pd.api.types.is_numeric_dtype(X[f])
        cent = ice - ice[:, [0]]
        summ.append({"feature": f, "grid_points": len(grid), "pd_min": pdp.min(), "pd_max": pdp.max(),
                     "pd_range": pdp.max() - pdp.min(),
                     "pd_spearman_vs_grid": float(stats.spearmanr(grid.astype(float), pdp).statistic)
                     if numeric and np.std(pdp) > 0 else np.nan,
                     "ice_heterogeneity": float(cent.std(axis=0).mean())})
        curves[f] = (grid, pdp)
    if not summ:
        raise NotApplicable("No profiled feature has two or more distinct values.")
    s = pd.DataFrame(summ).sort_values(["pd_range", "feature"], ascending=[False, True]).reset_index(drop=True)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    show = s["feature"].tolist()[:6]
    fig = make_subplots(rows=1, cols=len(show), subplot_titles=show)
    for i, f in enumerate(show, start=1):
        g, p = curves[f]
        fig.add_trace(go.Scatter(x=[str(v) for v in g] if g.dtype == object else g, y=p, mode="lines+markers",
                                 name=f, showlegend=False), row=1, col=i)
    fig.update_layout(title=f"Partial dependence — model '{model}' (top {len(show)} by PD range)")
    return Outcome({"features_profiled": len(s), "rows_used_for_ice": len(X),
                    "largest_effect_feature": s.loc[0, "feature"], "largest_pd_range": s.loc[0, "pd_range"]},
                   {"Feature effect summary": s, "PD and ICE by grid value": pd.DataFrame(rows)}, [fig],
                   rows_used=len(X),
                   notes=dropped_note(dropped) + [f"Seeded sample of {len(X)} rows; scores are the model output.",
                                                  "PD/ICE substitute values independently of other features, "
                                                  "which can create unrealistic rows when features are correlated."])


@register("ml.monotonicity", "Monotonicity of the model response vs expected sign", "Explainability", _SUP,
          params=(_MODEL, _FEATS, P("expected_signs", "dict",
                                    help='Expected direction per feature, e.g. {"dti": 1, "income": -1}'),
                  P("grid_points", "integer", default=20), P("max_rows", "integer", default=500),
                  P("tolerance", "number", default=0.0,
                    help="Score change in the wrong direction ignored as numerical noise"),
                  P("grid_lower_quantile", "number", default=0.01), P("grid_upper_quantile", "number", default=0.99)),
          description="""Checks that the model's score moves in the business-expected direction when one
feature changes and the others stay fixed. For each feature with an expected sign (+1: score should not
decrease as the feature increases; −1: should not increase), ICE curves of the model under validation are
computed on a seeded sample of rows over a quantile grid. Reports the share of ICE curves with at least one
step in the wrong direction (beyond the tolerance), the share of all grid steps that violate, the mean and
maximum violation size, the number of violating steps of the partial-dependence curve, and the
sign-adjusted Spearman correlation of PD with the grid (1 = perfectly monotone in the expected way).""",
          references=("EBA (2021), EBA/REP/2021/12 Report on the use of machine learning in IRB models",
                      "Goldstein et al. (2015), Peeking inside the black box: visualizing statistical learning "
                      "with plots of individual conditional expectation, JCGS 24",
                      "ECB Guide to internal models (2024), credit risk chapter — plausibility of risk drivers"))
def monotonicity(ctx: RunContext, model, expected_signs, features=None, grid_points=20, max_rows=500,
                 tolerance=0.0, grid_lower_quantile=0.01, grid_upper_quantile=0.99) -> Outcome:
    _check_quantiles(grid_lower_quantile, grid_upper_quantile)
    m = ctx.models[model]
    feats = model_features(m, ctx.df, features)
    signs = {}
    for f, v in expected_signs.items():
        if f not in feats:
            raise ValueError(f"'{f}' is not a model feature.")
        if float(v) not in (1.0, -1.0):
            raise ValueError(f"Expected sign for '{f}' must be +1 or -1, got {v}.")
        signs[f] = int(float(v))
    sub, dropped = complete(ctx.df, feats)
    if len(sub) < 10:
        raise NotApplicable("Fewer than 10 complete rows.")
    X = _sample_rows(sub[feats], int(max_rows), ctx.rng())
    rows = []
    for f in sorted(signs):
        if not pd.api.types.is_numeric_dtype(X[f]):
            raise ValueError(f"'{f}' is not numeric; monotonicity needs an ordered feature.")
        grid = _grid(X[f], int(grid_points), grid_lower_quantile, grid_upper_quantile)
        if len(grid) < 2:
            rows.append({"feature": f, "expected_sign": signs[f], "grid_points": len(grid)})
            continue
        ice = _ice(m, X, f, grid)
        d = np.diff(ice, axis=1) * signs[f]
        viol = d < -tolerance
        pdd = np.diff(ice.mean(axis=0)) * signs[f]
        pdp = ice.mean(axis=0)
        rows.append({"feature": f, "expected_sign": signs[f], "grid_points": len(grid),
                     "share_curves_violating": float(viol.any(axis=1).mean()),
                     "share_steps_violating": float(viol.mean()),
                     "mean_violation_size": float(-d[viol].mean()) if viol.any() else 0.0,
                     "max_violation_size": float(-d[viol].min()) if viol.any() else 0.0,
                     "pd_violating_steps": int((pdd < -tolerance).sum()),
                     "pd_signed_spearman": float(signs[f] * stats.spearmanr(grid, pdp).statistic)
                     if np.std(pdp) > 0 else np.nan,
                     "pd_change_low_to_high": float(pdp[-1] - pdp[0])})
    t = pd.DataFrame(rows)
    worst = t.sort_values(["share_curves_violating", "feature"], ascending=[False, True]).iloc[0]
    return Outcome({"features_checked": len(t), "rows_used_for_ice": len(X),
                    "max_share_curves_violating": worst.get("share_curves_violating", np.nan),
                    "feature_with_max_violation_share": worst["feature"]},
                   {"Monotonicity by feature": t}, rows_used=len(X),
                   notes=dropped_note(dropped) + [f"Seeded sample of {len(X)} rows; tolerance {tolerance}.",
                                                  "ICE substitutes values with other features fixed; violations "
                                                  "can arise in feature regions with no real data."])


@register("ml.explanation_stability", "Stability of feature importance across bootstrap resamples",
          "Explainability", _SUP,
          params=(_MODEL, _TARGET_OPT, _ACTUAL_OPT, _FEATS, _METRIC, P("n_boot", "integer", default=20),
                  P("n_repeats", "integer", default=3, help="Permutation repeats inside each resample"),
                  P("top_k", "integer", default=5), _THRESHOLD),
          description="""Permutation importance of the model under validation is computed on the full data and
on n_boot bootstrap resamples of the rows (seeded). For each resample, the Spearman and Kendall rank
correlation between its importance vector and the full-sample vector is computed. Summary: mean, minimum
and 5th percentile of those correlations. Per feature: full-sample rank, mean / SD / min / max rank across
resamples and the share of resamples in which the feature is in the top k. Unstable rankings mean
explanations (and the stories built on them) depend on the particular sample.""",
          references=("Fisher, Rudin & Dominici (2019), All models are wrong, but many are useful, JMLR 20",
                      "Alvarez-Melis & Jaakkola (2018), On the robustness of interpretability methods, ICML WHI",
                      "Molnar et al. (2022), General pitfalls of model-agnostic interpretation methods, "
                      "xxAI workshop, LNAI 13200"))
def explanation_stability(ctx: RunContext, model, target=None, actual=None, features=None, metric=None,
                          n_boot=20, n_repeats=3, top_k=5, threshold=0.5) -> Outcome:
    m, feats, sub, X, y, task, dropped = _model_xy(ctx, model, features, target, actual)
    if len(feats) < 2:
        raise NotApplicable("Rank stability needs at least two features.")
    name, hib, fn = _metric_spec(metric, task, for_importance=True)
    full = _perm_importance(m, X, y, fn, hib, threshold, n_repeats, ctx.seed).mean(axis=1)
    n = len(y)
    imps, sp, kt = [], [], []
    for b in range(int(n_boot)):
        i = ctx.rng(b + 1).integers(0, n, n)
        if task == "classification" and len(np.unique(y[i])) < 2:
            continue
        v = _perm_importance(m, X.iloc[i], y[i], fn, hib, threshold, n_repeats, ctx.seed + b + 1).mean(axis=1)
        imps.append(v)
        sp.append(stats.spearmanr(v, full).statistic)
        kt.append(stats.kendalltau(v, full).statistic)
    if not imps:
        raise NotApplicable("No usable bootstrap resample.")
    I = np.asarray(imps)
    ranks = np.apply_along_axis(lambda r: stats.rankdata(-r, method="min"), 1, I)
    full_rank = stats.rankdata(-full, method="min")
    tab = pd.DataFrame({"feature": feats, "full_sample_importance": full, "full_sample_rank": full_rank,
                        "mean_rank": ranks.mean(axis=0), "sd_rank": ranks.std(axis=0, ddof=1) if len(I) > 1 else np.nan,
                        "min_rank": ranks.min(axis=0), "max_rank": ranks.max(axis=0),
                        f"share_in_top_{int(top_k)}": (ranks <= top_k).mean(axis=0)})
    tab = tab.sort_values(["full_sample_rank", "feature"]).reset_index(drop=True)
    sp, kt = np.asarray(sp, float), np.asarray(kt, float)
    per = pd.DataFrame({"resample": np.arange(1, len(sp) + 1), "spearman_vs_full": sp, "kendall_vs_full": kt})
    return Outcome({"metric": name, "resamples": len(I), "mean_spearman": float(np.nanmean(sp)),
                    "min_spearman": float(np.nanmin(sp)), "p5_spearman": float(np.nanpercentile(sp, 5)),
                    "mean_kendall": float(np.nanmean(kt)),
                    "top_feature_full": tab.loc[0, "feature"],
                    "top_feature_share_rank_1": float((ranks[:, feats.index(tab.loc[0, "feature"])] == 1).mean())},
                   {"Rank stability by feature": tab, "Per resample": per}, rows_used=n,
                   notes=dropped_note(dropped) + [_HELDOUT_NOTE, f"Seeded with {ctx.seed}."])


# ── robustness and sensitivity (real model) ────────────────────────────────

def _perturb_cols(X, feats, vary):
    cols = list(vary) if vary else _numeric_feats(X, feats, 3)
    bad = [c for c in cols if c not in feats or not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(f"{bad} are not numeric model features.")
    if not cols:
        raise NotApplicable("No continuous numeric model features to perturb.")
    return cols


def _delta_stats(base, new, threshold, classifier) -> dict:
    d = new - base
    out = {"mean_score_change": float(d.mean()), "mean_abs_score_change": float(np.abs(d).mean()),
           "p95_abs_score_change": float(np.percentile(np.abs(d), 95)), "max_abs_score_change": float(np.abs(d).max())}
    if classifier:
        out["share_label_flips"] = float(((base >= threshold) != (new >= threshold)).mean())
    return out


@register("ml.noise_robustness", "Robustness to Gaussian input noise (real model)", "Robustness", _SUP,
          params=(_MODEL, _TARGET_OPT, _ACTUAL_OPT, _FEATS,
                  P("perturb", "columns", required=False,
                    help="Features to perturb; default numeric model features with >= 3 distinct values"),
                  P("noise_levels", "list", required=False,
                    help="Noise SD as share of feature SD; default 0.01, 0.05, 0.1, 0.25"),
                  P("n_repeats", "integer", default=5), _THRESHOLD, _METRIC),
          description="""Adds independent Gaussian noise N(0, (level·SD_j)²) to each perturbed feature j (SD from
the rows provided), re-scores the model under validation and compares with the unperturbed score, for
several noise levels and seeded repeats. Per level: mean, mean absolute, 95th-percentile and maximum score
change; share of predicted labels that flip at the threshold (classifiers); and, when an outcome is given,
the metric under noise and its change from baseline. Measures how sensitive outputs and performance are
to measurement error in the inputs; evaluate on held-out data.""",
          references=("Goodfellow, Shlens & Szegedy (2015), Explaining and harnessing adversarial examples, ICLR",
                      "Hendrycks & Dietterich (2019), Benchmarking neural network robustness to common "
                      "corruptions and perturbations, ICLR",
                      "EBA (2021), EBA/REP/2021/12 Report on the use of machine learning in IRB models"))
def noise_robustness(ctx: RunContext, model, target=None, actual=None, features=None, perturb=None,
                     noise_levels=None, n_repeats=5, threshold=0.5, metric=None) -> Outcome:
    m, feats, sub, X, y, task, dropped = _model_xy(ctx, model, features, target, actual, need_y=False)
    cols = _perturb_cols(X, feats, perturb)
    levels = sorted({float(v) for v in (noise_levels or [0.01, 0.05, 0.1, 0.25])})
    clf = hasattr(m.obj, "predict_proba")
    fn = name = None
    if y is not None:
        name, _, fn = _metric_spec(metric, task)
    base = predict_scores(m, X)
    base_metric = fn(y, base, threshold) if fn else np.nan
    sd = X[cols].astype(float).std(ddof=1).to_numpy()
    rows = []
    for li, lev in enumerate(levels):
        for r in range(int(n_repeats)):
            rng = ctx.rng(1000 * (li + 1) + r)
            Xn = X.copy()
            noise = rng.normal(0.0, 1.0, (len(X), len(cols))) * (lev * sd)
            for j, c in enumerate(cols):
                Xn[c] = X[c].astype(float).to_numpy() + noise[:, j]
            s = predict_scores(m, Xn)
            row = {"noise_level": lev, "repeat": r + 1} | _delta_stats(base, s, threshold, clf)
            if fn:
                row["metric_under_noise"] = fn(y, s, threshold)
            rows.append(row)
    per = pd.DataFrame(rows)
    agg = per.drop(columns="repeat").groupby("noise_level").mean().reset_index()
    if fn:
        agg["metric_sd_over_repeats"] = per.groupby("noise_level")["metric_under_noise"].std(ddof=1).to_numpy()
        agg["metric_change"] = agg["metric_under_noise"] - base_metric
    top = agg.iloc[-1]
    summary = {"features_perturbed": len(cols), "levels": len(levels), "repeats": int(n_repeats), "n": len(X),
               "max_level": top["noise_level"], "mean_abs_score_change_at_max_level": top["mean_abs_score_change"]}
    if clf:
        summary["share_label_flips_at_max_level"] = top["share_label_flips"]
    if fn:
        summary |= {"metric": name, "baseline_metric": base_metric, "metric_change_at_max_level": top["metric_change"]}
    return Outcome(summary, {"Robustness by noise level": agg, "Per repeat": per}, rows_used=len(X),
                   notes=dropped_note(dropped) + [_HELDOUT_NOTE, f"Perturbed: {', '.join(cols)}. Noise seeded "
                                                  f"from {ctx.seed}; integer-valued features receive continuous noise."])


@register("ml.sensitivity", "One-at-a-time sensitivity of the model score", "Robustness", _SUP,
          params=(_MODEL, _FEATS, P("vary", "columns", required=False,
                                    help="Features to shift; default numeric model features with >= 3 values"),
                  P("shift_std", "number", default=1.0, help="Shift size in feature standard deviations"),
                  P("shift_pct", "number", required=False,
                    help="If given, relative shift instead (0.1 = ±10% of each value)"),
                  P("threshold", "number", required=False, help="Classifier cut-off for label-flip shares")),
          description="""Each feature in turn is shifted up and down for every row — by ±shift_std standard
deviations (absolute shift) or by ±shift_pct of its value (relative) — with all other features unchanged,
and the model under validation is re-scored. Per feature and direction: mean score change, mean absolute,
95th-percentile and maximum absolute change, mean score after the shift and (classifiers with threshold)
share of labels flipping. Ranks features by their local influence on the score; also reveals
counter-intuitive signs.""",
          references=("Saltelli et al. (2008), Global Sensitivity Analysis: The Primer, Wiley",
                      "Saltelli & Annoni (2010), How to avoid a perfunctory sensitivity analysis, "
                      "Environmental Modelling & Software 25"))
def sensitivity(ctx: RunContext, model, features=None, vary=None, shift_std=1.0, shift_pct=None,
                threshold=None) -> Outcome:
    m, feats, sub, X, _, _, dropped = _model_xy(ctx, model, features, need_y=False)
    cols = _perturb_cols(X, feats, vary)
    base = predict_scores(m, X)
    clf = threshold is not None
    rows = []
    for f in cols:
        x = X[f].astype(float).to_numpy()
        for direction, sgn in (("up", 1.0), ("down", -1.0)):
            Xs = X.copy()
            Xs[f] = x * (1 + sgn * shift_pct) if shift_pct is not None else x + sgn * shift_std * np.std(x, ddof=1)
            s = predict_scores(m, Xs)
            rows.append({"feature": f, "direction": direction,
                         "shift": f"{sgn * shift_pct:+.4g} relative" if shift_pct is not None
                         else f"{sgn * shift_std * np.std(x, ddof=1):+.6g}",
                         "mean_score_after": float(s.mean())} | _delta_stats(base, s, threshold or 0.5, clf))
    t = pd.DataFrame(rows)
    rank = t.groupby("feature")["mean_abs_score_change"].max().sort_values(ascending=False)
    t["feature_rank"] = t["feature"].map({f: i + 1 for i, f in enumerate(rank.index)})
    t = t.sort_values(["feature_rank", "direction"]).reset_index(drop=True)
    return Outcome({"features": len(cols), "n": len(X), "base_mean_score": float(base.mean()),
                    "shift": f"±{shift_pct} relative" if shift_pct is not None else f"±{shift_std} SD",
                    "most_sensitive_feature": rank.index[0], "its_mean_abs_score_change": float(rank.iloc[0])},
                   {"Sensitivity by feature": t}, rows_used=len(X),
                   notes=dropped_note(dropped) + [_HELDOUT_NOTE, "Shifts may move rows outside the observed "
                                                  "range of a feature (extrapolation)."])


@register("ml.scenario_stress", "Scenario stress test of the model (feature shocks)", "Robustness", _SUP,
          params=(_MODEL, _FEATS,
                  P("shocks", "dict", help='Feature -> shock, e.g. {"income": 0.9, "dti": 1.2}; a value can also be '
                                            '{"add": 0.5} or {"multiply": 1.1}'),
                  P("shock_type", "string", default="multiplicative", choices=("multiplicative", "additive"),
                    help="How plain-number shocks are applied: x·v or x + v"),
                  P("threshold", "number", required=False, help="Cut-off for the share above threshold"),
                  P("weight", required=False, help="Exposure / weight column for weighted portfolio averages")),
          description="""Applies a scenario of simultaneous shocks to model inputs (multiplicative x·v or
additive x + v per feature) to every row, re-scores the model under validation and reports the
portfolio-level outcome: mean (and exposure-weighted mean) score before and after, absolute and relative
change, the share of rows above the threshold before and after, the share of rows whose label changes,
and score quantiles before and after. Shocks are applied mechanically: they do not propagate to
correlated inputs that are not shocked.""",
          references=("EBA (2018), EBA/GL/2018/04 Guidelines on institutions' stress testing",
                      "Basel Committee on Banking Supervision (2018), Stress testing principles"))
def scenario_stress(ctx: RunContext, model, shocks, features=None, shock_type="multiplicative", threshold=None,
                    weight=None) -> Outcome:
    m = ctx.models[model]
    feats = model_features(m, ctx.df, features)
    sub, dropped = complete(ctx.df, _uniq([*feats, weight]))
    if sub.empty:
        raise NotApplicable("No complete rows.")
    X = sub[feats]
    Xs = X.copy()
    applied = []
    for f in sorted(shocks):
        if f not in feats or not pd.api.types.is_numeric_dtype(X[f]):
            raise ValueError(f"'{f}' is not a numeric model feature.")
        v = shocks[f]
        if isinstance(v, dict):
            if len(v) != 1 or next(iter(v)) not in ("add", "multiply"):
                raise ValueError(f"Shock for '{f}' must be a number or {{'add': x}} / {{'multiply': x}}.")
            kind, val = next(iter(v.items()))
        else:
            kind, val = ("multiply" if shock_type == "multiplicative" else "add"), v
        val = float(val)
        x = X[f].astype(float)
        Xs[f] = x * val if kind == "multiply" else x + val
        applied.append({"feature": f, "shock": kind, "value": val, "mean_before": float(x.mean()),
                        "mean_after": float(Xs[f].mean())})
    b, s = predict_scores(m, X), predict_scores(m, Xs)
    w = num(sub, weight).to_numpy() if weight else np.ones(len(b))
    summary = {"n": len(b), "shocked_features": len(applied), "mean_score_base": float(b.mean()),
               "mean_score_stressed": float(s.mean()), "mean_score_change": float(s.mean() - b.mean()),
               "relative_change": float(s.mean() / b.mean() - 1) if b.mean() != 0 else np.nan}
    if weight:
        wb, ws = float(np.average(b, weights=w)), float(np.average(s, weights=w))
        summary |= {"weighted_mean_score_base": wb, "weighted_mean_score_stressed": ws,
                    "weighted_relative_change": ws / wb - 1 if wb else np.nan,
                    "sum_weight_x_score_base": float(np.sum(w * b)), "sum_weight_x_score_stressed": float(np.sum(w * s))}
    if threshold is not None:
        summary |= {"share_above_threshold_base": float((b >= threshold).mean()),
                    "share_above_threshold_stressed": float((s >= threshold).mean()),
                    "share_label_changed": float(((b >= threshold) != (s >= threshold)).mean())}
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    qt = pd.DataFrame({"quantile": qs, "score_base": np.quantile(b, qs), "score_stressed": np.quantile(s, qs)})
    return Outcome(summary, {"Shocks applied": pd.DataFrame(applied), "Score quantiles": qt}, rows_used=len(b),
                   notes=dropped_note(dropped) + ["Shocks are applied only to the listed features; correlated "
                                                  "inputs are not moved and the model is not re-estimated."])


@register("ml.missing_value_robustness", "Robustness to missing / imputed inputs", "Robustness", _SUP,
          params=(_MODEL, _TARGET_OPT, _ACTUAL_OPT, _FEATS,
                  P("vary", "columns", required=False, help="Features to blank out; default all model features"),
                  _THRESHOLD, _METRIC),
          description="""For each feature in turn, the model under validation is re-scored with that feature
(a) set to missing (NaN) for every row — models that cannot handle missing values are reported as such —
and (b) replaced by its median (numeric) or mode (categorical), i.e. simple imputation. Per feature and
treatment: mean, mean absolute and maximum score change and, when an outcome is given, the metric and its
change from baseline. Shows how much each input matters and how the model behaves when data are missing
in production.""",
          references=("Saar-Tsechansky & Provost (2007), Handling missing values when applying classification "
                      "models, JMLR 8",
                      "Little & Rubin (2019), Statistical Analysis with Missing Data, 3rd ed."))
def missing_value_robustness(ctx: RunContext, model, target=None, actual=None, features=None, vary=None,
                             threshold=0.5, metric=None) -> Outcome:
    m, feats, sub, X, y, task, dropped = _model_xy(ctx, model, features, target, actual, need_y=False)
    cols = list(vary) if vary else feats
    bad = [c for c in cols if c not in feats]
    if bad:
        raise ValueError(f"{bad} are not model features.")
    clf = hasattr(m.obj, "predict_proba")
    fn = name = None
    if y is not None:
        name, _, fn = _metric_spec(metric, task)
    base = predict_scores(m, X)
    bm = fn(y, base, threshold) if fn else np.nan
    rows, fails = [], 0
    for f in cols:
        numeric = pd.api.types.is_numeric_dtype(X[f])
        fill = X[f].median() if numeric else X[f].mode().sort_values().iloc[0]
        for treat, val in (("set to NaN", np.nan), ("median / mode", fill)):
            Xs = X.copy()
            Xs[f] = val
            row = {"feature": f, "treatment": treat, "fill_value": val}
            try:
                s = np.asarray(predict_scores(m, Xs), dtype=float)
                if not np.all(np.isfinite(s)):
                    raise ValueError("non-finite scores")
                row |= {"model_error": ""} | _delta_stats(base, s, threshold, clf)
                if fn:
                    row |= {"metric": fn(y, s, threshold), "metric_change": fn(y, s, threshold) - bm}
            except Exception as exc:
                fails += treat == "set to NaN"
                row["model_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            rows.append(row)
    t = pd.DataFrame(rows)
    imp = t[t["treatment"] == "median / mode"]
    summary = {"features": len(cols), "n": len(X), "features_where_model_fails_on_nan": int(fails)}
    if "mean_abs_score_change" in imp and imp["mean_abs_score_change"].notna().any():
        top = imp.loc[imp["mean_abs_score_change"].idxmax()]
        summary |= {"max_mean_abs_change_imputed": top["mean_abs_score_change"],
                    "most_affected_feature_imputed": top["feature"]}
    if fn:
        summary |= {"metric": name, "baseline_metric": bm}
    return Outcome(summary, {"Missing-value robustness": t}, rows_used=len(X),
                   notes=dropped_note(dropped) + [_HELDOUT_NOTE, "Medians / modes are taken from the rows provided."])


@register("ml.extrapolation", "Out-of-range extrapolation vs the training sample", "Robustness",
          _SUP + _UNS,
          params=(P("features", "columns", help="Model input columns"),
                  P("sample", required=False, help="Sample column; omit when the current data are another table"),
                  P("reference_value", "string", required=False, help="Value marking the training (reference) sample"),
                  P("current_value", "string", required=False, help="Value marking the sample being scored"),
                  P("other", "table", required=False,
                    help="Second loaded table with the rows being scored (active table = training reference)"),
                  P("quantile", "number", default=0.01,
                    help="Also report the share outside the reference [q, 1−q] quantile range")),
          description="""Share of current (scoring) rows whose inputs fall outside what the model saw in
training. Numeric features: share below the reference minimum, above the maximum, and outside the
reference [q, 1−q] quantile range. Categorical features: share of rows with a level never seen in the
reference. Row level: share of rows with at least one feature outside the reference range, and the share
whose Mahalanobis distance from the reference mean exceeds the reference sample's maximum and 99th
percentile distance (multivariate extrapolation, numeric features). Model outputs in these regions rest
on extrapolation.""",
          references=("Mahalanobis (1936), On the generalised distance in statistics",
                      "EBA (2017), EBA/GL/2017/16 Guidelines on PD estimation, LGD estimation and the treatment "
                      "of defaulted exposures — representativeness of data"))
def extrapolation(ctx: RunContext, features, sample=None, reference_value=None, current_value=None, other=None,
                  quantile=0.01) -> Outcome:
    if other is not None:
        ref, cur, rl, cl = ctx.df, ctx.tables[other], ctx.source_name or "reference", other
    elif sample:
        ref, cur, rl, cl = split_samples(ctx.df, sample, reference_value, current_value)
    else:
        raise ValueError("Give `sample` (a sample column) or `other` (a second table).")
    missing = [f for f in features if f not in cur.columns]
    if missing:
        raise ValueError(f"{missing} are not in the current sample table.")
    if len(ref) < 2 or len(cur) < 1:
        raise NotApplicable("Reference needs >= 2 rows and current >= 1 row.")
    rows, flags = [], np.zeros(len(cur), bool)
    qflags = np.zeros(len(cur), bool)
    numf = []
    for f in features:
        r, c = ref[f].dropna(), cur[f]
        if pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(c):
            numf.append(f)
            cv = pd.to_numeric(c, errors="coerce").to_numpy(dtype=float)
            lo, hi = float(r.min()), float(r.max())
            ql, qh = np.quantile(r.to_numpy(dtype=float), [quantile, 1 - quantile])
            below, above = cv < lo, cv > hi
            qout = (cv < ql) | (cv > qh)
            flags |= below | above
            qflags |= qout
            rows.append({"feature": f, "type": "numeric", "ref_min": lo, "ref_max": hi, "ref_q_low": ql,
                         "ref_q_high": qh, "share_below_min": float(below.mean()), "share_above_max": float(above.mean()),
                         "share_outside_range": float((below | above).mean()),
                         "share_outside_quantile_range": float(qout.mean()), "share_unseen_level": np.nan,
                         "current_missing": int(c.isna().sum())})
        else:
            seen = set(r.astype(str))
            un = (~c.astype(str).isin(seen) & c.notna()).to_numpy()
            flags |= un
            qflags |= un
            rows.append({"feature": f, "type": "categorical", "share_unseen_level": float(un.mean()),
                         "share_outside_range": float(un.mean()), "current_missing": int(c.isna().sum())})
    t = pd.DataFrame(rows).sort_values(["share_outside_range", "feature"], ascending=[False, True])
    summary = {"reference": rl, "current": cl, "n_reference": len(ref), "n_current": len(cur),
               "share_rows_any_feature_outside_range": float(flags.mean()),
               f"share_rows_any_feature_outside_q{quantile:g}_range": float(qflags.mean())}
    notes = ["Missing values are not counted as out of range."]
    if len(numf) >= 1:
        R = ref[numf].dropna().to_numpy(dtype=float)
        C = cur[numf].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
        if len(R) > len(numf) and len(C):
            mu, Si = R.mean(axis=0), np.linalg.pinv(np.atleast_2d(np.cov(R, rowvar=False)))
            md = lambda A: np.sqrt(np.einsum("ij,jk,ik->i", A - mu, Si, A - mu))
            dr, dc = md(R), md(C)
            summary |= {"share_rows_mahalanobis_above_ref_max": float((dc > dr.max()).mean()),
                        "share_rows_mahalanobis_above_ref_p99": float((dc > np.quantile(dr, 0.99)).mean())}
            notes.append("Mahalanobis distances use the reference mean and (pseudo-inverse) covariance; rows "
                         "with missing numeric features are excluded from that part.")
    return Outcome(summary, {"Extrapolation by feature": t}, rows_used=len(ref) + len(cur), notes=notes)


# ── data leakage ───────────────────────────────────────────────────────────

@register("ml.leakage_screen", "Target-leakage screen: single-feature predictive power", "Data leakage", _SUP,
          params=(P("features", "columns"), _TARGET_OPT, _ACTUAL_OPT),
          description="""Flags candidate leakage (features that encode the outcome, e.g. post-event
information) by measuring how well EACH FEATURE ALONE predicts the outcome. Binary target: direction-free
AUC = max(AUC, 1 − AUC) of the raw feature (numeric) or of its in-sample event-rate encoding (categorical,
optimistic), plus |Spearman| (numeric); continuous outcome: |Spearman| and |Pearson| (numeric) or the
correlation ratio η (categorical). For features with missing values, the AUC / |Spearman| of the
missing-value indicator is also reported, because missingness that depends on the outcome is a classic
leak. Sorted by single-feature strength; near-perfect predictors deserve a data-lineage check.""",
          references=("Kaufman et al. (2012), Leakage in data mining: formulation, detection, and avoidance, "
                      "ACM TKDD 6",
                      "Hand (2006), Classifier technology and the illusion of progress, Statistical Science 21"))
def leakage_screen(ctx: RunContext, features, target=None, actual=None) -> Outcome:
    ycol, task = _task(target, actual)
    df = ctx.df
    y_all = _yvals(df, ycol, task)
    okY = np.isfinite(y_all)
    if task == "classification":
        require_two_classes(y_all[okY], ycol)
    rows = []
    for f in features:
        if f == ycol:
            continue
        x = df[f]
        miss = x.isna().to_numpy()
        ok = okY & ~miss
        y = y_all[ok]
        row = {"feature": f, "n_non_missing": int(ok.sum()), "missing_share": float(miss[okY].mean())}
        numeric = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
        if ok.sum() >= 3 and x[ok].nunique() > 1 and len(np.unique(y)) > 1:
            if numeric:
                xv = x[ok].to_numpy(dtype=float)
                row["abs_spearman"] = abs(float(stats.spearmanr(xv, y).statistic))
                if task == "classification":
                    a = _auc(y.astype(int), xv)
                    row["single_feature_auc"] = max(a, 1 - a)
                else:
                    row["abs_pearson"] = abs(float(np.corrcoef(xv, y)[0, 1]))
            else:
                xs = x[ok].astype(str).to_numpy()
                means = pd.Series(y).groupby(xs).mean()
                enc = pd.Series(xs).map(means).to_numpy()
                if task == "classification":
                    a = _auc(y.astype(int), enc)
                    row["single_feature_auc"] = max(a, 1 - a)
                else:
                    sst = np.sum((y - y.mean()) ** 2)
                    row["eta_correlation_ratio"] = float(np.sqrt(np.sum((enc - y.mean()) ** 2) / sst)) if sst else np.nan
        if 0 < miss[okY].sum() < okY.sum():
            mi = miss[okY].astype(float)
            if task == "classification":
                a = _auc(y_all[okY].astype(int), mi)
                row["missing_indicator_auc"] = max(a, 1 - a)
            else:
                row["missing_indicator_abs_spearman"] = abs(float(stats.spearmanr(mi, y_all[okY]).statistic))
        rows.append(row)
    t = pd.DataFrame(rows)
    key = "single_feature_auc" if task == "classification" else "abs_spearman"
    if key not in t:
        t[key] = np.nan
    if task == "regression" and "eta_correlation_ratio" in t:
        t["strength"] = t[[key, "eta_correlation_ratio"]].max(axis=1)
    else:
        t["strength"] = t[key]
    t = t.sort_values(["strength", "feature"], ascending=[False, True], na_position="last").reset_index(drop=True)
    summary = {"features_screened": len(t), "n": int(okY.sum()), "strongest_feature": t.loc[0, "feature"],
               "strongest_single_feature_" + ("auc" if task == "classification" else "association"): t.loc[0, "strength"]}
    if task == "classification":
        summary["max_abs_spearman"] = float(t["abs_spearman"].max()) if "abs_spearman" in t else np.nan
    return Outcome(summary, {"Single-feature association with outcome": t.drop(columns="strength")}, rows_used=int(okY.sum()),
                   notes=["Each feature is evaluated on its own non-missing rows. Event-rate encodings of "
                          "categorical features are in-sample and therefore optimistic."])


@register("ml.train_test_duplicates", "Duplicate rows / IDs across train and test samples", "Data leakage",
          _SUP + _UNS,
          params=(P("sample"), P("reference_value", "string", required=False, help="Value marking the training sample"),
                  P("current_value", "string", required=False, help="Value marking the test sample"),
                  P("features", "columns", required=False, help="Columns that define a duplicate; default all except sample"),
                  P("id", required=False)),
          description="""Counts rows of the test sample whose values on the chosen columns exactly match a row
of the training sample (contamination makes test performance optimistic), duplicates within each sample,
and — with an id column — entities present in both samples (overlap is a leak for out-of-sample tests,
expected for out-of-time tests on the same portfolio). Matching uses exact value hashes; missing values
match each other.""",
          references=("Kaufman et al. (2012), Leakage in data mining: formulation, detection, and avoidance, "
                      "ACM TKDD 6",
                      "Allamanis (2019), The adverse effects of code duplication in machine learning models "
                      "of code, Onward!"))
def train_test_duplicates(ctx: RunContext, sample, reference_value=None, current_value=None, features=None,
                          id=None) -> Outcome:
    ref, cur, rl, cl = split_samples(ctx.df, sample, reference_value, current_value)
    cols = list(features) if features else [c for c in ctx.df.columns if c not in (sample, id)]
    hr = pd.util.hash_pandas_object(ref[cols], index=False).to_numpy()
    hc = pd.util.hash_pandas_object(cur[cols], index=False).to_numpy()
    in_ref = np.isin(hc, hr)
    summary = {"train_sample": rl, "test_sample": cl, "n_train": len(ref), "n_test": len(cur),
               "columns_compared": len(cols), "test_rows_duplicating_train": int(in_ref.sum()),
               "share_test_rows_in_train": float(in_ref.mean()) if len(cur) else np.nan,
               "duplicate_rows_within_train": int(len(hr) - len(np.unique(hr))),
               "duplicate_rows_within_test": int(len(hc) - len(np.unique(hc)))}
    tables = {}
    if id:
        ir, ic = set(ref[id].dropna()), set(cur[id].dropna())
        both = ir & ic
        summary |= {"ids_in_both": len(both), "share_test_ids_in_train": len(both) / len(ic) if ic else np.nan}
    if in_ref.any():
        tables["Test rows found in train (first 50)"] = cur.loc[in_ref, cols].head(50).reset_index(drop=True)
    return Outcome(summary, tables, rows_used=len(ref) + len(cur))


# ── multicollinearity ──────────────────────────────────────────────────────

@register("ml.vif", "Variance inflation factors", "Multicollinearity", _SUP,
          params=(P("features", "columns"),),
          description="""VIF_j = 1 / (1 − R²_j), where R²_j is from regressing feature j on all other features
with an intercept (statsmodels variance_inflation_factor). Also tolerance 1/VIF and R²_j. VIF measures how
much the variance of a linear coefficient is inflated by collinearity; for non-linear ML models it
indicates redundancy that destabilises importance measures and explanations. Perfect collinearity gives
an infinite VIF.""",
          references=("Belsley, Kuh & Welsch (1980), Regression Diagnostics, Wiley",
                      "Kutner, Nachtsheim & Neter (2004), Applied Linear Regression Models, 4th ed., ch. 10"))
def vif(ctx: RunContext, features) -> Outcome:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    sub, dropped = _numeric_matrix(ctx, features)
    if len(features) < 2 or len(sub) <= len(features) + 1:
        raise NotApplicable("Need at least 2 features and more rows than features.")
    X = np.column_stack([np.ones(len(sub)), sub[features].to_numpy(dtype=float)])
    v = np.array([variance_inflation_factor(X, j + 1) for j in range(len(features))], dtype=float)
    t = pd.DataFrame({"feature": features, "vif": v, "tolerance": 1 / v, "r2_on_other_features": 1 - 1 / v})
    t = t.sort_values(["vif", "feature"], ascending=[False, True]).reset_index(drop=True)
    return Outcome({"features": len(features), "n": len(sub), "max_vif": float(t["vif"].max()),
                    "max_vif_feature": t.loc[0, "feature"], "mean_vif": float(np.mean(v))},
                   {"VIF": t}, notes=dropped_note(dropped), rows_used=len(sub))


@register("ml.condition_index", "Condition number and Belsley variance-decomposition proportions",
          "Multicollinearity", _SUP,
          params=(P("features", "columns"), P("include_intercept", "boolean", default=True)),
          description="""Belsley–Kuh–Welsch collinearity diagnostics: the design matrix (with an intercept
column by default, NOT centred) is scaled to unit column length and decomposed by SVD. Condition index
η_k = μ_max / μ_k for each singular value; the condition number is the largest index. Variance-
decomposition proportions π_kj = (v_jk²/μ_k²) / Σ_k (v_jk²/μ_k²) show which variables share each
near-dependency (a high index with two or more large proportions). Also reports the condition number of
the correlation matrix, √(λ_max/λ_min).""",
          references=("Belsley, Kuh & Welsch (1980), Regression Diagnostics: Identifying Influential Data and "
                      "Sources of Collinearity, Wiley",
                      "Belsley (1991), Conditioning Diagnostics: Collinearity and Weak Data in Regression, Wiley"))
def condition_index(ctx: RunContext, features, include_intercept=True) -> Outcome:
    sub, dropped = _numeric_matrix(ctx, features)
    if len(sub) <= len(features):
        raise NotApplicable("Need more rows than features.")
    X = sub[features].to_numpy(dtype=float)
    names = list(features)
    if include_intercept:
        X = np.column_stack([np.ones(len(X)), X])
        names = ["(intercept)"] + names
    norms = np.linalg.norm(X, axis=0)
    if np.any(norms == 0):
        raise NotApplicable("A column is identically zero.")
    _, mu, Vt = np.linalg.svd(X / norms, full_matrices=False)
    with np.errstate(divide="ignore"):
        ci = mu.max() / mu
        phi = (Vt.T ** 2) / mu ** 2
    pi = phi / phi.sum(axis=1, keepdims=True)                   # rows: variables, cols: dimensions
    t = pd.DataFrame({"dimension": np.arange(1, len(mu) + 1), "singular_value": mu, "eigenvalue": mu ** 2,
                      "condition_index": ci})
    for j, nm in enumerate(names):
        t[f"prop_{nm}"] = pi[j]
    t = t.sort_values("condition_index").reset_index(drop=True)
    ev = np.linalg.eigvalsh(np.corrcoef(sub[features].to_numpy(dtype=float), rowvar=False)) if len(features) > 1 \
        else np.array([1.0])
    return Outcome({"condition_number": float(ci.max()), "n": len(sub), "variables": len(names),
                    "correlation_matrix_condition_number": float(np.sqrt(ev.max() / ev.min())) if ev.min() > 0 else np.inf},
                   {"Condition indices and variance proportions": t}, notes=dropped_note(dropped), rows_used=len(sub))


@register("ml.correlation_pairs", "All pairwise feature correlations, sorted", "Multicollinearity", _SUP + _UNS,
          params=(P("features", "columns"),),
          description="""Every pair of numeric features with its Pearson correlation (and two-sided t-test
p-value, H0: ρ = 0), Spearman rank correlation and pairwise-complete n, sorted by the larger of |Pearson|
and |Spearman|. No cut-off is applied: the full ranked list is reported.""",
          references=("Kutner, Nachtsheim & Neter (2004), Applied Linear Regression Models, 4th ed.",))
def correlation_pairs(ctx: RunContext, features) -> Outcome:
    numf = [f for f in features if pd.api.types.is_numeric_dtype(ctx.df[f])]
    skipped = [f for f in features if f not in numf]
    if len(numf) < 2:
        raise NotApplicable("Need at least two numeric features.")
    D = ctx.df[numf].astype(float)
    pear, spear = D.corr("pearson"), D.corr("spearman")
    nn = D.notna().astype(int)
    cnt = nn.T @ nn
    rows = []
    for i, a in enumerate(numf):
        for b in numf[i + 1:]:
            r, n = pear.loc[a, b], int(cnt.loc[a, b])
            tstat = r * np.sqrt((n - 2) / (1 - r ** 2)) if n > 2 and abs(r) < 1 else np.nan
            p = float(2 * stats.t.sf(abs(tstat), n - 2)) if np.isfinite(tstat) else (0.0 if abs(r) == 1 else np.nan)
            rows.append({"feature_1": a, "feature_2": b, "pearson": r, "pearson_p_value": p,
                         "spearman": spear.loc[a, b], "n": n})
    t = pd.DataFrame(rows)
    t["max_abs"] = t[["pearson", "spearman"]].abs().max(axis=1)
    t = t.sort_values(["max_abs", "feature_1", "feature_2"], ascending=[False, True, True]).reset_index(drop=True)
    return Outcome({"pairs": len(t), "max_abs_correlation": float(t.loc[0, "max_abs"]),
                    "most_correlated_pair": f"{t.loc[0, 'feature_1']} / {t.loc[0, 'feature_2']}"},
                   {"Correlation pairs": t.drop(columns="max_abs")}, rows_used=len(D),
                   notes=([f"Non-numeric features skipped: {skipped}."] if skipped else [])
                   + ["Correlations use pairwise-complete rows."])


# ── unsupervised ───────────────────────────────────────────────────────────

def _cluster_data(ctx: RunContext, features, segment, standardize):
    sub, dropped = _numeric_matrix(ctx, features, [segment])
    X = sub[features].to_numpy(dtype=float)
    if standardize:
        sd = X.std(axis=0, ddof=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    lab = sub[segment].astype(str).to_numpy()
    k = len(np.unique(lab))
    if k < 2 or k >= len(lab):
        raise NotApplicable(f"Need 2 <= clusters < rows (got {k} clusters, {len(lab)} rows).")
    return X, lab, k, dropped


_SEG = P("segment", help="Cluster / segment label assigned by the model")


@register("ml.cluster_quality", "Cluster validity indices (silhouette, Davies–Bouldin, Calinski–Harabasz, WCSS)",
          "Clustering", _UNS,
          params=(P("features", "columns", help="Variables the clustering was built on"), _SEG,
                  P("standardize", "boolean", default=False, help="z-score features first (match the model's space)"),
                  P("silhouette_sample", "integer", default=5000, help="Max rows for silhouette (seeded sample)")),
          description="""Internal validity of given cluster labels in feature space: mean silhouette
s = (b − a)/max(a, b) (−1..1, higher = better separated; per cluster mean and share of negative
silhouettes), Davies–Bouldin index (lower = better), Calinski–Harabasz pseudo-F (higher = better),
within-cluster sum of squares (inertia / WCSS), between-cluster SS and R² = BCSS/TSS. Distances are
Euclidean in the (optionally standardised) feature space; compute in the space the model clustered in.""",
          references=("Rousseeuw (1987), Silhouettes, J. Comput. Appl. Math. 20",
                      "Davies & Bouldin (1979), IEEE TPAMI 1",
                      "Caliński & Harabasz (1974), Communications in Statistics 3"))
def cluster_quality(ctx: RunContext, features, segment, standardize=False, silhouette_sample=5000) -> Outcome:
    from sklearn import metrics as skm
    X, lab, k, dropped = _cluster_data(ctx, features, segment, standardize)
    n = len(lab)
    idx = np.arange(n) if n <= silhouette_sample else np.sort(ctx.rng().choice(n, int(silhouette_sample), replace=False))
    if len(np.unique(lab[idx])) < 2:
        raise NotApplicable("The silhouette sample contains a single cluster.")
    sil = skm.silhouette_samples(X[idx], lab[idx])
    mu = X.mean(axis=0)
    rows, wcss, bcss = [], 0.0, 0.0
    for c in sorted(np.unique(lab), key=str):
        Xc = X[lab == c]
        cen = Xc.mean(axis=0)
        w = float(((Xc - cen) ** 2).sum())
        wcss += w
        bcss += len(Xc) * float(((cen - mu) ** 2).sum())
        sc = sil[lab[idx] == c]
        rows.append({"cluster": c, "n": len(Xc), "share": len(Xc) / n, "within_ss": w,
                     "mean_distance_to_centroid": float(np.sqrt(((Xc - cen) ** 2).sum(axis=1)).mean()),
                     "mean_silhouette": float(sc.mean()) if len(sc) else np.nan,
                     "share_negative_silhouette": float((sc < 0).mean()) if len(sc) else np.nan})
    tss = float(((X - mu) ** 2).sum())
    return Outcome({"n": n, "clusters": k, "silhouette": float(sil.mean()),
                    "davies_bouldin": float(skm.davies_bouldin_score(X, lab)),
                    "calinski_harabasz": float(skm.calinski_harabasz_score(X, lab)), "wcss_inertia": wcss,
                    "bcss": bcss, "r2_bcss_over_tss": bcss / tss if tss > 0 else np.nan},
                   {"Per cluster": pd.DataFrame(rows)}, rows_used=n,
                   notes=dropped_note(dropped) + ([f"Silhouette on a seeded sample of {len(idx)} rows."]
                                                  if len(idx) < n else []))


@register("ml.cluster_sizes", "Cluster size distribution and concentration (HHI)", "Clustering", _UNS,
          params=(_SEG,),
          description="""Count and share of each cluster; Herfindahl–Hirschman index HHI = Σ s_k², normalised
HHI* = (HHI − 1/K)/(1 − 1/K) (0 = equal sizes, 1 = all in one cluster), effective number of clusters 1/HHI,
Shannon entropy and Pielou evenness H/ln K, largest and smallest shares. Very small clusters are hard to
validate and unstable; dominant clusters indicate poor segmentation.""",
          references=("Herfindahl (1950); Hirschman (1964), The paternity of an index, AER 54",
                      "Pielou (1966), The measurement of diversity, J. Theoretical Biology 13"))
def cluster_sizes(ctx: RunContext, segment) -> Outcome:
    s = ctx.df[segment]
    vc = s.dropna().astype(str).value_counts()
    vc = vc.reindex(sorted(vc.index, key=str))
    k, n = len(vc), int(vc.sum())
    if k < 1:
        raise NotApplicable("No labels.")
    sh = vc.to_numpy() / n
    hhi = float(np.sum(sh ** 2))
    H = float(-np.sum(sh * np.log(sh)))
    t = pd.DataFrame({"cluster": vc.index, "n": vc.to_numpy(), "share": sh})
    return Outcome({"clusters": k, "n": n, "hhi": hhi,
                    "hhi_normalised": (hhi - 1 / k) / (1 - 1 / k) if k > 1 else np.nan,
                    "effective_clusters": 1 / hhi, "entropy": H, "pielou_evenness": H / np.log(k) if k > 1 else np.nan,
                    "largest_share": float(sh.max()), "smallest_share": float(sh.min()),
                    "smallest_cluster_n": int(vc.min())},
                   {"Cluster sizes": t}, rows_used=n, notes=dropped_note(int(s.isna().sum()), "rows without a label"))


@register("ml.cluster_stability", "Cluster stability under bootstrap re-clustering (ARI, Jaccard)", "Clustering", _UNS,
          params=(P("features", "columns", help="Variables the clustering was built on"), _SEG,
                  P("model", "model", required=False,
                    help="Loaded clustering model to refit (needs fit + predict); default KMeans with K = #clusters"),
                  P("n_boot", "integer", default=20), P("standardize", "boolean", default=False)),
          description="""Bootstrap resamples of the rows (seeded) are re-clustered — by refitting a clone of the
given clustering model, or by KMeans with the same number of clusters (n_init=10, seeded) — and every row is
assigned with the refitted model's predict. Per resample: adjusted Rand index (ARI) between the given labels
and the re-clustering (1 = identical partition, ~0 = chance agreement). Per original cluster: Hennig's
cluster-wise stability = mean over resamples of the maximum Jaccard similarity with any re-clustered group.
With the KMeans default, low agreement can also reflect that the original algorithm is not KMeans.""",
          references=("Hubert & Arabie (1985), Comparing partitions, J. Classification 2",
                      "Hennig (2007), Cluster-wise assessment of cluster stability, CSDA 52",
                      "Lange et al. (2004), Stability-based validation of clustering solutions, Neural Computation 16"))
def cluster_stability(ctx: RunContext, features, segment, model=None, n_boot=20, standardize=False) -> Outcome:
    from sklearn.base import clone
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    X, lab, k, dropped = _cluster_data(ctx, features, segment, standardize)
    n = len(lab)
    if model:
        est, _ = _refittable(ctx, model)
        if not hasattr(est, "predict"):
            raise NotApplicable(f"Model '{model}' has no predict method, so new rows cannot be assigned.")
        method = f"refit clone of model '{model}'"
        Xin = pd.DataFrame(X, columns=features)
    else:
        est, method, Xin = None, f"KMeans (K={k}, n_init=10)", X
    clusters = sorted(np.unique(lab), key=str)
    aris, jac = [], {c: [] for c in clusters}
    for b in range(int(n_boot)):
        i = ctx.rng(b + 1).integers(0, n, n)
        e = clone(est) if est is not None else KMeans(n_clusters=k, n_init=10, random_state=ctx.seed + b)
        fitX = Xin.iloc[i] if isinstance(Xin, pd.DataFrame) else Xin[i]
        e.fit(fitX)
        new = np.asarray(e.predict(Xin)).astype(str)
        aris.append(adjusted_rand_score(lab, new))
        ct = pd.crosstab(lab, new)
        inter = ct.to_numpy(dtype=float)
        union = inter.sum(axis=1, keepdims=True) + inter.sum(axis=0, keepdims=True) - inter
        J = (inter / union).max(axis=1)
        for c, v in zip(ct.index, J):
            jac[c].append(float(v))
    aris = np.asarray(aris)
    per = pd.DataFrame({"cluster": clusters, "n": [int((lab == c).sum()) for c in clusters],
                        "mean_jaccard": [np.mean(jac[c]) for c in clusters],
                        "min_jaccard": [np.min(jac[c]) for c in clusters]})
    return Outcome({"method": method, "resamples": len(aris), "clusters": k, "n": n,
                    "mean_ari": float(aris.mean()), "sd_ari": float(aris.std(ddof=1)) if len(aris) > 1 else np.nan,
                    "min_ari": float(aris.min()), "max_ari": float(aris.max()),
                    "least_stable_cluster": per.sort_values(["mean_jaccard", "cluster"]).iloc[0]["cluster"]},
                   {"Cluster-wise stability (Jaccard)": per,
                    "ARI per resample": pd.DataFrame({"resample": np.arange(1, len(aris) + 1), "ari": aris})},
                   rows_used=n, notes=dropped_note(dropped) + [
                       f"Seeded from {ctx.seed}. Jaccard is computed on all rows after assigning them with the "
                       "re-fitted model (a variant of Hennig 2007, which uses the resampled rows)."])


@register("ml.pca", "PCA explained variance, loadings, Bartlett sphericity and KMO", "Dimensionality", _UNS,
          params=(P("features", "columns"), P("standardize", "boolean", default=True,
                                              help="PCA on the correlation matrix (True) or covariance matrix"),
                  P("n_components", "integer", required=False, help="Components to show loadings for (default all, max 10)")),
          description="""Principal component analysis: eigenvalue, explained-variance ratio and cumulative ratio
per component; loadings (eigenvector × √eigenvalue = correlation of the feature with the component when
standardised); number of components needed for 80/90/95% cumulative variance. Bartlett's test of sphericity
χ² = −(n − 1 − (2p + 5)/6)·ln|R| with p(p−1)/2 df (H0: correlation matrix = identity) and the
Kaiser–Meyer–Olkin measure of sampling adequacy (overall and per variable).""",
          references=("Jolliffe (2002), Principal Component Analysis, 2nd ed., Springer",
                      "Bartlett (1950), Tests of significance in factor analysis, Brit. J. Psych. Stat. Section 3",
                      "Kaiser (1974), An index of factorial simplicity, Psychometrika 39"))
def pca(ctx: RunContext, features, standardize=True, n_components=None) -> Outcome:
    sub, dropped = _numeric_matrix(ctx, features)
    p, n = len(features), len(sub)
    if p < 2 or n <= p:
        raise NotApplicable("Need at least 2 features and more rows than features.")
    X = sub[features].to_numpy(dtype=float)
    R = np.corrcoef(X, rowvar=False)
    M = R if standardize else np.cov(X, rowvar=False)
    ev, vec = np.linalg.eigh(M)
    order = np.argsort(ev)[::-1]
    ev, vec = np.clip(ev[order], 0, None), vec[:, order]
    vec = vec * np.where(vec[np.abs(vec).argmax(axis=0), np.arange(p)] < 0, -1, 1)   # deterministic sign
    ratio = ev / ev.sum()
    cum = np.cumsum(ratio)
    var_t = pd.DataFrame({"component": [f"PC{i}" for i in range(1, p + 1)], "eigenvalue": ev,
                          "explained_ratio": ratio, "cumulative_ratio": cum})
    kshow = min(int(n_components or p), p, 10)
    load = pd.DataFrame(vec[:, :kshow] * np.sqrt(ev[:kshow]), columns=[f"PC{i}" for i in range(1, kshow + 1)])
    load.insert(0, "feature", features)
    det = np.linalg.det(R)
    bart = -(n - 1 - (2 * p + 5) / 6) * np.log(det) if det > 0 else np.inf
    dfb = p * (p - 1) / 2
    Ri = np.linalg.pinv(R)
    part = -Ri / np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    off = ~np.eye(p, dtype=bool)
    r2, q2 = (R ** 2) * off, (part ** 2) * off
    kmo = float(r2.sum() / (r2.sum() + q2.sum()))
    load["kmo_msa"] = r2.sum(axis=0) / (r2.sum(axis=0) + q2.sum(axis=0))
    need = {f"components_for_{int(q * 100)}pct": int(np.searchsorted(cum, q - 1e-12) + 1) for q in (0.8, 0.9, 0.95)}
    return Outcome({"features": p, "n": n, "pc1_explained_ratio": float(ratio[0]), **need,
                    "bartlett_chi2": float(bart), "bartlett_df": dfb, "bartlett_p_value": float(stats.chi2.sf(bart, dfb)),
                    "kmo": kmo, "matrix": "correlation" if standardize else "covariance"},
                   {"Explained variance": var_t, "Loadings": load}, notes=dropped_note(dropped), rows_used=n)


# ── surrogate fallback ─────────────────────────────────────────────────────

@register("ml.surrogate_explainability",
          "SURROGATE explainability (RandomForest stand-in — NOT the model under validation)",
          "Explainability", _SUP,
          params=(P("features", "columns"), _TARGET_OPT, _ACTUAL_OPT,
                  P("score", required=False, help="Model score column: fit the surrogate to mimic the model "
                                                  "(global surrogate) instead of the outcome"),
                  P("test_size", "number", default=0.3), P("max_depth", "integer", default=5),
                  P("n_estimators", "integer", default=200), P("n_repeats", "integer", default=5),
                  P("max_shap_rows", "integer", default=500)),
          description="""FALLBACK when the model itself is not available. A RandomForest SURROGATE (seeded) is
fitted on a stratified/random training split and evaluated on the held-out split. With `score` it is a
global surrogate of the model under validation (it learns the model's score; fidelity = held-out R² and
RMSE between surrogate and model score); otherwise it learns the outcome (held-out AUC or R²). Importances
reported for the SURROGATE on the held-out split: permutation importance, mean |SHAP| (TreeExplainer) and
impurity importance from training. These describe the data / the score surface as learned by a different
model, NOT the internals of the model under validation; low fidelity makes them uninformative.""",
          references=("Molnar (2022), Interpretable Machine Learning, 2nd ed., ch. 8.6 (global surrogate)",
                      "Craven & Shavlik (1996), Extracting tree-structured representations of trained networks, NIPS",
                      "Breiman (2001), Random forests, Machine Learning 45"))
def surrogate_explainability(ctx: RunContext, features, target=None, actual=None, score=None, test_size=0.3,
                             max_depth=5, n_estimators=200, n_repeats=5, max_shap_rows=500) -> Outcome:
    import shap
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    if score:
        ycol, task, fit_to = score, "regression", f"model score '{score}' (global surrogate)"
    else:
        ycol, task = _task(target, actual)
        fit_to = f"outcome '{ycol}'"
    sub, dropped = complete(ctx.df, _uniq([*features, ycol]))
    y = _yvals(sub, ycol, task)
    ok = np.isfinite(y)
    dropped += int((~ok).sum())
    sub, y = sub[ok], y[ok]
    X = sub[list(features)].copy()
    coded = [f for f in features if not pd.api.types.is_numeric_dtype(X[f])]
    for f in coded:
        X[f] = pd.Categorical(X[f].astype(str), categories=sorted(X[f].astype(str).unique())).codes
    X = X.astype(float)
    if task == "classification":
        require_two_classes(y, ycol)
        y = y.astype(int)
    if len(y) < 20:
        raise NotApplicable("Fewer than 20 complete rows.")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=ctx.seed,
                                          stratify=y if task == "classification" else None)
    Model = RandomForestClassifier if task == "classification" else RandomForestRegressor
    rf = Model(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=ctx.seed, n_jobs=1).fit(Xtr, ytr)
    s_te = predict_scores(rf, Xte)
    name, hib, fn = _metric_spec("auc" if task == "classification" else "rmse", task)
    perf = {"holdout_auc": _auc(yte, s_te)} if task == "classification" else \
        {"holdout_r2": _reg_metrics(yte, s_te)["r2"], "holdout_rmse": fn(yte, s_te, 0.5)}
    imp = _perm_importance(rf, Xte, yte, fn, hib, 0.5, n_repeats, ctx.seed).mean(axis=1)
    Xs = _sample_rows(Xte, int(max_shap_rows), ctx.rng())
    sv = shap.TreeExplainer(rf).shap_values(Xs)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    t = pd.DataFrame({"feature": list(features), "surrogate_permutation_importance_holdout": imp,
                      "surrogate_mean_abs_shap_holdout": np.abs(sv).mean(axis=0),
                      "surrogate_impurity_importance_train": rf.feature_importances_})
    t = t.sort_values(["surrogate_permutation_importance_holdout", "feature"], ascending=[False, True]).reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    summary = {"surrogate": "RandomForest SURROGATE — NOT the model under validation", "fitted_to": fit_to,
               "n_train": len(ytr), "n_holdout": len(yte), **perf,
               "top_feature_surrogate": t.loc[0, "feature"]}
    notes = dropped_note(dropped) + [
        "SURROGATE — not the model under validation: a RandomForest (max_depth "
        f"{max_depth}, {n_estimators} trees, seed {ctx.seed}) fitted on {len(ytr)} rows and evaluated on a "
        f"held-out split of {len(yte)} rows. Its importances describe {fit_to} as learned by the surrogate, "
        "not the internal logic of the validated model. Load the model (load_model) and use "
        "ml.permutation_importance / ml.shap_importance for the real model."]
    if coded:
        notes.append(f"Categorical features ordinal-coded for the surrogate: {coded}.")
    return Outcome(summary, {"SURROGATE feature importance (held-out)": t}, rows_used=len(y), notes=notes)
