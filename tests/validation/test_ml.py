"""Machine-learning validation tests (ask/validation/t_ml.py): known answers vs sklearn / scipy /
statsmodels, NotApplicable paths, NaN handling and determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as skm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from ask.validation import core
from ask.validation.core import RunContext, run_test
from ask.validation.models import LoadedModel

FEATS = ["x1", "x2", "x3", "noise"]


def _lm(name, obj, feats=FEATS, flavour="sklearn-like"):
    return LoadedModel(name, "memory", obj, flavour, "", list(feats))


@pytest.fixture(scope="module")
def cls():
    rng = np.random.default_rng(7)
    n = 800
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n), "x3": rng.normal(0, 1, n),
                       "noise": rng.normal(0, 1, n)})
    logit = -1.0 + 1.5 * df.x1 - 1.0 * df.x2 + 0.5 * df.x3
    df["y"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["sample"] = np.where(np.arange(n) < 500, "train", "test")
    tr = df[df["sample"] == "train"]
    lr = LogisticRegression().fit(tr[FEATS], tr["y"])
    rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0).fit(tr[FEATS], tr["y"])
    dt = DecisionTreeClassifier(max_depth=3, random_state=0).fit(tr[FEATS], tr["y"])
    df["p_lr"] = lr.predict_proba(df[FEATS])[:, 1]
    models = {"lr": _lm("lr", lr), "rf": _lm("rf", rf), "dt": _lm("dt", dt),
              "fn": _lm("fn", lambda X: lr.predict_proba(X)[:, 1], flavour="callable")}
    return df, models, lr


@pytest.fixture(scope="module")
def reg():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(5, 2, n), "x2": rng.normal(0, 1, n), "x3": rng.uniform(1, 3, n),
                       "noise": rng.normal(0, 1, n)})
    df["a"] = 2.0 * df.x1 - 1.5 * df.x2 + 0.5 * df.x3 + rng.normal(0, 1, n)
    df["sample"] = np.where(np.arange(n) < 250, "train", "test")
    tr = df[df["sample"] == "train"]
    lin = LinearRegression().fit(tr[FEATS], tr["a"])
    df["pred"] = lin.predict(df[FEATS])
    return df, {"lin": _lm("lin", lin)}, lin


def _ctx(df, models=None, **kw):
    return RunContext(df=df, models=models or {}, source_name="synthetic", **kw)


def _ok(res):
    assert res.status == "ok", res.error
    return res


# ── classification performance ─────────────────────────────────────────────

def test_classification_metrics_match_sklearn(cls):
    df, models, lr = cls
    t = df[df["sample"] == "test"]
    r = _ok(run_test("ml.classification_metrics", _ctx(t), {"target": "y", "score": "p_lr", "threshold": 0.4,
                                                             "beta": 2}))
    y, s = t["y"].to_numpy(), t["p_lr"].to_numpy()
    yh = (s >= 0.4).astype(int)
    S = r.summary
    assert S["accuracy"] == pytest.approx(skm.accuracy_score(y, yh))
    assert S["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, yh))
    assert S["precision"] == pytest.approx(skm.precision_score(y, yh))
    assert S["recall"] == pytest.approx(skm.recall_score(y, yh))
    assert S["f1"] == pytest.approx(skm.f1_score(y, yh))
    assert S["f_beta"] == pytest.approx(skm.fbeta_score(y, yh, beta=2))
    assert S["mcc"] == pytest.approx(skm.matthews_corrcoef(y, yh))
    assert S["cohen_kappa"] == pytest.approx(skm.cohen_kappa_score(y, yh))
    assert S["log_loss"] == pytest.approx(skm.log_loss(y, s))
    assert S["roc_auc"] == pytest.approx(skm.roc_auc_score(y, s))
    assert S["pr_auc"] == pytest.approx(skm.average_precision_score(y, s))
    assert S["brier"] == pytest.approx(skm.brier_score_loss(y, s))
    tn, fp, fn, tp = skm.confusion_matrix(y, yh).ravel()
    cm = r.tables["Confusion matrix"]
    assert cm["predicted_1"].tolist() == [tp, fp] and cm["predicted_0"].tolist() == [fn, tn]
    # the same numbers when the model scores the rows itself
    r2 = _ok(run_test("ml.classification_metrics", _ctx(t, models), {"target": "y", "model": "lr", "threshold": 0.4,
                                                                      "beta": 2}))
    assert r2.summary["roc_auc"] == pytest.approx(S["roc_auc"])


def test_classification_metrics_bootstrap_nan_and_single_class(cls):
    df, _, _ = cls
    d = df.copy()
    d.loc[:9, "p_lr"] = np.nan
    a = _ok(run_test("ml.classification_metrics", _ctx(d), {"target": "y", "score": "p_lr", "n_boot": 50}))
    b = _ok(run_test("ml.classification_metrics", _ctx(d), {"target": "y", "score": "p_lr", "n_boot": 50}))
    assert a.rows_used == len(d) - 10 and any("10 rows" in n for n in a.notes)
    assert a.tables["Metrics"].equals(b.tables["Metrics"])
    m = a.tables["Metrics"].set_index("metric")
    assert m.loc["roc_auc", "ci_low_95"] < m.loc["roc_auc", "value"] < m.loc["roc_auc", "ci_high_95"]
    one = d.assign(y=0)
    assert run_test("ml.classification_metrics", _ctx(one), {"target": "y", "score": "p_lr"}).status == "not_applicable"
    assert run_test("ml.classification_metrics", _ctx(d), {"target": "y"}).status == "error"


def test_threshold_sweep_counts_and_optimum(cls):
    df, _, _ = cls
    th = [0.2, 0.4, 0.5, 0.6]
    r = _ok(run_test("ml.threshold_sweep", _ctx(df), {"target": "y", "score": "p_lr", "thresholds": th,
                                                      "cost_fp": 1, "cost_fn": 5}))
    t = r.tables["Metrics by threshold"]
    y, s = df["y"].to_numpy(), df["p_lr"].to_numpy()
    for _, row in t.iterrows():
        tn, fp, fn, tp = skm.confusion_matrix(y, (s >= row["threshold"]).astype(int)).ravel()
        assert (row["TP"], row["FP"], row["TN"], row["FN"]) == (tp, fp, tn, fn)
        assert row["mcc"] == pytest.approx(skm.matthews_corrcoef(y, (s >= row["threshold"]).astype(int)))
    assert r.summary["cost_optimal_threshold"] == t.loc[t["expected_cost"].idxmin(), "threshold"]
    assert r.summary["bayes_threshold_if_calibrated"] == pytest.approx(1 / 6)
    full = _ok(run_test("ml.threshold_sweep", _ctx(df), {"target": "y", "score": "p_lr"}))
    fpr, tpr, _ = skm.roc_curve(y, s)
    assert full.summary["max_youden_j"] <= np.max(tpr - fpr) + 1e-12
    assert full.summary["max_youden_j"] > np.max(tpr - fpr) - 0.05


def test_multiclass_matches_sklearn():
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = np.where(X.a > 0.5, 2, np.where(X.b > 0, 1, 0))
    flip = rng.uniform(size=n) < 0.15
    y[flip] = rng.integers(0, 3, flip.sum())
    df = X.assign(y=y.astype(float))
    lr = LogisticRegression().fit(X, y)
    df["pred"] = lr.predict(X)
    r = _ok(run_test("ml.multiclass_metrics", _ctx(df), {"target": "y", "predicted": "pred"}))
    yt, yp = df["y"].astype(int), df["pred"]
    assert r.summary["accuracy"] == pytest.approx(skm.accuracy_score(yt, yp))
    assert r.summary["f1_macro"] == pytest.approx(skm.f1_score(yt, yp, average="macro"))
    assert r.summary["f1_weighted"] == pytest.approx(skm.f1_score(yt, yp, average="weighted"))
    assert r.summary["precision_micro"] == pytest.approx(skm.precision_score(yt, yp, average="micro"))
    assert r.summary["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(yt, yp))
    assert r.summary["mcc"] == pytest.approx(skm.matthews_corrcoef(yt, yp))
    rm = _ok(run_test("ml.multiclass_metrics", _ctx(df, {"m": _lm("m", lr, ["a", "b"])}),
                      {"target": "y", "model": "m"}))
    assert rm.summary["accuracy"] == pytest.approx(r.summary["accuracy"])
    assert rm.summary["log_loss"] == pytest.approx(skm.log_loss(yt, lr.predict_proba(X)))
    assert rm.summary["roc_auc_ovr_macro"] == pytest.approx(
        skm.roc_auc_score(yt, lr.predict_proba(X), multi_class="ovr"))


def test_probability_calibration_known_values(cls):
    df, _, _ = cls
    r = _ok(run_test("ml.probability_calibration", _ctx(df), {"target": "y", "score": "p_lr"}))
    t = r.tables["Reliability table"]
    y, s = df["y"].to_numpy(), df["p_lr"].to_numpy()
    assert r.summary["brier"] == pytest.approx(skm.brier_score_loss(y, s))
    assert r.summary["ece"] == pytest.approx(float(np.sum(t["n"] / len(y) * (t["observed_rate"] - t["mean_predicted"]).abs())))
    assert t["n"].sum() == len(y)
    # well-specified logistic model: slope near 1
    assert 0.6 < r.summary["calibration_slope"] < 1.5
    bad = df.assign(p_lr=df["p_lr"] * 3)
    assert run_test("ml.probability_calibration", _ctx(bad), {"target": "y", "score": "p_lr"}).status == "error"


def test_gains_lift_hand_example():
    df = pd.DataFrame({"y": [1, 1, 0, 1, 0, 0, 0, 0, 1, 0], "s": np.arange(10, 0, -1) / 10})
    r = _ok(run_test("ml.gains_lift", _ctx(df), {"target": "y", "score": "s", "bins": 5}))
    t = r.tables["Gains and lift"]
    assert t["events"].tolist() == [2, 1, 0, 0, 1]
    assert t["cum_event_capture"].iloc[0] == pytest.approx(0.5)
    assert r.summary["top_band_lift"] == pytest.approx(1.0 / 0.4)
    assert r.summary["ks_exact"] == pytest.approx(np.max(np.subtract(*skm.roc_curve(df.y, df.s)[1::-1])))


# ── regression ────────────────────────────────────────────────────────────

def test_regression_metrics_match_sklearn(reg):
    df, models, _ = reg
    r = _ok(run_test("ml.regression_metrics", _ctx(df), {"actual": "a", "predicted": "pred", "n_features": 4}))
    a, p = df["a"].to_numpy(), df["pred"].to_numpy()
    S = r.summary
    assert S["rmse"] == pytest.approx(np.sqrt(skm.mean_squared_error(a, p)))
    assert S["mae"] == pytest.approx(skm.mean_absolute_error(a, p))
    assert S["mape"] == pytest.approx(skm.mean_absolute_percentage_error(a, p))
    assert S["r2"] == pytest.approx(skm.r2_score(a, p))
    assert S["median_ae"] == pytest.approx(skm.median_absolute_error(a, p))
    assert S["max_error"] == pytest.approx(skm.max_error(a, p))
    assert S["explained_variance"] == pytest.approx(skm.explained_variance_score(a, p))
    n = len(a)
    assert S["adjusted_r2"] == pytest.approx(1 - (1 - S["r2"]) * (n - 1) / (n - 5))
    assert S["bias_mean_predicted_minus_actual"] == pytest.approx(np.mean(p - a))
    rm = _ok(run_test("ml.regression_metrics", _ctx(df, models), {"actual": "a", "model": "lin"}))
    assert rm.summary["adjusted_r2"] == pytest.approx(S["adjusted_r2"])
    small = pd.DataFrame({"a": [0.0, 2.0, 4.0], "p": [1.0, 2.0, 2.0]})
    rs = _ok(run_test("ml.regression_metrics", _ctx(small), {"actual": "a", "predicted": "p"}))
    assert rs.summary["mape"] == pytest.approx(0.25)                       # actual = 0 excluded
    assert rs.summary["smape"] == pytest.approx((2 / 1 + 0 + 4 / 6) / 3)


def test_residual_diagnostics(reg):
    df, models, _ = reg
    r = _ok(run_test("ml.residual_diagnostics", _ctx(df), {"actual": "a", "predicted": "pred"}))
    assert r.tables["Residuals by prediction bin"]["n"].sum() == len(df)
    assert r.summary["mz_slope"] == pytest.approx(1.0, abs=0.1)
    import statsmodels.api as sm
    fit = sm.OLS(df["a"], sm.add_constant(df["pred"])).fit()
    assert r.summary["mz_intercept"] == pytest.approx(fit.params.iloc[0])


# ── overfitting ───────────────────────────────────────────────────────────

def test_train_test_gap(cls, reg):
    df, models, _ = cls
    r = _ok(run_test("ml.train_test_gap", _ctx(df, models), {"target": "y", "model": "rf", "sample": "sample",
                                                              "n_boot": 100}))
    t = r.tables["Train vs test"].set_index("metric")
    tr = df[df["sample"] == "train"]
    rf = models["rf"].obj
    assert t.loc["auc", "train"] == pytest.approx(skm.roc_auc_score(tr.y, rf.predict_proba(tr[FEATS])[:, 1]))
    assert t.loc["auc", "gap_train_minus_test"] == pytest.approx(t.loc["auc", "train"] - t.loc["auc", "test"])
    assert t.loc["auc", "gap_ci_low_95"] <= t.loc["auc", "gap_ci_high_95"]
    assert r.summary["train_sample"] == "train"
    again = run_test("ml.train_test_gap", _ctx(df, models), {"target": "y", "model": "rf", "sample": "sample",
                                                             "n_boot": 100})
    assert again.tables["Train vs test"].equals(r.tables["Train vs test"])
    rdf, rmodels, _ = reg
    rr = _ok(run_test("ml.train_test_gap", _ctx(rdf), {"actual": "a", "predicted": "pred", "sample": "sample",
                                                      "metrics": ["rmse", "r2"], "n_boot": 50}))
    assert set(rr.tables["Train vs test"]["metric"]) == {"rmse", "r2"}
    bad = run_test("ml.train_test_gap", _ctx(rdf), {"actual": "a", "predicted": "pred", "sample": "sample",
                                                    "metrics": ["auc"]})
    assert bad.status == "error"


def test_cross_validation_refits_real_model(cls):
    df, models, _ = cls
    r = _ok(run_test("ml.cross_validation", _ctx(df, models), {"model": "lr", "target": "y", "k": 4}))
    per = r.tables["Per fold"]
    assert per["fold"].nunique() == 4 and per["n_test"].sum() == 3 * len(df)      # 3 metrics x all rows
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    ref = cross_val_score(LogisticRegression(), df[FEATS], df["y"], scoring="roc_auc",
                          cv=StratifiedKFold(4, shuffle=True, random_state=core.DEFAULT_SEED))
    assert r.summary["auc_cv_mean"] == pytest.approx(ref.mean())
    na = run_test("ml.cross_validation", _ctx(df, models), {"model": "fn", "target": "y"})
    assert na.status == "not_applicable" and "refit" in na.error


def test_learning_curve(cls):
    df, models, _ = cls
    r = _ok(run_test("ml.learning_curve", _ctx(df, models), {"model": "dt", "target": "y", "k": 3,
                                                              "train_sizes": [0.2, 1.0]}))
    t = r.tables["Learning curve"]
    assert len(t) == 2 and t["n_train_mean"].iloc[0] < t["n_train_mean"].iloc[1]
    assert len(r.figures) == 1


# ── explainability ────────────────────────────────────────────────────────

def test_permutation_importance_deterministic_and_sensible(cls):
    df, models, _ = cls
    p = {"model": "lr", "target": "y", "n_repeats": 5}
    a = _ok(run_test("ml.permutation_importance", _ctx(df, models), p))
    b = _ok(run_test("ml.permutation_importance", _ctx(df, models), p))
    assert a.run_id == b.run_id and a.tables["Permutation importance"].equals(b.tables["Permutation importance"])
    t = a.tables["Permutation importance"]
    assert t.loc[0, "feature"] == "x1"
    assert t.set_index("feature").loc["noise", "importance_mean"] < 0.01
    from sklearn.inspection import permutation_importance
    ref = permutation_importance(models["lr"].obj, df[FEATS], df["y"], scoring="roc_auc", n_repeats=5,
                                 random_state=core.DEFAULT_SEED)
    assert t.set_index("feature").loc[FEATS, "importance_mean"].to_numpy() == pytest.approx(ref.importances_mean)
    c = _ok(run_test("ml.permutation_importance", _ctx(df, models), p | {"metric": "log_loss"}))
    assert c.tables["Permutation importance"].loc[0, "feature"] == "x1"


def test_shap_tree_and_model_agnostic(cls):
    df, models, _ = cls
    r = _ok(run_test("ml.shap_importance", _ctx(df, models), {"model": "rf", "max_rows": 100}))
    assert "TreeExplainer" in r.summary["method"]
    t = r.tables["SHAP importance"]
    assert t.loc[0, "feature"] in ("x1", "x2")
    assert t.set_index("feature").loc["x1", "direction_spearman"] > 0.5
    assert t.set_index("feature").loc["x2", "direction_spearman"] < -0.5
    again = _ok(run_test("ml.shap_importance", _ctx(df, models), {"model": "rf", "max_rows": 100}))
    assert again.tables["SHAP importance"].equals(t)


@pytest.mark.slow
def test_shap_model_agnostic_exact_and_permutation(cls):
    """Model-agnostic SHAP (numba JIT compilation makes the first call take several seconds)."""
    df, models, _ = cls
    p = {"model": "lr", "max_rows": 40, "background_rows": 20}
    a = _ok(run_test("ml.shap_importance", _ctx(df, models), p))
    b = _ok(run_test("ml.shap_importance", _ctx(df, models), p))
    assert "exact" in a.summary["method"] and a.summary["additivity_max_error"] < 1e-6
    assert a.tables["SHAP importance"].equals(b.tables["SHAP importance"])
    assert a.tables["SHAP importance"].loc[0, "feature"] == "x1"
    wide = df.copy()
    extra = [f"z{i}" for i in range(8)]
    for i, c in enumerate(extra):
        wide[c] = np.random.default_rng(i).normal(size=len(df))
    feats = FEATS + extra
    m = LogisticRegression().fit(wide[feats], wide["y"])
    ctx = _ctx(wide, {"w": _lm("w", m, feats)})
    c = _ok(run_test("ml.shap_importance", ctx, {"model": "w", "max_rows": 20, "background_rows": 10}))
    d = _ok(run_test("ml.shap_importance", ctx, {"model": "w", "max_rows": 20, "background_rows": 10}))
    assert "permutation" in c.summary["method"]
    assert c.tables["SHAP importance"].equals(d.tables["SHAP importance"])


def test_partial_dependence_and_monotonicity(cls):
    df, models, lr = cls
    r = _ok(run_test("ml.partial_dependence", _ctx(df, models), {"model": "lr", "explain": ["x1", "x2"],
                                                                  "grid_points": 10, "max_rows": 100}))
    s = r.tables["Feature effect summary"].set_index("feature")
    assert s.loc["x1", "pd_spearman_vs_grid"] == pytest.approx(1.0)
    assert s.loc["x2", "pd_spearman_vs_grid"] == pytest.approx(-1.0)
    m = _ok(run_test("ml.monotonicity", _ctx(df, models), {"model": "lr", "expected_signs": {"x1": 1, "x2": 1},
                                                            "max_rows": 100}))
    t = m.tables["Monotonicity by feature"].set_index("feature")
    assert t.loc["x1", "share_curves_violating"] == 0.0           # logistic in x1 with positive coefficient
    assert t.loc["x2", "share_curves_violating"] == 1.0           # negative coefficient vs expected +1
    assert m.summary["feature_with_max_violation_share"] == "x2"
    bad = run_test("ml.monotonicity", _ctx(df, models), {"model": "lr", "expected_signs": {"x1": 2}})
    assert bad.status == "error"


def test_explanation_stability(cls):
    df, models, _ = cls
    r = _ok(run_test("ml.explanation_stability", _ctx(df, models), {"model": "lr", "target": "y", "n_boot": 5,
                                                                     "top_k": 2}))
    assert r.summary["resamples"] == 5 and r.summary["top_feature_full"] == "x1"
    assert -1 <= r.summary["min_spearman"] <= r.summary["mean_spearman"] <= 1


# ── robustness ────────────────────────────────────────────────────────────

def test_noise_robustness(cls):
    df, models, _ = cls
    p = {"model": "lr", "target": "y", "noise_levels": [0, 0.1, 0.5], "n_repeats": 3}
    a = _ok(run_test("ml.noise_robustness", _ctx(df, models), p))
    b = _ok(run_test("ml.noise_robustness", _ctx(df, models), p))
    t = a.tables["Robustness by noise level"]
    assert t.loc[t["noise_level"] == 0, "mean_abs_score_change"].iloc[0] == 0.0
    assert t["mean_abs_score_change"].is_monotonic_increasing
    assert a.tables["Per repeat"].equals(b.tables["Per repeat"])
    assert a.summary["baseline_metric"] == pytest.approx(skm.roc_auc_score(df.y, df.p_lr))


def test_sensitivity_linear_known_answer(reg):
    df, models, lin = reg
    r = _ok(run_test("ml.sensitivity", _ctx(df, models), {"model": "lin", "shift_std": 1.0}))
    t = r.tables["Sensitivity by feature"]
    up = t[(t["feature"] == "x1") & (t["direction"] == "up")].iloc[0]
    assert up["mean_score_change"] == pytest.approx(lin.coef_[0] * df["x1"].std(ddof=1))
    assert r.summary["most_sensitive_feature"] == "x1"
    rp = _ok(run_test("ml.sensitivity", _ctx(df, models), {"model": "lin", "shift_pct": 0.1, "vary": ["x3"]}))
    d = rp.tables["Sensitivity by feature"].iloc[0]
    assert abs(d["mean_score_change"]) == pytest.approx(abs(lin.coef_[2]) * 0.1 * df["x3"].mean())


def test_scenario_stress_linear_known_answer(reg):
    df, models, lin = reg
    r = _ok(run_test("ml.scenario_stress", _ctx(df, models),
                     {"model": "lin", "shocks": {"x1": 1.1, "x2": {"add": 0.5}}, "weight": "x3"}))
    expected = lin.coef_[0] * 0.1 * df["x1"].mean() + lin.coef_[1] * 0.5
    assert r.summary["mean_score_change"] == pytest.approx(expected)
    assert r.summary["shocked_features"] == 2 and "weighted_relative_change" in r.summary
    bad = run_test("ml.scenario_stress", _ctx(df, models), {"model": "lin", "shocks": {"zzz": 1.1}})
    assert bad.status == "error"


def test_missing_value_robustness(cls):
    df, models, _ = cls
    r = _ok(run_test("ml.missing_value_robustness", _ctx(df, models), {"model": "lr", "target": "y"}))
    assert r.summary["features_where_model_fails_on_nan"] == 4       # LogisticRegression rejects NaN
    t = r.tables["Missing-value robustness"]
    med = t[t["treatment"] == "median / mode"].set_index("feature")
    assert med.loc["x1", "mean_abs_score_change"] > med.loc["noise", "mean_abs_score_change"]


def test_extrapolation_hand_example():
    df = pd.DataFrame({"x": [0, 1, 2, 3, 4, -1, 5, 2], "c": list("aabbbacz"),
                       "s": ["train"] * 5 + ["test"] * 3})
    r = _ok(run_test("ml.extrapolation", _ctx(df), {"features": ["x", "c"], "sample": "s", "quantile": 0.0}))
    t = r.tables["Extrapolation by feature"].set_index("feature")
    assert t.loc["x", "share_below_min"] == pytest.approx(1 / 3)
    assert t.loc["x", "share_above_max"] == pytest.approx(1 / 3)
    assert t.loc["c", "share_unseen_level"] == pytest.approx(2 / 3)
    assert r.summary["share_rows_any_feature_outside_range"] == pytest.approx(1.0)
    r2 = _ok(run_test("ml.extrapolation", RunContext(df=df[df.s == "train"], tables={"new": df[df.s == "test"]}),
                      {"features": ["x"], "other": "new"}))
    assert r2.summary["n_current"] == 3


# ── leakage & multicollinearity ─────────────────────────────────────────────

def test_leakage_screen_flags_leak(cls):
    df, _, _ = cls
    d = df.assign(leak=df["y"] + np.random.default_rng(0).normal(0, 0.01, len(df)),
                  cat=np.where(df["x1"] > 0, "hi", "lo"), miss=np.where(df["y"] == 1, np.nan, df["x3"]))
    r = _ok(run_test("ml.leakage_screen", _ctx(d), {"target": "y", "features": FEATS + ["leak", "cat", "miss"]}))
    t = r.tables["Single-feature association with outcome"].set_index("feature")
    assert r.summary["strongest_feature"] in ("leak", "miss")
    assert t.loc["leak", "single_feature_auc"] == pytest.approx(1.0)
    assert t.loc["miss", "missing_indicator_auc"] == pytest.approx(1.0)
    a = skm.roc_auc_score(d.y, d.x2)
    assert t.loc["x2", "single_feature_auc"] == pytest.approx(max(a, 1 - a))
    assert np.isfinite(t.loc["cat", "single_feature_auc"])


def test_train_test_duplicates_hand_example():
    df = pd.DataFrame({"a": [1, 2, 3, 1, 9, 2, 2], "b": list("xyzxqyy"), "id": [1, 2, 3, 4, 5, 2, 7],
                       "s": ["train"] * 4 + ["test"] * 3})
    r = _ok(run_test("ml.train_test_duplicates", _ctx(df), {"sample": "s", "features": ["a", "b"], "id": "id"}))
    S = r.summary
    assert S["test_rows_duplicating_train"] == 2 and S["duplicate_rows_within_train"] == 1
    assert S["duplicate_rows_within_test"] == 1 and S["ids_in_both"] == 1


def test_vif_condition_index_and_correlations(cls):
    df, _, _ = cls
    d = df.assign(x4=df["x1"] + 0.3 * df["x2"] + np.random.default_rng(1).normal(0, 0.2, len(df)))
    feats = ["x1", "x2", "x3", "x4"]
    r = _ok(run_test("ml.vif", _ctx(d), {"features": feats}))
    t = r.tables["VIF"].set_index("feature")
    r2 = LinearRegression().fit(d[["x2", "x3", "x4"]], d["x1"]).score(d[["x2", "x3", "x4"]], d["x1"])
    assert t.loc["x1", "vif"] == pytest.approx(1 / (1 - r2))
    assert np.diag(np.linalg.inv(d[feats].corr().to_numpy())) == pytest.approx(t.loc[feats, "vif"].to_numpy())
    c = _ok(run_test("ml.condition_index", _ctx(d), {"features": feats}))
    X = np.column_stack([np.ones(len(d)), d[feats].to_numpy()])
    assert c.summary["condition_number"] == pytest.approx(np.linalg.cond(X / np.linalg.norm(X, axis=0)))
    props = c.tables["Condition indices and variance proportions"].filter(like="prop_")
    assert props.sum(axis=0).to_numpy() == pytest.approx(np.ones(5))
    p = _ok(run_test("ml.correlation_pairs", _ctx(d), {"features": feats}))
    top = p.tables["Correlation pairs"].iloc[0]
    assert {top["feature_1"], top["feature_2"]} == {"x1", "x4"}
    from scipy import stats
    pr = stats.pearsonr(d.x1, d.x4)
    assert top["pearson"] == pytest.approx(pr.statistic) and top["pearson_p_value"] == pytest.approx(pr.pvalue, abs=1e-12)
    assert len(p.tables["Correlation pairs"]) == 6


# ── unsupervised ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(5)
    centres = np.array([[0, 0], [6, 0], [0, 6]])
    X = np.vstack([rng.normal(c, 0.7, (100, 2)) for c in centres])
    km = KMeans(3, n_init=10, random_state=0).fit(X)
    df = pd.DataFrame(X, columns=["f1", "f2"]).assign(cl=km.labels_)
    return df, km


def test_cluster_quality_matches_sklearn(blobs):
    df, km = blobs
    r = _ok(run_test("ml.cluster_quality", _ctx(df), {"features": ["f1", "f2"], "segment": "cl"}))
    X = df[["f1", "f2"]].to_numpy()
    assert r.summary["silhouette"] == pytest.approx(skm.silhouette_score(X, df.cl))
    assert r.summary["davies_bouldin"] == pytest.approx(skm.davies_bouldin_score(X, df.cl))
    assert r.summary["calinski_harabasz"] == pytest.approx(skm.calinski_harabasz_score(X, df.cl))
    assert r.summary["wcss_inertia"] == pytest.approx(km.inertia_, rel=1e-6)
    one = df.assign(cl=0)
    assert run_test("ml.cluster_quality", _ctx(one), {"features": ["f1", "f2"], "segment": "cl"}).status == "not_applicable"


def test_cluster_sizes_hhi():
    df = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 30 + ["c"] * 20 + [None] * 3})
    r = _ok(run_test("ml.cluster_sizes", _ctx(df), {"segment": "c"}))
    hhi = 0.5 ** 2 + 0.3 ** 2 + 0.2 ** 2
    assert r.summary["hhi"] == pytest.approx(hhi)
    assert r.summary["hhi_normalised"] == pytest.approx((hhi - 1 / 3) / (2 / 3))
    assert r.summary["effective_clusters"] == pytest.approx(1 / hhi) and r.rows_used == 100


def test_cluster_stability(blobs):
    df, km = blobs
    p = {"features": ["f1", "f2"], "segment": "cl", "n_boot": 5}
    r = _ok(run_test("ml.cluster_stability", _ctx(df), p))
    assert r.summary["mean_ari"] > 0.95
    assert (r.tables["Cluster-wise stability (Jaccard)"]["mean_jaccard"] > 0.9).all()
    rm = _ok(run_test("ml.cluster_stability", _ctx(df, {"km": _lm("km", km, ["f1", "f2"])}), p | {"model": "km"}))
    assert rm.summary["mean_ari"] > 0.95 and "refit" in rm.summary["method"]
    again = run_test("ml.cluster_stability", _ctx(df), p)
    assert again.summary == r.summary


def test_pca_matches_sklearn(cls):
    df, _, _ = cls
    d = df.assign(x4=df["x1"] + 0.3 * df["x2"] + np.random.default_rng(2).normal(0, 0.3, len(df)))
    feats = ["x1", "x2", "x3", "x4"]
    r = _ok(run_test("ml.pca", _ctx(d), {"features": feats}))
    Z = (d[feats] - d[feats].mean()) / d[feats].std(ddof=1)
    ref = PCA().fit(Z)
    assert r.tables["Explained variance"]["explained_ratio"].to_numpy() == pytest.approx(ref.explained_variance_ratio_)
    R = d[feats].corr().to_numpy()
    n, p = len(d), 4
    assert r.summary["bartlett_chi2"] == pytest.approx(-(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R)))
    assert 0 < r.summary["kmo"] <= 1


def test_surrogate_is_labelled_and_out_of_sample(cls):
    df, _, _ = cls
    r = _ok(run_test("ml.surrogate_explainability", _ctx(df), {"features": FEATS, "target": "y",
                                                                "n_estimators": 30, "max_shap_rows": 100}))
    assert "SURROGATE" in r.test_name and "NOT the model" in r.summary["surrogate"]
    assert any("SURROGATE — not the model under validation" in n for n in r.notes)
    assert r.summary["n_holdout"] == 240 and 0.6 < r.summary["holdout_auc"] <= 1
    t = r.tables["SURROGATE feature importance (held-out)"]
    assert t.loc[0, "feature"] in ("x1", "x2")
    g = _ok(run_test("ml.surrogate_explainability", _ctx(df), {"features": FEATS, "score": "p_lr",
                                                                "n_estimators": 30, "max_shap_rows": 50}))
    assert "global surrogate" in g.summary["fitted_to"] and g.summary["holdout_r2"] > 0.5


def test_every_ml_test_declared_well():
    core.load_all()
    ids = [k for k in core.REGISTRY if k.startswith("ml.")]
    assert len(ids) >= 28
    for k in ids:
        spec = core.REGISTRY[k]
        assert spec.references and len(spec.description) > 80, k
