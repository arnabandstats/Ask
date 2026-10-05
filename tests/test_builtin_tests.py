"""Every built-in deterministic test, run for real.

Part 1 runs each Test Category of each model type ON ITS OWN through the app's
runner (the same path the chat uses) and checks it produces exactly its outputs:
its Execution_Metrics rows, its Excel sheets, its charts, its files — and nothing
belonging to another test.

Part 2 checks the numbers: every statistical function in the engine's toolbox is
compared with an independent reference (sklearn / statsmodels / numpy) or a case
whose answer is known by hand.

The engine (ask/analysis/test_engine.py) is verbatim legacy code and is not
modified; the toolbox is reached through run_pipeline's closure.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from ask.analysis import runner
from ask.sources.loaders import load_path

pytestmark = pytest.mark.slow

CLF, REG = "Supervised: Classification", "Supervised: Regression"
CLU, DIM = "Unsupervised: Clustering", "Unsupervised: Dimensionality Reduction"
CV = "Computer Vision & Image Processing"

# Surrogate-model results say so in their names (they are not the model under validation).
CV_ROW = "5-Fold CV Average Score (SURROGATE RandomForest, not the model under validation)"
IMP_CHART = "Feature Importances (SURROGATE RandomForest, not the model under validation)"
ROBUST_ROWS = {"Robustness Baseline Score (SURROGATE, hold-out)", "Robustness Noisy Score (SURROGATE, hold-out)",
               "Performance Degradation (SURROGATE, hold-out)"}

# What each Test Category must produce when run alone.
#   rows   -> "Test" values it adds to the Execution_Metrics sheet
#   sheets -> Excel sheets it writes
#   charts -> Plotly chart titles; pngs -> number of static (matplotlib) figures
#   files  -> glob of extra files written to the output folder
EXPECTED: dict[tuple[str, str], dict] = {
    (CLF, "Performance Metrics"): dict(rows={"Classification Performance"},
                                       sheets={"PlotData_ROC", "PlotData_ConfusionMatrix"},
                                       charts={"ROC Curve", "Confusion Matrix"}),
    (CLF, "Class Imbalance Handling"): dict(sheets={"SMOTE_Imbalance"}),
    (CLF, "Ranking"): dict(rows={"Gini Rank Ordering"}),
    (CLF, "Statistical Diagnostics"): dict(sheets={"VIF_Diagnostics"}),
    (CLF, "Validation & Sampling"): dict(rows={CV_ROW}),
    (CLF, "Explainability"): dict(sheets={"PlotData_FeatureImportance"},
                                  charts={IMP_CHART}, pngs=1),
    (CLF, "Bias–Variance Analysis"): dict(sheets={"PlotData_LearningCurve"},
                                          charts={"Learning Curve (Surrogate)"}),
    (CLF, "Robustness & Sensitivity"): dict(rows=ROBUST_ROWS),
    (CLF, "Drift Detection"): dict(rows={"Population Stability Index (PSI)"}),

    (REG, "Performance Metrics"): dict(rows={"Regression Performance"},
                                       sheets={"PlotData_Residuals"}, charts={"Residuals"}),
    (REG, "Statistical Diagnostics"): dict(sheets={"VIF_Diagnostics"}),
    (REG, "Validation & Sampling"): dict(rows={CV_ROW}),
    (REG, "Explainability"): dict(sheets={"PlotData_FeatureImportance"},
                                  charts={IMP_CHART}, pngs=1),
    (REG, "Bias–Variance Analysis"): dict(sheets={"PlotData_LearningCurve"},
                                          charts={"Learning Curve (Surrogate)"}),
    (REG, "Robustness & Sensitivity"): dict(rows=ROBUST_ROWS),
    (REG, "Drift Detection"): dict(rows={"Population Stability Index (PSI)"}),

    (CLU, "Clustering Metrics"): dict(rows={"Clustering Metrics"}),
    (CLU, "Granularity"): dict(rows={"Herfindahl–Hirschman Index (HHI)"}),

    (DIM, "Dimensionality Reduction Metrics"): dict(sheets={"PlotData_PCA"},
                                                    charts={"PCA Explained Variance"}),
    (DIM, "Diagnostics"): dict(sheets={"PlotData_Correlation"}, charts={"Correlation Heatmap"}),

    (CV, "Data Quality Assessment"): dict(sheets={"CV_Quality_Metrics"}),
    (CV, "Validation & Robustness"): dict(files="noisy_*"),
}
ALWAYS_SHEETS = {"Test Dictionary"}          # written by every run
TABULAR_ALWAYS = {"Variable_Summary"}        # written by every tabular run


# ── synthetic data ─────────────────────────────────────────────────────────

def _clf_df(n=500, seed=1):
    rng = np.random.default_rng(seed)
    x1, x2, x3, x4 = (rng.normal(size=n) for _ in range(4))
    p = 1 / (1 + np.exp(-(2.5 * x1 + 0.5 * x2)))
    y = (rng.random(n) < p).astype(int)
    y[:20] = 1                                     # make sure both classes are well represented
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y,
                         "score": np.clip(p + rng.normal(0, 0.03, n), 0, 1),
                         "split": np.where(np.arange(n) < 0.7 * n, "train", "test")})


def _reg_df(n=500, seed=2):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = (rng.normal(size=n) for _ in range(3))
    y = 3 * x1 - 2 * x2 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y, "y_pred": y + rng.normal(0, 0.3, n),
                         "split": np.where(np.arange(n) < 0.7 * n, "dev", "oot")})


def _clu_df(n_per=120, seed=3):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [8, 8], [-8, 8]])
    pts = np.vstack([c + rng.normal(0, 0.6, (n_per, 2)) for c in centers])
    labels = np.repeat(["a", "b", "c"], n_per)
    return pd.DataFrame({"f1": pts[:, 0], "f2": pts[:, 1], "true_cluster": labels, "cluster": labels})


def _dim_df(n=300, seed=4):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return pd.DataFrame({"a": base, "b": 2 * base + rng.normal(0, 0.05, n), "c": rng.normal(size=n),
                         "id": np.arange(n), "score": rng.random(n)})


def _write_images(folder):
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    board = (np.indices((128, 128)).sum(axis=0) // 8 % 2 * 255).astype(np.uint8)
    sharp = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(folder / "sharp.png"), sharp)
    cv2.imwrite(str(folder / "blurry.png"), cv2.GaussianBlur(sharp, (31, 31), 10))
    (folder / "notes.txt").write_text("not an image")
    return folder


DATASETS = {CLF: (_clf_df, "y", "score", "split"), REG: (_reg_df, "y", "y_pred", "split"),
            CLU: (_clu_df, "true_cluster", "cluster", None), DIM: (_dim_df, "id", "score", None),
            CV: (_clf_df, "y", "score", None)}


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    base = tmp_path_factory.mktemp("builtin")
    out = {}
    for mt, (make, *_cols) in DATASETS.items():
        p = base / f"{mt.split(':')[-1].strip().split()[0].lower()}.csv"
        make().to_csv(p, index=False)
        out[mt] = load_path(p)[0]
    out["images"] = _write_images(base / "images")
    return out


def _run(sources, tmp_path, model_type, tests):
    _, obs, pred, split = DATASETS[model_type]
    image_dir = str(sources["images"]) if model_type == CV else None
    results = runner.run_model_tests(sources[model_type], model_type, obs, pred, split, tests,
                                     tmp_path, image_dir=image_dir, log=lambda m: None)
    return results[model_type]


def _titles(res):
    return {f.layout.title.text for f in res["plotly_figs"]}


def _rows(res):
    em = res["tables"].get("Execution_Metrics")
    return set(em["Test"]) if em is not None else set()


# ── Part 1: each built-in test, run on its own ─────────────────────────────

def test_every_catalog_entry_has_an_expectation():
    """Adding a test to the engine without adding it here must fail."""
    catalog = {(mt, cat) for mt, tests in runner.test_catalog().items() for cat, _ in tests}
    assert catalog == set(EXPECTED)


@pytest.mark.parametrize("model_type,test", sorted(EXPECTED), ids=lambda v: v)
def test_builtin_test_runs_alone(sources, tmp_path, model_type, test):
    exp = EXPECTED[(model_type, test)]
    res = _run(sources, tmp_path, model_type, [test])

    assert res["error"] is None, res["error"]
    sheets = set(res["tables"])
    always = ALWAYS_SHEETS | (set() if model_type == CV else TABULAR_ALWAYS)
    produced_sheets = sheets - always - {"Execution_Metrics"}

    # exactly this test's outputs — nothing from any other test
    assert _rows(res) == exp.get("rows", set())
    assert produced_sheets == exp.get("sheets", set())
    assert _titles(res) == exp.get("charts", set())
    assert len(res["mpl_pngs"]) == exp.get("pngs", 0)
    assert always <= sheets

    if "files" in exp:
        assert sorted(p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in res["extra_files"]) == \
               ["noisy_blurry.png", "noisy_sharp.png"]
    else:
        assert res["extra_files"] == []
    if model_type != CV:
        assert res["excel_path"] and res["html_path"]

    # metric values are real numbers, not NaN/empty
    if exp.get("rows"):
        em = res["tables"]["Execution_Metrics"].drop(columns="Test")
        assert em.notna().any(axis=1).all(), em


@pytest.mark.parametrize("model_type", [CLF, REG, CLU, DIM, CV])
def test_full_battery_is_the_union(sources, tmp_path, model_type):
    """Running every test of a model type at once produces all of their outputs."""
    res = _run(sources, tmp_path, model_type, None)
    assert res["error"] is None, res["error"]
    mine = [v for (mt, _), v in EXPECTED.items() if mt == model_type]
    assert _rows(res) == set().union(*(e.get("rows", set()) for e in mine))
    assert _titles(res) == set().union(*(e.get("charts", set()) for e in mine))
    assert set().union(*(e.get("sheets", set()) for e in mine)) <= set(res["tables"])


def test_engine_state_restored_after_run(sources, tmp_path):
    from ask.analysis import test_engine as te
    mp = te._import_main_pipeline()
    before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in mp.CONFIG.items()}
    hook = mp.export_plotly_to_html
    _run(sources, tmp_path, CLF, ["Ranking"])
    assert mp.CONFIG == before and mp.export_plotly_to_html == hook and mp.html_visualizations == []


def test_engine_state_restored_after_failure(sources, tmp_path, monkeypatch):
    from ask.analysis import test_engine as te
    mp = te._import_main_pipeline()
    before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in mp.CONFIG.items()}

    def explode(self):
        raise RuntimeError("boom")
    monkeypatch.setattr(type(mp), "run_pipeline", explode)
    res = _run(sources, tmp_path, CLF, ["Ranking"])
    assert res["error"] == "RuntimeError: boom"
    assert mp.CONFIG == before and mp.html_visualizations == []


def test_params_are_applied(sources, tmp_path):
    """clf_threshold flows through to the classification metrics."""
    src = sources[CLF]
    lo = runner.run_model_tests(src, CLF, "y", "score", None, ["Performance Metrics"], tmp_path / "lo",
                                params={"clf_threshold": 0.1}, log=lambda m: None)[CLF]
    hi = runner.run_model_tests(src, CLF, "y", "score", None, ["Performance Metrics"], tmp_path / "hi",
                                params={"clf_threshold": 0.9}, log=lambda m: None)[CLF]
    rec = lambda r: r["tables"]["Execution_Metrics"]["rec"].iloc[0]  # noqa: E731
    assert rec(lo) > rec(hi)


# ── Part 2: the numbers ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def stx():
    from ask.analysis import test_engine as te
    mp = te._import_main_pipeline()
    assert mp is not None, te._MAIN_PIPELINE_IMPORT_ERROR
    return inspect.getclosurevars(type(mp).run_pipeline).nonlocals["stx"]


class TestPerformanceMetrics:
    def test_classification_matches_sklearn(self, stx):
        from sklearn import metrics as M
        df = _clf_df()
        m, fpr, tpr, cm = stx.compute_classification_metrics(df["y"], df["score"], 0.5)
        yc = (df["score"] >= 0.5).astype(int)
        assert m["acc"] == pytest.approx(M.accuracy_score(df["y"], yc))
        assert m["prec"] == pytest.approx(M.precision_score(df["y"], yc))
        assert m["rec"] == pytest.approx(M.recall_score(df["y"], yc))
        assert m["f1"] == pytest.approx(M.f1_score(df["y"], yc))
        assert m["auc"] == pytest.approx(M.roc_auc_score(df["y"], df["score"]))
        assert m["pr_auc"] == pytest.approx(M.average_precision_score(df["y"], df["score"]))
        assert (cm == M.confusion_matrix(df["y"], yc)).all() and cm.sum() == len(df)
        assert fpr[0] == 0 and tpr[-1] == 1

    def test_classification_single_class_gives_nan_auc(self, stx):
        y = pd.Series([1, 1, 1, 1])
        m, fpr, tpr, _ = stx.compute_classification_metrics(y, pd.Series([0.2, 0.9, 0.6, 0.7]), 0.5)
        assert np.isnan(m["auc"]) and np.isnan(m["pr_auc"]) and fpr is None

    def test_regression_matches_sklearn(self, stx):
        from sklearn import metrics as M
        df = _reg_df()
        m = stx.compute_regression_metrics(df["y"], df["y_pred"])
        assert m["rmse"] == pytest.approx(np.sqrt(M.mean_squared_error(df["y"], df["y_pred"])))
        assert m["mae"] == pytest.approx(M.mean_absolute_error(df["y"], df["y_pred"]))
        assert m["r2"] == pytest.approx(M.r2_score(df["y"], df["y_pred"]))
        assert m["mad"] == pytest.approx(np.median(np.abs(df["y"] - np.median(df["y"]))))

    def test_regression_perfect_prediction(self, stx):
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        m = stx.compute_regression_metrics(y, y)
        assert m["rmse"] == 0 and m["mae"] == 0 and m["r2"] == 1 and m["mad"] == 1.0


class TestRanking:
    def test_gini_is_2auc_minus_1(self, stx):
        from sklearn.metrics import roc_auc_score
        df = _clf_df()
        assert stx.compute_gini(df["y"], df["score"]) == pytest.approx(2 * roc_auc_score(df["y"], df["score"]) - 1)

    def test_gini_extremes(self, stx):
        y = pd.Series([0, 0, 1, 1])
        assert stx.compute_gini(y, pd.Series([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
        assert stx.compute_gini(y, pd.Series([0.9, 0.8, 0.2, 0.1])) == pytest.approx(-1.0)
        assert np.isnan(stx.compute_gini(pd.Series([1, 1]), pd.Series([0.3, 0.4])))


class TestStatisticalDiagnostics:
    def test_independent_features_have_vif_near_one(self, stx):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["a", "b", "c"])
        assert stx.compute_vif(X)["VIF"].max() < 1.1

    def test_collinear_feature_flagged_and_sorted(self, stx):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=1000), rng.normal(size=1000)
        X = pd.DataFrame({"a": a, "b": b, "c": a + b + rng.normal(0, 0.01, 1000), "d": rng.normal(size=1000)})
        vif = stx.compute_vif(X)
        assert list(vif["VIF"]) == sorted(vif["VIF"], reverse=True)
        assert vif.set_index("Feature").loc["c", "VIF"] > 100
        assert vif.set_index("Feature").loc["d", "VIF"] < 1.1

    def test_matches_statsmodels(self, stx):
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = _reg_df()[["x1", "x2", "x3"]]
        got = stx.compute_vif(X).set_index("Feature")["VIF"]
        for i, c in enumerate(X.columns):
            assert got[c] == pytest.approx(variance_inflation_factor(X.values, i))


class TestClassImbalance:
    def test_smote_balances_classes(self, stx):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(220, 3)), columns=list("abc"))
        y = pd.Series([0] * 200 + [1] * 20)
        X_res, y_res = stx.apply_smote(X, y, 5)
        counts = pd.Series(y_res).value_counts()
        assert counts[0] == counts[1] == 200 and len(X_res) == 400
        pd.testing.assert_frame_equal(X_res.iloc[:220].reset_index(drop=True), X)   # originals kept

    def test_smote_handles_missing_values(self, stx):
        X = pd.DataFrame({"a": [np.nan] + list(range(29)), "b": range(30)}, dtype=float)
        y = pd.Series([0] * 24 + [1] * 6)
        _, y_res = stx.apply_smote(X, y, 3)
        assert pd.Series(y_res).value_counts().nunique() == 1


class TestValidationAndBiasVariance:
    def test_surrogate_type(self, stx):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        assert isinstance(stx.get_surrogate_model(CLF), RandomForestClassifier)
        assert isinstance(stx.get_surrogate_model(REG, depth=3), RandomForestRegressor)
        assert stx.get_surrogate_model(REG, depth=3).max_depth == 3

    def test_cross_validation_reasonable_and_deterministic(self, stx):
        df = _clf_df()
        X = df[["x1", "x2", "x3", "x4"]]
        mean, std = stx.compute_cross_validation(X, df["y"], CLF)
        assert 0.7 < mean <= 1.0 and 0 <= std < 0.1
        assert stx.compute_cross_validation(X, df["y"], CLF) == (mean, std)

    def test_cross_validation_regression_r2(self, stx):
        df = _reg_df()
        mean, _ = stx.compute_cross_validation(df[["x1", "x2", "x3"]], df["y"], REG)
        assert mean > 0.8                                         # R² of a well-specified surrogate

    def test_learning_curve_shapes(self, stx):
        df = _clf_df()
        sizes, tr, te_ = stx.compute_learning_curve(df[["x1", "x2"]], df["y"], CLF)
        assert len(sizes) == len(tr) == len(te_) == 5
        assert list(sizes) == sorted(sizes) and ((0 <= tr) & (tr <= 1)).all()
        assert (tr >= te_ - 0.05).all()                           # train score is not below test score


class TestExplainability:
    def test_binary_classification(self, stx):
        df = _clf_df()
        X = df[["x1", "x2", "x3", "x4"]]
        imp, shap_m, basis = stx.compute_explainability(X, df["y"], CLF, 5)
        assert basis == "positive class" and shap_m.shape == X.shape
        assert imp["Importance"].sum() == pytest.approx(1.0)
        assert imp.iloc[-1]["Feature"] == "x1"                   # the true driver ranks first
        assert np.abs(shap_m).mean(axis=0).argmax() == 0          # SHAP agrees

    def test_two_features_two_classes_stays_2d(self, stx):
        """The engine documents a SHAP shape trap when n_features == n_classes."""
        df = _clf_df()
        _, shap_m, basis = stx.compute_explainability(df[["x1", "x2"]], df["y"], CLF, 3)
        assert shap_m.shape == (len(df), 2) and basis == "positive class"

    def test_multiclass(self, stx):
        df = _clf_df()
        y = pd.cut(df["x1"], 3, labels=[0, 1, 2]).astype(int)
        _, shap_m, basis = stx.compute_explainability(df[["x1", "x2", "x3"]], y, CLF, 3)
        assert basis == "mean across 3 classes" and shap_m.shape == (len(df), 3) and (shap_m >= 0).all()

    def test_regression(self, stx):
        df = _reg_df()
        imp, shap_m, basis = stx.compute_explainability(df[["x1", "x2", "x3"]], df["y"], REG, 5)
        assert basis == "" and shap_m.shape == (len(df), 3)
        assert set(imp.tail(2)["Feature"]) == {"x1", "x2"}       # x3 is pure noise

    def test_top_15_features_only(self, stx):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(200, 20)), columns=[f"f{i}" for i in range(20)])
        imp, shap_m, _ = stx.compute_explainability(X, (X["f0"] > 0).astype(int), CLF, 3)
        assert len(imp) == 15 and shap_m.shape == (200, 20)


class TestRobustness:
    def test_scores(self, stx):
        df = _reg_df()
        base, noisy = stx.compute_robustness(df[["x1", "x2", "x3"]], df["y"], REG)
        assert 0.8 < base <= 1.0 and noisy <= base + 0.02

    def test_classification_scores_are_accuracies(self, stx):
        df = _clf_df()
        base, noisy = stx.compute_robustness(df[["x1", "x2"]], df["y"], CLF)
        assert 0 <= noisy <= 1 and 0.7 < base <= 1


class TestDrift:
    def test_identical_distributions(self, stx):
        s = pd.Series(np.random.default_rng(0).normal(size=5000))
        assert stx.calculate_psi(s, s.copy()) < 1e-3

    def test_shift_increases_psi(self, stx):
        rng = np.random.default_rng(0)
        train = pd.Series(rng.normal(size=5000))
        small = stx.calculate_psi(train, pd.Series(rng.normal(0.1, 1, 5000)))
        large = stx.calculate_psi(train, pd.Series(rng.normal(1.0, 1, 5000)))
        assert small < 0.1 < 0.25 < large

    def test_matches_hand_computation(self, stx):
        train = pd.Series(np.arange(100, dtype=float))           # deciles of 10 values each
        test = pd.Series(np.arange(10, dtype=float))             # everything in the first decile
        e = np.full(10, 0.1) + 1e-4
        a = np.array([1.0] + [0.0] * 9) + 1e-4
        e, a = e / e.sum(), a / a.sum()
        expected = round(float(np.sum((e - a) * np.log(e / a))), 4)
        assert stx.calculate_psi(train, test, bins=10) == pytest.approx(expected)

    def test_empty_and_nan(self, stx):
        assert np.isnan(stx.calculate_psi(pd.Series([], dtype=float), pd.Series([1.0])))
        assert np.isnan(stx.calculate_psi(pd.Series([np.nan, np.nan]), pd.Series([1.0])))


class TestClustering:
    def test_metrics_match_sklearn(self, stx):
        from sklearn import metrics as M
        df = _clu_df()
        X, labels = df[["f1", "f2"]], df["cluster"]
        m = stx.compute_clustering_metrics(X, labels)
        codes = pd.factorize(labels.astype(str), sort=True)[0]
        assert m["sil"] == pytest.approx(M.silhouette_score(X, codes))
        assert m["dbi"] == pytest.approx(M.davies_bouldin_score(X, codes))
        assert m["ch"] == pytest.approx(M.calinski_harabasz_score(X, codes))
        assert m["sil"] > 0.8 and m["dbi"] < 0.3 and m["wcss"] > 0

    def test_bad_labels_score_worse(self, stx):
        df = _clu_df()
        good = stx.compute_clustering_metrics(df[["f1", "f2"]], df["cluster"])
        shuffled = df["cluster"].sample(frac=1, random_state=0).reset_index(drop=True)
        bad = stx.compute_clustering_metrics(df[["f1", "f2"]], shuffled)
        assert bad["sil"] < good["sil"] and bad["dbi"] > good["dbi"]

    @pytest.mark.parametrize("labels,expected", [
        (["a"] * 10, 1.0),
        (["a", "b"] * 5, 0.5),
        (["a", "b", "c", "d"] * 3, 0.25),
        (["a"] * 9 + ["b"], 0.82),
    ])
    def test_hhi(self, stx, labels, expected):
        assert stx.compute_hhi(labels) == pytest.approx(expected)


class TestDimensionalityReduction:
    def test_pca_variance(self, stx):
        df = _dim_df()
        evr = stx.compute_pca_variance(df[["a", "b", "c"]])
        assert evr.sum() == pytest.approx(1.0) and list(evr) == sorted(evr, reverse=True)
        assert evr[0] > 0.7                                        # a and b are almost the same signal

    def test_pca_matches_sklearn(self, stx):
        from sklearn.decomposition import PCA
        X = _dim_df()[["a", "b", "c"]]
        np.testing.assert_allclose(stx.compute_pca_variance(X), PCA().fit(X).explained_variance_ratio_)

    def test_correlation(self, stx):
        X = _dim_df()[["a", "b", "c"]]
        corr = stx.compute_correlation(X)
        pd.testing.assert_frame_equal(corr, X.corr())
        assert corr.loc["a", "b"] > 0.99 and abs(corr.loc["a", "c"]) < 0.2


class TestComputerVision:
    @pytest.fixture
    def images(self, tmp_path):
        import cv2
        d = _write_images(tmp_path / "imgs")
        return cv2.imread(str(d / "sharp.png")), cv2.imread(str(d / "blurry.png"))

    def test_blur_detection(self, stx, images):
        sharp, blurry = images
        lv_s, ed_s, status_s = stx.analyze_image_quality(sharp, 100.0)
        lv_b, ed_b, status_b = stx.analyze_image_quality(blurry, 100.0)
        assert lv_s > lv_b and ed_s > ed_b
        assert 0 <= ed_b <= ed_s <= 1
        assert status_s.endswith("Sharp") and status_b.endswith("Blurry")

    def test_threshold_controls_status(self, stx, images):
        sharp, _ = images
        lv, _, _ = stx.analyze_image_quality(sharp, 100.0)
        assert stx.analyze_image_quality(sharp, lv + 1)[2].endswith("Blurry")

    def test_noisy_image(self, stx, images):
        sharp, _ = images
        noisy = stx.generate_noisy_image(sharp)
        assert noisy.shape == sharp.shape and noisy.dtype == np.uint8 and (noisy != sharp).any()

    def test_quality_sheet_values(self, sources, tmp_path):
        res = _run(sources, tmp_path, CV, ["Data Quality Assessment"])
        sheet = res["tables"]["CV_Quality_Metrics"].set_index("Filename")
        assert set(sheet.index) == {"sharp.png", "blurry.png"}               # notes.txt ignored
        assert sheet.loc["sharp.png", "Laplacian Variance"] > sheet.loc["blurry.png", "Laplacian Variance"]

    def test_missing_image_dir_is_skipped_cleanly(self, sources, tmp_path):
        res = runner.run_model_tests(sources[CV], CV, "y", "score", None, None, tmp_path,
                                     image_dir=str(tmp_path / "no_images"), log=lambda m: None)[CV]
        assert res["error"] is None and "CV_Quality_Metrics" not in res["tables"]


class TestProfiling:
    def test_variable_summary(self, stx):
        df = pd.DataFrame({"n": [1.0, None, 3.0, 4.0], "c": ["a", "b", None, None]})
        s = stx.get_variable_summary(df).set_index("Variable")
        assert s.loc["n", "% Missing"] == 25.0 and s.loc["c", "% Missing"] == 50.0
        assert s.loc["n", "Type"] == "Numeric" and s.loc["c", "Type"] == "Categorical"
        assert s.loc["n", "Non-missing Count"] == 3
