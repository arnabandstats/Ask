"""Deterministic Test Lab engine.

Copied VERBATIM from the original single-file app (lines 13811-14719 and
14804-14812): the former stats_toolbox.py + main_pipeline.py and the
run_deterministic_tests() wrapper. Do not edit the test logic here; call it.

Heavy dependencies (cv2, shap, sklearn, statsmodels, imblearn, plotly) are
imported lazily inside _import_main_pipeline(), so importing this module is cheap.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


TESTCAT_ALL_OPTION = "All tests"

_MAIN_PIPELINE_IMPORT_ERROR: Optional[str] = None


def _parse_test_categories(mp, model_categories: list[str]) -> list[str]:
    """Ordered, de-duplicated Test Category names (the part before ':' in
    each TEST_MAPPINGS entry) for the given model categories — the same
    parse run_pipeline uses to build the Test Dictionary sheet."""
    out: list[str] = []
    for mc in model_categories:
        for entry in mp.TEST_MAPPINGS.get(mc, []):
            cat = entry.split(":", 1)[0].strip() if ":" in entry else entry.strip()
            if cat and cat not in out:
                out.append(cat)
    return out


# ── Former stats_toolbox.py + main_pipeline.py, now embedded below ──
#
# WHY A LAZILY-BUILT NAMESPACE CLASS, NOT PLAIN MODULE-LEVEL FUNCTIONS:
#
#   1. Heavy, Test-Lab-only dependencies (cv2, shap, xlsxwriter,
#      statsmodels, imbalanced-learn, scikit-learn, plotly) must stay OUT
#      of the app module's top-level imports — Repository/Document
#      Intelligence and "Chat with data" all have to keep working even if
#      those packages aren't installed. The two-file design got this for
#      free because `import main_pipeline` (and, through it,
#      `import stats_toolbox`) only ran the first time the Test Lab tab
#      was opened. Building the classes below INSIDE the lazy loader
#      function `_import_main_pipeline()` reproduces that exactly: the
#      `import cv2`/`import shap`/… lines only execute on first call, and
#      the result is cached in `_TEST_LAB_NS` afterward (mirroring
#      Python's own sys.modules cache for a real `import`).
#
#   2. run_deterministic_tests() (below, Phase 5B) needs to temporarily
#      MONKEYPATCH export_plotly_to_html / export_matplotlib_to_html and
#      mutate-then-restore a shared CONFIG dict, once per test run — the
#      same `mp.attribute = …` contract it used against a real imported
#      `main_pipeline` module object. A class instance satisfies that
#      contract identically: assigning to an instance attribute shadows
#      the class's method WITHOUT auto-binding `self` (exactly like
#      reassigning a plain module-level function), and restoring the
#      original bound method afterward puts it back exactly as it was.
#      See run_deterministic_tests()'s docstring for the patch/restore
#      code itself.
#
#   3. Nesting each former file's functions inside its own class (instead
#      of dumping them at the app module's own top level) avoids name
#      collisions — e.g. stats_toolbox.py has its own `extract_python_code`,
#      unrelated to the app module's own pandas-agent helper of the same name.
#
# Every method body below is a verbatim, line-for-line port of the
# original module function of the same name; only the call surface
# changed (`stx.foo(...)` stays a call on an instance instead of a module,
# `CONFIG`/`TEST_MAPPINGS` became `self.CONFIG`/`self.TEST_MAPPINGS`, and
# bare `export_plotly_to_html(...)` calls became `self.export_plotly_to_html(...)`
# so the monkeypatch in point 2 above still takes effect).

_TEST_LAB_NS: Any = None   # built once and cached — see point 1 above


def _import_main_pipeline():
    """Lazily build (and cache) the Deterministic Test Lab engine — a
    verbatim port of the former main_pipeline.py + stats_toolbox.py.
    Returns the namespace object (kept under the name `mp` at call sites,
    matching the original `import main_pipeline as mp`), or None with the
    failure recorded in _MAIN_PIPELINE_IMPORT_ERROR."""
    global _TEST_LAB_NS, _MAIN_PIPELINE_IMPORT_ERROR
    if _TEST_LAB_NS is not None:
        return _TEST_LAB_NS
    try:
        # ---- heavy, Test-Lab-only imports (deferred on purpose — see
        #      the module docstring above) ----
        import base64
        import cv2
        import shap
        import xlsxwriter  # noqa: F401 — engine name used by pd.ExcelWriter below
        import plotly.express as px
        import plotly.graph_objects as go
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
            roc_curve, average_precision_score, confusion_matrix,
            accuracy_score, precision_score, recall_score, f1_score,
            silhouette_score, davies_bouldin_score, calinski_harabasz_score,
        )
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import learning_curve, cross_val_score
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import LabelEncoder
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from imblearn.over_sampling import SMOTE

        # ════════════════════════════════════════════════════════════
        # Former stats_toolbox.py — one method per statistical or
        # computer-vision metric/diagnostic the Test Lab can run.
        # Every method is stateless (pure function of its arguments);
        # @staticmethod/@classmethod are used purely for namespacing.
        # ════════════════════════════════════════════════════════════
        class _StatsToolbox:
            """The Deterministic Test Lab's metric/diagnostic library —
            each method mirrors one line of a TEST_MAPPINGS entry."""

            @staticmethod
            def extract_python_code(text):
                """Pull a fenced ```python``` (or bare ```) code block out
                of an LLM response. Unused by run_pipeline() itself — kept
                for parity with the original toolbox module, which also
                never called it internally."""
                match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
                if match:
                    return match.group(1).strip()
                match_generic = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
                if match_generic:
                    return match_generic.group(1).strip()
                return text.strip()

            @staticmethod
            def calculate_psi(train_series, test_series, bins=10):
                """Population Stability Index between two numeric series,
                via quantile binning on the train series. Used by the
                Drift Detection test."""
                train_series, test_series = train_series.dropna(), test_series.dropna()
                if len(train_series) == 0 or len(test_series) == 0:
                    return np.nan
                try:
                    _, bin_edges = pd.qcut(train_series, q=bins, retbins=True, duplicates='drop')
                    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
                    train_dist = pd.cut(train_series, bins=bin_edges).value_counts(normalize=True).sort_index() + 1e-4
                    test_dist = pd.cut(test_series, bins=bin_edges).value_counts(normalize=True).sort_index() + 1e-4
                    train_dist /= train_dist.sum()
                    test_dist /= test_dist.sum()
                    return round(np.sum((train_dist - test_dist) * np.log(train_dist / test_dist)), 4)
                except Exception:
                    return np.nan

            @staticmethod
            def analyze_image_quality(img_array, blur_thresh):
                """Laplacian-variance blur check + Canny edge density for
                one image. Used by the Computer Vision "Data Quality
                Assessment" test."""
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                status = "🟢 Sharp" if laplacian_var >= blur_thresh else "🔴 Blurry"
                return laplacian_var, edge_density, status

            @staticmethod
            def generate_noisy_image(img_array):
                """Gaussian-noise-perturbed copy of an image. Used by the
                Computer Vision "Validation & Robustness" test."""
                noise = np.random.normal(0, 25, img_array.shape).astype(np.uint8)
                return cv2.add(img_array, noise)

            @staticmethod
            def get_variable_summary(df):
                """Per-column missing %, non-missing count, and inferred
                type (Numeric/Categorical) — populates the Variable_Summary
                sheet that every run writes regardless of MODEL_TYPE."""
                all_num_cols = df.select_dtypes(include=np.number).columns.tolist()
                return pd.DataFrame({
                    "Variable": df.columns,
                    "% Missing": (df.isnull().mean() * 100).round(2),
                    "Non-missing Count": df.notnull().sum(),
                    "Type": ["Numeric" if c in all_num_cols else "Categorical" for c in df.columns]
                })

            @staticmethod
            def compute_classification_metrics(y_true, y_pred, thresh):
                """Accuracy/precision/recall/F1/AUC/PR-AUC at a probability
                threshold, plus ROC points and the confusion matrix. Used
                by the classification "Performance Metrics" test."""
                y_pred_class = (y_pred >= thresh).astype(int)
                metrics = {
                    "acc": accuracy_score(y_true, y_pred_class),
                    "prec": precision_score(y_true, y_pred_class, zero_division=0),
                    "rec": recall_score(y_true, y_pred_class, zero_division=0),
                    "f1": f1_score(y_true, y_pred_class, zero_division=0),
                    "auc": roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan,
                    "pr_auc": average_precision_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan
                }
                fpr, tpr, _ = roc_curve(y_true, y_pred) if len(np.unique(y_true)) == 2 else (None, None, None)
                cm = confusion_matrix(y_true, y_pred_class)
                return metrics, fpr, tpr, cm

            @staticmethod
            def compute_regression_metrics(y_true, y_pred):
                """RMSE/MAE/R²/MAD for a regression's observed vs.
                predicted. Used by the regression "Performance Metrics"
                test."""
                return {
                    "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                    "mad": np.median(np.abs(y_true - np.median(y_true)))
                }

            @staticmethod
            def compute_gini(y_true, y_pred):
                """Gini coefficient (2*AUC - 1) for binary classification.
                Used by the "Ranking" test."""
                if len(np.unique(y_true)) == 2:
                    return 2 * roc_auc_score(y_true, y_pred) - 1
                return np.nan

            @staticmethod
            def compute_vif(X):
                """Variance Inflation Factor per feature (multicollinearity
                check). Used by the "Statistical Diagnostics" test."""
                vif_data = pd.DataFrame({"Feature": X.columns})
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
                return vif_data.sort_values("VIF", ascending=False)

            @staticmethod
            def apply_smote(X, y_true, neighbors):
                """SMOTE-resampled (X, y) for class-imbalance handling.
                Used by the "Class Imbalance Handling" test."""
                sm = SMOTE(k_neighbors=neighbors, random_state=42)
                return sm.fit_resample(X.fillna(0), y_true)

            @staticmethod
            def get_surrogate_model(model_type, depth=5):
                """RandomForest surrogate matching the model type — the
                shared model behind cross-validation, learning-curve,
                explainability and robustness tests below."""
                if "Classification" in model_type:
                    return RandomForestClassifier(max_depth=depth, random_state=42)
                return RandomForestRegressor(max_depth=depth, random_state=42)

            @classmethod
            def compute_cross_validation(cls, X, y_true, model_type):
                """Mean/std of 5-fold CV score for the surrogate model.
                Used by the "Validation & Sampling" test."""
                model = cls.get_surrogate_model(model_type)
                scores = cross_val_score(model, X.fillna(0), y_true, cv=5)
                return scores.mean(), scores.std()

            @classmethod
            def compute_learning_curve(cls, X, y_true, model_type):
                """Train/test learning-curve scores across training sizes
                (capped at 3,000 rows for speed). Used by the
                "Bias–Variance Analysis" test."""
                model = cls.get_surrogate_model(model_type)
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X.fillna(0)[:3000], y_true[:3000], cv=3)
                return train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)

            @classmethod
            def compute_explainability(cls, X, y_true, model_type, depth):
                """Feature importances + a 2-D SHAP matrix from a fitted
                surrogate model. Used by the "Explainability" test.

                Returns (importances, shap_matrix, basis) where shap_matrix is
                ALWAYS (n_samples, n_features) and `basis` says what the class
                axis was collapsed to — empty for a regressor, which has no
                class axis and so needs no qualifier on the chart.

                That collapse is the whole point of this method. TreeExplainer
                returns a different shape per task, and since shap 0.45 a
                CLASSIFIER returns one 3-D ndarray of
                (n_samples, n_features, n_classes) instead of the old list of
                per-class arrays — so the long-standing
                `shap_values[1] if isinstance(shap_values, list)` check silently
                stopped firing and passed the raw 3-D array to summary_plot.
                shap then has to guess what a 3-D array means, and it guesses
                by shape: when n_features == n_classes (two features, two
                classes) it cannot tell the class axis from a second feature
                axis, concludes it was handed INTERACTION values, and draws a
                feature-by-feature interaction grid whose axis reads "SHAP
                interaction value". Every number in that grid is a class score
                plotted as if it were an interaction — the chart was not ugly,
                it was meaningless. Collapsing here means only a 2-D matrix can
                ever reach a plot, so shap has nothing left to guess."""
                model = cls.get_surrogate_model(model_type, depth)
                X_clean = X.fillna(0)
                model.fit(X_clean, y_true)
                imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}) \
                        .sort_values("Importance", ascending=True).tail(15)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_clean)

                # Legacy list-of-arrays (shap < 0.45) and the modern 3-D
                # ndarray are the same information in two containers, so both
                # collapse the same way: keep the positive class for a binary
                # task — signed, and what "the model's SHAP values" means for a
                # binary classifier — and average magnitudes for multiclass,
                # where no single class is the answer and the bar chart is
                # reading magnitudes anyway.
                basis = ""
                if isinstance(shap_values, list):
                    n_classes = len(shap_values)
                    if n_classes == 2:
                        values = np.asarray(shap_values[1])
                        basis = "positive class"
                    else:
                        values = np.mean([np.abs(np.asarray(v)) for v in shap_values], axis=0)
                        basis = f"mean across {n_classes} classes"
                else:
                    values = np.asarray(shap_values)
                    if values.ndim == 3:
                        n_classes = values.shape[2]
                        if n_classes == 2:
                            values = values[:, :, 1]
                            basis = "positive class"
                        else:
                            values = np.mean(np.abs(values), axis=2)
                            basis = f"mean across {n_classes} classes"

                # A regressor is already 2-D; anything else here is a shape
                # this code has not been taught, and a wrong chart is worse
                # than a missing one.
                if values.ndim != 2 or values.shape[1] != X_clean.shape[1]:
                    raise ValueError(
                        f"Unexpected SHAP value shape {values.shape} for "
                        f"{X_clean.shape[1]} feature(s) — refusing to plot it.")
                return imp, values, basis

            @classmethod
            def compute_robustness(cls, X, y_true, model_type):
                """Baseline vs. Gaussian-noise-perturbed model score. Used
                by the "Robustness & Sensitivity" test."""
                X_clean = X.fillna(0)
                noise = np.random.normal(0, 0.05, X_clean.shape)
                X_noisy = X_clean + noise * X_clean.std().values

                model = cls.get_surrogate_model(model_type)
                model.fit(X_clean, y_true)
                baseline_score = model.score(X_clean, y_true)
                noisy_score = model.score(X_noisy, y_true)
                return baseline_score, noisy_score

            @staticmethod
            def compute_clustering_metrics(X, y_pred):
                """Silhouette / Davies–Bouldin / Calinski–Harabasz / WCSS
                for the predicted cluster labels. Used by the "Clustering
                Metrics" test."""
                y_pred_encoded = LabelEncoder().fit_transform(y_pred.astype(str))
                X_clean = X.fillna(0)
                kmeans = KMeans(n_clusters=len(np.unique(y_pred_encoded)), random_state=42).fit(X_clean)
                return {
                    "sil": silhouette_score(X_clean, y_pred_encoded),
                    "dbi": davies_bouldin_score(X_clean, y_pred_encoded),
                    "ch": calinski_harabasz_score(X_clean, y_pred_encoded),
                    "wcss": kmeans.inertia_
                }

            @staticmethod
            def compute_hhi(y_pred):
                """Herfindahl–Hirschman Index of predicted-label
                concentration. Used by the "Granularity" test."""
                return (pd.Series(y_pred).value_counts(normalize=True) ** 2).sum()

            @staticmethod
            def compute_pca_variance(X):
                """Explained-variance ratio per PCA component. Used by the
                "Dimensionality Reduction Metrics" test."""
                pca = PCA().fit(X.fillna(0))
                return pca.explained_variance_ratio_

            @staticmethod
            def compute_correlation(X):
                """Feature correlation matrix. Used by the "Diagnostics"
                test."""
                return X.fillna(0).corr()

        stx = _StatsToolbox()

        # ════════════════════════════════════════════════════════════
        # Former main_pipeline.py — CONFIG, the TEST_MAPPINGS glossary,
        # the two HTML-export hooks, and run_pipeline() itself. Kept as
        # instance state/methods (not plain module globals) so
        # run_deterministic_tests() can monkeypatch/restore it exactly as
        # it did against a real imported module — see the block comment
        # above this function for why.
        # ════════════════════════════════════════════════════════════
        class _MainPipeline:
            """One Deterministic Test Lab run: CONFIG in, an Excel report
            + interactive-visualizations HTML file out. Every public
            method mirrors main_pipeline.py's module-level function of the
            same name."""

            def __init__(self):
                # ---- Configuration (headless inputs; the Test Lab UI
                #      overwrites these per run — see run_deterministic_tests) ----
                self.CONFIG = {
                    # I/O Settings
                    "DATA_PATH": "Data/Data - num.xlsx",   # Path to your tabular data
                    "IMAGE_DIR": "sample_images/",         # Path to folder if running Computer Vision tests
                    "OUTPUT_DIR": "Output",

                    # Model & Data Definition
                    "MODEL_TYPE": "Supervised: Regression",  # Choose from the TEST_MAPPINGS keys
                    "OBSERVED_COL": "Y",
                    "PREDICTED_COL": "Y_pred",
                    "SPLIT_COL": "Test_Train_Flag",        # Set to None if not applicable

                    # Test Hyperparameters
                    "PARAMS": {
                        "clf_threshold": 0.5,
                        "surrogate_depth": 5,
                        "smote_neighbors": 5,
                        "psi_bins": 10,
                        "blur_threshold": 100.0
                    },

                    # Test-category filter: None runs the full battery for MODEL_TYPE.
                    # Or provide a set/list of "Test Category" names — the part before
                    # the ":" in each TEST_MAPPINGS entry, e.g.
                    # {"Explainability", "Drift Detection"} — to run only those tests.
                    # Profiling (Variable_Summary) and the Test Dictionary sheet
                    # always run.
                    "SELECTED_TESTS": None
                }

                # Glossary mapping for the Test Dictionary output
                self.TEST_MAPPINGS = {
                    "Supervised: Classification": [
                        "Performance Metrics: Accuracy, Precision, Recall / Sensitivity, F1-score, ROC Curve, AUC, PR-AUC, Gini coefficient, Confusion Matrix.",
                        "Class Imbalance Handling: Confusion matrix analysis, Class weights, SMOTE, Threshold tuning.",
                        "Ranking: Gini Rank Ordering.",
                        "Statistical Diagnostics: Variance Inflation Factor (VIF).",
                        "Validation & Sampling: Cross-Validation (k-fold).",
                        "Explainability: SHAP, Feature Importance.",
                        "Bias–Variance Analysis: Learning curves.",
                        "Robustness & Sensitivity: Random noise perturbation.",
                        "Drift Detection: Population Stability Index (PSI)."
                    ],
                    "Supervised: Regression": [
                        "Performance Metrics: RMSE, MAE, R², MAD.",
                        "Statistical Diagnostics: Variance Inflation Factor (VIF).",
                        "Validation & Sampling: Cross-Validation (k-fold).",
                        "Explainability: SHAP, Feature Importance.",
                        "Bias–Variance Analysis: Learning curves.",
                        "Robustness & Sensitivity: Random noise perturbation.",
                        "Drift Detection: Population Stability Index (PSI)."
                    ],
                    "Unsupervised: Clustering": [
                        "Clustering Metrics: Silhouette Score, Davies–Bouldin Index (DBI), Calinski–Harabasz Index (CH score), WCSS / Inertia.",
                        "Granularity: Herfindahl–Hirschman Index (HHI)."
                    ],
                    "Unsupervised: Dimensionality Reduction": [
                        "Dimensionality Reduction Metrics: Explained variance (PCA).",
                        "Diagnostics: Correlation clustering."
                    ],
                    "Computer Vision & Image Processing": [
                        "Data Quality Assessment: Laplacian variance measures, Edge-density methods.",
                        "Validation & Robustness: Random noise perturbation."
                    ]
                }

                # HTML payload accumulated by the two export_* hooks below,
                # written out as one file at the end of run_pipeline().
                self.html_visualizations = []

                # Theme setup (a module-level side effect in the original
                # main_pipeline.py; applied once per _MainPipeline instance
                # here instead — same "runs exactly once" behavior thanks
                # to the _TEST_LAB_NS cache in _import_main_pipeline()).
                self.COLOR_SEQ = ["#2C3696", "#F36717", "#F9B248"]
                plt.rcParams.update({
                    "axes.facecolor": "#FFFFFF", "figure.facecolor": "#FFFFFF", "text.color": "#000000",
                    "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
                    "axes.prop_cycle": plt.cycler('color', self.COLOR_SEQ)
                })

            def _test_enabled(self, category):
                """True if this Test Category should run — CONFIG["SELECTED_TESTS"]
                is None for the full battery, or a collection of Test Category names."""
                selected = self.CONFIG.get("SELECTED_TESTS")
                return selected is None or category in selected

            def apply_consulting_theme(self, fig):
                """Shared Plotly layout/axis styling applied to every chart
                the pipeline exports."""
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Helvetica Neue, Inter, sans-serif", color="#000000"),
                    title_font=dict(size=18, color="#000000"), margin=dict(t=50, l=40, r=20, b=40),
                    colorway=self.COLOR_SEQ, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_xaxes(showline=True, linewidth=1, linecolor='#CCCCCC', gridcolor='#EEEEEE', zeroline=False)
                fig.update_yaxes(showline=True, linewidth=1, linecolor='#CCCCCC', gridcolor='#EEEEEE', zeroline=False)
                return fig

            def export_plotly_to_html(self, fig):
                """Appends a Plotly figure to the HTML payload.
                NOTE — MONKEYPATCH SEAM: run_deterministic_tests() (Phase
                5B, below) temporarily replaces this exact instance
                attribute so it can also capture the figure for inline UI
                display before calling through to this original. See that
                function's docstring for the full patch/restore contract."""
                self.html_visualizations.append(
                    self.apply_consulting_theme(fig).to_html(full_html=False, include_plotlyjs='cdn'))

            def export_matplotlib_to_html(self, fig):
                """Encodes a Matplotlib figure to base64 and appends to the
                HTML payload. Same monkeypatch seam as export_plotly_to_html
                above."""
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
                buf.seek(0)
                b64_string = base64.b64encode(buf.read()).decode('utf-8')
                img_tag = f'<div style="margin: 20px 0;"><img src="data:image/png;base64,{b64_string}" /></div>'
                self.html_visualizations.append(img_tag)

            def save_dataframe(self, writer, df, sheet_name):
                """Saves a dataframe to Excel, handling naming collisions
                cleanly (Excel sheet names cap at 31 characters)."""
                name = sheet_name[:31]   # Excel limit
                if not df.empty:
                    df.to_excel(writer, sheet_name=name, index=False)

            def run_pipeline(self):
                """The Test Lab's actual model-validation run: writes the
                Test Dictionary sheet, routes to the Computer Vision or
                tabular pipeline per CONFIG["MODEL_TYPE"], runs every
                applicable _StatsToolbox test for that category, and saves
                the Excel report + interactive-visualizations HTML file.
                Verbatim port of main_pipeline.run_pipeline()."""
                os.makedirs(self.CONFIG["OUTPUT_DIR"], exist_ok=True)
                excel_path = os.path.join(self.CONFIG["OUTPUT_DIR"], "Model_Evaluation_Report.xlsx")
                html_path = os.path.join(self.CONFIG["OUTPUT_DIR"], "Interactive_Visualizations.html")

                writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")

                # 1. Generate Test Dictionary
                test_dict_data = []
                for m_type, tests in self.TEST_MAPPINGS.items():
                    for t in tests:
                        cat, desc = t.split(":", 1) if ":" in t else (t, "")
                        test_dict_data.append({"Model Category": m_type, "Test Category": cat.strip(), "Description/Metrics": desc.strip()})

                self.save_dataframe(writer, pd.DataFrame(test_dict_data), "Test Dictionary")
                print("[OK] Initialized Output Engine and Test Dictionary.")

                # 2. Computer Vision Pipeline Route
                if "Computer Vision" in self.CONFIG["MODEL_TYPE"]:
                    print("[CV] Running Computer Vision Pipeline...")
                    cv_results = []
                    if not os.path.exists(self.CONFIG["IMAGE_DIR"]):
                        print(f"Directory {self.CONFIG['IMAGE_DIR']} not found. Skipping CV tests.")
                        return

                    for filename in os.listdir(self.CONFIG["IMAGE_DIR"]):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(self.CONFIG["IMAGE_DIR"], filename)
                            img = cv2.imread(img_path)

                            # Quality Assessment
                            if self._test_enabled("Data Quality Assessment"):
                                lap_var, edge_den, status = stx.analyze_image_quality(img, self.CONFIG["PARAMS"]["blur_threshold"])
                                cv_results.append({
                                    "Filename": filename, "Laplacian Variance": lap_var,
                                    "Edge Density": edge_den, "Blur Status": status
                                })

                            # Robustness (Save an example of noisy image)
                            if self._test_enabled("Validation & Robustness"):
                                noisy = stx.generate_noisy_image(img)
                                cv2.imwrite(os.path.join(self.CONFIG["OUTPUT_DIR"], f"noisy_{filename}"), noisy)

                    self.save_dataframe(writer, pd.DataFrame(cv_results), "CV_Quality_Metrics")
                    writer.close()
                    print(f"[OK] Pipeline complete. Check {self.CONFIG['OUTPUT_DIR']} for results.")
                    return

                # 3. Tabular Data Pipeline Route
                if not os.path.exists(self.CONFIG["DATA_PATH"]):
                    print(f"[ERROR] Data file {self.CONFIG['DATA_PATH']} not found.")
                    return

                print("[INFO] Loading Data and Running Tabular Pipeline...")
                df = pd.read_excel(self.CONFIG["DATA_PATH"])
                obs_col = self.CONFIG["OBSERVED_COL"]
                pred_col = self.CONFIG["PREDICTED_COL"]
                split_col = self.CONFIG["SPLIT_COL"]

                perf_df = df.dropna(subset=[obs_col, pred_col]).copy()
                y_true, y_pred = perf_df[obs_col], perf_df[pred_col]
                num_cols = perf_df.select_dtypes(include=np.number).columns.tolist()
                X = perf_df[num_cols].drop(columns=[obs_col, pred_col, split_col], errors='ignore')

                # Base Metrics & Profiling
                var_summary = stx.get_variable_summary(df)
                self.save_dataframe(writer, var_summary, "Variable_Summary")

                # 4. Statistical Test Execution (Mapped directly to TEST_MAPPINGS list logic)
                metrics_log = []

                if "Classification" in self.CONFIG["MODEL_TYPE"]:
                    if self._test_enabled("Performance Metrics"):
                        print("-> Running Classification Metrics...")
                        metrics, fpr, tpr, cm = stx.compute_classification_metrics(y_true, y_pred, self.CONFIG["PARAMS"]["clf_threshold"])
                        metrics_log.append({"Test": "Classification Performance", **metrics})

                        if fpr is not None:
                            roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
                            self.save_dataframe(writer, roc_df, "PlotData_ROC")
                            fig_roc = px.line(roc_df, x="FPR", y="TPR", title="ROC Curve")
                            fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                            self.export_plotly_to_html(fig_roc)

                            cm_df = pd.DataFrame(cm)
                            self.save_dataframe(writer, cm_df, "PlotData_ConfusionMatrix")
                            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Oranges', title="Confusion Matrix")
                            self.export_plotly_to_html(fig_cm)

                    if self._test_enabled("Ranking"):
                        gini = stx.compute_gini(y_true, y_pred)
                        metrics_log.append({"Test": "Gini Rank Ordering", "Value": gini})

                    if self._test_enabled("Class Imbalance Handling"):
                        print("-> Running Imbalance (SMOTE)...")
                        try:
                            _, y_res = stx.apply_smote(X, y_true, self.CONFIG["PARAMS"]["smote_neighbors"])
                            imbalance_df = pd.DataFrame({
                                "Original": y_true.value_counts(),
                                "Post_SMOTE": y_res.value_counts()
                            }).reset_index().rename(columns={"index": "Class"})
                            self.save_dataframe(writer, imbalance_df, "SMOTE_Imbalance")
                        except Exception as e:
                            print(f"SMOTE skipped: {e}")

                elif "Regression" in self.CONFIG["MODEL_TYPE"]:
                    if self._test_enabled("Performance Metrics"):
                        print("-> Running Regression Metrics...")
                        metrics = stx.compute_regression_metrics(y_true, y_pred)
                        metrics_log.append({"Test": "Regression Performance", **metrics})

                        res_df = pd.DataFrame({"Predicted": y_pred, "Residual Error": (y_true - y_pred)})
                        self.save_dataframe(writer, res_df, "PlotData_Residuals")
                        fig_res = px.scatter(res_df, x="Predicted", y="Residual Error", title="Residuals")
                        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
                        self.export_plotly_to_html(fig_res)

                if "Supervised" in self.CONFIG["MODEL_TYPE"]:
                    print("-> Running VIF, CV, Learning Curves & SHAP...")
                    if self._test_enabled("Statistical Diagnostics"):
                        try:
                            vif_df = stx.compute_vif(X)
                            self.save_dataframe(writer, vif_df, "VIF_Diagnostics")
                        except Exception:
                            pass

                    if self._test_enabled("Validation & Sampling"):
                        mean_sc, std_sc = stx.compute_cross_validation(X, y_true, self.CONFIG["MODEL_TYPE"])
                        metrics_log.append({"Test": "5-Fold CV Average Score", "Mean": mean_sc, "StdDev": std_sc})

                    if self._test_enabled("Bias–Variance Analysis"):
                        t_size, t_mean, test_mean = stx.compute_learning_curve(X, y_true, self.CONFIG["MODEL_TYPE"])
                        lc_df = pd.DataFrame({"Train_Size": t_size, "Train_Score": t_mean, "Test_Score": test_mean})
                        self.save_dataframe(writer, lc_df, "PlotData_LearningCurve")
                        fig_lc = go.Figure()
                        fig_lc.add_trace(go.Scatter(x=t_size, y=t_mean, mode='lines+markers', name='Train Score'))
                        fig_lc.add_trace(go.Scatter(x=t_size, y=test_mean, mode='lines+markers', name='Test Score'))
                        fig_lc.update_layout(title="Learning Curve (Surrogate)", xaxis_title="Training Size", yaxis_title="Score")
                        self.export_plotly_to_html(fig_lc)

                    if self._test_enabled("Explainability"):
                        try:
                            imp, shap_matrix, shap_basis = stx.compute_explainability(X, y_true, self.CONFIG["MODEL_TYPE"], self.CONFIG["PARAMS"]["surrogate_depth"])
                            self.save_dataframe(writer, imp, "PlotData_FeatureImportance")
                            fig_imp = px.bar(imp, x="Importance", y="Feature", orientation='h', title="Feature Importances")
                            self.export_plotly_to_html(fig_imp)

                            # max_display matches the 15 rows kept in `imp`, so
                            # the chart and the saved sheet show the same
                            # features rather than disagreeing.
                            shap_top_n = 15
                            shap.summary_plot(shap_matrix, X.fillna(0), plot_type="bar",
                                              max_display=shap_top_n, show=False)

                            # Size the figure AFTER the call, never before:
                            # summary_plot creates its own figure and resizes
                            # it, so a plt.figure(figsize=…) beforehand is
                            # simply discarded. Height tracks the number of
                            # bars, which is what stops a wide feature set
                            # rendering as a squashed stack of slivers.
                            fig_shap = plt.gcf()
                            n_bars = min(int(X.shape[1]), shap_top_n)
                            fig_shap.set_size_inches(8.0, max(2.8, 0.38 * n_bars + 1.5))
                            ax_shap = fig_shap.axes[0]
                            ax_shap.set_xlabel("mean(|SHAP value|)"
                                               + (f" — {shap_basis}" if shap_basis else ""))
                            ax_shap.set_title("Explainability — global feature importance (surrogate model)")
                            # constrained rather than tight: tight_layout has no
                            # idea how wide a long feature name is until it is
                            # drawn, which is how the axis label ended up
                            # sliced off the right edge before.
                            fig_shap.set_layout_engine("constrained")

                            self.export_matplotlib_to_html(fig_shap)
                            plt.close(fig_shap)
                        except Exception as e:
                            print(f"Explainability execution failed: {e}")

                    if self._test_enabled("Robustness & Sensitivity"):
                        base_sc, noise_sc = stx.compute_robustness(X, y_true, self.CONFIG["MODEL_TYPE"])
                        metrics_log.append({"Test": "Robustness Baseline Score", "Value": base_sc})
                        metrics_log.append({"Test": "Robustness Noisy Score", "Value": noise_sc})
                        metrics_log.append({"Test": "Performance Degradation", "Value": base_sc - noise_sc})

                if "Clustering" in self.CONFIG["MODEL_TYPE"]:
                    print("-> Running Clustering Metrics...")
                    if self._test_enabled("Clustering Metrics"):
                        metrics = stx.compute_clustering_metrics(X, y_pred)
                        metrics_log.append({"Test": "Clustering Metrics", **metrics})
                    if self._test_enabled("Granularity"):
                        hhi = stx.compute_hhi(y_pred)
                        metrics_log.append({"Test": "Herfindahl–Hirschman Index (HHI)", "Value": hhi})

                if "Dimensionality Reduction" in self.CONFIG["MODEL_TYPE"]:
                    print("-> Running PCA and Correlation Diagnostics...")
                    if self._test_enabled("Dimensionality Reduction Metrics"):
                        evr = stx.compute_pca_variance(X)
                        pca_df = pd.DataFrame({"Component": range(1, len(evr)+1), "Explained Variance Ratio": evr})
                        self.save_dataframe(writer, pca_df, "PlotData_PCA")
                        fig_pca = px.bar(pca_df, x="Component", y="Explained Variance Ratio", title="PCA Explained Variance")
                        self.export_plotly_to_html(fig_pca)

                    if self._test_enabled("Diagnostics"):
                        corr = stx.compute_correlation(X)
                        self.save_dataframe(writer, corr.reset_index(), "PlotData_Correlation")
                        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
                        self.export_plotly_to_html(fig_corr)

                if split_col and split_col in df.columns and self._test_enabled("Drift Detection"):
                    print("-> Running Drift Detection (PSI)...")
                    train_data = perf_df[perf_df[split_col] == perf_df[split_col].unique()[0]][pred_col]
                    test_data = perf_df[perf_df[split_col] == perf_df[split_col].unique()[-1]][pred_col]
                    psi_val = stx.calculate_psi(train_data, test_data, self.CONFIG["PARAMS"]["psi_bins"])
                    metrics_log.append({"Test": "Population Stability Index (PSI)", "Value": psi_val})

                # Save aggregated metrics to Excel
                self.save_dataframe(writer, pd.DataFrame(metrics_log), "Execution_Metrics")
                writer.close()
                print(f"[OK] Data processing complete. Metrics and raw plotting data saved to {excel_path}.")

                # Write HTML file
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write("<html><head><title>Evaluation Visualizations</title></head><body style='font-family: Arial, sans-serif; padding: 20px;'>")
                    f.write("<h2>Model Evaluation - Interactive Visualizations</h2>")
                    for html_fig in self.html_visualizations:
                        f.write(html_fig)
                    f.write("</body></html>")
                print(f"[OK] Visualizations bundled into {html_path}.")

        _TEST_LAB_NS = _MainPipeline()
        _MAIN_PIPELINE_IMPORT_ERROR = None
        return _TEST_LAB_NS
    except Exception as exc:
        _MAIN_PIPELINE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return None


def _category_slug(name: str) -> str:
    """Filesystem-safe folder name for one TEST_MAPPINGS category."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


class _LineLogWriter(io.TextIOBase):
    """File-like adapter forwarding complete lines to a log callback — used
    to surface main_pipeline's print() output in the UI processing log."""

    def __init__(self, log_fn: Callable[[str], None]):
        self._log_fn = log_fn
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._log_fn(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._log_fn(self._buf)
        self._buf = ""


def _ensure_excel_input(data_path: str, df: pd.DataFrame, log_fn=print) -> str:
    """
    main_pipeline.run_pipeline() loads its input with pd.read_excel, so any
    non-Excel selection (CSV / pickle / parquet) is written from the
    already-loaded DataFrame to a temporary .xlsx and that path is handed
    to the pipeline. Pure I/O adaptation — no test logic involved.
    """
    if str(data_path).lower().endswith((".xlsx", ".xls")):
        return str(data_path)
    tmp_dir = Path(tempfile.mkdtemp(prefix="ask_testlab_"))
    tmp_path = tmp_dir / (Path(str(data_path)).stem + ".xlsx")
    df.to_excel(tmp_path, index=False)
    log_fn(f"[adapt] Input converted to a temporary Excel copy for the pipeline: {tmp_path}")
    return str(tmp_path)


def run_deterministic_tests(mp, data_path: str, categories: list[str],
                            observed_col: str, predicted_col: str,
                            split_col: Optional[str], image_dir: str,
                            output_root: str, params: dict,
                            selected_tests: Optional[set[str]] = None,
                            log_fn=print) -> dict:
    """
    Execute main_pipeline.run_pipeline() once per selected TEST_MAPPINGS
    category. Each category writes to <output_root>/<category-slug>/ with
    main_pipeline's own file names (Model_Evaluation_Report.xlsx,
    Interactive_Visualizations.html, noisy_*.png), so multi-category runs
    never overwrite each other. html_visualizations — a module global that
    run_pipeline itself never clears — is cleared per category so repeat
    runs don't accumulate stale figures.

    `selected_tests` — None runs the full battery; otherwise a set of Test
    Category names handed to main_pipeline's CONFIG["SELECTED_TESTS"] gate
    so run_pipeline() itself skips the unselected tests.

    Returns {category: {output_dir, excel_path, html_path, tables,
                        plotly_figs, mpl_pngs, extra_files, error}}.
    """
    results: dict = {}

    # run_pipeline()'s print() output is redirected into log_fn below; if
    # log_fn itself prints (it does by default), route that to the REAL
    # stdout so it can't recurse back into the redirect writer.
    real_stdout = sys.stdout

    def _safe_log(msg: str) -> None:
        with contextlib.redirect_stdout(real_stdout):
            log_fn(msg)

    for cat in categories:
        out_dir = Path(output_root) / _category_slug(cat)
        out_dir.mkdir(parents=True, exist_ok=True)
        cat_res = {
            "category": cat, "output_dir": str(out_dir), "error": None,
            "excel_path": None, "html_path": None, "tables": {},
            "plotly_figs": [], "mpl_pngs": [], "extra_files": [],
        }
        results[cat] = cat_res
        log_fn(f"==== {cat} ====")

        config_backup = {k: (dict(v) if isinstance(v, dict) else v)
                         for k, v in mp.CONFIG.items()}
        orig_plotly = mp.export_plotly_to_html
        orig_mpl    = mp.export_matplotlib_to_html

        def _cap_plotly(fig, _res=cat_res, _orig=orig_plotly):
            _res["plotly_figs"].append(fig)
            _orig(fig)

        def _cap_mpl(fig, _res=cat_res, _orig=orig_mpl):
            # PNG must be grabbed NOW — run_pipeline closes the figure
            # right after this export call returns.
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
                _res["mpl_pngs"].append(buf.getvalue())
            except Exception:
                pass
            _orig(fig)

        try:
            mp.CONFIG.update({
                "DATA_PATH":     str(data_path),
                "IMAGE_DIR":     str(image_dir),
                "OUTPUT_DIR":    str(out_dir),
                "MODEL_TYPE":    cat,
                "OBSERVED_COL":  observed_col,
                "PREDICTED_COL": predicted_col,
                "SPLIT_COL":     split_col,
                "PARAMS":        dict(params),
                "SELECTED_TESTS": (set(selected_tests) if selected_tests
                                   else None),
            })
            mp.html_visualizations.clear()
            mp.export_plotly_to_html     = _cap_plotly
            mp.export_matplotlib_to_html = _cap_mpl
            stream = _LineLogWriter(_safe_log)
            with contextlib.redirect_stdout(stream):
                mp.run_pipeline()
            stream.flush()
        except Exception as exc:
            cat_res["error"] = f"{type(exc).__name__}: {exc}"
            log_fn(f"[ERROR] {cat}: {cat_res['error']}")
        finally:
            mp.export_plotly_to_html     = orig_plotly
            mp.export_matplotlib_to_html = orig_mpl
            mp.CONFIG.clear()
            mp.CONFIG.update(config_backup)
            mp.html_visualizations.clear()

        excel_path = out_dir / "Model_Evaluation_Report.xlsx"
        html_path  = out_dir / "Interactive_Visualizations.html"
        if excel_path.exists():
            cat_res["excel_path"] = str(excel_path)
            try:
                cat_res["tables"] = pd.read_excel(excel_path, sheet_name=None)
            except Exception as exc:
                log_fn(f"[WARN] Could not read back {excel_path.name}: {exc}")
        if html_path.exists():
            cat_res["html_path"] = str(html_path)
        cat_res["extra_files"] = sorted(
            str(p) for p in out_dir.glob("noisy_*") if p.is_file())
    return results



def _sheet_view_for_category(sheet: str, sdf: pd.DataFrame, cat: str) -> pd.DataFrame:
    """The saved workbook's Test Dictionary lists EVERY model category
    (main_pipeline writes the full glossary); for display and the results
    chat, show only the rows of the category that was actually run."""
    if sheet == "Test Dictionary" and "Model Category" in sdf.columns:
        filtered = sdf[sdf["Model Category"] == cat]
        if len(filtered):
            return filtered
    return sdf
