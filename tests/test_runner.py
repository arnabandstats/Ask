"""The adapters around the verbatim test engine and data-quality checks."""
from __future__ import annotations

import pandas as pd
import pytest

from ask.analysis import runner
from ask.sources.loaders import load_path

EXPECTED_MODEL_TYPES = ["Supervised: Classification", "Supervised: Regression",
                        "Unsupervised: Clustering", "Unsupervised: Dimensionality Reduction",
                        "Computer Vision & Image Processing"]


class TestCatalog:
    def test_model_types(self):
        assert list(runner.test_catalog()) == EXPECTED_MODEL_TYPES

    def test_classification_tests(self):
        cats = [c for c, _ in runner.test_catalog()["Supervised: Classification"]]
        assert cats == ["Performance Metrics", "Class Imbalance Handling", "Ranking",
                        "Statistical Diagnostics", "Validation & Sampling", "Explainability",
                        "Bias–Variance Analysis", "Robustness & Sensitivity", "Drift Detection"]

    def test_catalog_text(self):
        text = runner.catalog_text()
        assert "Supervised: Regression:" in text and "  - Drift Detection: Population Stability" in text

    @pytest.mark.slow
    def test_catalog_matches_engine(self):
        """The ast-read catalog must equal what the engine itself uses."""
        from ask.analysis import test_engine as te
        mp = te._import_main_pipeline()
        assert mp is not None, te._MAIN_PIPELINE_IMPORT_ERROR
        parsed = {mt: [f"{c}: {d}" for c, d in tests] for mt, tests in runner.test_catalog().items()}
        assert parsed == mp.TEST_MAPPINGS


class TestValidation:
    @pytest.fixture
    def src(self, portfolio_csv):
        return load_path(portfolio_csv)[0]

    def test_unknown_model_type(self, src, tmp_path):
        with pytest.raises(ValueError, match="Unknown model_type"):
            runner.run_model_tests(src, "Supervised: Magic", "default_flag", "pd_score", None, None, tmp_path)

    def test_unknown_column(self, src, tmp_path):
        with pytest.raises(KeyError, match="Column 'target' not in portfolio.csv"):
            runner.run_model_tests(src, "Supervised: Classification", "target", "pd_score", None, None, tmp_path)

    def test_unknown_test_name(self, src, tmp_path):
        with pytest.raises(ValueError, match="Unknown test"):
            runner.run_model_tests(src, "Supervised: Classification", "default_flag", "pd_score",
                                   None, ["Vibes Check"], tmp_path)


class TestDataQuality:
    def test_tables_and_figures(self, portfolio_csv):
        src = load_path(portfolio_csv)[0]
        res = runner.run_quality(src, log=lambda m: None)
        assert set(res["tables"]) == {"Missing Values", "Validity Checks", "Outlier Summary (IQR)",
                                      "Column Validity", "Descriptive Statistics"}
        assert len(res["mpl_pngs"]) == len(res["png_captions"]) >= 2
        assert all(p.startswith(b"\x89PNG") for p in res["mpl_pngs"])

    def test_detects_missing_and_duplicates(self, tmp_path):
        df = pd.DataFrame({"a": [1, 1, None, 4], "b": ["x", "x", "y", None]})
        p = tmp_path / "dirty.csv"
        df.to_csv(p, index=False)
        res = runner.run_quality(load_path(p)[0], log=lambda m: None)
        miss = res["tables"]["Missing Values"].set_index("column")["missing_count"]
        assert miss["a"] == 1 and miss["b"] == 1
        text = runner.summarize_quality(res)
        assert "1 duplicate row" in text


@pytest.mark.slow
class TestRealEngineRun:
    def test_classification_run_matches_independent_metrics(self, portfolio_csv, tmp_path):
        from sklearn.metrics import roc_auc_score
        src = load_path(portfolio_csv)[0]
        results = runner.run_model_tests(src, "Supervised: Classification", "default_flag", "pd_score",
                                         "sample", ["Performance Metrics", "Drift Detection"], tmp_path,
                                         log=lambda m: None)
        res = results["Supervised: Classification"]
        assert res["error"] is None
        metrics = res["tables"]["Execution_Metrics"]
        auc = metrics.loc[metrics["Test"] == "Classification Performance", "auc"].iloc[0]
        assert auc == pytest.approx(roc_auc_score(src.df["default_flag"], src.df["pd_score"]), abs=1e-3)
        assert (tmp_path / "Supervised_Classification" / "Model_Evaluation_Report.xlsx").exists()
        assert res["plotly_figs"]
        text = runner.summarize_test_results(results, "Supervised: Classification")
        assert "Execution_Metrics" in text and "Test Dictionary" not in text

    def test_non_excel_input_converted(self, portfolio_csv, tmp_path):
        src = load_path(portfolio_csv)[0]
        results = runner.run_model_tests(src, "Supervised: Regression", "pd_score", "x1", None,
                                         ["Performance Metrics"], tmp_path, log=lambda m: None)
        assert results["Supervised: Regression"]["error"] is None
