"""Tool dispatch: argument handling, errors, artifacts, notes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ask import config
from ask.agent import tools as T
from ask.agent.tools import SCHEMAS, ToolContext, dispatch
from ask.sources.registry import SourceRegistry


@pytest.fixture
def ctx(tmp_path, sample_repo, portfolio_csv):
    reg = SourceRegistry()
    reg.load(sample_repo)
    reg.load(portfolio_csv)
    return ToolContext(registry=reg, output_dir=tmp_path / "out")


def call(ctx, tool, **args):
    return dispatch(ctx, tool, json.dumps(args))


class TestSchemas:
    def test_every_schema_has_an_implementation(self):
        names = {s["function"]["name"] for s in SCHEMAS}
        assert names == set(T._IMPL)

    def test_required_args_exist_in_properties(self):
        for s in SCHEMAS:
            params = s["function"]["parameters"]
            assert set(params["required"]) <= set(params["properties"]), s["function"]["name"]

    def test_run_tests_enum_matches_catalog(self):
        from ask.analysis.runner import test_catalog
        rt = next(s for s in SCHEMAS if s["function"]["name"] == "run_tests")
        assert rt["function"]["parameters"]["properties"]["model_type"]["enum"] == list(test_catalog())


class TestDispatch:
    def test_unknown_tool(self, ctx):
        assert call(ctx, "rm_rf") == "Unknown tool rm_rf."

    def test_bad_json(self, ctx):
        assert dispatch(ctx, "search", "{not json").startswith("Arguments were not valid JSON")

    def test_errors_become_text(self, ctx):
        assert call(ctx, "read_file", file="nope.py").startswith("Error: File 'nope.py'")
        assert call(ctx, "data_overview", source="ghost").startswith("Error: No loaded data source")

    def test_unexpected_argument_is_reported(self, ctx):
        assert call(ctx, "search", query="x", bogus=1).startswith("Error:")

    def test_none_arguments_dropped(self, ctx):
        assert "matching" in dispatch(ctx, "grep", json.dumps({"pattern": "MIN_ACCURACY", "source": None}))

    def test_used_tools_tracked(self, ctx):
        call(ctx, "search", query="gate")
        call(ctx, "list_tests")
        assert ctx.used == {"search", "list_tests"}

    def test_long_output_truncated(self, ctx, monkeypatch):
        monkeypatch.setattr(config, "MAX_TOOL_OUTPUT_CHARS", 100)
        out = call(ctx, "list_files")
        assert len(out) < 200 and out.endswith("(output truncated; narrow the request)")


class TestToolBehaviour:
    def test_load_path_and_unload(self, ctx, docx_file):
        out = call(ctx, "load_path", path=str(docx_file))
        assert out.startswith("Loaded [docs] policy.docx") and ctx.sources_changed
        assert call(ctx, "unload_source", name="policy.docx") == "Unloaded policy.docx."
        assert "policy.docx" not in ctx.registry.sources

    def test_load_missing_path(self, ctx, tmp_path):
        assert call(ctx, "load_path", path=str(tmp_path / "nope")).startswith("Path not found")

    def test_query_data_show_creates_table_artifact_and_note(self, ctx):
        out = call(ctx, "query_data", expression="df['segment'].value_counts()", show=True, title="Counts")
        assert out.startswith("Series (3 values")
        art = ctx.artifacts[-1]
        assert art["type"] == "table" and art["title"] == "Counts" and art["rows"] == 3
        assert ctx.notes and "value_counts" in ctx.notes[-1]

    def test_query_data_rejection(self, ctx):
        assert call(ctx, "query_data", expression="df.to_csv('x')").startswith("Expression rejected")

    def test_sheet_switch(self, ctx, tmp_path, portfolio_df):
        import pandas as pd
        p = tmp_path / "book.xlsx"
        with pd.ExcelWriter(p) as w:
            portfolio_df.to_excel(w, sheet_name="dev", index=False)
            portfolio_df.head(7).to_excel(w, sheet_name="oot", index=False)
        call(ctx, "load_path", path=str(p))
        assert call(ctx, "query_data", expression="len(df)", source="book.xlsx", sheet="oot") == "7"
        assert call(ctx, "query_data", expression="len(df)", source="book.xlsx",
                    sheet="nope").startswith("Error: Sheet 'nope'")

    def test_make_chart_saves_plotly_json(self, ctx):
        out = call(ctx, "make_chart", kind="histogram", x="pd_score")
        art = ctx.artifacts[-1]
        assert "shown to the user" in out and art["type"] == "plotly"
        assert Path(art["path"]).exists() and json.loads(Path(art["path"]).read_text())["data"]

    def test_run_data_quality_artifacts(self, ctx):
        out = call(ctx, "run_data_quality")
        groups = {a["group"] for a in ctx.artifacts}
        assert groups == {"Data quality · portfolio.csv"}
        assert any(a["type"] == "image" and Path(a["path"]).exists() for a in ctx.artifacts)
        assert "Missing Values" in out

    def test_overview_and_compare_record_evidence(self, ctx, tmp_path, sample_repo):
        call(ctx, "overview")
        assert ctx.evidence.lines("model_repo", "README.md")
        v2 = tmp_path / "train_v2.py"
        v2.write_text((sample_repo / "src" / "train.py").read_text().replace("42", "7"))
        call(ctx, "load_path", path=str(v2))
        out = call(ctx, "compare", a="train_v2.py", b="model_repo")
        assert "Exact diff with the closest match (src/train.py)" in out

    @pytest.mark.slow
    def test_run_tests_artifacts(self, ctx):
        out = call(ctx, "run_tests", model_type="Supervised: Classification", observed_col="default_flag",
                   predicted_col="pd_score", split_col="sample", tests=["Performance Metrics"])
        types = {a["type"] for a in ctx.artifacts}
        assert {"table", "plotly", "file"} <= types and "Execution_Metrics" in out
        assert all(a["group"].startswith("Model tests · Supervised: Classification") for a in ctx.artifacts)
