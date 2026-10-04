"""Charts: every kind, many columns at once, forgiving column names, computed results,
and error messages the agent can act on."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ask.agent.tools import ToolContext, dispatch
from ask.analysis import charts
from ask.analysis.charts import ChartError, build
from ask.sources.registry import SourceRegistry

DEMO = Path(__file__).resolve().parent.parent / "demo data" / "Data - num.xlsx"


@pytest.fixture
def sales():
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "store_id": rng.choice([f"S{i}" for i in range(8)], n),
        "quantity": rng.integers(1, 20, n),
        "unit_price": rng.uniform(1, 50, n),
        "revenue": rng.uniform(10, 900, n),
        "profit": rng.normal(50, 20, n),
        "flag": rng.choice(["train", "test"], n),
        "is_promo": rng.choice([True, False], n),
        "TimeStamp": pd.date_range("2024-01-01", periods=n, freq="D"),
    })


def _panels(fig):
    return len(fig.data)


class TestManyColumns:
    def test_histogram_without_x_draws_every_numeric_column(self, sales):
        fig, notes = build(sales, "histogram")
        assert _panels(fig) == 4                         # quantity, unit_price, revenue, profit (bool excluded)
        titles = [a.text for a in fig.layout.annotations]
        assert titles == ["quantity", "unit_price", "revenue", "profit"]
        assert fig.layout.title.text == "Histograms of 4 numeric columns" and notes == []

    @pytest.mark.parametrize("kind,trace", [("box", "box"), ("violin", "violin")])
    def test_box_and_violin_grids(self, sales, kind, trace):
        fig, _ = build(sales, kind, columns=["revenue", "profit"])
        assert {t.type for t in fig.data} == {trace} and _panels(fig) == 2

    def test_non_numeric_columns_skipped_with_note(self, sales):
        fig, notes = build(sales, "histogram", columns=["revenue", "store_id", "profit"])
        assert _panels(fig) == 2 and "skipped non-numeric column(s): store_id" in notes[0]

    def test_single_column_list_is_a_normal_chart(self, sales):
        fig, _ = build(sales, "histogram", columns=["revenue"])
        assert fig.layout.title.text == "Distribution of revenue"

    def test_panel_cap(self):
        wide = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 30)), columns=[f"c{i}" for i in range(30)])
        fig, notes = build(wide, "histogram")
        assert _panels(fig) == charts.MAX_PANELS and "first 24 of 30" in notes[0]

    def test_no_numeric_columns(self):
        with pytest.raises(ChartError, match="No numeric columns"):
            build(pd.DataFrame({"a": ["x", "y"]}), "histogram")


class TestKinds:
    def test_count_of_categories(self, sales):
        fig, _ = build(sales, "count", x="store_id")
        assert fig.data[0].type == "bar" and sum(fig.data[0].y) == len(sales)

    def test_bar_aggregates_repeated_x(self, sales):
        fig, notes = build(sales, "bar", x="store_id", y="revenue")
        got = dict(zip(fig.data[0].x, fig.data[0].y))
        assert got == pytest.approx(sales.groupby("store_id")["revenue"].sum().to_dict())
        assert "aggregated by store_id with sum" in notes[0]

    def test_bar_with_explicit_agg(self, sales):
        fig, notes = build(sales, "bar", x="flag", y="profit", agg="mean")
        got = dict(zip(fig.data[0].x, fig.data[0].y))
        assert got == pytest.approx(sales.groupby("flag")["profit"].mean().to_dict()) and notes == []

    def test_line_over_time_is_sorted(self, sales):
        shuffled = sales.sample(frac=1, random_state=1)
        fig, _ = build(shuffled, "line", x="TimeStamp", y="revenue")
        xs = list(fig.data[0].x)
        assert xs == sorted(xs)

    def test_area(self, sales):
        assert build(sales, "area", x="TimeStamp", y="profit")[0].data[0].type == "scatter"

    def test_scatter_with_trendline(self, sales):
        fig, _ = build(sales, "scatter", x="unit_price", y="revenue", trendline=True)
        assert len(fig.data) == 2                        # points + OLS line

    def test_scatter_samples_large_data(self):
        big = pd.DataFrame({"a": np.arange(30_000), "b": np.arange(30_000)})
        fig, notes = build(big, "scatter", x="a", y="b")
        assert len(fig.data[0].x) == charts.MAX_POINTS and "random sample" in notes[0]

    def test_pie_counts_and_values(self, sales):
        counts = build(sales, "pie", x="flag")[0]
        assert sum(counts.data[0].values) == len(sales)
        values = build(sales, "pie", x="flag", y="revenue")[0]
        assert sum(values.data[0].values) == pytest.approx(sales["revenue"].sum())

    def test_correlation_heatmap(self, sales):
        fig, _ = build(sales, "heatmap")
        z = np.array(fig.data[0].z)
        assert z.shape == (4, 4) and np.allclose(np.diag(z), 1.0)

    def test_crosstab_heatmap(self, sales):
        fig, _ = build(sales, "heatmap", x="flag", y="store_id")
        assert np.array(fig.data[0].z).sum() == len(sales)

    @pytest.mark.parametrize("alias,kind", [("hist", "histogram"), ("boxplot", "box"), ("pies", "pie"),
                                            ("correlation", "heatmap"), ("Bars", "bar")])
    def test_kind_aliases(self, sales, alias, kind):
        build(sales, alias, x="revenue" if kind in ("histogram", "box") else "flag",
              y="revenue" if kind == "bar" else None)


class TestForgivingInputs:
    @pytest.mark.parametrize("given", ["REVENUE", "Revenue", "unit price", "UnitPrice"])
    def test_case_and_spacing_fixed(self, sales, given):
        build(sales, "histogram", x=given)

    def test_typo_gets_suggestion(self, sales):
        with pytest.raises(ChartError, match="Did you mean 'revenue'"):
            build(sales, "histogram", x="revenu")

    @pytest.mark.parametrize("kind,args,msg", [
        ("scatter", {"x": "revenue"}, "needs x and y"),
        ("line", {"x": "TimeStamp"}, "needs x and y"),
        ("pie", {}, "needs x"),
        ("count", {}, "needs x"),
        ("bar", {"x": "flag", "y": "revenue", "agg": "average"}, "agg must be one of"),
    ])
    def test_missing_arguments_explained(self, sales, kind, args, msg):
        with pytest.raises(ChartError, match=msg):
            build(sales, kind, **args)

    def test_empty_table(self):
        with pytest.raises(ChartError, match="empty"):
            build(pd.DataFrame(), "histogram")


class TestComputedResults:
    def test_frame_from_series_with_period_index(self, sales):
        res = sales.groupby(sales["TimeStamp"].dt.to_period("M"))["revenue"].sum()
        frame = charts.frame_from_result(res)
        assert list(frame.columns) == ["TimeStamp", "revenue"]
        assert pd.api.types.is_datetime64_any_dtype(frame["TimeStamp"])

    def test_scalar_rejected(self):
        with pytest.raises(ChartError, match="not a single value"):
            charts.frame_from_result(3.5)


class TestTool:
    @pytest.fixture
    def ctx(self, tmp_path, sales):
        p = tmp_path / "sales.parquet"
        sales.to_parquet(p)
        reg = SourceRegistry()
        reg.load(p)
        return ToolContext(registry=reg, output_dir=tmp_path / "out")

    def call(self, ctx, **args):
        return dispatch(ctx, "make_chart", json.dumps(args))

    def test_all_numeric_histograms_in_one_call(self, ctx):
        out = self.call(ctx, kind="histogram")
        assert out.startswith("Chart 'Histograms of 4 numeric columns' created")
        assert ctx.artifacts[-1]["type"] == "plotly" and Path(ctx.artifacts[-1]["path"]).exists()

    def test_expression_monthly_trend(self, ctx):
        out = self.call(ctx, kind="line",
                        expression="df.groupby(df['TimeStamp'].dt.to_period('M'))['revenue'].sum()")
        assert "created" in out and len(ctx.artifacts) == 1

    def test_unsafe_expression_rejected_as_fixable(self, ctx):
        out = self.call(ctx, kind="line", expression="df.to_csv('x.csv')")
        assert out.startswith("Chart not created: expression rejected") and "call make_chart again" in out

    def test_errors_tell_the_agent_to_retry(self, ctx):
        out = self.call(ctx, kind="histogram", x="revenu")
        assert "Did you mean 'revenue'" in out and "call make_chart again" in out
        assert ctx.artifacts == []

    def test_notes_reported(self, ctx):
        out = self.call(ctx, kind="bar", x="store_id", y="revenue")
        assert "Notes: revenue aggregated by store_id with sum" in out


@pytest.mark.skipif(not DEMO.exists(), reason="demo data not present")
def test_demo_data_all_numeric_histograms():
    """The exact case that used to fail: histograms of every numeric column of the demo workbook."""
    from ask.sources.loaders import load_path
    [src] = load_path(DEMO)
    fig, _ = build(src.df, "histogram")
    assert [a.text for a in fig.layout.annotations] == charts.numeric_columns(src.df)
    assert len(fig.data) == 8
