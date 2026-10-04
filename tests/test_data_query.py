"""Safe pandas expressions, result formatting, data overview, charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ask.analysis import data_query as dq
from ask.analysis.data_query import UnsafeExpression, evaluate


@pytest.fixture
def df(portfolio_df):
    return portfolio_df


class TestEvaluateAllowed:
    def test_groupby_mean_matches_pandas(self, df):
        res = evaluate("df.groupby('segment')['default_flag'].mean()", df, {})
        pd.testing.assert_series_equal(res, df.groupby("segment")["default_flag"].mean())

    def test_scalar(self, df):
        assert evaluate("len(df)", df, {}) == len(df)

    def test_boolean_filter(self, df):
        res = evaluate("df[df['pd_score'] > 0.5]['default_flag'].mean()", df, {})
        assert res == pytest.approx(df[df["pd_score"] > 0.5]["default_flag"].mean())

    def test_numpy_and_lambda(self, df):
        res = evaluate("df['x1'].apply(lambda v: np.sign(v)).value_counts()", df, {})
        assert res.sum() == len(df)

    def test_comprehension(self, df):
        assert evaluate("[c for c in df.columns if c.startswith('x')]", df, {}) == ["x1", "x2"]

    def test_other_tables_via_dfs(self, df):
        other = df.head(5)
        assert evaluate("len(dfs['small'])", df, {"small": other}) == 5

    def test_code_fence_stripped(self, df):
        assert evaluate("```python\nlen(df)\n```", df, {}) == len(df)

    def test_allowed_to_methods(self, df):
        assert evaluate("df['segment'].unique().tolist()", df, {})
        assert isinstance(evaluate("df.head(2).to_dict()", df, {}), dict)

    def test_does_not_mutate_data(self, df):
        before = df.copy()
        evaluate("df.assign(z=1).shape", df, {})
        pd.testing.assert_frame_equal(df, before)


class TestEvaluateRejected:
    @pytest.mark.parametrize("expr,why", [
        ("df.__class__", "private attribute"),
        ("df._data", "private attribute"),
        ("().__class__.__bases__[0].__subclasses__()", "private attribute"),
        ("pd.read_csv('secrets.csv')", "read_csv"),
        ("df.to_csv('out.csv')", "to_csv"),
        ("df.to_pickle('x.pkl')", "to_pickle"),
        ("np.save('x', df)", "save"),
        ("np.load('x.npy')", "load"),
        ("pd.io.common", "io"),
        ("df.query('x1 > 0')", "query"),
        ("df.eval('x1 + 1')", "eval"),
        ("__import__('os')", "name '__import__'"),
        ("open('x.txt')", "name 'open'"),
        ("eval('1+1')", "name 'eval'"),
        ("getattr(df, 'shape')", "name 'getattr'"),
        ("os.system('dir')", "system|name 'os'"),
    ])
    def test_blocked(self, df, expr, why):
        with pytest.raises(UnsafeExpression, match=why):
            evaluate(expr, df, {})

    @pytest.mark.parametrize("expr", ["x = 1", "import os", "df['a'] = 1", "for i in range(3): pass"])
    def test_statements_rejected(self, df, expr):
        with pytest.raises(UnsafeExpression, match="Not a single Python expression"):
            evaluate(expr, df, {})

    def test_no_builtins_leak(self, df):
        with pytest.raises(UnsafeExpression):
            evaluate("print('hi')", df, {})


class TestFormatResult:
    def test_dataframe(self, df):
        text, frame = dq.format_result(df, max_rows=10)
        assert text.startswith(f"DataFrame {len(df)} rows × 6 cols")
        assert f"… ({len(df) - 10} more rows)" in text and frame is df

    def test_series_becomes_frame(self, df):
        text, frame = dq.format_result(df["segment"].value_counts())
        assert text.startswith("Series (3 values") and isinstance(frame, pd.DataFrame)

    def test_array_and_scalar(self):
        assert dq.format_result(np.arange(4))[0].startswith("array shape (4,)")
        assert dq.format_result(3.5) == ("3.5", None)


class TestOverview:
    def test_contents(self, df):
        out = dq.overview(df, "portfolio")
        assert out.startswith(f"portfolio: {len(df):,} rows × 6 columns")
        for col in df.columns:
            assert f"\n{col} | " in out
        assert "numeric summary:" in out and "first 5 rows:" in out

    def test_handles_unhashable_and_nulls(self):
        weird = pd.DataFrame({"lists": [[1], [2], None], "n": [1.0, None, 3.0]})
        out = dq.overview(weird, "w")
        assert "lists | object" in out and "n | float64 | 2 |" in out


class TestCharts:
    @pytest.mark.parametrize("kind,x,y", [("histogram", "pd_score", None), ("bar", "segment", None),
                                          ("scatter", "x1", "pd_score"), ("box", "pd_score", None),
                                          ("line", "x1", "x2")])
    def test_kinds(self, df, kind, x, y):
        fig = dq.make_chart(df, kind, x, y, title="t")
        assert fig.layout.title.text == "t" and len(fig.data) >= 1

    def test_aggregated_bar_is_correct(self, df):
        fig = dq.make_chart(df, "bar", "segment", "default_flag", agg="mean")
        expected = df.groupby("segment")["default_flag"].mean()
        got = dict(zip(fig.data[0].x, fig.data[0].y))
        assert got == pytest.approx(expected.to_dict())

    def test_unknown_column(self, df):
        with pytest.raises(KeyError, match="Column 'nope' not found"):
            dq.make_chart(df, "histogram", "nope")

    def test_unknown_kind(self, df):
        with pytest.raises(ValueError, match="kind must be one of"):
            dq.make_chart(df, "pie", "segment")

    def test_large_scatter_sampled(self):
        big = pd.DataFrame({"a": np.arange(30_000), "b": np.arange(30_000)})
        fig = dq.make_chart(big, "scatter", "a", "b")
        assert len(fig.data[0].x) == 20_000
