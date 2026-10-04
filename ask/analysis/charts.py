"""Charts of a loaded table (or of a computed result) for the chat.

Covers what people ask for in practice: one column or many at once (a grid of small
charts), category counts, trends over time, relationships, correlations, shares. Column
names are matched forgivingly, and every error says how to fix the call, so the agent can
retry instead of giving up.
"""
from __future__ import annotations

import difflib
import math

import numpy as np
import pandas as pd

KINDS = ["histogram", "box", "violin", "bar", "count", "line", "area", "scatter", "pie", "heatmap"]
MULTI_KINDS = {"histogram", "box", "violin"}      # can show many columns as a grid
MAX_PANELS = 24
MAX_POINTS = 20_000
PALETTE = ["#3d5a8a", "#d9502b", "#7a8b6f", "#c49a3a", "#6b5b95", "#4f8fa3", "#a35d6a", "#5b8c85"]


class ChartError(ValueError):
    """A problem with the request; the message says how to fix it."""


# ── column handling ────────────────────────────────────────────────────────

def resolve_column(df: pd.DataFrame, name: str | None) -> str | None:
    """Exact name, else a case/space-insensitive match, else a ChartError with suggestions."""
    if name is None or name == "":
        return None
    cols = [str(c) for c in df.columns]
    if name in cols:
        return name
    key = lambda s: "".join(str(s).lower().split()).replace("_", "")  # noqa: E731
    matches = [c for c in cols if key(c) == key(name)]
    if len(matches) == 1:
        return matches[0]
    close = difflib.get_close_matches(name, cols, n=3, cutoff=0.5)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise ChartError(f"Column {name!r} not found.{hint} Columns: {', '.join(cols[:60])}")


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.select_dtypes(include=[np.number]).columns
            if not pd.api.types.is_bool_dtype(df[c])]


def _style(fig, title: str, height: int | None = None):
    fig.update_layout(title=title or "", template="plotly_white", colorway=PALETTE,
                      margin=dict(t=56 if title else 20, l=8, r=8, b=8),
                      font=dict(family="Inter, system-ui, sans-serif", size=12))
    if height:
        fig.update_layout(height=height)
    return fig


# ── chart kinds ────────────────────────────────────────────────────────────

def _grid(df: pd.DataFrame, kind: str, cols: list[str], title: str, nbins: int | None,
          log_y: bool, notes: list[str]):
    """One small chart per column."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    non_numeric = [c for c in cols if c not in numeric_columns(df)]
    if non_numeric:
        notes.append(f"skipped non-numeric column(s): {', '.join(non_numeric)}")
        cols = [c for c in cols if c not in non_numeric]
    if not cols:
        raise ChartError(f"No numeric columns to draw a {kind} of. Numeric columns: "
                         f"{', '.join(numeric_columns(df)) or 'none'}.")
    if len(cols) > MAX_PANELS:
        notes.append(f"showing the first {MAX_PANELS} of {len(cols)} columns")
        cols = cols[:MAX_PANELS]
    if len(cols) == 1:
        return _single(df, kind, cols[0], None, None, title, nbins, log_y, notes)

    per_row = 3 if len(cols) > 4 else 2
    rows = math.ceil(len(cols) / per_row)
    fig = make_subplots(rows=rows, cols=per_row, subplot_titles=cols,
                        horizontal_spacing=0.08, vertical_spacing=min(0.12, 0.6 / rows))
    for i, c in enumerate(cols):
        r, k = i // per_row + 1, i % per_row + 1
        series = df[c].dropna()
        if kind == "histogram":
            trace = go.Histogram(x=series, nbinsx=nbins or 30, marker_color=PALETTE[0], showlegend=False)
        elif kind == "box":
            trace = go.Box(y=series, name=c, marker_color=PALETTE[0], boxpoints="outliers", showlegend=False)
        else:
            trace = go.Violin(y=series, name=c, line_color=PALETTE[0], box_visible=True, showlegend=False)
        fig.add_trace(trace, row=r, col=k)
    if log_y:
        fig.update_yaxes(type="log")
    plural = {"histogram": "Histograms", "box": "Box plots", "violin": "Violin plots"}[kind]
    return _style(fig, title or f"{plural} of {len(cols)} numeric columns", 240 * rows + 60)


def _single(df, kind, x, y, color, title, nbins, log_y, notes, agg=None, trendline=False):
    import plotly.express as px

    data = df
    if kind == "histogram":
        if x is None:
            raise ChartError("A histogram needs a column (x), or several via columns=[...].")
        fig = px.histogram(data, x=x, color=color, nbins=nbins, log_y=log_y)
    elif kind in ("box", "violin"):
        if x is None and y is None:
            raise ChartError(f"A {kind} plot needs a numeric column (x or y).")
        value, group = (y, x) if y else (x, None)
        fn = px.box if kind == "box" else px.violin
        fig = fn(data, x=group, y=value, color=color, log_y=log_y)
    elif kind in ("bar", "count") and (y is None or kind == "count"):
        if x is None:
            raise ChartError("A count/bar chart of categories needs x (the category column).")
        counts = df[x].astype("string").fillna("(missing)").value_counts()
        if len(counts) > 30:
            notes.append(f"showing the 30 most frequent of {len(counts)} values")
        data = counts.head(30).rename_axis(x).reset_index(name="count")
        fig = px.bar(data, x=x, y="count", log_y=log_y)
    elif kind in ("bar", "line", "area"):
        if x is None or y is None:
            raise ChartError(f"A {kind} chart needs x and y (or use kind='count' for category counts).")
        if agg or df[x].duplicated().any():
            how = agg or ("sum" if kind == "bar" else "mean")
            if not agg:
                notes.append(f"{y} aggregated by {x} with {how} (x has repeated values)")
            data = getattr(df.groupby(x, dropna=False, observed=True)[y], how)().reset_index()
        data = data.sort_values(x) if kind in ("line", "area") else data
        if kind == "bar" and len(data) > 50:
            notes.append(f"showing the top 50 of {len(data)} bars")
            data = data.sort_values(y, ascending=False).head(50)
        fn = {"bar": px.bar, "line": px.line, "area": px.area}[kind]
        fig = fn(data, x=x, y=y, color=color if color in data.columns else None, log_y=log_y)
    elif kind == "scatter":
        if x is None or y is None:
            raise ChartError("A scatter plot needs x and y.")
        if len(data) > MAX_POINTS:
            notes.append(f"plotted a random sample of {MAX_POINTS:,} of {len(data):,} rows")
            data = data.sample(MAX_POINTS, random_state=0)
        try:
            fig = px.scatter(data, x=x, y=y, color=color, log_y=log_y,
                             trendline="ols" if trendline else None, opacity=0.7)
        except Exception:                          # trendline needs numeric x/y and statsmodels
            fig = px.scatter(data, x=x, y=y, color=color, log_y=log_y, opacity=0.7)
            if trendline:
                notes.append("trend line skipped (needs numeric x and y)")
    elif kind == "pie":
        if x is None:
            raise ChartError("A pie chart needs x (the category column), optionally y (values).")
        if y:
            data = df.groupby(x, dropna=False, observed=True)[y].sum().reset_index()
            fig = px.pie(data, names=x, values=y)
        else:
            data = df[x].astype("string").fillna("(missing)").value_counts().head(12)
            fig = px.pie(names=data.index, values=data.values)
            if df[x].nunique() > 12:
                notes.append("showing the 12 largest categories")
    else:
        raise ChartError(f"kind must be one of: {', '.join(KINDS)}")
    if color is None and kind not in ("pie",):
        fig.update_traces(marker_color=PALETTE[0], selector=dict(type="bar"))
        fig.update_traces(marker_color=PALETTE[0], selector=dict(type="histogram"))
    default = {"histogram": f"Distribution of {x}", "box": f"Spread of {y or x}",
               "violin": f"Spread of {y or x}", "count": f"Counts of {x}", "pie": f"Share by {x}",
               "scatter": f"{y} vs {x}"}.get(kind, f"{y or x}" + (f" by {x}" if y and x else ""))
    return _style(fig, title or default)


def _heatmap(df, cols, x, y, title, notes):
    import plotly.express as px

    if x and y:                                        # two categories: a crosstab
        table = pd.crosstab(df[y].astype("string"), df[x].astype("string"))
        fig = px.imshow(table, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        return _style(fig, title or f"{y} × {x} (counts)")
    nums = [c for c in (cols or numeric_columns(df)) if c in numeric_columns(df)]
    if len(nums) < 2:
        raise ChartError("A correlation heatmap needs at least two numeric columns.")
    corr = df[nums].corr()
    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale="RdBu_r")
    return _style(fig, title or "Correlation between numeric columns", max(360, 34 * len(nums) + 120))


# ── entry point ────────────────────────────────────────────────────────────

def build(df: pd.DataFrame, kind: str, x: str | None = None, y: str | None = None,
          color: str | None = None, title: str | None = None, agg: str | None = None,
          columns: list[str] | None = None, nbins: int | None = None, log_y: bool = False,
          trendline: bool = False):
    """Returns (figure, notes). Raises ChartError with an actionable message."""
    kind = (kind or "").lower().strip()
    kind = {"hist": "histogram", "histograms": "histogram", "boxplot": "box", "bars": "bar",
            "counts": "count", "correlation": "heatmap", "corr": "heatmap", "lines": "line",
            "scatterplot": "scatter", "donut": "pie"}.get(kind, kind)
    if kind not in KINDS and kind.endswith("s") and kind[:-1] in KINDS:
        kind = kind[:-1]                                   # "pies", "areas", ...
    if kind not in KINDS:
        raise ChartError(f"Unknown chart kind {kind!r}. Use one of: {', '.join(KINDS)}.")
    if agg and agg not in {"mean", "sum", "count", "median", "min", "max"}:
        raise ChartError("agg must be one of mean, sum, count, median, min, max.")
    if df is None or df.empty:
        raise ChartError("The table is empty, so there is nothing to chart.")
    notes: list[str] = []
    x, y, color = resolve_column(df, x), resolve_column(df, y), resolve_column(df, color)
    cols = [resolve_column(df, c) for c in columns] if columns else None

    if kind == "heatmap":
        return _heatmap(df, cols, x, y, title, notes), notes
    if kind in MULTI_KINDS and (cols or (x is None and y is None)):
        return _grid(df, kind, cols or numeric_columns(df), title, nbins, log_y, notes), notes
    return _single(df, kind, x, y, color, title, nbins, log_y, notes, agg=agg, trendline=trendline), notes


def frame_from_result(res) -> pd.DataFrame:
    """Turn a query result (Series / DataFrame / scalar) into a chartable table."""
    if isinstance(res, pd.Series):
        name = res.name if res.name is not None else "value"
        out = res.to_frame(name=str(name))
    elif isinstance(res, pd.DataFrame):
        out = res.copy()
    else:
        raise ChartError("The expression must produce a table or a series to chart, not a single value.")
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    out.columns = [str(c) if not isinstance(c, tuple) else "_".join(map(str, c)) for c in out.columns]
    for c in out.columns:                               # Periods (e.g. .dt.to_period('M')) -> timestamps
        if isinstance(out[c].dtype, pd.PeriodDtype):
            out[c] = out[c].dt.to_timestamp()
    return out
