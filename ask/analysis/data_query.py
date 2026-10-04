"""Answering questions about a loaded table: overview, safe pandas expressions, charts.

query_data evaluates ONE pandas/numpy expression. The expression is parsed and
checked first: no imports, no dunder/private attributes, no file or network
I/O methods, and only whitelisted names. That keeps "ask anything about the
data" deterministic and side-effect free without running arbitrary code.
"""
from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

_SAFE_BUILTINS = {
    "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
    "sorted": sorted, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool, "range": range,
    "zip": zip, "enumerate": enumerate, "any": any, "all": all, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
}
_BLOCKED_ATTRS = {"eval", "exec", "system", "popen", "pipe", "load", "loads", "save",
                  "savez", "tofile", "fromfile", "loadtxt", "savetxt", "genfromtxt",
                  "memmap", "ctypeslib", "lib", "testing", "os", "sys", "io", "query",
                  "api", "compat", "core", "util", "style"}
_ALLOWED_TO = {"to_frame", "to_dict", "to_list", "to_numpy", "to_string", "to_period",
               "to_datetime", "to_numeric", "to_timedelta", "to_timestamp", "to_records",
               "to_markdown", "tolist"}


class UnsafeExpression(ValueError):
    pass


def _check(tree: ast.AST, allowed_names: set[str]) -> None:
    lambda_args: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            lambda_args.update(a.arg for a in node.args.args)
        if isinstance(node, (ast.comprehension,)):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    lambda_args.add(n.id)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                             ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)):
            raise UnsafeExpression(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("_"):
                raise UnsafeExpression(f"private attribute '{a}' is not allowed")
            if a in _BLOCKED_ATTRS or a.startswith("read_") or (a.startswith("to_") and a not in _ALLOWED_TO):
                raise UnsafeExpression(f"'.{a}' is not allowed (no file/system access)")
        if isinstance(node, ast.Name) and node.id not in allowed_names and node.id not in lambda_args:
            raise UnsafeExpression(f"name '{node.id}' is not available; use df, dfs, pd, np")


def evaluate(expression: str, df: pd.DataFrame, dfs: dict[str, pd.DataFrame]) -> Any:
    expr = expression.strip()
    if expr.startswith("```"):
        expr = expr.strip("`").removeprefix("python").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"Not a single Python expression: {exc.msg}") from exc
    env = {"df": df, "dfs": dfs, "pd": pd, "np": np, **_SAFE_BUILTINS}
    _check(tree, set(env))
    # Names go in globals (not locals) so lambdas and comprehensions can see df/np/pd too.
    return eval(compile(tree, "<query_data>", "eval"), {"__builtins__": {}, **env})


def format_result(res: Any, max_rows: int = 60) -> tuple[str, pd.DataFrame | None]:
    """Text for the model, plus a DataFrame worth showing to the user (if any)."""
    with pd.option_context("display.max_columns", 50, "display.width", 200,
                           "display.max_colwidth", 60):
        if isinstance(res, pd.DataFrame):
            shown = res.head(max_rows)
            more = f"\n… ({len(res) - max_rows} more rows)" if len(res) > max_rows else ""
            return f"DataFrame {res.shape[0]} rows × {res.shape[1]} cols\n{shown.to_string()}{more}", res
        if isinstance(res, pd.Series):
            shown = res.head(max_rows)
            more = f"\n… ({len(res) - max_rows} more)" if len(res) > max_rows else ""
            frame = res.to_frame(name=res.name if res.name is not None else "value")
            return f"Series ({len(res)} values, dtype {res.dtype})\n{shown.to_string()}{more}", frame
        if isinstance(res, np.ndarray):
            return f"array shape {res.shape}\n{np.array2string(res, threshold=200)}", None
        return repr(res)[:8000], None


def overview(df: pd.DataFrame, name: str) -> str:
    n_rows, n_cols = df.shape
    lines = [f"{name}: {n_rows:,} rows × {n_cols} columns", "", "column | dtype | non-null | unique | example"]
    for c in df.columns[:120]:
        s = df[c]
        try:
            uniq = s.nunique(dropna=True)
        except TypeError:
            uniq = "?"
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        lines.append(f"{c} | {s.dtype} | {s.notna().sum():,} | {uniq} | {str(ex)[:40]}")
    if n_cols > 120:
        lines.append(f"… {n_cols - 120} more columns")
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            lines += ["", "numeric summary:", num.describe().T.round(4).head(60).to_string()]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        lines += ["", "first 5 rows:", df.head(5).to_string()[:6000]]
    return "\n".join(lines)


def make_chart(df: pd.DataFrame, kind: str, x: str | None, y: str | None = None,
               color: str | None = None, title: str | None = None, agg: str | None = None):
    import plotly.express as px

    for col in (x, y, color):
        if col and col not in df.columns:
            raise KeyError(f"Column '{col}' not found. Columns: {', '.join(map(str, df.columns[:40]))}")
    data = df
    if agg and x and y:
        data = getattr(df.groupby(x, dropna=False)[y], agg)().reset_index()
    kind = kind.lower()
    if kind == "histogram":
        fig = px.histogram(data, x=x, color=color)
    elif kind == "bar":
        if y is None and x:
            data = df[x].value_counts().head(30).rename_axis(x).reset_index(name="count")
            y = "count"
        fig = px.bar(data, x=x, y=y, color=color)
    elif kind == "line":
        fig = px.line(data.sort_values(x) if x else data, x=x, y=y, color=color)
    elif kind == "scatter":
        if len(data) > 20_000:
            data = data.sample(20_000, random_state=0)
        fig = px.scatter(data, x=x, y=y, color=color)
    elif kind == "box":
        fig = px.box(data, x=x if y else None, y=y or x, color=color)
    else:
        raise ValueError("kind must be one of histogram, bar, line, scatter, box")
    fig.update_layout(title=title or "", margin=dict(t=48 if title else 16, l=8, r=8, b=8),
                      template="plotly_white",
                      colorway=["#3d5a8a", "#d9502b", "#7a8b6f", "#c49a3a", "#6b5b95", "#4f8fa3"])
    if not color:
        fig.update_traces(marker_color="#3d5a8a")
    return fig
