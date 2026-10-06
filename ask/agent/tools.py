"""Tools the agent can call, their JSON schemas, and the dispatcher.

Tools return plain text for the model. Anything the user should SEE (tables,
charts, figures, report files) is added to ctx.artifacts and rendered by the UI.
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ask import config
from ask.analysis import compare as cmp
from ask.analysis import data_query as dq
from ask.analysis import runner, simulate
from ask.agent import guardrails
from ask.agent import validation_tools as vt
from ask.retrieval import search as rs
from ask.sources import databricks
from ask.sources.loaders import LoadError
from ask.sources.paths import find_paths
from ask.sources.registry import SourceRegistry

TEXT_TOOLS = {"search", "grep", "read_file", "list_files", "compare", "overview"} | vt.TEXT
DATA_TOOLS = {"data_overview", "query_data", "make_chart", "run_data_quality", "list_tests",
              "run_tests", "simulate_data"} | vt.DATA


@dataclass
class ToolContext:
    registry: SourceRegistry
    output_dir: Path
    status: Callable[[str], None] = lambda msg: None
    evidence: rs.Evidence = field(default_factory=rs.Evidence)
    artifacts: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)        # analysis results kept for follow-ups
    used: set[str] = field(default_factory=set)
    sources_changed: bool = False
    runs: list[str] = field(default_factory=list)            # validation run_ids produced this turn
    judge_model: str | None = None                           # model for LLM-judge tests
    guard: list[str] = field(default_factory=list)           # guardrail events shown under the answer


# ── artifacts ──────────────────────────────────────────────────────────────

def _table_artifact(df: pd.DataFrame, title: str, group: str | None = None, max_rows: int = 2000) -> dict:
    shown = df.head(max_rows)
    return {"type": "table", "title": title, "group": group,
            "data": shown.to_json(orient="split", date_format="iso", default_handler=str),
            "rows": int(len(df))}


def _save(ctx: ToolContext, suffix: str, data: bytes | str) -> str:
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    p = ctx.output_dir / f"{uuid.uuid4().hex[:10]}{suffix}"
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_bytes(data)
    return str(p)


def _fig_artifact(ctx: ToolContext, fig, title: str | None = None, group: str | None = None) -> dict:
    if title is None:
        try:
            title = fig.layout.title.text or "Chart"
        except Exception:
            title = "Chart"
    return {"type": "plotly", "title": title, "group": group, "path": _save(ctx, ".json", fig.to_json())}


def _png_artifact(ctx: ToolContext, png: bytes, title: str, group: str | None = None) -> dict:
    return {"type": "image", "title": title, "group": group, "path": _save(ctx, ".png", png)}


# ── tool implementations ───────────────────────────────────────────────────

def t_load_path(ctx: ToolContext, path: str, kind: str | None = None) -> str:
    if databricks.url_in(path):
        path = databricks.path_from_url(databricks.url_in(path)) or ""
        if not path:
            return databricks.URL_HELP
    mentions = find_paths(path) or []
    resolved = next((m.path for m in mentions if m.path), None)
    target = resolved or Path(path.strip().strip("\"'`"))
    if not target.exists():
        if not databricks.looks_like_path(path):
            return f"Path not found: {path}. Ask the user to check it."
        target = Path(databricks.normalise(path))      # fetched through the Databricks API
    ctx.status(f"Loading {target.name}")
    try:
        srcs = ctx.registry.load(target, kind)
    except LoadError as exc:
        return f"Could not load {target}: {exc}"
    ctx.sources_changed = True
    return "\n".join(f"Loaded [{s.kind}] {s.name}: {s.summary()}" for s in srcs) + \
        "\n\nCurrently loaded:\n" + ctx.registry.describe()


def t_unload(ctx: ToolContext, name: str) -> str:
    try:
        src = ctx.registry.get(name)
    except KeyError as exc:
        return str(exc)
    ctx.registry.remove(src.name)
    ctx.sources_changed = True
    return f"Unloaded {src.name}."


def t_search(ctx: ToolContext, query: str, source: str | None = None) -> str:
    ctx.status(f"Searching for “{query}”")
    return rs.search(ctx.registry, ctx.evidence, query, source)


def t_grep(ctx: ToolContext, pattern: str, regex: bool = False, source: str | None = None,
           path_glob: str | None = None, case_sensitive: bool = False) -> str:
    ctx.status(f"Grepping “{pattern}”")
    return rs.grep(ctx.registry, ctx.evidence, pattern, regex, source, path_glob, case_sensitive)


def t_read_file(ctx: ToolContext, file: str, start_line: int | None = None,
                end_line: int | None = None) -> str:
    ctx.status(f"Reading {file}")
    return rs.read_file(ctx.registry, ctx.evidence, file, start_line, end_line)


def t_overview(ctx: ToolContext, source: str | None = None) -> str:
    ctx.status("Mapping the repository")
    return rs.overview(ctx.registry, ctx.evidence, source)


def t_list_files(ctx: ToolContext, source: str | None = None, glob: str | None = None) -> str:
    return rs.list_files(ctx.registry, source, glob)


def _data(ctx: ToolContext, source: str | None, sheet: str | None):
    src = ctx.registry.get(source, "data")
    if sheet:
        if sheet not in src.sheets:
            raise KeyError(f"Sheet '{sheet}' not in {src.name}. Sheets: {', '.join(map(str, src.sheets))}")
        src.df = src.sheets[sheet]
    return src


def t_data_overview(ctx: ToolContext, source: str | None = None, sheet: str | None = None) -> str:
    src = _data(ctx, source, sheet)
    return dq.overview(src.df, src.name)


def t_query_data(ctx: ToolContext, expression: str, source: str | None = None,
                 sheet: str | None = None, show: bool = False, title: str | None = None) -> str:
    src = _data(ctx, source, sheet)
    ctx.status("Querying data")
    dfs = {s.name: s.df for s in ctx.registry.of_kind("data")}
    try:
        res = dq.evaluate(expression, src.df, dfs)
    except dq.UnsafeExpression as exc:
        return f"Expression rejected: {exc}"
    text, frame = dq.format_result(res)
    if show and frame is not None:
        ctx.artifacts.append(_table_artifact(frame, title or "Result"))
    ctx.notes.append(f"query_data `{expression}` on {src.name}:\n{text[:1500]}")
    return text


def t_make_chart(ctx: ToolContext, kind: str, x: str | None = None, y: str | None = None,
                 color: str | None = None, title: str | None = None, agg: str | None = None,
                 columns: list[str] | None = None, expression: str | None = None,
                 nbins: int | None = None, log_y: bool = False, trendline: bool = False,
                 source: str | None = None, sheet: str | None = None) -> str:
    from ask.analysis import charts

    src = _data(ctx, source, sheet)
    ctx.status("Drawing chart")
    data = src.df
    try:
        if expression:                     # chart a computed result, e.g. a monthly sum
            dfs = {s.name: s.df for s in ctx.registry.of_kind("data")}
            try:
                data = charts.frame_from_result(dq.evaluate(expression, src.df, dfs))
            except dq.UnsafeExpression as exc:
                raise charts.ChartError(f"expression rejected: {exc}") from exc
            if x is None and y is None and len(data.columns) == 2 and kind not in charts.MULTI_KINDS:
                x, y = map(str, data.columns)
        fig, notes = charts.build(data, kind, x, y, color, title, agg, columns, nbins, log_y, trendline)
    except charts.ChartError as exc:
        return (f"Chart not created: {exc} Fix the arguments and call make_chart again "
                "(don't tell the user it failed unless it keeps failing).")
    except Exception as exc:
        return (f"Chart not created: {type(exc).__name__}: {exc}. Try another kind or explicit "
                f"x/y/columns. Columns: {', '.join(map(str, data.columns[:60]))}")
    fig_title = fig.layout.title.text or kind
    ctx.artifacts.append(_fig_artifact(ctx, fig, fig_title))
    extra = f" Notes: {'; '.join(notes)}." if notes else ""
    return f"Chart '{fig_title}' created and shown to the user.{extra}"


def t_run_data_quality(ctx: ToolContext, source: str | None = None, sheet: str | None = None) -> str:
    src = _data(ctx, source, sheet)
    ctx.status(f"Running data-quality checks on {src.name}")
    res = runner.run_quality(src, log=lambda m: ctx.status(m))
    group = f"Data quality · {src.name}"
    for name, tdf in res.get("tables", {}).items():
        if isinstance(tdf, pd.DataFrame) and not tdf.empty:
            ctx.artifacts.append(_table_artifact(tdf, name, group))
    for cap, png in zip(res.get("png_captions", []), res.get("mpl_pngs", [])):
        ctx.artifacts.append(_png_artifact(ctx, png, cap, group))
    text = runner.summarize_quality(res)
    ctx.notes.append(f"Data-quality results for {src.name}:\n{text[:6000]}")
    return text


def t_list_tests(ctx: ToolContext) -> str:
    return runner.catalog_text()


def t_run_tests(ctx: ToolContext, model_type: str, observed_col: str, predicted_col: str,
                split_col: str | None = None, tests: list[str] | None = None,
                source: str | None = None, sheet: str | None = None,
                params: dict | None = None, image_dir: str | None = None) -> str:
    src = _data(ctx, source, sheet)
    ctx.status(f"Running {model_type} tests on {src.name} (first run loads the test libraries)")
    out_dir = ctx.output_dir / f"tests_{time.strftime('%Y%m%d_%H%M%S')}"
    results = runner.run_model_tests(src, model_type, observed_col, predicted_col, split_col,
                                     tests, out_dir, params, image_dir,
                                     log=lambda m: ctx.status(m[:120]))
    from ask.analysis.test_engine import _sheet_view_for_category
    group = f"Model tests · {model_type} · {src.name}"
    for cat, res in results.items():
        for sheet_name, sdf in (res.get("tables") or {}).items():
            if sheet_name.startswith("PlotData_") or sdf.empty:
                continue
            ctx.artifacts.append(_table_artifact(_sheet_view_for_category(sheet_name, sdf, cat), sheet_name, group))
        for fig in res.get("plotly_figs") or []:
            ctx.artifacts.append(_fig_artifact(ctx, fig, group=group))
        for i, png in enumerate(res.get("mpl_pngs") or [], 1):
            ctx.artifacts.append(_png_artifact(ctx, png, f"Figure {i}", group))
        for key, label in (("excel_path", "Excel report"), ("html_path", "Interactive charts (HTML)")):
            if res.get(key):
                ctx.artifacts.append({"type": "file", "title": label, "group": group, "path": res[key]})
    text = runner.summarize_test_results(results, model_type)
    ctx.notes.append(f"Model test results ({model_type}, observed={observed_col}, "
                     f"predicted={predicted_col}, split={split_col}) on {src.name}:\n{text[:8000]}")
    return text


def t_simulate_data(ctx: ToolContext, n: int, columns: list[dict], seed: int | None = None,
                    name: str | None = None) -> str:
    seed = config.VALIDATION_SEED if seed is None else int(seed)
    spec = {"n": int(n), "columns": columns, "seed": seed}
    ctx.status(f"Simulating {int(n):,} rows")
    df = simulate.generate(**spec)
    src = ctx.registry.add_simulated(name or "simulated", df, spec)
    ctx.sources_changed = True
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        stats = df.describe(include="all").T.to_string()
    text = (f"Loaded [data] {src.name}: SIMULATED, {len(df):,} rows × {df.shape[1]} columns, seed {seed} "
            f"(reproducible). It is now the active table for make_chart / query_data / tests.\n"
            f"Spec: {columns}\nSample statistics:\n{stats}")
    ctx.notes.append(text[:3000])
    return text


def t_compare(ctx: ToolContext, a: str, b: str) -> str:
    ctx.status(f"Comparing {a} with {b}")
    text = cmp.compare(ctx.registry, ctx.evidence, a, b)
    ctx.notes.append(f"compare {a} vs {b}:\n{text[:3000]}")
    return text


# ── schemas ────────────────────────────────────────────────────────────────

def _fn(name: str, desc: str, props: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required or []}}}


_S = {"type": "string"}
_SIM_DISTS, _SIM_HELP = list(simulate.DISTRIBUTIONS), simulate.help_text()
_SRC = {"type": "string", "description": "Loaded source name. Omit when only one fits."}
_SHEET = {"type": "string", "description": "Excel sheet to use (makes it the active table)."}

SCHEMAS = [
    _fn("load_path", "Load a repository folder, document(s) or data file from the user's machine.",
        {"path": _S, "kind": {"type": "string", "enum": ["repo", "docs", "data"],
                              "description": "Only if the user said what it is."}}, ["path"]),
    _fn("unload_source", "Unload a loaded source.", {"name": _S}, ["name"]),
    _fn("search", "Ranked keyword search over loaded repos/documents. Returns line-numbered snippets.",
        {"query": _S, "source": _SRC}, ["query"]),
    _fn("grep", "Find exact text or a regex in loaded repos/documents, with line numbers.",
        {"pattern": _S, "regex": {"type": "boolean"}, "source": _SRC,
         "path_glob": {"type": "string", "description": "e.g. '*.py' or 'src/*'"},
         "case_sensitive": {"type": "boolean"}}, ["pattern"]),
    _fn("read_file", f"Read lines of a file (max {config.MAX_READ_LINES} per call), line-numbered.",
        {"file": {"type": "string", "description": "Path as shown by other tools; may be 'source:path'."},
         "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["file"]),
    _fn("overview", "Structure of a loaded repo/docs source: projects it contains (for a workspace "
        "of several repos), top-level folders with file counts, and each README's opening lines. "
        "Call this FIRST for broad questions (what is this, what's happening, architecture, summary).",
        {"source": _SRC}),
    _fn("list_files", "List files in loaded repos/documents.",
        {"source": _SRC, "glob": {"type": "string"}}),
    _fn("simulate_data", "Generate a SIMULATED table (synthetic random data) and load it as the active "
        "data source, e.g. 1000 draws from a normal distribution, a mix of columns, a toy default "
        "flag. Seeded and reproducible. Then use make_chart / query_data / tests on it. "
        "Distributions and params: " + _SIM_HELP,
        {"n": {"type": "integer", "description": "Number of rows (max 1,000,000)."},
         "columns": {"type": "array", "description": "One entry per column.", "items": {
             "type": "object", "properties": {
                 "name": _S, "distribution": {"type": "string", "enum": _SIM_DISTS},
                 "params": {"type": "object", "description": "e.g. {\"mean\": 0, \"var\": 1}"}},
             "required": ["name", "distribution"]}},
         "seed": {"type": "integer", "description": "Random seed (default: the validation seed)."},
         "name": {"type": "string", "description": "Source name (default 'simulated')."}},
        ["n", "columns"]),
    _fn("data_overview", "Columns, dtypes, null counts, summary stats and first rows of a table.",
        {"source": _SRC, "sheet": _SHEET}),
    _fn("query_data", "Evaluate ONE pandas expression. Names: df (active table), dfs (dict of all "
        "tables by name), pd, np. No assignments, imports or file I/O. Example: "
        "df.groupby('segment')['default'].mean().sort_values()",
        {"expression": _S, "source": _SRC, "sheet": _SHEET,
         "show": {"type": "boolean", "description": "Also show the result table to the user."},
         "title": _S}, ["expression"]),
    _fn("make_chart", "Draw a chart and show it to the user. "
        "histogram/box/violin: one column (x), several (columns=[...]), or omit both for ALL "
        "numeric columns as a grid of small charts. count: frequencies of a category (x). "
        "bar/line/area: y by x (repeated x values are aggregated). scatter: y vs x (trendline "
        "optional). pie: shares of a category. heatmap: correlation of numeric columns, or a "
        "count crosstab when x and y are categories. expression: chart the result of a pandas "
        "expression instead of the table, e.g. "
        "df.groupby(df['TimeStamp'].dt.to_period('M'))['revenue'].sum().",
        {"kind": {"type": "string", "enum": ["histogram", "box", "violin", "count", "bar", "line",
                                             "area", "scatter", "pie", "heatmap"]},
         "x": _S, "y": _S, "color": {"type": "string", "description": "Column to colour/group by."},
         "columns": {"type": "array", "items": _S, "description": "Several columns (histogram/box/violin/heatmap)."},
         "expression": {"type": "string", "description": "Pandas expression whose result is charted."},
         "agg": {"type": "string", "enum": ["mean", "sum", "count", "median", "min", "max"],
                 "description": "Aggregate y by x first."},
         "nbins": {"type": "integer"}, "log_y": {"type": "boolean"},
         "trendline": {"type": "boolean", "description": "Scatter only: add a linear trend line."},
         "title": _S, "source": _SRC, "sheet": _SHEET}, ["kind"]),
    _fn("run_data_quality", "Run the built-in deterministic data-quality checks (missing values, "
        "validity, IQR outliers, descriptive stats, plots).", {"source": _SRC, "sheet": _SHEET}),
    _fn("list_tests", "List the built-in deterministic model-validation tests by model type.", {}),
    _fn("run_tests", "Run the built-in deterministic model-validation tests on a table.",
        {"model_type": {"type": "string", "enum": list(runner.test_catalog().keys())},
         "observed_col": _S, "predicted_col": _S,
         "split_col": {"type": "string", "description": "Train/test flag column, used for drift (PSI)."},
         "tests": {"type": "array", "items": _S,
                   "description": "Test category names to run; omit for all."},
         "source": _SRC, "sheet": _SHEET,
         "params": {"type": "object", "description": "Overrides: clf_threshold, surrogate_depth, "
                    "smote_neighbors, psi_bins, blur_threshold"},
         "image_dir": {"type": "string", "description": "Image folder for computer-vision tests."}},
        ["model_type", "observed_col", "predicted_col"]),
    _fn("compare", "Exact comparison: two files (line diff), two repo/doc sources (file-level diff), "
        "a file against a folder (closest matching files + diff with the best match) or two tables "
        "(schema and statistics).",
        {"a": {"type": "string", "description": "File ('source:path') or source name."},
         "b": {"type": "string", "description": "File ('source:path') or source name."}}, ["a", "b"]),
    *vt.schemas(_fn, _S, _SRC, _SHEET),
]

_IMPL = {
    "load_path": t_load_path, "unload_source": t_unload, "search": t_search, "grep": t_grep,
    "read_file": t_read_file, "list_files": t_list_files, "overview": t_overview, "data_overview": t_data_overview,
    "query_data": t_query_data, "make_chart": t_make_chart, "run_data_quality": t_run_data_quality,
    "list_tests": t_list_tests, "run_tests": t_run_tests, "compare": t_compare,
    "simulate_data": t_simulate_data,
    **vt.IMPL,
}


def dispatch(ctx: ToolContext, name: str, arguments: str) -> str:
    fn = _IMPL.get(name)
    if fn is None:
        return f"Unknown tool {name}."
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return (f"Arguments were not valid JSON: {exc}. Pass names and short values only; tools read "
                "loaded documents themselves, so never put document text in arguments.")
    if not isinstance(args, dict):
        return "Arguments must be a JSON object of parameter names to values."
    ctx.used.add(name)
    try:
        out = fn(ctx, **{k: v for k, v in args.items() if v is not None})
    except (KeyError, ValueError, LoadError, TypeError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return f"Error: {msg}"
    except Exception as exc:
        if type(exc).__name__ in {"TableError", "DatabricksError"}:      # readable by design
            return f"Error: {exc}"
        return f"Error: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)[-1500:]}"
    if len(out) > config.MAX_TOOL_OUTPUT_CHARS:
        out = out[: config.MAX_TOOL_OUTPUT_CHARS] + "\n… (output truncated; narrow the request)"
    hits = guardrails.scan(out)            # indirect prompt injection from loaded sources
    if hits:
        ctx.guard.append(f"Text that looks like instructions was found in the {name} output "
                         f"({', '.join(hits)}); it was treated as data.")
        out = guardrails.tool_notice(hits) + out
    return out
