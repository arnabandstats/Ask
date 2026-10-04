"""Thin adapters that CALL the verbatim test engine and data-quality checks.

No test logic lives here: we pick inputs, call run_deterministic_tests() /
run_data_quality_checks(), and serialise what they return.
"""
from __future__ import annotations

import ast
import functools
import tempfile
from pathlib import Path

import pandas as pd

from ask.sources.loaders import Source

_ENGINE_FILE = Path(__file__).with_name("test_engine.py")

DEFAULT_PARAMS = {"clf_threshold": 0.5, "surrogate_depth": 5, "smote_neighbors": 5,
                  "psi_bins": 10, "blur_threshold": 100.0}


@functools.lru_cache(maxsize=1)
def test_catalog() -> dict[str, list[tuple[str, str]]]:
    """{model type: [(test category, description)]}, read from the engine source
    with ast (no heavy imports), so Settings and the agent can list tests instantly."""
    tree = ast.parse(_ENGINE_FILE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == "TEST_MAPPINGS"):
            raw = ast.literal_eval(node.value)
            return {mt: [tuple(s.strip() for s in e.split(":", 1)) if ":" in e else (e, "")
                         for e in entries] for mt, entries in raw.items()}
    return {}


def catalog_text() -> str:
    lines = []
    for mt, tests in test_catalog().items():
        lines.append(f"{mt}:")
        lines += [f"  - {cat}: {desc}" for cat, desc in tests]
    return "\n".join(lines)


def run_model_tests(src: Source, model_type: str, observed_col: str, predicted_col: str,
                    split_col: str | None, tests: list[str] | None, output_dir: Path,
                    params: dict | None = None, image_dir: str | None = None,
                    log=print) -> dict:
    from ask.analysis import test_engine as te

    catalog = test_catalog()
    if model_type not in catalog:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose one of: {', '.join(catalog)}")
    df = src.df
    if model_type != "Computer Vision & Image Processing":
        for col in (observed_col, predicted_col, split_col):
            if col and col not in df.columns:
                raise KeyError(f"Column '{col}' not in {src.name}. Columns: {', '.join(map(str, df.columns[:60]))}")
    valid = {c for c, _ in catalog[model_type]}
    selected = None
    if tests:
        unknown = [t for t in tests if t not in valid]
        if unknown:
            raise ValueError(f"Unknown test(s) for {model_type}: {unknown}. Valid: {sorted(valid)}")
        selected = set(tests)

    mp = te._import_main_pipeline()
    if mp is None:
        raise RuntimeError(f"Test engine unavailable: {te._MAIN_PIPELINE_IMPORT_ERROR}")

    # The engine reads its input with pd.read_excel; give it the active table.
    data_path = src.path
    first_sheet = next(iter(src.sheets)) if src.sheets else None
    is_excel = str(data_path).lower().endswith((".xlsx", ".xls"))
    if not is_excel or (src.df is not src.sheets.get(first_sheet)):
        tmp = Path(tempfile.mkdtemp(prefix="ask_tests_")) / (Path(data_path).stem + ".xlsx")
        df.to_excel(tmp, index=False)
        data_path = str(tmp)

    output_dir.mkdir(parents=True, exist_ok=True)
    return te.run_deterministic_tests(
        mp, data_path, [model_type], observed_col, predicted_col, split_col,
        image_dir or str(output_dir), str(output_dir), {**DEFAULT_PARAMS, **(params or {})},
        selected_tests=selected, log_fn=log)


def summarize_test_results(results: dict, model_type: str, max_rows: int = 40) -> str:
    from ask.analysis.test_engine import _sheet_view_for_category

    parts = []
    for cat, res in results.items():
        parts.append(f"==== {cat} ====")
        if res.get("error"):
            parts.append(f"RUN ERROR: {res['error']}")
        for sheet, sdf in (res.get("tables") or {}).items():
            if sheet == "Test Dictionary":
                continue
            if sheet.startswith("PlotData_"):
                parts.append(f"--- {sheet}: {sdf.shape[0]} rows (plot data, omitted)")
                continue
            sdf = _sheet_view_for_category(sheet, sdf, model_type)
            with pd.option_context("display.width", 220, "display.max_columns", 30):
                body = sdf.head(max_rows).to_string(index=False)
            more = f"\n… ({len(sdf) - max_rows} more rows)" if len(sdf) > max_rows else ""
            parts.append(f"--- {sheet} ({sdf.shape[0]}×{sdf.shape[1]}) ---\n{body}{more}")
        titles = []
        for fig in res.get("plotly_figs") or []:
            try:
                titles.append(fig.layout.title.text or "(untitled)")
            except Exception:
                titles.append("(untitled)")
        if titles:
            parts.append("charts produced: " + "; ".join(titles))
        if res.get("mpl_pngs"):
            parts.append(f"static figures produced: {len(res['mpl_pngs'])}")
        if res.get("excel_path"):
            parts.append(f"Excel report: {res['excel_path']}")
        if res.get("html_path"):
            parts.append(f"HTML charts: {res['html_path']}")
    return "\n".join(parts)


def run_quality(src: Source, log=print) -> dict:
    from ask.analysis.data_quality import run_data_quality_checks
    return run_data_quality_checks(src.df, log_fn=log)


def summarize_quality(res: dict, max_rows: int = 40) -> str:
    parts = [res.get("summary", "")]
    for name, tdf in res.get("tables", {}).items():
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            body = tdf.head(max_rows).to_string(index=False) if hasattr(tdf, "to_string") else str(tdf)
        more = f"\n… ({len(tdf) - max_rows} more rows)" if hasattr(tdf, "__len__") and len(tdf) > max_rows else ""
        parts.append(f"--- {name} ---\n{body}{more}")
    if res.get("png_captions"):
        parts.append("figures produced: " + "; ".join(res["png_captions"]))
    return "\n".join(p for p in parts if p)
