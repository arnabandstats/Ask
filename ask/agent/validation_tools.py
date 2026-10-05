"""Validator tools: the deterministic test library, model artefacts, Unity Catalog
tables, static code review and documentation checklists.

Each tool returns plain text for the model; tables and charts go to ctx.artifacts.
Every test run is saved (ask/validation/store.py) and identified by a run_id the
answer cites as [test:<run_id>].
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import pandas as pd

from ask import config
from ask.validation import core as vcore
from ask.validation import store as vstore


def _tools():
    from ask.agent import tools    # tools imports this module; import lazily to avoid a cycle
    return tools


# ── running tests ──────────────────────────────────────────────────────────

def _run_ctx(ctx, spec: vcore.TestSpec, source: str | None, sheet: str | None) -> vcore.RunContext:
    reg = ctx.registry
    needs_data = any(p.kind in {"column", "columns"} for p in spec.params)
    df, name = None, ""
    if needs_data or source:
        src = _tools()._data(ctx, source, sheet)
        df, name = src.df, src.name
    judge = None
    if spec.kind == "judge":
        from ask.validation.judge import make_judge
        judge = make_judge(ctx.judge_model)
    return vcore.RunContext(df=df, tables={s.name: s.df for s in reg.of_kind("data")},
                            models=dict(reg.models), seed=config.VALIDATION_SEED, judge=judge,
                            source_name=name)


def _record(ctx, res: vcore.TestResult, show: bool = True, group: str | None = None) -> None:
    t = _tools()
    vstore.save(res)
    ctx.runs.append(res.run_id)
    if show and res.status == "ok":
        g = group or f"{res.test_name} · test:{res.run_id}"
        for name, tdf in res.tables.items():
            if isinstance(tdf, pd.DataFrame) and not tdf.empty:
                ctx.artifacts.append(t._table_artifact(tdf, name, g))
        for fig in res.figures:
            ctx.artifacts.append(t._fig_artifact(ctx, fig, group=g))
    ctx.notes.append(res.to_text(max_rows=12)[:3000])


def t_list_validation_tests(ctx, model_type: str | None = None, area: str | None = None,
                            query: str | None = None) -> str:
    if model_type and model_type not in vcore.MODEL_TYPES:
        return f"Unknown model_type '{model_type}'. Use one of: {', '.join(vcore.MODEL_TYPES)}"
    specs = vcore.catalog(model_type, area, query)
    if not specs:
        return "No tests match. Try a broader query, or omit area/query."
    lines, cur = [], None
    for s in specs:
        if s.area != cur:
            cur = s.area
            lines.append(f"\n{cur}:")
        req = [p.name for p in s.params if p.required]
        opt = [p.name for p in s.params if not p.required]
        judge = " [LLM judge]" if s.kind == "judge" else ""
        lines.append(f"  - {s.id} — {s.name}{judge}. needs: {', '.join(req) or '-'}"
                     + (f"; optional: {', '.join(opt)}" if opt else ""))
    head = f"{len(specs)} tests" + (f" for {model_type}" if model_type else "") + \
        ". Call describe_test for parameter details before running one you haven't used."
    return head + "\n".join(lines)


def t_describe_test(ctx, test_id: str) -> str:
    return vcore.describe(vcore.get(test_id))


def t_run_validation_test(ctx, test_id: str, params: dict | None = None, source: str | None = None,
                          sheet: str | None = None, show: bool = True) -> str:
    spec = vcore.get(test_id)
    ctx.status(f"Running {spec.name}")
    rctx = _run_ctx(ctx, spec, source, sheet)
    res = vcore.run_test(spec.id, rctx, params or {})
    _record(ctx, res, show)
    return res.to_text()


def _suite_specs(model_type: str, mapping: dict, areas: list[str] | None, include_judge: bool):
    names = set(mapping)
    out = []
    for s in vcore.catalog(model_type):
        if model_type != "general" and model_type not in s.model_types:
            continue                      # generic data tests: run them explicitly
        if (s.kind == "judge" and not include_judge) or not s.suite:
            continue
        if areas and not any(a.lower() in s.area.lower() for a in areas):
            continue
        req = {p.name for p in s.params if p.required}
        declared = {p.name for p in s.params}
        if req <= names and (req or declared & names):
            out.append(s)
    return out


def t_run_validation_suite(ctx, model_type: str, columns: dict, source: str | None = None,
                           sheet: str | None = None, areas: list[str] | None = None,
                           include_judge: bool = False) -> str:
    if model_type not in vcore.MODEL_TYPES:
        return f"Unknown model_type '{model_type}'. Use one of: {', '.join(vcore.MODEL_TYPES)}"
    columns = dict(columns)
    if "pd" in columns and "score" not in columns:
        columns["score"] = columns["pd"]        # a PD ranks risk too: discrimination tests take `score`
    specs = _suite_specs(model_type, columns, areas, include_judge)
    if not specs:
        return ("No test's required inputs are covered by these columns. Call list_validation_tests "
                f"for {model_type} to see what each test needs.")
    if len(specs) > config.MAX_SUITE_TESTS:
        specs = specs[:config.MAX_SUITE_TESTS]
    ok, other = [], []
    for i, spec in enumerate(specs, 1):
        ctx.status(f"Suite {i}/{len(specs)}: {spec.name}")
        params = {k: v for k, v in columns.items() if spec.param(k) is not None}
        res = vcore.run_test(spec.id, _run_ctx(ctx, spec, source, sheet), params)
        _record(ctx, res, show=True, group=f"Suite · {model_type} · {spec.area}")
        if res.status == "ok":
            head = "; ".join(f"{k} = {vcore.fmt(v)}" for k, v in list(res.summary.items())[:8])
            note = f" (notes: {' '.join(res.notes)[:200]})" if res.notes else ""
            ok.append(f"- [test:{res.run_id}] {spec.area} · {spec.name}: {head}{note}")
        else:
            other.append(f"- {spec.id}: {res.status} — {res.error[:200]}")
    text = [f"Ran {len(specs)} {model_type} tests with columns {columns}. Full tables are shown to "
            "the user; call get_test_run(run_id) for any result's detail.", *ok]
    if other:
        text += ["Not run / not applicable:", *other]
    return "\n".join(text)


def t_get_test_run(ctx, run_id: str) -> str:
    rec = vstore.load(run_id.strip().removeprefix("test:").strip("[]"))
    parts = [f"[test:{rec['run_id']}] {rec['test_name']} ({rec['test_id']}) — {rec['status']}",
             f"Params: {rec['params']}", f"Results: {rec['summary']}"]
    for name, t in rec.get("tables", {}).items():
        df = pd.DataFrame(t["data"], columns=t["columns"])
        parts.append(f"--- {name} ({len(df)} rows) ---\n{df.head(40).to_string(index=False)}")
    if rec.get("notes"):
        parts.append("Notes: " + " ".join(rec["notes"]))
    parts.append(f"Provenance: data sha256={rec['data_fingerprint']}, rows={rec['rows_in']}, "
                 f"seed={rec['seed']}, versions={rec['versions']}, run at {rec['started']}")
    return "\n".join(parts)


# ── models and tables ──────────────────────────────────────────────────────

def t_load_model(ctx, location: str, name: str | None = None) -> str:
    from ask.sources import databricks
    from ask.validation.models import load_model
    loc = location.strip().strip("\"'`")
    if not loc.startswith(("runs:/", "models:/")) and not Path(loc).exists() and databricks.looks_like_path(loc):
        loc = str(databricks.fetch(loc))       # /Workspace or /Volumes file through the API
    ctx.status(f"Loading model {Path(loc).name}")
    m = load_model(loc, name)
    ctx.registry.models[m.name] = m
    ctx.sources_changed = True
    return (f"Loaded model {m.describe()}.\nNote for the user: loading a pickle runs code stored in "
            "the file, so only load artefacts from the model owner's controlled location. "
            f"Pass model='{m.name}' to tests that take a model.")


def t_load_table(ctx, table: str, columns: list[str] | None = None, where: str | None = None,
                 max_rows: int | None = None, key: list[str] | None = None, name: str | None = None) -> str:
    from ask.sources import tables
    ctx.status(f"Reading {table}")
    spec = {"table": table, "columns": columns, "where": where, "max_rows": max_rows, "key": key}
    loaded = tables.load_table(**spec)
    src = ctx.registry.add_table(name or table.split(".")[-1], loaded, spec)
    ctx.sources_changed = True
    return f"Loaded [data] {src.name}: {loaded.describe()}.\nSQL: {loaded.sql}"


def t_profile_table(ctx, table: str, columns: list[str] | None = None, key: list[str] | None = None,
                    where: str | None = None) -> str:
    from ask.sources import tables
    ctx.status(f"Profiling {table} in SQL")
    prof, extra, backend = tables.profile_table(table, columns, key, where)
    ctx.artifacts.append(_tools()._table_artifact(prof, f"Profile · {table}"))
    text = (f"Full-population profile of {table} ({extra['rows']:,} rows, computed in {backend}"
            + (f", filter: {where}" if where else "") + ")")
    if key:
        text += (f"\nDuplicate keys on {key}: {extra['duplicate_keys']:,} keys covering "
                 f"{extra['rows_in_duplicate_keys']:,} rows")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        text += "\n" + prof.to_string(index=False)
    ctx.notes.append(text[:4000])
    return text


def t_query_table(ctx, sql: str, name: str | None = None) -> str:
    from ask.sources import tables
    ctx.status("Running SQL")
    df, clean, backend = tables.query_table(sql)
    loaded = tables.TableLoad(df, name or "query", len(df), False, "", clean, backend)
    src = ctx.registry.add_table(name or "query_result", loaded, {"sql": clean})
    ctx.sources_changed = True
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        preview = df.head(30).to_string(index=False)
    return f"Loaded [data] {src.name}: {len(df):,} rows × {df.shape[1]} columns from:\n{clean}\n{preview}"


# ── code and documentation review ──────────────────────────────────────────

def _text_files(ctx, source: str | None, path_glob: str | None, kinds=("repo", "docs")):
    src = ctx.registry.get(source, *kinds)
    files = {k: v for k, v in src.files.items() if not path_glob or fnmatch.fnmatch(k, path_glob)
             or fnmatch.fnmatch(Path(k).name, path_glob)}
    return src, files


def _cite(src, rel: str, a: int, b: int | None = None) -> str:
    return f"[{rel}:L{a}-{b or a}]"


def t_scan_code(ctx, source: str | None = None, rules: list[str] | None = None,
                path_glob: str | None = None, include_info: bool = False) -> str:
    from ask.validation import code_scan
    src, files = _text_files(ctx, source, path_glob, ("repo",))
    ctx.status(f"Scanning {len(files)} files in {src.name}")
    for rel in [r for r in files if r.lower().endswith(".ipynb")]:
        raw = Path(src.path) / rel          # raw JSON keeps execution counts and error outputs
        try:
            files[rel] = raw.read_text(encoding="utf-8")
        except OSError:
            pass                            # same line numbers either way (scanner renders cells)
    findings = code_scan.scan(files, rules, include_info)
    if not findings:
        return f"No static-analysis findings in {len(files)} files of {src.name} (rules: {rules or 'all'})."
    rows = []
    for f in findings:
        ctx.evidence.add(src.name, f.file, f.line, f.end_line)
        rows.append({"rule": f.rule, "category": f.category, "file": f.file, "line": f.line,
                     "title": f.title, "message": f.message, "snippet": f.snippet})
    table = pd.DataFrame(rows)
    ctx.artifacts.append(_tools()._table_artifact(table, f"Code scan · {src.name}", "Code review"))
    summary = code_scan.summarise(findings)
    lines = [f"{len(findings)} findings in {src.name} ({len(files)} files scanned). Each line below "
             "was shown to you, so you may cite it. Read the surrounding code before concluding: "
             "rules are heuristics, and a finding is only confirmed once you have read it in context.",
             summary.to_string(index=False), ""]
    for f in findings[:150]:
        lines.append(f"- {f.rule} {f.title} {_cite(src, f.file, f.line, f.end_line)}: {f.message} | `{f.snippet[:140]}`")
    if len(findings) > 150:
        lines.append(f"… {len(findings) - 150} more (narrow with rules or path_glob).")
    ctx.notes.append("\n".join(lines)[:5000])
    return "\n".join(lines)


def t_list_standards(ctx, model_type: str | None = None) -> str:
    from ask.validation import doc_review
    stds = doc_review.list_standards(model_type)
    return "\n".join(f"- {s.id}: {s.title} ({len(s.requirements)} requirements; applies to "
                     f"{', '.join(s.applies_to)}). {s.version_note}" for s in stds) or "No standards found."


def t_check_documentation(ctx, standard: str, source: str | None = None,
                          model_type: str | None = None) -> str:
    from ask.validation import doc_review
    std = doc_review.load_standard(standard)
    src, files = _text_files(ctx, source, None, ("docs", "repo"))
    ctx.status(f"Checking {src.name} against {std.title}")
    checks = doc_review.check(files, std, model_type)
    table = doc_review.checks_table(checks)
    ctx.artifacts.append(_tools()._table_artifact(table, f"{std.title} · {src.name}", "Documentation review"))
    lines = [f"{std.title} — {len(checks)} requirements checked against {src.name} by KEYWORD SEARCH. "
             "Status says whether matching text was found, NOT whether it is adequate: read the cited "
             "lines (read_file) and judge adequacy yourself before stating a gap or a finding.",
             f"{std.version_note}"]
    missing = []
    for c in checks:
        r = c.requirement
        if not c.hits:
            missing.append(f"- {r.id} [{r.topic}] {r.requirement[:220]} (source: {r.source})")
            continue
        lines.append(f"\n{r.id} [{r.topic}] {c.status} (keyword coverage {c.coverage:.0%}) — {r.requirement} "
                     f"(source: {r.source})")
        for h in c.hits:
            ctx.evidence.add(src.name, h.file, h.line, h.line)
            lines.append(f"    {_cite(src, h.file, h.line)} {h.text.strip()[:180]}")
    if missing:
        lines.append(f"\nNo matching text found for {len(missing)} requirements (confirm with search/grep "
                     "before calling any of them a gap; the wording may differ):")
        lines += missing
    text = "\n".join(lines)
    ctx.notes.append(text[:6000])
    return text


def t_cross_check(ctx, doc_source: str, code_source: str | None = None,
                  data_source: str | None = None) -> str:
    from ask.validation import doc_review
    docs = ctx.registry.get(doc_source, "docs", "repo")
    out = []
    if code_source:
        code = ctx.registry.get(code_source, "repo")
        ctx.status("Cross-checking documented values against code")
        vt = doc_review.cross_check_values(docs.files, code.files)
        ctx.artifacts.append(_tools()._table_artifact(vt, "Documented values vs code", "Consistency"))
        import re
        for _, r in vt.iterrows():           # the lines shown here become citable
            ctx.evidence.add(docs.name, r["Document"], int(r["Line"]), int(r["Line"]))
            for f, ln in re.findall(r"\[([^\[\]]+):L(\d+)\]", str(r["Code matches"])):
                ctx.evidence.add(code.name, f, int(ln), int(ln))
        with pd.option_context("display.width", 240, "display.max_columns", 12, "display.max_colwidth", 60):
            out.append(f"Documented values vs code ({len(vt)} values):\n{vt.head(80).to_string(index=False)}")
    if data_source:
        data = ctx.registry.get(data_source, "data")
        ct = doc_review.cross_check_columns(docs.files, list(map(str, data.df.columns)))
        ctx.artifacts.append(_tools()._table_artifact(ct, "Documented variables vs data columns", "Consistency"))
        with pd.option_context("display.width", 240, "display.max_columns", 12, "display.max_colwidth", 60):
            out.append(f"Documented variables vs data columns:\n{ct.head(80).to_string(index=False)}")
    if not out:
        return "Pass code_source and/or data_source to compare the documentation with."
    text = "\n\n".join(out)
    ctx.notes.append(text[:5000])
    return text


# ── schemas ────────────────────────────────────────────────────────────────

def schemas(_fn, _S, _SRC, _SHEET) -> list[dict]:
    mt = {"type": "string", "enum": list(vcore.MODEL_TYPES),
          "description": "; ".join(f"{k} = {v}" for k, v in vcore.MODEL_TYPES.items())}
    strs = {"type": "array", "items": _S}
    return [
        _fn("list_validation_tests", "Catalog of the deterministic validation tests (and LLM-judge tests, "
            "marked), grouped by area, with the inputs each needs. Filter by model_type, area "
            "(e.g. Calibration, Discrimination, Stability, Back-testing, Fairness) or query words.",
            {"model_type": mt, "area": _S, "query": _S}),
        _fn("describe_test", "Full description of one test: what it measures, H0, parameters, references.",
            {"test_id": _S}, ["test_id"]),
        _fn("run_validation_test", "Run ONE deterministic validation test and save the result with a "
            "run_id. Map the test's parameter names to column names (or values) in params, e.g. "
            "{\"target\": \"default_flag\", \"pd\": \"pd_12m\", \"grade\": \"rating\"}. Cite the "
            "result as [test:<run_id>].",
            {"test_id": _S, "params": {"type": "object", "description": "parameter name -> column name or value"},
             "source": _SRC, "sheet": _SHEET,
             "show": {"type": "boolean", "description": "Show the result tables/charts (default true)."}},
            ["test_id"]),
        _fn("run_validation_suite", "Run every test for a model type whose required inputs are covered "
            "by `columns` (parameter name -> column/value, e.g. {\"target\": \"dflt\", \"pd\": \"pd\", "
            "\"grade\": \"grade\", \"sample\": \"sample\", \"period\": \"year\"}). Use for 'run the "
            "tests / validate the model'. Optional areas narrow it (e.g. [\"Calibration\"]).",
            {"model_type": mt, "columns": {"type": "object"}, "source": _SRC, "sheet": _SHEET,
             "areas": strs, "include_judge": {"type": "boolean",
                                              "description": "Also run LLM-judge tests (GenAI)."}},
            ["model_type", "columns"]),
        _fn("get_test_run", "Retrieve a saved test result by run_id (all tables and provenance).",
            {"run_id": _S}, ["run_id"]),
        _fn("load_model", "Load a model artefact so tests can score it: a .pkl/.joblib file (local, "
            "/Workspace or /Volumes), or an MLflow URI (runs:/..., models:/name/1, "
            "models:/catalog.schema.model/1).", {"location": _S, "name": _S}, ["location"]),
        _fn("load_table", "Load a Unity Catalog / Delta table (catalog.schema.table) as a data source. "
            "Tables larger than the row limit are sampled deterministically (same rows every run). "
            "Select only the columns you need.",
            {"table": _S, "columns": strs, "where": {"type": "string", "description": "SQL condition"},
             "max_rows": {"type": "integer"}, "key": {**strs, "description": "Key columns (sampling/order)"},
             "name": _S}, ["table"]),
        _fn("profile_table", "Full-population data-quality profile of a Unity Catalog table computed in "
            "SQL (rows, missing, distinct, min/max/mean/std, duplicate keys). Use for big tables.",
            {"table": _S, "columns": strs, "key": strs, "where": _S}, ["table"]),
        _fn("query_table", "Run ONE read-only SQL SELECT on Unity Catalog (e.g. a GROUP BY that "
            "aggregates a big table to grade/period level) and load the result as a data source.",
            {"sql": _S, "name": _S}, ["sql"]),
        _fn("scan_code", "Deterministic static code review of a loaded repo: unseeded randomness, "
            "hard-coded values/paths/dates, data leakage patterns, silent error handling, secrets, "
            "unpinned dependencies, notebook execution order, SQL issues. Returns citable findings.",
            {"source": _SRC, "rules": strs, "path_glob": _S,
             "include_info": {"type": "boolean", "description": "Also informational rules (TODOs, docstrings)."}}),
        _fn("list_standards", "List the regulatory/standard documentation checklists available.",
            {"model_type": mt}),
        _fn("check_documentation", "Check loaded documentation against a regulatory/standard checklist "
            "(e.g. ECB Guide to internal models, CRR IRB, IFRS 9, EU AI Act, SR 11-7). Returns, per "
            "requirement, the matching document lines (citable) or 'no evidence found'.",
            {"standard": _S, "source": _SRC, "model_type": mt}, ["standard"]),
        _fn("cross_check", "Consistency between documentation and implementation: numeric parameters "
            "stated in the docs vs the same values in code, and documented variables vs data columns.",
            {"doc_source": _S, "code_source": _S, "data_source": _S}, ["doc_source"]),
    ]


IMPL = {
    "list_validation_tests": t_list_validation_tests, "describe_test": t_describe_test,
    "run_validation_test": t_run_validation_test, "run_validation_suite": t_run_validation_suite,
    "get_test_run": t_get_test_run, "load_model": t_load_model, "load_table": t_load_table,
    "profile_table": t_profile_table, "query_table": t_query_table, "scan_code": t_scan_code,
    "list_standards": t_list_standards, "check_documentation": t_check_documentation,
    "cross_check": t_cross_check,
}
TEXT = {"scan_code", "check_documentation", "cross_check"}
DATA = {"run_validation_test", "run_validation_suite", "get_test_run", "load_model", "load_table",
        "profile_table", "query_table"}
