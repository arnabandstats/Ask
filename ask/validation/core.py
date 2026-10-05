"""The deterministic validation-test library: registry, inputs, results, provenance.

Every test is a plain Python function registered with @register. The agent never
computes a statistic itself: it picks a test from the catalog, maps columns to
the test's parameters, and run_test() does the rest:

  1. checks and converts the parameters (columns exist, numbers are numbers),
  2. fingerprints exactly the data the test reads (SHA-256 of those columns),
  3. calls the test with a fixed random seed,
  4. returns a TestResult carrying the numbers AND the provenance needed to
     reproduce them: test id + version, parameters, data fingerprint, rows,
     seed, library versions, and a run_id derived from all of those.

Same inputs -> same run_id -> same numbers. That is what makes a result citable
as [test:<run_id>] in an answer or a report.

Parameter names are a shared vocabulary so one column mapping can drive a whole
suite (see ROLES below). A test only declares the roles it needs.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

LIBRARY_VERSION = "1.0.0"
DEFAULT_SEED = 20240601

# ── model types ────────────────────────────────────────────────────────────
# Keys are what tests declare and what the agent passes; labels are for people.
MODEL_TYPES: dict[str, str] = {
    "pd": "PD / rating & scoring models (IRB, application/behavioural scorecards)",
    "lgd": "LGD models (IRB, IFRS 9, ELBE / LGD in-default)",
    "ead": "EAD / CCF models",
    "ifrs9": "IFRS 9 ECL (staging, lifetime PD, ECL measurement)",
    "ews": "Early-warning systems",
    "satellite": "Macro / satellite / stress-testing regressions and time series",
    "var": "Market risk VaR / ES (back-testing, P&L attribution)",
    "pricing": "Pricing and valuation models",
    "ccr": "Counterparty credit risk exposure (EE/EPE/PFE) and CVA",
    "aml": "AML / transaction-monitoring and screening models",
    "ml_classification": "Machine-learning classifiers",
    "ml_regression": "Machine-learning regressors",
    "ml_unsupervised": "Clustering / segmentation / dimensionality reduction",
    "genai": "GenAI / LLM / RAG / agents",
    "general": "Any model (data quality, generic statistics)",
}

# ── parameter vocabulary ───────────────────────────────────────────────────
# Use these names whenever a test needs that kind of input, so suites can pass
# one mapping to every test. A test may add its own specific parameters.
ROLES: dict[str, str] = {
    "target": "observed binary outcome, 1 = event (default, SAR, fraud, ...)",
    "score": "model output where HIGHER = RISKIER (probability or score)",
    "pd": "predicted probability of default, in [0, 1]",
    "grade": "rating grade / pool / score band (categorical, ordered by risk if possible)",
    "segment": "segment / portfolio / sub-population",
    "period": "observation period or snapshot date (for time series of a metric)",
    "sample": "sample flag column (e.g. dev / oot / validation, train / test)",
    "reference_value": "value of `sample` that marks the reference (development) sample",
    "current_value": "value of `sample` that marks the current / comparison sample",
    "actual": "observed continuous outcome (realised LGD, CCF, loss, P&L, ...)",
    "predicted": "predicted continuous outcome",
    "weight": "observation weight / exposure for weighted statistics",
    "features": "list of explanatory variable columns",
    "feature": "one explanatory variable column",
    "protected": "protected / sensitive attribute for fairness tests",
    "id": "entity / account / customer identifier",
    "date": "event or observation date",
    "ead": "exposure at default", "limit": "credit limit", "drawn": "drawn amount",
    "pnl": "P&L series (losses negative unless `loss_positive`)",
    "var": "VaR forecast series (positive number = loss amount)",
    "es": "expected-shortfall forecast series (positive number = loss amount)",
    "question": "prompt / question text", "answer": "generated answer text",
    "reference": "ground-truth / reference answer text", "contexts": "retrieved context text",
    "model": "name of a model loaded with load_model",
    "other": "name of a second loaded table (reference data, benchmark, population, ...)",
}


# ── declarations ───────────────────────────────────────────────────────────

PARAM_KINDS = {"column", "columns", "number", "integer", "string", "boolean", "list", "dict",
               "table", "model"}


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                       # one of PARAM_KINDS
    required: bool = True
    default: Any = None
    help: str = ""
    choices: tuple = ()

    def __post_init__(self):
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"Param {self.name}: unknown kind {self.kind}")


def P(name: str, kind: str = "column", help: str = "", default: Any = None,
      required: bool | None = None, choices: tuple = ()) -> Param:
    """Shorthand: a param is optional when it has a default (or required=False)."""
    if required is None:
        required = default is None
    return Param(name, kind, required, default, help or ROLES.get(name, ""), tuple(choices))


@dataclass
class Outcome:
    """What a test function returns. Keep `summary` to headline numbers."""
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list = field(default_factory=list)          # plotly figures
    notes: list[str] = field(default_factory=list)       # caveats, assumptions, small-sample warnings
    rows_used: int | None = None


class NotApplicable(Exception):
    """Raise when the data cannot support the test (e.g. one class only)."""


@dataclass
class RunContext:
    """Everything a test can read besides its parameters."""
    df: pd.DataFrame | None = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    seed: int = DEFAULT_SEED
    judge: Callable[[str, str], str] | None = None       # (system, user) -> text; judge tests only
    source_name: str = ""

    def rng(self, offset: int = 0) -> np.random.Generator:
        """A FRESH generator each call, so a test's randomness never depends on call order elsewhere."""
        return np.random.default_rng(self.seed + offset)


@dataclass(frozen=True)
class TestSpec:
    id: str                          # "<module>.<name>", e.g. "pd.jeffreys"
    name: str
    area: str                        # Discrimination, Calibration, Stability, ...
    model_types: tuple[str, ...]
    params: tuple[Param, ...]
    description: str                 # what it tests, H0, how to read the output
    references: tuple[str, ...]
    fn: Callable
    kind: str = "statistical"        # statistical | judge (LLM-assisted; not bit-reproducible)
    version: str = "1"
    suite: bool = True               # False: needs deliberate inputs; never run by a suite automatically

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)


REGISTRY: dict[str, TestSpec] = {}


def register(id: str, name: str, area: str, model_types, params=(), description: str = "",
             references=(), kind: str = "statistical", version: str = "1", suite: bool = True):
    """Decorator: add a test function to the catalog.

    The function's signature is fn(ctx: RunContext, **params) -> Outcome.
    """
    unknown = [m for m in model_types if m not in MODEL_TYPES]
    if unknown:
        raise ValueError(f"{id}: unknown model types {unknown}")

    def deco(fn):
        if id in REGISTRY and REGISTRY[id].fn.__qualname__ != fn.__qualname__:
            raise ValueError(f"duplicate test id {id}")
        REGISTRY[id] = TestSpec(id, name, area, tuple(model_types), tuple(params),
                                description.strip(), tuple(references), fn, kind, version, suite)
        return fn
    return deco


# ── common helpers for test modules ────────────────────────────────────────

def num(df: pd.DataFrame, col: str) -> pd.Series:
    """A column as float, with non-numeric values turned into NaN."""
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def binary(df: pd.DataFrame, col: str) -> pd.Series:
    """A 0/1 outcome column. Accepts bool, 0/1, 'Y'/'N', 'yes'/'no', 'true'/'false'."""
    s = df[col]
    if s.dtype == bool:
        return s.astype(float)
    if s.dtype == object or pd.api.types.is_string_dtype(s):     # pandas 3: str dtype, not object
        m = s.astype(str).str.strip().str.lower().map(
            {"1": 1.0, "0": 0.0, "y": 1.0, "n": 0.0, "yes": 1.0, "no": 0.0, "true": 1.0,
             "false": 0.0, "1.0": 1.0, "0.0": 0.0})
        if m.notna().sum() >= s.notna().sum() * 0.99:
            return m
    out = pd.to_numeric(s, errors="coerce").astype(float)
    vals = set(out.dropna().unique())
    if not vals <= {0.0, 1.0}:
        raise ValueError(f"Column '{col}' is not binary 0/1 (values like {sorted(vals)[:6]})")
    return out


def complete(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """Rows with no missing value in `cols`, and how many rows were dropped."""
    cols = [c for c in cols if c]
    sub = df[cols].dropna()
    return sub, len(df) - len(sub)


def dropped_note(n: int, what: str = "rows with missing inputs") -> list[str]:
    return [f"{n} {what} excluded."] if n else []


def require_two_classes(y: pd.Series | np.ndarray, label: str = "target") -> None:
    u = np.unique(np.asarray(y)[~pd.isna(np.asarray(y))])
    if len(u) < 2:
        raise NotApplicable(f"{label} has a single class ({u.tolist()}); the test needs both 0 and 1.")


def split_samples(df: pd.DataFrame, sample: str, reference_value=None, current_value=None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """(reference rows, current rows, ref label, cur label) from a sample column.

    With no values given and exactly two distinct values, the first in sorted
    order is the reference (so 'dev' < 'oot', 'test' > 'train' needs explicit values;
    we special-case common names)."""
    vals = [v for v in pd.unique(df[sample].dropna())]
    def _match(v):
        return [x for x in vals if str(x).lower() == str(v).lower()]
    if reference_value is not None and current_value is not None:
        r, c = _match(reference_value), _match(current_value)
        if not r or not c:
            raise ValueError(f"'{sample}' values are {vals[:10]}; got reference={reference_value}, current={current_value}")
        ref, cur = r[0], c[0]
    else:
        if len(vals) != 2 and reference_value is None:
            raise ValueError(f"'{sample}' has {len(vals)} values {vals[:10]}; pass reference_value and current_value.")
        pref = ["train", "training", "dev", "development", "ref", "reference", "insample", "in_sample",
                "build", "base", "baseline"]
        if reference_value is not None:
            ref = _match(reference_value)[0]
        else:
            ref = next((v for v in vals if str(v).lower() in pref), sorted(vals, key=str)[0])
        rest = [v for v in vals if v != ref]
        if current_value is not None:
            cur = _match(current_value)[0]
        elif len(rest) == 1:
            cur = rest[0]
        else:
            raise ValueError(f"'{sample}' has values {vals[:10]}; pass current_value.")
    return df[df[sample] == ref], df[df[sample] == cur], str(ref), str(cur)


def table_or_none(ctx: RunContext, name: str | None) -> pd.DataFrame | None:
    return ctx.tables.get(name) if name else None


def fmt(x: Any, digits: int = 4) -> Any:
    """Round floats for display; leave everything else."""
    if isinstance(x, (float, np.floating)):
        if math.isnan(x) or math.isinf(x):
            return float(x)
        return round(float(x), digits)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ── results and provenance ─────────────────────────────────────────────────

@dataclass
class TestResult:
    test_id: str
    test_name: str
    area: str
    kind: str
    status: str                      # ok | not_applicable | error
    run_id: str
    params: dict
    source: str
    data_fingerprint: str
    rows_in: int
    rows_used: int | None
    seed: int
    versions: dict
    started: str
    seconds: float
    summary: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str = ""
    references: tuple[str, ...] = ()

    def provenance(self) -> str:
        return (f"run_id={self.run_id} · test={self.test_id} v{self.versions.get('test')} · "
                f"source={self.source or '-'} · data sha256={self.data_fingerprint} · "
                f"rows in/used={self.rows_in}/{self.rows_used if self.rows_used is not None else '-'} · "
                f"seed={self.seed} · code {self.versions.get('code')} · library v{self.versions.get('library')}")

    def to_text(self, max_rows: int = 30) -> str:
        head = f"[test:{self.run_id}] {self.test_name} ({self.test_id}) — {self.status.upper()}"
        if self.kind == "judge":
            head += " — LLM-JUDGE-BASED: not a statistical test; outputs cached by prompt hash"
        parts = [head]
        if self.error:
            parts.append(f"Reason: {self.error}")
        if self.summary:
            parts.append("Results: " + "; ".join(f"{k} = {fmt(v)}" for k, v in self.summary.items()))
        for name, t in self.tables.items():
            with pd.option_context("display.width", 220, "display.max_columns", 30,
                                   "display.float_format", lambda v: f"{v:.6g}"):
                body = t.head(max_rows).to_string(index=False)
            more = f"\n… ({len(t) - max_rows} more rows)" if len(t) > max_rows else ""
            parts.append(f"--- {name} ({len(t)} rows) ---\n{body}{more}")
        if self.notes:
            parts.append("Notes: " + " ".join(self.notes))
        parts.append("Provenance: " + self.provenance())
        return "\n".join(parts)

    def to_json(self) -> dict:
        return {
            "run_id": self.run_id, "test_id": self.test_id, "test_name": self.test_name,
            "area": self.area, "kind": self.kind, "status": self.status, "params": self.params,
            "source": self.source, "data_fingerprint": self.data_fingerprint,
            "rows_in": self.rows_in, "rows_used": self.rows_used, "seed": self.seed,
            "versions": self.versions, "started": self.started, "seconds": self.seconds,
            "summary": {k: _jsonable(v) for k, v in self.summary.items()},
            "tables": {k: json.loads(v.to_json(orient="split", date_format="iso", default_handler=str))
                       for k, v in self.tables.items()},
            "notes": self.notes, "error": self.error, "references": list(self.references),
        }


def _jsonable(v: Any) -> Any:
    v = fmt(v, 10)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def fingerprint(df: pd.DataFrame | None, cols: list[str] | None = None,
                extra: list[pd.DataFrame] | None = None) -> str:
    """SHA-256 over the exact values a test reads (selected columns, row order kept)."""
    h = hashlib.sha256()
    frames = []
    if df is not None:
        frames.append(df[cols] if cols else df)
    frames += [t for t in (extra or []) if t is not None]
    for f in frames:
        h.update(",".join(map(str, f.columns)).encode())
        try:
            h.update(pd.util.hash_pandas_object(f, index=False).values.tobytes())
        except TypeError:          # unhashable cells (lists/dicts): hash their text
            h.update(pd.util.hash_pandas_object(f.astype(str), index=False).values.tobytes())
    return h.hexdigest()[:16]


_CODE_HASHES: dict[str, str] = {}


def code_hash(spec: TestSpec) -> str:
    """SHA-256 of the source of the test's module and of this core module, so a run_id
    changes whenever the code that produced the numbers changes."""
    import inspect
    import sys
    mod = spec.fn.__module__
    if mod not in _CODE_HASHES:
        h = hashlib.sha256()
        for m in (sys.modules[mod], sys.modules[__name__]):
            try:
                h.update(inspect.getsource(m).replace("\r\n", "\n").encode())
            except (OSError, TypeError):
                h.update(m.__name__.encode())
        _CODE_HASHES[mod] = h.hexdigest()[:12]
    return _CODE_HASHES[mod]


def _versions(spec: TestSpec) -> dict:
    import scipy
    out = {"library": LIBRARY_VERSION, "test": spec.version, "code": code_hash(spec), "numpy": np.__version__,
           "pandas": pd.__version__, "scipy": scipy.__version__}
    try:
        import sklearn
        out["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return out


def _coerce(spec: TestSpec, raw: dict, ctx: RunContext) -> tuple[dict, list[str], list[pd.DataFrame]]:
    """Validate parameters. Returns (kwargs, data columns read, other tables read)."""
    unknown = set(raw) - {p.name for p in spec.params}
    if unknown:
        raise ValueError(f"{spec.id} has no parameter(s) {sorted(unknown)}. "
                         f"Parameters: {', '.join(p.name for p in spec.params)}")
    df = ctx.df
    cols = list(df.columns) if df is not None else []
    out, used, others = {}, [], []

    def _col(name: str, v: str) -> str:
        if v in cols:
            return v
        low = {str(c).lower(): c for c in cols}
        if str(v).lower() in low:
            return low[str(v).lower()]
        raise KeyError(f"Column '{v}' (for {name}) is not in the table. Columns: "
                       f"{', '.join(map(str, cols[:60]))}")

    for p in spec.params:
        if p.name not in raw or raw[p.name] is None:
            if p.required:
                raise ValueError(f"{spec.id} needs '{p.name}' ({p.kind}): {p.help}")
            out[p.name] = p.default
            continue
        v = raw[p.name]
        if p.kind == "column":
            if df is None:
                raise ValueError(f"{spec.id} needs a data table")
            v = _col(p.name, str(v))
            used.append(v)
        elif p.kind == "columns":
            if isinstance(v, str):
                v = [s.strip() for s in v.split(",") if s.strip()]
            v = [_col(p.name, str(c)) for c in v]
            used += v
        elif p.kind == "number":
            v = float(v)
        elif p.kind == "integer":
            v = int(v)
        elif p.kind == "boolean":
            v = v if isinstance(v, bool) else str(v).lower() in {"true", "1", "yes"}
        elif p.kind == "string":
            v = str(v)
        elif p.kind == "list":
            v = list(v) if isinstance(v, (list, tuple)) else [s.strip() for s in str(v).split(",")]
        elif p.kind == "dict":
            if isinstance(v, str):
                v = json.loads(v)
            if not isinstance(v, dict):
                raise ValueError(f"'{p.name}' must be an object")
        elif p.kind == "table":
            if v not in ctx.tables:
                match = [k for k in ctx.tables if str(v).lower() in k.lower()]
                if len(match) != 1:
                    raise KeyError(f"No loaded table '{v}' for {p.name}. Tables: {', '.join(ctx.tables)}")
                v = match[0]
            others.append(ctx.tables[v])
        elif p.kind == "model":
            if v not in ctx.models:
                raise KeyError(f"No loaded model '{v}'. Load it with load_model first. "
                               f"Loaded: {', '.join(ctx.models) or 'none'}")
        if p.choices and v not in p.choices:
            raise ValueError(f"'{p.name}' must be one of {list(p.choices)}, got {v!r}")
        out[p.name] = v
    return out, list(dict.fromkeys(used)), others


def run_test(test_id: str, ctx: RunContext, params: dict | None = None) -> TestResult:
    """Run one registered test. Never raises for test-level problems: they come back
    as status error / not_applicable so the agent can report them."""
    spec = get(test_id)
    raw = {k: v for k, v in (params or {}).items() if v is not None}
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    versions = _versions(spec)
    try:
        kwargs, used, others = _coerce(spec, raw, ctx)
    except (KeyError, ValueError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return TestResult(spec.id, spec.name, spec.area, spec.kind, "error", "-", raw, ctx.source_name,
                          "-", 0 if ctx.df is None else len(ctx.df), None, ctx.seed, versions, started,
                          0.0, error=f"Invalid parameters: {msg}", references=spec.references)
    needs_data = any(p.kind in {"column", "columns"} for p in spec.params)
    fp = fingerprint(ctx.df if needs_data else None, used or None, others)
    for p in spec.params:                 # a model's identity is part of the inputs
        if p.kind == "model" and kwargs.get(p.name):
            m = ctx.models[kwargs[p.name]]
            versions[f"model:{kwargs[p.name]}"] = getattr(m, "sha256", "") or getattr(m, "location", "")
    if spec.kind == "judge":
        versions["judge_model"] = getattr(ctx.judge, "model", "unknown") if ctx.judge else "none"
    canon = json.dumps({k: _jsonable(v) for k, v in sorted(kwargs.items())}, sort_keys=True, default=str)
    extra = json.dumps({k: v for k, v in versions.items() if k.startswith(("model:", "judge_model", "code"))},
                       sort_keys=True)
    run_id = hashlib.sha256(f"{spec.id}|{spec.version}|{LIBRARY_VERSION}|{canon}|{fp}|{ctx.seed}|{extra}"
                            .encode()).hexdigest()[:12]
    rows_in = 0 if ctx.df is None else len(ctx.df)
    res = TestResult(spec.id, spec.name, spec.area, spec.kind, "ok", run_id,
                     {k: _jsonable(v) for k, v in kwargs.items()}, ctx.source_name, fp,
                     rows_in, None, ctx.seed, versions, started, 0.0, references=spec.references)
    try:
        with np.errstate(all="ignore"):
            out = spec.fn(ctx, **kwargs)
        if not isinstance(out, Outcome):
            raise TypeError(f"{spec.id} returned {type(out).__name__}, not Outcome")
        res.summary, res.tables, res.figures = out.summary, out.tables, out.figures
        res.notes, res.rows_used = list(out.notes), out.rows_used
    except NotApplicable as exc:
        res.status, res.error = "not_applicable", str(exc)
    except Exception as exc:          # a test bug or bad data must not kill the turn
        res.status, res.error = "error", f"{type(exc).__name__}: {exc}"
    res.seconds = round(time.perf_counter() - t0, 3)
    return res


def get(test_id: str) -> TestSpec:
    load_all()
    if test_id in REGISTRY:
        return REGISTRY[test_id]
    low = test_id.lower().strip()
    hits = [k for k in REGISTRY if k.lower() == low or k.lower().endswith("." + low)]
    if len(hits) == 1:
        return REGISTRY[hits[0]]
    near = [k for k in REGISTRY if low in k.lower() or low in REGISTRY[k].name.lower()][:12]
    raise KeyError(f"Unknown test '{test_id}'." + (f" Did you mean: {', '.join(near)}?" if near
                                                   else " Use list_validation_tests."))


_LOADED = False


def load_all() -> None:
    """Import every test module once so their @register calls run."""
    global _LOADED
    if _LOADED:
        return
    import importlib
    import pkgutil

    import ask.validation as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if m.name.startswith("t_"):
            importlib.import_module(f"ask.validation.{m.name}")
    _LOADED = True


def catalog(model_type: str | None = None, area: str | None = None, query: str | None = None
            ) -> list[TestSpec]:
    load_all()
    out = []
    for s in REGISTRY.values():
        if model_type and model_type not in s.model_types and "general" not in s.model_types:
            continue
        if area and area.lower() not in s.area.lower():
            continue
        if query:
            hay = f"{s.id} {s.name} {s.area} {s.description}".lower()
            if not all(w in hay for w in query.lower().split()):
                continue
        out.append(s)
    return sorted(out, key=lambda s: (s.area, s.id))


def describe(spec: TestSpec) -> str:
    lines = [f"{spec.id} — {spec.name}", f"Area: {spec.area} · Model types: {', '.join(spec.model_types)}"
             + (" · LLM-judge-based" if spec.kind == "judge" else "")]
    if spec.description:
        lines.append(spec.description)
    lines.append("Parameters:")
    for p in spec.params:
        req = "required" if p.required else f"optional, default {p.default!r}"
        ch = f" one of {list(p.choices)}" if p.choices else ""
        lines.append(f"  - {p.name} ({p.kind}, {req}){ch}: {p.help}")
    if spec.references:
        lines.append("References: " + "; ".join(spec.references))
    return "\n".join(lines)
