"""Data review: profile, missingness, duplicates, validity, outliers, data types,
target definition, representativeness, time coverage, reconciliation, referential
integrity, leakage candidates, time consistency and univariate target association.

Every finding carries its evidence: the column, the count and example rows, so a
validator can write "table X, column Y, rows ...". Row numbers are 0-based
POSITIONS in the table as loaded (not index labels); when `id` is given, that
column's values are quoted next to them. Examples are always the first rows in
table order, so they are deterministic. Detailed evidence is returned as long
tables with one row per (check, column, example row).

No pass/fail thresholds. Where a check needs a cut-off by its definition (IQR
fence k, z-score cut-off, near-constant share), it is a parameter and the output
is a count, not a verdict. Name-based checks (probability-like or amount-like
column names, id-like columns) are heuristics and are labelled as such.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, num, register
from ask.validation.t_stability import _is_categorical, psi_table

_ALL = ("general", "pd", "lgd", "ead", "ifrs9", "ews", "aml", "satellite", "pricing", "ccr", "var",
        "ml_classification", "ml_regression", "ml_unsupervised")
_BINARY = ("general", "pd", "ifrs9", "ews", "aml", "ml_classification")

_EVID = (P("id", required=False, help="Identifier column whose values are quoted next to row numbers"),
         P("max_examples", "integer", default=10, help="Example rows listed per finding"))
_DAYFIRST = P("dayfirst", "boolean", default=False,
              help="Parse ambiguous text dates such as 01/02/2024 as day-first")

ROW_NOTE = ("Row numbers are 0-based positions in the table as loaded; examples are the first rows "
            "in table order.")


# ── evidence helpers ────────────────────────────────────────────────────────

def _pos(mask) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def _join(vals, k: int, total: int | None = None) -> str:
    vals = list(vals)
    total = len(vals) if total is None else total
    s = ", ".join(str(v) for v in vals[:k])
    return s + (f" (+{total - k} more)" if total > k else "")


def _ex(df: pd.DataFrame, pos, id, k: int) -> dict:
    """{'example_rows': '3, 7', 'example_ids': 'A3, A7'} for the first k positions."""
    pos = np.asarray(pos, dtype=int)
    out = {"example_rows": _join(pos, k)}
    if id:
        out["example_ids"] = _join(df[id].iloc[pos[:k]].tolist(), k, len(pos))
    return out


def _show(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return "<missing>"
    if isinstance(v, str):
        return repr(v)                 # quotes make leading/trailing whitespace visible
    return str(v)


def _evidence(df: pd.DataFrame, pos, k: int, check: str, column: str, id=None,
              values=None) -> list[dict]:
    pos = np.asarray(pos, dtype=int)[:k]
    vals = (list(values)[:k] if values is not None
            else df[column].iloc[pos].tolist() if column in df.columns else [""] * len(pos))
    rows = []
    for p, v in zip(pos, vals):
        r = {"check": check, "column": column, "row": int(p)}
        if id:
            r["id"] = df[id].iloc[p]
        r["value"] = _show(v) if not isinstance(v, str) or values is None else v
        rows.append(r)
    return rows


def _ev_frame(rows: list[dict], id=None) -> pd.DataFrame:
    cols = ["check", "column", "row"] + (["id"] if id else []) + ["value"]
    return pd.DataFrame(rows, columns=cols)


def _match_col(df: pd.DataFrame, name: str, what: str) -> str:
    if name in df.columns:
        return name
    low = {str(c).lower(): c for c in df.columns}
    if str(name).lower() in low:
        return low[str(name).lower()]
    raise ValueError(f"Column '{name}' ({what}) is not in the table. Columns: "
                     f"{', '.join(map(str, list(df.columns)[:60]))}")


# ── type helpers ────────────────────────────────────────────────────────────

_ID_NAME = re.compile(r"(^id$|^id_|_id$|^key$|_key$|_no$|_nr$|number$|^uuid|guid)", re.I)
_BOOL_TOKENS = {"0", "1", "0.0", "1.0", "true", "false", "y", "n", "yes", "no", "t", "f"}
_BIN_MAP = {"1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0, "y": 1.0, "n": 0.0, "yes": 1.0, "no": 0.0,
            "true": 1.0, "false": 0.0, "t": 1.0, "f": 0.0}
_DATE_RE = re.compile(
    r"^\s*(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{1,2}[ -][A-Za-z]{3,9}[ -,]*\d{2,4}|[A-Za-z]{3,9} \d{1,2},? \d{4})"
    r"([ T]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?\s*$")


def _is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s))


def _text_values(s: pd.Series) -> pd.Series:
    """Non-null values as str (object/str columns), keeping the original index."""
    x = s.dropna()
    return x.astype(str)


def _date_like_share(s: pd.Series) -> float:
    if pd.api.types.is_datetime64_any_dtype(s):
        return 1.0
    if not _is_text(s):
        return 0.0
    x = _text_values(s)
    return float(x.str.match(_DATE_RE).mean()) if len(x) else 0.0


def _numeric_parse(s: pd.Series) -> pd.Series:
    """Text values parsed as numbers after stripping whitespace (NaN where not a number)."""
    return pd.to_numeric(s.astype("object").where(s.notna()).map(
        lambda v: v.strip() if isinstance(v, str) else v), errors="coerce").astype(float)


def _to_dates(s: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Datetime series; ISO strings first, then dateutil (month-first unless dayfirst)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(None)
        return s
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        x = num(s.to_frame("v"), "v")
        ok = x.dropna()
        ints = ok[(ok == np.floor(ok))].astype("int64").astype(str)
        if len(ok) and len(ints) == len(ok):
            lens = set(ints.str.len())
            fmt_ = {8: "%Y%m%d", 6: "%Y%m", 4: "%Y"}.get(lens.pop()) if len(lens) == 1 else None
            if fmt_:
                out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
                out[ints.index] = pd.to_datetime(ints, format=fmt_, errors="coerce")
                return out
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[us]")
    t = s.astype("object").where(s.notna())
    t = t.map(lambda v: v.strip() if isinstance(v, str) else v)
    out = pd.to_datetime(t, errors="coerce", format="ISO8601")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert(None)
    rest = out.isna() & t.notna()
    if rest.any():
        out = out.astype("datetime64[us]")
        out[rest] = pd.to_datetime(t[rest], errors="coerce", format="mixed", dayfirst=dayfirst)
    return out.astype("datetime64[us]")


def _is_date_col(s: pd.Series, share: float = 0.9) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or (_is_text(s) and s.notna().any()
                                                       and _date_like_share(s) >= share)


def _semantic(s: pd.Series, name) -> str:
    """Inferred semantic type (heuristic, documented in data.profile)."""
    x = s.dropna()
    if len(x) == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    nun = x.nunique()
    if nun <= 2 and set(x.astype(str).str.strip().str.lower()) <= _BOOL_TOKENS:
        return "boolean"
    if _ID_NAME.search(str(name)) and nun >= 0.95 * len(x) and len(x) > 1:
        return "id-like"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    t = x.astype(str)
    if t.str.match(_DATE_RE).mean() >= 0.9:
        return "date"
    if _numeric_parse(x).notna().mean() >= 0.9:
        return "numeric-as-text"
    has_space = t.str.strip().str.contains(" ").mean()
    if nun == len(x) and len(x) >= 10 and has_space < 0.5:
        return "id-like"
    if t.str.len().mean() > 30 or (has_space > 0.5 and nun > 0.5 * len(x)):
        return "text"
    return "categorical"


def _binary(df: pd.DataFrame, col: str) -> pd.Series:
    """0/1 target as float (NaN for missing). Accepts bool, 0/1 and y/n, yes/no, true/false text."""
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float)
    else:
        out = s.astype("object").map(
            lambda v: _BIN_MAP.get(str(v).strip().lower(), -1.0) if pd.notna(v) else np.nan).astype(float)
    bad = out.notna() & ~out.isin([0.0, 1.0])
    if bad.any():
        raise ValueError(f"Target '{col}' is not binary 0/1: e.g. {s[bad].head(5).tolist()} at rows "
                         f"{_join(_pos(bad), 5)}")
    return out


def _canon_key(s: pd.Series) -> pd.Series:
    """Key values as comparable strings: integer-valued numbers lose '.0', text kept verbatim."""
    out = pd.Series(np.nan, index=s.index, dtype="object")
    nn = s.notna()
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        v = s[nn].astype(float)
        out[nn] = [str(int(a)) if float(a).is_integer() else repr(float(a)) for a in v]
    else:
        out[nn] = s[nn].astype(str)
    return out


def _composite(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [_canon_key(df[c]) for c in cols]
    miss = pd.concat([p.isna() for p in parts], axis=1).any(axis=1)
    key = parts[0].fillna("")
    for p in parts[1:]:
        key = key + "\x1f" + p.fillna("")
    return key.where(~miss)


def _auc(y: np.ndarray, x: np.ndarray) -> float:
    """Mann–Whitney AUC = P(x_event > x_non-event) + 0.5·P(tie)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float, float, int]:
    """(V, chi2, p, dof) without continuity correction; missing values form their own level."""
    ct = pd.crosstab(a.astype("object").where(a.notna(), "<missing>").astype(str),
                     b.astype("object").where(b.notna(), "<missing>").astype(str))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan"), float("nan"), float("nan"), 0
    chi2, p, dof, _ = stats.chi2_contingency(ct.to_numpy(), correction=False)
    n = ct.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))), float(chi2), float(p), int(dof)


def _period_series(s: pd.Series, freq: str, dayfirst: bool = False):
    """Period-typed series, or None when the column cannot be read as dates."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").dropna()
        if len(v) and (v == v.round()).all() and v.between(1800, 2200).all():
            return None              # calendar years: handled as integer periods, not as dates
    d = _to_dates(s, dayfirst)
    nn = int(s.notna().sum())
    if nn == 0 or d.notna().sum() < 0.9 * nn:
        return None
    return d.dt.to_period(freq)


# ── profile ─────────────────────────────────────────────────────────────────

@register("data.profile", "Column profile (types, missing, distinct, distribution)", "Data quality", _ALL,
          params=(P("columns", "columns", required=False, help="Columns to profile; default all"),
                  P("top_n", "integer", default=5, help="Most frequent values listed per column"),
                  P("near_constant", "number", default=0.95,
                    help="Mode share at or above which a column is marked near-constant (descriptive)"),
                  P("high_cardinality", "integer", default=50,
                    help="Distinct-value count above which a categorical/text/id column is marked high-cardinality")),
          description="""One row per column: storage dtype, inferred semantic type, count, missing count and %,
distinct values, most frequent values with shares, mode share, constant / near-constant flags (mode share
>= near_constant), high-cardinality flag, and for numeric columns (and numbers stored as text, after parsing)
min, max, mean, standard deviation (ddof=1), quantiles (1/5/25/50/75/95/99 %, linear interpolation),
zero count and negative count.
Semantic type is a heuristic: boolean = at most two values from {0,1,true,false,y,n,yes,no,t,f};
date = datetime dtype or >= 90% of text values look like dates; id-like = name looks like an identifier
(id, *_id, *_key, *_no, *number) with >= 95% distinct values, or text with all values distinct, >= 10 values
and no spaces; numeric-as-text = >= 90% of text values parse as numbers; text = long strings or mostly
multi-word free text; otherwise categorical. Read the flags as descriptions, not findings.""",
          references=("ECB Guide to internal models (2024), general topics: data quality",
                      "BCBS 239 (2013), Principles for effective risk data aggregation and risk reporting",
                      "Siddiqi (2006), Credit Risk Scorecards, ch. 6 (data review)"))
def profile(ctx: RunContext, columns=None, top_n=5, near_constant=0.95, high_cardinality=50) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    cols = columns or list(df.columns)
    if not cols:
        raise NotApplicable("The table has no columns.")
    n = len(df)
    rows = []
    for c in cols:
        s = df[c]
        x = s.dropna()
        sem = _semantic(s, c)
        nun = int(x.nunique())
        vc = x.astype(str).value_counts() if len(x) else pd.Series(dtype=int)
        vc = vc.reset_index()
        vc.columns = ["v", "n"]
        vc = vc.sort_values(["n", "v"], ascending=[False, True], kind="mergesort")
        top = "; ".join(f"{v}: {k} ({k / len(x):.1%})" for v, k in zip(vc["v"].head(top_n), vc["n"].head(top_n)))
        mode_share = float(vc["n"].iloc[0] / len(x)) if len(x) else float("nan")
        r = {"column": c, "dtype": str(s.dtype), "semantic_type": sem, "n": n,
             "missing": int(s.isna().sum()), "missing_pct": float(s.isna().mean()) if n else float("nan"),
             "distinct": nun, "distinct_pct": nun / len(x) if len(x) else float("nan"),
             "top_values": top, "mode_share": mode_share, "constant": nun <= 1,
             "near_constant": bool(mode_share >= near_constant) if len(x) else False,
             "high_cardinality": bool(sem in {"categorical", "text", "id-like"} and nun > high_cardinality)}
        if sem in {"numeric", "numeric-as-text"} or (sem in {"boolean", "id-like"}
                                                     and pd.api.types.is_numeric_dtype(s)
                                                     and not pd.api.types.is_bool_dtype(s)):
            v = (s.astype(float) if pd.api.types.is_numeric_dtype(s) else _numeric_parse(s)).dropna()
            if len(v):
                q = np.quantile(v, [.01, .05, .25, .5, .75, .95, .99])
                r.update({"min": float(v.min()), "p1": q[0], "p5": q[1], "p25": q[2], "median": q[3],
                          "p75": q[4], "p95": q[5], "p99": q[6], "max": float(v.max()),
                          "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                          "zeros": int((v == 0).sum()), "negatives": int((v < 0).sum())})
        elif sem == "date":
            d = _to_dates(s).dropna()
            if len(d):
                r.update({"min": str(d.min().date()), "max": str(d.max().date())})
        rows.append(r)
    out = pd.DataFrame(rows)
    return Outcome({"rows": n, "columns": len(cols),
                    "columns_with_missing": int((out["missing"] > 0).sum()),
                    "constant_columns": int(out["constant"].sum()),
                    "near_constant_columns": int(out["near_constant"].sum()),
                    "high_cardinality_columns": int(out["high_cardinality"].sum()),
                    "total_missing_cells": int(out["missing"].sum())},
                   {"Column profile": out}, rows_used=n,
                   notes=["Semantic types and the constant / near-constant / high-cardinality flags are "
                          "heuristic descriptions; check them against the data dictionary.",
                          "Statistics for numbers stored as text are computed after parsing; values that do not "
                          "parse are ignored there (see data.type_consistency)."])


# ── missingness ─────────────────────────────────────────────────────────────

def _homogeneity(miss: pd.Series, group: pd.Series) -> tuple[float, float, int]:
    ct = pd.crosstab(group.astype(str), miss)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan"), float("nan"), 0
    chi2, p, dof, _ = stats.chi2_contingency(ct.to_numpy(), correction=False)
    return float(chi2), float(p), int(dof)


@register("data.missingness", "Missing values by column, period and segment", "Data quality", _ALL,
          params=(P("columns", "columns", required=False, help="Columns to check; default all"),
                  P("period", required=False), P("segment", required=False), *_EVID),
          description="""Count and share of missing values (NaN/None/NaT) per column, with example rows. Blank or
whitespace-only text values are counted separately (they are not missing to pandas but usually are to the
business). With `period` and/or `segment`: the missing share of every column with missing values per period
/ segment, and a chi-square test of homogeneity of the missing rate across periods / segments
(H0: the probability of being missing is the same in every period / segment; Pearson chi-square on the
missing-vs-present × group table, no continuity correction).""",
          references=("Little & Rubin (2019), Statistical Analysis with Missing Data, 3rd ed., ch. 1",
                      "ECB Guide to internal models (2024), general topics: data quality"))
def missingness(ctx: RunContext, columns=None, period=None, segment=None, id=None, max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    cols = columns or [c for c in df.columns if c not in {period, segment, id}]
    n = len(df)
    if n == 0:
        raise NotApplicable("The table is empty.")
    rows, ev = [], []
    for c in cols:
        m = df[c].isna()
        blank = pd.Series(False, index=df.index)
        if _is_text(df[c]):
            blank = df[c].astype("object").map(lambda v: isinstance(v, str) and v.strip() == "")
        pm, pb = _pos(m), _pos(blank)
        r = {"column": c, "n": n, "missing": len(pm), "missing_pct": len(pm) / n,
             "blank_text": len(pb)}
        r.update(_ex(df, pm, id, max_examples) if len(pm) else {"example_rows": ""})
        rows.append(r)
        ev += _evidence(df, pm, max_examples, "missing", c, id)
        ev += _evidence(df, pb, max_examples, "blank text", c, id)
    out = pd.DataFrame(rows).sort_values(["missing", "column"], ascending=[False, True], kind="mergesort")
    tables = {"Missing by column": out}
    with_missing = [c for c in cols if df[c].isna().any()]
    tests = []
    for by, label in ((period, "period"), (segment, "segment")):
        if not by or not with_missing:
            continue
        g = df[by].astype("object").where(df[by].notna(), "<missing>").astype(str)
        long = []
        for c in with_missing:
            t = df[c].isna().groupby(g).agg(["size", "sum"]).reset_index()
            t.columns = [label, "n", "missing"]
            t.insert(0, "column", c)
            t["missing_pct"] = t["missing"] / t["n"]
            long.append(t)
            chi2, p, dof = _homogeneity(df[c].isna(), g)
            tests.append({"column": c, "by": label, "chi2": chi2, "dof": dof, "p_value": p})
        tables[f"Missing by {label}"] = pd.concat(long, ignore_index=True)
    if tests:
        tables["Missing-rate homogeneity tests"] = pd.DataFrame(tests)
    tables["Evidence"] = _ev_frame(ev, id)
    figs = []
    if with_missing:
        import plotly.express as px
        top = out[out["missing"] > 0].head(30)
        figs.append(px.bar(top, x="column", y="missing_pct", title="Share missing by column"))
    return Outcome({"rows": n, "columns_checked": len(cols), "columns_with_missing": len(with_missing),
                    "missing_cells": int(out["missing"].sum()),
                    "missing_cell_pct": float(out["missing"].sum() / (n * len(cols))),
                    "rows_with_any_missing": int(df[cols].isna().any(axis=1).sum()),
                    "blank_text_cells": int(out["blank_text"].sum())},
                   tables, figs, [ROW_NOTE], rows_used=n)


@register("data.missing_patterns", "Missingness pattern combinations", "Data quality", _ALL,
          params=(P("columns", "columns", required=False, help="Columns; default those with any missing value"),
                  P("max_patterns", "integer", default=20), *_EVID),
          description="""Groups rows by WHICH of the columns are missing together (the missingness pattern) and
counts rows per pattern, most frequent first, with example rows. Shows whether values go missing jointly
(e.g. one source system not delivering a block of fields) or independently, and whether the data are
monotone-missing.""",
          references=("Little & Rubin (2019), Statistical Analysis with Missing Data, 3rd ed., ch. 1.3",
                      "van Buuren (2018), Flexible Imputation of Missing Data, 2nd ed., ch. 4.1"))
def missing_patterns(ctx: RunContext, columns=None, max_patterns=20, id=None, max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    cols = columns or [c for c in df.columns if df[c].isna().any()]
    if not cols:
        raise NotApplicable("No column has missing values.")
    M = df[cols].isna()
    pats, inv = np.unique(M.to_numpy(), axis=0, return_inverse=True)
    inv = inv.ravel()
    rows = []
    for g, pat in enumerate(pats):
        pos = np.flatnonzero(inv == g)
        lab = ", ".join(c for c, v in zip(cols, pat) if v) or "<complete>"
        rows.append({"missing_columns": lab, "n_missing_columns": int(pat.sum()),
                     "rows": len(pos), "pct": len(pos) / len(df), "first_row": int(pos[0]),
                     **_ex(df, pos, id, max_examples)})
    out = pd.DataFrame(rows).sort_values(["rows", "first_row"], ascending=[False, True], kind="mergesort")
    shown = out.head(max_patterns).drop(columns="first_row")
    notes = [ROW_NOTE]
    if len(out) > max_patterns:
        notes.append(f"{len(out) - max_patterns} rarer patterns not listed.")
    return Outcome({"patterns": len(out), "complete_rows": int((~M.any(axis=1)).sum()),
                    "rows_with_missing": int(M.any(axis=1).sum()),
                    "most_common_incomplete_pattern": next((r for r in out["missing_columns"]
                                                            if r != "<complete>"), "")},
                   {"Missingness patterns": shown}, notes=notes, rows_used=len(df))


@register("data.missing_vs_target", "Missingness vs target (event rate when missing vs present)",
          "Data quality", _BINARY,
          params=(P("target"), P("columns", "columns", required=False,
                                 help="Columns; default all with missing values except the target"), *_EVID),
          description="""For every column with missing values: event (default) rate among rows where the column is
missing vs where it is present, the difference, Fisher's exact test (two-sided) and Pearson chi-square
(no continuity correction) on the 2×2 missing × target table. H0: missingness is independent of the target.
A dependence means the data are not MCAR with respect to the outcome and that the treatment of missing values
matters for the model (missing is informative). Rows with a missing target are excluded.""",
          references=("Little & Rubin (2019), Statistical Analysis with Missing Data, 3rd ed.",
                      "Fisher (1922), On the interpretation of chi-square from contingency tables, JRSS 85"))
def missing_vs_target(ctx: RunContext, target, columns=None, id=None, max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    y = _binary(df, target)
    keep = y.notna()
    cols = columns or [c for c in df.columns if c != target and df.loc[keep, c].isna().any()]
    cols = [c for c in cols if c != target]
    if not cols:
        raise NotApplicable("No column has missing values among rows with a known target.")
    yk = y[keep]
    rows = []
    for c in cols:
        m = df.loc[keep, c].isna()
        a, b = int((m & (yk == 1)).sum()), int((m & (yk == 0)).sum())
        cc, d = int((~m & (yk == 1)).sum()), int((~m & (yk == 0)).sum())
        tab = np.array([[a, b], [cc, d]])
        if (a + b) == 0 or (cc + d) == 0:
            fp = chi2 = p = float("nan")
        else:
            fp = float(stats.fisher_exact(tab).pvalue)
            if tab.sum(axis=0).min() > 0:
                chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
            else:
                chi2, p = float("nan"), float("nan")
        pos = _pos(df[c].isna() & keep)
        rows.append({"column": c, "n_missing": a + b, "events_missing": a,
                     "event_rate_missing": a / (a + b) if a + b else float("nan"),
                     "n_present": cc + d, "events_present": cc,
                     "event_rate_present": cc / (cc + d) if cc + d else float("nan"),
                     "difference": (a / (a + b) if a + b else np.nan) - (cc / (cc + d) if cc + d else np.nan),
                     "fisher_p_value": fp, "chi2": float(chi2), "chi2_p_value": float(p),
                     **_ex(df, pos, id, max_examples)})
    out = pd.DataFrame(rows)
    i = out["fisher_p_value"].fillna(2).idxmin()
    return Outcome({"columns": len(out), "n": int(keep.sum()), "overall_event_rate": float(yk.mean()),
                    "smallest_fisher_p_value": float(out.loc[i, "fisher_p_value"]),
                    "smallest_p_column": out.loc[i, "column"]},
                   {"Missingness vs target": out},
                   notes=[ROW_NOTE, "p-values are per column, not adjusted for multiple testing."]
                   + ([f"{int((~keep).sum())} rows with missing target excluded."] if (~keep).any() else []),
                   rows_used=int(keep.sum()))


def _em_mvn(X: np.ndarray, max_iter: int = 2000, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray, int]:
    """ML mean and covariance of a multivariate normal with missing values (EM, Dempster et al. 1977)."""
    n, p = X.shape
    M = np.isnan(X)
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0))
    pats, inv = np.unique(M, axis=0, return_inverse=True)
    groups = [(pats[g], np.flatnonzero(inv.ravel() == g)) for g in range(len(pats))]
    it = 0
    for it in range(1, max_iter + 1):
        T1, T2 = np.zeros(p), np.zeros((p, p))
        for pat, idx in groups:
            o, m = ~pat, pat
            Xh = np.empty((len(idx), p))
            Xh[:, o] = X[np.ix_(idx, o)]
            C = np.zeros((p, p))
            if m.any():
                Soo, Smo = S[np.ix_(o, o)], S[np.ix_(m, o)]
                B = np.linalg.solve(Soo, Smo.T).T
                Xh[:, m] = mu[m] + (Xh[:, o] - mu[o]) @ B.T
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ Smo.T
            T1 += Xh.sum(axis=0)
            T2 += Xh.T @ Xh + len(idx) * C
        mu_new = T1 / n
        S_new = T2 / n - np.outer(mu_new, mu_new)
        delta = max(np.max(np.abs(mu_new - mu)), np.max(np.abs(S_new - S)))
        mu, S = mu_new, S_new
        if delta < tol:
            break
    return mu, S, it


@register("data.littles_mcar", "Little's MCAR test", "Data quality", _ALL,
          params=(P("features", "columns", help="Numeric columns (at least 2) with missing values"),),
          description="""Little's (1988) test of H0: the data are Missing Completely At Random. Rows are grouped by
missingness pattern j (observed variables o_j, n_j rows, observed-variable means ȳ_j). With μ, Σ the
maximum-likelihood estimates of the multivariate-normal mean and covariance from ALL rows (EM algorithm),
d² = Σ_j n_j (ȳ_j − μ_oj)' Σ_oj⁻¹ (ȳ_j − μ_oj), asymptotically chi-square with Σ_j p_j − p degrees of freedom
(p_j observed variables in pattern j, p variables). A small p-value is evidence against MCAR (missingness
depends on the data); a large p-value does not prove MCAR and the test cannot distinguish MAR from MNAR.
Assumes approximate multivariate normality; rows with every variable missing carry no information and are
excluded.""",
          references=("Little (1988), A test of missing completely at random for multivariate data with missing "
                      "values, JASA 83(404), 1198-1202",
                      "Dempster, Laird & Rubin (1977), Maximum likelihood from incomplete data via the EM "
                      "algorithm, JRSS B 39(1)"))
def littles_mcar(ctx: RunContext, features) -> Outcome:
    if len(features) < 2:
        raise ValueError("Little's test needs at least 2 columns.")
    X = np.column_stack([num(ctx.df, c).to_numpy() for c in features])
    allmiss = np.isnan(X).all(axis=1)
    X = X[~allmiss]
    n, p = X.shape
    M = np.isnan(X)
    if not M.any():
        raise NotApplicable("No missing values in these columns; MCAR is moot.")
    if (~M).sum(axis=0).min() < 2:
        raise NotApplicable("A column has fewer than 2 observed values.")
    mu, S, iters = _em_mvn(X)
    if np.linalg.matrix_rank(S) < p:
        raise NotApplicable("Estimated covariance matrix is singular (collinear or constant columns).")
    pats, inv = np.unique(M, axis=0, return_inverse=True)
    inv = inv.ravel()
    d2, df_, rows = 0.0, 0, []
    for g, pat in enumerate(pats):
        idx = np.flatnonzero(inv == g)
        o = ~pat
        diff = X[np.ix_(idx, o)].mean(axis=0) - mu[o]
        contrib = len(idx) * float(diff @ np.linalg.solve(S[np.ix_(o, o)], diff))
        d2 += contrib
        df_ += int(o.sum())
        rows.append({"missing_columns": ", ".join(c for c, mm in zip(features, pat) if mm) or "<complete>",
                     "rows": len(idx), "observed_variables": int(o.sum()), "d2_contribution": contrib})
    df_ -= p
    if df_ <= 0:
        raise NotApplicable("Only one missingness pattern; the test has no degrees of freedom.")
    pval = float(stats.chi2.sf(d2, df_))
    est = pd.DataFrame({"variable": features, "em_mean": mu, "em_sd": np.sqrt(np.diag(S)),
                        "observed": (~M).sum(axis=0)})
    return Outcome({"d2": d2, "dof": df_, "p_value": pval, "patterns": len(pats), "n": n, "em_iterations": iters},
                   {"Patterns": pd.DataFrame(rows), "EM estimates": est}, rows_used=n,
                   notes=([f"{int(allmiss.sum())} rows with all variables missing excluded."] if allmiss.any() else [])
                   + ["Asymptotic chi-square; assumes multivariate normality. Non-numeric values are treated as missing."])


# ── duplicates ──────────────────────────────────────────────────────────────

_MAX_GROUPS = 500


def _dup_groups(df: pd.DataFrame, key: pd.Series, id, k: int) -> pd.DataFrame:
    """One row per value of `key` that occurs more than once (NaN keys ignored), in order of first row."""
    kk = key.dropna()
    counts = kk.map(kk.value_counts())
    dup = kk[counts > 1]
    rows = []
    for _, idx in dup.groupby(dup, sort=False).groups.items():
        pos = np.sort(np.asarray(idx, dtype=int))      # df has a RangeIndex: labels are positions
        rows.append({"rows": len(pos), "first_row": int(pos[0]), **_ex(df, pos, id, k)})
    out = pd.DataFrame(rows, columns=["rows", "first_row", "example_rows"] + (["example_ids"] if id else []))
    return out.sort_values("first_row", kind="mergesort").reset_index(drop=True)


@register("data.duplicates", "Duplicate rows and duplicate keys", "Data quality", _ALL,
          params=(P("keys", "columns", required=False, help="Key columns that should identify a row uniquely"),
                  P("date", required=False, help="Snapshot / observation date added to the key"),
                  P("ignore", "columns", required=False,
                    help="Columns ignored when comparing full rows (e.g. load timestamps)"),
                  P("round_digits", "integer", default=6, help="Numeric rounding for near-duplicate detection"),
                  *_EVID),
          description="""(1) Full-row duplicates: rows identical in every column (except `ignore`), grouped, with the
row numbers of each group. (2) Key duplicates: rows sharing the same `keys` (+ `date`) values, with example key
values, rows per key, and whether the duplicated rows conflict (differ in some non-key column) or are exact
repeats; rows with a missing key value are counted separately. (3) Near-duplicates: rows that are not exact
duplicates but become identical after trimming and lower-casing text, collapsing internal whitespace and rounding
numbers to `round_digits` decimals. Missing values compare equal to each other.""",
          references=("BCBS 239 (2013), Principle 3 (accuracy and integrity)",
                      "ECB Guide to internal models (2024), general topics: data quality"))
def duplicates(ctx: RunContext, keys=None, date=None, ignore=None, round_digits=6, id=None,
               max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise NotApplicable("The table is empty.")
    d = df.reset_index(drop=True)
    cols = [c for c in d.columns if c not in set(ignore or [])]
    full = _composite(d.astype("object").where(d.notna(), "\x00NA"), cols)
    tables, notes = {}, [ROW_NOTE]
    g_full = _dup_groups(d, full, id, max_examples)
    full_rows = int(g_full["rows"].sum()) if len(g_full) else 0
    tables["Full-row duplicate groups"] = g_full.drop(columns="first_row").head(_MAX_GROUPS)
    summary = {"rows": n, "full_duplicate_groups": len(g_full), "rows_in_full_duplicates": full_rows,
               "redundant_full_duplicates": full_rows - len(g_full)}

    # near duplicates
    norm = pd.DataFrame(index=d.index)
    for c in cols:
        s = d[c]
        if _is_text(s):
            norm[c] = s.astype("object").map(lambda v: re.sub(r"\s+", " ", v.strip().lower())
                                             if isinstance(v, str) else v)
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            norm[c] = s.astype(float).round(round_digits)
        else:
            norm[c] = s
    near = _composite(norm.astype("object").where(norm.notna(), "\x00NA"), cols)
    near_counts = near.map(near.value_counts())
    raw_variants = pd.DataFrame({"near": near, "full": full}).groupby("near")["full"].transform("nunique")
    near_mask = (near_counts > 1) & (raw_variants > 1)
    g_near = _dup_groups(d, near.where(near_mask), id, max_examples)
    tables["Near-duplicate groups"] = g_near.drop(columns="first_row").head(_MAX_GROUPS)
    summary["near_duplicate_groups"] = len(g_near)

    if keys:
        kcols = list(keys) + ([date] if date and date not in keys else [])
        key = _composite(d, kcols)
        miss = key.isna()
        g_key = _dup_groups(d, key, id, max_examples)
        if len(g_key):
            first = g_key["first_row"].to_numpy()
            kv = d[kcols].iloc[first].reset_index(drop=True)
            g_key = pd.concat([kv, g_key], axis=1)
            distinct_full = pd.DataFrame({"k": key, "f": full}).dropna().groupby("k")["f"].nunique()
            g_key["conflicting"] = key.iloc[first].map(distinct_full).to_numpy() > 1
        tables["Duplicate keys"] = g_key.drop(columns="first_row").head(_MAX_GROUPS)
        summary.update({"key_columns": ", ".join(kcols), "duplicate_keys": len(g_key),
                        "rows_with_duplicate_keys": int(g_key["rows"].sum()) if len(g_key) else 0,
                        "conflicting_duplicate_keys": int(g_key["conflicting"].sum()) if len(g_key) else 0,
                        "rows_with_missing_key": int(miss.sum())})
        if miss.any():
            tables["Rows with missing key"] = pd.DataFrame([_ex(d, _pos(miss), id, max_examples)])
    if any(len(t) >= _MAX_GROUPS for t in tables.values()):
        notes.append(f"Group tables are capped at {_MAX_GROUPS} groups; counts in the summary are complete.")
    return Outcome(summary, tables, notes=notes, rows_used=n)


# ── validity rules and automatic range checks ────────────────────────────────

_RULE_KEYS = {"min", "max", "allowed", "regex", "not_null", "unique"}


def _compare_values(s: pd.Series, bound, dayfirst=False):
    """(values comparable with bound, bound) — dates if the column is date-like, else numbers."""
    if _is_date_col(s) or isinstance(bound, pd.Timestamp):
        return _to_dates(s, dayfirst), pd.Timestamp(bound)
    return _numeric_parse(s) if _is_text(s) else s.astype(float), float(bound)


@register("data.validity_rules", "Validity rules (ranges, allowed values, patterns, not-null)",
          "Data quality", _ALL,
          params=(P("rules", "dict", help='{column: {"min": .., "max": .., "allowed": [..], "regex": "..", '
                                          '"not_null": true, "unique": true}}'),
                  *_EVID, _DAYFIRST),
          description="""Checks user-supplied business rules per column and reports, per rule, the number of values
checked, the number violating and example rows with their values. Rules: min / max (inclusive; dates when the
column is a date column, numbers otherwise; non-null values that cannot be read as a number/date count as
violations), allowed (value must be in the list; numeric columns compare numerically, text compares exactly,
so case and whitespace variants are violations), regex (full match on the text value), not_null, unique
(non-null values must not repeat; every row of a repeated value is a violation). Except for not_null, rules
apply to non-missing values only.""",
          references=("BCBS 239 (2013), Principle 3 (accuracy and integrity)",
                      "ECB Guide to internal models (2024), general topics: data quality"))
def validity_rules(ctx: RunContext, rules, id=None, max_examples=10, dayfirst=False) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    if not rules:
        raise ValueError("Give at least one rule.")
    rows, ev, any_bad = [], [], pd.Series(False, index=df.index)
    for col_in, spec in rules.items():
        col = _match_col(df, col_in, "rule")
        if not isinstance(spec, dict):
            raise ValueError(f"Rule for '{col_in}' must be an object like {{\"min\": 0}}")
        unknown = set(spec) - _RULE_KEYS
        if unknown:
            raise ValueError(f"Unknown rule key(s) {sorted(unknown)} for '{col_in}'; use {sorted(_RULE_KEYS)}")
        s = df[col]
        nn = s.notna()
        for rule, val in spec.items():
            if rule == "not_null":
                if not val:
                    continue
                bad = ~nn
            elif rule in {"min", "max"}:
                x, b = _compare_values(s, val, dayfirst)
                unread = nn & x.isna()
                bad = unread | ((x < b) if rule == "min" else (x > b)).fillna(False).astype(bool)
            elif rule == "allowed":
                if not isinstance(val, (list, tuple)):
                    raise ValueError(f"'allowed' for '{col_in}' must be a list")
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    allowed = pd.to_numeric(pd.Series(list(val), dtype="object"), errors="coerce").dropna()
                    bad = nn & ~s.astype(float).isin(allowed.astype(float))
                else:
                    bad = nn & ~s.astype("object").map(str).isin([str(v) for v in val])
            elif rule == "regex":
                rx = re.compile(str(val))
                bad = nn & ~s.astype("object").map(lambda v: bool(rx.fullmatch(str(v))) if pd.notna(v) else True)
            else:  # unique
                if not val:
                    continue
                bad = nn & s.duplicated(keep=False)
            bad = bad.astype(bool)
            pos = _pos(bad)
            checked = int(len(s) if rule == "not_null" else nn.sum())
            rows.append({"column": col, "rule": rule, "rule_value": str(val), "n_checked": checked,
                         "violations": len(pos), "violation_pct": len(pos) / checked if checked else float("nan"),
                         **(_ex(df, pos, id, max_examples) if len(pos) else {"example_rows": ""}),
                         "example_values": _join([_show(v) for v in s.iloc[pos[:max_examples]]], max_examples,
                                                 len(pos))})
            ev += _evidence(df, pos, max_examples, f"rule {rule}={val}", col, id)
            any_bad |= bad.to_numpy()
    out = pd.DataFrame(rows)
    return Outcome({"rules": len(out), "rules_with_violations": int((out["violations"] > 0).sum()),
                    "total_violations": int(out["violations"].sum()),
                    "rows_violating_any_rule": int(any_bad.sum()),
                    "rows_violating_pct": float(any_bad.mean())},
                   {"Rule results": out, "Evidence": _ev_frame(ev, id)}, notes=[ROW_NOTE], rows_used=len(df))


_PROB_NAME = re.compile(r"(?:^|[^a-z])(pd|prob|proba|probability|lgd|ccf)(?:[^a-z]|$)", re.I)
_AMOUNT_NAME = re.compile(r"(?:^|[^a-z])(amount|amt|balance|exposure|ead|limit|drawn|outstanding|principal|"
                          r"collateral|notional)(?:[^a-z]|$)", re.I)


@register("data.range_checks", "Automatic range checks (probabilities, amounts, dates)", "Data quality", _ALL,
          params=(P("as_of", "string", required=False, help="As-of / extraction date; dates after it are flagged"),
                  P("min_date", "string", required=False, help="Dates before this are flagged"),
                  P("prob_columns", "columns", required=False,
                    help="Columns that must lie in [0,1]; default: names like pd/prob/lgd/ccf (heuristic)"),
                  P("amount_columns", "columns", required=False,
                    help="Columns that must be >= 0; default: names like amount/balance/exposure/ead/limit (heuristic)"),
                  P("date_columns", "columns", required=False, help="Date columns; default auto-detected"),
                  *_EVID, _DAYFIRST),
          description="""Rule-free plausibility checks with example rows:
probability-like columns outside [0, 1]; negative values in amount-like columns; dates after `as_of`
(future dates); dates before `min_date`; and text in date columns that cannot be parsed as a date.
Column selection by name is a HEURISTIC (regex on the column name) and reported as such in the `basis` column;
pass the column lists explicitly to override. LGD and CCF can legitimately fall outside [0, 1] (recovery costs,
over-recoveries, drawings above the limit), so findings there are candidates for review, not errors.
Future-date and min-date checks run only when the dates are given; the current clock is never used.""",
          references=("CRR (Regulation (EU) No 575/2013), Art. 179 and 181 (estimation data requirements)",
                      "ECB Guide to internal models (2024), credit risk chapter: data"))
def range_checks(ctx: RunContext, as_of=None, min_date=None, prob_columns=None, amount_columns=None,
                 date_columns=None, id=None, max_examples=10, dayfirst=False) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    n = len(df)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]
    candidates = num_cols + [c for c in df.columns if c not in num_cols and _semantic(df[c], c) == "numeric-as-text"]
    probs = [(c, "user-specified") for c in prob_columns] if prob_columns else \
        [(c, "name heuristic") for c in candidates if _PROB_NAME.search(str(c))]
    amts = [(c, "user-specified") for c in amount_columns] if amount_columns else \
        [(c, "name heuristic") for c in candidates if _AMOUNT_NAME.search(str(c))
         and not _PROB_NAME.search(str(c))]
    dates = [(c, "user-specified") for c in date_columns] if date_columns else \
        [(c, "auto-detected") for c in df.columns if _is_date_col(df[c])]
    t_asof = pd.Timestamp(as_of) if as_of else None
    t_min = pd.Timestamp(min_date) if min_date else None
    rows, ev, notes = [], [], [ROW_NOTE]

    def add(col, check, basis, x, bad):
        pos = _pos(bad)
        xn = x.dropna()
        rows.append({"column": col, "check": check, "basis": basis, "n_checked": int(x.notna().sum()),
                     "violations": len(pos), "violation_pct": len(pos) / max(int(x.notna().sum()), 1),
                     "min": _show(xn.min()) if len(xn) else "", "max": _show(xn.max()) if len(xn) else "",
                     **(_ex(df, pos, id, max_examples) if len(pos) else {"example_rows": ""})})
        ev.extend(_evidence(df, pos, max_examples, check, col, id))

    for c, basis in probs:
        x = df[c].astype(float) if pd.api.types.is_numeric_dtype(df[c]) else _numeric_parse(df[c])
        add(c, "probability outside [0,1]", basis, x, ((x < 0) | (x > 1)).fillna(False))
    for c, basis in amts:
        x = df[c].astype(float) if pd.api.types.is_numeric_dtype(df[c]) else _numeric_parse(df[c])
        add(c, "negative amount", basis, x, (x < 0).fillna(False))
    for c, basis in dates:
        x = _to_dates(df[c], dayfirst)
        if _is_text(df[c]):
            add(c, "unparseable date", basis, df[c], (df[c].notna() & x.isna()))
        if t_asof is not None:
            add(c, f"date after as_of {t_asof.date()}", basis, x, (x > t_asof).fillna(False))
        if t_min is not None:
            add(c, f"date before min_date {t_min.date()}", basis, x, (x < t_min).fillna(False))
    if dates and t_asof is None:
        notes.append("No as_of date given: future-date check skipped.")
    if not rows:
        raise NotApplicable("No probability-like, amount-like or date columns found; pass the column lists.")
    out = pd.DataFrame(rows)
    if any(b == "name heuristic" for _, b in probs + amts):
        notes.append("Columns marked 'name heuristic' were selected by their names only; confirm their meaning.")
    return Outcome({"checks": len(out), "checks_with_violations": int((out["violations"] > 0).sum()),
                    "total_violations": int(out["violations"].sum()),
                    "probability_columns": len(probs), "amount_columns": len(amts), "date_columns": len(dates)},
                   {"Range checks": out, "Evidence": _ev_frame(ev, id)}, notes=notes, rows_used=n)


# ── outliers ────────────────────────────────────────────────────────────────

@register("data.outliers", "Outliers per column (IQR, z-score, robust MAD z-score, percentiles)",
          "Data quality", _ALL,
          params=(P("columns", "columns", required=False, help="Numeric columns; default all numeric non-id columns"),
                  P("iqr_k", "number", default=1.5, help="Tukey fence multiplier k"),
                  P("z", "number", default=3.0, help="|z-score| cut-off"),
                  P("mad_z", "number", default=3.5, help="|robust z| cut-off (Iglewicz–Hoaglin)"),
                  P("lower_pct", "number", default=0.01, help="Lower percentile cap"),
                  P("upper_pct", "number", default=0.99, help="Upper percentile cap"), *_EVID),
          description="""For each numeric column and method: the lower/upper bound, the number of values below and
above, and example rows. Methods: Tukey IQR fences [Q1 − k·IQR, Q3 + k·IQR]; z-score |x − mean|/sd > z
(sd with ddof=1; sensitive to the outliers themselves); robust z = 0.6745·(x − median)/MAD > mad_z (MAD = median
absolute deviation; not computable when MAD = 0); percentile caps (values strictly below the lower_pct or above
the upper_pct quantile — the rows that capping/winsorising would change). The cut-offs are method parameters,
not judgements; heavy-tailed variables (amounts, exposures) will show many points by construction.""",
          references=("Tukey (1977), Exploratory Data Analysis",
                      "Iglewicz & Hoaglin (1993), How to Detect and Handle Outliers, ASQC",
                      "NIST/SEMATECH e-Handbook of Statistical Methods, 1.3.5.17 (detection of outliers)"))
def outliers(ctx: RunContext, columns=None, iqr_k=1.5, z=3.0, mad_z=3.5, lower_pct=0.01, upper_pct=0.99,
             id=None, max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    if not 0 <= lower_pct < upper_pct <= 1:
        raise ValueError("Need 0 <= lower_pct < upper_pct <= 1.")
    cols = columns or [c for c in df.columns if _semantic(df[c], c) in {"numeric", "numeric-as-text"}]
    rows, ev, notes = [], [], [ROW_NOTE]
    for c in cols:
        x = df[c].astype(float) if pd.api.types.is_numeric_dtype(df[c]) else _numeric_parse(df[c])
        v = x.dropna()
        if len(v) < 3:
            notes.append(f"{c}: fewer than 3 numeric values, skipped.")
            continue
        q1, q3, med = np.quantile(v, [.25, .75, .5])
        sd = v.std(ddof=1)
        mad = float(np.median(np.abs(v - med)))
        bounds = [("IQR", q1 - iqr_k * (q3 - q1), q3 + iqr_k * (q3 - q1)),
                  ("z-score", v.mean() - z * sd, v.mean() + z * sd) if sd > 0 else ("z-score", np.nan, np.nan),
                  ("robust MAD z", med - mad_z * mad / 0.6745, med + mad_z * mad / 0.6745) if mad > 0
                  else ("robust MAD z", np.nan, np.nan),
                  ("percentile", *np.quantile(v, [lower_pct, upper_pct]))]
        for meth, lo, hi in bounds:
            if np.isnan(lo):
                notes.append(f"{c}: {meth} not computable (zero spread).")
                continue
            low, high = (x < lo).fillna(False), (x > hi).fillna(False)
            pos = _pos(low | high)
            rows.append({"column": c, "method": meth, "n": len(v), "lower_bound": float(lo), "upper_bound": float(hi),
                         "below": int(low.sum()), "above": int(high.sum()), "outliers": len(pos),
                         "outlier_pct": len(pos) / len(v),
                         **(_ex(df, pos, id, max_examples) if len(pos) else {"example_rows": ""})})
            ev += _evidence(df, pos, max_examples, meth, c, id)
    if not rows:
        raise NotApplicable("No numeric column with enough values.")
    out = pd.DataFrame(rows)
    return Outcome({"columns": out["column"].nunique(),
                    "iqr_outliers": int(out.loc[out["method"] == "IQR", "outliers"].sum()),
                    "zscore_outliers": int(out.loc[out["method"] == "z-score", "outliers"].sum()),
                    "mad_outliers": int(out.loc[out["method"] == "robust MAD z", "outliers"].sum())},
                   {"Outliers by column and method": out, "Evidence": _ev_frame(ev, id)}, notes=notes,
                   rows_used=len(df))


def _grubbs_crit(n: int, alpha: float, two_sided: bool) -> float:
    t = stats.t.isf(alpha / (2 * n if two_sided else n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))


def _grubbs_p(G: float, n: int, two_sided: bool) -> float:
    den = (n - 1) ** 2 - n * G * G
    if den <= 0:
        return 0.0
    t = np.sqrt(n * (n - 2) * G * G / den)
    return float(min(1.0, (2 * n if two_sided else n) * stats.t.sf(t, n - 2)))


@register("data.grubbs", "Grubbs test / generalized ESD for outliers in one column", "Data quality", _ALL,
          params=(P("column"), P("alternative", "string", default="two-sided", choices=("two-sided", "max", "min")),
                  P("alpha", "number", default=0.05, help="Significance level for the critical values"),
                  P("max_outliers", "integer", default=1,
                    help="Upper bound r for Rosner's generalized ESD (1 = plain Grubbs)"), *_EVID),
          description="""Grubbs (1969) test, H0: no outlier, the data are a sample from one normal distribution.
G = max|x − mean|/s (two-sided), (max − mean)/s or (mean − min)/s (one-sided); s with ddof=1.
p-value = n·P(T > t) (one-sided) or 2n·P(T > t) (two-sided), capped at 1, with t = √(n(n−2)G²/((n−1)² − nG²)) and
T ~ t(n−2); the critical value G_crit = ((n−1)/√n)·√(t_c²/(n−2+t_c²)), t_c the upper α/n (α/(2n)) t quantile.
With max_outliers = r > 1, Rosner's (1983) generalized ESD: remove the most extreme value r times, R_i vs λ_i;
the number of outliers is the largest i with R_i > λ_i (the procedure's own definition). Assumes normality —
on skewed data (amounts) the test flags the tail, not errors.""",
          references=("Grubbs (1969), Procedures for detecting outlying observations in samples, Technometrics 11(1)",
                      "Rosner (1983), Percentage points for a generalized ESD many-outlier procedure, "
                      "Technometrics 25(2)",
                      "NIST/SEMATECH e-Handbook of Statistical Methods, 1.3.5.17"))
def grubbs(ctx: RunContext, column, alternative="two-sided", alpha=0.05, max_outliers=1, id=None,
           max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    x = num(df, column)
    v = x.dropna()
    n = len(v)
    if n < 3:
        raise NotApplicable("Grubbs test needs at least 3 values.")
    if v.std(ddof=1) == 0:
        raise NotApplicable("No variance.")
    if not 1 <= max_outliers <= n - 2:
        raise ValueError("max_outliers must be between 1 and n-2.")
    two = alternative == "two-sided"
    vals, pos_all = v.to_numpy(), _pos(x.notna())
    rows = []
    for i in range(max_outliers):
        m = len(vals)
        if m < 3:
            break
        mean, s = vals.mean(), vals.std(ddof=1)
        if s == 0:
            break
        dev = (vals - mean) if alternative == "max" else (mean - vals) if alternative == "min" else np.abs(vals - mean)
        j = int(np.argmax(dev))
        R = float(dev[j] / s)
        lam = _grubbs_crit(m, alpha, two)      # = Rosner's lambda_i for the current sample size m
        r ={"step": i + 1, "n": m, "row": int(pos_all[j])}
        if id:
            r["id"] = df[id].iloc[pos_all[j]]
        r.update({"value": float(vals[j]), "statistic": R, "critical_value": float(lam),
                  "p_value": _grubbs_p(R, m, two)})
        rows.append(r)
        vals, pos_all = np.delete(vals, j), np.delete(pos_all, j)
    out = pd.DataFrame(rows)
    first = out.iloc[0]
    exceed = np.flatnonzero(out["statistic"].to_numpy() > out["critical_value"].to_numpy())
    k_out = int(exceed.max() + 1) if len(exceed) else 0
    summary = {"G": float(first["statistic"]), "p_value": float(first["p_value"]),
               "critical_value": float(first["critical_value"]), "alpha": alpha, "n": n,
               "suspect_row": int(first["row"]), "suspect_value": float(first["value"])}
    if id:
        summary["suspect_id"] = first["id"]
    if max_outliers > 1:
        summary["gesd_outliers"] = k_out
        summary["gesd_rows"] = _join(out["row"].head(k_out).tolist(), max_examples)
    return Outcome(summary, {"Grubbs / ESD steps": out},
                   notes=[ROW_NOTE, "Assumes the non-outlying data are normal."]
                   + ([f"{int(x.isna().sum())} missing or non-numeric values excluded."] if x.isna().any() else []),
                   rows_used=n)


# ── data types ──────────────────────────────────────────────────────────────

def _shape(v: str) -> str:
    """Format signature of a date string: digits -> d, letters -> a."""
    return re.sub(r"\d", "d", re.sub(r"[A-Za-z]", "a", v.strip()))


@register("data.type_consistency", "Data-type and formatting consistency", "Data quality", _ALL,
          params=(P("columns", "columns", required=False, help="Text columns; default all non-numeric, non-date columns"),
                  *_EVID),
          description="""For text (string/object) columns: numbers stored as text (values that parse as numbers, and
the values that do not); mixed content (some values numeric, some dates, some other text) and mixed Python
types in object columns; inconsistent date formats (several format signatures such as dddd-dd-dd vs dd/dd/dddd
among date-like values); leading/trailing whitespace; and inconsistent category spelling (distinct raw labels
that become identical after lower-casing, trimming and collapsing internal whitespace, e.g. 'Retail', 'retail ',
'RETAIL'). Every finding lists counts and example rows with the raw value quoted.""",
          references=("BCBS 239 (2013), Principle 3 (accuracy and integrity)",
                      "ECB Guide to internal models (2024), general topics: data quality"))
def type_consistency(ctx: RunContext, columns=None, id=None, max_examples=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    cols = columns or [c for c in df.columns if _is_text(df[c])]
    cols = [c for c in cols if _is_text(df[c])]
    if not cols:
        raise NotApplicable("No text columns to check.")
    rows, variants, formats, ev = [], [], [], []
    for c in cols:
        s = df[c]
        nn = s.notna()
        obj = s.astype("object")
        sval = obj.map(lambda v: v if isinstance(v, str) else (str(v) if pd.notna(v) else None))
        types = obj[nn].map(lambda v: type(v).__name__).value_counts().sort_index()
        numeric = _numeric_parse(s).notna() & nn
        datelike = sval.map(lambda v: bool(_DATE_RE.match(v)) if isinstance(v, str) else False) & nn
        other = nn & ~numeric & ~datelike
        ws = obj.map(lambda v: isinstance(v, str) and v != v.strip())
        r = {"column": c, "n_non_missing": int(nn.sum()),
             "python_types": "; ".join(f"{k}: {v}" for k, v in types.items()),
             "numeric_values": int(numeric.sum()), "date_like_values": int(datelike.sum()),
             "other_text_values": int(other.sum()),
             "mixed_content": int((numeric.sum() > 0) + (datelike.sum() > 0) + (other.sum() > 0)) > 1
             or len(types) > 1,
             "leading_trailing_whitespace": int(ws.sum())}
        if numeric.sum() and other.sum():
            ev += _evidence(df, _pos(other), max_examples, "non-numeric value in mostly numeric column"
                            if numeric.sum() >= other.sum() else "numeric value in text column", c, id)
        elif numeric.sum() and not other.sum() and not datelike.sum():
            ev += _evidence(df, _pos(numeric), max_examples, "number stored as text", c, id)
        ev += _evidence(df, _pos(ws), max_examples, "leading/trailing whitespace", c, id)
        # date formats
        shapes = sval[datelike].map(_shape)
        r["date_formats"] = int(shapes.nunique())
        if shapes.nunique() > 1:
            for shp, idx in shapes.groupby(shapes, sort=True).groups.items():
                pos = _pos(df.index.isin(idx))
                formats.append({"column": c, "format": shp, "values": len(pos),
                                **_ex(df, pos, id, max_examples),
                                "example_values": _join([repr(v) for v in sval.iloc[pos[:3]]], 3, len(pos))})
                ev += _evidence(df, pos, max_examples, f"date format {shp}", c, id)
        # category spelling variants
        txt = sval[nn & ~numeric & ~datelike]
        normed = txt.map(lambda v: re.sub(r"\s+", " ", v.strip().lower()))
        nvar = 0
        for key, idx in normed.groupby(normed, sort=True).groups.items():
            raw = txt.loc[idx]
            if raw.nunique() > 1:
                nvar += 1
                for lab in sorted(raw.unique()):
                    pos = _pos(df.index.isin(raw.index[raw == lab]))
                    variants.append({"column": c, "normalised_label": key, "raw_label": repr(lab),
                                     "rows": len(pos), **_ex(df, pos, id, max_examples)})
                    ev += _evidence(df, pos, max_examples, f"spelling variant of '{key}'", c, id)
        r["labels_with_spelling_variants"] = nvar
        rows.append(r)
    out = pd.DataFrame(rows)
    tables = {"Type checks by column": out}
    if variants:
        tables["Category spelling variants"] = pd.DataFrame(variants)
    if formats:
        tables["Date formats"] = pd.DataFrame(formats)
    tables["Evidence"] = _ev_frame(ev, id)
    return Outcome({"columns": len(out), "columns_with_mixed_content": int(out["mixed_content"].sum()),
                    "columns_with_whitespace": int((out["leading_trailing_whitespace"] > 0).sum()),
                    "whitespace_values": int(out["leading_trailing_whitespace"].sum()),
                    "columns_with_several_date_formats": int((out["date_formats"] > 1).sum()),
                    "labels_with_spelling_variants": int(out["labels_with_spelling_variants"].sum())},
                   tables, notes=[ROW_NOTE, "Values in the evidence are quoted so whitespace is visible."],
                   rows_used=len(df))


# ── target / default definition ─────────────────────────────────────────────

def _canon_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


@register("data.target_sanity", "Target / default-definition sanity", "Data quality", _BINARY,
          params=(P("target", help="Target column as stored (values are checked before any conversion)"),
                  P("period", required=False), P("segment", required=False),
                  P("allowed", "list", default=["0", "1"], help="Allowed target values"),
                  P("event_value", "string", default="1", help="Value that marks an event (default)"),
                  P("as_of", "string", required=False, help="Data extraction / as-of date"),
                  P("horizon_months", "integer", default=12, help="Outcome window length in months"),
                  P("freq", "string", default="M", choices=("D", "W", "M", "Q", "Y"),
                    help="Period frequency when period holds dates"), *_EVID, _DAYFIRST),
          description="""Checks the target as stored: value counts with allowed / not allowed (numbers compare
numerically, so 1, 1.0 and '1' are the same value), missing targets, example rows of disallowed values. Event
prevalence overall, per period and per segment (event = `event_value`, among rows with an allowed value), with a
chi-square test of homogeneity of the event rate across periods / segments (H0: equal event rates).
Observation-window completeness (with `period` and `as_of`): the observation date of a period is the LAST day
of the period at `freq`; the outcome window ends `horizon_months` later; the window is complete when it ends on
or before `as_of`. Periods with incomplete windows typically show artificially low default rates.""",
          references=("CRR (Regulation (EU) No 575/2013), Art. 178 (definition of default)",
                      "EBA/GL/2016/07, Guidelines on the application of the definition of default",
                      "EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures"))
def target_sanity(ctx: RunContext, target, period=None, segment=None, allowed=None, event_value="1",
                  as_of=None, horizon_months=12, freq="M", id=None, max_examples=10, dayfirst=False) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    allowed = allowed if allowed is not None else ["0", "1"]
    s = df[target]
    canon = s.astype("object").map(_canon_val)
    allowed_c = {_canon_val(a) for a in allowed}
    ev_c = _canon_val(event_value)
    if ev_c not in allowed_c:
        raise ValueError(f"event_value {event_value!r} is not among allowed {allowed}")
    miss = s.isna()
    ok = canon.map(lambda v: v in allowed_c) & ~miss
    bad = ~ok & ~miss
    event = (canon == ev_c) & ok
    vals = pd.DataFrame({"value": s.astype("object").map(_show), "ok": ok, "miss": miss})
    vt = []
    for v, idx in vals.groupby("value", sort=True).groups.items():
        pos = np.sort(np.asarray(idx, dtype=int))
        is_miss = bool(miss.iloc[pos[0]])
        vt.append({"value": v, "rows": len(pos), "share": len(pos) / len(df),
                   "status": "missing" if is_miss else ("allowed" if ok.iloc[pos[0]] else "NOT allowed"),
                   **(_ex(df, pos, id, max_examples) if not ok.iloc[pos[0]] else {"example_rows": ""})})
    tables = {"Target values": pd.DataFrame(vt)}
    nv = int(ok.sum())
    if nv == 0:
        raise NotApplicable("No row has an allowed target value.")
    summary = {"rows": len(df), "valid_target_rows": nv, "events": int(event.sum()),
               "event_rate": float(event.sum() / nv), "missing_target": int(miss.sum()),
               "invalid_target": int(bad.sum())}
    notes = [ROW_NOTE]

    def by(col, label):
        g = df[col].astype("object").where(df[col].notna(), "<missing>")
        d = pd.DataFrame({"g": g, "n": 1, "valid": ok, "ev": event, "miss": miss, "bad": bad})
        t = d.groupby("g", sort=False).agg(rows=("n", "sum"), valid=("valid", "sum"), events=("ev", "sum"),
                                           missing_target=("miss", "sum"), invalid_target=("bad", "sum"))
        t = t.reset_index().rename(columns={"g": label})
        t["event_rate"] = t["events"] / t["valid"].replace(0, np.nan)
        ct = np.column_stack([t["events"], t["valid"] - t["events"]])
        ct = ct[ct.sum(axis=1) > 0]
        chi = (float("nan"), float("nan"))
        if len(ct) > 1 and (ct.sum(axis=0) > 0).all():
            c2, p, _, _ = stats.chi2_contingency(ct, correction=False)
            chi = (float(c2), float(p))
        return t, chi

    if period:
        t, chi = by(period, "period")
        per = _period_series(df[period], freq, dayfirst)
        if per is not None:
            first = pd.Series(per.values, index=df.index).groupby(
                df[period].astype("object").where(df[period].notna(), "<missing>"), sort=False).first()
            t["obs_date"] = t["period"].map(lambda v: first.get(v)).map(
                lambda p: p.end_time.normalize() if isinstance(p, pd.Period) else pd.NaT)
            t = t.sort_values("obs_date", kind="mergesort", na_position="last")
            if as_of:
                a = pd.Timestamp(as_of)
                t["window_end"] = t["obs_date"].map(lambda d: d + pd.DateOffset(months=horizon_months)
                                                    if pd.notna(d) else pd.NaT)
                t["window_complete"] = t["window_end"].map(lambda d: bool(d <= a) if pd.notna(d) else False)
                inc = t[~t["window_complete"]]
                summary.update({"incomplete_window_periods": len(inc),
                                "rows_in_incomplete_periods": int(inc["rows"].sum()),
                                "events_in_incomplete_periods": int(inc["events"].sum())})
            t["obs_date"] = t["obs_date"].dt.date.astype(str)
            if "window_end" in t:
                t["window_end"] = t["window_end"].map(lambda d: str(d.date()) if pd.notna(d) else "")
        else:
            t = t.sort_values("period", key=lambda s: s.astype(str), kind="mergesort")
            if as_of:
                notes.append("Period values could not be read as dates; window completeness not assessed.")
        t["period"] = t["period"].astype(str)
        tables["By period"] = t.reset_index(drop=True)
        summary.update({"periods": len(t), "period_rate_chi2": chi[0], "period_rate_p_value": chi[1],
                        "min_events_in_a_period": int(t["events"].min())})
    if segment:
        t, chi = by(segment, "segment")
        t["segment"] = t["segment"].astype(str)
        tables["By segment"] = t.sort_values("segment", kind="mergesort").reset_index(drop=True)
        summary.update({"segments": len(t), "segment_rate_chi2": chi[0], "segment_rate_p_value": chi[1]})
    return Outcome(summary, tables, notes=notes, rows_used=nv)


# ── representativeness and time coverage ────────────────────────────────────

@register("data.representativeness", "Sample vs population representativeness", "Data quality", _ALL,
          params=(P("other", "table", help="Population (or reference) table to compare the active table with"),
                  P("features", "columns", required=False, help="Variables; default all common columns"),
                  P("bins", "integer", default=10), P("id", required=False, help="Identifier column to skip")),
          description="""Per variable, the active table (sample) against a population table: PSI-like distance
Σ (sample% − pop%)·ln(sample%/pop%) with bins fixed on the population (quantile bins for numeric, one bin per
level for categorical/≤ 20 levels, missing its own bin); a two-sample Kolmogorov–Smirnov test (numeric) or a
chi-square test of homogeneity (categorical), H0: same distribution; missing shares; and coverage of categories:
levels present in the population but absent from the sample (with the population share they represent) and
levels in the sample that do not exist in the population.""",
          references=("CRR (Regulation (EU) No 575/2013), Art. 174 and 179 (representativeness of data)",
                      "EBA/GL/2017/16, Guidelines on PD and LGD estimation, section 4 (representativeness)",
                      "ECB Guide to internal models (2024), credit risk chapter: representativeness"))
def representativeness(ctx: RunContext, other, features=None, bins=10, id=None) -> Outcome:
    smp, pop = ctx.df.reset_index(drop=True), ctx.tables[other].reset_index(drop=True)
    feats = features or [c for c in smp.columns if c in pop.columns and c != id]
    missing_in_pop = [f for f in feats if f not in pop.columns]
    if missing_in_pop:
        raise ValueError(f"Not in the population table: {missing_in_pop}")
    if not feats:
        raise NotApplicable("No common columns.")
    rows, cover = [], []
    for f in feats:
        cat = _is_categorical(pop[f])
        t = psi_table(pop[f], smp[f], bins, categorical=cat)
        r = {"variable": f, "type": "categorical" if cat else "numeric", "n_sample": len(smp),
             "n_population": len(pop), "sample_missing_pct": float(smp[f].isna().mean()),
             "population_missing_pct": float(pop[f].isna().mean()),
             "PSI": float(t["psi_contribution"].sum())}
        if cat:
            ct = np.vstack([t["cur_count"], t["ref_count"]])
            ct = ct[:, ct.sum(axis=0) > 0]
            if ct.shape[1] > 1:
                chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
                r.update({"test": "chi-square", "statistic": float(chi2), "p_value": float(p)})
            ps = pop[f].dropna().astype(str)
            ss = set(smp[f].dropna().astype(str))
            absent = sorted(set(ps) - ss)
            extra = sorted(ss - set(ps))
            r.update({"pop_levels_absent_in_sample": len(absent),
                      "pop_share_of_absent_levels": float(ps.isin(absent).sum() / len(pop)) if len(pop) else np.nan,
                      "sample_levels_not_in_population": len(extra)})
            for lev in absent:
                cover.append({"variable": f, "level": lev, "issue": "in population, absent from sample",
                              "population_rows": int((ps == lev).sum()), "sample_rows": 0})
            for lev in extra:
                sr = _pos(smp[f].astype("object").map(lambda v: pd.notna(v) and str(v) == lev))
                cover.append({"variable": f, "level": lev, "issue": "in sample, absent from population",
                              "population_rows": 0, "sample_rows": len(sr), "example_rows": _join(sr, 10)})
        else:
            a, b = num(smp, f).dropna(), num(pop, f).dropna()
            if len(a) >= 1 and len(b) >= 1:
                ks = stats.ks_2samp(a, b)
                r.update({"test": "Kolmogorov-Smirnov", "statistic": float(ks.statistic), "p_value": float(ks.pvalue)})
        rows.append(r)
    out = pd.DataFrame(rows)
    tables = {"Representativeness by variable": out}
    if cover:
        tables["Category coverage"] = pd.DataFrame(cover)
    i = out["PSI"].idxmax()
    return Outcome({"variables": len(out), "n_sample": len(smp), "n_population": len(pop),
                    "max_PSI": float(out.loc[i, "PSI"]), "max_PSI_variable": out.loc[i, "variable"],
                    "levels_absent_in_sample": int(sum(1 for c in cover if c["sample_rows"] == 0))},
                   tables, rows_used=len(smp) + len(pop),
                   notes=["Tests assume independent samples; when the sample is drawn from the population they are "
                          "conservative (overlap makes the distributions more alike).",
                          "PSI bins are fixed on the population; zero-share bins are floored at 1e-4."])


@register("data.time_coverage", "Time coverage: records per period and gaps", "Data quality", _ALL,
          params=(P("period", help="Period or date column"),
                  P("freq", "string", default="M", choices=("D", "W", "M", "Q", "Y"),
                    help="Frequency used when the column holds dates"),
                  P("segment", required=False), _DAYFIRST),
          description="""Number of records per period (dates are bucketed at `freq`), change versus the previous
period, missing periods (gaps) in the calendar sequence between the first and last period, rows with a missing
or unreadable period, and with `segment`, the periods in which a segment has no records. Integer period codes
such as 202401 are read as yyyymm; when the column is neither dates nor integers, only counts are given.""",
          references=("CRR (Regulation (EU) No 575/2013), Art. 180(1)(h) and 181(1)(j) (length of data series)",
                      "EBA/GL/2017/16, Guidelines on PD and LGD estimation"))
def time_coverage(ctx: RunContext, period, freq="M", segment=None, dayfirst=False) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    per = _period_series(df[period], freq, dayfirst)
    notes, tables = [], {}
    if per is not None:
        key = per
        unread = int((df[period].notna() & per.isna()).sum())
        observed = sorted(per.dropna().unique())
        full = pd.period_range(observed[0], observed[-1], freq=observed[0].freq) if observed else []
        gaps = [p for p in full if p not in set(observed)]
        counts = key.value_counts().reindex(full, fill_value=0)
        t = pd.DataFrame({"period": [str(p) for p in full], "records": counts.to_numpy()})
    else:
        unread = 0
        key = df[period]
        is_int = pd.api.types.is_integer_dtype(df[period])
        observed = sorted(key.dropna().unique(), key=lambda v: (str(type(v)), v))
        counts = key.value_counts()
        if is_int and observed:
            full = list(range(int(observed[0]), int(observed[-1]) + 1))
            gaps = [p for p in full if p not in set(observed)]
            counts = counts.reindex(full, fill_value=0)
        else:
            full, gaps = observed, []
            counts = counts.reindex(observed)
            notes.append("Period values are not dates or integers; gaps cannot be determined.")
        t = pd.DataFrame({"period": [str(p) for p in counts.index], "records": counts.to_numpy()})
    if len(t) == 0:
        raise NotApplicable("No period values.")
    t["change_vs_previous_pct"] = t["records"].pct_change().replace([np.inf, -np.inf], np.nan)
    t["gap"] = t["records"] == 0
    tables["Records per period"] = t
    summary = {"periods_observed": len(observed), "first_period": str(observed[0]), "last_period": str(observed[-1]),
               "expected_periods": len(full), "gaps": len(gaps), "gap_periods": _join([str(g) for g in gaps], 24),
               "missing_period_rows": int(df[period].isna().sum()), "unreadable_period_rows": unread,
               "min_records": int(t.loc[~t["gap"], "records"].min()), "max_records": int(t["records"].max())}
    if segment:
        k = key.astype(str).where(key.notna())
        ct = pd.crosstab(k,df[segment].astype("object").where(df[segment].notna(), "<missing>").astype(str))
        ct = ct.reindex([str(p) for p in full], fill_value=0)
        long = ct.reset_index().melt(id_vars=ct.index.name or "row_0", var_name="segment", value_name="records")
        long.columns = ["period", "segment", "records"]
        zero = long[long["records"] == 0].sort_values(["segment", "period"], kind="mergesort")
        tables["Records per period and segment"] = long.sort_values(["segment", "period"], kind="mergesort")
        tables["Segment gaps"] = zero.reset_index(drop=True)
        summary["segment_period_gaps"] = len(zero)
    import plotly.express as px
    figs = [px.bar(t, x="period", y="records", title=f"Records per period ({period})")]
    return Outcome(summary, tables, figs, notes, rows_used=int(df[period].notna().sum()))


# ── reconciliation and referential integrity ────────────────────────────────

@register("data.reconciliation", "Reconciliation against a source / reference table", "Data quality", _ALL,
          params=(P("other", "table", help="Source / reference table (B); the active table is A"),
                  P("keys", "columns", help="Key columns (same names in both tables)"),
                  P("compare", "columns", required=False, help="Columns to compare; default all common non-key columns"),
                  P("tolerance", "number", default=0.0, help="Absolute tolerance for numeric differences"),
                  P("rel_tolerance", "number", default=0.0, help="Relative tolerance (share of |B|)"),
                  P("sum_columns", "columns", required=False,
                    help="Columns whose totals are reconciled; default the numeric compare columns"),
                  P("group", required=False, help="Column for totals by group (in both tables)"),
                  P("max_examples", "integer", default=10)),
          description="""Compares table A (active) with table B (`other`) on the key columns: row counts, distinct keys,
duplicate keys in each table, keys only in A and only in B (with example keys and row numbers), and for matched
keys the number of differing values per column (numeric: |A − B| > tolerance + rel_tolerance·|B|, or one side
missing; text: exact inequality; both missing counts as equal), with the largest and summed differences and
example keys. Totals reconciliation: sum per column in A and B on all rows and on matched keys only, overall
and by `group`. Keys are compared as text after normalising integer-valued numbers (1 and 1.0 match).""",
          references=("BCBS 239 (2013), Principles 3 and 4 (accuracy, integrity, completeness)",
                      "ECB Guide to internal models (2024), general topics: data quality"))
def reconciliation(ctx: RunContext, other, keys, compare=None, tolerance=0.0, rel_tolerance=0.0,
                   sum_columns=None, group=None, max_examples=10) -> Outcome:
    A = ctx.df.reset_index(drop=True)
    B0 = ctx.tables[other]
    bmap = {c: _match_col(B0, c, "in other table") for c in list(keys) + list(compare or []) + list(sum_columns or [])
            + ([group] if group else [])}
    B = B0.reset_index(drop=True).rename(columns={v: k for k, v in bmap.items()})
    common = [c for c in A.columns if c in B.columns and c not in keys]
    compare = list(compare) if compare else common
    ka, kb = _composite(A, keys), _composite(B, keys)
    da, db = ka[ka.notna()].duplicated(keep=False), kb[kb.notna()].duplicated(keep=False)
    notes = [ROW_NOTE + " Row numbers refer to each table separately."]
    if da.any() or db.any():
        notes.append("Duplicate keys found; value comparison uses the first row of each key.")
    first_a = pd.Series(np.arange(len(A)), index=ka.index)[ka.notna() & ~ka.duplicated()]
    first_b = pd.Series(np.arange(len(B)), index=kb.index)[kb.notna() & ~kb.duplicated()]
    ia = pd.Series(first_a.to_numpy(), index=ka[first_a.index].to_numpy())
    ib = pd.Series(first_b.to_numpy(), index=kb[first_b.index].to_numpy())
    # ia / ib are in row order, so these lists are in row order of their table
    only_a = list(ia.index[~ia.index.isin(ib.index)])
    only_b = list(ib.index[~ib.index.isin(ia.index)])
    both = list(ia.index[ia.index.isin(ib.index)])

    def keyrows(lst, idx, frame, label):
        rows = []
        for k in lst[:max_examples]:
            r = {c: frame[c].iloc[idx[k]] for c in keys}
            r[f"row_in_{label}"] = int(idx[k])
            rows.append(r)
        return pd.DataFrame(rows, columns=list(keys) + [f"row_in_{label}"])

    tables = {"Keys only in A": keyrows(only_a, ia, A, "A"), "Keys only in B": keyrows(only_b, ib, B, "B")}
    ra, rb = ia[both].to_numpy(), ib[both].to_numpy()
    diffs, ex = [], []
    for c in compare:
        if c not in B.columns:
            raise ValueError(f"Compare column '{c}' is not in the other table.")
        a, b = A[c].iloc[ra].reset_index(drop=True), B[c].iloc[rb].reset_index(drop=True)
        isnum = pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) \
            and not pd.api.types.is_bool_dtype(a)
        if isnum:
            fa, fb = a.astype(float), b.astype(float)
            d = (fa - fb)
            bad = (d.abs() > tolerance + rel_tolerance * fb.abs()) | (fa.isna() != fb.isna())
            extra = {"max_abs_difference": float(d.abs().max()) if d.notna().any() else np.nan,
                     "sum_difference": float(d.sum())}
        else:
            sa, sb = _canon_key(a), _canon_key(b)
            bad = (sa != sb) & ~(sa.isna() & sb.isna())
            extra = {"max_abs_difference": np.nan, "sum_difference": np.nan}
        bad = bad.fillna(False).astype(bool)
        p = _pos(bad)
        diffs.append({"column": c, "type": "numeric" if isnum else "text", "matched_keys": len(both),
                      "differences": len(p), "difference_pct": len(p) / len(both) if both else np.nan, **extra,
                      "example_rows_A": _join(ra[p], max_examples), "example_rows_B": _join(rb[p], max_examples)})
        for j in p[:max_examples]:
            r = {"column": c, **{k: A[k].iloc[ra[j]] for k in keys}, "row_in_A": int(ra[j]),
                 "row_in_B": int(rb[j]), "value_A": _show(a.iloc[j]), "value_B": _show(b.iloc[j])}
            ex.append(r)
    tables["Value differences by column"] = pd.DataFrame(diffs)
    tables["Value differences (examples)"] = pd.DataFrame(ex)
    sums = list(sum_columns) if sum_columns else [d["column"] for d in diffs if d["type"] == "numeric"]
    tot = []
    for c in sums:
        if c not in B.columns:
            raise ValueError(f"Sum column '{c}' is not in the other table.")
        ta, tb = float(num(A, c).sum()), float(num(B, c).sum())
        tma, tmb = float(num(A, c).iloc[ra].sum()), float(num(B, c).iloc[rb].sum())
        tot.append({"column": c, "total_A": ta, "total_B": tb, "difference": ta - tb,
                    "rel_difference": (ta - tb) / tb if tb else np.nan,
                    "matched_total_A": tma, "matched_total_B": tmb, "matched_difference": tma - tmb})
    if tot:
        tables["Totals"] = pd.DataFrame(tot)
    if group and sums:
        ga = A.assign(_g=A[group].astype("object").where(A[group].notna(), "<missing>").astype(str))
        gb = B.assign(_g=B[group].astype("object").where(B[group].notna(), "<missing>").astype(str))
        rows = []
        for c in sums:
            sa = ga.assign(_v=num(A, c)).groupby("_g")["_v"].sum()
            sb = gb.assign(_v=num(B, c)).groupby("_g")["_v"].sum()
            for g in sorted(set(sa.index) | set(sb.index)):
                x, y = float(sa.get(g, 0.0)), float(sb.get(g, 0.0))
                rows.append({"group": g, "column": c, "total_A": x, "total_B": y, "difference": x - y,
                             "rel_difference": (x - y) / y if y else np.nan})
        tables["Totals by group"] = pd.DataFrame(rows)
    return Outcome({"rows_A": len(A), "rows_B": len(B), "distinct_keys_A": len(ia), "distinct_keys_B": len(ib),
                    "duplicate_key_rows_A": int(da.sum()), "duplicate_key_rows_B": int(db.sum()),
                    "missing_key_rows_A": int(ka.isna().sum()), "missing_key_rows_B": int(kb.isna().sum()),
                    "keys_only_in_A": len(only_a), "keys_only_in_B": len(only_b), "matched_keys": len(both),
                    "columns_with_differences": int(sum(d["differences"] > 0 for d in diffs)),
                    "value_differences": int(sum(d["differences"] for d in diffs))},
                   tables, notes=notes, rows_used=len(A) + len(B))


@register("data.referential_integrity", "Referential integrity (foreign key vs reference table)",
          "Data quality", _ALL,
          params=(P("column", help="Foreign-key column in the active table"),
                  P("other", "table", help="Reference (master) table"),
                  P("other_column", "string", required=False, help="Key column in the reference table; default same name"),
                  *_EVID),
          description="""Foreign-key values of the active table that do not exist in the reference table (orphans):
number of orphan rows and distinct orphan values, each orphan value with its row count and example rows; also
missing foreign keys, duplicate keys in the reference table (which make the reference ambiguous) and reference
keys never used. Values are compared as text after normalising integer-valued numbers (1 and 1.0 match).""",
          references=("BCBS 239 (2013), Principle 3 (accuracy and integrity)",
                      "Codd (1970), A relational model of data for large shared data banks, CACM 13(6)"))
def referential_integrity(ctx: RunContext, column, other, other_column=None, id=None, max_examples=10) -> Outcome:
    df, ref = ctx.df.reset_index(drop=True), ctx.tables[other]
    rc = _match_col(ref, other_column or column, "reference key")
    fk, rk = _canon_key(df[column]), _canon_key(ref[rc])
    rset = set(rk.dropna())
    orphan = fk.notna() & ~fk.isin(rset)
    rows = []
    vals = fk[orphan]
    for v, idx in vals.groupby(vals, sort=False).groups.items():
        pos = np.sort(np.asarray(idx, dtype=int))
        rows.append({"orphan_value": v, "rows": len(pos), "first_row": int(pos[0]), **_ex(df, pos, id, max_examples)})
    out = pd.DataFrame(rows, columns=["orphan_value", "rows", "first_row", "example_rows"]
                       + (["example_ids"] if id else []))
    out = out.sort_values(["rows", "first_row"], ascending=[False, True], kind="mergesort").drop(columns="first_row")
    used = set(fk.dropna())
    return Outcome({"rows": len(df), "orphan_rows": int(orphan.sum()), "orphan_values": len(out),
                    "orphan_pct": float(orphan.mean()) if len(df) else np.nan,
                    "missing_foreign_key": int(fk.isna().sum()), "reference_keys": len(rset),
                    "duplicate_reference_keys": int(rk.dropna().duplicated().sum()),
                    "unused_reference_keys": len(rset - used)},
                   {"Orphan values": out.reset_index(drop=True),
                    "Evidence": _ev_frame(_evidence(df, _pos(orphan), max_examples, "orphan foreign key", column, id),
                                          id)},
                   notes=[ROW_NOTE], rows_used=len(df))


# ── leakage candidates ──────────────────────────────────────────────────────

def _assoc_row(x: pd.Series, y: pd.Series, sem: str) -> dict:
    """AUC/Gini for ordered columns, Cramér's V for categorical ones (y has no NaN)."""
    r = {"auc": np.nan, "gini": np.nan, "cramers_v": np.nan, "p_value": np.nan}
    if sem in {"numeric", "numeric-as-text", "date"} or (sem == "id-like" and pd.api.types.is_numeric_dtype(x)):
        if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            v = x.astype(float)
        elif sem == "date":
            v = (_to_dates(x) - pd.Timestamp("1970-01-01")).dt.total_seconds()
        else:
            v = _numeric_parse(x)
        ok = v.notna()
        if ok.sum() and y[ok].nunique() == 2:
            a = _auc(y[ok].to_numpy(), v[ok].to_numpy())
            n1, n0 = int((y[ok] == 1).sum()), int((y[ok] == 0).sum())
            mw = stats.mannwhitneyu(v[ok & (y == 1)], v[ok & (y == 0)], alternative="two-sided") if n1 and n0 else None
            r.update({"auc": a, "gini": 2 * a - 1, "p_value": float(mw.pvalue) if mw else np.nan})
        r["n_used"] = int(ok.sum())
    elif sem == "id-like":
        ok = x.notna()
        rank = x[ok].astype(str).rank(method="average")
        if y[ok].nunique() == 2:
            a = _auc(y[ok].to_numpy(), rank.to_numpy())
            r.update({"auc": a, "gini": 2 * a - 1})
        r["n_used"] = int(ok.sum())
    else:
        v, chi2, p, _ = _cramers_v(x, y)
        r.update({"cramers_v": v, "p_value": p, "n_used": len(x)})
    cand = [v for v in (abs(r["gini"]), r["cramers_v"]) if not np.isnan(v)]
    r["strength"] = max(cand) if cand else np.nan
    return r


@register("data.leakage_candidates", "Target-leakage candidates in the data", "Data quality", _BINARY,
          params=(P("target"), P("features", "columns", required=False, help="Columns to screen; default all others"),
                  P("date", required=False, help="Observation (snapshot) date column"),
                  P("observation_date", "string", required=False,
                    help="Single observation date for all rows (instead of a date column)"),
                  *_EVID, _DAYFIRST),
          description="""Screens columns that may carry information from after the observation date:
(1) date-like columns with values later than the observation date (row-wise `date` column or a single
`observation_date`) — count, share and example rows; (2) univariate association of every column with the target:
AUC and Gini (with Mann–Whitney p-value) for numeric and date columns, Cramér's V (chi-square p-value) for
categorical ones (missing as its own level); (3) id-like columns, where any association (AUC of the identifier's
order) suggests sequencing or batch effects (e.g. identifiers assigned after default). Columns are sorted by
strength = max(|Gini|, Cramér's V); there is no threshold — near-perfect association (|Gini| or V close to 1) is the
classic leakage signature, but domain review decides. Heuristic screen, not proof of leakage.""",
          references=("Kaufman, Rosset, Perlich & Stitelman (2012), Leakage in data mining: formulation, detection, "
                      "and avoidance, ACM TKDD 6(4)",
                      "EBA/GL/2017/16, Guidelines on PD estimation (risk drivers, information at the observation date)"))
def leakage_candidates(ctx: RunContext, target, features=None, date=None, observation_date=None, id=None,
                       max_examples=10, dayfirst=False) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    y_all = _binary(df, target)
    keep = y_all.notna()
    feats = features or [c for c in df.columns if c not in {target, date, id}]
    d = df[keep]
    y = y_all[keep]
    if y.nunique() < 2:
        raise NotApplicable("Target has a single class.")
    obs = None
    if date:
        obs = _to_dates(df[date], dayfirst)
    elif observation_date:
        obs = pd.Series(pd.Timestamp(observation_date), index=df.index)
    rows, post, ev = [], [], []
    for c in feats:
        sem = _semantic(df[c], c)
        r = {"column": c, "semantic_type": sem, **_assoc_row(d[c], y, sem)}
        rows.append(r)
        if obs is not None and sem == "date":
            fd = _to_dates(df[c], dayfirst)
            later = (fd > obs).fillna(False)
            both = (fd.notna() & obs.notna())
            pos = _pos(later)
            post.append({"column": c, "rows_compared": int(both.sum()), "rows_after_observation": len(pos),
                         "share_after_observation": len(pos) / int(both.sum()) if both.sum() else np.nan,
                         **(_ex(df, pos, id, max_examples) if len(pos) else {"example_rows": ""})})
            ev += _evidence(df, pos, max_examples, "date after observation date", c, id)
    out = pd.DataFrame(rows).sort_values(["strength", "column"], ascending=[False, True], kind="mergesort",
                                         na_position="last")
    tables = {"Association with target": out.reset_index(drop=True)}
    summary = {"columns_screened": len(out), "n": int(keep.sum()),
               "strongest_column": out.iloc[0]["column"], "strongest_strength": float(out.iloc[0]["strength"]),
               "id_like_columns": int((out["semantic_type"] == "id-like").sum())}
    notes = [ROW_NOTE, "AUC below 0.5 means higher values go with non-events; strength uses |Gini|."]
    if post:
        tables["Dates after observation date"] = pd.DataFrame(post)
        tables["Evidence"] = _ev_frame(ev, id)
        summary["date_columns_with_post_observation_values"] = int(sum(p["rows_after_observation"] > 0 for p in post))
    elif obs is None:
        notes.append("No observation date given: post-observation date check skipped.")
    if (~keep).any():
        notes.append(f"{int((~keep).sum())} rows with missing target excluded from association measures.")
    return Outcome(summary, tables, notes=notes, rows_used=int(keep.sum()))


# ── time consistency ────────────────────────────────────────────────────────

@register("data.time_consistency", "Time consistency per entity (date order, overlapping intervals)",
          "Data quality", _ALL,
          params=(P("id", help="Entity identifier"),
                  P("date", required=False, help="Date that should not decrease within an entity in table order"),
                  P("start_date", required=False, help="Validity-interval start column"),
                  P("end_date", required=False, help="Validity-interval end column (missing = open-ended)"),
                  P("inclusive_end", "boolean", default=True,
                    help="End date is inclusive (next interval may start the day after)"),
                  P("max_examples", "integer", default=10), _DAYFIRST),
          description="""(1) With `date`: within each entity, in table order, rows whose date is earlier than the
previous row's date (order reversals), and repeated (entity, date) pairs. (2) With `start_date`/`end_date`:
intervals with start after end; overlapping intervals per entity (sorted by start, an interval overlaps when it
starts on or before the latest end of the entity's earlier intervals — strictly before when inclusive_end is
false); gaps (start more than one day after the latest earlier end for inclusive ends, after it for exclusive
ends). A missing end date means open-ended; rows with a missing start are counted and excluded.""",
          references=("BCBS 239 (2013), Principle 3 (accuracy and integrity)",
                      "Snodgrass (1999), Developing Time-Oriented Database Applications in SQL, ch. 5 (sequenced keys)"))
def time_consistency(ctx: RunContext, id, date=None, start_date=None, end_date=None, inclusive_end=True,
                     max_examples=10, dayfirst=False) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    if not date and not (start_date and end_date):
        raise ValueError("Give `date`, or both `start_date` and `end_date`.")
    summary, tables, ev = {"rows": len(df), "entities": int(df[id].nunique())}, {}, []
    if date:
        d = _to_dates(df[date], dayfirst)
        prev = d.groupby(df[id], sort=False).shift()
        rev = (d < prev).fillna(False)
        dup = d.notna() & pd.DataFrame({"i": _canon_key(df[id]), "d": d}).duplicated(keep=False)
        pos = _pos(rev)
        tables["Date order reversals"] = pd.DataFrame(
            [{"row": int(p), "id": df[id].iloc[p], "date": str(d.iloc[p].date()), "previous_date": str(prev.iloc[p].date())}
             for p in pos[:max_examples]], columns=["row", "id", "date", "previous_date"])
        summary.update({"date_reversals": len(pos), "entities_with_reversals": int(df.loc[rev, id].nunique()),
                        "duplicate_id_date_rows": int(dup.sum())})
        ev += _evidence(df, pos, max_examples, "date earlier than previous row of the same id", date, id)
        ev += _evidence(df, _pos(dup), max_examples, "repeated (id, date)", date, id)
    if start_date and end_date:
        s, e = _to_dates(df[start_date], dayfirst), _to_dates(df[end_date], dayfirst)
        no_start = s.isna()
        e_eff = e.fillna(pd.Timestamp("2900-01-01"))       # open-ended
        inv = (s > e).fillna(False)
        w = pd.DataFrame({"id": _canon_key(df[id]), "s": s, "e": e_eff, "row": np.arange(len(df))})[~no_start]
        w = w.sort_values(["id", "s", "e", "row"], kind="mergesort")
        prev_max = w.groupby("id", sort=False)["e"].cummax().groupby(w["id"], sort=False).shift()
        one = pd.Timedelta(days=1)
        ov = (w["s"] <= prev_max) if inclusive_end else (w["s"] < prev_max)
        gap = (w["s"] > prev_max + one) if inclusive_end else (w["s"] > prev_max)
        ov, gap = ov.fillna(False), gap.fillna(False)
        ex = []
        for i in w.index[ov.to_numpy()][:max_examples]:
            ent = w[w["id"] == w.at[i, "id"]]
            earlier = ent.loc[:i].iloc[:-1]
            hit = earlier[(earlier["e"] >= w.at[i, "s"]) if inclusive_end else (earlier["e"] > w.at[i, "s"])]
            ex.append({"row": int(w.at[i, "row"]), "id": df[id].iloc[w.at[i, "row"]],
                       "start": str(s.iloc[w.at[i, "row"]].date()),
                       "end": str(e.iloc[w.at[i, "row"]].date()) if pd.notna(e.iloc[w.at[i, "row"]]) else "open",
                       "overlaps_row": int(hit["row"].iloc[0]) if len(hit) else -1})
        tables["Overlapping intervals"] = pd.DataFrame(ex, columns=["row", "id", "start", "end", "overlaps_row"])
        ov_rows = np.sort(w.loc[ov, "row"].to_numpy())
        gap_rows = np.sort(w.loc[gap, "row"].to_numpy())
        summary.update({"intervals": int((~no_start).sum()), "missing_start_rows": int(no_start.sum()),
                        "start_after_end": int(inv.sum()), "overlapping_intervals": len(ov_rows),
                        "entities_with_overlaps": int(w.loc[ov, "id"].nunique()), "gaps": len(gap_rows),
                        "open_ended_intervals": int((e.isna() & ~no_start).sum())})
        ev += _evidence(df, _pos(inv), max_examples, "start after end", start_date, id)
        ev += _evidence(df, ov_rows, max_examples, "overlapping interval", start_date, id)
        ev += _evidence(df, gap_rows, max_examples, "gap before interval", start_date, id)
    tables["Evidence"] = _ev_frame(ev, id)
    return Outcome(summary, tables, notes=[ROW_NOTE], rows_used=len(df))


# ── univariate association with target ──────────────────────────────────────

def _iv_table(x: pd.Series, y: pd.Series, bins: int, categorical: bool) -> pd.DataFrame:
    """Per-bin WoE / IV. WoE = ln(%non-events / %events); 0.5 is added to every cell when any cell is 0."""
    if categorical:
        b = x.astype("object").where(x.notna(), "<missing>").astype(str)
    else:
        v = x.astype(float)
        edges = np.unique(np.nanquantile(v.dropna(), np.linspace(0, 1, bins + 1))) if v.notna().any() else np.array([])
        if len(edges) >= 2:
            edges[0], edges[-1] = -np.inf, np.inf
            b = pd.cut(v, edges, include_lowest=True).astype(str)
        else:
            b = v.astype(str)
        b = b.where(v.notna(), "<missing>")
    t = pd.DataFrame({"bin": b, "ev": y, "n": 1}).groupby("bin", sort=False).agg(n=("n", "sum"), events=("ev", "sum"))
    if not categorical:
        order = [str(c) for c in pd.unique(b[v.notna()].iloc[np.argsort(v[v.notna()].to_numpy(), kind="mergesort")])]
        t = t.reindex([o for o in order if o in t.index] + (["<missing>"] if "<missing>" in t.index else []))
    else:
        t = t.sort_index()
    t["non_events"] = t["n"] - t["events"]
    e, ne = t["events"].astype(float), t["non_events"].astype(float)
    if (e == 0).any() or (ne == 0).any():
        e, ne = e + 0.5, ne + 0.5
    de, dn = e / e.sum(), ne / ne.sum()
    t["event_rate"] = t["events"] / t["n"]
    t["woe"] = np.log(dn / de)
    t["iv_contribution"] = (dn - de) * t["woe"]
    return t.reset_index()


@register("data.target_association", "Univariate association with the target (AUC/Gini, Cramér's V, IV)",
          "Data quality", _BINARY,
          params=(P("target"), P("features", "columns", required=False, help="Default all columns except target"),
                  P("bins", "integer", default=10, help="Quantile bins for numeric IV")),
          description="""One row per variable: n, missing share; numeric variables: AUC (Mann–Whitney, ties count
one half; missing excluded), Gini = 2·AUC − 1 and Information Value on `bins` quantile bins (missing its own
bin); categorical variables (non-numeric or ≤ 20 levels): Cramér's V = √(χ²/(n·(min(r,c) − 1))) without
continuity correction, its chi-square p-value, and IV over levels. WoE = ln(%non-events/%events) per bin;
IV = Σ (%non-events − %events)·WoE; when any bin has zero events or non-events, 0.5 is added to every cell of
that variable. Descriptive ranking only — no thresholds.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 6 (WoE and IV)",
                      "Hanley & McNeil (1982), The meaning and use of the area under a ROC curve, Radiology 143",
                      "Cramér (1946), Mathematical Methods of Statistics"))
def target_association(ctx: RunContext, target, features=None, bins=10) -> Outcome:
    df = ctx.df.reset_index(drop=True)
    y_all = _binary(df, target)
    keep = y_all.notna()
    d, y = df[keep], y_all[keep]
    if y.nunique() < 2:
        raise NotApplicable("Target has a single class.")
    feats = features or [c for c in df.columns if c != target]
    rows, detail = [], []
    for f in feats:
        x = d[f]
        isnum = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
        cat = (not isnum) or x.nunique(dropna=True) <= 20
        r = {"variable": f, "type": "categorical" if cat else "numeric", "n": len(x),
             "missing_pct": float(x.isna().mean()), "auc": np.nan, "gini": np.nan, "cramers_v": np.nan,
             "chi2_p_value": np.nan}
        if isnum:
            ok = x.notna()
            if ok.sum() and y[ok].nunique() == 2:
                a = _auc(y[ok].to_numpy(), x[ok].astype(float).to_numpy())
                r.update({"auc": a, "gini": 2 * a - 1})
        if cat:
            v, _, p, _ = _cramers_v(x, y)
            r.update({"cramers_v": v, "chi2_p_value": p})
        t = _iv_table(x, y, bins, cat)
        r["iv"] = float(t["iv_contribution"].sum())
        r["bins"] = len(t)
        rows.append(r)
        detail.append(t.assign(variable=f))
    out = pd.DataFrame(rows).sort_values(["iv", "variable"], ascending=[False, True], kind="mergesort")
    woe = pd.concat(detail, ignore_index=True)
    woe = woe[["variable"] + [c for c in woe.columns if c != "variable"]]
    return Outcome({"variables": len(out), "n": int(keep.sum()), "event_rate": float(y.mean()),
                    "max_iv": float(out["iv"].max()), "max_iv_variable": out.iloc[0]["variable"]},
                   {"Univariate association": out.reset_index(drop=True), "WoE by bin": woe},
                   notes=[f"{int((~keep).sum())} rows with missing target excluded."] if (~keep).any() else [],
                   rows_used=int(keep.sum()))
