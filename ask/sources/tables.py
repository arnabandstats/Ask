"""Unity Catalog / Delta tables: load (small or sampled), profile and aggregate in SQL.

Two backends, tried in order:
  1. an active Spark session (a notebook or job on a Databricks cluster, or
     Databricks Connect when it is installed and configured);
  2. a Databricks SQL warehouse through the databricks-sdk Statement Execution API
     (set ASK_SQL_WAREHOUSE_ID). This works from a Databricks App and from a laptop.

Big tables are never pulled whole. load_table() counts rows first; above
config.MAX_TABLE_ROWS it takes a DETERMINISTIC hash sample (xxhash64 of the key
columns, or of every selected column) and orders rows by that hash, so the
same table and settings always give the same rows in the same order — and so
the same data fingerprint and test results. profile_table() and query_table()
compute on the full population inside the warehouse.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from ask import config

_IDENT = re.compile(r"^[A-Za-z0-9_`\-]+(\.[A-Za-z0-9_`\-]+){0,2}$")
_FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|"
                        r"copy|optimize|vacuum|refresh|call|set|use|msck|restore|clone)\b", re.I)
HASH_BUCKETS = 1_000_000


class TableError(Exception):
    """Readable error for the chat."""


@dataclass
class TableLoad:
    df: pd.DataFrame
    table: str
    total_rows: int
    sampled: bool
    sample_rule: str
    sql: str
    backend: str

    def describe(self) -> str:
        s = (f"{self.table}: {len(self.df):,} rows × {self.df.shape[1]} columns loaded via {self.backend}"
             f" (table has {self.total_rows:,} rows matching the filter)")
        if self.sampled:
            s += f". DETERMINISTIC SAMPLE: {self.sample_rule}"
        return s


# ── identifiers and SQL safety ─────────────────────────────────────────────

def _q(name: str) -> str:
    """Quote a table or column identifier with backticks, part by part."""
    name = name.strip()
    if not _IDENT.match(name):
        raise TableError(f"`{name}` is not a valid identifier (expected catalog.schema.table or a column).")
    return ".".join(f"`{p.strip('`')}`" for p in name.split("."))


def check_select(sql: str) -> str:
    """Only one read-only SELECT / WITH statement."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise TableError("Only one SQL statement is allowed.")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise TableError("Only SELECT (or WITH … SELECT) queries are allowed.")
    bad = _FORBIDDEN.search(re.sub(r"'[^']*'", "''", s))
    if bad:
        raise TableError(f"`{bad.group(0)}` is not allowed in a read-only query.")
    return s


def _where(where: str | None) -> str:
    if not where:
        return ""
    w = where.strip()
    if ";" in w or _FORBIDDEN.search(re.sub(r"'[^']*'", "''", w)):
        raise TableError("The filter must be a plain WHERE condition.")
    return f" WHERE {w}"


# ── backends ───────────────────────────────────────────────────────────────

def _spark():
    try:
        from pyspark.sql import SparkSession
        s = SparkSession.getActiveSession()
        if s is not None:
            return s
    except Exception:
        pass
    try:
        from databricks.connect import DatabricksSession
        return DatabricksSession.builder.getOrCreate()
    except Exception:
        return None


def _warehouse_sql(sql: str) -> pd.DataFrame:
    import os
    wid = os.getenv("ASK_SQL_WAREHOUSE_ID", "").strip()
    if not wid:
        raise TableError("No Spark session here and ASK_SQL_WAREHOUSE_ID is not set. Set it to a SQL "
                         "warehouse ID (SQL Warehouses → your warehouse → Connection details).")
    from ask.sources.databricks import DatabricksError, client
    try:
        w = client()
    except DatabricksError as exc:
        raise TableError(str(exc)) from exc
    from databricks.sdk.service.sql import Disposition, Format, StatementState
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wid, wait_timeout="50s",
        disposition=Disposition.EXTERNAL_LINKS, format=Format.ARROW_STREAM)
    import time
    while resp.status and resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        resp = w.statement_execution.get_statement(resp.statement_id)
    state = resp.status.state if resp.status else None
    if state != StatementState.SUCCEEDED:
        err = resp.status.error.message if resp.status and resp.status.error else state
        raise TableError(f"SQL failed: {err}")
    import io
    import urllib.request

    import pyarrow as pa
    frames = []
    chunk = resp.result
    while chunk is not None:
        for link in chunk.external_links or []:
            with urllib.request.urlopen(link.external_link, timeout=300) as fh:   # pre-signed, no auth header
                frames.append(pa.ipc.open_stream(io.BytesIO(fh.read())).read_all().to_pandas())
        nxt = chunk.next_chunk_index if chunk.external_links else None
        chunk = (w.statement_execution.get_statement_result_chunk_n(resp.statement_id, nxt)
                 if nxt is not None else None)
    if not frames:
        cols = [c.name for c in (resp.manifest.schema.columns or [])] if resp.manifest else []
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def run_sql(sql: str) -> tuple[pd.DataFrame, str]:
    spark = _spark()
    if spark is not None:
        return spark.sql(sql).toPandas(), "Spark"
    return _warehouse_sql(sql), "SQL warehouse"


# ── public API ─────────────────────────────────────────────────────────────

def load_table(table: str, columns: list[str] | None = None, where: str | None = None,
               max_rows: int | None = None, key: list[str] | None = None) -> TableLoad:
    max_rows = max_rows or config.MAX_TABLE_ROWS
    t = _q(table)
    cols = ", ".join(_q(c) for c in columns) if columns else "*"
    w = _where(where)
    total = int(run_sql(f"SELECT COUNT(*) AS n FROM {t}{w}")[0].iloc[0, 0])
    hash_cols = ", ".join(_q(c) for c in (key or columns or [])) or "*"
    h = f"pmod(xxhash64({hash_cols}), {HASH_BUCKETS})"
    sampled, rule = total > max_rows, ""
    if sampled:
        k = max(1, math.floor(HASH_BUCKETS * max_rows / total))
        cond = f"{h} < {k}"
        w = f"{w} AND {cond}" if w else f" WHERE {cond}"
        rule = (f"rows with pmod(xxhash64({hash_cols}), {HASH_BUCKETS}) < {k} "
                f"(≈{k / HASH_BUCKETS:.2%} of {total:,} rows); same rows every run")
    order = ", ".join(_q(c) for c in key) if key else h
    sql = f"SELECT {cols} FROM {t}{w} ORDER BY {order}"
    df, backend = run_sql(sql)
    return TableLoad(df, table, total, sampled, rule, sql, backend)


def profile_table(table: str, columns: list[str] | None = None, key: list[str] | None = None,
                  where: str | None = None) -> tuple[pd.DataFrame, dict, str]:
    """Full-population column profile computed in SQL: count, nulls, distinct, min, max,
    mean and std for numeric columns; duplicate count on the key."""
    t, w = _q(table), _where(where)
    schema, _ = run_sql(f"DESCRIBE TABLE {t}")
    schema = schema[~schema.iloc[:, 0].astype(str).str.startswith("#")]
    types = {str(r.iloc[0]): str(r.iloc[1]).lower() for _, r in schema.iterrows() if str(r.iloc[0]).strip()}
    cols = columns or list(types)
    numeric = ("int", "bigint", "smallint", "tinyint", "double", "float", "decimal", "long")
    parts = ["COUNT(*) AS `__rows`"]
    for i, c in enumerate(cols):
        qc = _q(c)
        parts += [f"COUNT({qc}) AS `nn_{i}`", f"COUNT(DISTINCT {qc}) AS `nd_{i}`",
                  f"CAST(MIN({qc}) AS STRING) AS `mn_{i}`", f"CAST(MAX({qc}) AS STRING) AS `mx_{i}`"]
        if types.get(c, "").startswith(numeric):
            parts += [f"AVG({qc}) AS `av_{i}`", f"STDDEV_SAMP({qc}) AS `sd_{i}`"]
    agg, backend = run_sql(f"SELECT {', '.join(parts)} FROM {t}{w}")
    r = agg.iloc[0]
    n = int(r["__rows"])
    rows = []
    for i, c in enumerate(cols):
        nn = int(r[f"nn_{i}"])
        rows.append({"column": c, "type": types.get(c, ""), "rows": n, "missing": n - nn,
                     "missing_pct": (n - nn) / n if n else float("nan"), "distinct": int(r[f"nd_{i}"]),
                     "min": r[f"mn_{i}"], "max": r[f"mx_{i}"],
                     "mean": r.get(f"av_{i}"), "std": r.get(f"sd_{i}")})
    extra = {"rows": n, "backend": backend}
    if key:
        k = ", ".join(_q(c) for c in key)
        d, _ = run_sql(f"SELECT COUNT(*) AS groups, COALESCE(SUM(c), 0) AS rows FROM "
                       f"(SELECT {k}, COUNT(*) AS c FROM {t}{w} GROUP BY {k} HAVING COUNT(*) > 1)")
        extra["duplicate_keys"] = int(d.iloc[0, 0])
        extra["rows_in_duplicate_keys"] = int(d.iloc[0, 1])
    return pd.DataFrame(rows), extra, backend


def query_table(sql: str, max_rows: int = 200_000) -> tuple[pd.DataFrame, str, str]:
    s = check_select(sql)
    df, backend = run_sql(f"SELECT * FROM ({s}) AS q LIMIT {int(max_rows) + 1}")
    if len(df) > max_rows:
        raise TableError(f"The query returns more than {max_rows:,} rows. Aggregate it in SQL "
                         "(GROUP BY) or use load_table, which samples deterministically.")
    return df, s, backend


def looks_like_table(text: str) -> bool:
    s = text.strip().strip("`'\"")
    return bool(re.match(r"^[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*\.[A-Za-z_][\w-]*$", s)) and "/" not in s
