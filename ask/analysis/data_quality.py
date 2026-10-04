"""Data-quality checks.

Copied VERBATIM from the original single-file app (lines 14826-15016).
Pure pandas/matplotlib, no LLM.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd

ACCENT = "#334155"   # histogram colour, from the original app


DQ_MAX_PLOT_COLS = 12   # cap individually-plotted columns so render time/size stay sane


def _fig_to_png_bytes(fig) -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def run_data_quality_checks(df: pd.DataFrame, log_fn=print) -> dict:
    """Basic data-quality assessment over a loaded DataFrame. Returns a dict
    shaped like the Test Lab's per-category results ({"tables", "mpl_pngs",
    ...}) so it can reuse the same table/image rendering and chat-context
    serialization patterns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    log_fn(f"Assessing {n_rows:,} rows × {n_cols} columns "
           f"({len(numeric_cols)} numeric, {len(cat_cols)} categorical/other) …")

    # ── Missing values ──
    log_fn("Computing missing-value assessment …")
    miss_count = df.isna().sum()
    miss_pct = (miss_count / max(n_rows, 1) * 100).round(2)
    missing_tbl = pd.DataFrame({
        "column":        df.columns,
        "dtype":         [str(df[c].dtype) for c in df.columns],
        "missing_count": miss_count.values,
        "missing_pct":   miss_pct.values,
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    nonzero_miss = missing_tbl[missing_tbl["missing_count"] > 0]
    miss_png = None
    if len(nonzero_miss):
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(nonzero_miss))))
        ax.barh(nonzero_miss["column"][::-1], nonzero_miss["missing_pct"][::-1], color="#B91C1C")
        ax.set_xlabel("Missing %")
        ax.set_title("Missing values by column")
        fig.tight_layout()
        miss_png = _fig_to_png_bytes(fig)

    # ── Descriptive statistics ──
    log_fn("Computing descriptive statistics …")
    desc_tbl = (df.describe(include="all").transpose()
                .reset_index().rename(columns={"index": "column"}))

    # ── Outlier assessment (IQR method, numeric columns) ──
    log_fn("Running outlier assessment (IQR method) on numeric columns …")
    outlier_rows = []
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            lo = hi = None
            n_out = 0
        else:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((s < lo) | (s > hi)).sum())
        outlier_rows.append({
            "column": c, "q1": q1, "q3": q3, "iqr": iqr,
            "lower_bound": lo, "upper_bound": hi,
            "outlier_count": n_out,
            "outlier_pct": round(n_out / len(s) * 100, 2) if len(s) else 0.0,
        })
    outlier_tbl = (pd.DataFrame(outlier_rows).sort_values("outlier_pct", ascending=False)
                   .reset_index(drop=True) if outlier_rows
                   else pd.DataFrame(columns=["column", "outlier_count", "outlier_pct"]))

    plot_numeric = numeric_cols[:DQ_MAX_PLOT_COLS]

    def _grid(cols: list[str], per_row: int, cell_w: float, cell_h: float):
        nrows = math.ceil(len(cols) / per_row)
        fig, axes = plt.subplots(nrows, per_row, figsize=(cell_w * per_row, cell_h * nrows))
        axes = np.array(axes).reshape(-1)
        for ax in axes[len(cols):]:
            ax.axis("off")
        return fig, axes

    # ── Outlier boxplots (numeric columns) ──
    box_png = None
    if plot_numeric:
        fig, axes = _grid(plot_numeric, min(4, len(plot_numeric)), 4, 3)
        for ax, c in zip(axes, plot_numeric):
            ax.boxplot(df[c].dropna(), vert=True)
            ax.set_title(c, fontsize=10)
        fig.suptitle("Outlier boxplots (numeric columns)"
                     + (f" — first {DQ_MAX_PLOT_COLS}" if len(numeric_cols) > DQ_MAX_PLOT_COLS else ""))
        fig.tight_layout()
        box_png = _fig_to_png_bytes(fig)

    # ── Distribution histograms (numeric columns) ──
    log_fn("Rendering histograms for numeric columns …")
    hist_png = None
    if plot_numeric:
        fig, axes = _grid(plot_numeric, min(4, len(plot_numeric)), 4, 3)
        for ax, c in zip(axes, plot_numeric):
            ax.hist(df[c].dropna(), bins=30, color=ACCENT)
            ax.set_title(c, fontsize=10)
        fig.suptitle("Distribution histograms (numeric columns)"
                     + (f" — first {DQ_MAX_PLOT_COLS}" if len(numeric_cols) > DQ_MAX_PLOT_COLS else ""))
        fig.tight_layout()
        hist_png = _fig_to_png_bytes(fig)

    # ── Frequency plots (categorical columns; skip very high cardinality) ──
    log_fn("Rendering frequency plots for categorical columns …")
    plot_cat = [c for c in cat_cols if df[c].nunique(dropna=True) <= 50][:DQ_MAX_PLOT_COLS]
    skipped_cat = [c for c in cat_cols if c not in plot_cat]
    freq_png = None
    if plot_cat:
        fig, axes = _grid(plot_cat, min(3, len(plot_cat)), 5, 3.2)
        for ax, c in zip(axes, plot_cat):
            vc = df[c].value_counts(dropna=True).head(10)
            ax.bar(vc.index.astype(str), vc.values, color="#0e7490")
            ax.set_title(c, fontsize=10)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
        fig.suptitle("Frequency plots (categorical columns, top 10 values)"
                     + (" — high-cardinality columns skipped" if skipped_cat else ""))
        fig.tight_layout()
        freq_png = _fig_to_png_bytes(fig)

    # ── Data validity checks ──
    log_fn("Running data validity checks …")
    n_dupe_rows = int(df.duplicated().sum())
    validity_rows = []
    for c in df.columns:
        s = df[c]
        n_unique = s.nunique(dropna=True)
        validity_rows.append({
            "column":      c,
            "dtype":       str(s.dtype),
            "n_unique":    n_unique,
            "pct_unique":  round(n_unique / max(n_rows, 1) * 100, 2),
            "is_constant": bool(n_unique <= 1),
            "is_id_like":  bool(n_unique == n_rows and n_rows > 0),
        })
    validity_tbl = pd.DataFrame(validity_rows)
    constant_cols = validity_tbl.loc[validity_tbl["is_constant"], "column"].tolist()
    id_like_cols  = validity_tbl.loc[validity_tbl["is_id_like"], "column"].tolist()

    validity_summary = pd.DataFrame([
        {"check": "Duplicate rows", "value": n_dupe_rows,
         "detail": f"{n_dupe_rows / max(n_rows, 1) * 100:.2f}% of rows" if n_rows else ""},
        {"check": "Constant columns", "value": len(constant_cols),
         "detail": ", ".join(constant_cols) if constant_cols else "none"},
        {"check": "ID-like columns (100% unique)", "value": len(id_like_cols),
         "detail": ", ".join(id_like_cols) if id_like_cols else "none"},
        {"check": "Columns with any missing values", "value": int((miss_count > 0).sum()),
         "detail": ", ".join(nonzero_miss["column"].tolist()) if len(nonzero_miss) else "none"},
    ])

    summary_text = (
        f"{n_rows:,} rows × {n_cols} columns · {len(numeric_cols)} numeric · "
        f"{len(cat_cols)} categorical/other · {n_dupe_rows:,} duplicate row(s) · "
        f"{int((miss_count > 0).sum())} column(s) with missing values."
    )

    tables = {
        "Missing Values":            missing_tbl,
        "Validity Checks":           validity_summary,
        "Outlier Summary (IQR)":     outlier_tbl,
        "Column Validity":           validity_tbl,
        "Descriptive Statistics":    desc_tbl,
    }
    png_pairs = [
        ("Missing values by column",              miss_png),
        ("Distribution histograms",                hist_png),
        ("Outlier boxplots",                       box_png),
        ("Categorical frequency plots (top 10)",   freq_png),
    ]
    mpl_pngs      = [png for _, png in png_pairs if png is not None]
    png_captions  = [cap for cap, png in png_pairs if png is not None]

    log_fn("Data quality assessment complete.")
    return {
        "tables":                  tables,
        "mpl_pngs":                mpl_pngs,
        "png_captions":            png_captions,
        "summary":                 summary_text,
        "skipped_high_cardinality": skipped_cat,
    }

