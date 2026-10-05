"""Population and characteristic stability: PSI, CSI, distribution-shift tests.

All comparisons are reference sample vs current sample, taken either from a
`sample` column of the active table or from a second loaded table (`other`).
Bins are always fixed on the REFERENCE sample, so the result does not change
when the current sample changes size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import (NotApplicable, Outcome, P, RunContext, num, register,
                                 split_samples)

_ALL = ("pd", "lgd", "ead", "ifrs9", "ews", "aml", "ml_classification", "ml_regression",
        "ml_unsupervised", "satellite", "general")

_SAMPLES = (
    P("sample", help="Sample column; omit when comparing with another table (other)", required=False),
    P("reference_value", "string", required=False),
    P("current_value", "string", required=False),
    P("other", "table", help="Second loaded table holding the current sample (instead of sample column)",
      required=False),
)


def _ref_cur(ctx: RunContext, sample, reference_value, current_value, other):
    if other is not None:
        return ctx.df, ctx.tables[other], ctx.source_name or "reference", other
    if not sample:
        raise ValueError("Give either `sample` (a sample column) or `other` (a second table).")
    return split_samples(ctx.df, sample, reference_value, current_value)


def _is_categorical(s: pd.Series, max_levels: int = 20) -> bool:
    return (not pd.api.types.is_numeric_dtype(s)) or s.nunique(dropna=True) <= max_levels


def psi_table(ref: pd.Series, cur: pd.Series, bins: int = 10, eps: float = 1e-4,
              categorical: bool | None = None) -> pd.DataFrame:
    """Per-bin PSI contributions. Numeric: quantile bins on the reference, open-ended
    outer edges. Categorical (or <= 20 levels): one bin per level. Missing values are
    their own bin, because a shift in missingness is itself instability."""
    if categorical is None:
        categorical = _is_categorical(ref)
    if categorical:
        r = ref.astype("object").where(ref.notna(), "<missing>").astype(str)
        c = cur.astype("object").where(cur.notna(), "<missing>").astype(str)
        levels = sorted(set(r) | set(c), key=str)
        rc = r.value_counts().reindex(levels, fill_value=0)
        cc = c.value_counts().reindex(levels, fill_value=0)
        labels = levels
    else:
        rn, cn = pd.to_numeric(ref, errors="coerce"), pd.to_numeric(cur, errors="coerce")
        edges = np.unique(np.nanquantile(rn.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([rn.min(), rn.max()])
        edges[0], edges[-1] = -np.inf, np.inf
        rb, cb = pd.cut(rn, edges, include_lowest=True), pd.cut(cn, edges, include_lowest=True)
        cats = rb.cat.categories
        rc = rb.value_counts().reindex(cats, fill_value=0)
        cc = cb.value_counts().reindex(cats, fill_value=0)
        labels = [str(c) for c in cats]
        if rn.isna().any() or cn.isna().any():
            rc = pd.concat([rc, pd.Series([int(rn.isna().sum())])])
            cc = pd.concat([cc, pd.Series([int(cn.isna().sum())])])
            labels = labels + ["<missing>"]
    rc, cc = np.asarray(rc, float), np.asarray(cc, float)
    rp = np.maximum(rc / max(rc.sum(), 1), eps)
    cp = np.maximum(cc / max(cc.sum(), 1), eps)
    contrib = (cp - rp) * np.log(cp / rp)
    return pd.DataFrame({"bin": labels, "ref_count": rc.astype(int), "cur_count": cc.astype(int),
                         "ref_pct": rc / max(rc.sum(), 1), "cur_pct": cc / max(cc.sum(), 1),
                         "psi_contribution": contrib})


@register("stability.psi", "Population Stability Index (PSI)", "Stability", _ALL,
          params=(P("column", help="Score, PD, grade or any variable to compare"), *_SAMPLES,
                  P("bins", "integer", default=10, help="Quantile bins for numeric columns"),
                  P("categorical", "boolean", required=False,
                    help="Force categorical binning (default: auto, <= 20 levels)")),
          description="""PSI = Σ (cur% − ref%)·ln(cur%/ref%) over bins fixed on the reference sample.
Missing values form their own bin. Reported per bin so the driving bins are visible.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 8",
                      "Yurdakul (2018), Statistical properties of the population stability index"))
def psi(ctx: RunContext, column, sample=None, reference_value=None, current_value=None,
        other=None, bins=10, categorical=None) -> Outcome:
    ref, cur, rl, cl = _ref_cur(ctx, sample, reference_value, current_value, other)
    if column not in cur.columns:
        raise ValueError(f"'{column}' is not in the current sample table")
    t = psi_table(ref[column], cur[column], bins, categorical=categorical)
    total = float(t["psi_contribution"].sum())
    # Under H0 (same distribution), n·PSI ~ chi2(k-1) approximately with n = 1/(1/n_ref + 1/n_cur).
    n_eff = 1 / (1 / max(len(ref), 1) + 1 / max(len(cur), 1))
    k = int((t["ref_count"] + t["cur_count"] > 0).sum())
    p = float(stats.chi2.sf(total * n_eff, max(k - 1, 1)))
    return Outcome({"PSI": total, "bins": k, "chi2_approx_p_value": p, "reference": rl, "current": cl,
                    "n_reference": len(ref), "n_current": len(cur)},
                   {"PSI by bin": t}, rows_used=len(ref) + len(cur),
                   notes=["p-value uses the asymptotic chi-square approximation of n·PSI (Yurdakul 2018)."])


@register("stability.csi", "Characteristic Stability Index (CSI) for many variables", "Stability", _ALL,
          params=(P("features", "columns", required=False, help="Variables; default all except sample"),
                  *_SAMPLES, P("bins", "integer", default=10)),
          description="""PSI computed for every characteristic, sorted by value. Shows which input
variables drove a change in the score distribution.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, ch. 8",))
def csi(ctx: RunContext, features=None, sample=None, reference_value=None, current_value=None,
        other=None, bins=10) -> Outcome:
    ref, cur, rl, cl = _ref_cur(ctx, sample, reference_value, current_value, other)
    feats = features or [c for c in ref.columns if c != sample and c in cur.columns]
    rows = []
    for f in feats:
        t = psi_table(ref[f], cur[f], bins)
        rows.append({"variable": f, "CSI": float(t["psi_contribution"].sum()), "bins": len(t),
                     "ref_missing_pct": float(ref[f].isna().mean()),
                     "cur_missing_pct": float(cur[f].isna().mean())})
    out = pd.DataFrame(rows).sort_values("CSI", ascending=False)
    if out.empty:
        raise NotApplicable("No common variables to compare.")
    return Outcome({"variables": len(out), "max_CSI": float(out["CSI"].max()),
                    "max_CSI_variable": out.iloc[0]["variable"], "reference": rl, "current": cl},
                   {"CSI by variable": out}, rows_used=len(ref) + len(cur))


@register("stability.distribution_tests", "Two-sample distribution tests (KS, AD, Wasserstein, JS)",
          "Stability", _ALL,
          params=(P("column"), *_SAMPLES, P("bins", "integer", default=10)),
          description="""Numeric column: two-sample Kolmogorov–Smirnov, Anderson–Darling k-sample,
Mann–Whitney U, Wasserstein distance, Jensen–Shannon distance (on reference quantile bins).
Categorical column: chi-square test of homogeneity and Jensen–Shannon distance.
H0 for every test: both samples come from the same distribution.""",
          references=("scipy.stats ks_2samp / anderson_ksamp / mannwhitneyu / chi2_contingency",))
def distribution_tests(ctx: RunContext, column, sample=None, reference_value=None, current_value=None,
                       other=None, bins=10) -> Outcome:
    ref, cur, rl, cl = _ref_cur(ctx, sample, reference_value, current_value, other)
    r, c = ref[column], cur[column]
    rows = []
    t = psi_table(r, c, bins)
    rp, cp = t["ref_pct"].to_numpy(), t["cur_pct"].to_numpy()
    from scipy.spatial.distance import jensenshannon
    js = float(jensenshannon(rp, cp, base=2)) if rp.sum() and cp.sum() else float("nan")
    if _is_categorical(r):
        ct = np.vstack([t["ref_count"], t["cur_count"]])
        ct = ct[:, ct.sum(axis=0) > 0]
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        rows.append({"test": "Chi-square homogeneity", "statistic": chi2, "p_value": p, "dof": dof})
    else:
        rn, cn = num(ref, column).dropna(), num(cur, column).dropna()
        if len(rn) < 2 or len(cn) < 2:
            raise NotApplicable("Fewer than 2 numeric values in a sample.")
        ks = stats.ks_2samp(rn, cn)
        rows.append({"test": "Kolmogorov–Smirnov 2-sample", "statistic": ks.statistic, "p_value": ks.pvalue})
        ad = stats.anderson_ksamp([rn.to_numpy(), cn.to_numpy()])
        rows.append({"test": "Anderson–Darling k-sample", "statistic": ad.statistic,
                     "p_value": float(ad.pvalue if hasattr(ad, "pvalue") else ad.significance_level)})
        mw = stats.mannwhitneyu(rn, cn, alternative="two-sided")
        rows.append({"test": "Mann–Whitney U", "statistic": mw.statistic, "p_value": mw.pvalue})
        rows.append({"test": "Wasserstein distance", "statistic": stats.wasserstein_distance(rn, cn),
                     "p_value": np.nan})
    rows.append({"test": "Jensen–Shannon distance (base 2)", "statistic": js, "p_value": np.nan})
    out = pd.DataFrame(rows)
    return Outcome({r_["test"]: r_["statistic"] for r_ in rows} | {"reference": rl, "current": cl},
                   {"Distribution tests": out}, rows_used=len(ref) + len(cur),
                   notes=["Anderson–Darling p-values are capped by scipy to [0.001, 0.25]."]
                   if not _is_categorical(r) else [])


@register("stability.psi_over_time", "PSI of each period against a reference period", "Stability", _ALL,
          params=(P("column"), P("period"), P("reference_value", "string", required=False,
                                              help="Reference period; default the first period"),
                  P("bins", "integer", default=10)),
          description="PSI of the column for every period versus the reference period (bins on the reference).",
          references=("Siddiqi (2006)",))
def psi_over_time(ctx: RunContext, column, period, reference_value=None, bins=10) -> Outcome:
    df = ctx.df
    periods = sorted(df[period].dropna().unique(), key=lambda v: (str(type(v)), v))
    if len(periods) < 2:
        raise NotApplicable("Need at least two periods.")
    ref_p = periods[0] if reference_value is None else next(
        (p for p in periods if str(p) == str(reference_value)), None)
    if ref_p is None:
        raise ValueError(f"Reference period {reference_value} not found; periods: {periods[:12]}")
    ref = df.loc[df[period] == ref_p, column]
    rows = [{"period": str(p), "n": int((df[period] == p).sum()),
             "PSI": float(psi_table(ref, df.loc[df[period] == p, column], bins)["psi_contribution"].sum())}
            for p in periods]
    out = pd.DataFrame(rows)
    return Outcome({"reference_period": str(ref_p), "periods": len(periods),
                    "max_PSI": float(out["PSI"].max()),
                    "max_PSI_period": out.loc[out["PSI"].idxmax(), "period"]},
                   {"PSI by period": out}, rows_used=int(df[period].notna().sum()))
