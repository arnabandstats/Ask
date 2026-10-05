"""EAD / CCF model validation: realised CCF replication, back-testing, coverage, distribution, ranking.

Realised CCF (reference-date approach) per defaulted facility:
    CCF = (EAD_default − drawn_ref) / (limit_ref − drawn_ref)
with drawn and limit observed at the reference date (typically 12 months before default). Facilities whose
undrawn amount limit − drawn is <= `min_undrawn` (fully drawn or overdrawn) have no defined CCF; they are
excluded from CCF statistics and counted separately. Wherever a test takes the realised CCF it can be
given directly (`actual`) or computed from `ead`, `limit` and `drawn`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, dropped_note, num, register
from ask.validation.t_lgd import (_frame, correlation_outcome, distribution_tables, gauc, sorted_levels,
                                  ttest_rows)

_EAD = ("ead",)

_CCF_INPUTS = (
    P("actual", required=False, help="Realised CCF (give this, or ead + limit + drawn)"),
    P("ead", required=False, help="EAD at default (realised drawn amount at default)"),
    P("limit", required=False, help="Credit limit at the reference date"),
    P("drawn", required=False, help="Drawn amount at the reference date"),
    P("min_undrawn", "number", default=0.0, help="Facilities with limit − drawn <= this have no CCF (excluded)"),
)


def _ccf(ctx: RunContext, actual, ead, limit, drawn, min_undrawn, numeric=None, cats=None
         ) -> tuple[pd.DataFrame, list[str]]:
    """Frame with column `real` = realised CCF (+ requested extras), and notes on exclusions."""
    numeric = dict(numeric or {})
    if actual:
        d, dropped = _frame(ctx, {"real": actual, **numeric}, cats)
        return d, dropped_note(dropped)
    if not (ead and limit and drawn):
        raise ValueError("Give the realised CCF (`actual`) or all of `ead`, `limit` and `drawn`.")
    d, dropped = _frame(ctx, {"ead": ead, "limit": limit, "drawn": drawn, **numeric}, cats)
    und = d["limit"] - d["drawn"]
    excl = und <= min_undrawn
    notes = dropped_note(dropped)
    if excl.any():
        notes.append(f"{int(excl.sum())} facilities with undrawn amount <= {min_undrawn:g} have no defined CCF "
                     "and are excluded.")
    d = d[~excl].copy()
    d["real"] = (d["ead"] - d["drawn"]) / (d["limit"] - d["drawn"])
    return d, notes


@register("ead.realised_ccf", "Realised CCF computation and profile", "Replication", _EAD,
          params=(P("ead", help="EAD at default"), P("limit", help="Limit at the reference date"),
                  P("drawn", help="Drawn amount at the reference date"),
                  P("segment", required=False),
                  P("min_undrawn", "number", default=0.0, help="Undrawn amount at or below which CCF is undefined"),
                  P("utilisation_bands", "list", default=[0.25, 0.5, 0.75, 0.9],
                    help="Edges of drawn/limit bands for the CCF-by-utilisation table")),
          description="""Recomputes the realised CCF = (EAD − drawn) / (limit − drawn) per facility (reference-date
approach) and profiles it: mean, median, undrawn-weighted CCF Σ(EAD − drawn)/Σ(limit − drawn), counts below 0
(repayments before default) and above 1 (drawings beyond the limit), mean after flooring at 0 and capping at 1,
per segment and per utilisation band (drawn / limit at the reference date). Facilities with undrawn amount
≤ `min_undrawn` are excluded from the CCF and reported separately (count, overdrawn count, mean EAD / limit
and mean EAD − drawn), since a CCF is not defined for them.""",
          references=("Regulation (EU) No 575/2013 (CRR), Articles 166 and 182 (own estimates of conversion factors)",
                      "EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted "
                      "exposures (realised conversion factors, fixed-horizon reference date)",
                      "Jacobs (2010), An empirical study of exposure at default, Journal of Advanced Studies in "
                      "Finance 1(1)"))
def ead_realised_ccf(ctx: RunContext, ead, limit, drawn, segment=None, min_undrawn=0.0,
                     utilisation_bands=(0.25, 0.5, 0.75, 0.9)) -> Outcome:
    d, dropped = _frame(ctx, {"ead": ead, "limit": limit, "drawn": drawn}, {"seg": segment})
    if d.empty:
        raise NotApplicable("No complete facilities.")
    und = d["limit"] - d["drawn"]
    excl = und <= min_undrawn
    x, k = d[excl], d[~excl].copy()
    if k.empty:
        raise NotApplicable("No facility has an undrawn amount above min_undrawn.")
    k["ccf"] = (k["ead"] - k["drawn"]) / (k["limit"] - k["drawn"])

    def block(s: pd.DataFrame) -> dict:
        c = s["ccf"]
        return {"n": len(s), "mean_ccf": float(c.mean()), "median_ccf": float(c.median()),
                "undrawn_weighted_ccf": float((s["ead"] - s["drawn"]).sum() / (s["limit"] - s["drawn"]).sum()),
                "mean_ccf_floored_capped": float(c.clip(0, 1).mean()), "n_ccf_below_0": int((c < 0).sum()),
                "n_ccf_above_1": int((c > 1).sum()), "sd_ccf": float(c.std(ddof=1)) if len(c) > 1 else np.nan}

    groups = [(g, k[k["seg"] == g]) for g in sorted_levels(k["seg"])] if segment else []
    seg_t = pd.DataFrame([{"group": str(g)} | block(s) for g, s in groups + [("ALL", k)] if len(s)])
    edges = sorted(float(v) for v in utilisation_bands)
    has_lim = k["limit"] > 0
    util = (k.loc[has_lim, "drawn"] / k.loc[has_lim, "limit"])
    band = pd.cut(util, [-np.inf, *edges, np.inf], right=False)
    util_rows = []
    for cat in band.cat.categories:
        s = k.loc[util.index[band == cat]]
        if len(s):
            util_rows.append({"utilisation_band": str(cat), "mean_utilisation": float(util[band == cat].mean())}
                             | block(s))
    a = seg_t.iloc[-1]
    summary = {"n_ccf": int(a["n"]), "mean_ccf": a["mean_ccf"], "median_ccf": a["median_ccf"],
               "undrawn_weighted_ccf": a["undrawn_weighted_ccf"], "n_ccf_below_0": int(a["n_ccf_below_0"]),
               "n_ccf_above_1": int(a["n_ccf_above_1"]), "n_excluded_no_undrawn": len(x),
               "n_overdrawn": int((d["drawn"] > d["limit"]).sum())}
    tabs = {"Realised CCF by segment": seg_t, "Realised CCF by utilisation": pd.DataFrame(util_rows)}
    if len(x):
        tabs["Facilities without undrawn amount"] = pd.DataFrame([{
            "n": len(x), "n_overdrawn": int((x["drawn"] > x["limit"]).sum()),
            "mean_ead_over_limit": float((x["ead"] / x["limit"].where(x["limit"] > 0)).mean()),
            "mean_ead_minus_drawn": float((x["ead"] - x["drawn"]).mean()),
            "total_ead_minus_drawn": float((x["ead"] - x["drawn"]).sum())}])
    notes = dropped_note(dropped)
    if (~has_lim).any():
        notes.append(f"{int((~has_lim).sum())} facilities with limit <= 0 left out of the utilisation table.")
    return Outcome(summary, tabs, notes=notes, rows_used=len(d))


@register("ead.ccf_backtest", "CCF back-test t-test (realised vs estimated, per pool and portfolio)",
          "Calibration", _EAD,
          params=(*_CCF_INPUTS, P("predicted", help="Estimated CCF per facility (or the CCF of its pool)"),
                  P("grade", required=False, help="CCF pool / grade"),
                  P("alternative", "string", default="greater", choices=("greater", "two-sided", "less"),
                    help="H1 on realised − estimated: 'greater' = estimates too low")),
          description="""ECB-style back-test of CCF estimates, per pool and portfolio: with d_i = realised CCF_i −
estimated CCF_i, T = √N · mean(d)/s_d ~ Student t(N−1). Default H0: estimated CCF ≥ realised (one-sided,
p = 1 − t_{N−1}(T)). Facilities without undrawn amount are excluded (no CCF). Assumes independent facilities
and an approximately normal mean.""",
          references=("ECB (2019), Instructions for reporting the validation results of internal models — "
                      "IRB Pillar I models for credit risk (CCF back-testing t-test)",
                      "Regulation (EU) No 575/2013 (CRR), Article 182"))
def ead_ccf_backtest(ctx: RunContext, predicted, actual=None, ead=None, limit=None, drawn=None, min_undrawn=0.0,
                     grade=None, alternative="greater") -> Outcome:
    d, notes = _ccf(ctx, actual, ead, limit, drawn, min_undrawn, {"pred": predicted}, {"grade": grade})
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 facilities with a realised CCF.")
    t = ttest_rows(d, "grade" if grade else None, alternative)
    a = t.iloc[-1]
    if t["p_value"].isna().any():
        notes.append("p-value not computed where n < 2 or the differences have zero variance.")
    return Outcome({"mean_realised": a["mean_realised"], "mean_estimated": a["mean_estimated"],
                    "mean_difference": a["mean_difference"], "t_statistic": a["t_statistic"],
                    "p_value": a["p_value"], "n": int(a["n"]), "alternative": alternative},
                   {"CCF back-test by pool": t}, notes=notes, rows_used=len(d))


@register("ead.coverage_ratio", "EAD coverage ratio (predicted vs realised EAD)", "Calibration", _EAD,
          params=(P("ead", help="Realised EAD at default"), P("predicted", help="Predicted EAD (at the reference date)"),
                  P("segment", required=False)),
          description="""Exposure-level calibration of EAD: coverage ratio = Σ predicted EAD / Σ realised EAD (< 1 = EAD
under-estimated in aggregate), share of facilities with predicted < realised EAD, mean and median of the
facility ratio predicted/realised (facilities with realised EAD > 0), and a paired t-test of realised −
predicted EAD (H1: realised > predicted, one-sided). Overall and per segment.""",
          references=("Regulation (EU) No 575/2013 (CRR), Articles 166 and 182",
                      "BCBS (2005), Working Paper No. 14, Studies on the Validation of Internal Rating Systems"))
def ead_coverage_ratio(ctx: RunContext, ead, predicted, segment=None) -> Outcome:
    d, dropped = _frame(ctx, {"real": ead, "pred": predicted}, {"seg": segment})
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 complete facilities.")
    groups = [(g, d[d["seg"] == g]) for g in sorted_levels(d["seg"])] if segment else []
    rows = []
    for g, s in groups + [("ALL", d)]:
        sr, sp = float(s["real"].sum()), float(s["pred"].sum())
        pos = s["real"] > 0
        ratio = s.loc[pos, "pred"] / s.loc[pos, "real"]
        diff = s["real"] - s["pred"]
        t, p = np.nan, np.nan
        if len(s) > 1 and diff.std(ddof=1) > 0:
            r = stats.ttest_1samp(diff, 0.0, alternative="greater")
            t, p = float(r.statistic), float(r.pvalue)
        rows.append({"group": str(g), "n": len(s), "sum_predicted": sp, "sum_realised": sr,
                     "coverage_ratio": sp / sr if sr else np.nan, "share_under_predicted": float((s["pred"] < s["real"]).mean()),
                     "mean_ratio": float(ratio.mean()) if len(ratio) else np.nan,
                     "median_ratio": float(ratio.median()) if len(ratio) else np.nan,
                     "t_statistic": t, "p_value_underestimation": p})
    tab = pd.DataFrame(rows)
    a = tab.iloc[-1]
    return Outcome({"coverage_ratio": a["coverage_ratio"], "share_under_predicted": a["share_under_predicted"],
                    "median_ratio": a["median_ratio"], "t_statistic": a["t_statistic"],
                    "p_value_underestimation": a["p_value_underestimation"], "n": int(a["n"])},
                   {"EAD coverage": tab}, notes=dropped_note(dropped), rows_used=len(d))


@register("ead.ccf_distribution", "Realised / estimated CCF distribution and mass at bounds", "Distribution", _EAD,
          params=(*_CCF_INPUTS, P("predicted", required=False, help="Optional estimated CCF"),
                  P("tol", "number", default=1e-6), P("bins", "integer", default=20)),
          description="""Shape of the CCF distribution: moments, quantiles, share exactly at 0 and 1 (within `tol`), share
below 0 (repayment before default) and above 1 (drawing beyond the limit), share near each bound, bimodality
coefficient, and a histogram on [0, 1] with out-of-range counts — for realised and (optionally) estimated
CCF. Facilities without undrawn amount have no CCF and are excluded. Descriptive.""",
          references=("EBA/GL/2017/16, Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures",
                      "Pfister et al. (2013), Good things peak in pairs: a note on the bimodality coefficient, "
                      "Frontiers in Psychology 4"))
def ead_ccf_distribution(ctx: RunContext, actual=None, ead=None, limit=None, drawn=None, min_undrawn=0.0,
                         predicted=None, tol=1e-6, bins=20) -> Outcome:
    d, notes = _ccf(ctx, actual, ead, limit, drawn, min_undrawn)
    if len(d) < 2:
        raise NotApplicable("Fewer than 2 realised CCFs.")
    vals = {"realised": d["real"]}
    if predicted:
        p = num(ctx.df, predicted).loc[d.index].dropna()
        vals["estimated"] = p
    t, h = distribution_tables(vals, 0.0, 1.0, tol, bins)
    r = t.iloc[0]
    return Outcome({"n": int(r["n"]), "mean": r["mean"], "median": r["median"], "share_at_0": r["share_at_0"],
                    "share_at_1": r["share_at_1"], "share_below_0": r["share_below_0"],
                    "share_above_1": r["share_above_1"], "bimodality_coefficient": r["bimodality_coefficient"]},
                   {"Distribution summary": t, "Histogram": h}, notes=notes, rows_used=len(d))


@register("ead.ccf_ranking", "CCF ranking power: generalised AUC and rank correlations", "Discrimination", _EAD,
          params=(*_CCF_INPUTS, P("predicted", help="Estimated CCF"), P("segment", required=False),
                  P("confidence", "number", default=0.95)),
          description="""Ranking power of CCF estimates: generalised AUC = P(CCF_est_i > CCF_est_j | CCF_real_i >
CCF_real_j) + ½ P(tie in estimate), over pairs with different realised CCF, with a U-statistic standard
error and z-test of H0: gAUC = 0.5; plus Pearson, Spearman and Kendall τ-b correlations of realised vs
estimated CCF (overall and per segment, H0: no association).""",
          references=("ECB (2019), Instructions for reporting the validation results of internal models — "
                      "IRB Pillar I models for credit risk (generalised AUC)",
                      "DeLong, DeLong & Clarke-Pearson (1988), Biometrics 44(3)"))
def ead_ccf_ranking(ctx: RunContext, predicted, actual=None, ead=None, limit=None, drawn=None, min_undrawn=0.0,
                    segment=None, confidence=0.95) -> Outcome:
    d, notes = _ccf(ctx, actual, ead, limit, drawn, min_undrawn, {"pred": predicted}, {"seg": segment})
    if len(d) < 4:
        raise NotApplicable("Fewer than 4 facilities with a realised CCF.")
    g = gauc(d["pred"], d["real"])
    zc = stats.norm.ppf(0.5 + confidence / 2)
    z0 = (g["gAUC"] - 0.5) / g["se"] if g["se"] > 0 else np.nan
    corr = correlation_outcome(d, "seg" if segment else None, confidence, 0)
    summary = {"gAUC": g["gAUC"], "gAUC_se": g["se"], "gAUC_ci_lower": max(0.0, g["gAUC"] - zc * g["se"]),
               "gAUC_ci_upper": min(1.0, g["gAUC"] + zc * g["se"]),
               "gAUC_p_vs_0.5": float(2 * stats.norm.sf(abs(z0))) if g["se"] > 0 else np.nan} | corr.summary
    return Outcome(summary, {"Correlations": corr.tables["Correlations"]},
                   notes=notes + ["gAUC: pairs with equal realised CCF not compared; estimate ties count 1/2."],
                   rows_used=len(d))
