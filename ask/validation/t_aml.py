"""AML transaction-monitoring, screening and customer-risk-rating validation.

Tests for the quantitative parts of an AML model validation:
  * alert volumes and the alert → case → SAR funnel by scenario, segment, period;
  * scenario overlap and the unique SAR contribution of each rule;
  * threshold tuning: below-the-line (BTL) bands, reviewed-BTL productive rates with exact Clopper–Pearson
    intervals, above-the-line (ATL) threshold sweeps, and sample-size planning for BTL/ATL testing;
  * segmentation (peer-group) quality and drift, data completeness of TM-critical fields, Benford's law;
  * name-screening effectiveness from a test-case table;
  * alert ageing / backlog;
  * customer risk rating: class distribution, migration matrix, concordance with SAR outcomes.

Binary flags (alert, case, SAR, productive, hit) accept 0/1, bool, Y/N, yes/no, true/false.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import (NotApplicable, Outcome, P, RunContext, binary, dropped_note, num, register)

_AML = ("aml",)
_REFS_TM = ("Wolfsberg Group (2024), Statement on Effective Monitoring for Suspicious Activity",
            "FFIEC BSA/AML Examination Manual, Suspicious Activity Reporting / Transaction Monitoring",
            "OCC Bulletin 2011-12 / Federal Reserve SR 11-7, Supervisory Guidance on Model Risk Management",
            "Directive (EU) 2015/849 (AMLD4) as amended, art. 8 and 33")


# ── helpers ────────────────────────────────────────────────────────────────

def clopper_pearson(k, n, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Exact two-sided Clopper–Pearson interval for k successes in n trials (vectorised)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    a = 1 - confidence
    with np.errstate(all="ignore"):
        lo = np.where(k > 0, stats.beta.ppf(a / 2, k, n - k + 1), 0.0)
        hi = np.where(k < n, stats.beta.ppf(1 - a / 2, k + 1, n - k), 1.0)
    lo, hi = np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)
    return lo, hi


def cp_upper_one_sided(k, n, confidence: float = 0.95) -> np.ndarray:
    """One-sided exact upper bound: beta.ppf(confidence, k+1, n−k); equals 1 − (1−conf)^(1/n) for k = 0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(all="ignore"):
        return np.where(n > 0, np.where(k < n, stats.beta.ppf(confidence, k + 1, n - k), 1.0), np.nan)


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    return binary(df, col)


def _rate_table(g: pd.DataFrame, by: str, k: str, n: str, confidence: float, rate: str) -> pd.DataFrame:
    lo, hi = clopper_pearson(g[k], g[n], confidence)
    g = g.copy()
    g[rate] = g[k] / g[n].where(g[n] > 0)
    g[f"ci_low_{confidence:.0%}"], g[f"ci_high_{confidence:.0%}"] = lo, hi
    return g


def _homogeneity(k: np.ndarray, n: np.ndarray) -> tuple[float, float, int]:
    """Chi-square test that the event rate is equal across groups (2 × G table)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    keep = n > 0
    ct = np.vstack([k[keep], n[keep] - k[keep]])
    ct = ct[:, ct.sum(axis=0) > 0]
    if ct.shape[1] < 2 or (ct.sum(axis=1) == 0).any():
        return float("nan"), float("nan"), 0
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    return float(chi2), float(p), int(dof)


def _sorted_levels(s: pd.Series, order=None) -> list:
    levels = list(pd.unique(s.dropna()))
    if order:
        miss = [v for v in levels if str(v) not in {str(o) for o in order}]
        if miss:
            raise ValueError(f"Values {miss[:10]} are not in `order`.")
        pos = {str(o): i for i, o in enumerate(order)}
        return sorted(levels, key=lambda v: pos[str(v)])
    return sorted(levels, key=lambda v: (str(type(v)), v))


# ── alert volumes and funnel ───────────────────────────────────────────────

@register("aml.alert_rate", "Alert rate overall and by scenario, segment and period", "Alert analysis", _AML,
          params=(P("alert", help="Alert flag on the monitored population (1 = alerted)"),
                  P("scenario", required=False, help="Scenario / rule that generated the alert"),
                  P("segment", required=False), P("period", required=False),
                  P("confidence", "number", default=0.95, help="Level of the Clopper–Pearson intervals")),
          description="""Alert rate = alerted units / monitored units (rows of the population table: customers,
accounts or transactions per period), overall and by segment, period and scenario, each with an exact
Clopper–Pearson interval. For scenarios the rate is (rows alerted by the scenario) / (all monitored rows).
Chi-square tests of homogeneity (H0: equal alert rate across segments / across periods) show whether
differences are beyond sampling noise. Unexplained jumps across periods point to data feed or parameter
changes; segments with near-zero alert rates may be insufficiently covered.""",
          references=(*_REFS_TM, "Clopper & Pearson (1934), Biometrika 26(4)"))
def alert_rate(ctx: RunContext, alert, scenario=None, segment=None, period=None, confidence=0.95) -> Outcome:
    df = ctx.df
    a = _flag(df, alert)
    keep = a.notna()
    d = df[keep].assign(_a=a[keep])
    n, k = len(d), int(d["_a"].sum())
    if n == 0:
        raise NotApplicable("No rows with a valid alert flag.")
    lo, hi = clopper_pearson(k, n, confidence)
    summary = {"population": n, "alerts": k, "alert_rate": k / n, "ci_low": float(lo), "ci_high": float(hi)}
    tables = {}
    for col, name in ((segment, "segment"), (period, "period")):
        if not col:
            continue
        g = d.groupby(d[col].astype(str), sort=True)["_a"].agg(population="count", alerts="sum").reset_index()
        g = g.rename(columns={col: name})
        tables[f"Alert rate by {name}"] = _rate_table(g, name, "alerts", "population", confidence, "alert_rate")
        chi2, p, dof = _homogeneity(g["alerts"].to_numpy(), g["population"].to_numpy())
        summary[f"{name}_homogeneity_chi2"], summary[f"{name}_homogeneity_p"] = chi2, p
    if scenario:
        s = d.loc[d["_a"] == 1, scenario].astype("object").where(d.loc[d["_a"] == 1, scenario].notna(), "<none>")
        g = s.astype(str).value_counts().sort_index().rename_axis("scenario").reset_index(name="alerts")
        g["population"] = n
        g = _rate_table(g, "scenario", "alerts", "population", confidence, "alert_rate")
        g["share_of_alerts"] = g["alerts"] / max(k, 1)
        tables["Alert rate by scenario"] = g
    notes = dropped_note(int((~keep).sum()), "rows with missing alert flag")
    return Outcome(summary, tables, notes=notes, rows_used=n)


@register("aml.alert_funnel", "Alert → case → SAR conversion funnel by scenario", "Alert analysis", _AML,
          params=(P("target", help="SAR filed (1) for the alert"), P("case", required=False,
                                                                     help="Escalated to a case (1)"),
                  P("scenario", required=False), P("period", required=False),
                  P("confidence", "number", default=0.95)),
          description="""On the alert table (one row per alert): numbers of alerts, cases and SARs and the
conversion rates alert → case, case → SAR and alert → SAR (the productive-alert rate), each with an exact
Clopper–Pearson interval, by scenario (and by period if given). A chi-square test of homogeneity
(H0: equal productive rate in all scenarios) is reported. Scenarios with persistently low productivity are
candidates for tuning; SARs recorded without a case are counted as a data-consistency check.""",
          references=(*_REFS_TM, "Clopper & Pearson (1934), Biometrika 26(4)"))
def alert_funnel(ctx: RunContext, target, case=None, scenario=None, period=None, confidence=0.95) -> Outcome:
    df = ctx.df
    sar = _flag(df, target)
    d = pd.DataFrame({"sar": sar}, index=df.index)
    if case:
        d["case"] = _flag(df, case)
    d = d.dropna()
    dropped = len(df) - len(d)
    if d.empty:
        raise NotApplicable("No alerts with valid flags.")
    notes = dropped_note(dropped, "alerts with missing flags")
    if case:
        bad = int(((d["sar"] == 1) & (d["case"] == 0)).sum())
        if bad:
            notes.append(f"{bad} SARs have no case flag (data inconsistency).")
    groups = [("all", pd.Series("all", index=d.index))]
    if scenario:
        groups.append(("scenario", df.loc[d.index, scenario].astype("object").fillna("<none>").astype(str)))
    if period:
        groups.append(("period", df.loc[d.index, period].astype("object").fillna("<none>").astype(str)))
    tables, summary = {}, {}
    for name, key in groups:
        agg = {"alerts": ("sar", "count"), "sars": ("sar", "sum")}
        if case:
            agg["cases"] = ("case", "sum")
        g = d.assign(_k=key).groupby("_k", sort=True).agg(**agg).reset_index().rename(columns={"_k": name})
        g = _rate_table(g, name, "sars", "alerts", confidence, "alert_to_SAR")
        if case:
            g["alert_to_case"] = g["cases"] / g["alerts"]
            g["case_to_SAR"] = g["sars"] / g["cases"].where(g["cases"] > 0)
        if name == "all":
            r = g.iloc[0]
            summary = {"alerts": int(r["alerts"]), "sars": int(r["sars"]), "alert_to_SAR": float(r["alert_to_SAR"])}
            if case:
                summary |= {"cases": int(r["cases"]), "alert_to_case": float(r["alert_to_case"]),
                            "case_to_SAR": float(r["case_to_SAR"])}
            tables["Funnel overall"] = g
        else:
            tables[f"Funnel by {name}"] = g
            chi2, p, dof = _homogeneity(g["sars"].to_numpy(), g["alerts"].to_numpy())
            summary[f"{name}_homogeneity_chi2"], summary[f"{name}_homogeneity_p"] = chi2, p
    return Outcome(summary, tables, notes=notes, rows_used=len(d))


@register("aml.scenario_overlap", "Scenario overlap (Jaccard) and unique SAR contribution of each rule",
          "Alert analysis", _AML,
          params=(P("id", help="Alerted entity (customer / account)"), P("scenario"),
                  P("target", required=False, help="SAR flag; an entity counts as SAR if any of its rows is 1")),
          description="""On the alert table: for each pair of scenarios the Jaccard index |A∩B|/|A∪B| of their
alerted-entity sets and the overlap count; for each scenario the entities it alerts that no other scenario
alerts (unique coverage) and, with a SAR flag, the SAR entities it catches, the SAR entities ONLY it catches
(unique SAR contribution — what would be lost if it were switched off) and its share of all SAR entities. Highly
overlapping scenarios with no unique SAR contribution are redundancy candidates; a rule with a large unique
contribution is critical.""",
          references=(*_REFS_TM, "Jaccard (1912), New Phytologist 11(2)"))
def scenario_overlap(ctx: RunContext, id, scenario, target=None) -> Outcome:
    df = ctx.df
    cols = [id, scenario]
    d = df[cols].dropna().astype(str)
    dropped = len(df) - len(d)
    if target:
        d["sar"] = _flag(df, target).loc[d.index].fillna(0)
    scen = sorted(d[scenario].unique())
    if len(scen) < 2:
        raise NotApplicable("Need at least two scenarios.")
    sets = {s: set(d.loc[d[scenario] == s, id]) for s in scen}
    jac = pd.DataFrame(index=scen, columns=scen, dtype=float)
    pairs = []
    for i, a in enumerate(scen):
        for b in scen:
            u = len(sets[a] | sets[b])
            jac.loc[a, b] = len(sets[a] & sets[b]) / u if u else np.nan
        for b in scen[i + 1:]:
            pairs.append({"scenario_1": a, "scenario_2": b, "overlap_entities": len(sets[a] & sets[b]),
                          "jaccard": float(jac.loc[a, b])})
    n_scen = d.groupby(id)[scenario].nunique()
    sar_ent = set(d.loc[d["sar"] == 1, id]) if target else set()
    rows = []
    for s in scen:
        only = {e for e in sets[s] if n_scen[e] == 1}
        r = {"scenario": s, "entities": len(sets[s]), "unique_entities": len(only),
             "unique_share": len(only) / len(sets[s]) if sets[s] else np.nan}
        if target:
            caught = sets[s] & sar_ent
            r |= {"sar_entities": len(caught), "unique_sar_entities": len(caught & only),
                  "share_of_all_sar_entities": len(caught) / len(sar_ent) if sar_ent else np.nan}
        rows.append(r)
    contrib = pd.DataFrame(rows)
    pt = pd.DataFrame(pairs).sort_values("jaccard", ascending=False, kind="mergesort").reset_index(drop=True)
    jt = jac.reset_index().rename(columns={"index": "scenario"})
    summary = {"scenarios": len(scen), "entities": int(d[id].nunique()),
               "mean_pairwise_jaccard": float(pt["jaccard"].mean()),
               "max_pairwise_jaccard": float(pt["jaccard"].iloc[0]),
               "max_pair": f"{pt['scenario_1'].iloc[0]} / {pt['scenario_2'].iloc[0]}"}
    if target:
        summary |= {"sar_entities": len(sar_ent),
                    "scenarios_without_unique_sar": int((contrib["unique_sar_entities"] == 0).sum())}
    return Outcome(summary, {"Contribution by scenario": contrib, "Jaccard matrix": jt, "Pairwise overlap": pt},
                   notes=dropped_note(dropped), rows_used=len(d))


# ── threshold testing ──────────────────────────────────────────────────────

_THRESH = (P("value", help="Monitored quantity the scenario thresholds (amount, count, velocity, ...)"),
           P("threshold", "number", help="Current threshold"),
           P("direction", "string", default="above", choices=("above", "below"),
             help="'above': alert when value >= threshold; 'below': alert when value <= threshold"))


def _bands(value: pd.Series, threshold: float, bands, direction: str) -> pd.Series:
    """Band label per row: 'ATL', 'BTL -10%..0%' style labels, or 'beyond' the last band."""
    pct = sorted(float(b) for b in bands)
    if not pct or pct[0] <= 0:
        raise ValueError("bands must be positive percentages, e.g. [10, 20, 30, 40, 50].")
    sign = 1 if direction == "above" else -1
    # distance below the threshold in % of |threshold| (positive = BTL side)
    dist = sign * (threshold - value) / abs(threshold) * 100
    lab = pd.Series("beyond", index=value.index, dtype=object)
    lab[dist <= 0] = "ATL"
    prev = 0.0
    for p in pct:
        lab[(dist > prev) & (dist <= p)] = f"BTL {prev:g}%–{p:g}%"
        prev = p
    lab[value.isna()] = np.nan
    return lab


def _band_order(pct) -> list[str]:
    pct = sorted(float(b) for b in pct)
    prev, out = 0.0, ["ATL"]
    for p in pct:
        out.append(f"BTL {prev:g}%–{p:g}%")
        prev = p
    return out + ["beyond"]


@register("aml.btl_bands", "Below-the-line population bands around a scenario threshold", "Threshold tuning",
          _AML,
          params=(*_THRESH, P("bands", "list", default=[10, 20, 30, 40, 50],
                              help="Upper edges of BTL bands in % below the threshold"),
                  P("id", required=False, help="Entity id (to count distinct entities per band)")),
          description="""Splits the monitored population by distance to the current threshold: above the line
(alerting), BTL bands (e.g. 0–10%, 10–20%, … below the threshold; for 'below' scenarios the mirror image) and
beyond the last band. Reports rows, distinct entities and share per band. This is the sampling frame for
below-the-line testing: each band is sampled and reviewed (see aml.sample_size and aml.btl_review) to show
whether lowering the threshold would surface productive alerts.""",
          references=(*_REFS_TM,))
def btl_bands(ctx: RunContext, value, threshold, direction="above", bands=(10, 20, 30, 40, 50), id=None) -> Outcome:
    if threshold == 0:
        raise ValueError("threshold must be non-zero (bands are % of the threshold).")
    v = num(ctx.df, value)
    lab = _bands(v, threshold, bands, direction)
    d = pd.DataFrame({"band": lab, "value": v})
    if id:
        d["id"] = ctx.df[id]
    d = d.dropna(subset=["band"])
    order = _band_order(bands)
    agg = {"rows": ("value", "count"), "min_value": ("value", "min"), "max_value": ("value", "max")}
    if id:
        agg["entities"] = ("id", "nunique")
    g = d.groupby("band").agg(**agg).reindex(order).fillna({"rows": 0}).reset_index()
    g["rows"] = g["rows"].astype(int)
    g["share"] = g["rows"] / max(len(d), 1)
    btl = g[g["band"].str.startswith("BTL")]["rows"].sum()
    return Outcome({"population": len(d), "atl_rows": int(g.loc[g["band"] == "ATL", "rows"].iloc[0]),
                    "btl_rows": int(btl), "threshold": threshold, "direction": direction},
                   {"Population by band": g}, notes=dropped_note(int(v.isna().sum()), "rows with missing value"),
                   rows_used=len(d))


@register("aml.btl_review", "Productive rate in reviewed below-the-line samples (exact CIs)", "Threshold tuning",
          _AML,
          params=(P("target", help="Review outcome: 1 = productive (would have been escalated / SAR)"),
                  P("band", required=False, help="Band label column of the reviewed sample"),
                  P("value", required=False, help="Monitored value (to derive bands instead of `band`)"),
                  P("threshold", "number", required=False), P("direction", "string", default="above",
                                                                choices=("above", "below")),
                  P("bands", "list", default=[10, 20, 30, 40, 50]),
                  P("population_counts", "dict", required=False,
                    help="Population size per band {band: N} to extrapolate productive counts"),
                  P("confidence", "number", default=0.95)),
          description="""For each BTL band of the reviewed sample: reviewed items, productive items, productive
rate with exact two-sided Clopper–Pearson interval and the one-sided exact upper bound at the chosen confidence
(for zero productive items this is 1 − (1 − confidence)^(1/n), the 'rule of three' ≈ 3/n at 95%). With
population sizes per band, the expected number of productive cases in the whole band (rate × N) and its upper
bound are extrapolated. Assumes simple random sampling within each band.""",
          references=("Clopper & Pearson (1934), Biometrika 26(4)", "Hanley & Lippman-Hand (1983), JAMA 249(13)",
                      *_REFS_TM[:2]))
def btl_review(ctx: RunContext, target, band=None, value=None, threshold=None, direction="above",
               bands=(10, 20, 30, 40, 50), population_counts=None, confidence=0.95) -> Outcome:
    df = ctx.df
    y = _flag(df, target)
    if band:
        lab = df[band].astype("object").where(df[band].notna(), np.nan)
        lab = lab.where(lab.isna(), lab.astype(str))
        order = None
    elif value and threshold is not None:
        lab = _bands(num(df, value), threshold, bands, direction)
        order = _band_order(bands)
    else:
        raise ValueError("Give `band`, or `value` with `threshold`.")
    d = pd.DataFrame({"band": lab, "y": y}).dropna()
    if d.empty:
        raise NotApplicable("No reviewed items with a band and an outcome.")
    g = d.groupby("band")["y"].agg(reviewed="count", productive="sum").reset_index()
    if order:
        g = g.set_index("band").reindex([o for o in order if o in set(g["band"])]).reset_index()
    else:
        g = g.sort_values("band", kind="mergesort").reset_index(drop=True)
    g["productive"] = g["productive"].astype(int)
    g = _rate_table(g, "band", "productive", "reviewed", confidence, "productive_rate")
    g[f"upper_bound_one_sided_{confidence:.0%}"] = cp_upper_one_sided(g["productive"], g["reviewed"], confidence)
    notes = dropped_note(len(df) - len(d), "rows without band or outcome")
    if population_counts:
        pc = {str(k): float(v) for k, v in population_counts.items()}
        unknown = set(pc) - set(g["band"].astype(str))
        if unknown:
            notes.append(f"population_counts for bands not in the sample ignored: {sorted(unknown)}.")
        g["population"] = g["band"].astype(str).map(pc)
        g["expected_productive_in_population"] = g["productive_rate"] * g["population"]
        g["upper_productive_in_population"] = g[f"upper_bound_one_sided_{confidence:.0%}"] * g["population"]
    k, n = int(g["productive"].sum()), int(g["reviewed"].sum())
    lo, hi = clopper_pearson(k, n, confidence)
    return Outcome({"reviewed": n, "productive": k, "productive_rate": k / n, "ci_low": float(lo),
                    "ci_high": float(hi), "bands": len(g), "confidence": confidence},
                   {"Productive rate by band": g}, notes=notes, rows_used=len(d))


@register("aml.atl_threshold_sweep", "Above-the-line threshold sweep (alerts, SARs, marginal yield)",
          "Threshold tuning", _AML,
          params=(P("value"), P("target", help="SAR / productive outcome"),
                  P("thresholds", "list", required=False, help="Thresholds to evaluate (default: quantile grid)"),
                  P("n_thresholds", "integer", default=20),
                  P("current_threshold", "number", required=False),
                  P("direction", "string", default="above", choices=("above", "below"))),
          description="""For each candidate threshold t: alerts generated (value ≥ t, or ≤ t for 'below'), SARs
captured, SAR yield (SARs / alerts), share of all SARs captured, and the marginal SAR yield of the additional
alerts when moving from the next stricter threshold (ΔSARs / Δalerts) — the quantity to compare with the
investigation cost. With the current threshold, changes in alerts and SARs relative to it are shown. Outcomes
are only known for investigated items, so rows below the current threshold should come from a reviewed BTL
sample; otherwise SARs below the line are understated.""",
          references=(*_REFS_TM,))
def atl_threshold_sweep(ctx: RunContext, value, target, thresholds=None, n_thresholds=20, current_threshold=None,
                        direction="above") -> Outcome:
    df = ctx.df
    d = pd.DataFrame({"v": num(df, value), "y": _flag(df, target)}).dropna()
    if d.empty:
        raise NotApplicable("No rows with value and outcome.")
    if thresholds:
        ts = np.unique(np.asarray([float(t) for t in thresholds]))
    else:
        ts = np.unique(np.quantile(d["v"], np.linspace(0, 1, int(n_thresholds) + 1)[:-1]))
    if current_threshold is not None:
        ts = np.unique(np.r_[ts, float(current_threshold)])
    ts = ts[::-1] if direction == "above" else ts            # strictest first
    v, y = d["v"].to_numpy(), d["y"].to_numpy()
    total = y.sum()
    rows = []
    for t in ts:
        m = v >= t if direction == "above" else v <= t
        rows.append({"threshold": float(t), "alerts": int(m.sum()), "sars": int(y[m].sum())})
    g = pd.DataFrame(rows)
    g["sar_yield"] = g["sars"] / g["alerts"].where(g["alerts"] > 0)
    g["share_of_sars_captured"] = g["sars"] / total if total else np.nan
    da, ds = g["alerts"].diff(), g["sars"].diff()
    g["additional_alerts"] = da
    g["additional_sars"] = ds
    g["marginal_sar_yield"] = ds / da.where(da > 0)
    summary = {"rows": len(d), "total_sars": int(total), "thresholds": len(g)}
    if current_threshold is not None:
        cur = g.loc[np.isclose(g["threshold"], current_threshold)].iloc[0]
        g["alerts_vs_current"] = g["alerts"] - cur["alerts"]
        g["sars_vs_current"] = g["sars"] - cur["sars"]
        summary |= {"current_alerts": int(cur["alerts"]), "current_sars": int(cur["sars"]),
                    "current_sar_yield": float(cur["sar_yield"])}
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_scatter(x=g["alerts"], y=g["sars"], mode="lines+markers", text=g["threshold"].round(4).astype(str),
                    name="threshold sweep")
    fig.update_layout(title="SARs captured vs alerts generated", xaxis_title="alerts", yaxis_title="SARs")
    return Outcome(summary, {"Threshold sweep": g}, figures=[fig],
                   notes=dropped_note(len(df) - len(d)) + [
                       "Outcomes below the current threshold are only meaningful if they come from reviewed BTL "
                       "samples."], rows_used=len(d))


def discovery_sample_size_binomial(confidence: float, tolerable_rate: float, expected_errors: int = 0) -> int:
    """Smallest n with P(X ≤ c | n, p_t) ≤ 1 − confidence (binomial attribute sampling)."""
    c, p, beta = int(expected_errors), float(tolerable_rate), 1 - float(confidence)
    if c == 0:
        return int(math.ceil(math.log(beta) / math.log(1 - p) - 1e-12))
    hi = int(stats.chi2.ppf(confidence, 2 * (c + 1)) / (2 * p) * 2 + 50)
    n = np.arange(c + 1, hi + 1)
    ok = np.nonzero(stats.binom.cdf(c, n, p) <= beta)[0]
    return int(n[ok[0]])


def discovery_sample_size_hypergeom(confidence: float, tolerable_rate: float, population: int,
                                    expected_errors: int = 0) -> tuple[int, int]:
    """(n, D): smallest n with P(X ≤ c | N, D, n) ≤ 1 − confidence, D = ceil(p_t·N) deviations in the population."""
    N, c = int(population), int(expected_errors)
    D = int(math.ceil(tolerable_rate * N - 1e-9))
    if D <= c:
        return N, D
    n = np.arange(c + 1, N + 1)
    ok = np.nonzero(stats.hypergeom.cdf(c, N, D, n) <= 1 - confidence)[0]
    return (int(n[ok[0]]) if len(ok) else N), D


@register("aml.sample_size", "Sample size for BTL/ATL testing (discovery and estimation sampling)",
          "Threshold tuning", ("aml", "general"),
          params=(P("confidence", "number", default=0.95),
                  P("tolerable_rate", "number", default=0.05, help="Tolerable / detectable productive rate"),
                  P("expected_errors", "integer", default=0,
                    help="Productive items allowed in the sample (0 = discovery sampling)"),
                  P("population", "integer", required=False, help="Population size (hypergeometric / FPC)"),
                  P("margin", "number", required=False, help="Margin of error for estimating a rate"),
                  P("expected_rate", "number", default=0.5, help="Anticipated rate for the estimation sample")),
          description="""Attribute (discovery) sampling: the smallest n such that, if the true productive rate
were the tolerable rate p_t, observing at most c productive items would have probability ≤ 1 − confidence.
Binomial: P(X ≤ c | n, p_t) ≤ 1 − confidence (for c = 0, n = ⌈ln(1 − conf)/ln(1 − p_t)⌉; 59 for 95% / 5%).
Hypergeometric (finite population N, D = ⌈p_t·N⌉ productive items): P(X ≤ c | N, D, n) ≤ 1 − confidence.
Estimation sampling: n₀ = z²·p(1 − p)/E² for a two-sided margin E, with finite-population correction
n = n₀/(1 + (n₀ − 1)/N). If c or fewer productive items are found, one can state with the given confidence that
the rate is below p_t.""",
          references=("AICPA (2019), Audit Guide: Audit Sampling, attribute sampling tables",
                      "Cochran (1977), Sampling Techniques, 3rd ed., ch. 4",
                      "Guy, Carmichael & Whittington (2002), Audit Sampling: An Introduction, 5th ed."))
def sample_size(ctx: RunContext, confidence=0.95, tolerable_rate=0.05, expected_errors=0, population=None,
                margin=None, expected_rate=0.5) -> Outcome:
    if not 0 < confidence < 1 or not 0 < tolerable_rate < 1:
        raise ValueError("confidence and tolerable_rate must be in (0, 1).")
    if expected_errors < 0:
        raise ValueError("expected_errors must be >= 0.")
    rows = []
    nb = discovery_sample_size_binomial(confidence, tolerable_rate, expected_errors)
    rows.append({"method": "Discovery / attribute sampling (binomial)", "sample_size": nb,
                 "detail": f"P(X ≤ {expected_errors} | n, {tolerable_rate:g}) ≤ {1 - confidence:.4g}"})
    summary = {"binomial_sample_size": nb}
    if population:
        nh, D = discovery_sample_size_hypergeom(confidence, tolerable_rate, population, expected_errors)
        rows.append({"method": "Discovery / attribute sampling (hypergeometric)", "sample_size": nh,
                     "detail": f"N = {population}, D = {D} productive items at the tolerable rate"})
        summary["hypergeometric_sample_size"] = nh
    if margin:
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        n0 = z ** 2 * expected_rate * (1 - expected_rate) / margin ** 2
        rows.append({"method": "Estimation of a rate (normal approximation)", "sample_size": int(math.ceil(n0)),
                     "detail": f"z = {z:.4f}, p = {expected_rate:g}, E = {margin:g}"})
        summary["estimation_sample_size"] = int(math.ceil(n0))
        if population:
            nf = n0 / (1 + (n0 - 1) / population)
            rows.append({"method": "Estimation of a rate with finite-population correction",
                         "sample_size": int(math.ceil(nf)), "detail": f"N = {population}"})
            summary["estimation_sample_size_fpc"] = int(math.ceil(nf))
    summary |= {"confidence": confidence, "tolerable_rate": tolerable_rate, "expected_errors": expected_errors}
    return Outcome(summary, {"Sample sizes": pd.DataFrame(rows)})


# ── segmentation, data quality, Benford ────────────────────────────────────

@register("aml.segmentation_quality", "Segmentation / peer-group quality and drift", "Segmentation", _AML,
          params=(P("segment", help="Segment / peer-group / cluster label"),
                  P("features", "columns", help="Key activity variables (volumes, counts, ...)"),
                  P("period", required=False, help="Period column for segment population drift"),
                  P("max_silhouette_rows", "integer", default=5000,
                    help="Rows sampled for the silhouette coefficient")),
          description="""Peer groups should be internally homogeneous and different from each other. Per segment
and variable: n, mean, median, SD, coefficient of variation, IQR (within-group dispersion). Per variable:
Kruskal–Wallis H test (H0: same distribution in all segments) with effect size ε² = (H − G + 1)/(n − G), and
η² = between-group SS / total SS. Silhouette coefficient of the segmentation on standardised variables
(−1..1; sample of rows, Euclidean). With a period column: segment shares by period and PSI of each period's
segment mix against the first period (population drift between peer groups).""",
          references=("Kruskal & Wallis (1952), JASA 47(260)", "Rousseeuw (1987), J. Comput. Appl. Math. 20",
                      "Tomczak & Tomczak (2014), Trends in Sport Sciences 21(1) (ε² effect size)", *_REFS_TM[:1]))
def segmentation_quality(ctx: RunContext, segment, features, period=None, max_silhouette_rows=5000) -> Outcome:
    df = ctx.df
    work = pd.DataFrame({f: num(df, f) for f in features}, index=df.index)
    work["_seg"] = df[segment]
    d = work.dropna()
    d = d.assign(_seg=d["_seg"].astype(str))
    segs = sorted(d["_seg"].unique())
    if len(segs) < 2:
        raise NotApplicable("Need at least two segments.")
    rows, tests = [], []
    for f in features:
        for s in segs:
            x = d.loc[d["_seg"] == s, f]
            q1, q3 = x.quantile([0.25, 0.75])
            rows.append({"segment": s, "variable": f, "n": len(x), "mean": x.mean(), "median": x.median(),
                         "sd": x.std(ddof=1), "cv": x.std(ddof=1) / abs(x.mean()) if x.mean() else np.nan,
                         "iqr": q3 - q1})
        groups = [d.loc[d["_seg"] == s, f].to_numpy() for s in segs]
        n, G = len(d), len(segs)
        try:
            H, p = stats.kruskal(*groups)
        except ValueError:
            H, p = np.nan, np.nan
        tot = ((d[f] - d[f].mean()) ** 2).sum()
        between = sum(len(g) * (g.mean() - d[f].mean()) ** 2 for g in groups)
        tests.append({"variable": f, "kruskal_wallis_H": float(H), "p_value": float(p),
                      "epsilon_squared": float((H - G + 1) / (n - G)) if np.isfinite(H) else np.nan,
                      "eta_squared": float(between / tot) if tot > 0 else np.nan})
    tables = {"Between-segment tests": pd.DataFrame(tests), "Within-segment dispersion": pd.DataFrame(rows)}
    summary = {"segments": len(segs), "n": len(d),
               "min_kruskal_wallis_p": float(np.nanmin([t["p_value"] for t in tests])),
               "mean_eta_squared": float(np.nanmean([t["eta_squared"] for t in tests]))}
    notes = dropped_note(len(df) - len(d))
    Z = d[list(features)].to_numpy(float)
    sd = Z.std(axis=0)
    if (sd > 0).any() and len(d) > len(segs):
        from sklearn.metrics import silhouette_score
        Z = (Z[:, sd > 0] - Z[:, sd > 0].mean(axis=0)) / sd[sd > 0]
        lab = d["_seg"].to_numpy()
        if len(d) > max_silhouette_rows:
            idx = np.sort(ctx.rng(3).choice(len(d), int(max_silhouette_rows), replace=False))
            Z, lab = Z[idx], lab[idx]
            notes.append(f"Silhouette computed on a random sample of {int(max_silhouette_rows)} rows.")
        if len(np.unique(lab)) >= 2:
            summary["silhouette"] = float(silhouette_score(Z, lab, metric="euclidean"))
    if period:
        from ask.validation.t_stability import psi_table
        per = df.loc[d.index, period]
        mix = pd.crosstab(per.astype(str), d["_seg"], normalize="index").sort_index()
        tables["Segment shares by period"] = mix.reset_index()
        ps = sorted(per.dropna().unique(), key=lambda v: (str(type(v)), v))
        if len(ps) >= 2:
            ref = d.loc[per == ps[0], "_seg"]
            psis = [{"period": str(p), "n": int((per == p).sum()),
                     "PSI_vs_first": float(psi_table(ref, d.loc[per == p, "_seg"], categorical=True)
                                           ["psi_contribution"].sum())} for p in ps]
            tables["Segment mix PSI by period"] = pd.DataFrame(psis)
            summary["max_segment_mix_PSI"] = float(max(r["PSI_vs_first"] for r in psis))
    return Outcome(summary, tables, notes=notes, rows_used=len(d))


_PLACEHOLDERS = ["", "unknown", "unk", "n/a", "na", "null", "none", "xx", "xxx", "?", "-", "0000", "not available"]


@register("aml.data_completeness", "Completeness of TM-critical fields", "Data quality", _AML,
          params=(P("columns", "columns", help="TM-critical fields (counterparty, country, amount, ...)"),
                  P("amount", required=False, help="Amount column: also count zero / negative amounts"),
                  P("period", required=False), P("segment", required=False),
                  P("placeholders", "list", required=False,
                    help="Values treated as missing (default: '', unknown, n/a, null, none, xx, ?, -, ...)")),
          description="""Share of records in which each transaction-monitoring-critical field is missing: null
values plus placeholder values that defeat the rules (blank, 'UNKNOWN', 'N/A', 'XX', ...; case-insensitive), and
for the amount column non-numeric, zero and negative amounts. Broken down by period and segment when given, so
that feed outages and channels with systematically poor data are visible. Missing counterparty / country /
amount information silently disables scenarios that depend on them.""",
          references=("BCBS 239 (2013), Principles for effective risk data aggregation and risk reporting",
                      "Regulation (EU) 2023/1113 (transfer of funds: payer/payee information)", *_REFS_TM[:2]))
def data_completeness(ctx: RunContext, columns, amount=None, period=None, segment=None, placeholders=None) -> Outcome:
    df = ctx.df
    ph = {str(p).strip().lower() for p in (placeholders if placeholders is not None else _PLACEHOLDERS)}
    miss = pd.DataFrame(index=df.index)
    rows = []
    for c in columns:
        s = df[c]
        null = s.isna()
        txt = s.astype("object").where(~null, None)
        place = (~null) & txt.astype(str).str.strip().str.lower().isin(ph)
        m = null | place
        r = {"field": c, "records": len(s), "null": int(null.sum()), "placeholder": int(place.sum())}
        if amount and c == amount:
            a = pd.to_numeric(s, errors="coerce")
            nonnum = (~null) & (~place) & a.isna()
            nonpos = a.notna() & (a <= 0)
            r |= {"non_numeric": int(nonnum.sum()), "zero_or_negative": int(nonpos.sum())}
            m = m | nonnum | nonpos
        r["missing_total"] = int(m.sum())
        r["missing_share"] = float(m.mean()) if len(m) else np.nan
        rows.append(r)
        miss[c] = m
    t = pd.DataFrame(rows).sort_values("missing_share", ascending=False, kind="mergesort").reset_index(drop=True)
    tables = {"Completeness by field": t}
    for col, name in ((period, "period"), (segment, "segment")):
        if col:
            g = miss.groupby(df[col].astype(str), sort=True).mean()
            g.insert(0, "records", df.groupby(df[col].astype(str), sort=True).size())
            tables[f"Missing share by {name}"] = g.reset_index().rename(columns={col: name})
    any_m = miss.any(axis=1)
    return Outcome({"records": len(df), "fields": len(columns), "worst_field": t.iloc[0]["field"],
                    "worst_missing_share": float(t.iloc[0]["missing_share"]),
                    "share_records_with_any_missing": float(any_m.mean())},
                   tables, rows_used=len(df))


def _leading_digits(x: np.ndarray, two: bool) -> np.ndarray:
    e = np.floor(np.log10(x))
    m = np.round(x / 10.0 ** e, 9)                  # mantissa in [1, 10)
    e = np.where(m >= 10, e + 1, e)
    m = np.round(x / 10.0 ** e, 9)
    return np.floor(m * 10 + 1e-9).astype(int) if two else np.floor(m + 1e-9).astype(int)


_NIGRINI = {False: [(0.006, "close conformity"), (0.012, "acceptable conformity"),
                    (0.015, "marginally acceptable conformity"), (np.inf, "nonconformity")],
            True: [(0.0012, "close conformity"), (0.0018, "acceptable conformity"),
                   (0.0022, "marginally acceptable conformity"), (np.inf, "nonconformity")]}


@register("aml.benford", "Benford's law first-digit / first-two-digits test", "Data quality", ("aml", "general"),
          params=(P("amount", help="Transaction amounts"),
                  P("digits", "string", default="first", choices=("first", "first_two")),
                  P("min_amount", "number", default=10.0, help="Amounts below this are excluded")),
          description="""Compares the leading-digit distribution of positive amounts ≥ min_amount with Benford's
law P(d) = log10(1 + 1/d) (d = 1..9, or 10..99 for the first-two-digits test). Pearson χ² goodness of fit
(H0: amounts follow Benford; df = 8 or 89; over-powered for large n), mean absolute deviation
MAD = mean|observed% − expected%| with Nigrini's published MAD conformity ranges (part of the MAD test
definition), and per-digit z statistics with continuity correction. Deviations (spikes just below reporting
thresholds, round amounts) can indicate structuring or data issues; many legitimate amount distributions
(prices, fees, capped amounts) do not follow Benford, so read in context.""",
          references=("Benford (1938), Proc. American Philosophical Society 78(4)",
                      "Nigrini (2012), Benford's Law: Applications for Forensic Accounting, Auditing, and Fraud "
                      "Detection, Wiley", "Nigrini (1996), J. American Taxation Association 18(1)"))
def benford(ctx: RunContext, amount, digits="first", min_amount=10.0) -> Outcome:
    a = num(ctx.df, amount)
    n_all = len(a)
    x = a[(a > 0) & (a >= min_amount)].to_numpy()
    two = digits == "first_two"
    if len(x) < 10:
        raise NotApplicable(f"{len(x)} positive amounts ≥ {min_amount}.")
    d = _leading_digits(x, two)
    support = np.arange(10, 100) if two else np.arange(1, 10)
    exp = np.log10(1 + 1 / support)
    obs = pd.Series(d).value_counts().reindex(support, fill_value=0).to_numpy()
    n = len(x)
    po = obs / n
    chi2 = float(((obs - n * exp) ** 2 / (n * exp)).sum())
    dof = len(support) - 1
    mad = float(np.mean(np.abs(po - exp)))
    z = (np.abs(po - exp) - 1 / (2 * n)) / np.sqrt(exp * (1 - exp) / n)
    z = np.maximum(z, 0)
    t = pd.DataFrame({"digit": support, "observed_count": obs, "observed_share": po, "benford_share": exp,
                      "difference": po - exp, "z_stat": z, "z_p_value": 2 * stats.norm.sf(z)})
    rng_label = next(lab for cut, lab in _NIGRINI[two] if mad <= cut)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_bar(x=support, y=po, name="observed")
    fig.add_scatter(x=support, y=exp, name="Benford", mode="lines+markers")
    fig.update_layout(title=f"Benford {digits.replace('_', ' ')} digit test", xaxis_title="leading digit(s)",
                      yaxis_title="share")
    return Outcome({"n": n, "chi2": chi2, "df": dof, "p_value": float(stats.chi2.sf(chi2, dof)), "MAD": mad,
                    "nigrini_mad_range": rng_label, "max_abs_z": float(z.max()),
                    "max_z_digit": int(support[int(np.argmax(z))])},
                   {"Digit distribution": t}, figures=[fig],
                   notes=[f"{n_all - n} rows excluded (missing, non-positive or below {min_amount:g})."]
                   if n_all - n else [], rows_used=n)


# ── screening ──────────────────────────────────────────────────────────────

@register("aml.screening_effectiveness", "Name-screening effectiveness from test cases", "Screening", _AML,
          params=(P("expected_hit", help="Test case should hit (1 = true match planted)"),
                  P("actual_hit", help="Screening engine produced a hit (1)"),
                  P("variation", required=False, help="Fuzzy-variation type of the test case"),
                  P("score", required=False, help="Engine match score (for a threshold table)"),
                  P("confidence", "number", default=0.95)),
          description="""From a labelled test-case set (true matches with name variations — typos, transliteration,
word order, missing tokens, aliases — and true non-matches): detection rate = TP/(TP + FN) and false-positive
rate = FP/(FP + TN) with exact Clopper–Pearson intervals, precision, and the same by variation type, which
reveals the fuzzy-matching weaknesses. With the engine's match score: detection and false-positive rates if the
alert threshold were set at each score quantile. Results reflect the test set's composition, not production
prevalence.""",
          references=("Wolfsberg Group (2019), Guidance on Sanctions Screening",
                      "Clopper & Pearson (1934), Biometrika 26(4)", *_REFS_TM[2:3]))
def screening_effectiveness(ctx: RunContext, expected_hit, actual_hit, variation=None, score=None,
                            confidence=0.95) -> Outcome:
    df = ctx.df
    d = pd.DataFrame({"e": _flag(df, expected_hit), "a": _flag(df, actual_hit)}, index=df.index)
    if variation:
        d["v"] = df[variation].astype("object").fillna("<none>").astype(str)
    if score:
        d["s"] = num(df, score)
    d = d.dropna()
    if d.empty:
        raise NotApplicable("No complete test cases.")

    def _metrics(x):
        tp = int(((x["e"] == 1) & (x["a"] == 1)).sum()); fn = int(((x["e"] == 1) & (x["a"] == 0)).sum())
        fp = int(((x["e"] == 0) & (x["a"] == 1)).sum()); tn = int(((x["e"] == 0) & (x["a"] == 0)).sum())
        dl, dh = clopper_pearson(tp, tp + fn, confidence)
        fl, fh = clopper_pearson(fp, fp + tn, confidence)
        return {"cases": len(x), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "detection_rate": tp / (tp + fn) if tp + fn else np.nan, "detection_ci_low": float(dl),
                "detection_ci_high": float(dh), "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
                "fpr_ci_low": float(fl), "fpr_ci_high": float(fh),
                "precision": tp / (tp + fp) if tp + fp else np.nan}
    m = _metrics(d)
    tables = {"Overall": pd.DataFrame([m])}
    if variation:
        tables["By variation type"] = pd.DataFrame([{"variation": v} | _metrics(d[d["v"] == v])
                                                    for v in sorted(d["v"].unique())])
    if score:
        qs = np.unique(np.quantile(d["s"], np.linspace(0, 1, 21)))
        pos, neg = d["e"] == 1, d["e"] == 0
        tables["By score threshold"] = pd.DataFrame(
            [{"threshold": float(q), "detection_rate": float((d.loc[pos, "s"] >= q).mean()) if pos.any() else np.nan,
              "false_positive_rate": float((d.loc[neg, "s"] >= q).mean()) if neg.any() else np.nan,
              "hits": int((d["s"] >= q).sum())} for q in qs])
    keys = ("cases", "detection_rate", "detection_ci_low", "detection_ci_high", "false_positive_rate",
            "fpr_ci_low", "fpr_ci_high", "precision")
    return Outcome({k: m[k] for k in keys}, tables, notes=dropped_note(len(df) - len(d), "incomplete test cases"),
                   rows_used=len(d))


# ── alert ageing ───────────────────────────────────────────────────────────

def _dates(s: pd.Series) -> pd.Series:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", format="mixed")


@register("aml.alert_aging", "Alert ageing and backlog", "Alert analysis", _AML,
          params=(P("date", help="Alert creation date"),
                  P("closed_date", required=False, help="Alert closure date (empty = still open)"),
                  P("as_of", "string", required=False, help="Reference date (default: latest date in the data)"),
                  P("segment", required=False, help="Grouping (scenario, team, segment)"),
                  P("buckets", "list", default=[30, 60, 90, 180], help="Age bucket edges in days")),
          description="""Open alerts at the reference date (created on or before it, not closed by it): count,
age in days (median, 75th / 90th percentile, max) and distribution over age buckets; closed alerts: time to
close (mean, median, 90th percentile); by group when given; and the month-end backlog (alerts open at each
month end) with monthly inflow and closures. Growing backlogs or long-tail ageing indicate investigation
capacity problems that delay SAR filing beyond regulatory deadlines.""",
          references=(*_REFS_TM[:2], "Directive (EU) 2015/849, art. 33 (prompt reporting of suspicions)"))
def alert_aging(ctx: RunContext, date, closed_date=None, as_of=None, segment=None, buckets=(30, 60, 90, 180)) -> Outcome:
    df = ctx.df
    c = _dates(df[date])
    cl = _dates(df[closed_date]) if closed_date else pd.Series(pd.NaT, index=df.index)
    keep = c.notna()
    if not keep.any():
        raise NotApplicable("No valid creation dates.")
    ref = pd.Timestamp(as_of) if as_of else max(c.max(), cl.max() if cl.notna().any() else c.max())
    c, cl = c[keep], cl[keep]
    grp = df.loc[keep, segment].astype("object").fillna("<none>").astype(str) if segment else None
    created = c <= ref
    open_ = created & (cl.isna() | (cl > ref))
    closed = created & cl.notna() & (cl <= ref)
    age = (ref - c[open_]).dt.days
    ttc = (cl[closed] - c[closed]).dt.days
    edges = sorted(float(b) for b in buckets)
    labels = [f"0–{edges[0]:g}"] + [f"{a:g}–{b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f">{edges[-1]:g}"]
    bk = pd.cut(age, [-np.inf, *edges, np.inf], labels=labels, right=True)
    btab = bk.value_counts().reindex(labels, fill_value=0).rename_axis("age_days").reset_index(name="open_alerts")
    btab["share"] = btab["open_alerts"] / max(len(age), 1)
    neg = int((ttc < 0).sum())

    def _stats(a, t):
        return {"open_alerts": len(a), "open_median_age": float(a.median()) if len(a) else np.nan,
                "open_p75_age": float(a.quantile(0.75)) if len(a) else np.nan,
                "open_p90_age": float(a.quantile(0.9)) if len(a) else np.nan,
                "open_max_age": float(a.max()) if len(a) else np.nan, "closed_alerts": len(t),
                "mean_days_to_close": float(t.mean()) if len(t) else np.nan,
                "median_days_to_close": float(t.median()) if len(t) else np.nan,
                "p90_days_to_close": float(t.quantile(0.9)) if len(t) else np.nan}
    summary = {"as_of": str(ref.date())} | _stats(age, ttc)
    tables = {"Open alerts by age bucket": btab}
    if segment:
        tables["By group"] = pd.DataFrame([{"group": g} | _stats(age[grp[open_] == g], ttc[grp[closed] == g])
                                           for g in sorted(grp.unique())])
    months = pd.period_range(c.min().to_period("M"), ref.to_period("M"), freq="M")
    if len(months) <= 240:
        rows = []
        for m in months:
            end = min(m.to_timestamp(how="end"), ref)
            start = m.to_timestamp(how="start")
            rows.append({"month": str(m), "inflow": int(((c >= start) & (c <= end)).sum()),
                         "closed": int((cl.notna() & (cl >= start) & (cl <= end)).sum()),
                         "open_at_month_end": int(((c <= end) & (cl.isna() | (cl > end))).sum())})
        tables["Monthly backlog"] = pd.DataFrame(rows)
    notes = dropped_note(int((~keep).sum()), "alerts without a valid creation date")
    if neg:
        notes.append(f"{neg} alerts closed before they were created (data error).")
    if not closed_date:
        notes.append("No closure date given: all alerts treated as open.")
    return Outcome(summary, tables, notes=notes, rows_used=int(keep.sum()))


# ── customer risk rating ───────────────────────────────────────────────────

_CRR_ORDER = P("order", "list", required=False, help="Risk classes from lowest to highest risk")


@register("aml.crr_distribution", "Customer risk rating: distribution across risk classes over time",
          "Customer risk rating", _AML,
          params=(P("grade", help="Customer risk class"), P("period", required=False), _CRR_ORDER),
          description="""Counts and shares of customers per risk class (in risk order), overall and per period.
With periods: chi-square test of homogeneity of the class mix across periods (H0: same distribution; over-powered
for large populations) and PSI of each period against the first. Concentration in one class, or empty high-risk
classes, question the discriminating power of the risk-rating methodology.""",
          references=("EBA/GL/2021/02, ML/TF Risk Factors Guidelines (Title I, risk assessment)",
                      "FATF (2014), Risk-Based Approach Guidance for the Banking Sector", "Siddiqi (2006), ch. 8 (PSI)"))
def crr_distribution(ctx: RunContext, grade, period=None, order=None) -> Outcome:
    df = ctx.df
    g = df[grade]
    keep = g.notna() & (df[period].notna() if period else True)
    d = df[keep]
    levels = _sorted_levels(d[grade], order)
    lab = d[grade]
    counts = lab.value_counts().reindex(levels, fill_value=0)
    t = pd.DataFrame({"risk_class": [str(v) for v in levels], "customers": counts.to_numpy(),
                      "share": counts.to_numpy() / max(len(d), 1)})
    summary = {"customers": len(d), "classes": len(levels), "largest_class": str(counts.idxmax()),
               "largest_class_share": float(counts.max() / max(len(d), 1))}
    tables = {"Distribution": t}
    if period:
        ps = sorted(d[period].unique(), key=lambda v: (str(type(v)), v))
        ct = pd.crosstab(d[period], lab).reindex(index=ps, columns=levels, fill_value=0)
        sh = ct.div(ct.sum(axis=1), axis=0)
        sh.index = [str(p) for p in ps]
        tables["Share by period"] = sh.rename_axis("period").reset_index()
        if len(ps) >= 2:
            m = ct.to_numpy().T
            m = m[m.sum(axis=1) > 0]
            chi2, p, dof, _ = stats.chi2_contingency(m, correction=False)
            from ask.validation.t_stability import psi_table
            ref = lab[d[period] == ps[0]]
            psis = [float(psi_table(ref, lab[d[period] == q], categorical=True)["psi_contribution"].sum())
                    for q in ps]
            tables["PSI by period"] = pd.DataFrame({"period": [str(q) for q in ps], "PSI_vs_first": psis})
            summary |= {"homogeneity_chi2": float(chi2), "homogeneity_df": int(dof), "homogeneity_p": float(p),
                        "max_PSI_vs_first": float(max(psis))}
    return Outcome(summary, tables, notes=dropped_note(int((~keep).sum())), rows_used=len(d))


@register("aml.crr_migration", "Customer risk rating migration matrix", "Customer risk rating", _AML,
          params=(P("id"), P("period"), P("grade"), _CRR_ORDER,
                  P("reference_value", "string", required=False, help="From-period (default: first)"),
                  P("current_value", "string", required=False, help="To-period (default: last)")),
          description="""Joins customers present in both periods and tabulates the risk-class migration matrix
(counts and row percentages, classes in risk order), the shares upgraded to higher risk, downgraded and stable,
the share moving more than one class, and customers only in one period (new / exited). Excessive stability can
mean ratings are not refreshed; large unexplained movements can mean unstable inputs.""",
          references=("EBA/GL/2021/02, ML/TF Risk Factors Guidelines (keeping risk assessments up to date)",
                      "Jafry & Schuermann (2004), J. Banking & Finance 28(11) (migration matrices)"))
def crr_migration(ctx: RunContext, id, period, grade, order=None, reference_value=None, current_value=None) -> Outcome:
    df = ctx.df[[id, period, grade]].dropna()
    ps = sorted(df[period].unique(), key=lambda v: (str(type(v)), v))
    if len(ps) < 2:
        raise NotApplicable("Need at least two periods.")

    def _pick(v, default):
        if v is None:
            return default
        hit = [p for p in ps if str(p) == str(v)]
        if not hit:
            raise ValueError(f"Period {v} not found; periods: {[str(p) for p in ps[:12]]}")
        return hit[0]
    p0, p1 = _pick(reference_value, ps[0]), _pick(current_value, ps[-1])
    levels = _sorted_levels(df[grade], order)
    pos = {str(v): i for i, v in enumerate(levels)}
    a = df[df[period] == p0].drop_duplicates(id, keep="last").set_index(id)[grade]
    b = df[df[period] == p1].drop_duplicates(id, keep="last").set_index(id)[grade]
    both = a.index.intersection(b.index)
    if len(both) == 0:
        raise NotApplicable("No customer is present in both periods.")
    fa, fb = a.loc[both].astype(str), b.loc[both].astype(str)
    names = [str(v) for v in levels]
    m = pd.crosstab(fa, fb).reindex(index=names, columns=names, fill_value=0)
    rowpct = m.div(m.sum(axis=1).where(m.sum(axis=1) > 0), axis=0)
    step = fb.map(pos) - fa.map(pos)
    summary = {"from_period": str(p0), "to_period": str(p1), "customers_in_both": len(both),
               "share_stable": float((step == 0).mean()), "share_to_higher_risk": float((step > 0).mean()),
               "share_to_lower_risk": float((step < 0).mean()), "share_moved_more_than_one": float((step.abs() > 1).mean()),
               "new_customers": int(len(b.index.difference(a.index))),
               "exited_customers": int(len(a.index.difference(b.index)))}
    return Outcome(summary, {"Migration counts": m.rename_axis("from \\ to").reset_index(),
                             "Migration row %": rowpct.rename_axis("from \\ to").reset_index()},
                   notes=dropped_note(len(ctx.df) - len(df)), rows_used=len(both) * 2)


def cochran_armitage(k: np.ndarray, n: np.ndarray, scores: np.ndarray | None = None) -> tuple[float, float]:
    """Cochran–Armitage trend test Z and two-sided p (Agresti 2013, §5.3.5 form with N in the variance)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    s = np.arange(len(k), dtype=float) if scores is None else np.asarray(scores, float)
    N, R = n.sum(), k.sum()
    p = R / N
    T = float(np.sum(s * (k - n * p)))
    V = p * (1 - p) * (np.sum(n * s ** 2) - np.sum(n * s) ** 2 / N)
    if V <= 0:
        return float("nan"), float("nan")
    z = T / math.sqrt(V)
    return z, float(2 * stats.norm.sf(abs(z)))


@register("aml.crr_sar_concordance", "Customer risk rating concordance with SAR outcomes", "Customer risk rating",
          _AML,
          params=(P("grade", help="Customer risk class"), P("target", help="SAR filed on the customer (1)"),
                  _CRR_ORDER, P("confidence", "number", default=0.95)),
          description="""Do higher risk classes have more SARs? Per class (in risk order): customers, SAR
customers, SAR rate with exact Clopper–Pearson interval, share of all SARs. Whether the SAR rate is monotone
non-decreasing in risk and the number of reversals; Cochran–Armitage trend test (H0: no linear trend of SAR
rate in class rank; scores 0..K−1); Kendall's tau-b between class rank and SAR flag; AUC of the ordinal class
as a SAR ranker (ties count ½; AUC = (Somers' D + 1)/2). SARs concentrated in low-risk classes challenge the
calibration of the risk factors.""",
          references=("Cochran (1954), Biometrics 10(4); Armitage (1955), Biometrics 11(3)",
                      "Agresti (2013), Categorical Data Analysis, 3rd ed., §5.3", "Kendall (1945), Biometrika 33(3)",
                      "EBA/GL/2021/02, ML/TF Risk Factors Guidelines"))
def crr_sar_concordance(ctx: RunContext, grade, target, order=None, confidence=0.95) -> Outcome:
    df = ctx.df
    y = _flag(df, target)
    keep = y.notna() & df[grade].notna()
    g, y = df.loc[keep, grade], y[keep]
    if y.nunique() < 2:
        raise NotApplicable("SAR flag has a single value.")
    levels = _sorted_levels(g, order)
    pos = {str(v): i for i, v in enumerate(levels)}
    r = g.astype(str).map(pos)
    t = pd.DataFrame({"risk_class": [str(v) for v in levels]})
    t["customers"] = r.value_counts().reindex(range(len(levels)), fill_value=0).to_numpy()
    t["sars"] = y.groupby(r).sum().reindex(range(len(levels)), fill_value=0).to_numpy().astype(int)
    t = _rate_table(t, "risk_class", "sars", "customers", confidence, "sar_rate")
    t["share_of_sars"] = t["sars"] / t["sars"].sum()
    rates = t.loc[t["customers"] > 0, "sar_rate"].to_numpy()
    rev = int((np.diff(rates) < 0).sum())
    z, pz = cochran_armitage(t["sars"].to_numpy(), t["customers"].to_numpy())
    tau = stats.kendalltau(r.to_numpy(), y.to_numpy())
    pos_r, neg_r = r[y == 1].to_numpy(), r[y == 0].to_numpy()
    auc = float(stats.mannwhitneyu(pos_r, neg_r).statistic / (len(pos_r) * len(neg_r)))
    return Outcome({"customers": int(keep.sum()), "sars": int(y.sum()), "classes": len(levels),
                    "monotone_non_decreasing": rev == 0, "reversals": rev, "cochran_armitage_z": z,
                    "cochran_armitage_p": pz, "kendall_tau_b": float(tau.statistic),
                    "kendall_p": float(tau.pvalue), "AUC": auc, "somers_d": 2 * auc - 1,
                    "share_sars_in_top_class": float(t["share_of_sars"].iloc[-1])},
                   {"SAR rate by risk class": t}, notes=dropped_note(int((~keep).sum())), rows_used=int(keep.sum()))
