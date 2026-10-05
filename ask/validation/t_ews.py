"""Early-warning-system (EWS) and alerting performance: hit rate, false alarms, lead time, triggers.

Data layout:
  - The active table is the POPULATION: one row per obligor/account (`id`) with the event date
    (`event_date`: default, SAR, ...; empty = no event).
  - Signals (alerts) come either from a `signal_date` column of the population table (one trigger per id),
    or from a second loaded table `signals` with one row per alert (`sig_id`, `sig_date`, optional
    `sig_type`); with `sig_score` + `threshold` only rows with score >= threshold are alerts.

Definitions (lead = event date − alert date, in days):
  - An alert is TRUE if min_lead_days <= lead <= lookback_days (it precedes the event inside the window).
  - It is FALSE if the id has no event, or the event comes more than lookback_days later; false alerts dated
    after observation_end − lookback_days cannot be judged yet (censored) and are left out.
  - Alerts with lead < min_lead_days (on or after the event) are late and left out of precision.
  - Hit rate (recall) = events with at least one true alert / events.
  - False-alarm rate = non-event ids with at least one judgeable alert / non-event ids.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ask.validation.core import NotApplicable, Outcome, P, RunContext, num, register
from ask.validation.t_lgd import _wilson, flag, sorted_levels

_EWS = ("ews",)
_EWS_AML = ("ews", "aml")

_POP = (P("id", help="Obligor / account id (population table, one row per id)"),
        P("event_date", help="Default / event date per id (empty = no event)"))
_SIG = (P("signal_date", required=False, help="Trigger date per id in the population table (one signal per id)"),
        P("signals", "table", required=False, help="Loaded table with one row per alert"),
        P("sig_id", "string", required=False, help="Id column in `signals` (default: same name as id)"),
        P("sig_date", "string", default="date", help="Alert date column in `signals`"),
        P("sig_type", "string", required=False, help="Trigger type column in `signals`"),
        P("sig_score", "string", required=False, help="Score column in `signals` (alerts = score >= threshold)"),
        P("threshold", "number", required=False),
        P("observation_end", "string", required=False,
          help="Last date of the observation window (YYYY-MM-DD); default the latest date in the data"))
_WIN = (P("lookback_days", "integer", default=365, help="Window before the event in which an alert counts"),
        P("min_lead_days", "integer", default=1, help="Minimum lead for an alert to count as early"))


def _dates(s: pd.Series, label: str) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    bad = int((s.notna() & out.isna()).sum())
    if bad:
        raise ValueError(f"{bad} non-empty values of '{label}' are not dates (e.g. {s[s.notna() & out.isna()].iloc[0]!r}).")
    return out


def _load(ctx: RunContext, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
          threshold, observation_end, need_type=False):
    """(population [id, edate], alerts [id, sdate, stype, edate], observation end, notes)."""
    df = ctx.df
    notes = []
    pop = pd.DataFrame({"id": df[id], "edate": _dates(df[event_date], event_date) if event_date else pd.NaT})
    if pop["id"].isna().any():
        notes.append(f"{int(pop['id'].isna().sum())} population rows without id excluded.")
        pop = pop[pop["id"].notna()]
    if pop["id"].duplicated().any():
        raise ValueError(f"'{id}' must be unique in the population table (one row per id).")
    if signals:
        t = ctx.tables[signals]
        sid = sig_id or id
        for c in [sid, sig_date] + [c for c in (sig_type, sig_score) if c]:
            if c not in t.columns:
                raise ValueError(f"Column '{c}' not in the signals table. Columns: {', '.join(map(str, t.columns))}")
        sig = pd.DataFrame({"id": t[sid], "sdate": _dates(t[sig_date], sig_date),
                            "stype": t[sig_type].astype(str) if sig_type else "all"})
        if sig_score:
            if threshold is None:
                raise ValueError("sig_score needs a threshold.")
            sc = pd.to_numeric(t[sig_score], errors="coerce")
            sig = sig[(sc >= threshold).to_numpy()]
    elif signal_date:
        sig = pd.DataFrame({"id": df[id], "sdate": _dates(df[signal_date], signal_date), "stype": "all"})
    else:
        raise ValueError("Give `signal_date` (column of the population table) or a `signals` table.")
    if need_type and not sig_type:
        raise ValueError("This test needs `signals` with a `sig_type` column.")
    n0 = len(sig)
    sig = sig.dropna(subset=["id", "sdate"])
    if signals and n0 - len(sig):
        notes.append(f"{n0 - len(sig)} alerts without id/date excluded.")
    unknown = ~sig["id"].isin(pop["id"])
    if unknown.any():
        notes.append(f"{int(unknown.sum())} alerts for ids outside the population excluded.")
        sig = sig[~unknown]
    sig = sig.merge(pop, on="id", how="left")
    sig["lead"] = (sig["edate"] - sig["sdate"]) / pd.Timedelta(days=1)
    if observation_end:
        end = pd.Timestamp(observation_end)
    else:
        end = max(pop["edate"].max() if pop["edate"].notna().any() else pd.Timestamp.min,
                  sig["sdate"].max() if len(sig) else pd.Timestamp.min)
        notes.append(f"Observation end taken as the latest date in the data ({end.date()}).")
    return pop, sig, end, notes


def _metrics(pop: pd.DataFrame, sig: pd.DataFrame, end, lookback: int, min_lead: int) -> dict:
    lead = sig["lead"]
    true = (lead >= min_lead) & (lead <= lookback)
    judge_by = end - pd.Timedelta(days=lookback)
    false = (lead.isna() | (lead > lookback)) & (sig["sdate"] <= judge_by)
    late = lead < min_lead
    censored = ~true & ~false & ~late
    ev = pop["edate"].notna()
    hit_ids = set(sig.loc[true, "id"])
    n_ev = int(ev.sum())
    tp = int(pop.loc[ev, "id"].isin(hit_ids).sum())
    fp_ids = set(sig.loc[(sig["sdate"] <= judge_by) & sig["edate"].isna(), "id"])
    nonev = pop.loc[~ev, "id"]
    fp = int(nonev.isin(fp_ids).sum())
    n_true, n_false = int(true.sum()), int(false.sum())
    return {"ids": len(pop), "events": n_ev, "hits": tp, "hit_rate": tp / n_ev if n_ev else np.nan,
            "non_event_ids": len(nonev), "false_alarm_ids": fp,
            "false_alarm_rate": fp / len(nonev) if len(nonev) else np.nan,
            "id_precision": tp / (tp + fp) if tp + fp else np.nan,
            "alerts": len(sig), "true_alerts": n_true, "false_alerts": n_false,
            "alert_precision": n_true / (n_true + n_false) if n_true + n_false else np.nan,
            "late_alerts": int(late.sum()), "censored_alerts": int(censored.sum()),
            "base_rate": n_ev / len(pop) if len(pop) else np.nan, "_true": true, "_false": false}


def _clean(m: dict) -> dict:
    return {k: v for k, v in m.items() if not k.startswith("_")}


@register("ews.hit_rate", "EWS hit rate, false-alarm rate and precision", "Discrimination", _EWS_AML,
          params=(*_POP, *_SIG, *_WIN, P("confidence", "number", default=0.95)),
          description="""Id-level and alert-level performance of an early-warning or alerting system over a lookback window.
Hit rate (recall) = events preceded by at least one alert with min_lead_days ≤ lead ≤ lookback_days / events.
False-alarm rate = non-event ids with at least one judgeable alert / non-event ids. Id precision = hits /
(hits + false-alarm ids). Alert precision = true alerts / (true + false alerts). Lift = id precision / base
event rate. Wilson CIs on the rates. Alerts too recent to judge (no event yet but window not elapsed) and
alerts on/after the event are excluded from precision and counted.""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (early-warning indicators, "
                      "monitoring framework)",
                      "Wilson (1927), JASA 22(158)"))
def ews_hit_rate(ctx: RunContext, id, event_date, signal_date=None, signals=None, sig_id=None, sig_date="date",
                 sig_type=None, sig_score=None, threshold=None, observation_end=None, lookback_days=365,
                 min_lead_days=1, confidence=0.95) -> Outcome:
    pop, sig, end, notes = _load(ctx, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
                                 threshold, observation_end)
    m = _clean(_metrics(pop, sig, end, lookback_days, min_lead_days))
    if m["events"] == 0:
        raise NotApplicable("No events in the population.")
    m["lift"] = m["id_precision"] / m["base_rate"] if m["base_rate"] else np.nan
    rows = []
    for name, k, n in (("hit_rate", m["hits"], m["events"]),
                       ("false_alarm_rate", m["false_alarm_ids"], m["non_event_ids"]),
                       ("id_precision", m["hits"], m["hits"] + m["false_alarm_ids"]),
                       ("alert_precision", m["true_alerts"], m["true_alerts"] + m["false_alerts"])):
        lo, hi = _wilson(k, n, confidence)
        rows.append({"metric": name, "count": k, "denominator": n, "rate": k / n if n else np.nan,
                     "ci_low": lo, "ci_high": hi})
    cm = pd.DataFrame({"": ["event", "no event"], "alerted": [m["hits"], m["false_alarm_ids"]],
                       "not alerted": [m["events"] - m["hits"], m["non_event_ids"] - m["false_alarm_ids"]]})
    notes.append(f"Window: alerts {min_lead_days}–{lookback_days} days before the event count as hits.")
    return Outcome(m | {"lookback_days": lookback_days}, {"Rates": pd.DataFrame(rows), "Id-level confusion": cm},
                   notes=notes, rows_used=len(pop))


def _first_true_lead(sig: pd.DataFrame, true: pd.Series) -> pd.Series:
    """Per hit id, the lead of the EARLIEST alert inside the window."""
    return sig[true].groupby("id")["lead"].max()


@register("ews.lead_time", "EWS lead time distribution and cumulative capture", "Timeliness", _EWS,
          params=(*_POP, *_SIG, *_WIN,
                  P("lead_grid", "list", default=[0, 30, 60, 90, 180, 270, 365],
                    help="Lead times (days) for the cumulative capture table")),
          description="""Lead time = days from the FIRST alert inside the window [event − lookback_days, event −
min_lead_days] to the event, per hit event: mean, median, quantiles (P10, P25, P75, P90), min, max. Cumulative
capture by lead time: share of all events (and of hit events) with a lead of at least d days, for each d in
`lead_grid`. Descriptive.""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (early-warning indicators)",))
def ews_lead_time(ctx: RunContext, id, event_date, signal_date=None, signals=None, sig_id=None, sig_date="date",
                  sig_type=None, sig_score=None, threshold=None, observation_end=None, lookback_days=365,
                  min_lead_days=1, lead_grid=(0, 30, 60, 90, 180, 270, 365)) -> Outcome:
    pop, sig, end, notes = _load(ctx, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
                                 threshold, observation_end)
    m = _metrics(pop, sig, end, lookback_days, min_lead_days)
    lead = _first_true_lead(sig, m["_true"]).to_numpy(float)
    if len(lead) == 0:
        raise NotApplicable("No event was preceded by an alert inside the window.")
    q = np.quantile(lead, [0.1, 0.25, 0.5, 0.75, 0.9])
    summary = {"hit_events": len(lead), "events": m["events"], "mean_lead_days": lead.mean(), "median_lead_days": q[2],
               "p10_lead_days": q[0], "p25_lead_days": q[1], "p75_lead_days": q[3], "p90_lead_days": q[4],
               "min_lead_days": lead.min(), "max_lead_days": lead.max()}
    cap = pd.DataFrame([{"lead_at_least_days": float(g), "events_captured": int((lead >= float(g)).sum()),
                         "share_of_all_events": float((lead >= float(g)).sum() / m["events"]),
                         "share_of_hit_events": float((lead >= float(g)).mean())}
                        for g in sorted(float(v) for v in lead_grid)])
    return Outcome(summary, {"Cumulative capture by lead time": cap,
                             "Lead time per hit event": pd.DataFrame({"lead_days": np.sort(lead)})},
                   notes=notes, rows_used=len(pop))


@register("ews.trigger_performance", "Performance per trigger type (hit rate, precision, lift)", "Discrimination",
          _EWS_AML,
          params=(*_POP, *_SIG, *_WIN),
          description="""For each trigger type and all triggers together: alerts, ids alerted, hits (events preceded by an
alert of that type inside the window), hit rate = hits / all events, alert precision (true / judgeable
alerts of that type), id precision = hits / (hits + non-event ids alerted by that type), lift = id precision /
base event rate, median lead of the earliest in-window alert, and the number of hits ONLY that trigger caught
(its unique contribution). Descriptive; triggers overlap, so hit rates do not add up.""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (early-warning indicators)",))
def ews_trigger_performance(ctx: RunContext, id, event_date, signal_date=None, signals=None, sig_id=None,
                            sig_date="date", sig_type=None, sig_score=None, threshold=None, observation_end=None,
                            lookback_days=365, min_lead_days=1) -> Outcome:
    pop, sig, end, notes = _load(ctx, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
                                 threshold, observation_end, need_type=True)
    allm = _metrics(pop, sig, end, lookback_days, min_lead_days)
    if allm["events"] == 0:
        raise NotApplicable("No events in the population.")
    hit_by_type = sig[allm["_true"]].groupby("id")["stype"].agg(lambda s: set(s))
    rows = []
    for ty in [*sorted_levels(sig["stype"]), "ALL"]:
        s = sig if ty == "ALL" else sig[sig["stype"] == ty]
        m = _metrics(pop, s, end, lookback_days, min_lead_days)
        lead = _first_true_lead(s, m["_true"])
        only = int(sum(1 for v in hit_by_type if v == {ty})) if ty != "ALL" else np.nan
        rows.append({"trigger": str(ty), "alerts": len(s), "ids_alerted": int(s["id"].nunique()), "hits": m["hits"],
                     "hit_rate": m["hit_rate"], "alert_precision": m["alert_precision"],
                     "id_precision": m["id_precision"],
                     "lift": m["id_precision"] / m["base_rate"] if m["base_rate"] else np.nan,
                     "median_lead_days": float(lead.median()) if len(lead) else np.nan, "unique_hits": only})
    t = pd.DataFrame(rows)
    best = t[t["trigger"] != "ALL"].sort_values(["hit_rate", "trigger"], ascending=[False, True]).iloc[0]
    return Outcome({"trigger_types": len(t) - 1, "events": allm["events"], "hit_rate_all": float(t.iloc[-1]["hit_rate"]),
                    "highest_hit_rate_trigger": best["trigger"], "highest_hit_rate": float(best["hit_rate"])},
                   {"Trigger performance": t}, notes=notes, rows_used=len(pop))


@register("ews.recall_by_horizon", "Recall and precision by warning horizon (e.g. 3 / 6 / 12 months)",
          "Timeliness", _EWS_AML,
          params=(*_POP, *_SIG, P("horizons", "list", default=[91, 182, 365], help="Windows in days"),
                  P("min_lead_days", "integer", default=1), P("confidence", "number", default=0.95)),
          description="""For each warning horizon h: recall = events with an alert between min_lead_days and h days before
the event / events (Wilson CI), alert precision and id precision with the same window, and the false-alarm
rate. Shows how much warning the system gives: recall at 3 months vs 12 months.""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (early-warning indicators)",))
def ews_recall_by_horizon(ctx: RunContext, id, event_date, signal_date=None, signals=None, sig_id=None,
                          sig_date="date", sig_type=None, sig_score=None, threshold=None, observation_end=None,
                          horizons=(91, 182, 365), min_lead_days=1, confidence=0.95) -> Outcome:
    pop, sig, end, notes = _load(ctx, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
                                 threshold, observation_end)
    rows = []
    for h in sorted(int(float(v)) for v in horizons):
        m = _clean(_metrics(pop, sig, end, h, min_lead_days))
        lo, hi = _wilson(m["hits"], m["events"], confidence)
        rows.append({"horizon_days": h, "events": m["events"], "hits": m["hits"], "recall": m["hit_rate"],
                     "recall_ci_low": lo, "recall_ci_high": hi, "alert_precision": m["alert_precision"],
                     "id_precision": m["id_precision"], "false_alarm_rate": m["false_alarm_rate"]})
    t = pd.DataFrame(rows)
    if t["events"].iloc[0] == 0:
        raise NotApplicable("No events in the population.")
    return Outcome({f"recall_{int(h)}d": r for h, r in zip(t["horizon_days"], t["recall"])} |
                   {"events": int(t["events"].iloc[0])}, {"Recall by horizon": t}, notes=notes, rows_used=len(pop))


@register("ews.alert_workload", "Alert workload per period", "Operational", _EWS_AML,
          params=(*_POP, *_SIG, P("frequency", "string", default="M", choices=("W", "M", "Q", "Y")),
                  P("lookback_days", "integer", default=365), P("min_lead_days", "integer", default=1)),
          description="""Alerts per calendar period (week / month / quarter / year): number of alerts, distinct ids
alerted, ids alerted for the first time, alerts per 1,000 ids in the population, and how many of the period's
alerts turned out true (event within the window) or false (judgeable, no event); per trigger type when
`sig_type` is given. Shows the case-handling workload the alert threshold implies.""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (monitoring framework)",))
def ews_alert_workload(ctx: RunContext, id, event_date, signal_date=None, signals=None, sig_id=None,
                       sig_date="date", sig_type=None, sig_score=None, threshold=None, observation_end=None,
                       frequency="M", lookback_days=365, min_lead_days=1) -> Outcome:
    pop, sig, end, notes = _load(ctx, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
                                 threshold, observation_end)
    if sig.empty:
        raise NotApplicable("No alerts.")
    m = _metrics(pop, sig, end, lookback_days, min_lead_days)
    s = sig.assign(per=sig["sdate"].dt.to_period(frequency).astype(str), true=m["_true"], false=m["_false"])
    first = s.groupby("id")["sdate"].transform("min") == s["sdate"]
    s["first"] = first & ~s.duplicated(["id", "sdate"])
    t = s.groupby("per", sort=True).agg(alerts=("id", "size"), ids_alerted=("id", "nunique"),
                                        new_ids=("first", "sum"), true_alerts=("true", "sum"),
                                        false_alerts=("false", "sum")).reset_index().rename(columns={"per": "period"})
    t["alerts_per_1000_ids"] = t["alerts"] / len(pop) * 1000
    tabs = {"Alerts per period": t}
    if sig_type:
        bt = s.pivot_table(index="per", columns="stype", values="id", aggfunc="size", fill_value=0)
        tabs["Alerts per period and trigger"] = bt.reset_index().rename(columns={"per": "period"})
    return Outcome({"periods": len(t), "alerts": len(sig), "mean_alerts_per_period": float(t["alerts"].mean()),
                    "max_alerts_per_period": int(t["alerts"].max()),
                    "mean_ids_alerted_per_period": float(t["ids_alerted"].mean()), "population_ids": len(pop)},
                   tabs, notes=notes + ["Periods without alerts are not listed."], rows_used=len(sig))


@register("ews.persistence", "Signal persistence and flip-flop rate", "Stability", _EWS,
          params=(P("id"), P("period", help="Observation date / period of the panel"),
                  P("signal", required=False, help="0/1 signal status per id and period"),
                  P("score", required=False, help="Score (signal = score >= threshold), instead of `signal`"),
                  P("threshold", "number", required=False)),
          description="""On a panel of signal status per id and period (consecutive observations of the same id):
persistence P(on_t | on_{t−1}), switch-on rate P(on_t | off_{t−1}), overall flip rate (status changes /
consecutive pairs), on-spell lengths (mean, median, share of one-period spells) and flip-flop ids (ids with two
or more separate on-spells). High flip-flop means unstable alerts that create repeated case work. Gaps in the
panel are not filled: consecutive rows of an id are treated as consecutive periods.""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (early-warning indicators)",))
def ews_persistence(ctx: RunContext, id, period, signal=None, score=None, threshold=None) -> Outcome:
    df = ctx.df
    if signal:
        on = flag(df[signal], signal)
    elif score is not None and threshold is not None:
        sc = num(df, score)
        on = (sc >= threshold).astype(float).where(sc.notna())
    else:
        raise ValueError("Give `signal`, or `score` with `threshold`.")
    d = pd.DataFrame({"id": df[id], "per": df[period], "on": on}).dropna()
    if d.duplicated(["id", "per"]).any():
        raise ValueError("Duplicate (id, period) rows.")
    d = d.sort_values(["id", "per"], kind="mergesort")
    prev = d.groupby("id")["on"].shift()
    pair = prev.notna()
    a, b = prev[pair], d.loc[pair, "on"]
    if len(a) == 0:
        raise NotApplicable("No id has two consecutive observations.")
    tm = pd.crosstab(a.astype(int), b.astype(int)).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    start = (d["on"] == 1) & (prev != 1)          # spell starts (prev NaN or 0)
    spell_id = start.astype(int).groupby(d["id"]).cumsum()
    spells = d[d["on"] == 1].assign(spell=spell_id[d["on"] == 1]).groupby(["id", "spell"]).size()
    per_id = spells.groupby(level="id").size()
    summary = {"ids": int(d["id"].nunique()), "observations": len(d), "share_on": float(d["on"].mean()),
               "persistence": float(tm.loc[1, 1] / tm.loc[1].sum()) if tm.loc[1].sum() else np.nan,
               "switch_on_rate": float(tm.loc[0, 1] / tm.loc[0].sum()) if tm.loc[0].sum() else np.nan,
               "flip_rate": float((a != b).mean()), "on_spells": len(spells),
               "mean_spell_length": float(spells.mean()) if len(spells) else np.nan,
               "share_one_period_spells": float((spells == 1).mean()) if len(spells) else np.nan,
               "flip_flop_ids": int((per_id >= 2).sum()),
               "share_alerted_ids_flip_flop": float((per_id >= 2).mean()) if len(per_id) else np.nan}
    tm.index.name, tm.columns.name = "status_t-1", "status_t"
    lens = spells.value_counts().sort_index()
    return Outcome(summary, {"Transition counts": tm.reset_index(),
                             "On-spell lengths": pd.DataFrame({"spell_length": lens.index.astype(int),
                                                               "spells": lens.to_numpy()})},
                   notes=["Consecutive observations of an id are treated as consecutive periods."], rows_used=len(d))


@register("ews.score_lift", "Precision@k and lift by score band", "Discrimination", _EWS_AML,
          params=(P("target", help="1 = event within the horizon after the scoring date"), P("score"),
                  P("n_bins", "integer", default=10), P("top_k", "list", default=[0.01, 0.05, 0.1],
                                                        help="Population shares for precision@k")),
          description="""For a risk score (higher = riskier): bands by score quantiles (band 1 = highest scores) with n,
events, event rate, lift = band event rate / overall rate, cumulative capture of events and cumulative lift;
precision@k = event rate among the top k share of scores (ties at the cut-off split proportionally, so the
result does not depend on row order), recall@k and lift@k; and the AUC.""",
          references=("Siddiqi (2006), Credit Risk Scorecards, Wiley (gains / lift tables)",
                      "Provost & Fawcett (2013), Data Science for Business, O'Reilly (lift, precision@k)"))
def ews_score_lift(ctx: RunContext, target, score, n_bins=10, top_k=(0.01, 0.05, 0.1)) -> Outcome:
    d = pd.DataFrame({"y": flag(ctx.df[target], target), "s": num(ctx.df, score)})
    n0 = len(d)
    d = d.dropna()
    if d["y"].nunique() < 2:
        raise NotApplicable("Target has a single class.")
    n, ev = len(d), float(d["y"].sum())
    base = ev / n
    q = d["s"].rank(method="average", ascending=False, pct=True)
    band = np.ceil(q * n_bins).clip(1, n_bins).astype(int)
    rows, cum_n, cum_e = [], 0, 0.0
    for b in sorted(band.unique()):
        s = d[band == b]
        cum_n += len(s)
        cum_e += s["y"].sum()
        rows.append({"band": int(b), "n": len(s), "min_score": float(s["s"].min()), "max_score": float(s["s"].max()),
                     "events": int(s["y"].sum()), "event_rate": float(s["y"].mean()), "lift": float(s["y"].mean() / base),
                     "cum_share_population": cum_n / n, "cum_capture": cum_e / ev, "cum_lift": (cum_e / ev) / (cum_n / n)})
    srt = np.sort(d["s"].to_numpy())[::-1]
    pk = []
    for k in sorted(float(v) for v in top_k):
        m = max(1, int(round(k * n)))
        cut = srt[m - 1]
        above = d["s"] > cut
        at = d["s"] == cut
        e = float(d.loc[above, "y"].sum() + d.loc[at, "y"].sum() * (m - above.sum()) / at.sum())
        pk.append({"top_share": k, "top_n": m, "expected_events": e, "precision_at_k": e / m, "recall_at_k": e / ev,
                   "lift_at_k": (e / m) / base})
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(d["y"], d["s"]))
    pkt = pd.DataFrame(pk)
    out = {"auc": auc, "base_rate": base, "n": n, "events": int(ev)}
    for _, r in pkt.iterrows():
        out[f"precision_at_{r['top_share']:g}"] = r["precision_at_k"]
        out[f"lift_at_{r['top_share']:g}"] = r["lift_at_k"]
    return Outcome(out, {"Lift by score band": pd.DataFrame(rows), "Precision at k": pkt},
                   notes=[f"{n0 - n} rows with missing inputs excluded."] if n0 - n else [], rows_used=n)


@register("ews.dpd_comparison", "EWS timing vs days-past-due (does EWS fire before 30 dpd?)", "Timeliness", _EWS,
          params=(P("id"), P("dpd_date", help="Date the account first reached 30 days past due"),
                  P("event_date", required=False, help="Default date, to compare both lead times to default"),
                  *_SIG, P("lookback_days", "integer", default=365,
                           help="Window before (and after) the 30-dpd date in which alerts are considered")),
          description="""For accounts that reached 30 days past due: share where the EWS fired BEFORE the 30-dpd date (first
alert within lookback_days before it), share where it fired only after (within lookback_days after), and share
never alerted around it; distribution of days of advance warning over the dpd backstop (median, mean,
quartiles). With `event_date`, median lead to default of the EWS alert vs of the 30-dpd date (an EWS adds value
only if it warns earlier than the arrears backstop).""",
          references=("EBA/GL/2020/06, Guidelines on loan origination and monitoring (early-warning indicators)",
                      "IFRS 9 Financial Instruments, paragraph 5.5.11 (30 days past due rebuttable presumption)"))
def ews_dpd_comparison(ctx: RunContext, id, dpd_date, event_date=None, signal_date=None, signals=None, sig_id=None,
                       sig_date="date", sig_type=None, sig_score=None, threshold=None, observation_end=None,
                       lookback_days=365) -> Outcome:
    pop, sig, end, notes = _load(ctx, id, event_date, signal_date, signals, sig_id, sig_date, sig_type, sig_score,
                                 threshold, observation_end)
    dd = pd.DataFrame({"id": ctx.df[id], "ddate": _dates(ctx.df[dpd_date], dpd_date)})
    dd = dd[dd["id"].isin(pop["id"]) & dd["ddate"].notna()]
    if dd.empty:
        raise NotApplicable("No account reached 30 days past due.")
    s = sig.merge(dd, on="id", how="inner")
    gap = (s["ddate"] - s["sdate"]) / pd.Timedelta(days=1)        # > 0: alert before 30 dpd
    before = s[(gap > 0) & (gap <= lookback_days)].groupby("id")["sdate"].min()
    after = s[(gap <= 0) & (gap >= -lookback_days)].groupby("id")["sdate"].min()
    dd = dd.set_index("id")
    dd["first_before"] = before.reindex(dd.index)
    dd["first_after"] = after.reindex(dd.index)
    dd["status"] = np.where(dd["first_before"].notna(), "before 30 dpd",
                            np.where(dd["first_after"].notna(), "only on/after 30 dpd", "not alerted"))
    adv = ((dd["ddate"] - dd["first_before"]) / pd.Timedelta(days=1)).dropna().to_numpy()
    n = len(dd)
    summary = {"n_reached_30dpd": n, "share_before_30dpd": float((dd["status"] == "before 30 dpd").mean()),
               "share_only_after": float((dd["status"] == "only on/after 30 dpd").mean()),
               "share_not_alerted": float((dd["status"] == "not alerted").mean())}
    if len(adv):
        q = np.quantile(adv, [0.25, 0.5, 0.75])
        summary |= {"median_days_ahead": q[1], "mean_days_ahead": float(adv.mean()), "p25_days_ahead": q[0],
                    "p75_days_ahead": q[2]}
    tabs = {"Timing vs 30 dpd": dd["status"].value_counts().reindex(
        ["before 30 dpd", "only on/after 30 dpd", "not alerted"], fill_value=0).rename_axis("status")
        .reset_index(name="accounts").assign(share=lambda t: t["accounts"] / n)}
    if event_date:
        e = pop.set_index("id")["edate"].reindex(dd.index)
        ews_lead = ((e - dd["first_before"]) / pd.Timedelta(days=1)).dropna()
        dpd_lead = ((e - dd["ddate"]) / pd.Timedelta(days=1)).dropna()
        summary |= {"median_ews_lead_to_default": float(ews_lead.median()) if len(ews_lead) else np.nan,
                    "median_30dpd_lead_to_default": float(dpd_lead.median()) if len(dpd_lead) else np.nan,
                    "n_defaulted": int(e.notna().sum())}
    return Outcome(summary, tabs, notes=notes, rows_used=n)
