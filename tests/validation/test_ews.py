"""Early-warning-system tests (t_ews.py): known-answer checks on a small hand-built portfolio."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ask.validation.core import RunContext, run_test


def _ok(res):
    assert res.status == "ok", res.error
    return res


@pytest.fixture
def ctx():
    pop = pd.DataFrame({"cid": [1, 2, 3, 4, 5, 6],
                        "default_date": ["2021-06-30", "2021-06-30", "2021-06-30", None, None, None],
                        "dpd30": ["2021-04-30", "2021-03-31", "2021-05-31", None, None, None]})
    alerts = pd.DataFrame({
        "cid": [1, 1, 2, 2, 4, 5, 99],
        "date": ["2021-03-01", "2021-05-01", "2020-01-01", "2021-07-15", "2020-06-01", "2022-12-01", "2021-01-01"],
        "trigger": ["A", "B", "A", "A", "B", "A", "A"]})
    return RunContext(df=pop, tables={"alerts": alerts}, source_name="ews")


BASE = {"id": "cid", "event_date": "default_date", "signals": "alerts", "observation_end": "2022-12-31"}


def test_hit_rate_hand(ctx):
    r = _ok(run_test("ews.hit_rate", ctx, BASE))
    s = r.summary
    # hits: id1 only; false-alarm ids: id4 (id5's alert is too recent to judge)
    assert s["events"] == 3 and s["hits"] == 1 and s["hit_rate"] == pytest.approx(1 / 3)
    assert s["false_alarm_ids"] == 1 and s["false_alarm_rate"] == pytest.approx(1 / 3)
    assert s["id_precision"] == pytest.approx(0.5)
    assert s["true_alerts"] == 2 and s["false_alerts"] == 2 and s["late_alerts"] == 1 and s["censored_alerts"] == 1
    assert s["alert_precision"] == pytest.approx(0.5) and s["lift"] == pytest.approx(1.0)
    assert any("outside the population" in n for n in r.notes)


def test_hit_rate_from_signal_column():
    pop = pd.DataFrame({"cid": [1, 2, 3], "ev": ["2021-06-30", None, "2021-06-30"],
                        "sig": ["2021-01-01", "2020-01-01", None]})
    r = _ok(run_test("ews.hit_rate", RunContext(df=pop), {"id": "cid", "event_date": "ev", "signal_date": "sig",
                                                         "observation_end": "2022-12-31"}))
    assert r.summary["hit_rate"] == pytest.approx(0.5) and r.summary["false_alarm_rate"] == pytest.approx(1.0)


def test_lead_time_hand(ctx):
    r = _ok(run_test("ews.lead_time", ctx, BASE | {"lead_grid": [0, 90, 180]}))
    assert r.summary["hit_events"] == 1 and r.summary["median_lead_days"] == pytest.approx(121)
    cap = r.tables["Cumulative capture by lead time"].set_index("lead_at_least_days")
    assert cap.loc[90, "share_of_all_events"] == pytest.approx(1 / 3) and cap.loc[180, "events_captured"] == 0


def test_trigger_performance_hand(ctx):
    r = _ok(run_test("ews.trigger_performance", ctx, BASE | {"sig_type": "trigger"}))
    t = r.tables["Trigger performance"].set_index("trigger")
    assert t.loc["A", "hits"] == 1 and t.loc["B", "hits"] == 1
    assert t.loc["A", "unique_hits"] == 0
    # A: true alert id1, false id2-2020 (id5 censored) -> 0.5 ; B: true id1, false id4 -> 0.5
    assert t.loc["A", "alert_precision"] == pytest.approx(0.5) and t.loc["B", "alert_precision"] == pytest.approx(0.5)
    assert t.loc["A", "id_precision"] == pytest.approx(1.0)
    assert run_test("ews.trigger_performance", ctx, BASE).status == "error"


def test_recall_by_horizon_hand(ctx):
    r = _ok(run_test("ews.recall_by_horizon", ctx, BASE | {"horizons": [30, 90, 182]}))
    assert r.summary["recall_30d"] == 0 and r.summary["recall_90d"] == pytest.approx(1 / 3)
    assert r.summary["recall_182d"] == pytest.approx(1 / 3)


def test_alert_workload(ctx):
    r = _ok(run_test("ews.alert_workload", ctx, BASE | {"sig_type": "trigger", "frequency": "Y"}))
    t = r.tables["Alerts per period"].set_index("period")
    assert t.loc["2021", "alerts"] == 3 and t.loc["2021", "new_ids"] == 1
    assert t.loc["2020", "new_ids"] == 2 and r.summary["alerts"] == 6
    assert t.loc["2021", "alerts_per_1000_ids"] == pytest.approx(500)


def test_persistence_hand():
    df = pd.DataFrame({"cid": ["x"] * 5 + ["y"] * 2, "m": [1, 2, 3, 4, 5, 1, 2], "s": [0, 1, 1, 0, 1, 1, 1]})
    r = _ok(run_test("ews.persistence", RunContext(df=df), {"id": "cid", "period": "m", "signal": "s"}))
    s = r.summary
    assert s["flip_rate"] == pytest.approx(3 / 5) and s["persistence"] == pytest.approx(2 / 3)
    assert s["switch_on_rate"] == pytest.approx(1.0) and s["on_spells"] == 3
    assert s["mean_spell_length"] == pytest.approx(5 / 3) and s["flip_flop_ids"] == 1
    r2 = _ok(run_test("ews.persistence", RunContext(df=df.assign(sc=df["s"] * 10)),
                      {"id": "cid", "period": "m", "score": "sc", "threshold": 5}))
    assert r2.summary == s


def test_score_lift_hand():
    df = pd.DataFrame({"y": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0], "s": [10, 9, 9, 7, 6, 5, 4, 3, 2, 1]})
    r = _ok(run_test("ews.score_lift", RunContext(df=df), {"target": "y", "score": "s", "n_bins": 5,
                                                           "top_k": [0.2, 0.3]}))
    pk = r.tables["Precision at k"].set_index("top_share")
    assert pk.loc[0.2, "expected_events"] == pytest.approx(1.5)      # 1 + tie at 9 split 1/2
    assert pk.loc[0.3, "precision_at_k"] == pytest.approx(2 / 3)
    assert r.summary["auc"] == pytest.approx(roc_auc_score(df["y"], df["s"]))
    bands = r.tables["Lift by score band"]
    assert bands["n"].sum() == 10 and bands["cum_capture"].iloc[-1] == pytest.approx(1.0)


def test_dpd_comparison_hand(ctx):
    r = _ok(run_test("ews.dpd_comparison", ctx, {"id": "cid", "dpd_date": "dpd30", "event_date": "default_date",
                                                 "signals": "alerts"}))
    s = r.summary
    # id1 alert 2021-03-01 before 30dpd (60 days); id2: 2020-01-01 is > 365 days before, 2021-07-15 after; id3 none
    assert s["n_reached_30dpd"] == 3
    assert s["share_before_30dpd"] == pytest.approx(1 / 3) and s["share_only_after"] == pytest.approx(1 / 3)
    assert s["median_days_ahead"] == pytest.approx(60)
    assert s["median_30dpd_lead_to_default"] == pytest.approx(61)


def test_determinism_and_errors(ctx):
    a = run_test("ews.hit_rate", ctx, BASE)
    b = run_test("ews.hit_rate", ctx, dict(BASE))
    assert a.run_id == b.run_id and a.summary == b.summary
    assert run_test("ews.hit_rate", ctx, {"id": "cid", "event_date": "default_date"}).status == "error"
    nodef = RunContext(df=ctx.df.assign(default_date=None), tables=ctx.tables)
    assert run_test("ews.hit_rate", nodef, BASE).status == "not_applicable"
