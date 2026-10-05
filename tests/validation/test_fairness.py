"""Fairness tests (ask/validation/t_fairness.py): hand-built confusion tables, cross-checks against
scipy / sklearn, NotApplicable paths, NaN handling and determinism."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ask.validation import core
from ask.validation.core import RunContext, run_test
from ask.validation.models import LoadedModel


def _cells(group, tp, fn, fp, tn):
    y = [1] * (tp + fn) + [0] * (fp + tn)
    yhat = [1] * tp + [0] * fn + [1] * fp + [0] * tn
    return pd.DataFrame({"grp": group, "y": y, "yhat": yhat})


@pytest.fixture
def hand():
    """A: n=100, TP 30 FN 10 FP 10 TN 50.  B: n=50, TP 15 FN 10 FP 10 TN 15."""
    df = pd.concat([_cells("A", 30, 10, 10, 50), _cells("B", 15, 10, 10, 15)], ignore_index=True)
    df["score"] = np.where(df["yhat"] == 1, 0.8, 0.2)
    return df


def _ctx(df, models=None):
    return RunContext(df=df, models=models or {}, source_name="synthetic")


def _ok(res):
    assert res.status == "ok", res.error
    return res


P = {"target": "y", "protected": "grp", "predicted": "yhat"}


def test_group_metrics_hand_table(hand):
    r = _ok(run_test("fairness.group_metrics", _ctx(hand), P))
    t = r.tables["Metrics by group"].set_index("group")
    assert t.loc["A", "tpr"] == pytest.approx(0.75) and t.loc["B", "tpr"] == pytest.approx(0.6)
    assert t.loc["A", "fpr"] == pytest.approx(10 / 60) and t.loc["B", "fpr"] == pytest.approx(0.4)
    assert t.loc["A", "ppv"] == pytest.approx(0.75) and t.loc["B", "npv"] == pytest.approx(0.6)
    assert t.loc["B", "for"] == pytest.approx(10 / 25) and t.loc["A", "fdr"] == pytest.approx(0.25)
    assert r.summary["reference_group"] == "A" and r.summary["favourable_rate_max_minus_min"] == pytest.approx(0.1)
    # score + threshold gives the same decisions
    s = _ok(run_test("fairness.group_metrics", _ctx(hand), {"target": "y", "protected": "grp", "score": "score"}))
    assert s.tables["Metrics by group"].equals(r.tables["Metrics by group"])


def test_demographic_parity_disparate_impact(hand):
    r = _ok(run_test("fairness.demographic_parity", _ctx(hand), P))
    S = r.summary
    assert S["statistical_parity_difference"] == pytest.approx(0.1)
    assert S["disparate_impact_min_over_max"] == pytest.approx(0.5 / 0.6)
    assert S["lowest_ratio_vs_reference"] == pytest.approx(0.5 / 0.6) and S["lowest_ratio_group"] == "B"
    t = r.tables["Parity by group"]
    row = t[(t["metric"] == "favourable_rate") & (t["group"] == "B")].iloc[0]
    p = 85 / 150
    z = (0.5 - 0.6) / np.sqrt(p * (1 - p) * (1 / 50 + 1 / 100))
    assert row["z"] == pytest.approx(z) and row["p_value"] == pytest.approx(2 * stats.norm.sf(abs(z)))
    ct = pd.crosstab(hand["grp"], hand["yhat"] == 0)
    assert S["chi2_p_value"] == pytest.approx(stats.chi2_contingency(ct, correction=False).pvalue)
    # favourable label 1 flips the rates: 0.4 (A) vs 0.5 (B)
    r1 = _ok(run_test("fairness.demographic_parity", _ctx(hand), P | {"favourable_label": 1}))
    assert r1.summary["disparate_impact_min_over_max"] == pytest.approx(0.8)
    # explicit reference group
    rb = _ok(run_test("fairness.demographic_parity", _ctx(hand), P | {"reference_group": "b"}))
    assert rb.summary["reference_group"] == "B"
    assert run_test("fairness.demographic_parity", _ctx(hand), P | {"reference_group": "Z"}).status == "error"


def test_equal_opportunity_equalized_odds_predictive_parity(hand):
    eo = _ok(run_test("fairness.equal_opportunity", _ctx(hand), P))
    assert eo.summary["tpr_max_minus_min"] == pytest.approx(0.15)
    assert eo.summary["tpr_min_over_max"] == pytest.approx(0.8)
    pos = hand[hand.y == 1]
    assert eo.summary["chi2_p_value"] == pytest.approx(
        stats.chi2_contingency(pd.crosstab(pos.grp, pos.yhat), correction=False).pvalue)
    od = _ok(run_test("fairness.equalized_odds", _ctx(hand), P))
    assert od.summary["fpr_max_minus_min"] == pytest.approx(0.4 - 10 / 60)
    assert od.summary["equalized_odds_difference"] == pytest.approx(0.4 - 10 / 60)
    assert od.summary["equalized_odds_ratio"] == pytest.approx((10 / 60) / 0.4)
    pp = _ok(run_test("fairness.predictive_parity", _ctx(hand), P))
    assert pp.summary["ppv_max_minus_min"] == pytest.approx(0.15)
    assert pp.summary["npv_max_minus_min"] == pytest.approx(50 / 60 - 0.6)


def test_error_rate_balance_and_treatment_equality(hand):
    r = _ok(run_test("fairness.error_rate_balance", _ctx(hand), P))
    t = r.tables["Error rates by group"]
    b = t[t["group"] == "B"].set_index("metric")
    assert b.loc["fnr", "value"] == pytest.approx(0.4) and b.loc["fnr", "ratio_vs_reference"] == pytest.approx(1.6)
    assert b.loc["fpr", "ratio_vs_reference"] == pytest.approx(0.4 / (10 / 60))
    te = _ok(run_test("fairness.treatment_equality", _ctx(hand), P))
    assert te.summary["min_fn_fp_ratio"] == pytest.approx(1.0) and te.summary["max_over_min"] == pytest.approx(1.0)
    d2 = pd.concat([_cells("A", 30, 20, 10, 40), _cells("B", 15, 10, 10, 15)], ignore_index=True)
    te2 = _ok(run_test("fairness.treatment_equality", _ctx(d2), P))
    assert te2.summary["max_over_min"] == pytest.approx(2.0)


def test_group_difference_tests_match_scipy(hand):
    r = _ok(run_test("fairness.group_difference_tests", _ctx(hand), P))
    t = r.tables["Group difference tests"].set_index("quantity")
    ct = pd.crosstab(hand.grp, hand.y)
    ref = stats.chi2_contingency(ct, correction=False)
    row = t.loc["base rate (outcome)"]
    assert row["chi2"] == pytest.approx(ref.statistic) and row["p_value"] == pytest.approx(ref.pvalue)
    assert row["fisher_p_value"] == pytest.approx(stats.fisher_exact(ct.to_numpy()).pvalue)
    assert row["cramers_v"] == pytest.approx(np.sqrt(ref.statistic / len(hand)))


def test_intersectional_epsilon(hand):
    d = hand.copy()
    d["age"] = np.where(np.arange(len(d)) % 3 == 0, "old", "young")
    r = _ok(run_test("fairness.intersectional", _ctx(d), P | {"protected_2": "age"}))
    cell = d.assign(c=d.grp + " | " + d.age).groupby("c")["yhat"].mean()
    eps = max(np.log(cell.max()) - np.log(cell.min()), np.log((1 - cell).max()) - np.log((1 - cell).min()))
    assert r.summary["cells"] == 4 and r.summary["differential_fairness_epsilon"] == pytest.approx(eps)
    assert r.summary["smallest_cell_n"] == int(d.assign(c=d.grp + d.age).groupby("c").size().min())


def test_conditional_parity_simpson():
    """Rates differ overall (0.68 vs 0.32) but are identical within each stratum: MH odds ratio = 1."""
    def block(g, s, n, fav):
        return pd.DataFrame({"grp": g, "stratum": s, "y": [0, 1] * (n // 2), "yhat": [0] * fav + [1] * (n - fav)})
    d = pd.concat([block("A", "s1", 80, 64), block("B", "s1", 20, 16), block("A", "s2", 20, 4),
                   block("B", "s2", 80, 16)], ignore_index=True)
    dp = _ok(run_test("fairness.demographic_parity", _ctx(d), P))
    assert dp.summary["statistical_parity_difference"] == pytest.approx(0.36)
    r = _ok(run_test("fairness.conditional_parity", _ctx(d), P | {"segment": "stratum"}))
    t = r.tables["Mantel–Haenszel vs reference"].iloc[0]
    assert t["mh_odds_ratio"] == pytest.approx(1.0) and t["cmh_p_value"] == pytest.approx(1.0)


def _slow_delong_se(y, s):
    pos, neg = s[y == 1], s[y == 0]
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    return np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg))


@pytest.fixture
def scored():
    rng = np.random.default_rng(11)
    n = 1200
    g = rng.choice(["F", "M", "X"], n, p=[0.45, 0.45, 0.1])
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1 + x + 0.5 * (g == "M"))))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"grp": g, "x": x, "y": y, "p": p, "sex": (g == "M").astype(float)})


def test_auc_by_group_matches_sklearn_and_delong(scored):
    d = scored.copy()
    d.loc[:4, "p"] = np.nan
    r = _ok(run_test("fairness.auc_by_group", _ctx(d), {"target": "y", "protected": "grp", "score": "p"}))
    assert r.rows_used == len(d) - 5
    t = r.tables["AUC by group"].set_index("group")
    dd = d.dropna()
    for g in ("F", "M", "X"):
        x = dd[dd.grp == g]
        assert t.loc[g, "auc"] == pytest.approx(roc_auc_score(x.y, x.p))
        assert t.loc[g, "auc_se"] == pytest.approx(_slow_delong_se(x.y.to_numpy(), x.p.to_numpy()))
    m = dd.grp == "X"
    bpsn = roc_auc_score(np.r_[np.ones(((~m) & (dd.y == 1)).sum()), np.zeros((m & (dd.y == 0)).sum())],
                         np.r_[dd.p[(~m) & (dd.y == 1)], dd.p[m & (dd.y == 0)]])
    assert t.loc["X", "bpsn_auc"] == pytest.approx(bpsn)


def test_calibration_and_score_distribution(scored):
    r = _ok(run_test("fairness.calibration_by_group", _ctx(scored), {"target": "y", "protected": "grp", "score": "p",
                                                                      "bins": 5}))
    assert r.tables["Group x bin"]["n"].sum() == len(scored)
    t = r.tables["Calibration by group"].set_index("group")
    x = scored[scored.grp == "F"]
    assert t.loc["F", "observed_over_expected"] == pytest.approx(x.y.sum() / x.p.sum())
    assert (t["hl_p_value"] > 0).all()
    bad = scored.assign(p=scored.p * 5)
    assert run_test("fairness.calibration_by_group", _ctx(bad),
                    {"target": "y", "protected": "grp", "score": "p"}).status == "error"
    s = _ok(run_test("fairness.score_distribution", _ctx(scored), {"protected": "grp", "score": "p", "target": "y"}))
    tab = s.tables["Score distribution by group"]
    row = tab[(tab.subset == "all") & (tab.group == "M")].iloc[0]
    ref = scored[scored.grp == s.summary["reference_group"]].p
    assert row["ks_p_value"] == pytest.approx(stats.ks_2samp(scored[scored.grp == "M"].p, ref).pvalue)
    assert set(tab["subset"]) == {"all", "Y = 0", "Y = 1"}


def test_counterfactual_flip(scored):
    lr = LogisticRegression().fit(scored[["x", "sex"]], scored["y"])
    lr_blind = LogisticRegression().fit(scored[["x"]], scored["y"])
    models = {"m": LoadedModel("m", "mem", lr, "sklearn-like", "", ["x", "sex"]),
              "blind": LoadedModel("blind", "mem", lr_blind, "sklearn-like", "", ["x"])}
    r = _ok(run_test("fairness.counterfactual_flip", _ctx(scored, models), {"model": "m", "protected": "sex"}))
    t = r.tables["Counterfactual changes"].set_index(["original_value", "counterfactual_value"])
    assert t.loc[("0.0", "1.0"), "mean_score_change"] > 0          # positive coefficient on sex
    assert r.summary["share_rows_any_label_flip"] > 0
    flips = ((lr.predict_proba(scored[["x"]].assign(sex=1.0))[:, 1] >= 0.5)
             != (lr.predict_proba(scored[["x", "sex"]])[:, 1] >= 0.5))[scored.sex.to_numpy() == 0].mean()
    assert t.loc[("0.0", "1.0"), "share_label_flips"] == pytest.approx(flips)
    na = run_test("fairness.counterfactual_flip", _ctx(scored, models), {"model": "blind", "protected": "sex"})
    assert na.status == "not_applicable"


def test_edge_cases_and_determinism(hand):
    one = hand[hand.grp == "A"]
    assert run_test("fairness.demographic_parity", _ctx(one), P).status == "not_applicable"
    assert run_test("fairness.group_metrics", _ctx(hand), {"target": "y", "protected": "grp"}).status == "error"
    d = hand.copy()
    d.loc[[0, 1, 2], "grp"] = None
    r = _ok(run_test("fairness.equalized_odds", _ctx(d), P))
    assert r.rows_used == len(d) - 3 and any("3 rows" in n for n in r.notes)
    a = run_test("fairness.intersectional", _ctx(hand.assign(a2="z")), P | {"protected_2": "a2"})
    b = run_test("fairness.intersectional", _ctx(hand.assign(a2="z")), P | {"protected_2": "a2"})
    assert a.run_id == b.run_id and a.summary == b.summary


def test_every_fairness_test_declared_well():
    core.load_all()
    ids = [k for k in core.REGISTRY if k.startswith("fairness.")]
    assert len(ids) == 14
    for k in ids:
        assert core.REGISTRY[k].references and len(core.REGISTRY[k].description) > 80, k
