"""AML validation tests (ask/validation/t_aml.py): known answers and edge cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ask.validation.core import RunContext, run_test
from ask.validation.t_aml import (_leading_digits, clopper_pearson, cochran_armitage, cp_upper_one_sided,
                                  discovery_sample_size_binomial, discovery_sample_size_hypergeom)


def _ctx(df=None):
    return RunContext(df=df, source_name="aml")


def _ok(test_id, df, params):
    res = run_test(test_id, _ctx(df), params)
    assert res.status == "ok", f"{test_id}: {res.error}"
    return res


# ── exact intervals and sample sizes ───────────────────────────────────────

def test_clopper_pearson_matches_beta_ppf():
    lo, hi = clopper_pearson(3, 20, 0.95)
    assert lo == pytest.approx(stats.beta.ppf(0.025, 3, 18))
    assert hi == pytest.approx(stats.beta.ppf(0.975, 4, 17))
    lo, hi = clopper_pearson(0, 30, 0.9)
    assert lo == 0 and hi == pytest.approx(1 - 0.05 ** (1 / 30))
    lo, hi = clopper_pearson(30, 30, 0.95)
    assert hi == 1 and lo == pytest.approx(0.025 ** (1 / 30))
    assert cp_upper_one_sided(0, 59, 0.95) == pytest.approx(1 - 0.05 ** (1 / 59))
    # agrees with scipy's own exact binomial CI
    ci = stats.binomtest(7, 50).proportion_ci(0.95, method="exact")
    assert clopper_pearson(7, 50)[0] == pytest.approx(ci.low) and clopper_pearson(7, 50)[1] == pytest.approx(ci.high)


def test_sample_size_known_values():
    assert discovery_sample_size_binomial(0.95, 0.05, 0) == 59
    assert discovery_sample_size_binomial(0.95, 0.05, 1) == 93          # AICPA attribute table
    assert discovery_sample_size_binomial(0.90, 0.10, 0) == 22
    n, D = discovery_sample_size_hypergeom(0.95, 0.05, 1000)
    assert D == 50
    assert stats.hypergeom.cdf(0, 1000, 50, n) <= 0.05 < stats.hypergeom.cdf(0, 1000, 50, n - 1)
    assert n <= 59
    res = _ok("aml.sample_size", None, {"population": 1000, "margin": 0.05})
    assert res.summary["binomial_sample_size"] == 59
    assert res.summary["estimation_sample_size"] == 385                # 1.96² · 0.25 / 0.05² = 384.1
    n0 = stats.norm.ppf(0.975) ** 2 * 0.25 / 0.0025
    assert res.summary["estimation_sample_size_fpc"] == math.ceil(n0 / (1 + (n0 - 1) / 1000))
    assert run_test("aml.sample_size", _ctx(), {"tolerable_rate": 1.5}).status == "error"


# ── alerts and funnel ──────────────────────────────────────────────────────

def test_alert_rate_by_group():
    df = pd.DataFrame({"alert": [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                       "scenario": ["S1", None, None, None, "S2", "S1", None, None, None, None],
                       "seg": list("AAAAABBBBB"), "month": ["m1", "m2"] * 5})
    res = _ok("aml.alert_rate", df, {"alert": "alert", "scenario": "scenario", "segment": "seg", "period": "month"})
    assert res.summary["alert_rate"] == pytest.approx(0.3)
    lo, hi = clopper_pearson(3, 10)
    assert res.summary["ci_low"] == pytest.approx(lo)
    seg = res.tables["Alert rate by segment"].set_index("segment")
    assert seg.loc["A", "alert_rate"] == pytest.approx(0.4) and seg.loc["B", "alert_rate"] == pytest.approx(0.2)
    sc = res.tables["Alert rate by scenario"].set_index("scenario")
    assert sc.loc["S1", "alerts"] == 2 and sc.loc["S1", "alert_rate"] == pytest.approx(0.2)


def test_alert_funnel_hand_numbers():
    df = pd.DataFrame({"scenario": ["A"] * 10 + ["B"] * 10,
                       "case": [1] * 4 + [0] * 6 + [1] * 2 + [0] * 8,
                       "sar": [1] * 2 + [0] * 8 + [1] + [0] * 9})
    res = _ok("aml.alert_funnel", df, {"target": "sar", "case": "case", "scenario": "scenario"})
    assert res.summary["alerts"] == 20 and res.summary["sars"] == 3 and res.summary["cases"] == 6
    assert res.summary["case_to_SAR"] == pytest.approx(0.5)
    t = res.tables["Funnel by scenario"].set_index("scenario")
    assert t.loc["A", "alert_to_SAR"] == pytest.approx(0.2) and t.loc["B", "case_to_SAR"] == pytest.approx(0.5)
    bad = df.assign(case=0)
    r = _ok("aml.alert_funnel", bad, {"target": "sar", "case": "case"})
    assert any("no case flag" in n for n in r.notes)


def test_scenario_overlap_jaccard_and_unique_sars():
    df = pd.DataFrame({"cust": ["c1", "c2", "c3", "c2", "c3", "c4", "c5"],
                       "rule": ["R1", "R1", "R1", "R2", "R2", "R2", "R3"],
                       "sar": [1, 0, 1, 0, 0, 1, 0]})
    res = _ok("aml.scenario_overlap", df, {"id": "cust", "scenario": "rule", "target": "sar"})
    j = res.tables["Jaccard matrix"].set_index("scenario")
    assert j.loc["R1", "R2"] == pytest.approx(2 / 4) and j.loc["R1", "R3"] == 0 and j.loc["R2", "R2"] == 1
    c = res.tables["Contribution by scenario"].set_index("scenario")
    # SAR entities: c1 (R1 only), c3 (R1 and R2), c4 (R2 only)
    assert c.loc["R1", "sar_entities"] == 2 and c.loc["R1", "unique_sar_entities"] == 1
    assert c.loc["R2", "unique_sar_entities"] == 1 and c.loc["R3", "unique_sar_entities"] == 0
    assert res.summary["sar_entities"] == 3 and res.summary["scenarios_without_unique_sar"] == 1


# ── thresholds ─────────────────────────────────────────────────────────────

def test_btl_bands_and_review():
    df = pd.DataFrame({"amt": [100, 150, 95, 85, 75, 40, np.nan], "id": list("abcdefg")})
    res = _ok("aml.btl_bands", df, {"value": "amt", "threshold": 100, "bands": [10, 20, 30], "id": "id"})
    t = res.tables["Population by band"].set_index("band")
    assert t.loc["ATL", "rows"] == 2 and t.loc["BTL 0%–10%", "rows"] == 1
    assert t.loc["BTL 10%–20%", "rows"] == 1 and t.loc["BTL 20%–30%", "rows"] == 1 and t.loc["beyond", "rows"] == 1
    below = _ok("aml.btl_bands", pd.DataFrame({"x": [5, 10, 10.5, 12]}),
                {"value": "x", "threshold": 10, "direction": "below", "bands": [10, 50]})
    tb = below.tables["Population by band"].set_index("band")
    assert tb.loc["ATL", "rows"] == 2 and tb.loc["BTL 0%–10%", "rows"] == 1 and tb.loc["BTL 10%–50%", "rows"] == 1

    rev = pd.DataFrame({"band": ["b1"] * 59 + ["b2"] * 40, "prod": [0] * 59 + [1] * 4 + [0] * 36})
    res = _ok("aml.btl_review", rev, {"target": "prod", "band": "band", "population_counts": {"b1": 1000, "b2": 500}})
    t = res.tables["Productive rate by band"].set_index("band")
    assert t.loc["b1", "upper_bound_one_sided_95%"] == pytest.approx(1 - 0.05 ** (1 / 59))
    assert t.loc["b1", "upper_bound_one_sided_95%"] < 0.05           # 59 clean items: rate < 5% at 95%
    assert t.loc["b2", "ci_low_95%"] == pytest.approx(stats.beta.ppf(0.025, 4, 37))
    assert t.loc["b2", "expected_productive_in_population"] == pytest.approx(0.1 * 500)
    vals = pd.DataFrame({"amt": [95, 92, 85, 81], "prod": [1, 0, 0, 0]})
    res = _ok("aml.btl_review", vals, {"target": "prod", "value": "amt", "threshold": 100, "bands": [10, 20]})
    assert list(res.tables["Productive rate by band"]["reviewed"]) == [2, 2]


def test_atl_threshold_sweep_hand():
    df = pd.DataFrame({"v": [10, 20, 30, 40, 50, 60], "sar": [0, 0, 1, 0, 1, 1]})
    res = _ok("aml.atl_threshold_sweep", df, {"value": "v", "target": "sar", "thresholds": [60, 40, 20],
                                              "current_threshold": 40})
    t = res.tables["Threshold sweep"].set_index("threshold")
    assert t.loc[60, "alerts"] == 1 and t.loc[60, "sars"] == 1
    assert t.loc[40, "alerts"] == 3 and t.loc[40, "sars"] == 2 and t.loc[40, "marginal_sar_yield"] == pytest.approx(0.5)
    assert t.loc[20, "alerts"] == 5 and t.loc[20, "marginal_sar_yield"] == pytest.approx(0.5)
    assert t.loc[20, "sars_vs_current"] == 1 and res.summary["current_sar_yield"] == pytest.approx(2 / 3)
    auto = _ok("aml.atl_threshold_sweep", df, {"value": "v", "target": "sar", "n_thresholds": 3})
    assert auto.summary["thresholds"] == 3


# ── segmentation, completeness, Benford ────────────────────────────────────

def test_segmentation_quality():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"seg": np.repeat(["low", "mid", "high"], 200),
                       "vol": np.r_[rng.normal(1, 1, 200), rng.normal(3, 1, 200), rng.normal(6, 1, 200)],
                       "cnt": rng.poisson(5, 600).astype(float),
                       "q": np.tile(["2024Q1", "2024Q2"], 300)})
    res = _ok("aml.segmentation_quality", df, {"segment": "seg", "features": ["vol", "cnt"], "period": "q"})
    t = res.tables["Between-segment tests"].set_index("variable")
    H = stats.kruskal(*[df.loc[df["seg"] == s, "vol"] for s in ["high", "low", "mid"]]).statistic
    assert t.loc["vol", "kruskal_wallis_H"] == pytest.approx(H)
    assert t.loc["vol", "p_value"] < 1e-50 and t.loc["cnt", "p_value"] > 0.001
    assert t.loc["vol", "eta_squared"] > 0.8 and res.summary["silhouette"] > 0
    assert res.summary["max_segment_mix_PSI"] == pytest.approx(0, abs=1e-9)
    small = _ok("aml.segmentation_quality", df, {"segment": "seg", "features": ["vol"], "max_silhouette_rows": 100})
    again = _ok("aml.segmentation_quality", df, {"segment": "seg", "features": ["vol"], "max_silhouette_rows": 100})
    assert small.summary["silhouette"] == again.summary["silhouette"]
    assert run_test("aml.segmentation_quality", _ctx(df.assign(seg="one")),
                    {"segment": "seg", "features": ["vol"]}).status == "not_applicable"


def test_data_completeness_hand():
    df = pd.DataFrame({"cpty": ["A", None, "UNKNOWN", "B", " "], "ctry": ["DE", "FR", "XX", "IT", "ES"],
                       "amt": [10, -5, 0, "abc", 3], "m": ["1", "1", "2", "2", "2"]})
    res = _ok("aml.data_completeness", df, {"columns": ["cpty", "ctry", "amt"], "amount": "amt", "period": "m"})
    t = res.tables["Completeness by field"].set_index("field")
    assert t.loc["cpty", "missing_total"] == 3 and t.loc["cpty", "null"] == 1 and t.loc["cpty", "placeholder"] == 2
    assert t.loc["ctry", "missing_total"] == 1
    assert t.loc["amt", "zero_or_negative"] == 2 and t.loc["amt", "non_numeric"] == 1
    assert res.summary["worst_field"] in {"cpty", "amt"} and res.summary["worst_missing_share"] == pytest.approx(0.6)
    bym = res.tables["Missing share by period"].set_index("period")
    assert bym.loc["1", "cpty"] == pytest.approx(0.5)


def test_leading_digits():
    x = np.array([1000.0, 999.99, 20.0, 99.0, 0.0123, 1e6, 3.0e-7])
    assert list(_leading_digits(x, False)) == [1, 9, 2, 9, 1, 1, 3]
    assert list(_leading_digits(x, True)) == [10, 99, 20, 99, 12, 10, 30]


def test_benford_conforming_vs_uniform_digits():
    rng = np.random.default_rng(42)
    benf = pd.DataFrame({"amt": 10 ** rng.uniform(1, 6, 20000)})      # log-uniform → exactly Benford
    r = _ok("aml.benford", benf, {"amount": "amt"})
    assert r.summary["p_value"] > 0.01 and r.summary["MAD"] < 0.006
    assert r.summary["nigrini_mad_range"] == "close conformity"
    uni = pd.DataFrame({"amt": rng.integers(1, 10, 20000) * 10.0 ** rng.integers(1, 5, 20000)})
    u = _ok("aml.benford", uni, {"amount": "amt"})
    assert u.summary["p_value"] < 1e-100 and u.summary["nigrini_mad_range"] == "nonconformity"
    exp = np.log10(1 + 1 / np.arange(1, 10))
    obs = u.tables["Digit distribution"]["observed_count"].to_numpy()
    assert u.summary["chi2"] == pytest.approx(stats.chisquare(obs, exp * obs.sum()).statistic)
    two = _ok("aml.benford", benf, {"amount": "amt", "digits": "first_two"})
    assert two.summary["df"] == 89 and two.summary["p_value"] > 0.001
    assert run_test("aml.benford", _ctx(pd.DataFrame({"amt": [1.0, 2.0]})), {"amount": "amt"}).status == \
        "not_applicable"


# ── screening, ageing ──────────────────────────────────────────────────────

def test_screening_effectiveness_hand():
    df = pd.DataFrame({"exp": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                       "act": [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                       "var": ["typo", "typo", "translit", "translit", None, None, None, None, None, None],
                       "score": [0.95, 0.9, 0.85, 0.6, 0.88, 0.2, 0.1, 0.3, 0.4, 0.5]})
    res = _ok("aml.screening_effectiveness", df, {"expected_hit": "exp", "actual_hit": "act",
                                                  "variation": "var", "score": "score"})
    assert res.summary["detection_rate"] == pytest.approx(0.75)
    assert res.summary["false_positive_rate"] == pytest.approx(1 / 6)
    assert res.summary["precision"] == pytest.approx(0.75)
    lo, hi = clopper_pearson(3, 4)
    assert res.summary["detection_ci_low"] == pytest.approx(lo)
    v = res.tables["By variation type"].set_index("variation")
    assert v.loc["typo", "detection_rate"] == 1 and v.loc["translit", "detection_rate"] == 0.5
    assert res.tables["By score threshold"]["detection_rate"].iloc[0] == 1


def test_alert_aging_hand():
    df = pd.DataFrame({"created": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-03-01"],
                       "closed": ["2024-01-11", None, "2024-03-31", None],
                       "team": ["x", "x", "y", "y"]})
    res = _ok("aml.alert_aging", df, {"date": "created", "closed_date": "closed", "as_of": "2024-03-31",
                                      "segment": "team"})
    assert res.summary["open_alerts"] == 2
    assert res.summary["open_max_age"] == 76 and res.summary["open_median_age"] == pytest.approx((76 + 30) / 2)
    assert res.summary["closed_alerts"] == 2 and res.summary["mean_days_to_close"] == pytest.approx((10 + 59) / 2)
    m = res.tables["Monthly backlog"].set_index("month")
    assert m.loc["2024-01", "open_at_month_end"] == 1 and m.loc["2024-02", "open_at_month_end"] == 2
    assert m.loc["2024-03", "open_at_month_end"] == 2 and m.loc["2024-03", "closed"] == 1
    b = res.tables["Open alerts by age bucket"].set_index("age_days")
    assert b.loc["0–30", "open_alerts"] == 1 and b.loc["60–90", "open_alerts"] == 1


# ── customer risk rating ───────────────────────────────────────────────────

ORDER = ["low", "medium", "high"]


def test_crr_distribution_and_migration():
    df = pd.DataFrame({"cust": list("abcdef") * 2, "q": ["q1"] * 6 + ["q2"] * 6,
                       "crr": ["low", "low", "medium", "medium", "high", "low",
                               "low", "medium", "medium", "high", "high", "high"]})
    res = _ok("aml.crr_distribution", df, {"grade": "crr", "period": "q", "order": ORDER})
    d = res.tables["Distribution"]
    assert list(d["risk_class"]) == ORDER and list(d["customers"]) == [4, 4, 4]
    assert "homogeneity_p" in res.summary
    mig = _ok("aml.crr_migration", df, {"id": "cust", "period": "q", "grade": "crr", "order": ORDER})
    m = mig.tables["Migration counts"].set_index("from \\ to")
    assert m.loc["low", "medium"] == 1 and m.loc["low", "high"] == 1 and m.loc["medium", "high"] == 1
    assert mig.summary["share_stable"] == pytest.approx(3 / 6)
    assert mig.summary["share_to_higher_risk"] == pytest.approx(3 / 6)
    assert mig.summary["share_moved_more_than_one"] == pytest.approx(1 / 6)
    bad = run_test("aml.crr_distribution", _ctx(df), {"grade": "crr", "order": ["low", "high"]})
    assert bad.status == "error"


def test_crr_sar_concordance_known_values():
    rng = np.random.default_rng(1)
    cls = rng.choice(ORDER, 3000, p=[0.6, 0.3, 0.1])
    p = pd.Series(cls).map({"low": 0.01, "medium": 0.04, "high": 0.15}).to_numpy()
    df = pd.DataFrame({"crr": cls, "sar": (rng.random(3000) < p).astype(int)})
    res = _ok("aml.crr_sar_concordance", df, {"grade": "crr", "target": "sar", "order": ORDER})
    assert res.summary["monotone_non_decreasing"] and res.summary["cochran_armitage_p"] < 1e-10
    rank = df["crr"].map({c: i for i, c in enumerate(ORDER)}).to_numpy()
    r = np.corrcoef(rank, df["sar"])[0, 1]
    assert res.summary["cochran_armitage_z"] ** 2 == pytest.approx(len(df) * r ** 2)   # Z² = N·r²
    from sklearn.metrics import roc_auc_score
    assert res.summary["AUC"] == pytest.approx(roc_auc_score(df["sar"], rank))
    z, _ = cochran_armitage(np.array([0, 0]), np.array([5, 5]))
    assert math.isnan(z)
    assert run_test("aml.crr_sar_concordance", _ctx(df.assign(sar=0)),
                    {"grade": "crr", "target": "sar"}).status == "not_applicable"


def test_determinism_and_nan_handling():
    df = pd.DataFrame({"alert": [1, 0, None, 1, 0], "seg": list("ababa")})
    a = _ok("aml.alert_rate", df, {"alert": "alert", "segment": "seg"})
    b = _ok("aml.alert_rate", df.copy(), {"alert": "alert", "segment": "seg"})
    assert a.run_id == b.run_id and a.summary == b.summary
    assert a.rows_used == 4 and any("1 rows" in n for n in a.notes)
