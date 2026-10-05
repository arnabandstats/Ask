"""PD / rating-system validation tests (ask/validation/t_pd.py): known answers and edge cases."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ask.validation import core
from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="portfolio", **kw)


def _ok(test_id, df, params, **kw):
    res = run_test(test_id, _ctx(df, **kw), params)
    assert res.status == "ok", f"{test_id}: {res.error}"
    return res


@pytest.fixture(scope="module")
def port():
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-3 + 1.1 * x)))
    y = rng.binomial(1, p)
    grade = pd.cut(p, [0, .01, .02, .04, .08, .16, 1], labels=False) + 1
    return pd.DataFrame({
        "y": y, "pd": p, "score": x, "bench": x + rng.normal(size=n), "grade": grade,
        "final": np.clip(grade + rng.choice([-1, 0, 0, 0, 1], n), 1, 6),
        "seg": rng.choice(["A", "B"], n), "year": rng.choice([2019, 2020, 2021, 2022, 2023], n),
        "s": rng.choice(["dev", "oot"], n), "ead": rng.uniform(1, 100, n)})


# ── discrimination ─────────────────────────────────────────────────────────

def test_auc_matches_sklearn_and_hand_delong_variance():
    pos, neg = [0.8, 0.6, 0.4], [0.5, 0.3, 0.2, 0.6]
    df = pd.DataFrame({"y": [1] * 3 + [0] * 4, "s": pos + neg})
    r = _ok("pd.auc", df, {"target": "y", "score": "s"})
    assert r.summary["AUC"] == pytest.approx(roc_auc_score(df["y"], df["s"]))
    assert r.summary["AUC"] == pytest.approx(2.375 / 3)
    # hand placement values (ties count 1/2): V10 = (1, 3.5/4, 2/4), V01 = (2/3, 1, 1, 1.5/3)
    v10, v01 = np.array([1, 0.875, 0.5]), np.array([2 / 3, 1, 1, 0.5])
    var = v10.var(ddof=1) / 3 + v01.var(ddof=1) / 4
    assert var == pytest.approx(0.0381944444, abs=1e-9)
    assert r.summary["AUC_SE_DeLong"] == pytest.approx(math.sqrt(var))
    assert r.summary["Gini"] == pytest.approx(2 * 2.375 / 3 - 1)


def test_auc_orientation_and_hanley_mcneil(port):
    a = _ok("pd.auc", port, {"target": "y", "score": "score"})
    b = _ok("pd.auc", port.assign(score=-port["score"]),
            {"target": "y", "score": "score", "higher_is_riskier": False})
    assert a.summary["AUC"] == pytest.approx(b.summary["AUC"])
    A, m = a.summary["AUC"], a.summary["defaults"]
    n = len(port) - m
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    hm = math.sqrt((A * (1 - A) + (m - 1) * (q1 - A * A) + (n - 1) * (q2 - A * A)) / (m * n))
    assert a.summary["AUC_SE_Hanley_McNeil"] == pytest.approx(hm)
    assert a.summary["AUC_CI_lower"] < A < a.summary["AUC_CI_upper"]
    assert len(a.figures) == 1


def test_auc_single_class_and_nan_handling():
    df = pd.DataFrame({"y": [0, 0, 0, 0], "s": [1, 2, 3, 4]})
    assert run_test("pd.auc", _ctx(df), {"target": "y", "score": "s"}).status == "not_applicable"
    df = pd.DataFrame({"y": [1, 1, 0, 0, 0, None], "s": [5, 4, 3, 2, np.nan, 1]})
    r = _ok("pd.auc", df, {"target": "y", "score": "s"})
    assert r.rows_used == 4 and any("2 rows" in n for n in r.notes)
    assert r.summary["AUC"] == 1.0


def test_cap_ar_equals_gini_and_somers_d(port):
    rd = port.assign(score=port["score"].round(1))      # heavy ties
    cap = _ok("pd.cap_accuracy_ratio", rd, {"target": "y", "score": "score"})
    auc = roc_auc_score(rd["y"], rd["score"])
    assert cap.summary["accuracy_ratio"] == pytest.approx(2 * auc - 1, abs=1e-12)
    rc = _ok("pd.rank_correlation", rd, {"target": "y", "score": "score"})
    assert rc.summary["somers_d_score_given_default"] == pytest.approx(2 * auc - 1)
    assert rc.summary["kendall_tau_b"] == pytest.approx(stats.kendalltau(rd["score"], rd["y"]).statistic)


def test_ks_matches_scipy(port):
    r = _ok("pd.ks", port, {"target": "y", "score": "score"})
    ref = stats.ks_2samp(port.loc[port.y == 1, "score"], port.loc[port.y == 0, "score"])
    assert r.summary["KS"] == pytest.approx(ref.statistic)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue)
    loc = r.summary["threshold"]
    f_bad = (port.loc[port.y == 1, "score"] <= loc).mean()
    f_good = (port.loc[port.y == 0, "score"] <= loc).mean()
    assert abs(f_good - f_bad) == pytest.approx(r.summary["KS"])


def test_information_value_hand():
    # band a: 10 defaults / 90 non-defaults; band b: 30 / 70
    df = pd.DataFrame({"y": [1] * 10 + [0] * 90 + [1] * 30 + [0] * 70, "g": ["a"] * 100 + ["b"] * 100})
    r = _ok("pd.information_value", df, {"target": "y", "grade": "g"})
    gd = np.array([90 / 160, 70 / 160])
    bd = np.array([10 / 40, 30 / 40])
    assert r.summary["IV"] == pytest.approx(float(np.sum((gd - bd) * np.log(gd / bd))))
    r2 = _ok("pd.information_value", df.assign(sc=np.arange(200)), {"target": "y", "score": "sc", "bins": 4})
    assert r2.summary["bands"] == 4


def test_divergence_and_pr_auc(port):
    r = _ok("pd.divergence", port, {"target": "y", "score": "score"})
    b, g = port.loc[port.y == 1, "score"], port.loc[port.y == 0, "score"]
    assert r.summary["divergence"] == pytest.approx((g.mean() - b.mean()) ** 2 / (0.5 * (b.var() + g.var())))
    pr = _ok("pd.pr_auc", port, {"target": "y", "score": "score"})
    assert pr.summary["average_precision"] == pytest.approx(average_precision_score(port.y, port.score))


def test_delong_compare(port):
    r = _ok("pd.delong_compare", port, {"target": "y", "score": "score", "benchmark": "bench"})
    assert r.summary["AUC_model"] == pytest.approx(roc_auc_score(port.y, port.score))
    assert r.summary["AUC_benchmark"] == pytest.approx(roc_auc_score(port.y, port.bench))
    single = _ok("pd.auc", port, {"target": "y", "score": "score"})
    assert r.tables["AUCs"]["SE"].iloc[0] == pytest.approx(single.summary["AUC_SE_DeLong"])
    assert r.summary["p_value_two_sided"] < 1e-6
    same = _ok("pd.delong_compare", port, {"target": "y", "score": "score", "benchmark": "score"})
    assert same.summary["AUC_difference"] == 0 and same.summary["SE_difference"] == pytest.approx(0, abs=1e-12)


def test_auc_by_segment_and_period(port):
    r = _ok("pd.auc_by_segment", port, {"target": "y", "score": "score", "segment": "seg"})
    t = r.tables["AUC by segment"].set_index("segment")
    sub = port[port.seg == "A"]
    assert t.loc["A", "AUC"] == pytest.approx(roc_auc_score(sub.y, sub.score))
    assert 0 <= r.summary["heterogeneity_p_value"] <= 1
    r = _ok("pd.auc_by_period", port, {"target": "y", "score": "score", "period": "year"})
    assert r.summary["periods"] == 5


def test_auc_change_ecb_and_independent(port):
    cur = _ok("pd.auc", port, {"target": "y", "score": "score"})
    r = _ok("pd.auc_change", port, {"target": "y", "score": "score", "reference_auc": 0.85})
    S = (0.85 - cur.summary["AUC"]) / cur.summary["AUC_SE_DeLong"]
    assert r.summary["statistic"] == pytest.approx(S)
    assert r.summary["p_value_deterioration"] == pytest.approx(1 - stats.norm.cdf(S))
    r2 = _ok("pd.auc_change", port, {"target": "y", "score": "score", "sample": "s"})
    t = r2.tables["AUC by sample"]
    assert r2.summary["statistic"] == pytest.approx((t.AUC[0] - t.AUC[1]) / math.sqrt(t.variance.sum()))
    r3 = _ok("pd.auc_change", port, {"target": "y", "score": "score", "sample": "s", "period": "year",
                                      "se_method": "hanley_mcneil"})
    assert r3.summary["se_method"] == "hanley_mcneil"
    # the within-period CDF transform leaves single-period AUC unchanged
    one = port[port.year == 2019]
    r4 = _ok("pd.auc_change", one, {"target": "y", "score": "score", "reference_auc": 0.8, "period": "year"})
    assert r4.summary["AUC_current"] == pytest.approx(roc_auc_score(one.y, one.score))


def test_gini_bootstrap_deterministic(port):
    a = _ok("pd.gini_bootstrap", port, {"target": "y", "score": "score", "n_boot": 300})
    b = _ok("pd.gini_bootstrap", port.copy(), {"target": "y", "score": "score", "n_boot": 300})
    assert a.summary == b.summary
    assert a.summary["percentile_CI_lower"] < a.summary["Gini"] < a.summary["percentile_CI_upper"]
    delong_se = 2 * _ok("pd.auc", port, {"target": "y", "score": "score"}).summary["AUC_SE_DeLong"]
    assert a.summary["bootstrap_SE"] == pytest.approx(delong_se, rel=0.3)
    c = run_test("pd.gini_bootstrap", _ctx(port, seed=1), {"target": "y", "score": "score", "n_boot": 300})
    assert c.summary["percentile_CI_lower"] != a.summary["percentile_CI_lower"]


def test_rank_ordering_inversions():
    # DRs by grade 1..4: 0.10, 0.05, 0.20, 0.15 -> adjacent inversions 2, pairwise inversions 2
    rows = []
    for g, d in zip([1, 2, 3, 4], [10, 5, 20, 15]):
        rows += [{"g": g, "y": 1}] * d + [{"g": g, "y": 0}] * (100 - d)
    df = pd.DataFrame(rows)
    r = _ok("pd.rank_ordering", df, {"target": "y", "grade": "g"})
    assert r.summary["adjacent_inversions"] == 2 and r.summary["pairwise_inversions"] == 2
    assert r.summary["spearman_rho"] == pytest.approx(stats.spearmanr([0, 1, 2, 3], [.1, .05, .2, .15]).statistic)
    r2 = _ok("pd.rank_ordering", df, {"target": "y", "grade": "g", "grade_order": ["2", "1", "4", "3"]})
    assert r2.summary["adjacent_inversions"] == 0 and r2.summary["spearman_rho"] == pytest.approx(1)
    bad = run_test("pd.rank_ordering", _ctx(df), {"target": "y", "grade": "g", "grade_order": ["1", "2"]})
    assert bad.status == "error" and "grade_order" in bad.error


# ── calibration ────────────────────────────────────────────────────────────

def _grade_df():
    # grade A: N=200, D=3, PD 0.01; grade B: N=100, D=9, PD 0.05
    return pd.DataFrame({"g": ["A"] * 200 + ["B"] * 100,
                         "y": [1] * 3 + [0] * 197 + [1] * 9 + [0] * 91,
                         "pd": [0.01] * 200 + [0.05] * 100})


def test_binomial_and_normal_tests_hand():
    r = _ok("pd.binomial_test", _grade_df(), {"target": "y", "pd": "pd", "grade": "g"})
    t = r.tables["Binomial test by grade"].set_index("grade")
    assert t.loc["B", "p_underestimation"] == pytest.approx(stats.binom.sf(8, 100, 0.05))
    assert t.loc["B", "p_overestimation"] == pytest.approx(stats.binom.cdf(9, 100, 0.05))
    assert t.loc["A", "p_two_sided"] == pytest.approx(stats.binomtest(3, 200, 0.01).pvalue)
    assert t.loc["Portfolio", "PD"] == pytest.approx((200 * .01 + 100 * .05) / 300)
    nt = _ok("pd.normal_test", _grade_df(), {"target": "y", "pd": "pd", "grade": "g"})
    z = (9 - 5) / math.sqrt(100 * .05 * .95)
    assert nt.tables["Normal test by grade"].set_index("grade").loc["B", "z"] == pytest.approx(z)


def test_jeffreys_matches_beta_cdf():
    r = _ok("pd.jeffreys_test", _grade_df().assign(e=1.0), {"target": "y", "pd": "pd", "grade": "g", "weight": "e"})
    t = r.tables["Jeffreys test by grade"].set_index("grade")
    assert t.loc["A", "p_value"] == pytest.approx(stats.beta.cdf(0.01, 3.5, 197.5))
    assert t.loc["B", "p_value"] == pytest.approx(stats.beta.cdf(0.05, 9.5, 91.5))
    pd_port = 7 / 300
    assert r.summary["p_value"] == pytest.approx(stats.beta.cdf(pd_port, 12.5, 288.5))
    assert t.loc["B", "exposure"] == 100


def test_vasicek_reduces_to_binomial_and_matches_simulation():
    df = _grade_df()
    r0 = _ok("pd.vasicek_test", df, {"target": "y", "pd": "pd", "grade": "g", "rho": 0.0})
    t0 = r0.tables["Vasicek test by grade"]
    assert np.allclose(t0["p_value_finite_N"], t0["p_value_binomial_independent"], rtol=1e-8)
    r = _ok("pd.vasicek_test", df, {"target": "y", "pd": "pd", "grade": "g", "rho": 0.15})
    t = r.tables["Vasicek test by grade"].set_index("grade")
    # correlation widens the default distribution: a high default count becomes less surprising
    assert t.loc["B", "p_value_finite_N"] > t.loc["B", "p_value_binomial_independent"]
    # asymptotic p-value vs Monte Carlo of the Vasicek default-rate distribution
    z = np.random.default_rng(0).standard_normal(400_000)
    dr = stats.norm.cdf((stats.norm.ppf(0.05) + math.sqrt(0.15) * z) / math.sqrt(0.85))
    assert t.loc["B", "p_value_asymptotic"] == pytest.approx((dr >= 0.09).mean(), abs=3e-3)
    # finite-N p-value vs Monte Carlo of the mixed binomial
    rng = np.random.default_rng(1)
    pz = stats.norm.cdf((stats.norm.ppf(0.05) + math.sqrt(0.15) * rng.standard_normal(200_000)) / math.sqrt(0.85))
    assert t.loc["B", "p_value_finite_N"] == pytest.approx((rng.binomial(100, pz) >= 9).mean(), abs=4e-3)
    assert any("rho = 0.15" in n for n in r.notes)


def test_hosmer_lemeshow_hand_and_chi_square_equivalence():
    df = _grade_df()
    r = _ok("pd.hosmer_lemeshow", df, {"target": "y", "pd": "pd", "grade": "g"})
    hl = (3 - 2) ** 2 / (200 * .01 * .99) + (9 - 5) ** 2 / (100 * .05 * .95)
    assert r.summary["HL"] == pytest.approx(hl)
    assert r.summary["dof"] == 2 and r.summary["p_value"] == pytest.approx(stats.chi2.sf(hl, 2))
    r2 = _ok("pd.hosmer_lemeshow", df, {"target": "y", "pd": "pd", "grade": "g", "dof": "g-2"})
    assert r2.summary["dof"] == 0 and math.isnan(r2.summary["p_value"])
    c = _ok("pd.chi_square_grades", df, {"target": "y", "pd": "pd", "grade": "g"})
    assert c.summary["chi2"] == pytest.approx(hl) and c.summary["dof"] == 2


def test_hosmer_lemeshow_deciles(port):
    r = _ok("pd.hosmer_lemeshow", port, {"target": "y", "pd": "pd"})
    assert r.summary["groups"] == 10
    t = r.tables["HL groups"]
    assert t["N"].sum() == len(port)


def test_spiegelhalter_hand():
    y, p = np.array([1, 0, 0, 1, 0]), np.array([0.8, 0.1, 0.3, 0.4, 0.2])
    r = _ok("pd.spiegelhalter", pd.DataFrame({"y": y, "p": p}), {"target": "y", "pd": "p"})
    z = np.sum((y - p) * (1 - 2 * p)) / math.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    assert r.summary["z"] == pytest.approx(z)


def test_brier_decomposition_exact_for_discrete_pds():
    df = _grade_df()
    r = _ok("pd.brier", df, {"target": "y", "pd": "pd"})
    s = r.summary
    assert s["brier"] == pytest.approx(brier_score_loss(df.y, df.pd))
    assert s["within_group_residual"] == pytest.approx(0, abs=1e-12)
    ob = 12 / 300
    assert s["uncertainty"] == pytest.approx(ob * (1 - ob))
    rel = (200 * (0.01 - 0.015) ** 2 + 100 * (0.05 - 0.09) ** 2) / 300
    assert s["reliability"] == pytest.approx(rel)
    assert s["brier_skill_score"] == pytest.approx(1 - s["brier"] / s["uncertainty"])


def test_calibration_in_the_large_hand():
    df = _grade_df().assign(w=[2.0] * 200 + [1.0] * 100)
    r = _ok("pd.calibration_in_the_large", df, {"target": "y", "pd": "pd", "weight": "w"})
    e, v = 7.0, 200 * .01 * .99 + 100 * .05 * .95
    assert r.summary["O_over_E"] == pytest.approx(12 / 7)
    assert r.summary["z"] == pytest.approx((12 - e) / math.sqrt(v))
    assert r.summary["exposure_weighted_DR"] == pytest.approx((3 * 2 + 9) / 500)


def test_calibration_slope_recovers_known_miscalibration():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.4, 40_000)
    lp = np.log(p / (1 - p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 0.7 * lp))))
    r = _ok("pd.calibration_slope", pd.DataFrame({"y": y, "p": p}), {"target": "y", "pd": "p"})
    assert r.summary["intercept"] == pytest.approx(0.3, abs=0.08)
    assert r.summary["slope"] == pytest.approx(0.7, abs=0.05)
    assert r.summary["p_value_slope_1"] < 1e-6 and r.summary["p_value_joint"] < 1e-6


def test_ece_hand():
    df = pd.DataFrame({"y": [0, 0, 0, 1, 1, 1, 1, 0], "p": [0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9]})
    r = _ok("pd.ece", df, {"target": "y", "pd": "p", "bins": 2, "strategy": "uniform"})
    # bin [0, .5): DR 0.25, mean PD 0.15; bin [.5, 1): DR 0.75, mean PD 0.85
    assert r.summary["ECE"] == pytest.approx(0.5 * 0.1 + 0.5 * 0.1)
    assert r.summary["MCE"] == pytest.approx(0.1)


def test_multi_period_and_long_run_hand():
    rows = []
    for yr, n, d, p in [(2020, 100, 3, .02), (2021, 100, 1, .02), (2022, 200, 6, .025)]:
        rows += [{"yr": yr, "y": 1, "p": p}] * d + [{"yr": yr, "y": 0, "p": p}] * (n - d)
    df = pd.DataFrame(rows)
    dev = np.array([.01, -.01, .005])
    r = _ok("pd.multi_period_normal_test", df, {"target": "y", "pd": "p", "period": "yr"})
    S = dev.sum() / (math.sqrt(3) * dev.std(ddof=1))
    assert r.summary["statistic"] == pytest.approx(S)
    assert r.summary["p_value"] == pytest.approx(stats.norm.sf(S))
    assert r.summary["periods_DR_above_PD"] == 2
    lr = _ok("pd.long_run_default_rate", df, {"target": "y", "pd": "p", "period": "yr"})
    assert lr.summary["long_run_average_DR"] == pytest.approx((0.03 + 0.01 + 0.03) / 3)
    assert lr.summary["pooled_DR"] == pytest.approx(10 / 400)
    assert lr.summary["average_PD"] == pytest.approx((0.02 + 0.02 + 0.025) / 3)


def test_pd_out_of_range_is_an_error():
    df = pd.DataFrame({"y": [0, 1, 0], "p": [5.0, 20.0, 1.0]})
    r = run_test("pd.jeffreys_test", _ctx(df), {"target": "y", "pd": "p"})
    assert r.status == "error" and "[0, 1]" in r.error


# ── rating system ──────────────────────────────────────────────────────────

def test_grade_concentration_hhi_and_ecb_test():
    df = pd.DataFrame({"g": [1] * 50 + [2] * 25 + [3] * 25, "e": [1.0] * 50 + [2.0] * 25 + [4.0] * 25})
    r = _ok("pd.grade_concentration", df, {"grade": "g", "weight": "e", "reference_cv": 0.2})
    s = r.summary
    assert s["HHI"] == pytest.approx(0.375)
    cv = math.sqrt(3 * ((0.5 - 1 / 3) ** 2 + 2 * (0.25 - 1 / 3) ** 2))
    assert s["CV"] == pytest.approx(cv)
    assert s["HI_ECB"] == pytest.approx(1 + math.log((cv ** 2 + 1) / 3) / math.log(3))
    z = math.sqrt(2) * (cv - 0.2) / math.sqrt(cv ** 2 * (0.5 + cv ** 2))
    assert s["p_value"] == pytest.approx(1 - stats.norm.cdf(z))
    ex = np.array([50, 50, 100]) / 200
    assert s["HHI_exposure"] == pytest.approx(np.sum(ex ** 2))
    # K includes an empty grade listed in grade_order
    r2 = _ok("pd.grade_concentration", df, {"grade": "g", "grade_order": [1, 2, 3, 4]})
    assert r2.summary["grades_K"] == 4


def test_migration_matrix_mwb_hand():
    counts = [[10, 5, 1], [2, 20, 4], [0, 3, 15]]
    rows = [(i + 1, j + 1) for i in range(3) for j in range(3) for _ in range(counts[i][j])]
    df = pd.DataFrame(rows, columns=["g0", "g1"])
    r = _ok("pd.migration_matrix", df, {"grade": "g0", "grade_to": "g1"})
    s = r.summary
    # upper: num = 1*5 + 2*1 + 1*4 = 11; M_up = 2*(5+1) + 1*4 = 16
    assert s["MWB_upper_downgrades"] == pytest.approx(11 / 16)
    # lower: num = 1*2 + 2*0 + 1*3 = 5; M_low = 1*2 + 2*(0+3) = 8
    assert s["MWB_lower_upgrades"] == pytest.approx(5 / 8)
    assert s["share_diagonal"] == pytest.approx(45 / 60)
    assert s["share_within_1_notch"] == pytest.approx(59 / 60)
    tr = 10 / 16 + 20 / 26 + 15 / 18
    assert s["mobility_shorrocks"] == pytest.approx((3 - tr) / 2)
    P = np.array(counts, float) / np.array(counts).sum(axis=1, keepdims=True)
    assert s["mobility_svd"] == pytest.approx(np.linalg.svd(P - np.eye(3), compute_uv=False).mean())
    z = r.tables["ECB z-tests"].set_index(["from_grade", "to_grade"])
    p12, p11 = 5 / 16, 10 / 16
    exp_z = (p11 - p12) / math.sqrt((p12 * (1 - p12) + p11 * (1 - p11) + 2 * p12 * p11) / 16)
    assert z.loc[("1", "2"), "z"] == pytest.approx(exp_z)


def test_migration_matrix_from_panel():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 1, 2, 3, 5], "t": ["2023"] * 4 + ["2024"] * 4,
                       "g": ["A", "B", "B", "C", "B", "B", "A", "C"]})
    r = _ok("pd.migration_matrix", df, {"grade": "g", "id": "id", "period": "t", "grade_order": ["A", "B", "C"]})
    c = r.tables["Migration counts"].set_index("from_grade")
    assert c.loc["A", "B"] == 1 and c.loc["B", "A"] == 1 and c.loc["C", "<not observed at end>"] == 1
    assert r.summary["obligors"] == 4


def test_overrides_hand():
    df = pd.DataFrame({"m": [1, 1, 2, 2, 3, 3, 3, 2], "f": [1, 2, 2, 4, 3, 1, 3, 2],
                       "y": [0, 0, 1, 1, 0, 1, 0, 0]})
    r = _ok("pd.overrides", df, {"grade": "m", "final_grade": "f", "target": "y"})
    s = r.summary
    assert s["overrides"] == 3 and s["downgrades_worse"] == 2 and s["upgrades_better"] == 1
    assert s["mean_abs_notches_when_overridden"] == pytest.approx((1 + 2 + 2) / 3)
    assert s["sign_test_p_value"] == pytest.approx(stats.binomtest(2, 3, 0.5).pvalue)
    assert s["DR_overridden"] == pytest.approx(2 / 3)


def test_adjacent_grade_dr_test_hand():
    rows = [{"g": 1, "y": 1}] * 5 + [{"g": 1, "y": 0}] * 95 + [{"g": 2, "y": 1}] * 15 + [{"g": 2, "y": 0}] * 85
    r = _ok("pd.adjacent_grade_dr_test", pd.DataFrame(rows), {"target": "y", "grade": "g"})
    t = r.tables["Adjacent grade tests"].iloc[0]
    pp = 20 / 200
    z = (0.15 - 0.05) / math.sqrt(pp * (1 - pp) * (2 / 100))
    assert t["z"] == pytest.approx(z) and t["p_value_z"] == pytest.approx(stats.norm.sf(z))
    assert t["p_value_fisher"] == pytest.approx(
        stats.fisher_exact([[15, 85], [5, 95]], alternative="greater").pvalue)


def test_grade_homogeneity_matches_scipy(port):
    r = _ok("pd.grade_homogeneity", port, {"target": "y", "grade": "grade", "segment": "seg"})
    t = r.tables["Homogeneity by grade"].set_index("grade")
    sub = port[port.grade == 6]
    chi = stats.chi2_contingency(pd.crosstab(sub.seg, sub.y).to_numpy(), correction=False)
    assert t.loc["6", "chi2"] == pytest.approx(chi.statistic)


def test_default_rate_series(port):
    r = _ok("pd.default_rate_series", port, {"target": "y", "period": "year", "pd": "pd", "weight": "ead"})
    t = r.tables["Default rate by period"].set_index("period")
    sub = port[port.year == 2020]
    ci = stats.binomtest(int(sub.y.sum()), len(sub)).proportion_ci(0.95, method="exact")
    assert t.loc["2020", "DR_CI_lower"] == pytest.approx(ci.low)
    assert t.loc["2020", "mean_PD"] == pytest.approx(sub.pd.mean())
    assert t.loc["2020", "exposure_weighted_DR"] == pytest.approx((sub.ead * sub.y).sum() / sub.ead.sum())


# ── default definition ─────────────────────────────────────────────────────

def test_default_definition_replication_hand():
    dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    a = pd.DataFrame({"id": "A", "d": dates, "dpd": [0] * 5 + [95] + [0] * 6, "amt": [0] * 5 + [500] + [0] * 6})
    b = pd.DataFrame({"id": "B", "d": dates, "dpd": [0] * 3 + [120] + [0] * 8, "amt": [0] * 3 + [50] + [0] * 8})
    df = pd.concat([a, b], ignore_index=True)
    df["ead"] = 10_000.0
    # A: trigger in June -> forward 12m flag = 1 on Jan–May. B's breach is immaterial (50 <= 100).
    df["flag"] = 0
    df.loc[(df.id == "A") & (df.d < "2022-06-01"), "flag"] = 1
    df.loc[(df.id == "B") & (df.d == dates[0]), "flag"] = 1           # one deliberate mismatch
    p = {"id": "id", "date": "d", "dpd": "dpd", "target": "flag", "past_due_amount": "amt", "exposure": "ead",
         "abs_threshold": 100, "rel_threshold": 0.01, "window_months": 12}
    r = _ok("pd.default_definition_replication", df, p)
    assert r.summary["mismatches"] == 1 and r.summary["provided_only"] == 1
    assert r.summary["sample_mismatching_ids"] == "B"
    # without materiality B's 120 dpd counts too: B Jan–Mar flagged
    r2 = _ok("pd.default_definition_replication", df, {**p, "abs_threshold": 0, "rel_threshold": 0})
    assert r2.summary["replicated_defaults"] == 5 + 3
    # status at the date itself
    r3 = _ok("pd.default_definition_replication", df, {**p, "window_months": 0})
    assert r3.summary["replicated_defaults"] == 1
    bad = run_test("pd.default_definition_replication", _ctx(df), {**p, "past_due_amount": None})
    assert bad.status == "error"


# ── library-level checks ───────────────────────────────────────────────────

def test_every_pd_test_is_registered_with_references():
    core.load_all()
    ids = [k for k in core.REGISTRY if k.startswith("pd.")]
    assert len(ids) >= 30
    for k in ids:
        spec = core.REGISTRY[k]
        assert spec.references and spec.description, k
        import inspect
        sig = set(inspect.signature(spec.fn).parameters) - {"ctx"}
        assert sig == {p.name for p in spec.params}, k


def test_same_inputs_same_numbers(port):
    a = run_test("pd.hosmer_lemeshow", _ctx(port), {"target": "y", "pd": "pd"})
    b = run_test("pd.hosmer_lemeshow", _ctx(port.copy()), {"target": "y", "pd": "pd"})
    assert a.run_id == b.run_id and a.summary == b.summary
    json_ = a.to_json()
    assert json_["status"] == "ok"
