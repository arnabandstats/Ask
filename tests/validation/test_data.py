"""Data-review tests (ask/validation/t_data.py): known answers and the evidence rows they cite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ask.validation import core
from ask.validation.core import RunContext, run_test
from ask.validation.t_data import _em_mvn, _grubbs_crit


def _ctx(df, **kw):
    return RunContext(df=df, source_name="loans", **kw)


def _ok(test_id, ctx, params):
    res = run_test(test_id, ctx, params)
    assert res.status == "ok", f"{test_id}: {res.error}"
    return res


def _rows(s: str) -> list[int]:
    return [int(v) for v in s.split(" (")[0].split(", ")] if s else []


# ── profile ─────────────────────────────────────────────────────────────────

def test_profile_known_values():
    df = pd.DataFrame({"loan_id": [101, 102, 103, 104, 105],
                       "x": [1.0, 2.0, 0.0, -3.0, np.nan],
                       "grade": ["a", "a", "b", "a", None],
                       "const": ["k"] * 5,
                       "flag": [0, 1, 0, 1, 0],
                       "amount_txt": ["1.5", "2", "3", "4", "5"],
                       "d": ["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30", "2024-05-31"]})
    res = _ok("data.profile", _ctx(df), {})
    t = res.tables["Column profile"].set_index("column")
    assert t.loc["x", "missing"] == 1 and t.loc["x", "missing_pct"] == pytest.approx(0.2)
    assert t.loc["x", "zeros"] == 1 and t.loc["x", "negatives"] == 1
    assert t.loc["x", "min"] == -3 and t.loc["x", "max"] == 2
    assert t.loc["x", "mean"] == pytest.approx(0.0)
    assert t.loc["x", "std"] == pytest.approx(np.std([1, 2, 0, -3], ddof=1))
    assert t.loc["x", "median"] == pytest.approx(0.5)
    assert t.loc["grade", "distinct"] == 2 and t.loc["grade", "mode_share"] == pytest.approx(0.75)
    assert t.loc["grade", "top_values"].startswith("a: 3 (75.0%)")
    assert t.loc["const", "constant"] and t.loc["const", "near_constant"]
    sem = t["semantic_type"]
    assert sem["loan_id"] == "id-like" and sem["flag"] == "boolean" and sem["x"] == "numeric"
    assert sem["amount_txt"] == "numeric-as-text" and sem["d"] == "date" and sem["grade"] == "categorical"
    assert t.loc["amount_txt", "mean"] == pytest.approx(3.1)
    assert res.summary["constant_columns"] == 1 and res.summary["total_missing_cells"] == 2


# ── missingness ─────────────────────────────────────────────────────────────

def test_missingness_counts_evidence_and_homogeneity():
    df = pd.DataFrame({"cid": [f"C{i}" for i in range(8)],
                       "x": [1, np.nan, 3, np.nan, 5, 6, 7, 8],
                       "s": ["a", "b", " ", "d", "", "f", "g", "h"],
                       "per": ["p1"] * 4 + ["p2"] * 4})
    res = _ok("data.missingness", _ctx(df), {"columns": ["x", "s"], "period": "per", "id": "cid"})
    t = res.tables["Missing by column"].set_index("column")
    assert t.loc["x", "missing"] == 2 and t.loc["x", "example_rows"] == "1, 3"
    assert t.loc["x", "example_ids"] == "C1, C3"
    assert t.loc["s", "missing"] == 0 and t.loc["s", "blank_text"] == 2
    ev = res.tables["Evidence"]
    assert ev.loc[ev["check"] == "blank text", "row"].tolist() == [2, 4]
    per = res.tables["Missing by period"]
    assert per.set_index("period").loc["p1", "missing_pct"] == pytest.approx(0.5)
    hom = res.tables["Missing-rate homogeneity tests"].iloc[0]
    chi2, p, _, _ = stats.chi2_contingency(np.array([[2, 2], [0, 4]]), correction=False)
    assert hom["chi2"] == pytest.approx(chi2) and hom["p_value"] == pytest.approx(p)
    assert res.summary["rows_with_any_missing"] == 2


def test_missing_patterns():
    df = pd.DataFrame({"a": [1, np.nan, np.nan, 4, np.nan], "b": [1, 2, np.nan, 4, np.nan], "c": [1] * 5})
    res = _ok("data.missing_patterns", _ctx(df), {"columns": ["a", "b"]})
    t = res.tables["Missingness patterns"].set_index("missing_columns")
    assert t.loc["a, b", "rows"] == 2 and t.loc["a, b", "example_rows"] == "2, 4"
    assert t.loc["a", "rows"] == 1 and t.loc["a", "example_rows"] == "1"
    assert t.loc["<complete>", "rows"] == 2
    assert res.summary["patterns"] == 3
    assert run_test("data.missing_patterns", _ctx(df[["c"]]), {}).status == "not_applicable"


def test_missing_vs_target_matches_fisher():
    # missing rows 0..3: 3 events; present rows 4..11: 1 event
    df = pd.DataFrame({"y": [1, 1, 1, 0] + [1, 0, 0, 0, 0, 0, 0, 0],
                       "x": [np.nan] * 4 + list(range(8))})
    res = _ok("data.missing_vs_target", _ctx(df), {"target": "y"})
    r = res.tables["Missingness vs target"].iloc[0]
    assert r["event_rate_missing"] == pytest.approx(0.75) and r["event_rate_present"] == pytest.approx(0.125)
    assert r["fisher_p_value"] == pytest.approx(stats.fisher_exact([[3, 1], [1, 7]]).pvalue)
    assert r["example_rows"] == "0, 1, 2, 3"


# ── Little's MCAR ───────────────────────────────────────────────────────────

def test_em_matches_closed_form_for_monotone_bivariate():
    rng = np.random.default_rng(3)
    x1 = rng.normal(0, 1, 200)
    x2 = 0.6 * x1 + rng.normal(0, 1, 200)
    x2[150:] = np.nan                        # monotone pattern: x2 missing in the last 50 rows
    mu, S, _ = _em_mvn(np.column_stack([x1, x2]))
    cc = slice(0, 150)
    b = np.cov(x1[cc], x2[cc], ddof=0)[0, 1] / np.var(x1[cc])
    mu2 = x2[cc].mean() + b * (x1.mean() - x1[cc].mean())     # Anderson (1957) ML estimator
    assert mu[0] == pytest.approx(x1.mean()) and mu[1] == pytest.approx(mu2, rel=1e-8)
    assert S[0, 0] == pytest.approx(np.var(x1), rel=1e-8)


def test_littles_mcar_detects_mar_and_not_mcar():
    rng = np.random.default_rng(7)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = 0.7 * x1 + rng.normal(0, 0.7, n)
    mcar = x2.copy()
    mcar[rng.random(n) < 0.3] = np.nan
    mar = x2.copy()
    mar[x1 > 0.3] = np.nan                   # missingness driven by x1
    r1 = _ok("data.littles_mcar", _ctx(pd.DataFrame({"a": x1, "b": mcar})), {"features": ["a", "b"]})
    r2 = _ok("data.littles_mcar", _ctx(pd.DataFrame({"a": x1, "b": mar})), {"features": ["a", "b"]})
    assert r1.summary["dof"] == 1 and r1.summary["p_value"] > 0.01
    assert r2.summary["p_value"] < 1e-10
    # d2 by hand for the 2-pattern case: the complete-pattern and the missing-b pattern
    rows = r2.tables["Patterns"]
    assert rows["d2_contribution"].sum() == pytest.approx(r2.summary["d2"])
    full = pd.DataFrame({"a": x1, "b": x2})
    assert run_test("data.littles_mcar", _ctx(full), {"features": ["a", "b"]}).status == "not_applicable"


# ── duplicates ──────────────────────────────────────────────────────────────

def test_duplicates_full_key_and_near():
    df = pd.DataFrame({"acc": [1, 2, 3, 1, 4, 2, 5],
                       "dt": ["2024-01", "2024-01", "2024-01", "2024-01", "2024-01", "2024-01", "2024-01"],
                       "seg": ["Retail", "SME", "Corp", "Retail", "SME", "SME ", "corp"],
                       "bal": [10.0, 20.0, 30.0, 10.0, 40.0, 25.0, 30.0]})
    df.loc[6, "acc"] = 3                      # row 6 = row 2 up to case: near-duplicate
    res = _ok("data.duplicates", _ctx(df), {"keys": ["acc"], "date": "dt"})
    s = res.summary
    assert s["full_duplicate_groups"] == 1 and s["rows_in_full_duplicates"] == 2
    assert res.tables["Full-row duplicate groups"].iloc[0]["example_rows"] == "0, 3"
    assert s["duplicate_keys"] == 3                     # acc 1, 2 and 3
    k = res.tables["Duplicate keys"].set_index("acc")
    assert k.loc[1, "example_rows"] == "0, 3" and not k.loc[1, "conflicting"]
    assert k.loc[2, "example_rows"] == "1, 5" and k.loc[2, "conflicting"]
    assert s["conflicting_duplicate_keys"] == 2
    assert s["near_duplicate_groups"] == 1
    assert res.tables["Near-duplicate groups"].iloc[0]["example_rows"] == "2, 6"


# ── validity rules and automatic checks ─────────────────────────────────────

def test_validity_rules_violations_and_rows():
    df = pd.DataFrame({"id": ["A", "B", "C", "D", "E"],
                       "ltv": [0.5, 1.7, -0.1, np.nan, 0.9],
                       "grade": ["G1", "G2", "g3", "G9", "G1"],
                       "iban": ["DE12", "DE1X", "FR99", None, "DE00"]})
    rules = {"ltv": {"min": 0, "max": 1.5, "not_null": True},
             "grade": {"allowed": ["G1", "G2", "G3"]},
             "iban": {"regex": r"[A-Z]{2}\d{2}"},
             "id": {"unique": True}}
    res = _ok("data.validity_rules", _ctx(df), {"rules": rules, "id": "id"})
    t = res.tables["Rule results"].set_index(["column", "rule"])
    assert t.loc[("ltv", "min"), "violations"] == 1 and t.loc[("ltv", "min"), "example_rows"] == "2"
    assert t.loc[("ltv", "max"), "example_rows"] == "1" and t.loc[("ltv", "max"), "example_ids"] == "B"
    assert t.loc[("ltv", "not_null"), "example_rows"] == "3"
    assert t.loc[("grade", "allowed"), "example_rows"] == "2, 3"
    assert t.loc[("iban", "regex"), "example_rows"] == "1" and t.loc[("iban", "regex"), "n_checked"] == 4
    assert t.loc[("id", "unique"), "violations"] == 0
    assert res.summary["rows_violating_any_rule"] == 3
    bad = run_test("data.validity_rules", _ctx(df), {"rules": {"ltv": {"maximum": 1}}})
    assert bad.status == "error" and "maximum" in bad.error


def test_range_checks_heuristics_and_dates():
    df = pd.DataFrame({"pd_12m": [0.01, 0.2, 1.2, -0.01],
                       "ead": [100.0, -5.0, 50.0, 0.0],
                       "start": ["2020-01-01", "2026-01-01", "1899-12-31", "not a date"],
                       "update_flag": [3, 4, 5, 6]})          # 'update' must not match 'pd'
    res = _ok("data.range_checks", _ctx(df), {"as_of": "2025-06-30", "min_date": "1900-01-01",
                                                 "date_columns": ["start"]})
    t = res.tables["Range checks"].set_index(["column", "check"])
    assert t.loc[("pd_12m", "probability outside [0,1]"), "example_rows"] == "2, 3"
    assert t.loc[("pd_12m", "probability outside [0,1]"), "basis"] == "name heuristic"
    assert t.loc[("ead", "negative amount"), "example_rows"] == "1"
    assert t.loc[("start", "date after as_of 2025-06-30"), "example_rows"] == "1"
    assert t.loc[("start", "date before min_date 1900-01-01"), "example_rows"] == "2"
    assert t.loc[("start", "unparseable date"), "example_rows"] == "3"
    assert "update_flag" not in res.tables["Range checks"]["column"].tolist()


# ── outliers ────────────────────────────────────────────────────────────────

def test_outliers_iqr_bounds_and_rows():
    x = list(range(1, 11)) + [100]
    df = pd.DataFrame({"x": x})
    res = _ok("data.outliers", _ctx(df), {"columns": ["x"]})
    t = res.tables["Outliers by column and method"].set_index("method")
    q1, q3 = np.quantile(x, [.25, .75])
    assert t.loc["IQR", "lower_bound"] == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert t.loc["IQR", "upper_bound"] == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert t.loc["IQR", "example_rows"] == "10" and t.loc["IQR", "above"] == 1
    med = np.median(x)
    mad = np.median(np.abs(np.array(x) - med))
    assert t.loc["robust MAD z", "upper_bound"] == pytest.approx(med + 3.5 * mad / 0.6745)
    assert t.loc["robust MAD z", "example_rows"] == "10"
    # z-score: 100 is (100-mean)/sd = 3.0 sd? check against the definition directly
    z = (np.array(x) - np.mean(x)) / np.std(x, ddof=1)
    assert t.loc["z-score", "outliers"] == int((np.abs(z) > 3).sum())


def test_grubbs_critical_values_and_nist_rosner_example():
    assert _grubbs_crit(10, 0.05, True) == pytest.approx(2.290, abs=5e-4)     # Grubbs tables
    assert _grubbs_crit(10, 0.05, False) == pytest.approx(2.176, abs=5e-4)
    x = [-0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56, 1.58, 1.65,
         1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10, 2.14, 2.15, 2.23, 2.24, 2.26,
         2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92, 2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30,
         4.64, 5.34, 5.42, 6.01]
    res = _ok("data.grubbs", _ctx(pd.DataFrame({"x": x})), {"column": "x", "max_outliers": 10})
    t = res.tables["Grubbs / ESD steps"]
    # NIST e-Handbook 1.3.5.17.3 (Rosner's example): R_i and lambda_i, 3 outliers
    assert t["statistic"].round(3).tolist() == [3.119, 2.943, 3.179, 2.810, 2.816, 2.848, 2.279, 2.310,
                                                2.102, 2.067]
    assert t["critical_value"].round(3).tolist() == [3.159, 3.151, 3.144, 3.136, 3.128, 3.120, 3.112,
                                                     3.103, 3.094, 3.085]
    assert res.summary["gesd_outliers"] == 3 and res.summary["gesd_rows"] == "53, 52, 51"
    # p-value at the critical value equals alpha
    from ask.validation.t_data import _grubbs_p
    assert _grubbs_p(_grubbs_crit(20, 0.05, True), 20, True) == pytest.approx(0.05)
    one = _ok("data.grubbs", _ctx(pd.DataFrame({"x": [1, 2, 3, 4, 50]})), {"column": "x"})
    xs = np.array([1, 2, 3, 4, 50.0])
    assert one.summary["G"] == pytest.approx((50 - xs.mean()) / xs.std(ddof=1))
    assert one.summary["suspect_row"] == 4


# ── data types ──────────────────────────────────────────────────────────────

def test_type_consistency_finds_each_issue():
    df = pd.DataFrame({"amt": ["10", "20.5", "abc", "30", "40"],
                       "seg": ["Retail", "retail ", "RETAIL", "SME", "SME"],
                       "dt": ["2024-01-31", "31/01/2024", "2024-02-29", "2024-03-31", "2024-04-30"],
                       "mixed": pd.Series([1, "a", 2.5, "b", None], dtype="object")})
    res = _ok("data.type_consistency", _ctx(df), {})
    t = res.tables["Type checks by column"].set_index("column")
    assert t.loc["amt", "numeric_values"] == 4 and t.loc["amt", "other_text_values"] == 1
    ev = res.tables["Evidence"]
    amt_ev = ev[(ev["column"] == "amt") & ev["check"].str.startswith("non-numeric")]
    assert amt_ev["row"].tolist() == [2] and amt_ev["value"].iloc[0] == "'abc'"
    assert t.loc["seg", "leading_trailing_whitespace"] == 1
    assert ev[(ev["column"] == "seg") & (ev["check"] == "leading/trailing whitespace")]["row"].tolist() == [1]
    v = res.tables["Category spelling variants"]
    assert set(v.loc[v["normalised_label"] == "retail", "raw_label"]) == {"'Retail'", "'retail '", "'RETAIL'"}
    assert v.set_index("raw_label").loc["'RETAIL'", "example_rows"] == "2"
    f = res.tables["Date formats"].set_index("format")
    assert f.loc["dd/dd/dddd", "example_rows"] == "1" and f.loc["dddd-dd-dd", "values"] == 4
    assert t.loc["mixed", "mixed_content"] and "int: 1" in t.loc["mixed", "python_types"]


# ── target ──────────────────────────────────────────────────────────────────

def test_target_sanity_values_prevalence_and_window():
    df = pd.DataFrame({"y": [0, 1, 0, 2, np.nan, 1, 0, 0],
                       "m": ["2024-01", "2024-01", "2024-01", "2024-01", "2024-07", "2024-07", "2024-07", "2024-07"],
                       "seg": list("AABBAABB")})
    res = _ok("data.target_sanity", _ctx(df), {"target": "y", "period": "m", "segment": "seg",
                                                "as_of": "2025-03-31", "horizon_months": 12})
    s = res.summary
    assert s["invalid_target"] == 1 and s["missing_target"] == 1 and s["valid_target_rows"] == 6
    assert s["event_rate"] == pytest.approx(2 / 6)
    v = res.tables["Target values"].set_index("value")
    assert v.loc["2.0", "status"] == "NOT allowed" and v.loc["2.0", "example_rows"] == "3"
    p = res.tables["By period"].set_index("period")
    assert p.loc["2024-01", "event_rate"] == pytest.approx(1 / 3)
    assert p.loc["2024-01", "obs_date"] == "2024-01-31" and p.loc["2024-01", "window_end"] == "2025-01-31"
    assert p.loc["2024-01", "window_complete"] and not p.loc["2024-07", "window_complete"]
    assert s["incomplete_window_periods"] == 1 and s["rows_in_incomplete_periods"] == 4
    g = res.tables["By segment"].set_index("segment")
    assert g.loc["A", "events"] == 2 and g.loc["B", "invalid_target"] == 1


def test_target_sanity_text_values():
    df = pd.DataFrame({"def": ["N", "Y", "N", "n"]})
    res = _ok("data.target_sanity", _ctx(df), {"target": "def", "allowed": ["N", "Y"], "event_value": "Y"})
    assert res.summary["invalid_target"] == 1 and res.summary["event_rate"] == pytest.approx(1 / 3)
    assert res.tables["Target values"].set_index("value").loc["'n'", "example_rows"] == "3"


# ── representativeness and time coverage ────────────────────────────────────

def test_representativeness_against_population():
    rng = np.random.default_rng(5)
    pop = pd.DataFrame({"x": rng.normal(0, 1, 400), "region": rng.choice(list("NSEW"), 400)})
    smp = pop.iloc[:100].copy()
    smp = smp[smp["region"] != "W"].reset_index(drop=True)
    smp.loc[0, "region"] = "Z"
    res = _ok("data.representativeness", RunContext(df=smp, tables={"population": pop}),
              {"other": "population"})
    t = res.tables["Representativeness by variable"].set_index("variable")
    ks = stats.ks_2samp(smp["x"], pop["x"])
    assert t.loc["x", "statistic"] == pytest.approx(ks.statistic) and t.loc["x", "p_value"] == pytest.approx(ks.pvalue)
    assert t.loc["region", "pop_levels_absent_in_sample"] == 1
    assert t.loc["region", "pop_share_of_absent_levels"] == pytest.approx((pop["region"] == "W").mean())
    cov = res.tables["Category coverage"].set_index("level")
    assert cov.loc["Z", "example_rows"] == "0" and cov.loc["W", "sample_rows"] == 0


def test_time_coverage_gaps():
    dates = pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-03", "2024-04-11", "2024-04-12"])
    df = pd.DataFrame({"d": dates, "seg": ["A", "B", "A", "A", "B"]})
    res = _ok("data.time_coverage", _ctx(df), {"period": "d", "segment": "seg"})
    assert res.summary["gaps"] == 1 and res.summary["gap_periods"] == "2024-03"
    t = res.tables["Records per period"].set_index("period")
    assert t.loc["2024-01", "records"] == 2 and t.loc["2024-03", "gap"]
    g = res.tables["Segment gaps"]
    assert ("2024-02", "B") in set(zip(g["period"], g["segment"]))
    ints = pd.DataFrame({"p": [202401, 202402, 202405]})
    r2 = _ok("data.time_coverage", _ctx(ints), {"period": "p"})
    assert r2.summary["gap_periods"] == "2024-03, 2024-04"


# ── reconciliation and referential integrity ────────────────────────────────

def test_reconciliation_keys_values_totals():
    a = pd.DataFrame({"k": [1, 2, 3, 4], "ead": [100.0, 200.0, 300.0, 50.0], "seg": ["R", "R", "C", "C"]})
    b = pd.DataFrame({"k": [1.0, 2.0, 3.0, 9.0], "ead": [100.0, 200.4, 330.0, 10.0], "seg": ["R", "R", "c", "C"]})
    res = _ok("data.reconciliation", RunContext(df=a, tables={"src": b}),
              {"other": "src", "keys": ["k"], "tolerance": 0.5, "group": "seg"})
    s = res.summary
    assert s["matched_keys"] == 3 and s["keys_only_in_A"] == 1 and s["keys_only_in_B"] == 1
    assert res.tables["Keys only in A"].iloc[0]["row_in_A"] == 3
    assert res.tables["Keys only in B"].iloc[0]["k"] == 9.0
    d = res.tables["Value differences by column"].set_index("column")
    assert d.loc["ead", "differences"] == 1 and d.loc["ead", "example_rows_A"] == "2"
    assert d.loc["ead", "max_abs_difference"] == pytest.approx(30.0)
    assert d.loc["seg", "differences"] == 1
    tot = res.tables["Totals"].set_index("column")
    assert tot.loc["ead", "total_A"] == 650 and tot.loc["ead", "total_B"] == pytest.approx(640.4)
    assert tot.loc["ead", "matched_difference"] == pytest.approx(600 - 630.4)
    g = res.tables["Totals by group"].set_index("group")
    assert g.loc["c", "total_B"] == 330 and g.loc["c", "total_A"] == 0


def test_referential_integrity_orphans():
    df = pd.DataFrame({"loan": ["L1", "L2", "L3", "L4", "L5"], "cust": [10, 11, 99, np.nan, 99]})
    ref = pd.DataFrame({"customer_id": [10, 11, 12, 12]})
    res = _ok("data.referential_integrity", RunContext(df=df, tables={"customers": ref}),
              {"column": "cust", "other": "customers", "other_column": "customer_id", "id": "loan"})
    s = res.summary
    assert s["orphan_rows"] == 2 and s["orphan_values"] == 1 and s["missing_foreign_key"] == 1
    assert s["duplicate_reference_keys"] == 1 and s["unused_reference_keys"] == 1
    o = res.tables["Orphan values"].iloc[0]
    assert o["orphan_value"] == "99" and o["example_rows"] == "2, 4" and o["example_ids"] == "L3, L5"


# ── leakage, time consistency, target association ───────────────────────────

def test_leakage_candidates():
    n = 40
    y = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({"y": y,
                       "obs": ["2023-12-31"] * n,
                       "default_date": ["2024-03-01" if v else None for v in y],
                       "recovery": y * 100.0 + np.arange(n) * 0.01,           # perfectly separating
                       "noise": np.random.default_rng(0).normal(size=n),
                       "acct_id": np.arange(n)})
    df.loc[0, "default_date"] = "2023-06-30"                                  # before obs: not leaking
    res = _ok("data.leakage_candidates", _ctx(df), {"target": "y", "date": "obs"})
    t = res.tables["Association with target"].set_index("column")
    assert t.loc["recovery", "auc"] == pytest.approx(1.0) and res.summary["strongest_strength"] == pytest.approx(1.0)
    assert t.loc["acct_id", "semantic_type"] == "id-like"
    post = res.tables["Dates after observation date"].set_index("column")
    assert post.loc["default_date", "rows_after_observation"] == n // 2
    assert _rows(post.loc["default_date", "example_rows"])[:3] == [1, 3, 5]


def test_time_consistency_reversals_and_overlaps():
    df = pd.DataFrame({"id": ["a", "a", "a", "b", "b", "c", "c"],
                       "snap": ["2024-01-31", "2024-03-31", "2024-02-29", "2024-01-31", "2024-01-31",
                                "2024-01-31", "2024-02-29"],
                       "start": ["2020-01-01", "2020-06-01", "2021-01-01", "2020-01-01", "2020-03-01",
                                 "2020-01-01", "2020-05-01"],
                       "end": ["2020-12-31", "2020-12-31", None, "2020-02-29", "2020-04-30",
                               "2020-03-31", "2020-01-15"]})
    res = _ok("data.time_consistency", _ctx(df), {"id": "id", "date": "snap", "start_date": "start",
                                                  "end_date": "end"})
    s = res.summary
    assert s["date_reversals"] == 1 and res.tables["Date order reversals"].iloc[0]["row"] == 2
    assert s["duplicate_id_date_rows"] == 2
    assert s["start_after_end"] == 1                              # row 6
    assert s["overlapping_intervals"] == 1
    ov = res.tables["Overlapping intervals"].iloc[0]
    assert ov["row"] == 1 and ov["overlaps_row"] == 0
    assert s["gaps"] == 1                     # c: May 1 after Mar 31 (b: Mar 1 follows Feb 29 directly)
    assert s["open_ended_intervals"] == 1


def test_target_association_hand_iv_and_cramers_v():
    # categorical: A has 1 event / 3 non-events, B has 3 events / 1 non-event
    df = pd.DataFrame({"y": [1, 0, 0, 0, 1, 1, 1, 0], "g": list("AAAABBBB"),
                       "x": [5.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 4.0]})
    res = _ok("data.target_association", _ctx(df), {"target": "y", "features": ["g", "x"], "bins": 2})
    t = res.tables["Univariate association"].set_index("variable")
    iv = (0.75 - 0.25) * np.log(0.75 / 0.25) + (0.25 - 0.75) * np.log(0.25 / 0.75)
    assert t.loc["g", "iv"] == pytest.approx(iv)
    assert t.loc["g", "cramers_v"] == pytest.approx(0.5)            # phi = (1*1-3*3)/sqrt(4*4*4*4) = -0.5
    from sklearn.metrics import roc_auc_score
    assert t.loc["x", "auc"] == pytest.approx(roc_auc_score(df["y"], df["x"]))
    assert t.loc["x", "gini"] == pytest.approx(2 * roc_auc_score(df["y"], df["x"]) - 1)


# ── cross-cutting ───────────────────────────────────────────────────────────

def test_rows_are_positions_not_index_labels():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]}, index=[100, 200, 300])
    res = _ok("data.missingness", _ctx(df), {})
    assert res.tables["Missing by column"].iloc[0]["example_rows"] == "1"


def test_text_binary_target_is_accepted():
    df = pd.DataFrame({"y": ["Y", "N", "N", "Y", "N", "Y"], "x": [3.0, 1, 2, 4, np.nan, np.nan]})
    res = _ok("data.missing_vs_target", _ctx(df), {"target": "y"})
    assert res.tables["Missingness vs target"].iloc[0]["n_missing"] == 2
    bad = run_test("data.missing_vs_target", _ctx(df.assign(y=["Y", "N", "maybe", "Y", "N", "Y"])), {"target": "y"})
    assert bad.status == "error" and "not binary" in bad.error


def test_determinism_and_registration():
    core.load_all()
    ids = [k for k in core.REGISTRY if k.startswith("data.")]
    assert len(ids) == 19
    for k in ids:
        spec = core.REGISTRY[k]
        assert spec.references and spec.description and "general" in spec.model_types
    df = pd.DataFrame({"a": [1.0, np.nan, 3, 3, 100], "b": ["x", "y", None, "y", "x "]})
    r1 = _ok("data.profile", _ctx(df), {})
    r2 = _ok("data.profile", _ctx(df.copy()), {})
    assert r1.run_id == r2.run_id
    pd.testing.assert_frame_equal(r1.tables["Column profile"], r2.tables["Column profile"])


def test_not_applicable_paths():
    df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [0, 0, 0]})
    assert run_test("data.grubbs", _ctx(df), {"column": "x"}).status == "not_applicable"
    assert run_test("data.target_association", _ctx(df), {"target": "y"}).status == "not_applicable"
    assert run_test("data.missing_vs_target", _ctx(df), {"target": "y"}).status == "not_applicable"
