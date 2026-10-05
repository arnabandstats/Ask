"""Documentation review support: standards YAML schema, keyword evidence, cross-checks."""
from __future__ import annotations

import re

import pytest
import yaml

from ask.validation import doc_review as dr
from ask.validation.core import MODEL_TYPES

YAMLS = sorted(dr.STANDARDS_DIR.glob("*.yaml"))


# ── YAML schema ────────────────────────────────────────────────────────────

def test_standards_directory_has_the_expected_standards():
    ids = {s.id for s in dr.list_standards()}
    expected = {"ecb_guide_internal_models", "crr_irb", "eba_gl_2017_16_pd_lgd", "eba_gl_2016_07_default",
                "eba_gl_2019_03_downturn_lgd", "ifrs9_impairment", "crr_market_risk_ima", "crr_ccr_imm",
                "eu_ai_act", "aml_transaction_monitoring", "sr_11_7", "generic_model_documentation",
                "genai_checklist"}
    assert expected <= ids
    assert len(YAMLS) == len(ids)


@pytest.mark.parametrize("path", YAMLS, ids=lambda p: p.name)
def test_yaml_schema(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("id", "title", "version_note", "applies_to", "requirements"):
        assert raw.get(key), f"{path.name}: missing {key}"
    assert raw["id"] == path.stem
    assert set(raw["applies_to"]) <= set(MODEL_TYPES)
    assert len(raw["requirements"]) >= 10
    for r in raw["requirements"]:
        for key in ("id", "topic", "requirement", "source", "keywords", "evidence_expected"):
            assert isinstance(r.get(key), (str, list)) and r[key], f"{path.name} {r.get('id')}: {key}"
        assert set(r) <= {"id", "topic", "requirement", "source", "model_types", "keywords", "patterns",
                          "evidence_expected"}, f"{r['id']}: unknown fields"
        assert set(r.get("model_types") or []) <= set(raw["applies_to"]), r["id"]
        assert isinstance(r["keywords"], list) and len(r["keywords"]) >= 3, r["id"]
        assert all(isinstance(k, str) and k.strip() for k in r["keywords"]), r["id"]
        for p in r.get("patterns") or []:
            re.compile(p)
        # one verifiable statement, not a topic heading
        assert len(r["requirement"].split()) >= 8 and r["requirement"].rstrip().endswith("."), r["id"]


def test_requirement_ids_unique_across_all_standards():
    ids = [r.id for s in dr.list_standards() for r in s.requirements]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_main_standards_are_deep():
    for sid, n in [("crr_irb", 50), ("ecb_guide_internal_models", 30), ("eba_gl_2017_16_pd_lgd", 25),
                   ("ifrs9_impairment", 20), ("eu_ai_act", 20), ("crr_market_risk_ima", 20)]:
        assert len(dr.load_standard(sid).requirements) >= n, sid


def test_every_keyword_matches_its_own_text():
    for s in dr.list_standards():
        for r in s.requirements:
            for k in r.keywords:
                assert dr._keyword_regex(k).search(k), (r.id, k)


def test_sr_11_7_is_labelled_a_us_benchmark():
    s = dr.load_standard("sr_11_7")
    assert "US" in s.version_note and "benchmark" in s.version_note


def test_parse_rejects_bad_files():
    good = {"id": "x", "title": "X", "version_note": "n", "applies_to": ["pd"],
            "requirements": [{"id": "A", "topic": "t", "requirement": "r", "source": "s",
                              "keywords": ["k"], "evidence_expected": "e"}]}
    assert dr._parse(good, "x").requirements[0].keywords == ("k",)
    with pytest.raises(ValueError, match="unknown model types"):
        dr._parse({**good, "applies_to": ["nope"]}, "x")
    with pytest.raises(ValueError, match="missing"):
        dr._parse({**good, "requirements": [{"id": "A"}]}, "x")
    dup = good["requirements"] * 2
    with pytest.raises(ValueError, match="duplicate"):
        dr._parse({**good, "requirements": dup}, "x")
    narrowed = [{**good["requirements"][0], "model_types": ["lgd"]}]
    with pytest.raises(ValueError, match="applies_to"):
        dr._parse({**good, "requirements": narrowed}, "x")


# ── lookup ─────────────────────────────────────────────────────────────────

def test_list_standards_by_model_type():
    pd_ids = {s.id for s in dr.list_standards("pd")}
    assert {"crr_irb", "ecb_guide_internal_models", "generic_model_documentation"} <= pd_ids
    assert "crr_market_risk_ima" not in pd_ids
    assert {s.id for s in dr.list_standards("genai")} >= {"genai_checklist", "eu_ai_act"}
    with pytest.raises(ValueError):
        dr.list_standards("nonsense")


def test_load_standard_fuzzy():
    assert dr.load_standard("crr_irb").id == "crr_irb"
    assert dr.load_standard("ECB Guide to internal models").id == "ecb_guide_internal_models"
    assert dr.load_standard("ecb guide").id == "ecb_guide_internal_models"
    assert dr.load_standard("SR 11-7").id == "sr_11_7"
    assert dr.load_standard("ifrs9 impairmnt").id == "ifrs9_impairment"  # close match
    with pytest.raises(KeyError, match="ambiguous"):
        dr.load_standard("CRR")
    with pytest.raises(KeyError, match="crr_irb"):
        dr.load_standard("completely unrelated")


# ── keyword evidence ───────────────────────────────────────────────────────

def _req(id, keywords, patterns=(), model_types=()):
    return dr.Requirement(id, "topic", "statement.", "source", tuple(model_types), tuple(keywords),
                          tuple(patterns), "expected")


def _std(*reqs):
    return dr.Standard("t", "Test", "note", ("pd", "lgd"), tuple(reqs))


FILES = {
    "b.md": "Intro\nThe margin of conservatism covers category A deficiencies.\n"
            "General estimation error is in MoC category C.\n",
    "a.md": "Nothing here\nWe mention estimation errors once.\n",
    "c.md": "A cure is defined.\nx\nx\nThe probation period is three months.\n",
}


def test_check_exact_hits_and_status():
    moc = _req("MOC", ["margin of conservatism", "category A", "estimation error"], [r"\bMoC\b"])
    none = _req("NONE", ["downturn", "economic downturn", "haircut"])
    weak = _req("WEAK", ["cure", "probation period", "exit from default"])
    res = dr.check(FILES, _std(moc, none, weak))
    assert [c.requirement.id for c in res] == ["MOC", "NONE", "WEAK"]
    c = res[0]
    assert c.status == "evidence found" and c.coverage == 1.0
    # b.md L2 and L3 both see 4 terms in their window; L3 is suppressed as adjacent; a.md L2 has 1
    assert [(h.file, h.line) for h in c.hits] == [("b.md", 2), ("a.md", 2)]
    assert c.hits[0].matched == "margin of conservatism, category A"
    assert c.hits[0].text == "The margin of conservatism covers category A deficiencies."
    assert c.hits[1].matched == "estimation error"
    assert res[1].status == "no evidence found" and res[1].hits == [] and res[1].coverage == 0.0
    w = res[2]
    assert w.status == "weak evidence" and w.coverage == pytest.approx(2 / 3, abs=1e-4)
    assert [(h.file, h.line) for h in w.hits] == [("c.md", 1), ("c.md", 4)]


def test_check_line_numbers_follow_splitlines():
    files = {"d.txt": "\r\n\r\nheader\r\nwe apply back testing daily\r\n"}
    res = dr.check(files, _std(_req("BT", ["back-testing", "daily", "overshooting"])))
    assert res[0].hits[0].line == 4
    assert res[0].hits[0].matched == "back-testing, daily"


def test_keyword_variants_and_word_boundaries():
    rx = dr._keyword_regex("back-testing")
    assert rx.search("Backtesting results") and rx.search("back testing") and rx.search("BACK-TESTING")
    assert dr._keyword_regex("PD").search("the PD model")
    assert not dr._keyword_regex("PD").search("updated weekly")
    assert dr._keyword_regex("override").search("overrides were monitored")


def test_check_ties_broken_by_file_then_line_and_max_hits():
    files = {"z.md": "alpha beta\n", "y.md": "\n\n\nalpha beta\n\n\nalpha beta\n"}
    res = dr.check(files, _std(_req("T", ["alpha", "beta", "gamma"])), max_hits=2)
    assert [(h.file, h.line) for h in res[0].hits] == [("y.md", 4), ("y.md", 7)]


def test_check_model_type_narrowing():
    std = _std(_req("ALL", ["x1", "x2", "x3"]), _req("LGD", ["x1", "x2", "x3"], model_types=["lgd"]))
    assert [c.requirement.id for c in dr.check({"f": "x1"}, std, model_type="pd")] == ["ALL"]
    assert len(dr.check({"f": "x1"}, std)) == 2
    with pytest.raises(ValueError):
        dr.check({"f": "x1"}, std, model_type="bogus")


def test_check_real_standard_is_deterministic_and_tabulates():
    doc = {"model_doc.md": "\n".join([
        "1 Purpose",
        "The PD model estimates the one-year default rate and is calibrated to the long-run average default rate.",
        "The margin of conservatism (MoC) covers category A, category B and category C.",
        "Out-of-time and out-of-sample tests were performed on a holdout sample.",
    ])}
    std = dr.load_standard("crr_irb")
    a = dr.check(doc, std, model_type="pd")
    b = dr.check(doc, std, model_type="pd")
    assert [(c.status, [(h.file, h.line) for h in c.hits]) for c in a] == \
           [(c.status, [(h.file, h.line) for h in c.hits]) for c in b]
    by_id = {c.requirement.id: c for c in a}
    assert by_id["CRR-180-01"].status == "evidence found"
    assert by_id["CRR-180-01"].hits[0].line == 2
    assert by_id["CRR-175-06"].hits[0].line == 4
    assert "CRR-181-01" not in by_id  # LGD-only requirement skipped for a PD model
    t = dr.checks_table(a)
    assert list(t.columns) == ["Requirement", "Topic", "Statement", "Source", "Keyword evidence",
                               "Keyword coverage", "Cited lines", "Matched terms", "Evidence expected"]
    assert set(t["Keyword evidence"]) <= set(dr.STATUSES)
    assert t.loc[t["Requirement"] == "CRR-180-01", "Cited lines"].iloc[0].startswith("[model_doc.md:L2]")


# ── numeric cross-check ────────────────────────────────────────────────────

def _row(df, stated):
    sub = df[df["Stated value"] == stated]
    assert len(sub) == 1, (stated, df["Stated value"].tolist())
    return sub.iloc[0]


def test_cross_check_values_known_answers():
    docs = {"doc.md": "\n".join([
        "The PD floor of 0.03% applies to corporates.",          # L1
        "Default occurs at 90 days past due.",                   # L2
        "Confidence level 99.9% and maximum LGD 45%.",           # L3
        "A spread of 150 bps is added.",                         # L4
        "The cap is 0,5% for retail.",                           # L5
        "Data cover at least five years.",                       # L6
        "Unmatched threshold of 0.123%.",                        # L7
        "See Article 178, section 4.2.1, version 3, dated 2023-01-31 and in 2019.",  # L8
    ])}
    code = {"params.py": "\n".join([
        "PD_FLOOR = 3e-4",            # L1
        "DPD_LIMIT = 90",             # L2
        "CONF = 0.999",               # L3
        "LGD_CAP = 0.45",             # L4
        "SPREAD = 0.015",             # L5
        "RETAIL_CAP = 0.005",         # L6
        "MIN_YEARS = 5",              # L7
    ]), "other.py": "floor = 3.0E-04\n"}
    df = dr.cross_check_values(docs, code)
    r = _row(df, "0.03%")
    assert r["Line"] == 1 and r["Unit"] == "%" and r["Context"] == "PD floor of"
    assert r["Code matches"] == "[other.py:L1]; [params.py:L1]"
    assert r["Found in code"]
    assert _row(df, "90 days")["Code matches"] == "[params.py:L2]"
    assert _row(df, "99.9%")["Code matches"] == "[params.py:L3]"
    r45 = _row(df, "45%")
    assert r45["Code matches"] == "[params.py:L4]" and r45["Context"] == "maximum LGD"
    assert _row(df, "150 bps")["Code matches"] == "[params.py:L5]"
    assert _row(df, "0,5%")["Code matches"] == "[params.py:L6]"
    assert _row(df, "five years")["Code matches"] == "[params.py:L7]"
    miss = _row(df, "0.123%")
    assert miss["Code matches"] == "none" and not miss["Found in code"]
    assert (df["Line"] == 8).sum() == 0  # article, section, version, date and year skipped


def test_cross_check_values_percent_also_matches_percent_number():
    df = dr.cross_check_values({"d": "LGD floor 10%"}, {"c.py": "floor_pct = 10\nfloor = 0.1"})
    assert _row(df, "10%")["Code matches"] == "[c.py:L2]; [c.py:L1]"


def test_cross_check_values_empty():
    df = dr.cross_check_values({"d": "no numbers here"}, {"c.py": "x = 1"})
    assert df.empty and "Code matches" in df.columns


# ── column cross-check ─────────────────────────────────────────────────────

def test_normalise_name():
    assert dr.normalise_name("LoanToValue") == "loan_to_value"
    assert dr.normalise_name("loan_to_value") == "loan_to_value"
    assert dr.normalise_name("Loan-to-value") == "loan_to_value"
    assert dr.normalise_name("LTVRatio") == "ltv_ratio"
    assert dr.normalise_name("pd12m") == "pd_12_m"


def test_cross_check_columns():
    docs = {"d.md": "We use loan_to_value and `dti` plus customerAge.\nDebt to income and Months On Book matter."}
    cols = ["LoanToValue", "dti", "debt_to_income", "months_on_book", "region"]
    df = dr.cross_check_columns(docs, cols).set_index("Normalised name")
    assert df.loc["customer_age", "Where"] == "documents only"
    assert df.loc["customer_age", "Document lines"] == "[d.md:L1]"
    assert df.loc["region", "Where"] == "data only"
    for k in ("loan_to_value", "dti", "debt_to_income", "months_on_book"):
        assert df.loc[k, "Where"] == "both", k
    assert df.loc["debt_to_income", "Document lines"] == "[d.md:L2]"
    assert df.loc["loan_to_value", "Data column"] == "LoanToValue"
    assert df.loc["loan_to_value", "Named in documents as"] == "loan_to_value"
    order = dr.cross_check_columns(docs, cols)["Where"].tolist()
    assert order == sorted(order, key={"documents only": 0, "data only": 1, "both": 2}.get)
