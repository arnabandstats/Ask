"""Core registry/provenance behaviour and the stability tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ask.validation import core
from ask.validation.core import RunContext, run_test


def _ctx(df, **kw):
    return RunContext(df=df, source_name="portfolio", **kw)


@pytest.fixture
def two_samples():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"score": rng.normal(0, 1, 2000), "grade": rng.choice(list("ABCD"), 2000),
                        "sample": "dev"})
    cur = pd.DataFrame({"score": rng.normal(0.5, 1, 1500), "grade": rng.choice(list("ABCD"), 1500,
                                                                               p=[.1, .2, .3, .4]),
                        "sample": "oot"})
    return pd.concat([ref, cur], ignore_index=True)


def test_every_test_is_well_formed():
    core.load_all()
    assert len(core.REGISTRY) >= 4
    for spec in core.REGISTRY.values():
        assert spec.id.count(".") == 1, spec.id
        assert spec.model_types and all(m in core.MODEL_TYPES for m in spec.model_types)
        assert spec.description, spec.id
        assert spec.kind in {"statistical", "judge"}


def test_psi_matches_hand_computation():
    df = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50 + ["a"] * 30 + ["b"] * 70,
                       "s": ["dev"] * 100 + ["oot"] * 100})
    res = run_test("stability.psi", _ctx(df), {"column": "g", "sample": "s"})
    assert res.status == "ok", res.error
    expected = (0.3 - 0.5) * np.log(0.3 / 0.5) + (0.7 - 0.5) * np.log(0.7 / 0.5)
    assert res.summary["PSI"] == pytest.approx(expected)
    assert res.summary["reference"] == "dev"


def test_same_inputs_same_run_id_and_numbers(two_samples):
    a = run_test("stability.psi", _ctx(two_samples), {"column": "score", "sample": "sample"})
    b = run_test("stability.psi", _ctx(two_samples.copy()), {"column": "score", "sample": "sample"})
    assert a.run_id == b.run_id and a.summary == b.summary
    changed = two_samples.copy()
    changed.loc[0, "score"] += 1
    c = run_test("stability.psi", _ctx(changed), {"column": "score", "sample": "sample"})
    assert c.run_id != a.run_id and c.data_fingerprint != a.data_fingerprint
    assert "run_id=" in a.to_text() and a.to_json()["run_id"] == a.run_id


def test_shift_is_detected(two_samples):
    res = run_test("stability.distribution_tests", _ctx(two_samples), {"column": "score", "sample": "sample"})
    assert res.status == "ok", res.error
    t = res.tables["Distribution tests"]
    assert t.loc[t["test"].str.startswith("Kolmogorov"), "p_value"].iloc[0] < 1e-6
    res = run_test("stability.csi", _ctx(two_samples), {"sample": "sample", "features": ["score", "grade"]})
    assert set(res.tables["CSI by variable"]["variable"]) == {"score", "grade"}


def test_bad_parameters_come_back_as_errors(two_samples):
    res = run_test("stability.psi", _ctx(two_samples), {"column": "nope", "sample": "sample"})
    assert res.status == "error" and "nope" in res.error
    res = run_test("stability.psi", _ctx(two_samples), {"column": "score", "bogus": 1})
    assert res.status == "error" and "bogus" in res.error
    with pytest.raises(KeyError):
        core.get("no.such_test")


def test_run_id_changes_when_the_test_code_changes(two_samples, monkeypatch):
    a = run_test("stability.psi", _ctx(two_samples), {"column": "score", "sample": "sample"})
    assert a.versions["code"] and f"code {a.versions['code']}" in a.provenance()
    monkeypatch.setitem(core._CODE_HASHES, "ask.validation.t_stability", "edited-code")
    b = run_test("stability.psi", _ctx(two_samples), {"column": "score", "sample": "sample"})
    assert b.run_id != a.run_id and b.summary == a.summary


def test_catalog_doc_is_up_to_date():
    from ask.validation import catalog_doc
    assert catalog_doc.DOC.read_text(encoding="utf-8") == catalog_doc.markdown(), \
        "docs/TEST_CATALOG.md is stale: run python -m ask.validation.catalog_doc"


def test_lookup_by_short_name():
    assert core.get("psi").id == "stability.psi"
