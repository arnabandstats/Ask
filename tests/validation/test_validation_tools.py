"""The agent-facing validation tools: catalog, runs, suites, audit trail, models, UC tables."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from ask.agent import faithfulness, router
from ask.agent.tools import ToolContext, dispatch
from ask.sources import tables
from ask.sources.registry import SourceRegistry
from ask.validation import store
from tests.conftest import calls, say, tool_call


def _ctx(tmp_path, reg):
    return ToolContext(registry=reg, output_dir=tmp_path / "out")


def call(ctx, tool, **args):
    return dispatch(ctx, tool, json.dumps(args))


@pytest.fixture
def drift_csv(tmp_path):
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"score": np.r_[rng.normal(0, 1, 800), rng.normal(0.4, 1, 600)],
                       "sample": ["dev"] * 800 + ["oot"] * 600})
    p = tmp_path / "drift.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def data_reg(drift_csv):
    reg = SourceRegistry()
    reg.load(drift_csv)
    return reg


class TestCatalogTools:
    def test_list_and_describe(self, tmp_path, data_reg):
        ctx = _ctx(tmp_path, data_reg)
        out = call(ctx, "list_validation_tests", model_type="pd", area="Stability")
        assert "stability.psi" in out and "needs:" in out
        assert "PSI" in call(ctx, "describe_test", test_id="stability.psi")
        assert "Unknown model_type" in call(ctx, "list_validation_tests", model_type="nope")

    def test_unknown_test_is_an_error_not_a_crash(self, tmp_path, data_reg):
        out = call(_ctx(tmp_path, data_reg), "run_validation_test", test_id="no.such")
        assert out.startswith("Error:")


class TestRuns:
    def test_run_saves_audit_record_and_shows_tables(self, tmp_path, data_reg):
        ctx = _ctx(tmp_path, data_reg)
        out = call(ctx, "run_validation_test", test_id="stability.psi",
                   params={"column": "score", "sample": "sample"})
        run_id = ctx.runs[0]
        assert f"[test:{run_id}]" in out and "Provenance:" in out
        assert store.exists(run_id)
        rec = store.load(run_id)
        assert rec["params"]["column"] == "score" and rec["summary"]["PSI"] > 0
        assert any(a["type"] == "table" for a in ctx.artifacts)
        detail = call(ctx, "get_test_run", run_id=run_id)
        assert "PSI by bin" in detail and "sha256" in detail

    def test_rerun_gives_same_run_id(self, tmp_path, data_reg):
        a, b = _ctx(tmp_path, data_reg), _ctx(tmp_path, data_reg)
        call(a, "run_validation_test", test_id="stability.psi", params={"column": "score", "sample": "sample"})
        call(b, "run_validation_test", test_id="stability.psi", params={"column": "score", "sample": "sample"})
        assert a.runs == b.runs

    def test_suite_runs_tests_whose_inputs_are_covered(self, tmp_path, data_reg):
        ctx = _ctx(tmp_path, data_reg)
        out = call(ctx, "run_validation_suite", model_type="pd",
                   columns={"column": "score", "sample": "sample"})
        assert "stability.psi" not in out or "[test:" in out
        assert "[test:" in out and len(ctx.runs) >= 2


class TestCitations:
    def test_unknown_run_ids_are_flagged(self, tmp_path, data_reg):
        ctx = _ctx(tmp_path, data_reg)
        call(ctx, "run_validation_test", test_id="stability.psi", params={"column": "score", "sample": "sample"})
        good = f"PSI is 0.12 [test:{ctx.runs[0]}]."
        assert faithfulness.check_test_citations(good, ctx.runs) == []
        bad = faithfulness.check_test_citations("PSI is 0.12 [test:0123456789ab].", ctx.runs)
        assert bad and "does not match" in bad[0]
        mangled = faithfulness.check_test_citations(f"PSI [test:{ctx.runs[0]}f].", ctx.runs)   # 13 chars
        assert mangled and "does not match" in mangled[0]
        missing = faithfulness.check_test_citations("The PSI is 0.12, so it moved.", ctx.runs)
        assert missing and "cites none" in missing[0]

    def test_findings_must_be_deficiencies_with_evidence(self):
        table = ("| ID | Area | Description | Evidence | Severity |\n|---|---|---|---|---|\n"
                 "| F-01 | Calibration | PDs too low | [test:0123456789ab] | High |\n"
                 "| F-02 | Discrimination | Good AUC | [test:0123456789ab] | — (positive) |\n"
                 "| F-03 | Code | Unseeded split | none | Medium |\n")
        probs = faithfulness.check_findings(table)
        assert len(probs) == 2
        assert probs[0].startswith("F-02 is not a deficiency") and probs[1].startswith("F-03 has no evidence")
        assert faithfulness.check_findings("| F-01 | Code | x | [src/a.py:L3-4] | High |") == []

    def test_router_repairs_an_uncited_test_answer(self, fake_llm, drift_csv, tmp_path):
        reg = SourceRegistry()
        reg.load(drift_csv)
        holder = {}

        def answer_with_id(kwargs):
            rid = next(i["output"].split("]")[0].split("test:")[1] for i in kwargs["input"]
                       if isinstance(i, dict) and i.get("type") == "function_call_output")
            holder["rid"] = rid
            return say(f"PSI between dev and oot is reported in [test:{rid}].")

        fake_llm.script = [
            calls(tool_call("run_validation_test", test_id="stability.psi",
                            params={"column": "score", "sample": "sample"})),
            say("The PSI is 0.25 which shows drift."),                 # uncited -> repair
            answer_with_id,
        ]
        turn = router.answer("check drift of score", reg, [], model="gpt-4.1", verify=True,
                             output_dir=tmp_path / "o", status=lambda m: None)
        assert turn.meta.get("repaired") and f"[test:{holder['rid']}]" in turn.content
        assert turn.meta["test_runs"] == [holder["rid"]]


class TestModels:
    def test_load_model_and_persist_in_records(self, tmp_path, data_reg):
        from sklearn.linear_model import LogisticRegression
        X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [1, 0, 1, 0, 1, 0]})
        m = LogisticRegression().fit(X, [0, 0, 0, 1, 1, 1])
        p = tmp_path / "pd_model.pkl"
        p.write_bytes(pickle.dumps(m))
        ctx = _ctx(tmp_path, data_reg)
        out = call(ctx, "load_model", location=str(p), name="pd_model")
        assert "Loaded model pd_model" in out and "runs code" in out        # the user is warned
        assert ctx.sources_changed and "pd_model" in data_reg.models
        assert "[model] pd_model" in data_reg.describe()
        recs = data_reg.records()
        fresh = SourceRegistry()
        assert fresh.restore(recs) == []
        assert "pd_model" in fresh.models

        from ask.validation.models import predict_scores
        s = predict_scores(fresh.models["pd_model"], X)
        assert s.shape == (6,) and np.all((s >= 0) & (s <= 1))


class TestTables:
    def test_select_guard(self):
        assert tables.check_select("SELECT a FROM c.s.t;") == "SELECT a FROM c.s.t"
        for bad in ["DROP TABLE x", "SELECT 1; DELETE FROM t", "WITH x AS (SELECT 1) DELETE FROM t",
                    "update t set a=1"]:
            with pytest.raises(tables.TableError):
                tables.check_select(bad)
        assert tables.check_select("SELECT 'delete' AS word FROM t")      # inside a string literal is fine
        with pytest.raises(tables.TableError):
            tables._q("c.s.t; drop")

    def test_big_table_is_hash_sampled_deterministically(self, monkeypatch):
        seen = []

        def fake(sql):
            seen.append(sql)
            if sql.startswith("SELECT COUNT(*)"):
                return pd.DataFrame({"n": [10_000_000]}), "Spark"
            return pd.DataFrame({"a": [1, 2]}), "Spark"

        monkeypatch.setattr(tables, "run_sql", fake)
        out = tables.load_table("main.risk.loans", columns=["a"], where="year = 2024", max_rows=1_000_000,
                                key=["a"])
        assert out.sampled and "same rows every run" in out.sample_rule
        assert "pmod(xxhash64(`a`), 1000000) < 100000" in seen[-1]
        assert seen[-1].endswith("ORDER BY `a`") and "WHERE year = 2024 AND" in seen[-1]

    def test_load_table_tool_registers_a_data_source(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tables, "run_sql", lambda sql: (
            (pd.DataFrame({"n": [3]}), "Spark") if "COUNT" in sql
            else (pd.DataFrame({"pd": [0.1, 0.2, 0.3], "y": [0, 1, 0]}), "Spark")))
        reg = SourceRegistry()
        ctx = _ctx(tmp_path, reg)
        out = call(ctx, "load_table", table="main.risk.loans", name="loans")
        assert "Loaded [data] loans" in out and reg.sources["loans"].df.shape == (3, 2)
        rec = next(r for r in reg.records() if r["name"] == "loans")
        assert rec["path"] == "uc://main.risk.loans" and rec["table"]["table"] == "main.risk.loans"
