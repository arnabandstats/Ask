"""The deterministic citation / claim checker."""
from __future__ import annotations

import pytest

from ask.agent import faithfulness as fv
from ask.retrieval.search import Evidence


@pytest.fixture
def ev_read_train():
    ev = Evidence()
    ev.add("model_repo", "src/train.py", 1, 16)
    return ev


class TestParsing:
    @pytest.mark.parametrize("text,ref,ranges", [
        ("[src/train.py:L7-8]", "src/train.py", [(7, 8)]),
        ("[src/train.py:L7]", "src/train.py", [(7, 7)]),
        ("[src/train.py:L7-L8]", "src/train.py", [(7, 8)]),
        ("[src/train.py:L7–8]", "src/train.py", [(7, 8)]),
        ("[src/train.py:L3-7,L89-101]", "src/train.py", [(3, 7), (89, 101)]),
        ("[src/train.py:L3-7, 9-10]", "src/train.py", [(3, 7), (9, 10)]),
        ("[repo:src/train.py:L8-7]", "repo:src/train.py", [(7, 8)]),
    ])
    def test_forms(self, text, ref, ranges):
        m = fv.CITATION.search(text)
        assert m and m.group(1) == ref and fv.parse_ranges(m.group(2)) == ranges

    @pytest.mark.parametrize("text", ["[link](http://x)", "[1]", "[train.py:7-8]", "list[int]"])
    def test_non_citations(self, text):
        assert fv.CITATION.search(text) is None


class TestVerify:
    def test_ok(self, repo_registry, ev_read_train):
        rep = fv.verify("The gate compares accuracy [src/train.py:L7-8].", repo_registry, ev_read_train, True)
        assert rep.ok and rep.citations[0].status == "ok"
        assert rep.citations[0].snippet.startswith("7| def passes_quality_gate")

    def test_not_read(self, repo_registry):
        rep = fv.verify("Gate [src/train.py:L7-8].", repo_registry, Evidence(), True)
        assert rep.citations[0].status == "not_read" and not rep.ok

    def test_partially_read_wide_range(self, repo_registry):
        ev = Evidence()
        ev.add("model_repo", "src/train.py", 7, 7)
        rep = fv.verify("Gate [src/train.py:L1-14].", repo_registry, ev, True)
        assert rep.citations[0].status == "not_read" and "only 1 of 14" in rep.citations[0].detail

    def test_bad_range(self, repo_registry, ev_read_train):
        rep = fv.verify("Gate [src/train.py:L900-910].", repo_registry, ev_read_train, True)
        assert rep.citations[0].status == "bad_range"

    def test_unknown_file(self, repo_registry, ev_read_train):
        rep = fv.verify("Gate [src/missing.py:L1-2].", repo_registry, ev_read_train, True)
        assert rep.citations[0].status == "unknown_file"

    def test_each_range_checked(self, repo_registry):
        ev = Evidence()
        ev.add("model_repo", "src/train.py", 7, 8)
        rep = fv.verify("Gate [src/train.py:L7-8,L11-14].", repo_registry, ev, True)
        assert [c.status for c in rep.citations] == ["ok", "not_read"]

    def test_hallucinated_identifier(self, repo_registry, ev_read_train):
        rep = fv.verify("It calls `magic_helper` [src/train.py:L7-8].", repo_registry, ev_read_train, True)
        assert any("magic_helper" in p for p in rep.problems)

    def test_real_identifier_and_filenames_pass(self, repo_registry, ev_read_train):
        rep = fv.verify("`passes_quality_gate()` in `train.py` uses `metrics` [src/train.py:L7-8].",
                        repo_registry, ev_read_train, True)
        assert rep.ok

    def test_uncited_grounded_answer_flagged(self, repo_registry):
        rep = fv.verify("The model is trained with a random forest. " * 10, repo_registry, Evidence(), True)
        assert rep.uncited and not rep.ok

    def test_general_knowledge_label_exempts(self, repo_registry):
        rep = fv.verify("General knowledge: PSI measures distribution shift. " * 10,
                        repo_registry, Evidence(), True)
        assert rep.ok

    def test_short_answers_not_flagged(self, repo_registry):
        assert fv.verify("Yes.", repo_registry, Evidence(), True).ok

    def test_to_meta_dedupes(self, repo_registry, ev_read_train):
        rep = fv.verify("A [src/train.py:L7-8]. B [src/train.py:L7-8].", repo_registry, ev_read_train, True)
        meta = rep.to_meta()
        assert meta["checked"] and len(meta["citations"]) == 1 and meta["problems"] == []


class TestVisualClaims:
    @pytest.mark.parametrize("text", ["The chart is shown below.", "See the chart.",
                                      "A histogram has been created above.",
                                      "Charts (ROC, confusion matrix) have been generated."])
    def test_claims_without_visual_flagged(self, text):
        assert fv.check_visual_claims(text, [])

    @pytest.mark.parametrize("text", ["The chart could not be generated.",
                                      "If you want a chart, let me know.",
                                      "Would you like a plot?"])
    def test_negations_and_offers_ok(self, text):
        assert fv.check_visual_claims(text, []) == []

    def test_claim_ok_when_visual_exists(self):
        assert fv.check_visual_claims("The chart is shown below.", [{"type": "plotly"}]) == []
