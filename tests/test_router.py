"""The agent turn: fast loading, tool loop, citation verification and repair."""
from __future__ import annotations

from pathlib import Path

import pytest

from ask import config
from ask.agent import router
from ask.sources.registry import SourceRegistry
from tests.conftest import calls, say, tool_call


def ask(reg, text, tmp_path, history=None, **kw):
    statuses = []
    turn = router.answer(text, reg, history or [], model=kw.pop("model", "gpt-4.1"),
                         verify=kw.pop("verify", True), output_dir=tmp_path / "out",
                         status=statuses.append, **kw)
    return turn, statuses


class TestFastLoad:
    def test_pure_load_needs_no_llm(self, sample_repo, tmp_path, fake_llm):
        reg = SourceRegistry()
        turn, _ = ask(reg, f"load this repo: {sample_repo}", tmp_path)
        assert fake_llm.requests == []
        assert turn.content.startswith("Loaded repo **model_repo**") and turn.sources_changed
        assert turn.meta == {"kind": "load"}

    def test_missing_path(self, tmp_path, fake_llm):
        turn, _ = ask(SourceRegistry(), f"load {tmp_path / 'ghost'}", tmp_path)
        assert turn.content.startswith("Path not found") and not turn.sources_changed

    def test_kind_hint_from_words_not_path(self, tmp_path, fake_llm):
        d = tmp_path / "Repository" / "policies"
        d.mkdir(parents=True)
        (d / "policy.md").write_text("# Policy\nValidate annually.\n")
        reg = SourceRegistry()
        ask(reg, f"load {d / 'policy.md'}", tmp_path)
        assert reg.sources["policy.md"].kind == "docs"       # "Repository" in the path is ignored

    def test_skipped_files_reported(self, sample_repo, tmp_path, fake_llm, monkeypatch):
        monkeypatch.setattr(config, "MAX_FILE_BYTES", 120)
        turn, _ = ask(SourceRegistry(), f"load {sample_repo}", tmp_path)
        assert "Skipped" in turn.content

    def test_question_with_path_goes_to_agent(self, sample_repo, tmp_path, fake_llm):
        fake_llm.script = [calls(tool_call("load_path", path=str(sample_repo))),
                           calls(tool_call("read_file", "c2", file="README.md")),
                           say("It trains a random forest [README.md:L3].")]
        reg = SourceRegistry()
        turn, _ = ask(reg, f"load {sample_repo} and tell me what it does", tmp_path)
        assert "model_repo" in reg.sources and turn.sources_changed
        assert turn.meta["verification"]["problems"] == []


class TestGroundedAnswers:
    def test_verified_citation(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [calls(tool_call("read_file", file="src/train.py", start_line=7, end_line=8)),
                           say("The gate compares accuracy to a minimum [src/train.py:L7-8].")]
        turn, statuses = ask(repo_registry, "how does the quality gate work?", tmp_path)
        v = turn.meta["verification"]
        assert [c["status"] for c in v["citations"]] == ["ok"] and v["problems"] == []
        assert "repaired" not in turn.meta and len(fake_llm.requests) == 2
        assert turn.meta["tools"] == ["read_file"] and turn.meta["api"] == "responses"
        assert "Checking citations against the source" in statuses

    def test_unread_citation_is_repaired(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("The gate is at [src/train.py:L7-8] and calls `magic_fn`."),
                           calls(tool_call("read_file", file="src/train.py", start_line=7, end_line=8)),
                           say("The gate compares accuracy [src/train.py:L7-8].")]
        turn, _ = ask(repo_registry, "where is the gate?", tmp_path)
        repair_prompt = fake_llm.requests[1]["input"][-1]["content"]
        assert "never shown to you" in repair_prompt and "magic_fn" in repair_prompt
        assert turn.meta["repaired"] and turn.meta["verification"]["problems"] == []
        assert turn.content == "The gate compares accuracy [src/train.py:L7-8]."

    def test_uncited_answer_with_repo_loaded_is_repaired(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("This app enforces strict faithfulness rules. " * 8),
                           calls(tool_call("overview")),
                           say("It trains a random forest and gates on accuracy [README.md:L1-3].")]
        turn, _ = ask(repo_registry, "what is happening in the repo?", tmp_path)
        assert "no citations" in fake_llm.requests[1]["input"][-1]["content"]
        assert turn.meta["repaired"] and turn.meta["verification"]["problems"] == []

    def test_general_knowledge_is_not_forced_to_cite(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("General knowledge: PSI measures population shift. " * 6)]
        turn, _ = ask(repo_registry, "what is PSI?", tmp_path)
        assert "repaired" not in turn.meta and len(fake_llm.requests) == 1

    def test_unfixable_problems_are_reported_not_hidden(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("See [src/train.py:L900-901]."), say("Still [src/train.py:L900-901].")]
        turn, _ = ask(repo_registry, "where?", tmp_path)
        assert turn.meta["repaired"]
        assert turn.meta["verification"]["citations"][0]["status"] == "bad_range"

    def test_verify_off(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("Unchecked [src/train.py:L7].")]
        turn, _ = ask(repo_registry, "q", tmp_path, verify=False)
        assert "verification" not in turn.meta and len(fake_llm.requests) == 1


class TestDataAndVisuals:
    @pytest.fixture
    def reg(self, portfolio_csv):
        r = SourceRegistry()
        r.load(portfolio_csv)
        return r

    def test_data_answer_not_citation_checked(self, reg, tmp_path, fake_llm):
        fake_llm.script = [calls(tool_call("query_data", expression="df['default_flag'].mean()")),
                           say("The default rate is about 50%.")]
        turn, _ = ask(reg, "what is the default rate?", tmp_path)
        assert "verification" not in turn.meta and "query_data" in turn.meta["notes"]

    def test_claimed_chart_without_chart_is_repaired(self, reg, tmp_path, fake_llm):
        fake_llm.script = [say("The chart is shown below."),
                           calls(tool_call("make_chart", kind="bar", x="segment", y="default_flag", agg="mean")),
                           say("The chart is shown below.")]
        turn, _ = ask(reg, "chart default rate by segment", tmp_path)
        assert "no chart was created" in fake_llm.requests[1]["input"][-1]["content"]
        assert turn.meta["repaired"] and turn.artifacts[0]["type"] == "plotly"

    def test_chart_claim_with_chart_is_fine(self, reg, tmp_path, fake_llm):
        fake_llm.script = [calls(tool_call("make_chart", kind="histogram", x="pd_score")),
                           say("The chart is shown below.")]
        turn, _ = ask(reg, "histogram of scores", tmp_path)
        assert "repaired" not in turn.meta and len(turn.artifacts) == 1


class TestLoopMechanics:
    def test_step_limit_forces_final_answer(self, repo_registry, tmp_path, fake_llm, monkeypatch):
        monkeypatch.setattr(config, "MAX_AGENT_STEPS", 2)
        fake_llm.script = [calls(tool_call("list_files")), calls(tool_call("list_files", "c2")),
                           say("General knowledge: best effort.")]
        turn, statuses = ask(repo_registry, "q", tmp_path)
        last = fake_llm.requests[-1]
        assert "tools" not in last and "Stop using tools" in last["input"][-1]["content"]
        assert "Writing the answer" in statuses and turn.content.startswith("General knowledge")

    def test_system_prompt_lists_sources(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("General knowledge: ok.")]
        ask(repo_registry, "q", tmp_path)
        assert "[repo] model_repo" in fake_llm.requests[0]["instructions"]

    def test_history_is_trimmed(self, repo_registry, tmp_path, fake_llm, monkeypatch):
        monkeypatch.setattr(config, "HISTORY_TURNS", 2)
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
        fake_llm.script = [say("General knowledge: ok.")]
        ask(repo_registry, "now", tmp_path, history=history)
        sent = [i["content"] for i in fake_llm.requests[0]["input"]]
        assert sent == ["m6", "m7", "m8", "m9", "now"]

    def test_reasoning_effort_passed(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("General knowledge: ok.")]
        ask(repo_registry, "q", tmp_path, model="gpt-5.6-luna", reasoning_effort="high")
        assert fake_llm.requests[0]["reasoning"] == {"effort": "high"}

    def test_empty_answer_placeholder(self, repo_registry, tmp_path, fake_llm):
        fake_llm.script = [say("")]
        turn, _ = ask(repo_registry, "q", tmp_path, verify=False)
        assert turn.content == "(no answer)"


def test_history_for_model_appends_notes():
    msgs = [{"role": "user", "content": "run tests"},
            {"role": "assistant", "content": "Done.", "meta": {"notes": "AUC 0.74"}},
            {"role": "assistant", "content": "Plain.", "meta": {}}]
    out = router.history_for_model(msgs)
    assert out[0] == {"role": "user", "content": "run tests"}
    assert out[1]["content"].startswith("Done.") and "AUC 0.74" in out[1]["content"]
    assert out[2] == {"role": "assistant", "content": "Plain."}
