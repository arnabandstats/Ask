"""Keyword search, grep, read_file, list_files, overview — and evidence tracking."""
from __future__ import annotations

import re

import pytest

from ask import config
from ask.retrieval import search as rs
from ask.retrieval.search import Evidence, tokens
from ask.sources.registry import SourceRegistry


def _numbered_lines(out: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"^\s*(\d+)\| ?(.*)$", out, re.M)}


class TestTokens:
    def test_splits_snake_and_camel(self):
        t = tokens("passes_quality_gate calculatePsi HTTPServer")
        for w in ("passes_quality_gate", "passes", "quality", "gate", "calculatepsi",
                  "calculate", "psi", "httpserver", "http", "server"):
            assert w in t

    def test_drops_stopwords_and_single_chars(self):
        assert tokens("what is the x of a") == []


class TestSearch:
    @pytest.mark.parametrize("query", ["quality gate accuracy threshold", "quality gate",
                                       "passes_quality_gate", "how is the quality gate defined"])
    def test_definition_outranks_mentions(self, repo_registry, query):
        # tests/test_train.py only *calls* passes_quality_gate; src/train.py defines it.
        out = rs.search(repo_registry, Evidence(), query)
        assert "src/train.py" in out.split("\n", 1)[0]

    def test_threshold_found_in_code_and_config(self, repo_registry):
        heads = [ln for ln in rs.search(repo_registry, Evidence(), "min accuracy").splitlines()
                 if ln.startswith("###")]
        assert {"src/train.py", "config.yaml"} <= {h.split()[1] for h in heads[:2]}

    def test_defined_tokens(self):
        toks = rs._defined_tokens("def passes_quality_gate(x):\nMIN_ACCURACY = 0.8\nclass PdModel:\n"
                                  "CREATE OR REPLACE VIEW risk.loans AS\nresult = fit()\n")
        assert {"passes_quality_gate", "min_accuracy", "pdmodel", "loans"} <= toks
        assert "result" not in toks          # an ordinary assignment is not a definition
        assert {"calculate", "psi"} <= rs._defined_tokens("def calculatePsi(e, a):")

    def test_definition_score_prefers_coverage(self):
        q = {"quality", "gate"}
        full = rs._definition_score(q, rs._defined_names("def passes_quality_gate(): pass"))
        partial = rs._definition_score(q, rs._defined_names("def test_gate(): pass"))
        assert full > partial > 0

    def test_snippets_are_real_lines(self, repo_registry):
        out = rs.search(repo_registry, Evidence(), "random forest seed")
        text = repo_registry.sources["model_repo"].files["src/train.py"].splitlines()
        for n, line in _numbered_lines(out.split("###")[1]).items():
            assert text[n - 1] == line

    def test_records_evidence(self, repo_registry):
        ev = Evidence()
        rs.search(repo_registry, ev, "passes_quality_gate")
        assert any(f == "src/train.py" for _, f in ev.files())

    def test_no_match(self, repo_registry):
        assert rs.search(repo_registry, Evidence(), "kubernetes helm chart").startswith("No matches")

    def test_stopword_only_query(self, repo_registry):
        assert "no searchable words" in rs.search(repo_registry, Evidence(), "what is the")

    def test_nothing_loaded(self):
        with pytest.raises(KeyError, match="No repository or documents"):
            rs.search(SourceRegistry(), Evidence(), "anything")

    def test_file_name_boost(self, repo_registry):
        out = rs.search(repo_registry, Evidence(), "utils")
        assert out.split("\n", 1)[0].count("src/utils.py") == 1


class TestGrep:
    def test_literal_with_line_numbers(self, repo_registry):
        ev = Evidence()
        out = rs.grep(repo_registry, ev, "MIN_ACCURACY = 0.85")
        assert "1 matching lines in 1 files" in out
        lines = _numbered_lines(out)
        assert lines[4] == "MIN_ACCURACY = 0.85"
        assert 4 in ev.lines("model_repo", "src/train.py")

    def test_literal_escapes_regex_chars(self, repo_registry):
        assert "matching lines" in rs.grep(repo_registry, Evidence(), 'metrics["accuracy"]')

    def test_regex(self, repo_registry):
        out = rs.grep(repo_registry, Evidence(), r"def \w+_model", regex=True)
        assert "train_model" in out

    def test_invalid_regex(self, repo_registry):
        assert rs.grep(repo_registry, Evidence(), "(unclosed", regex=True).startswith("Invalid regex")

    def test_case_sensitivity(self, repo_registry):
        assert "matching" in rs.grep(repo_registry, Evidence(), "min_accuracy")
        assert rs.grep(repo_registry, Evidence(), "MIN_accuracy", case_sensitive=True).startswith("No lines")

    def test_path_glob(self, repo_registry):
        out = rs.grep(repo_registry, Evidence(), "quality", path_glob="tests/*")
        assert "tests/test_train.py" in out and "src/train.py" not in out

    def test_max_hits(self, tmp_path):
        d = tmp_path / "many"
        d.mkdir()
        (d / "f.py").write_text("hit = 1\n" * 100)
        reg = SourceRegistry()
        reg.load(d)
        out = rs.grep(reg, Evidence(), "hit", max_hits=10)
        assert "10 matching lines" in out and "stopped at 10 matches" in out


class TestReadFile:
    def test_range(self, repo_registry):
        ev = Evidence()
        out = rs.read_file(repo_registry, ev, "src/train.py", 7, 8)
        assert "(lines 7-8 of" in out
        assert _numbered_lines(out) == {7: "def passes_quality_gate(metrics, min_accuracy=MIN_ACCURACY):",
                                        8: '    return metrics["accuracy"] >= min_accuracy'}
        assert ev.lines("model_repo", "src/train.py") == {7, 8}

    def test_whole_small_file(self, repo_registry):
        out = rs.read_file(repo_registry, Evidence(), "README.md")
        assert "(lines 1-3 of 3)" in out and "more lines" not in out

    def test_long_file_is_paged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "MAX_READ_LINES", 50)
        d = tmp_path / "big"
        d.mkdir()
        (d / "long.py").write_text("".join(f"line_{i} = {i}\n" for i in range(1, 201)))
        reg = SourceRegistry()
        reg.load(d)
        out = rs.read_file(reg, Evidence(), "long.py", 10)
        assert "(lines 10-59 of 200)" in out and "start_line=60" in out

    def test_start_beyond_end(self, repo_registry):
        assert "has only" in rs.read_file(repo_registry, Evidence(), "README.md", 99)

    def test_unknown_file(self, repo_registry):
        with pytest.raises(KeyError):
            rs.read_file(repo_registry, Evidence(), "nope.py")


class TestListAndOverview:
    def test_list_files_glob(self, repo_registry):
        out = rs.list_files(repo_registry, glob="*.py")
        assert "src/train.py" in out and "README.md" not in out

    def test_overview_single_repo(self, repo_registry):
        ev = Evidence()
        out = rs.overview(repo_registry, ev)
        assert "contents: src/ (2 files), tests/ (1 files)" in out
        assert "README.md (lines 1-3 of 3)" in out
        assert ev.lines("model_repo", "README.md") == {1, 2, 3}

    def test_overview_workspace_covers_every_project(self, workspace):
        reg = SourceRegistry()
        reg.load(workspace)
        ev = Evidence()
        out = rs.overview(reg, ev)
        assert "WORKSPACE of 3 separate projects" in out
        for name, phrase in [("alpha", "PD scores"), ("beta", "lineage graphs"), ("gamma", "RAG arena")]:
            assert f"### project {name}" in out and phrase in out
            assert ev.lines("workspace", f"{name}/README.md")


class TestEvidence:
    def test_accumulates(self):
        ev = Evidence()
        ev.add("s", "f.py", 1, 3)
        ev.add("s", "f.py", 10, 10)
        assert ev.lines("s", "f.py") == {1, 2, 3, 10}
        assert ev.lines("s", "other.py") == set()
        assert ev.files() == {("s", "f.py")}

    def test_prefix_only_with_several_multi_file_sources(self, sample_repo, workspace):
        reg = SourceRegistry()
        reg.load(sample_repo)
        assert rs.search(reg, Evidence(), "quality gate").startswith("### src/")
        reg.load(workspace)
        assert rs.search(reg, Evidence(), "quality gate").startswith("### model_repo:src/")
