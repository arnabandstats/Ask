"""Path extraction from chat messages."""
from __future__ import annotations

import os

import pytest

from ask.sources.paths import find_paths, is_pure_load_command, kind_hint


@pytest.fixture
def spaced_dir(tmp_path):
    d = tmp_path / "My Projects" / "model x"
    d.mkdir(parents=True)
    (d / "a.py").write_text("x = 1\n")
    return d


def _resolved(msg):
    return [m.path for m in find_paths(msg) if m.path]


class TestFindPaths:
    def test_unquoted_path_with_spaces(self, spaced_dir):
        assert _resolved(f"load this repo: {spaced_dir}") == [spaced_dir.resolve()]

    def test_trailing_words_are_trimmed(self, spaced_dir):
        assert _resolved(f"read {spaced_dir} and tell me what it does") == [spaced_dir.resolve()]

    @pytest.mark.parametrize("wrap", ['"{}"', "'{}'", "`{}`", "[{}]", "<{}>"])
    def test_wrapped_paths(self, spaced_dir, wrap):
        assert _resolved("load " + wrap.format(spaced_dir)) == [spaced_dir.resolve()]

    def test_forward_slashes(self, spaced_dir):
        assert _resolved("load " + str(spaced_dir).replace("\\", "/")) == [spaced_dir.resolve()]

    @pytest.mark.skipif(os.name != "nt", reason="git-bash paths are a Windows convention")
    def test_git_bash_style_path(self, spaced_dir):
        p = str(spaced_dir).replace("\\", "/")
        gitbash = "/" + p[0].lower() + p[2:]
        assert _resolved(f'load "{gitbash}"') == [spaced_dir.resolve()]

    def test_trailing_punctuation(self, spaced_dir):
        assert _resolved(f"please load {spaced_dir}.") == [spaced_dir.resolve()]

    def test_multiple_paths(self, tmp_path):
        a, b = tmp_path / "a.csv", tmp_path / "b.csv"
        a.write_text("x\n1\n"), b.write_text("x\n2\n")
        assert _resolved(f'compare "{a}" and "{b}"') == [a.resolve(), b.resolve()]

    def test_missing_path_is_reported_unresolved(self):
        ms = find_paths("load C:/definitely/not/here")
        assert len(ms) == 1 and ms[0].path is None and "definitely" in ms[0].raw

    def test_bare_relative_file(self, tmp_path, monkeypatch):
        (tmp_path / "data.csv").write_text("x\n1\n")
        monkeypatch.chdir(tmp_path)
        assert _resolved("read data.csv please") == [(tmp_path / "data.csv").resolve()]

    def test_no_paths_in_plain_question(self):
        assert find_paths("what is a population stability index?") == []

    def test_duplicates_collapsed(self, spaced_dir):
        assert len(_resolved(f'load "{spaced_dir}" "{spaced_dir}"')) == 1


class TestPureLoadCommand:
    @pytest.mark.parametrize("msg", ["load {p}", "please load this repo: {p}", "read the data from {p}",
                                     "{p}", "open [{p}]", "use {p} now"])
    def test_pure(self, spaced_dir, msg):
        text = msg.format(p=spaced_dir)
        assert is_pure_load_command(text, find_paths(text))

    @pytest.mark.parametrize("msg", ["load {p} and explain the training loop",
                                     "what does {p} do?", "load {p} then run the tests on the model"])
    def test_with_question(self, spaced_dir, msg):
        text = msg.format(p=spaced_dir)
        assert not is_pure_load_command(text, find_paths(text))

    def test_no_mentions(self):
        assert not is_pure_load_command("load it", [])


class TestKindHint:
    @pytest.mark.parametrize("text,kind", [
        ("load this repo", "repo"), ("open the codebase", "repo"),
        ("read the data", "data"), ("load the csv", "data"),
        ("load the policy document", "docs"), ("read the pdf", "docs"),
        ("load it", None)])
    def test_hints(self, text, kind):
        assert kind_hint(text) == kind
