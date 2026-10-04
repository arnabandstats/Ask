"""The per-chat source registry: loading, naming, lookup, file resolution, restore."""
from __future__ import annotations

import pytest

from ask.sources.registry import SourceRegistry


class TestLoadAndNames:
    def test_reloading_same_path_replaces(self, sample_repo):
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(sample_repo)
        assert list(reg.sources) == ["model_repo"]

    def test_same_name_different_path_gets_suffix(self, tmp_path):
        for parent in ("a", "b"):
            d = tmp_path / parent / "proj"
            d.mkdir(parents=True)
            (d / "m.py").write_text("x = 1\n")
        reg = SourceRegistry()
        reg.load(tmp_path / "a" / "proj")
        reg.load(tmp_path / "b" / "proj")
        assert list(reg.sources) == ["proj", "proj (2)"]

    def test_remove(self, repo_registry):
        assert repo_registry.remove("model_repo")
        assert not repo_registry.remove("model_repo")
        assert repo_registry.describe() == "Nothing is loaded yet."

    def test_records_and_restore(self, sample_repo, portfolio_csv):
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(portfolio_csv)
        recs = reg.records()
        fresh = SourceRegistry()
        assert fresh.restore(recs) == []
        assert list(fresh.sources) == ["model_repo", "portfolio.csv"]
        assert fresh.sources["portfolio.csv"].df.shape == reg.sources["portfolio.csv"].df.shape

    def test_restore_reports_missing_paths(self, tmp_path):
        errs = SourceRegistry().restore([{"name": "gone", "kind": "repo", "path": str(tmp_path / "x")}])
        assert len(errs) == 1 and errs[0].startswith("gone:") and "Path not found" in errs[0]


class TestGet:
    def test_default_is_most_recent_of_kind(self, sample_repo, portfolio_csv, tmp_path, portfolio_df):
        reg = SourceRegistry()
        reg.load(portfolio_csv)
        p2 = tmp_path / "second.csv"
        portfolio_df.to_csv(p2, index=False)
        reg.load(p2)
        reg.load(sample_repo)
        assert reg.get(None, "data").name == "second.csv"
        assert reg.get(None, "repo").name == "model_repo"

    def test_fuzzy_name(self, repo_registry):
        assert repo_registry.get("MODEL_REPO").name == "model_repo"
        assert repo_registry.get("model").name == "model_repo"

    def test_unknown_name_lists_loaded(self, repo_registry):
        with pytest.raises(KeyError, match="Loaded: model_repo"):
            repo_registry.get("nothing-like-it")

    def test_nothing_loaded(self):
        with pytest.raises(KeyError, match="No data source is loaded"):
            SourceRegistry().get(None, "data")

    def test_kind_filter(self, repo_registry):
        with pytest.raises(KeyError):
            repo_registry.get("model_repo", "data")


class TestResolveFile:
    @pytest.mark.parametrize("ref", ["src/train.py", "./src/train.py", "SRC/TRAIN.PY",
                                     "train.py", "model_repo:src/train.py",
                                     "model_repo::src/train.py", "model_repo/src/train.py",
                                     "src\\train.py", "`src/train.py`"])
    def test_variants(self, repo_registry, ref):
        src, rel = repo_registry.resolve_file(ref)
        assert (src.name, rel) == ("model_repo", "src/train.py")

    def test_missing_file(self, repo_registry):
        with pytest.raises(KeyError, match="not in any loaded"):
            repo_registry.resolve_file("src/nope.py")

    def test_ambiguous_across_sources(self, tmp_path):
        for n in ("one", "two"):
            d = tmp_path / n
            d.mkdir()
            (d / "main.py").write_text(f"name = '{n}'\n")
            (d / "other.py").write_text("x = 1\n")
        reg = SourceRegistry()
        reg.load(tmp_path / "one")
        reg.load(tmp_path / "two")
        with pytest.raises(KeyError, match="ambiguous"):
            reg.resolve_file("main.py")
        assert reg.resolve_file("two:main.py")[0].name == "two"

    def test_ambiguous_basename_within_source(self, tmp_path):
        d = tmp_path / "r"
        (d / "a").mkdir(parents=True)
        (d / "b").mkdir()
        (d / "a" / "util.py").write_text("a = 1\n")
        (d / "b" / "util.py").write_text("b = 1\n")
        reg = SourceRegistry()
        reg.load(d)
        with pytest.raises(KeyError):
            reg.resolve_file("util.py")
        assert reg.resolve_file("b/util.py")[1] == "b/util.py"

    def test_data_sources_are_not_files(self, portfolio_csv):
        reg = SourceRegistry()
        reg.load(portfolio_csv)
        with pytest.raises(KeyError):
            reg.resolve_file("portfolio.csv")


def test_describe_lists_columns(portfolio_csv, sample_repo):
    reg = SourceRegistry()
    reg.load(portfolio_csv)
    reg.load(sample_repo)
    d = reg.describe()
    assert "[data] portfolio.csv" in d and "columns: x1, x2, segment, default_flag" in d
    assert "[repo] model_repo" in d
