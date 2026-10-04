"""Comparisons: tables, files, trees, file-vs-folder."""
from __future__ import annotations

import pytest

from ask.analysis import compare as cmp
from ask.retrieval.search import Evidence
from ask.sources.registry import SourceRegistry


class TestTables:
    def test_schema_and_stats(self, portfolio_df):
        b = portfolio_df.drop(columns=["segment"]).assign(x1=portfolio_df["x1"] * 1.1)
        out = cmp.compare_tables(portfolio_df, b, "A", "B")
        assert "columns only in A: segment" in out
        assert "per-column comparison:" in out
        line = next(ln for ln in out.splitlines() if ln.strip().startswith("x1 "))
        assert "10.0" in line                                      # mean Δ% of +10%

    def test_identical_tables(self, portfolio_df):
        out = cmp.compare_tables(portfolio_df, portfolio_df.copy(), "A", "B")
        assert "row-aligned cell differences: 0 of" in out

    def test_cell_level_differences(self, portfolio_df):
        b = portfolio_df.copy()
        b.loc[0:4, "pd_score"] = -1
        out = cmp.compare_tables(portfolio_df, b, "A", "B")
        assert "row-aligned cell differences: 5 of" in out and "pd_score=5" in out

    def test_dtype_change(self, portfolio_df):
        b = portfolio_df.assign(default_flag=portfolio_df["default_flag"].astype(str))
        assert "dtype changes: default_flag: int64 -> " in cmp.compare_tables(portfolio_df, b, "A", "B")


class TestTexts:
    def _reg(self, tmp_path, a_text, b_text):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "f.py").write_text(a_text)
        (tmp_path / "b" / "f.py").write_text(b_text)
        reg = SourceRegistry()
        reg.load(tmp_path / "a" / "f.py")
        reg.load(tmp_path / "b" / "f.py")
        return reg

    def test_single_line_change(self, tmp_path):
        base = "".join(f"line {i}\n" for i in range(1, 21))
        reg = self._reg(tmp_path, base, base.replace("line 7\n", "line SEVEN\n"))
        a, b = list(reg.sources.values())
        ev = Evidence()
        out = cmp.compare_texts(ev, a, "f.py", b, "f.py")
        assert "1 changed blocks" in out and "- A7| line 7" in out and "+ B7| line SEVEN" in out
        assert ev.lines(a.name, "f.py") == {7} and ev.lines(b.name, "f.py") == {7}

    def test_identical(self, tmp_path):
        reg = self._reg(tmp_path, "x = 1\n", "x = 1\n")
        a, b = list(reg.sources.values())
        assert "identical" in cmp.compare_texts(Evidence(), a, "f.py", b, "f.py")


class TestDispatch:
    def test_two_data_sources(self, tmp_path, portfolio_df):
        portfolio_df.to_csv(tmp_path / "v1.csv", index=False)
        portfolio_df.head(10).to_csv(tmp_path / "v2.csv", index=False)
        reg = SourceRegistry()
        reg.load(tmp_path / "v1.csv")
        reg.load(tmp_path / "v2.csv")
        assert "B = v2.csv: 10 rows" in cmp.compare(reg, Evidence(), "v1.csv", "v2.csv")

    def test_two_folders(self, sample_repo, tmp_path):
        import shutil
        copy = tmp_path / "copy"
        shutil.copytree(sample_repo, copy)
        (copy / "src" / "train.py").write_text("changed\n")
        (copy / "NEW.md").write_text("# new\n")
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(copy)
        out = cmp.compare(reg, Evidence(), "model_repo", "copy")
        assert "changed: 1" in out and "only in B: 1" in out and "src/train.py" in out

    def test_document_vs_folder_finds_near_copy(self, sample_repo, tmp_path):
        v2 = tmp_path / "train_v2.py"
        v2.write_text((sample_repo / "src" / "train.py").read_text().replace("0.85", "0.80"))
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(v2)
        out = cmp.compare(reg, Evidence(), "train_v2.py", "model_repo")
        assert out.startswith("FILE = train_v2.py")
        ranked = [ln for ln in out.splitlines() if ln.startswith("  ")]
        assert ranked[0].endswith("src/train.py")
        assert "Exact diff with the closest match (src/train.py)" in out
        assert "MIN_ACCURACY = 0.80" in out and "MIN_ACCURACY = 0.85" in out

    def test_folder_vs_document_order_independent(self, sample_repo, tmp_path):
        v2 = tmp_path / "train_v2.py"
        v2.write_text((sample_repo / "src" / "train.py").read_text())
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(v2)
        assert cmp.compare(reg, Evidence(), "model_repo", "train_v2.py").startswith("FILE = train_v2.py")

    def test_unrelated_document_vs_folder(self, sample_repo, docx_file):
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(docx_file)
        out = cmp.compare(reg, Evidence(), "policy.docx", "model_repo")
        assert "No file in the folder is a near-copy" in out and "Exact diff" not in out

    def test_table_vs_repo_rejected(self, sample_repo, portfolio_csv):
        reg = SourceRegistry()
        reg.load(sample_repo)
        reg.load(portfolio_csv)
        with pytest.raises(ValueError):
            cmp.compare(reg, Evidence(), "portfolio.csv", "model_repo")

    def test_unknown_ref(self, repo_registry):
        with pytest.raises(KeyError):
            cmp.compare(repo_registry, Evidence(), "nothing.py", "model_repo")
