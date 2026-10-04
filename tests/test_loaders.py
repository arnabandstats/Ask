"""Reading repos, documents and tables; workspace detection; caching."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ask import config
from ask.sources import loaders
from ask.sources.loaders import LoadError, detect_projects, load_path


class TestRepoLoading:
    def test_repo_files_and_kind(self, sample_repo):
        [src] = load_path(sample_repo)
        assert src.kind == "repo"
        assert {"src/train.py", "src/utils.py", "tests/test_train.py", "README.md",
                "config.yaml", "explore.ipynb"} <= set(src.files)

    def test_skips_cache_dirs_and_binaries(self, sample_repo):
        [src] = load_path(sample_repo)
        assert not any("__pycache__" in f for f in src.files)
        assert "image.bin" not in src.files

    def test_text_is_exact(self, sample_repo):
        [src] = load_path(sample_repo)
        assert src.files["src/train.py"] == (sample_repo / "src" / "train.py").read_text()

    def test_notebook_cells_flattened(self, sample_repo):
        [src] = load_path(sample_repo)
        nb = src.files["explore.ipynb"]
        assert "# %% [cell 1] markdown" in nb and "# %% [cell 2] code" in nb
        assert "print(x)" in nb and "# [output]" in nb

    def test_summary_mentions_counts(self, sample_repo):
        [src] = load_path(sample_repo)
        assert f"{len(src.files)} files" in src.summary()

    def test_hint_overrides_kind(self, sample_repo):
        [src] = load_path(sample_repo, hint="docs")
        assert src.kind == "docs"

    def test_single_code_file(self, sample_repo):
        [src] = load_path(sample_repo / "src" / "train.py")
        assert src.kind == "repo" and list(src.files) == ["train.py"]
        assert src.path == str(sample_repo / "src" / "train.py")

    def test_skipped_files_recorded(self, tmp_path, monkeypatch):
        repo = tmp_path / "r"
        repo.mkdir()
        (repo / "ok.py").write_text("a = 1\n")
        (repo / "big.py").write_text("x" * 200)
        monkeypatch.setattr(config, "MAX_FILE_BYTES", 100)
        [src] = load_path(repo)
        assert "ok.py" in src.files and "big.py" not in src.files
        assert any(s.startswith("big.py") for s in src.skipped)


class TestDocuments:
    def test_docx_headings_and_tables(self, docx_file):
        [src] = load_path(docx_file)
        text = src.files["policy.docx"]
        assert src.kind == "docs"
        assert text.splitlines() == ["# Model Policy", "Models must be validated annually.",
                                     "## Scope", "| Tier | Frequency |", "| 1 | 12 months |"]

    def test_markdown_is_docs(self, tmp_path):
        p = tmp_path / "notes.md"
        p.write_text("# Notes\nhello\n")
        [src] = load_path(p)
        assert src.kind == "docs" and src.summary().startswith("MD")

    def test_docx_text_cached_on_disk(self, docx_file, isolated_data_dir, monkeypatch):
        load_path(docx_file)
        cached = list((isolated_data_dir / "text_cache").glob("*.txt"))
        assert len(cached) == 1
        loaders._TEXT_CACHE.clear()
        monkeypatch.setattr(loaders, "_read_docx", lambda p: pytest.fail("should use disk cache"))
        [src] = load_path(docx_file)
        assert "Model Policy" in src.files["policy.docx"]

    def test_cache_invalidated_when_file_changes(self, tmp_path):
        p = tmp_path / "a.py"
        p.write_text("v = 1\n")
        assert load_path(p)[0].files["a.py"] == "v = 1\n"
        p.write_text("v = 2  # changed, different size\n")
        assert "v = 2" in load_path(p)[0].files["a.py"]

    def test_pdf_without_text_raises(self, tmp_path, monkeypatch):
        p = tmp_path / "scan.pdf"
        p.write_bytes(b"%PDF-1.4 fake")

        class _Page:
            def extract_text(self):
                return ""

        class _Reader:
            def __init__(self, path):
                self.pages = [_Page(), _Page()]

        import pypdf
        monkeypatch.setattr(pypdf, "PdfReader", _Reader)
        with pytest.raises(LoadError, match="no extractable text"):
            load_path(p)

    def test_pdf_pages_are_marked(self, tmp_path, monkeypatch):
        p = tmp_path / "guide.pdf"
        p.write_bytes(b"%PDF-1.4 fake")

        class _Page:
            def __init__(self, t):
                self.t = t

            def extract_text(self):
                return self.t

        class _Reader:
            def __init__(self, path):
                self.pages = [_Page("Intro text"), _Page("Validation must be independent.")]

        import pypdf
        monkeypatch.setattr(pypdf, "PdfReader", _Reader)
        [src] = load_path(p)
        text = src.files["guide.pdf"]
        assert text.startswith("[page 1]") and "[page 2]\nValidation must be independent." in text
        assert src.summary() == "PDF, 2 pages"


class TestData:
    def test_csv(self, portfolio_csv, portfolio_df):
        [src] = load_path(portfolio_csv)
        assert src.kind == "data" and src.df.shape == portfolio_df.shape
        assert src.summary() == f"{len(portfolio_df):,} rows × {portfolio_df.shape[1]} columns"

    def test_excel_all_sheets(self, tmp_path, portfolio_df):
        p = tmp_path / "book.xlsx"
        with pd.ExcelWriter(p) as w:
            portfolio_df.to_excel(w, sheet_name="dev", index=False)
            portfolio_df.head(10).to_excel(w, sheet_name="oot", index=False)
        [src] = load_path(p)
        assert list(src.sheets) == ["dev", "oot"] and len(src.df) == len(portfolio_df)
        assert "2 sheets" in src.summary()

    @pytest.mark.parametrize("ext", [".parquet", ".pkl", ".tsv"])
    def test_other_formats(self, tmp_path, portfolio_df, ext):
        p = tmp_path / f"t{ext}"
        if ext == ".parquet":
            portfolio_df.to_parquet(p)
        elif ext == ".pkl":
            portfolio_df.to_pickle(p)
        else:
            portfolio_df.to_csv(p, sep="\t", index=False)
        [src] = load_path(p)
        assert src.df.shape == portfolio_df.shape

    def test_pickle_of_non_table(self, tmp_path):
        p = tmp_path / "obj.pkl"
        pd.to_pickle({"not": "a table"}, p)
        with pytest.raises(LoadError, match="not a table"):
            load_path(p)

    def test_folder_of_data_files(self, tmp_path, portfolio_df):
        d = tmp_path / "data"
        d.mkdir()
        portfolio_df.to_csv(d / "a.csv", index=False)
        portfolio_df.to_csv(d / "b.csv", index=False)
        srcs = load_path(d)
        assert [s.name for s in srcs] == ["a.csv", "b.csv"] and all(s.kind == "data" for s in srcs)

    def test_corrupt_csv_gives_readable_error(self, tmp_path):
        p = tmp_path / "bad.xlsx"
        p.write_text("this is not excel")
        with pytest.raises(LoadError, match="Couldn't read bad.xlsx"):
            load_path(p)


class TestErrors:
    def test_missing_path(self, tmp_path):
        with pytest.raises(LoadError, match="Path not found"):
            load_path(tmp_path / "nope")

    def test_unsupported_file(self, tmp_path):
        p = tmp_path / "movie.mp4"
        p.write_bytes(b"\x00\x00")
        with pytest.raises(LoadError, match="Unsupported file type"):
            load_path(p)

    def test_empty_folder(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(LoadError, match="No readable"):
            load_path(d)

    def test_unreadable_single_file(self, tmp_path):
        p = tmp_path / "blob.py"
        p.write_bytes(b"\x00\x00\x00binary")
        with pytest.raises(LoadError, match="binary"):
            load_path(p)


class TestWorkspace:
    def test_detects_projects(self, workspace):
        assert detect_projects(workspace) == ["alpha", "beta", "gamma"]

    def test_workspace_summary(self, workspace):
        [src] = load_path(workspace)
        assert src.projects == ["alpha", "beta", "gamma"]
        assert src.summary().startswith("workspace with 3 projects (alpha, beta, gamma)")

    def test_single_repo_is_not_a_workspace(self, sample_repo):
        assert detect_projects(sample_repo) == []

    def test_git_root_is_one_project(self, workspace):
        (workspace / ".git").mkdir()
        assert detect_projects(workspace) == []

    def test_nested_projects_two_levels_down(self, tmp_path):
        base = tmp_path / "ws"
        for n in ("one", "two"):
            d = base / "group" / n
            d.mkdir(parents=True)
            (d / "pyproject.toml").write_text("[project]\n")
            (d / "m.py").write_text("x=1\n")
        assert detect_projects(base) == ["group/one", "group/two"]

    def test_record_roundtrip(self, workspace):
        [src] = load_path(workspace)
        assert src.record() == {"name": "workspace", "kind": "repo", "path": str(workspace)}
        assert json.loads(json.dumps(src.record()))["path"] == str(workspace)
