"""Read repositories, documents and data files into memory.

No embeddings, no index build: a repo or document folder becomes a dict of
{relative path: text}. Text is cached per file on (path, mtime, size), so
reloading a chat or a folder only re-reads files that changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from ask import config
from ask.sources import databricks

CODE_EXTS = {
    ".py", ".ipynb", ".r", ".sql", ".sas", ".scala", ".java", ".js", ".ts", ".tsx",
    ".jsx", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".m", ".jl", ".sh",
    ".ps1", ".bat", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".xml",
    ".html", ".css", ".dockerfile", ".tf", ".kt", ".swift", ".rb", ".php", ".lua",
}
DOC_EXTS = {".pdf", ".docx", ".md", ".txt", ".rst", ".tex", ".rtf", ".htm"}
DATA_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pkl", ".pickle", ".feather"}
NAMED_TEXT_FILES = {"dockerfile", "makefile", "readme", "license", "requirements.txt", ".env.example"}

SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "dist",
    "build", ".idea", ".vscode", "site-packages", ".tox", ".eggs", "ask_data",
    "validator_cache", "validator_output",
}


class LoadError(Exception):
    """A path could not be loaded; the message is shown to the user as-is."""


@dataclass
class Source:
    name: str
    kind: str                      # "repo" | "docs" | "data"
    path: str
    files: dict[str, str] = field(default_factory=dict)       # repo/docs: relpath -> text
    df: pd.DataFrame | None = None                            # data: active table
    sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)           # unreadable files + reason
    projects: list[str] = field(default_factory=list)          # sub-project folders (relpaths)
    loaded_at: float = field(default_factory=time.time)
    _index: Any = None                                          # keyword index, built lazily

    def summary(self) -> str:
        if self.kind == "data":
            r, c = self.df.shape if self.df is not None else (0, 0)
            extra = f", {len(self.sheets)} sheets" if len(self.sheets) > 1 else ""
            return f"{r:,} rows × {c} columns{extra}"
        if len(self.files) == 1:
            text = next(iter(self.files.values()))
            pages = text.count("\n[page ") + text.startswith("[page ")
            size = f"{pages} pages" if pages else f"{text.count(chr(10)) + 1:,} lines"
            return f"{Path(next(iter(self.files))).suffix.lstrip('.').upper() or 'text'}, {size}"
        exts: dict[str, int] = {}
        for rel in self.files:
            e = Path(rel).suffix.lower() or Path(rel).name.lower()
            exts[e] = exts.get(e, 0) + 1
        top = ", ".join(f"{e} {n}" for e, n in sorted(exts.items(), key=lambda x: -x[1])[:6])
        lines = sum(t.count("\n") + 1 for t in self.files.values())
        s = f"{len(self.files)} files, {lines:,} lines ({top})"
        if self.projects:
            s = f"workspace with {len(self.projects)} projects ({', '.join(self.projects)}); " + s
        if self.skipped:
            s += f"; {len(self.skipped)} skipped"
        return s

    def record(self) -> dict:
        return {"name": self.name, "kind": self.kind, "path": self.path}


# ── per-file text cache ────────────────────────────────────────────────────

_TEXT_CACHE: dict[tuple[str, float, int], str] = {}
_SLOW_EXTS = {".pdf", ".docx"}       # extracted text is also cached on disk


def _cached_text(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    disk = None
    if p.suffix.lower() in _SLOW_EXTS:
        digest = hashlib.sha1(repr(key).encode()).hexdigest()[:20]
        disk = config.DATA_DIR / "text_cache" / f"{digest}.txt"
        if disk.exists():
            _TEXT_CACHE[key] = disk.read_text(encoding="utf-8")
            return _TEXT_CACHE[key]
    text = read_file_text(p)
    _TEXT_CACHE[key] = text
    if disk is not None:
        disk.parent.mkdir(parents=True, exist_ok=True)
        disk.write_text(text, encoding="utf-8")
    return text


# ── readers ────────────────────────────────────────────────────────────────

def _decode(raw: bytes) -> str:
    if b"\x00" in raw[:8192]:
        raise LoadError("binary file")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return _normalise_newlines(raw.decode(enc))
        except UnicodeDecodeError:
            continue
    return _normalise_newlines(raw.decode("utf-8", errors="replace"))


def _normalise_newlines(text: str) -> str:
    """CRLF / CR -> LF, so line numbers and matching are identical on every OS."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_pdf(p: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(p))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            txt = page.extract_text() or ""
        except Exception as exc:  # one bad page should not lose the document
            txt = f"(page text could not be extracted: {exc})"
        pages.append(f"[page {i}]\n{txt.strip()}")
    text = "\n\n".join(pages)
    if not text.replace("[page", "").strip(" \n0123456789]"):
        raise LoadError("PDF has no extractable text (it may be a scanned image)")
    return text


def _read_docx(p: Path) -> str:
    import docx
    import docx.table
    import docx.text.paragraph
    d = docx.Document(str(p))
    out: list[str] = []
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = docx.text.paragraph.Paragraph(child, d)
            txt = para.text.strip()
            if not txt:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading"):
                level = "".join(ch for ch in style if ch.isdigit()) or "1"
                txt = "#" * min(int(level), 6) + " " + txt
            out.append(txt)
        elif tag == "tbl":
            table = docx.table.Table(child, d)
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _read_ipynb(p: Path) -> str:
    nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    cells = nb.get("cells", [])
    out: list[str] = []
    for i, cell in enumerate(cells, 1):
        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else str(src)
        out.append(f"# %% [cell {i}] {cell.get('cell_type', 'code')}")
        out.append(src.rstrip())
        texts = []
        for o in cell.get("outputs", []) or []:
            t = o.get("text") or (o.get("data") or {}).get("text/plain")
            if t:
                texts.append("".join(t) if isinstance(t, list) else str(t))
        if texts:
            joined = "\n".join(texts).strip()
            if len(joined) > 2000:
                joined = joined[:2000] + "\n…(output truncated)"
            out.append("# [output]\n" + "\n".join("# " + ln for ln in joined.splitlines()))
    return "\n".join(out)


def read_file_text(p: Path) -> str:
    ext = p.suffix.lower()
    if p.stat().st_size > config.MAX_FILE_BYTES and ext not in {".pdf", ".docx"}:
        raise LoadError(f"larger than {config.MAX_FILE_BYTES // 1_000_000} MB")
    if ext == ".pdf":
        return _read_pdf(p)
    if ext == ".docx":
        return _read_docx(p)
    if ext == ".ipynb":
        return _read_ipynb(p)
    return _decode(p.read_bytes())


def read_table(p: Path) -> dict[str, pd.DataFrame]:
    """{sheet name: DataFrame}. Non-Excel files return a single entry."""
    ext = p.suffix.lower()
    if ext == ".csv":
        return {p.stem: pd.read_csv(p, low_memory=False)}
    if ext == ".tsv":
        return {p.stem: pd.read_csv(p, sep="\t", low_memory=False)}
    if ext in {".xlsx", ".xls"}:
        sheets = pd.read_excel(p, sheet_name=None)
        if not sheets:
            raise LoadError("workbook has no sheets")
        return sheets
    if ext == ".parquet":
        return {p.stem: pd.read_parquet(p)}
    if ext == ".feather":
        return {p.stem: pd.read_feather(p)}
    if ext in {".pkl", ".pickle"}:
        obj = pd.read_pickle(p)
        if isinstance(obj, pd.Series):
            obj = obj.to_frame()
        if not isinstance(obj, pd.DataFrame):
            raise LoadError(f"pickle holds a {type(obj).__name__}, not a table")
        return {p.stem: obj}
    raise LoadError(f"unsupported data format '{ext}'")


# ── classification + loading ───────────────────────────────────────────────

def _is_text_candidate(p: Path) -> bool:
    ext = p.suffix.lower()
    return ext in CODE_EXTS or ext in DOC_EXTS or p.name.lower() in NAMED_TEXT_FILES


def _walk(base: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
        for n in sorted(names):
            files.append(Path(root) / n)
            if len(files) > config.MAX_REPO_FILES * 4:
                return files
    return files


def _load_text_folder(name: str, kind: str, base: Path, paths: list[Path]) -> Source:
    src = Source(name=name, kind=kind, path=str(base))
    for p in paths:
        if len(src.files) >= config.MAX_REPO_FILES:
            src.skipped.append(f"(stopped after {config.MAX_REPO_FILES} files)")
            break
        rel = p.relative_to(base).as_posix()
        try:
            text = _cached_text(p)
        except LoadError as exc:
            src.skipped.append(f"{rel}: {exc}")
            continue
        except Exception as exc:
            src.skipped.append(f"{rel}: {type(exc).__name__}: {exc}")
            continue
        if text.strip():
            src.files[rel] = text
    return src


def _load_data_file(name: str, p: Path) -> Source:
    try:
        sheets = read_table(p)
    except LoadError:
        raise
    except Exception as exc:
        raise LoadError(f"Couldn't read {p.name}: {type(exc).__name__}: {exc}") from exc
    first = next(iter(sheets.values()))
    return Source(name=name, kind="data", path=str(p), df=first, sheets=sheets)


def load_path(path: Path, hint: str | None = None) -> list[Source]:
    """Load one path. Returns one or more sources (a folder of data files
    gives one data source per file). Raises LoadError with a readable message."""
    path = Path(path)
    if not path.exists():
        raw = str(path).replace("\\", "/")
        if databricks.looks_like_path(raw):
            return _load_databricks(raw, hint)
        raise LoadError(f"Path not found: {path}")
    name = path.name or str(path)

    if path.is_file():
        ext = path.suffix.lower()
        if ext in DATA_EXTS:
            return [_load_data_file(name, path)]
        if ext in DOC_EXTS or ext in CODE_EXTS or path.name.lower() in NAMED_TEXT_FILES:
            kind = "docs" if ext in DOC_EXTS else "repo"
            if hint in {"repo", "docs"}:
                kind = hint
            src = _load_text_folder(name, kind, path.parent, [path])
            if not src.files:
                reason = src.skipped[0].split(": ", 1)[-1] if src.skipped else "file is empty"
                raise LoadError(f"Couldn't read {path.name}: {reason}")
            src.path = str(path)
            return [src]
        raise LoadError(f"Unsupported file type '{ext or path.name}'. "
                        "Supported: code/text files, PDF, DOCX, Markdown, CSV, Excel, Parquet.")

    try:
        all_files = _walk(path)
    except PermissionError as exc:
        raise LoadError(f"Permission denied reading {path}: {exc}") from exc
    code = [p for p in all_files if p.suffix.lower() in CODE_EXTS or p.name.lower() in NAMED_TEXT_FILES]
    docs = [p for p in all_files if p.suffix.lower() in DOC_EXTS]
    data = [p for p in all_files if p.suffix.lower() in DATA_EXTS]

    if hint == "data" or (not code and not docs and data):
        if not data:
            raise LoadError(f"No data files (CSV, Excel, Parquet…) found in {path}")
        if len(data) > 20:
            raise LoadError(f"{path} contains {len(data)} data files; load a specific file instead")
        return [_load_data_file(p.name, p) for p in data]

    if not code and not docs:
        raise LoadError(f"No readable code or document files found in {path}")
    kind = hint if hint in {"repo", "docs"} else ("repo" if len(code) >= max(1, len(docs) // 3) else "docs")
    # A repo keeps its documents too (README, specs, PDFs) so both can be cited.
    text_files = sorted(set(code) | set(docs), key=lambda p: str(p).lower())
    src = _load_text_folder(name, kind, path, text_files)
    if not src.files:
        raise LoadError(f"Every file in {path} was unreadable ({len(src.skipped)} skipped)")
    src.projects = detect_projects(path)
    return [src]


def _load_databricks(raw: str, hint: str | None) -> list[Source]:
    """A /Workspace or /Volumes path that isn't mounted here (Databricks App, laptop):
    mirror it through the Databricks API, load the copy, and keep the Databricks path."""
    remote = databricks.normalise(raw)
    try:
        local = databricks.fetch(remote)
    except databricks.DatabricksError as exc:
        raise LoadError(str(exc)) from exc
    srcs = load_path(local, hint)
    for s in srcs:
        if local.is_dir() and s.kind == "data":     # data files in a folder keep their own path/name
            s.path = remote + "/" + Path(s.path).relative_to(local).as_posix()
        else:
            s.name = PurePosixPath(remote).name or s.name
            s.path = remote
    return srcs


PROJECT_MARKERS = (".git", "pyproject.toml", "setup.py", "requirements.txt", "package.json",
                   "databricks.yml", "environment.yml", "pom.xml", "go.mod", "Cargo.toml")


def _is_project(d: Path) -> bool:
    if any((d / m).exists() for m in PROJECT_MARKERS):
        return True
    return any(p.name.lower().startswith("readme") for p in d.iterdir() if p.is_file())


def detect_projects(base: Path) -> list[str]:
    """Sub-folders (one or two levels down) that look like separate projects.
    Returns [] when the folder is a single project, i.e. fewer than two are found."""
    if (base / ".git").exists():
        return []                       # the folder itself is one repository
    found: list[str] = []
    try:
        children = sorted(d for d in base.iterdir() if d.is_dir()
                          and d.name not in SKIP_DIRS and not d.name.startswith("."))
        for d in children:
            if _is_project(d):
                found.append(d.name)
                continue
            for sub in sorted(s for s in d.iterdir() if s.is_dir()
                              and s.name not in SKIP_DIRS and not s.name.startswith(".")):
                if (sub / ".git").exists() or any((sub / m).exists() for m in PROJECT_MARKERS[1:]):
                    found.append(f"{d.name}/{sub.name}")
    except OSError:
        return []
    return found if len(found) >= 2 else []
