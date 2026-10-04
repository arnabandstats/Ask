"""Keyword retrieval over loaded repo/doc sources: grep, ranked search, file reads.

Every function that shows file content to the model also records the exact
lines it showed in an Evidence object. The faithfulness check later uses it
to confirm that each citation points at lines the model actually saw.
"""
from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ask import config
from ask.sources.loaders import Source
from ask.sources.registry import SourceRegistry

WINDOW = 40          # lines per search window
STRIDE = 30
_STOP = set("""a an and are as at be by for from has have how i in is it its of on or
that the this to was were what when where which who why will with does do did can
not no yes you your we our they them there their then than if else into about all
any each per via use used using code file files function functions repo document""".split())


@dataclass
class Evidence:
    """Lines shown to the model this turn: {(source, relpath): {line numbers}}."""
    seen: dict[tuple[str, str], set[int]] = field(default_factory=lambda: defaultdict(set))

    def add(self, src: str, rel: str, start: int, end: int) -> None:
        self.seen[(src, rel)].update(range(start, end + 1))

    def lines(self, src: str, rel: str) -> set[int]:
        return self.seen.get((src, rel), set())

    def files(self) -> set[tuple[str, str]]:
        return {k for k, v in self.seen.items() if v}


def _numbered(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines)))
    return "\n".join(f"{start + i:>{width}}| {ln}" for i, ln in enumerate(lines))


# ── tokenisation ───────────────────────────────────────────────────────────

_SPLIT = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def tokens(text: str) -> list[str]:
    out = []
    for w in _SPLIT.findall(text):
        lw = w.lower()
        if len(lw) > 1 and lw not in _STOP:
            out.append(lw)
        parts = [p.lower() for chunk in w.split("_") for p in _CAMEL.findall(chunk)]
        if len(parts) > 1:
            out.extend(p for p in parts if len(p) > 1 and p not in _STOP)
    return out


# ── lazy per-source BM25 index (plain dicts, built on first search) ────────

# Names DEFINED on a line (functions, classes, constants, SQL objects): a window that
# defines what you asked about should outrank one that merely mentions it.
_DEFINES = re.compile(
    r"^\s*(?i:(?:async\s+)?(?:def|class|function|func|fn|sub|procedure))\s+([A-Za-z_]\w*)"
    r"|^\s*([A-Z][A-Z0-9_]{2,})\s*[:=]"                                   # CONSTANT = ...
    r"|^\s*(?i:create\s+(?:or\s+replace\s+)?(?:table|view|function|procedure))\s+([\w.]+)",
    re.M)


def _defined_names(text: str) -> list[tuple[str, set[str]]]:
    """(whole identifier, its word parts) for each name defined in `text`."""
    out = []
    for m in _DEFINES.finditer(text):
        raw = next(g for g in m.groups() if g).rsplit(".", 1)[-1]
        name = raw.lower()
        parts = {t for t in tokens(raw) if t != name} or {name}   # split camelCase before lowering
        out.append((name, parts))
    return out


def _defined_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for name, parts in _defined_names(text):
        out |= parts | {name}
    return out


def _definition_score(q: set[str], names: list[tuple[str, set[str]]]) -> float:
    """Best match between the query and one defined name, weighted by how much of the
    name the query covers: for "quality gate", `passes_quality_gate` (2 of 3 words)
    beats `test_gate` (1 of 2). Naming the identifier exactly is a full match."""
    best = 0.0
    for name, parts in names:
        if name in q:
            best = max(best, float(len(parts)) + 1)
            continue
        hit = q & parts
        if hit:
            best = max(best, len(hit) * len(hit) / len(parts))
    return best


class _Index:
    def __init__(self, src: Source):
        self.windows: list[tuple[str, int, int]] = []     # (relpath, start, end) 1-based
        self.tfs: list[Counter] = []
        self.path_toks: list[set[str]] = []
        self.defs: list[list[set[str]]] = []
        df: Counter = Counter()
        for rel, text in src.files.items():
            lines = text.splitlines() or [""]
            ptoks = set(tokens(rel))
            for s in range(0, max(1, len(lines)), STRIDE):
                e = min(len(lines), s + WINDOW)
                chunk = "\n".join(lines[s:e])
                tf = Counter(tokens(chunk))
                self.windows.append((rel, s + 1, e))
                self.tfs.append(tf)
                self.path_toks.append(ptoks)
                self.defs.append(_defined_names(chunk))
                df.update(tf.keys())
                if e >= len(lines):
                    break
        n = max(1, len(self.windows))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.avg_len = sum(sum(tf.values()) for tf in self.tfs) / n if self.tfs else 1.0

    def score(self, q: list[str]) -> list[tuple[float, int]]:
        k1, b = 1.4, 0.75
        out = []
        qset = set(q)
        for i, tf in enumerate(self.tfs):
            dl = sum(tf.values()) or 1
            s = 0.0
            for t in qset:
                f = tf.get(t)
                if f:
                    s += self.idf.get(t, 0) * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self.avg_len))
            s += 1.5 * len(qset & self.path_toks[i])       # file name matches matter in code
            s += 3.0 * _definition_score(qset, self.defs[i])   # the definition beats a mention
            if s > 0:
                out.append((s, i))
        out.sort(reverse=True)
        return out


def _index(src: Source) -> _Index:
    if src._index is None:
        src._index = _Index(src)
    return src._index


# ── tools ──────────────────────────────────────────────────────────────────

def _targets(reg: SourceRegistry, source: str | None) -> list[Source]:
    if source:
        return [reg.get(source, "repo", "docs")]
    srcs = reg.of_kind("repo", "docs")
    if not srcs:
        raise KeyError("No repository or documents are loaded. Ask the user for a path to load.")
    return srcs


def _prefix(reg: SourceRegistry, src: Source) -> str:
    # A single-file source is already identified by its file name.
    if len(src.files) == 1 or len(reg.of_kind("repo", "docs")) == 1:
        return ""
    return f"{src.name}:"


def search(reg: SourceRegistry, ev: Evidence, query: str, source: str | None = None,
           top_k: int = 8) -> str:
    q = tokens(query)
    if not q:
        return "Query has no searchable words; try grep with an exact identifier."
    hits = []
    for src in _targets(reg, source):
        idx = _index(src)
        for score, i in idx.score(q)[: top_k * 2]:
            hits.append((score, src, idx.windows[i]))
    hits.sort(key=lambda h: -h[0])
    out, per_file = [], Counter()
    for score, src, (rel, s, e) in hits:
        if per_file[(src.name, rel)] >= 2 or len(out) >= top_k:
            continue
        per_file[(src.name, rel)] += 1
        lines = src.files[rel].splitlines()[s - 1:e]
        ev.add(src.name, rel, s, e)
        out.append(f"### {_prefix(reg, src)}{rel}  (lines {s}-{e}, score {score:.1f})\n"
                   + _numbered(lines, s))
    return "\n\n".join(out) if out else f"No matches for: {query}"


def grep(reg: SourceRegistry, ev: Evidence, pattern: str, regex: bool = False,
         source: str | None = None, path_glob: str | None = None,
         case_sensitive: bool = False, context: int = 2, max_hits: int = 60) -> str:
    flags = 0 if case_sensitive else re.I
    try:
        rx = re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as exc:
        return f"Invalid regex: {exc}"
    out, n_hits, n_files = [], 0, 0
    for src in _targets(reg, source):
        for rel, text in src.files.items():
            if path_glob and not (fnmatch.fnmatch(rel, path_glob) or fnmatch.fnmatch(rel.split("/")[-1], path_glob)):
                continue
            lines = text.splitlines()
            matched = [i for i, ln in enumerate(lines) if rx.search(ln)]
            if not matched:
                continue
            n_files += 1
            block, last = [], -10
            for i in matched:
                if n_hits >= max_hits:
                    break
                n_hits += 1
                s, e = max(0, i - context), min(len(lines), i + context + 1)
                if s <= last:
                    s = last + 1
                if s < e:
                    if block and s > last + 1:
                        block.append("   …")
                    block.append(_numbered(lines[s:e], s + 1))
                    ev.add(src.name, rel, s + 1, e)
                    last = e - 1
            out.append(f"### {_prefix(reg, src)}{rel}  ({len(matched)} matches)\n" + "\n".join(block))
            if n_hits >= max_hits:
                break
    if not out:
        return f"No lines match {pattern!r}."
    more = f"\n\n(stopped at {max_hits} matches; narrow the pattern or use path_glob)" if n_hits >= max_hits else ""
    return f"{n_hits} matching lines in {n_files} files.\n\n" + "\n\n".join(out) + more


def read_file(reg: SourceRegistry, ev: Evidence, file: str,
              start_line: int | None = None, end_line: int | None = None) -> str:
    src, rel = reg.resolve_file(file)
    lines = src.files[rel].splitlines()
    total = len(lines)
    s = max(1, int(start_line or 1))
    e = min(total, int(end_line or (s + config.MAX_READ_LINES - 1)))
    if e - s + 1 > config.MAX_READ_LINES:
        e = s + config.MAX_READ_LINES - 1
    if s > total:
        return f"{rel} has only {total} lines."
    ev.add(src.name, rel, s, e)
    tail = f"\n… ({total - e} more lines; call read_file with start_line={e + 1})" if e < total else ""
    return f"### {_prefix(reg, src)}{rel}  (lines {s}-{e} of {total})\n" + _numbered(lines[s - 1:e], s) + tail


def _readme(files: dict[str, str], prefix: str) -> str | None:
    cands = [r for r in files if r.startswith(prefix) and "/" not in r[len(prefix):]
             and r[len(prefix):].lower().startswith("readme")]
    return min(cands, key=len) if cands else None


def _structure(src: Source, prefix: str) -> list[str]:
    """Top-level entries under `prefix` with file counts."""
    dirs: Counter = Counter()
    files = []
    for rel in src.files:
        if not rel.startswith(prefix):
            continue
        rest = rel[len(prefix):]
        if "/" in rest:
            dirs[rest.split("/", 1)[0] + "/"] += 1
        else:
            files.append(rest)
    out = [f"{d} ({n} files)" for d, n in sorted(dirs.items())]
    out += sorted(files)[:25]
    return out


def overview(reg: SourceRegistry, ev: Evidence, source: str | None = None,
             readme_lines: int = 40) -> str:
    """Structure of each loaded repo/docs source — and of each project inside a
    workspace — with the opening lines of every README."""
    out = []
    for src in _targets(reg, source):
        units = [(p, p.rstrip("/") + "/") for p in src.projects] or [(src.name, "")]
        head = (f"## {src.name} [{src.kind}] — {src.summary()}")
        if src.projects:
            head += (f"\nThis folder is a WORKSPACE of {len(src.projects)} separate projects. "
                     "Describe each one; do not present one project as the whole.")
            loose = [r for r in src.files if not any(r.startswith(p + "/") for p in src.projects)]
            if loose:
                head += f"\nFiles outside the projects: {', '.join(loose[:15])}"
        out.append(head)
        for label, prefix in units:
            n = sum(1 for r in src.files if r.startswith(prefix))
            lines = sum(t.count("\n") + 1 for r, t in src.files.items() if r.startswith(prefix))
            block = [f"\n### {'project ' if src.projects else ''}{label}  ({n} files, {lines:,} lines)",
                     "contents: " + ", ".join(_structure(src, prefix))]
            rm = _readme(src.files, prefix)
            if rm:
                text = src.files[rm].splitlines()
                k = min(len(text), readme_lines)
                ev.add(src.name, rm, 1, k)
                block.append(f"{_prefix(reg, src)}{rm} (lines 1-{k} of {len(text)}):\n" + _numbered(text[:k], 1))
            else:
                block.append("(no README)")
            out.append("\n".join(block))
    return "\n".join(out)


def list_files(reg: SourceRegistry, source: str | None = None, glob: str | None = None,
               limit: int = 300) -> str:
    out = []
    for src in _targets(reg, source):
        rels = [r for r in src.files if not glob or fnmatch.fnmatch(r, glob)
                or fnmatch.fnmatch(r.split("/")[-1], glob)]
        out.append(f"## {src.name} [{src.kind}] — {len(rels)} files")
        for r in rels[:limit]:
            out.append(f"{r}  ({src.files[r].count(chr(10)) + 1} lines)")
        if len(rels) > limit:
            out.append(f"… {len(rels) - limit} more (use glob to narrow)")
        if src.skipped and not glob:
            out.append(f"(skipped {len(src.skipped)}: " + "; ".join(src.skipped[:5]) + ")")
    return "\n".join(out)
