"""Find file-system paths inside a chat message.

Users type paths in many shapes: quoted, in brackets, Windows paths with
spaces, git-bash style (/c/Users/...), relative. We collect candidates and,
for unquoted ones, trim words off the end until the path exists, which is how
"load C:/My Projects/model x and summarise it" resolves to "C:/My Projects/model x".
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

_WRAPPED = re.compile(r'"([^"\n]+)"|\'([^\'\n]+)\'|`([^`\n]+)`|\[([^\[\]\n]+)\]|<([^<>\n]+)>')
_PATH_START = re.compile(r'(?<![\w/\\])(?:[A-Za-z]:[\\/]|\\\\|~[\\/]|\.{1,2}[\\/]|/)')
_BARE_FILE = re.compile(r'(?<![\w/\\:])([\w.\-]+\.[A-Za-z0-9]{1,6})\b')
_TRAILING = '.,;:!?)]}>"\''

_LOAD_VERBS = re.compile(
    r"\b(load|open|read|ingest|add|use|import|attach|index|analy[sz]e|look at|take)\b", re.I)
_FILLER = re.compile(
    r"\b(please|can|could|you|this|that|the|a|an|my|our|from|at|in|on|of|for|me|to|"
    r"repo|repository|codebase|code|folder|directory|dir|project|file|files|document|"
    r"documents|docs|doc|data|dataset|datasets|table|csv|excel|path|here|it|is|and|"
    r"now|also|load|open|read|ingest|add|use|import|attach|index|look|at|take)\b", re.I)


@dataclass
class PathMention:
    raw: str
    path: Path | None    # resolved, existing path; None if it does not exist


def _normalise(raw: str) -> str:
    s = raw.strip().strip(_TRAILING).strip()
    m = re.match(r"^/([a-zA-Z])/(.*)$", s)          # git-bash /c/Users -> C:/Users
    if m and os.name == "nt":
        s = f"{m.group(1).upper()}:/{m.group(2)}"
    if s.startswith("file://"):
        s = s[7:]
    return os.path.expanduser(s)


def _exists(s: str) -> Path | None:
    if not s:
        return None
    try:
        p = Path(s)
        return p.resolve() if p.exists() else None
    except (OSError, ValueError):
        return None


def _shrink_to_existing(text: str) -> tuple[str, Path | None]:
    """Try the longest prefix of `text` (cut at whitespace) that exists."""
    words = text.split()
    for n in range(len(words), 0, -1):
        cand = " ".join(words[:n]).rstrip(_TRAILING)
        found = _exists(_normalise(cand))
        if found:
            return cand, found
    first = words[0].rstrip(_TRAILING) if words else text
    return first, None


def find_paths(text: str) -> list[PathMention]:
    """All path-like mentions in `text`, existing or not (in order, de-duplicated)."""
    out: list[PathMention] = []
    seen: set[str] = set()
    consumed: list[tuple[int, int]] = []

    def add(raw: str, path: Path | None) -> None:
        key = str(path) if path else raw
        if key not in seen:
            seen.add(key)
            out.append(PathMention(raw, path))

    for m in _WRAPPED.finditer(text):
        inner = next(g for g in m.groups() if g is not None)
        p = _exists(_normalise(inner))
        if p or _PATH_START.match(inner.strip()):
            add(inner.strip(), p)
            consumed.append(m.span())

    def free(i: int) -> bool:
        return not any(a <= i < b for a, b in consumed)

    for line in text.splitlines():
        offset = text.find(line)
        for m in _PATH_START.finditer(line):
            if not free(offset + m.start()):
                continue
            raw, p = _shrink_to_existing(line[m.start():])
            # "/" alone or a lone word after a slash is almost never a path.
            if p is None and len(raw) < 3:
                continue
            add(raw, p)
            consumed.append((offset + m.start(), offset + m.start() + len(raw)))

    for m in _BARE_FILE.finditer(text):
        if free(m.start()):
            p = _exists(m.group(1))
            if p and p.is_file():
                add(m.group(1), p)
    return out


def is_pure_load_command(text: str, mentions: list[PathMention]) -> bool:
    """True when the message only asks to load paths (no question to answer)."""
    if not mentions:
        return False
    rest = text
    for pm in mentions:
        rest = rest.replace(pm.raw, " ")
    rest = re.sub(r"[\"'`\[\]<>:,.;!()]", " ", rest)
    if "?" in rest:
        return False
    leftover = _FILLER.sub(" ", rest).split()
    has_verb = bool(_LOAD_VERBS.search(text)) or not rest.strip()
    return has_verb and len(leftover) <= 2


def kind_hint(text: str) -> str | None:
    """Optional source kind the user named ('repo', 'docs' or 'data')."""
    t = text.lower()
    if re.search(r"\b(repo|repository|codebase|code base|source code|project)\b", t):
        return "repo"
    if re.search(r"\b(data|dataset|csv|excel|spreadsheet|table|parquet)\b", t):
        return "data"
    if re.search(r"\b(doc|docs|document|documents|pdf|policy|paper|report|word)\b", t):
        return "docs"
    return None
