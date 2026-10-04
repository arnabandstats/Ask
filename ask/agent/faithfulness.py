"""Deterministic citation check for repo/document answers.

No LLM judge. For each [path:Lx-y] citation we confirm that:
  - the file is in a loaded source,
  - the line range exists,
  - the model was actually shown those lines during this turn (Evidence),
and that `code identifiers` quoted near a citation really occur in the cited file.
Problems trigger one repair round in the router; the final report is shown to the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ask.retrieval.search import Evidence
from ask.sources.registry import SourceRegistry

# [path:L10-20]  also  [path:L10-20, L40-45]  and  [path:L10-20,L40]
_RANGE = r"L?\d+(?:\s*[-–]\s*L?\d+)?"
CITATION = re.compile(r"\[([^\[\]\n]{1,300}?):(L\d+(?:\s*[-–]\s*L?\d+)?(?:\s*,\s*" + _RANGE + r")*)\]")


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        nums = [int(n) for n in re.findall(r"\d+", part)]
        if nums:
            a, b = nums[0], nums[-1]
            out.append((min(a, b), max(a, b)))
    return out
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(\(\))?$")


@dataclass
class Citation:
    raw: str
    ref: str
    start: int
    end: int
    status: str = "ok"                  # ok | not_read | bad_range | unknown_file
    detail: str = ""
    source: str = ""
    file: str = ""
    snippet: str = ""


@dataclass
class Report:
    citations: list[Citation] = field(default_factory=list)
    identifier_issues: list[str] = field(default_factory=list)
    uncited: bool = False

    @property
    def problems(self) -> list[str]:
        out = [f"{c.raw}: {c.detail}" for c in self.citations if c.status != "ok"]
        out += self.identifier_issues
        if self.uncited:
            out.append("The answer has no citations, but a repo/documents are loaded. If the question "
                       "is about them, search and read the actual files and cite them. If it is "
                       "purely general knowledge unrelated to the loaded material, start the answer "
                       "with 'General knowledge:'.")
        return out

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_meta(self) -> dict:
        seen, items = set(), []
        for c in self.citations:
            key = (c.source, c.file, c.start, c.end)
            if key in seen:
                continue
            seen.add(key)
            items.append({"ref": c.raw, "status": c.status, "detail": c.detail,
                          "source": c.source, "file": c.file, "start": c.start,
                          "end": c.end, "snippet": c.snippet})
        return {"checked": True, "citations": items, "problems": self.problems}


def verify(answer: str, reg: SourceRegistry, ev: Evidence, grounded_turn: bool) -> Report:
    rep = Report()
    for m in CITATION.finditer(answer):
        for a, b in parse_ranges(m.group(2)):
            rep.citations.append(_check_one(m.group(0), m.group(1).strip(), a, b, reg, ev))

    _check_identifiers(answer, reg, ev, rep)
    text_claims = len(re.sub(r"```.*?```", "", answer, flags=re.S)) > 200
    general = answer.lstrip("*_# ").lower().startswith("general knowledge")
    rep.uncited = grounded_turn and not rep.citations and text_claims and not general
    return rep


def _check_one(raw: str, ref: str, a: int, b: int, reg: SourceRegistry, ev: Evidence) -> Citation:
    c = Citation(raw=raw, ref=ref, start=a, end=b)
    try:
        src, rel = reg.resolve_file(ref)
    except KeyError as exc:
        c.status, c.detail = "unknown_file", (exc.args[0] if exc.args else str(exc))
        return c
    c.source, c.file = src.name, rel
    lines = src.files[rel].splitlines()
    if a < 1 or b > len(lines):
        c.status, c.detail = "bad_range", f"{rel} has {len(lines)} lines"
        return c
    c.snippet = "\n".join(f"{i}| {lines[i - 1]}" for i in range(a, min(b, a + 11) + 1))
    seen = ev.lines(src.name, rel)
    span = set(range(a, b + 1))
    overlap = len(span & seen)
    if overlap == 0:
        c.status, c.detail = "not_read", "these lines were never shown to you in this turn"
    elif overlap < max(1, len(span) // 2):
        c.status, c.detail = "not_read", (f"only {overlap} of {len(span)} cited lines were "
                                          "shown to you; read them or narrow the range")
    return c


_VISUAL_CLAIM = re.compile(
    r"\b(chart|plot|graph|figure|histogram|visuali[sz]ation)s?\b[^.\n]{0,60}\b(is |are |has been |have been )?"
    r"(shown|displayed|below|above|attached|generated|created)\b|\bsee (the )?(chart|plot|graph|figure)", re.I)


def check_visual_claims(answer: str, artifacts: list[dict]) -> list[str]:
    """The answer must not claim a chart/figure the turn never produced."""
    has_visual = any(a.get("type") in {"plotly", "image"} for a in artifacts)
    claims = [m.group(0) for m in _VISUAL_CLAIM.finditer(answer)
              if not re.search(r"\b(not|cannot|can't|couldn't|n't|if you|want|would)\b", m.group(0), re.I)]
    if not has_visual and claims:
        return ["The answer says a chart/figure is shown, but no chart was created in this turn. "
                "Call make_chart (or the relevant test) to create it, or remove the claim."]
    return []


def _check_identifiers(answer: str, reg: SourceRegistry, ev: Evidence, rep: Report) -> None:
    """`identifiers` in a paragraph that cites files must appear in a cited or read file."""
    allowed = {s.name for s in reg.sources.values()}
    for s in reg.of_kind("data"):
        allowed.update(map(str, s.df.columns))
    for s in reg.of_kind("repo", "docs"):
        for rel in s.files:
            allowed.add(rel)
            allowed.add(rel.rsplit("/", 1)[-1])
    for para in re.split(r"\n\s*\n", re.sub(r"```.*?```", "", answer, flags=re.S)):
        cites = [c for c in rep.citations if c.raw in para and c.file]
        if not cites:
            continue
        pool = [reg.sources[c.source].files[c.file] for c in cites]
        pool += [reg.sources[s].files[f] for s, f in ev.files() if s in reg.sources and f in reg.sources[s].files]
        for span in re.findall(r"`([^`\n]{3,80})`", para):
            ident = span.strip()
            if not _IDENT.match(ident) or ident in allowed:
                continue
            bare = ident.removesuffix("()")
            last = bare.split(".")[-1]
            if not any(bare in text or last in text for text in pool):
                rep.identifier_issues.append(f"`{ident}` does not appear in the cited or read files")
