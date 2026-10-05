"""Documentation review support: regulatory checklists, keyword evidence, cross-checks.

The agent reviews model documentation against EXTERNALLY AVAILABLE standards
(CRR, EBA Guidelines, ECB Guide to internal models, IFRS 9, EU AI Act, AMLR/FATF,
SR 11-7 as a US benchmark) plus generic and GenAI checklists. Each standard is a
YAML file in `standards/` holding one verifiable statement per requirement.

Everything here is deterministic text search. `check()` finds lines that carry
the requirement's keywords and patterns and labels each requirement with its
KEYWORD EVIDENCE ("evidence found" / "weak evidence" / "no evidence found").
That label says whether the words are there, never whether the documentation
complies: the agent must read the cited lines ([file:Lx-y]) and judge adequacy.

Line numbers are 1-based positions in `text.splitlines()` of the text passed in,
so citations point exactly at the text the agent was given.
"""
from __future__ import annotations

import difflib
import functools
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ask.validation.core import MODEL_TYPES

STANDARDS_DIR = Path(__file__).with_name("standards")
STATUSES = ("evidence found", "weak evidence", "no evidence found")
_REQUIRED_STANDARD = ("id", "title", "version_note", "applies_to", "requirements")
_REQUIRED_REQUIREMENT = ("id", "topic", "requirement", "source", "keywords", "evidence_expected")


@dataclass(frozen=True)
class Requirement:
    id: str
    topic: str
    requirement: str
    source: str
    model_types: tuple = ()
    keywords: tuple = ()
    patterns: tuple = ()
    evidence_expected: str = ""


@dataclass(frozen=True)
class Standard:
    id: str
    title: str
    version_note: str
    applies_to: tuple
    requirements: tuple = ()


@dataclass
class Hit:
    file: str
    line: int
    text: str
    matched: str


@dataclass
class RequirementCheck:
    requirement: Requirement
    status: str
    hits: list = field(default_factory=list)
    coverage: float = 0.0


# ── loading ────────────────────────────────────────────────────────────────

def _tuple(v) -> tuple:
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,)
    return tuple(str(x) for x in v)


def _parse(raw: dict, origin: str) -> Standard:
    if not isinstance(raw, dict):
        raise ValueError(f"{origin}: top level must be a mapping")
    missing = [k for k in _REQUIRED_STANDARD if not raw.get(k)]
    if missing:
        raise ValueError(f"{origin}: missing {missing}")
    applies = _tuple(raw["applies_to"])
    unknown = [m for m in applies if m not in MODEL_TYPES]
    if unknown:
        raise ValueError(f"{origin}: unknown model types {unknown}")
    reqs, seen = [], set()
    for i, r in enumerate(raw["requirements"]):
        where = f"{origin} requirement #{i + 1}"
        if not isinstance(r, dict):
            raise ValueError(f"{where}: must be a mapping")
        missing = [k for k in _REQUIRED_REQUIREMENT if not r.get(k)]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        if r["id"] in seen:
            raise ValueError(f"{where}: duplicate id {r['id']}")
        seen.add(r["id"])
        mts = _tuple(r.get("model_types"))
        bad = [m for m in mts if m not in applies]
        if bad:
            raise ValueError(f"{where}: model_types {bad} not in the standard's applies_to")
        patterns = _tuple(r.get("patterns"))
        for p in patterns:
            re.compile(p)
        reqs.append(Requirement(str(r["id"]), str(r["topic"]).strip(), str(r["requirement"]).strip(),
                                str(r["source"]).strip(), mts, _tuple(r["keywords"]), patterns,
                                str(r["evidence_expected"]).strip()))
    return Standard(str(raw["id"]), str(raw["title"]).strip(), str(raw["version_note"]).strip(),
                    applies, tuple(reqs))


@functools.lru_cache(maxsize=1)
def _all_standards() -> tuple:
    import yaml
    out = []
    for path in sorted(STANDARDS_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as fh:
            out.append(_parse(yaml.safe_load(fh), path.name))
    ids = [s.id for s in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate standard ids {dup}")
    return tuple(sorted(out, key=lambda s: s.id))


def list_standards(model_type: str | None = None) -> list[Standard]:
    """All standards, or those whose `applies_to` includes `model_type`."""
    if model_type is not None and model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{model_type}'. Valid: {', '.join(MODEL_TYPES)}")
    return [s for s in _all_standards() if model_type is None or model_type in s.applies_to]


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def load_standard(id_or_title: str) -> Standard:
    """Find a standard by id or title: exact, then case/punctuation-insensitive, then a
    unique substring, then a unique close match. KeyError lists the options otherwise."""
    stds = _all_standards()
    for s in stds:
        if id_or_title == s.id:
            return s
    q = _norm(id_or_title)
    options = "; ".join(f"{s.id} ({s.title})" for s in stds)
    if not q:
        raise KeyError(f"Empty standard name. Options: {options}")
    exact = [s for s in stds if q in (_norm(s.id), _norm(s.title))]
    if len(exact) == 1:
        return exact[0]
    sub = [s for s in stds if q in _norm(s.id) or q in _norm(s.title)]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise KeyError(f"'{id_or_title}' is ambiguous: "
                       + "; ".join(f"{s.id} ({s.title})" for s in sub))
    names = {}
    for s in stds:
        names.setdefault(_norm(s.id), s)
        names.setdefault(_norm(s.title), s)
    close = {names[n].id: names[n] for n in difflib.get_close_matches(q, list(names), n=5, cutoff=0.6)}
    if len(close) == 1:
        return next(iter(close.values()))
    raise KeyError(f"No standard matches '{id_or_title}'. Options: {options}")


# ── keyword evidence ───────────────────────────────────────────────────────

def _keyword_regex(kw: str) -> re.Pattern:
    """Case-insensitive, whole-word, tolerant of space/hyphen/underscore variants
    ('back-testing' = 'backtesting' = 'back testing') and a plural s/es."""
    parts = [re.escape(p) for p in re.split(r"[\s\-_]+", kw.strip().lower()) if p]
    body = r"[\s\-_]*".join(parts)
    left = r"(?<![a-z0-9])" if re.match(r"[a-z0-9]", kw.strip().lower()) else ""
    right = r"(?:s|es)?(?![a-z0-9])" if re.search(r"[a-z0-9]$", kw.strip().lower()) else ""
    return re.compile(left + body + right, re.IGNORECASE)


def _terms(req: Requirement) -> list[tuple[str, re.Pattern]]:
    terms = [(k, _keyword_regex(k)) for k in dict.fromkeys(req.keywords)]
    terms += [(f"/{p}/", re.compile(p, re.IGNORECASE)) for p in dict.fromkeys(req.patterns)]
    return terms


def _applies(req: Requirement, model_type: str | None) -> bool:
    return model_type is None or not req.model_types or model_type in req.model_types


def check(files: dict[str, str], standard: Standard, model_type: str | None = None,
          max_hits: int = 3) -> list[RequirementCheck]:
    """Search the documents for each requirement's keywords/patterns.

    A line is a candidate hit when it contains at least one term. Candidates are
    ranked by the number of distinct terms in the line and its two neighbours
    (window ±1), ties broken by file then line; at most `max_hits` are kept, and no
    two kept hits are adjacent lines of the same file.

    Status (KEYWORD EVIDENCE, not compliance): "evidence found" when the best window
    holds at least two distinct terms (or the only term, when there is one);
    "weak evidence" when terms occur but never together; "no evidence found" otherwise.
    `coverage` = share of the requirement's terms found anywhere in the documents.
    Requirements narrowed to other model types are skipped when `model_type` is given.
    """
    if model_type is not None and model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{model_type}'")
    split = {name: str(text).splitlines() for name, text in sorted(files.items())}
    out = []
    for req in standard.requirements:
        if not _applies(req, model_type):
            continue
        terms = _terms(req)
        found_any: set[str] = set()
        candidates = []
        for name, lines in split.items():
            per_line = []
            for ln in lines:
                matched = [label for label, rx in terms if rx.search(ln)]
                per_line.append(matched)
                found_any.update(matched)
            for i, matched in enumerate(per_line):
                if not matched:
                    continue
                window = set()
                for j in (i - 1, i, i + 1):
                    if 0 <= j < len(per_line):
                        window.update(per_line[j])
                candidates.append((-len(window), name, i + 1, matched))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))
        hits, best = [], 0
        for neg, name, line, matched in candidates:
            if len(hits) >= max_hits:
                break
            if any(h.file == name and abs(h.line - line) <= 1 for h in hits):
                continue
            best = max(best, -neg)
            hits.append(Hit(name, line, split[name][line - 1].strip()[:400], ", ".join(matched)))
        n_terms = len(terms)
        if not found_any:
            status = "no evidence found"
        elif best >= min(2, n_terms):
            status = "evidence found"
        else:
            status = "weak evidence"
        coverage = round(len(found_any) / n_terms, 4) if n_terms else 0.0
        out.append(RequirementCheck(req, status, hits, coverage))
    return out


def checks_table(checks) -> pd.DataFrame:
    """One row per requirement; 'Keyword evidence' is a search result, not a compliance view."""
    rows = []
    for c in checks:
        r = c.requirement
        rows.append({
            "Requirement": r.id, "Topic": r.topic, "Statement": r.requirement, "Source": r.source,
            "Keyword evidence": c.status, "Keyword coverage": c.coverage,
            "Cited lines": "; ".join(f"[{h.file}:L{h.line}]" for h in c.hits),
            "Matched terms": " | ".join(h.matched for h in c.hits),
            "Evidence expected": r.evidence_expected,
        })
    return pd.DataFrame(rows, columns=["Requirement", "Topic", "Statement", "Source", "Keyword evidence",
                                       "Keyword coverage", "Cited lines", "Matched terms",
                                       "Evidence expected"])


# ── numeric parameters: documents vs code ──────────────────────────────────

_UNITS = {
    "%": "%", "percent": "%", "per cent": "%", "percentage points": "pp", "pp": "pp",
    "bps": "bp", "bp": "bp", "basis points": "bp", "basis point": "bp",
    "business days": "days", "business day": "days", "calendar days": "days", "days": "days",
    "day": "days", "dpd": "days", "months": "months", "month": "months", "years": "years",
    "year": "years", "weeks": "weeks", "week": "weeks",
}
_UNIT_RX = "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30}
_DOC_NUM = re.compile(
    r"(?<![\w.,/-])(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
    r"|(?:" + "|".join(_WORDS) + r"))"
    r"(?:\s*(?:-\s*)?(?P<unit>" + _UNIT_RX + r"))?(?![\w])", re.IGNORECASE)
_SKIP_BEFORE = re.compile(
    r"(?:\b(?:art|arts|article|articles|section|sections|chapter|chapters|para|paragraph|paragraphs|"
    r"figure|fig|table|page|p|pp|version|v|annex|point|points|recommendation|guideline|guidelines|"
    r"no|rule|step|appendix|title|part|ifrs|ias|gl|sr|rts|regulation|directive|tier|stage)\.?"
    r"|§|#)\s*$", re.IGNORECASE)
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b|\b\d{4}/\d{1,2}\b")
_CODE_NUM = re.compile(r"(?<![\w.])(\d[\d_]*(?:\.\d*)?(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)(?![\w.])")
_STOP = {"of", "the", "a", "an", "is", "are", "at", "to", "be", "set", "equal", "is set", "with", "and",
         "by", "for", "in", "on", "than", "least", "most", "up", "=", ":", "-", "–"}


def _to_float(s: str) -> float | None:
    s = s.strip().lower()
    if s in _WORDS:
        return float(_WORDS[s])
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s.replace("_", ""))
    except ValueError:
        return None


def _context(before: str, after: str) -> str:
    before = re.split(r"\d+(?:[.,]\d+)*\s*%?", before)[-1]  # words since the previous number
    words = re.findall(r"[A-Za-z][\w\-/()]*%?|[<>=≤≥]+", before)
    ctx = [w for w in words[-6:]]
    while ctx and ctx[0].lower() in _STOP:
        ctx.pop(0)
    text = " ".join(ctx)
    if not re.search(r"[A-Za-z]{2,}", text):
        tail = re.findall(r"[A-Za-z][\w\-/()]*", after)[:4]
        text = " ".join(tail)
    return text


def _candidates(value: float, unit: str | None) -> list[tuple[float, str]]:
    if unit == "%":
        return [(value / 100.0, "fraction"), (value, "percent number")]
    if unit == "bp":
        return [(value / 10000.0, "fraction"), (value, "bp number")]
    return [(value, "as stated")]


def _doc_values(doc_files: dict[str, str]) -> list[dict]:
    out = []
    for name, text in sorted(doc_files.items()):
        for ln_no, line in enumerate(str(text).splitlines(), start=1):
            dates = [m.span() for m in _DATE.finditer(line)]
            for m in _DOC_NUM.finditer(line):
                raw, unit_raw = m.group("num"), m.group("unit")
                if any(a <= m.start() < b for a, b in dates):
                    continue
                if raw.lower() in _WORDS and not unit_raw:
                    continue
                if _SKIP_BEFORE.search(line[:m.start()]):
                    continue
                # heading/list numbers like "4.2.1" or a leading "3." / "3)"
                if re.match(r"\.\d", line[m.end():m.end() + 2]):
                    continue
                if not line[:m.start()].strip() and re.match(r"[.)]\s", line[m.end():m.end() + 2]):
                    continue
                v = _to_float(raw)
                if v is None or not math.isfinite(v):
                    continue
                unit = _UNITS.get(unit_raw.lower()) if unit_raw else None
                if unit is None and 1900 <= v <= 2100 and float(v).is_integer():
                    continue  # a year
                ctx = _context(line[:m.start()], line[m.end():])
                if not ctx:
                    continue
                out.append({"file": name, "line": ln_no, "col": m.start(), "raw": m.group(0).strip(),
                            "value": v, "unit": unit, "context": ctx})
    return out


def _code_numbers(code_files: dict[str, str]) -> list[tuple[float, str, int, str]]:
    out = []
    for name, text in sorted(code_files.items()):
        for ln_no, line in enumerate(str(text).splitlines(), start=1):
            for m in _CODE_NUM.finditer(line):
                try:
                    v = float(m.group(1).replace("_", ""))
                except ValueError:
                    continue
                out.append((v, name, ln_no, m.group(1)))
    return out


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


def cross_check_values(doc_files: dict[str, str], code_files: dict[str, str]) -> pd.DataFrame:
    """Numeric parameters stated in documents, and where the same value appears in code.

    Document values are read with their unit (%, bp, pp, days, months, years, weeks;
    small number words such as "five years" too) and the words just before them as
    context. Percentages and basis points are searched in code both as a fraction
    (0.03% -> 0.0003) and as the stated number (0.03); others as stated. Code literals
    are matched numerically, so 0.0003, 3e-4 and 3.0E-04 all match. Article/section
    numbers, dates, years and heading numbers are skipped. A value found in code is
    only a textual coincidence until the agent reads the cited code line.
    """
    docs = _doc_values(doc_files)
    code = _code_numbers(code_files)
    rows = []
    for d in docs:
        matches, how = [], []
        for target, label in _candidates(d["value"], d["unit"]):
            for v, f, ln, lit in code:
                if _same(v, target):
                    ref = f"[{f}:L{ln}]"
                    if ref not in matches:
                        matches.append(ref)
                        how.append(f"{lit} ({label})")
        norm = ", ".join(f"{t:g}" for t, _ in _candidates(d["value"], d["unit"]))
        rows.append({
            "Document": d["file"], "Line": d["line"], "Cited line": f"[{d['file']}:L{d['line']}]",
            "Stated value": d["raw"], "Unit": d["unit"] or "", "Searched as": norm,
            "Context": d["context"], "Found in code": bool(matches),
            "Code matches": "; ".join(matches) if matches else "none",
            "Code literals": "; ".join(how),
        })
    return pd.DataFrame(rows, columns=["Document", "Line", "Cited line", "Stated value", "Unit",
                                       "Searched as", "Context", "Found in code", "Code matches",
                                       "Code literals"])


# ── variable names: documents vs data columns ──────────────────────────────

def _tokens(name: str) -> list[str]:
    toks = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", str(name))
    return [t.lower() for t in toks]


def normalise_name(name: str) -> str:
    """'LoanToValue', 'loan_to_value', 'loan-to-value', 'Loan to value' -> 'loan_to_value'."""
    return "_".join(_tokens(name))


_IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_]{0,63})`"
                    r"|(?<![\w.])([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)(?![\w])"
                    r"|(?<![\w.])([a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+)(?![\w])")


def cross_check_columns(doc_files: dict[str, str], columns: list[str]) -> pd.DataFrame:
    """Variables named in documents vs data columns, matched on normalised tokens.

    A data column counts as documented when its tokens appear in sequence in a
    document line with any separator (column `loan_to_value` matches "loan to value",
    "Loan-to-Value", "loanToValue"). Names counted as variables in the documents are
    identifier-like words only: snake_case, camelCase, or `backticked`; plain-English
    phrases are not treated as variable names. Rows: in both / documents only / data only.
    """
    lines = [(name, i, ln) for name, text in sorted(doc_files.items())
             for i, ln in enumerate(str(text).splitlines(), start=1)]
    by_norm: dict[str, dict] = {}
    for col in columns:
        key = normalise_name(col)
        if not key:
            continue
        rec = by_norm.setdefault(key, {"cols": [], "doc_names": [], "refs": []})
        rec["cols"].append(str(col))
    # data -> documents
    for key, rec in by_norm.items():
        toks = key.split("_")
        rx = re.compile(r"(?<![A-Za-z0-9])" + r"[\s_\-.]*".join(re.escape(t) for t in toks)
                        + r"(?![a-z0-9])", re.IGNORECASE)
        for name, i, ln in lines:
            if rx.search(ln):
                rec["refs"].append(f"[{name}:L{i}]")
    # documents -> data
    for name, i, ln in lines:
        for m in _IDENT.finditer(ln):
            ident = next(g for g in m.groups() if g)
            key = normalise_name(ident)
            if not key:
                continue
            rec = by_norm.setdefault(key, {"cols": [], "doc_names": [], "refs": []})
            if ident not in rec["doc_names"]:
                rec["doc_names"].append(ident)
            ref = f"[{name}:L{i}]"
            if ref not in rec["refs"]:
                rec["refs"].append(ref)
    rows = []
    for key in sorted(by_norm):
        rec = by_norm[key]
        in_docs, in_data = bool(rec["refs"]), bool(rec["cols"])
        rows.append({
            "Normalised name": key,
            "Data column": ", ".join(rec["cols"]),
            "Named in documents as": ", ".join(rec["doc_names"]),
            "Where": "both" if in_docs and in_data else ("documents only" if in_docs else "data only"),
            "Document lines": "; ".join(rec["refs"][:5]) + (" ..." if len(rec["refs"]) > 5 else ""),
        })
    order = {"documents only": 0, "data only": 1, "both": 2}
    rows.sort(key=lambda r: (order[r["Where"]], r["Normalised name"]))
    return pd.DataFrame(rows, columns=["Normalised name", "Data column", "Named in documents as", "Where",
                                       "Document lines"])
