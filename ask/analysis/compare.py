"""Deterministic comparison of two files, two text sources, or two tables.

The model gets exact diffs and statistics to explain; it never has to
"eyeball" two documents from memory.
"""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd

from ask.retrieval.search import Evidence
from ask.sources.loaders import Source
from ask.sources.registry import SourceRegistry

MAX_DIFF_LINES = 400


def _label(src: Source, rel: str) -> str:
    return rel if len(src.files) == 1 else f"{src.name}:{rel}"


def _resolve(reg: SourceRegistry, ref: str):
    """('source', Source) for a whole source, or ('file', (Source, relpath)).
    A loaded single document counts as a file."""
    def as_kind(src: Source):
        if src.kind != "data" and len(src.files) == 1:
            return "file", (src, next(iter(src.files)))
        return "source", src

    if ref in reg.sources:
        return as_kind(reg.sources[ref])
    try:
        return "file", reg.resolve_file(ref)
    except KeyError as file_err:
        try:
            return as_kind(reg.get(ref))
        except KeyError:
            raise KeyError(str(file_err)) from None


def compare(reg: SourceRegistry, ev: Evidence, a: str, b: str) -> str:
    ka, va = _resolve(reg, a)
    kb, vb = _resolve(reg, b)
    if ka == "source" and kb == "source":
        if va.kind == "data" and vb.kind == "data":
            return compare_tables(va.df, vb.df, va.name, vb.name)
        if va.kind != "data" and vb.kind != "data":
            return compare_trees(va, vb)
        raise ValueError("Can't diff a table against a repo/document; compare like with like.")
    if ka == "file" and kb == "file":
        (sa, ra), (sb, rb) = va, vb
        return compare_texts(ev, sa, ra, sb, rb)
    (file_src, rel), folder = (va, vb) if ka == "file" else (vb, va)
    if folder.kind == "data":
        raise ValueError("Can't diff a document against a table; ask about the table's contents instead.")
    return compare_file_to_source(ev, file_src, rel, folder)


def _token_set(text: str) -> set[str]:
    from ask.retrieval.search import tokens
    return set(tokens(text))


def compare_file_to_source(ev: Evidence, fsrc: Source, rel: str, folder: Source, top: int = 8) -> str:
    """Which files in `folder` best match one file, plus an exact diff with the closest one."""
    target = fsrc.files[rel]
    t_tokens = _token_set(target)
    t_lines = target.splitlines()
    scored = []
    for other_rel, text in folder.files.items():
        if folder is fsrc and other_rel == rel:
            continue
        o_tokens = _token_set(text)
        if not o_tokens or not t_tokens:
            continue
        vocab = len(t_tokens & o_tokens) / len(t_tokens | o_tokens)      # shared vocabulary
        cover = len(t_tokens & o_tokens) / len(t_tokens)                  # how much of the file it covers
        lines = difflib.SequenceMatcher(None, t_lines, text.splitlines(), autojunk=False).quick_ratio()
        scored.append((0.4 * vocab + 0.3 * cover + 0.3 * lines, vocab, cover, lines, other_rel))
    if not scored:
        return f"{folder.name} has no text files to compare against."
    scored.sort(reverse=True)
    out = [f"FILE = {_label(fsrc, rel)} ({len(t_lines)} lines)",
           f"FOLDER = {folder.name} ({len(folder.files)} files)", "",
           "closest files in the folder (score · shared vocabulary · coverage of FILE's terms · line similarity):"]
    out += [f"  {s:.0%} · {v:.0%} · {c:.0%} · {ln:.0%}  {r}" for s, v, c, ln, r in scored[:top]]
    best = scored[0]
    if best[3] >= 0.3:
        out += ["", f"Exact diff with the closest match ({best[4]}):",
                compare_texts(ev, fsrc, rel, folder, best[4])]
    else:
        out += ["", "No file in the folder is a near-copy of FILE (line similarity < 30%), so a line "
                "diff would be meaningless. To compare content (e.g. does the code implement the "
                "document), search both sides for each requirement and read the matching lines."]
    return "\n".join(out)


def compare_texts(ev: Evidence, sa: Source, ra: str, sb: Source, rb: str) -> str:
    la, lb = sa.files[ra].splitlines(), sb.files[rb].splitlines()
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    ratio = sm.ratio() if len(la) + len(lb) < 40_000 else sm.quick_ratio()
    out = [f"A = {_label(sa, ra)} ({len(la)} lines)", f"B = {_label(sb, rb)} ({len(lb)} lines)",
           f"line similarity: {ratio:.1%}", ""]
    n_changed = 0
    body: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        n_changed += 1
        if len(body) > MAX_DIFF_LINES:
            continue
        body.append(f"@@ {tag}: A L{i1 + 1}-{max(i1 + 1, i2)}  ->  B L{j1 + 1}-{max(j1 + 1, j2)}")
        for k in range(i1, min(i2, i1 + 40)):
            body.append(f"- A{k + 1}| {la[k]}")
        for k in range(j1, min(j2, j1 + 40)):
            body.append(f"+ B{k + 1}| {lb[k]}")
        if i2 > i1:
            ev.add(sa.name, ra, i1 + 1, i2)
        if j2 > j1:
            ev.add(sb.name, rb, j1 + 1, j2)
    if n_changed == 0:
        return "\n".join(out + ["The two files are identical."])
    out.append(f"{n_changed} changed blocks:")
    out += body
    if len(body) > MAX_DIFF_LINES:
        out.append("… diff truncated; read_file both files around the blocks above for detail")
    return "\n".join(out)


def compare_trees(a: Source, b: Source) -> str:
    fa, fb = set(a.files), set(b.files)
    only_a, only_b, both = sorted(fa - fb), sorted(fb - fa), sorted(fa & fb)
    changed = []
    for rel in both:
        if a.files[rel] != b.files[rel]:
            r = difflib.SequenceMatcher(None, a.files[rel].splitlines(),
                                        b.files[rel].splitlines(), autojunk=False).quick_ratio()
            changed.append((r, rel))
    changed.sort()
    out = [f"A = {a.name} ({len(fa)} files), B = {b.name} ({len(fb)} files)",
           f"identical: {len(both) - len(changed)}, changed: {len(changed)}, "
           f"only in A: {len(only_a)}, only in B: {len(only_b)}", ""]
    if changed:
        out.append("changed files (similarity, most different first):")
        out += [f"  {r:.0%}  {rel}" for r, rel in changed[:150]]
    if only_a:
        out.append("only in A:")
        out += [f"  {r}" for r in only_a[:150]]
    if only_b:
        out.append("only in B:")
        out += [f"  {r}" for r in only_b[:150]]
    out.append("\nUse compare with 'A:path' and 'B:path' (source-prefixed) to diff a specific file.")
    return "\n".join(out)


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, na: str, nb: str) -> str:
    ca, cb = list(map(str, a.columns)), list(map(str, b.columns))
    a = a.copy(); b = b.copy()
    a.columns, b.columns = ca, cb
    common = [c for c in ca if c in cb]
    out = [f"A = {na}: {a.shape[0]:,} rows × {a.shape[1]} cols",
           f"B = {nb}: {b.shape[0]:,} rows × {b.shape[1]} cols", ""]
    if set(ca) - set(cb):
        out.append("columns only in A: " + ", ".join(c for c in ca if c not in cb))
    if set(cb) - set(ca):
        out.append("columns only in B: " + ", ".join(c for c in cb if c not in ca))
    dtype_diff = [f"{c}: {a[c].dtype} -> {b[c].dtype}" for c in common if a[c].dtype != b[c].dtype]
    if dtype_diff:
        out.append("dtype changes: " + "; ".join(dtype_diff))

    rows = []
    for c in common:
        sa, sb = a[c], b[c]
        row = {"column": c, "null% A": round(sa.isna().mean() * 100, 2),
               "null% B": round(sb.isna().mean() * 100, 2)}
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            for stat in ("mean", "std", "min", "max"):
                va, vb = getattr(sa, stat)(), getattr(sb, stat)()
                row[f"{stat} A"], row[f"{stat} B"] = _r(va), _r(vb)
            row["mean Δ%"] = _r((sb.mean() - sa.mean()) / abs(sa.mean()) * 100) if sa.mean() else None
        else:
            ta = set(sa.dropna().astype(str).unique()[:5000])
            tb = set(sb.dropna().astype(str).unique()[:5000])
            row["unique A"], row["unique B"] = len(ta), len(tb)
            row["values only in B (sample)"] = ", ".join(sorted(tb - ta)[:5])
        rows.append(row)
    if rows:
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            out += ["", "per-column comparison:", pd.DataFrame(rows).to_string(index=False)]

    if a.shape == b.shape and ca == cb:
        try:
            diff_mask = ~((a == b) | (a.isna() & b.isna()))
            n = int(diff_mask.values.sum())
            out.append(f"\nrow-aligned cell differences: {n:,} of {a.size:,} cells")
            if n:
                per_col = diff_mask.sum()
                out.append("differing cells by column: " + ", ".join(
                    f"{c}={int(v)}" for c, v in per_col[per_col > 0].items()))
        except Exception:
            pass
    return "\n".join(out)


def _r(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v)
