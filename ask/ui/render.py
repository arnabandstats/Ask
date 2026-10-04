"""Render one stored chat message: text, artifacts and the citation check."""
from __future__ import annotations

import html
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from ask.agent.faithfulness import CITATION, parse_ranges

def message(m: dict, idx: int) -> None:
    """User turns: a grey bubble on the right. Assistant turns: plain text on the left.
    The layout comes from the st-key-umsg-* / st-key-amsg-* classes in styles.py."""
    if m["role"] == "user":
        with st.container(key=f"umsg-{idx}"):
            st.markdown(user_bubble(m["content"]), unsafe_allow_html=True)
        return
    with assistant_block(idx):
        st.markdown(_compact_citations(m["content"]))
        artifacts(m.get("artifacts") or [], idx)
        verification((m.get("meta") or {}).get("verification"), idx)


def assistant_block(idx: int | str):
    return st.container(key=f"amsg-{idx}")


def user_bubble(text: str) -> str:
    """The user's text, HTML-escaped (shown exactly as typed, never as Markdown/HTML),
    in a right-aligned grey bubble. Line breaks are kept by CSS (white-space: pre-wrap)."""
    return f"<div class='ks-user-row'><div class='ks-user'>{html.escape(text.strip()) or '&nbsp;'}</div></div>"


def _compact_citations(text: str) -> str:
    """[src:dir/file.py:L40-58] -> `file.py · L40–58` (full refs stay in the Sources panel)."""
    def short(m: re.Match) -> str:
        name = m.group(1).replace("\\", "/").split(":")[-1].split("/")[-1]
        lines = ", ".join(f"L{a}–{b}" if b != a else f"L{a}" for a, b in parse_ranges(m.group(2)))
        return f" `{name} · {lines}`"
    return CITATION.sub(short, text)


def artifacts(items: list[dict], idx: int) -> None:
    loose = [a for a in items if not a.get("group")]
    groups: dict[str, list[dict]] = {}
    for a in items:
        if a.get("group"):
            groups.setdefault(a["group"], []).append(a)
    for j, a in enumerate(loose):
        _one(a, f"{idx}-l{j}")
    for g, (name, arts) in enumerate(groups.items()):
        with st.expander(f"{name}  ·  {len(arts)} items", expanded=False):
            for j, a in enumerate(arts):
                _one(a, f"{idx}-g{g}-{j}")


def _one(a: dict, key: str) -> None:
    kind, title = a.get("type"), a.get("title") or ""
    try:
        if kind == "table":
            df = pd.read_json(StringIO(a["data"]), orient="split")
            note = f"  (first {len(df):,} of {a['rows']:,} rows)" if a.get("rows", 0) > len(df) else ""
            st.caption(title + note)
            st.dataframe(df, width="stretch", hide_index=True)
        elif kind == "plotly":
            import plotly.io as pio
            fig = pio.from_json(Path(a["path"]).read_text(encoding="utf-8"))
            st.plotly_chart(fig, width="stretch", key=f"fig-{key}")
        elif kind == "image":
            st.image(a["path"], caption=title)
        elif kind == "file":
            p = Path(a["path"])
            if p.exists() and p.stat().st_size < 25_000_000:
                st.download_button(f"Download {title}", p.read_bytes(), file_name=p.name, key=f"dl-{key}")
            st.caption(f"{title}: `{p}`")
    except FileNotFoundError:
        st.caption(f"{title}: output file no longer exists")
    except Exception as exc:
        st.caption(f"{title}: could not display ({type(exc).__name__}: {exc})")


def verification(v: dict | None, idx: int) -> None:
    if not v or not v.get("checked"):
        return
    cites = v.get("citations", [])
    ok = [c for c in cites if c["status"] == "ok"]
    problems = v.get("problems", [])
    if not cites and not problems:
        return
    if not problems:
        st.markdown(f"<div class='ks-check'>✓ {len(ok)} citation{'s' if len(ok) != 1 else ''} "
                    "checked against the source</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ks-check warn'>⚠ {len(problems)} point(s) could not be verified "
                    f"against the source ({len(ok)} of {len(cites)} citations verified)</div>",
                    unsafe_allow_html=True)
    if cites or problems:
        with st.expander("Sources", expanded=False):
            for c in cites:
                mark = "✓" if c["status"] == "ok" else "⚠"
                where = f"{c['source']} · {c['file']}" if c.get("file") else c["ref"]
                st.markdown(f"{mark} **{where}** · lines {c['start']}–{c['end']}"
                            + ("" if c["status"] == "ok" else f" — {c['detail']}"))
                if c.get("snippet"):
                    st.code(c["snippet"], language=None)
            for p in problems:
                if not any(p.startswith(c["ref"]) for c in cites):
                    st.markdown(f"⚠ {p}")
