"""Render one stored chat message: text, artifacts and the citation check."""
from __future__ import annotations

import html
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from ask.agent.faithfulness import CITATION, parse_ranges
from ask.validation.store import RUN_CITATION

def message(m: dict, idx: int, messages: list[dict] | None = None, title: str = "Chat") -> None:
    """User turns: a grey bubble on the right. Assistant turns: plain text on the left, then a
    small Save menu when `messages` (the whole chat) is given.
    The layout comes from the st-key-umsg-* / st-key-amsg-* classes in styles.py."""
    if m["role"] == "user":
        with st.container(key=f"umsg-{idx}"):
            st.markdown(user_bubble(m["content"]), unsafe_allow_html=True)
        return
    with assistant_block(idx):
        st.markdown(_compact_citations(m["content"]))
        artifacts(m.get("artifacts") or [], idx)
        verification((m.get("meta") or {}).get("verification"), idx)
        guard_note((m.get("meta") or {}).get("guard"))
        test_runs((m.get("meta") or {}).get("test_runs"))
        if messages:
            save_answer(messages, idx, title)


def save_answer(messages: list[dict], idx: int, title: str) -> None:
    """Download this answer (with its question) as Markdown. The whole chat is saved from the
    chat's ⋯ menu in the sidebar."""
    from ask import preferences
    from ask.memory import export
    st.download_button("", lambda: export.exchange_md(messages, idx, title, preferences.tool_name()),
                       file_name=export.file_name(title, f"-answer-{idx}"), mime="text/markdown",
                       icon=":material/download:", type="tertiary", key=f"save-{idx}",
                       help="Save this response (.md)", on_click="ignore")


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
    text = CITATION.sub(short, text)
    return RUN_CITATION.sub(lambda m: f" `test · {m.group(1)}`", text)


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


def guard_note(events: list[str] | None) -> None:
    """What the guardrails did for this answer (injection flagged, answer rewritten or withheld)."""
    if not events:
        return
    body = "<br>".join(html.escape(e) for e in events)
    st.markdown(f"<div class='ks-check warn'>🛡 Guardrails<br>{body}</div>", unsafe_allow_html=True)


def test_runs(run_ids: list[str] | None) -> None:
    """Provenance of every validation test run behind the answer (from the saved audit records)."""
    if not run_ids:
        return
    from ask.validation import store
    with st.expander(f"Test runs  ·  {len(run_ids)}", expanded=False):
        for rid in run_ids:
            try:
                r = store.load(rid)
            except KeyError:
                st.markdown(f"⚠ `{rid}` — record not found")
                continue
            judge = " · LLM judge" if r.get("kind") == "judge" else ""
            st.markdown(f"`{rid}` **{r['test_name']}** ({r['test_id']}){judge} — {r['status']}  \n"
                        f"source {r['source'] or '-'} · data sha256 {r['data_fingerprint']} · "
                        f"rows {r['rows_in']} · seed {r['seed']} · params `{r['params']}`")
