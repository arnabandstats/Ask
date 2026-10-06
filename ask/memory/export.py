"""Chats as Markdown: one answer with its question, or a whole chat.

Used by the Save button under every answer (browser download) and to keep a readable
copy of every chat next to the chat database (<history folder>/chats_md/<title>-<id>.md).
"""
from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path

import pandas as pd

MAX_TABLE_ROWS = 50            # rows of each result table written into the Markdown
MIRROR_DIR = "chats_md"


def _cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(a: dict) -> str:
    df = pd.read_json(StringIO(a["data"]), orient="split")
    shown = df.head(MAX_TABLE_ROWS)
    rows = ["| " + " | ".join(_cell(c) for c in shown.columns) + " |",
            "|" + "---|" * len(shown.columns)]
    rows += ["| " + " | ".join(_cell(v) for v in r) + " |" for r in shown.itertuples(index=False)]
    total = a.get("rows", len(df))
    if total > len(shown):
        rows.append(f"\n_First {len(shown)} of {total:,} rows._")
    return "\n".join(rows)


def _artifacts(items: list[dict]) -> list[str]:
    out = []
    for a in items:
        title = a.get("title") or "Result"
        group = f"{a['group']} · " if a.get("group") else ""
        if a.get("type") == "table":
            try:
                out.append(f"**{group}{title}**\n\n{_table(a)}")
            except Exception:
                out.append(f"**{group}{title}** (table could not be exported)")
        elif a.get("type") in {"plotly", "image"}:
            out.append(f"_{'Chart' if a['type'] == 'plotly' else 'Figure'}: {group}{title} (shown in the app)_")
        elif a.get("type") == "file":
            out.append(f"_File: {group}{title} — `{a.get('path', '')}`_")
    return out


def message_md(m: dict, assistant_name: str = "Assistant") -> str:
    if m["role"] == "user":
        return "## You\n\n" + m["content"].strip()
    parts = [f"## {assistant_name}", m["content"].strip()]
    parts += _artifacts(m.get("artifacts") or [])
    meta = m.get("meta") or {}
    if meta.get("test_runs"):
        parts.append("Test runs: " + ", ".join(f"`{r}`" for r in meta["test_runs"]))
    return "\n\n".join(parts)


def _header(title: str) -> str:
    return f"# {title}\n\n_Exported {time.strftime('%Y-%m-%d %H:%M')}_"


def exchange_md(messages: list[dict], idx: int, title: str, assistant_name: str = "Assistant") -> str:
    """The answer at messages[idx] with the question that led to it."""
    q = next((i for i in range(idx - 1, -1, -1) if messages[i]["role"] == "user"), None)
    picked = ([messages[q]] if q is not None else []) + [messages[idx]]
    return "\n\n".join([_header(title)] + [message_md(m, assistant_name) for m in picked]) + "\n"


def chat_md(messages: list[dict], title: str, assistant_name: str = "Assistant") -> str:
    body = "\n\n---\n\n".join(message_md(m, assistant_name) for m in messages)
    return f"{_header(title)}\n\n{body}\n"


def file_name(title: str, suffix: str = "") -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")[:60] or "chat"
    return f"{slug}{suffix}.md"


def mirror(folder: Path, cid: str, title: str, messages: list[dict], assistant_name: str = "Assistant") -> Path:
    """Write the chat's Markdown copy into the history folder (replacing an older copy, e.g.
    under a previous title)."""
    remove_mirror(folder, cid)
    d = folder / MIRROR_DIR
    d.mkdir(parents=True, exist_ok=True)
    p = d / file_name(title, f"-{cid}")
    p.write_text(chat_md(messages, title, assistant_name), encoding="utf-8")
    return p


def remove_mirror(folder: Path, cid: str) -> None:
    d = folder / MIRROR_DIR
    if d.is_dir():
        for old in d.glob(f"*-{cid}.md"):
            try:
                old.unlink()
            except OSError:
                pass
