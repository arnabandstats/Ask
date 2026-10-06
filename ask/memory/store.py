"""Chat persistence in a local SQLite file (ask_data/chats.db, or chats.db in the user's
chat history folder chosen in Settings).

A chat stores its messages (with artifacts and metadata) and the list of
sources that were loaded, so reopening a chat restores both.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

from ask import config

# Where chats live. The UI registers a resolver that returns the current session's chat history
# folder; it is asked on every call because Streamlit reruns a dialog or fragment without the
# main script, so a value set at the top of a run is not seen there. use_folder() overrides it
# for the current context (scripts, tests). None/blank everywhere = config.DB_PATH.
_folder: ContextVar[Path | None] = ContextVar("chat_history_folder", default=None)
_resolver: Callable[[], str | Path | None] | None = None


def set_resolver(fn: Callable[[], str | Path | None] | None) -> None:
    global _resolver
    _resolver = fn


def use_folder(folder: str | Path | None) -> None:
    """Store chats in `folder`/chats.db in this context (None or blank: back to the resolver/default)."""
    _folder.set(_as_dir(folder))


def _as_dir(folder) -> Path | None:
    return Path(folder).expanduser() if folder and str(folder).strip() else None


def _current() -> Path | None:
    f = _folder.get()
    if f is None and _resolver is not None:
        f = _as_dir(_resolver())
    return f


def folder() -> Path:
    """The folder holding the chat database in use."""
    return _current() or config.DB_PATH.parent


def db_path() -> Path:
    f = _current()
    return f / "chats.db" if f else config.DB_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    pinned INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    artifacts TEXT NOT NULL DEFAULT '[]',
    meta TEXT NOT NULL DEFAULT '{}',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
"""


@contextmanager
def _conn(path: Path | None = None):
    path = path or db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        con.executescript(_SCHEMA)
        if "pinned" not in {r[1] for r in con.execute("PRAGMA table_info(chats)")}:   # older databases
            con.execute("ALTER TABLE chats ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        yield con
        con.commit()
    finally:
        con.close()


def create_chat(title: str = "New chat") -> str:
    cid = uuid.uuid4().hex[:12]
    now = time.time()
    with _conn() as con:
        con.execute("INSERT INTO chats (id, title, created, updated) VALUES (?, ?, ?, ?)",
                    (cid, title, now, now))
    return cid


def list_chats(limit: int = 200) -> list[dict]:
    """Chats, pinned ones first, each group most recently updated first."""
    if not db_path().exists():
        return []
    with _conn() as con:
        rows = con.execute("SELECT id, title, updated, pinned FROM chats "
                           "ORDER BY pinned DESC, updated DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"pinned": bool(r["pinned"])} for r in rows]


def get_chat(cid: str, in_folder: str | Path | None = None) -> dict | None:
    """A chat with its messages. in_folder reads that history folder's database explicitly
    (for code that runs outside the session, e.g. a deferred download)."""
    path = Path(in_folder) / "chats.db" if in_folder else db_path()
    if not path.exists():
        return None
    with _conn(path) as con:
        row = con.execute("SELECT * FROM chats WHERE id = ?", (cid,)).fetchone()
        if row is None:
            return None
        msgs = con.execute("SELECT role, content, artifacts, meta FROM messages "
                           "WHERE chat_id = ? ORDER BY id", (cid,)).fetchall()
    return {"id": row["id"], "title": row["title"], "sources": json.loads(row["sources"]),
            "pinned": bool(row["pinned"]),
            "messages": [{"role": m["role"], "content": m["content"],
                          "artifacts": json.loads(m["artifacts"]), "meta": json.loads(m["meta"])}
                         for m in msgs]}


def add_message(cid: str, role: str, content: str, artifacts: list | None = None,
                meta: dict | None = None) -> None:
    now = time.time()
    with _conn() as con:
        con.execute("INSERT INTO messages (chat_id, role, content, artifacts, meta, created) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (cid, role, content, json.dumps(artifacts or [], default=str),
                     json.dumps(meta or {}, default=str), now))
        con.execute("UPDATE chats SET updated = ? WHERE id = ?", (now, cid))


def set_sources(cid: str, records: list[dict]) -> None:
    with _conn() as con:
        con.execute("UPDATE chats SET sources = ? WHERE id = ?", (json.dumps(records), cid))


def rename(cid: str, title: str) -> None:
    with _conn() as con:
        con.execute("UPDATE chats SET title = ? WHERE id = ?", (title.strip()[:80] or "Untitled", cid))


def set_pinned(cid: str, pinned: bool) -> None:
    with _conn() as con:
        con.execute("UPDATE chats SET pinned = ? WHERE id = ?", (int(bool(pinned)), cid))


def delete_chat(cid: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM chats WHERE id = ?", (cid,))
    from ask.memory import export
    export.remove_mirror(folder(), cid)


def title_from(text: str) -> str:
    t = " ".join(text.split())
    return t if len(t) <= 48 else t[:47].rstrip() + "…"
