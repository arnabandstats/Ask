"""Chat persistence in a local SQLite file (ask_data/chats.db).

A chat stores its messages (with artifacts and metadata) and the list of
sources that were loaded, so reopening a chat restores both.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from ask import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]'
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
def _conn():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        con.executescript(_SCHEMA)
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
    if not config.DB_PATH.exists():
        return []
    with _conn() as con:
        rows = con.execute("SELECT id, title, updated FROM chats ORDER BY updated DESC LIMIT ?",
                           (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_chat(cid: str) -> dict | None:
    if not config.DB_PATH.exists():
        return None
    with _conn() as con:
        row = con.execute("SELECT * FROM chats WHERE id = ?", (cid,)).fetchone()
        if row is None:
            return None
        msgs = con.execute("SELECT role, content, artifacts, meta FROM messages "
                           "WHERE chat_id = ? ORDER BY id", (cid,)).fetchall()
    return {"id": row["id"], "title": row["title"], "sources": json.loads(row["sources"]),
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


def delete_chat(cid: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM chats WHERE id = ?", (cid,))


def title_from(text: str) -> str:
    t = " ".join(text.split())
    return t if len(t) <= 48 else t[:47].rstrip() + "…"
