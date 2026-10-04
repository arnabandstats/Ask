"""SQLite chat memory."""
from __future__ import annotations

import time

from ask import config
from ask.memory import store


def test_no_db_until_first_chat():
    assert store.list_chats() == [] and store.get_chat("nope") is None
    assert not config.DB_PATH.exists()


def test_create_and_get():
    cid = store.create_chat("First")
    chat = store.get_chat(cid)
    assert chat == {"id": cid, "title": "First", "sources": [], "messages": []}


def test_messages_roundtrip_with_artifacts_and_meta():
    cid = store.create_chat()
    store.add_message(cid, "user", "hello")
    store.add_message(cid, "assistant", "hi", artifacts=[{"type": "table", "data": "{}"}],
                      meta={"verification": {"checked": True}})
    msgs = store.get_chat(cid)["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["artifacts"] == [{"type": "table", "data": "{}"}]
    assert msgs[1]["meta"]["verification"]["checked"] is True


def test_sources_persist():
    cid = store.create_chat()
    recs = [{"name": "repo", "kind": "repo", "path": "C:/x"}]
    store.set_sources(cid, recs)
    assert store.get_chat(cid)["sources"] == recs


def test_most_recently_updated_first():
    a = store.create_chat("a")
    time.sleep(0.01)
    b = store.create_chat("b")
    time.sleep(0.01)
    store.add_message(a, "user", "bump")
    assert [c["id"] for c in store.list_chats()] == [a, b]


def test_rename_and_delete_cascades():
    cid = store.create_chat("old")
    store.add_message(cid, "user", "x")
    store.rename(cid, "  new title  ")
    assert store.get_chat(cid)["title"] == "new title"
    store.rename(cid, "   ")
    assert store.get_chat(cid)["title"] == "Untitled"
    store.delete_chat(cid)
    assert store.get_chat(cid) is None
    import sqlite3
    with sqlite3.connect(config.DB_PATH) as con:
        assert con.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_title_from():
    assert store.title_from("  short   question ") == "short question"
    long = store.title_from("x" * 100)
    assert len(long) == 48 and long.endswith("…")


def test_unicode_and_non_json_meta():
    from pathlib import Path
    cid = store.create_chat("Ünïcødé ✓")
    store.add_message(cid, "assistant", "naïve — ✓", meta={"path": Path("C:/x")})
    chat = store.get_chat(cid)
    assert chat["title"] == "Ünïcødé ✓" and chat["messages"][0]["meta"]["path"] in ("C:/x", "C:\\x")
