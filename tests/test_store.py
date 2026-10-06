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
    assert chat == {"id": cid, "title": "First", "sources": [], "pinned": False, "messages": []}


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


def test_pinned_chats_come_first():
    a = store.create_chat("a")
    time.sleep(0.01)
    b = store.create_chat("b")
    store.set_pinned(a, True)
    chats = store.list_chats()
    assert [(c["id"], c["pinned"]) for c in chats] == [(a, True), (b, False)]
    store.set_pinned(a, False)
    assert [c["id"] for c in store.list_chats()] == [b, a]


def test_older_database_gets_the_pinned_column():
    import sqlite3
    config.DB_PATH.parent.mkdir(parents=True)
    with sqlite3.connect(config.DB_PATH) as con:          # schema before pins existed
        con.execute("CREATE TABLE chats (id TEXT PRIMARY KEY, title TEXT NOT NULL, created REAL NOT NULL, "
                    "updated REAL NOT NULL, sources TEXT NOT NULL DEFAULT '[]')")
        con.execute("INSERT INTO chats VALUES ('old1', 'Old chat', 1, 1, '[]')")
    assert store.list_chats() == [{"id": "old1", "title": "Old chat", "updated": 1, "pinned": False}]
    store.set_pinned("old1", True)
    assert store.get_chat("old1")["pinned"] is True


def test_each_history_folder_has_its_own_chats(tmp_path):
    mine, theirs = tmp_path / "alice", tmp_path / "bob"
    store.use_folder(mine)
    a = store.create_chat("alice's chat")
    store.use_folder(theirs)
    assert store.list_chats() == []
    store.create_chat("bob's chat")
    assert (mine / "chats.db").exists() and (theirs / "chats.db").exists()
    store.use_folder(mine)
    assert [c["id"] for c in store.list_chats()] == [a]
    store.use_folder("  ")
    assert store.db_path() == config.DB_PATH and store.list_chats() == []


def test_resolver_is_asked_on_every_call(tmp_path):
    """The UI's session folder must apply even where the main script didn't run first (a
    Streamlit dialog rerun): the store asks the resolver each time."""
    session = {"dir": str(tmp_path / "mine")}
    store.set_resolver(lambda: session["dir"])
    cid = store.create_chat("in my folder")
    store.rename(cid, "renamed")                       # e.g. from the Settings dialog
    assert store.get_chat(cid)["title"] == "renamed" and store.db_path() == tmp_path / "mine" / "chats.db"
    session["dir"] = ""                                # back to the default: a different database
    assert store.get_chat(cid) is None and store.db_path() == config.DB_PATH
    store.use_folder(tmp_path / "mine")                # an explicit folder overrides the resolver
    assert store.get_chat(cid)["title"] == "renamed"


def test_full_chat_download_reads_the_captured_folder(tmp_path):
    """The sidebar's 'Save full chat' builds its file when clicked, outside the session: it must
    read the folder captured when the menu was drawn, not whatever is current then."""
    from ask.ui import sidebar
    store.use_folder(tmp_path / "mine")
    cid = store.create_chat("Mine")
    store.add_message(cid, "user", "What is the PSI?")
    store.add_message(cid, "assistant", "The PSI is 0.087.")
    build = sidebar._full_chat_md(store.folder(), cid, "Mine")
    store.use_folder(None)                             # the click happens elsewhere
    md = build()
    assert md.startswith("# Mine") and "What is the PSI?" in md and "The PSI is 0.087." in md
    assert store.get_chat(cid) is None and store.get_chat(cid, in_folder=tmp_path / "mine")["title"] == "Mine"


def test_delete_removes_the_markdown_copy(tmp_path):
    from ask.memory import export
    store.use_folder(tmp_path)
    cid = store.create_chat("t")
    p = export.mirror(tmp_path, cid, "t", [{"role": "user", "content": "hi"}])
    assert p.exists()
    store.delete_chat(cid)
    assert not p.exists()


def test_unicode_and_non_json_meta():
    from pathlib import Path
    cid = store.create_chat("Ünïcødé ✓")
    store.add_message(cid, "assistant", "naïve — ✓", meta={"path": Path("C:/x")})
    chat = store.get_chat(cid)
    assert chat["title"] == "Ünïcødé ✓" and chat["messages"][0]["meta"]["path"] in ("C:/x", "C:\\x")
