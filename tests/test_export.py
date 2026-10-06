"""Chats as Markdown, and the per-user chat history folder preference."""
from __future__ import annotations

import pandas as pd
import pytest

from ask import preferences
from ask.memory import export

TABLE = {"type": "table", "title": "Results", "group": "Suite",
         "data": pd.DataFrame({"metric": ["gini", "a|b"], "value": [0.68, 1]}).to_json(orient="split"),
         "rows": 2}
CHAT = [
    {"role": "user", "content": "load C:/x", "artifacts": [], "meta": {}},
    {"role": "assistant", "content": "Loaded x.", "artifacts": [], "meta": {"kind": "load"}},
    {"role": "user", "content": "What is the Gini?", "artifacts": [], "meta": {}},
    {"role": "assistant", "content": "The Gini is 0.68 [test:abc123].",
     "artifacts": [TABLE, {"type": "plotly", "title": "ROC", "path": "x.json"}],
     "meta": {"test_runs": ["abc123"]}},
]


def test_exchange_has_the_question_and_the_answer_only():
    md = export.exchange_md(CHAT, 3, "Gini check", "Model Lens")
    assert md.startswith("# Gini check")
    assert "## You\n\nWhat is the Gini?" in md and "## Model Lens\n\nThe Gini is 0.68" in md
    assert "load C:/x" not in md
    assert "| metric | value |" in md and "| a\\|b | 1.0 |" in md        # pipes in cells escaped
    assert "_Chart: ROC (shown in the app)_" in md and "Test runs: `abc123`" in md


def test_full_chat_keeps_every_turn_in_order():
    md = export.chat_md(CHAT, "Gini check")
    order = [md.index(s) for s in ("load C:/x", "Loaded x.", "What is the Gini?", "The Gini is 0.68")]
    assert order == sorted(order) and md.count("\n---\n") == 3


def test_long_tables_are_cut_and_say_so():
    big = dict(TABLE, data=pd.DataFrame({"i": range(80)}).to_json(orient="split"), rows=80)
    md = export.message_md({"role": "assistant", "content": "x", "artifacts": [big]})
    assert "| 49 |" in md and "| 50 |" not in md and "First 50 of 80 rows" in md


def test_mirror_replaces_the_copy_under_an_old_title(tmp_path):
    first = export.mirror(tmp_path, "c1", "Old: title?", CHAT[:2])
    assert first.name == "Old_title-c1.md"
    second = export.mirror(tmp_path, "c1", "New title", CHAT)
    assert not first.exists() and second.read_text(encoding="utf-8").startswith("# New title")
    assert [p.name for p in (tmp_path / "chats_md").iterdir()] == ["New_title-c1.md"]


def test_history_dir_is_per_user(tmp_path):
    a = preferences.set_history_dir("alice@bank.eu", str(tmp_path / "alice"))
    assert a == str(tmp_path / "alice") and (tmp_path / "alice").is_dir()
    assert preferences.history_dir("alice@bank.eu") == a and preferences.history_dir("bob@bank.eu") == ""
    assert preferences.set_history_dir("alice@bank.eu", "  ") == ""
    assert preferences.history_dir("alice@bank.eu") == ""


def test_history_dir_must_be_absolute_and_writable(tmp_path):
    with pytest.raises(ValueError, match="full path"):
        preferences.set_history_dir("u", "relative/chats")
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(ValueError, match="Can't write"):
        preferences.set_history_dir("u", str(blocker / "sub"))
