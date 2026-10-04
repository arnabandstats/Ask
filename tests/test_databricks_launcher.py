"""The Databricks launcher notebook: format, syntax, and the parts that run outside Databricks."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "databricks_launcher.py"


def _cells() -> list[str]:
    text = NOTEBOOK.read_text(encoding="utf-8")
    return [c.strip("\n") for c in text.split("# COMMAND ----------")]


def _code_cells() -> list[str]:
    out = []
    for c in _cells():
        lines = [ln for ln in c.splitlines() if ln.strip()]
        if lines and not all(ln.startswith(("# MAGIC", "# Databricks notebook source")) for ln in lines):
            out.append(c)
    return out


def _cell_containing(marker: str) -> str:
    return next(c for c in _code_cells() if marker in c)


class _Widgets:
    def __init__(self, values):
        self.values = values

    def get(self, name):
        return self.values[name]

    def text(self, *a, **k):
        pass

    def dropdown(self, *a, **k):
        pass


class _FakeDbutils:
    def __init__(self, widgets, notebook_path="/Users/me/app/databricks_launcher"):
        self.widgets = _Widgets(widgets)
        path = notebook_path

        class _Ctx:
            def notebookPath(self):
                return type("O", (), {"get": lambda _s: path})()

        nb = type("NB", (), {"getContext": lambda _s: _Ctx()})()
        dbu = type("DBU", (), {"notebook": lambda _s: nb})()
        self.notebook = type("N", (), {"entry_point": type("E", (), {"getDbutils": lambda _s: dbu})()})()


def test_is_a_databricks_source_notebook():
    assert NOTEBOOK.read_text(encoding="utf-8").startswith("# Databricks notebook source\n")
    assert len(_cells()) >= 10


def test_every_code_cell_compiles():
    for cell in _code_cells():
        if cell.lstrip().startswith("# MAGIC %pip"):
            continue
        compile(cell, "<cell>", "exec")


def test_pip_cell_installs_the_generated_requirements():
    pip = next(c for c in _cells() if "%pip install" in c)
    assert "/tmp/ask_requirements.txt" in pip
    assert any("restartPython()" in c for c in _code_cells())


def test_requirements_cell_swaps_opencv_for_headless(tmp_path):
    cell = _cell_containing("_reqs = []")
    out_file = tmp_path / "reqs.txt"
    cell = cell.replace('"/Workspace" + ', "").replace("/tmp/ask_requirements.txt", str(out_file).replace("\\", "/"))
    ns = {"dbutils": _FakeDbutils({}, notebook_path=str(ROOT / "databricks_launcher"))}
    exec(cell, ns)
    reqs = out_file.read_text().split()
    assert "opencv-python-headless" in reqs and "opencv-python" not in reqs
    assert {"streamlit>=1.40", "openai>=1.0", "pandas", "pytest"} <= set(reqs)
    assert not any(r.startswith("#") or "  " in r for r in reqs)


def test_secrets_are_never_printed():
    start = _cell_containing("subprocess.Popen")
    for line in start.splitlines():
        if "print(" in line:
            assert "env[" not in line and "secrets" not in line


def test_start_cell_uses_proxy_safe_streamlit_flags():
    start = _cell_containing("subprocess.Popen")
    for flag in ('"--server.address", "0.0.0.0"', '"--server.headless", "true"',
                 '"--server.enableXsrfProtection", "false"', '"--server.enableCORS", "false"'):
        assert flag in start
    assert 'env["ASK_DATA_DIR"] = DATA_DIR' in start
    assert "start_new_session=True" in start and "/_stcore/health" in start


def test_backup_cell_copies_a_consistent_db(tmp_path):
    data, backup = tmp_path / "data", tmp_path / "vol" / "ask"
    data.mkdir()
    with sqlite3.connect(data / "chats.db") as con:
        con.execute("CREATE TABLE chats (id TEXT)")
        con.execute("INSERT INTO chats VALUES ('abc')")
    (data / "preferences.json").write_text('{"tool_name": "Ask"}')
    cell = _cell_containing("def backup_chats").replace("/tmp/chats_backup.db",
                                                        str(tmp_path / "snap.db").replace("\\", "/"))
    exec(cell, {"dbutils": _FakeDbutils({"data_dir": str(data), "backup_dir": str(backup)})})
    with sqlite3.connect(backup / "chats.db") as con:
        assert con.execute("SELECT id FROM chats").fetchall() == [("abc",)]
    assert (backup / "preferences.json").read_text() == '{"tool_name": "Ask"}'


def test_backup_without_folder_is_a_noop(tmp_path, capsys):
    cell = _cell_containing("def backup_chats")
    exec(cell, {"dbutils": _FakeDbutils({"data_dir": str(tmp_path), "backup_dir": ""})})
    assert "No backup folder set" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["app.py", "requirements.txt"])
def test_launcher_sits_next_to_the_app(name):
    assert (NOTEBOOK.parent / name).exists()
