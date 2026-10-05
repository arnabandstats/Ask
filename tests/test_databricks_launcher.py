"""The Databricks launcher notebook: format, syntax, and the parts that run outside Databricks."""
from __future__ import annotations

import ast
import json
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


def _functions(marker: str, ns: dict | None = None) -> dict:
    """Only the imports and function definitions of the cell containing `marker`."""
    tree = ast.parse(_cell_containing(marker))
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            and not (isinstance(n, ast.ImportFrom) and n.module.startswith("IPython"))]   # Databricks only
    ns = dict(ns or {})
    exec(compile(ast.Module(keep, []), "<cell>", "exec"), ns)
    return ns


class _Secrets:
    """Fake dbutils.secrets: {scope: {key: value}}."""

    def __init__(self, store):
        self.store = store

    def list(self, scope):
        return [type("K", (), {"key": k})() for k in self.store[scope]]

    def get(self, scope, key):
        return self.store[scope][key]


class _FakeDbutils:
    def __init__(self, secrets=None):
        self.secrets = _Secrets(secrets or {})


@pytest.fixture
def fake_tmp(tmp_path, monkeypatch):
    """Point tempfile.gettempdir() (where the launcher keeps its state file) at a test folder."""
    import tempfile
    d = tmp_path / "systmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


# ── format ──────────────────────────────────────────────────────────────

def test_is_a_databricks_source_notebook():
    assert NOTEBOOK.read_text(encoding="utf-8").startswith("# Databricks notebook source\n")


def test_every_code_cell_compiles():
    for cell in _code_cells():
        if cell.lstrip().startswith("# MAGIC %pip"):
            continue
        compile(cell, "<cell>", "exec")


def test_pip_installs_requirements_first():
    cells = _cells()
    assert cells[1].strip() == "# MAGIC %pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt"


def test_settings_defaults():
    settings = _cell_containing("SECRET_SCOPE =")
    ns = {}
    exec(settings, ns)
    assert ns["PORT"] == 8502 and ns["APP_FILE"] == "app.py"
    assert ns["SECRET_SCOPE"] == "OneLab-SecretScope"
    assert ns["AZURE_KEY_SECRET"] == "" and ns["AZURE_ENDPOINT"] == ""     # nothing secret committed


# ── start cell ──────────────────────────────────────────────────────────

def test_start_uses_azure_and_never_prints_secrets():
    start = _cell_containing("subprocess.Popen")
    assert 'app_env["USE_AZURE_OPENAI"] = "true"' in start
    assert 'app_env["AZURE_OPENAI_KEY"] = read_secret(' in start
    for var in ("AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_VERSION", "OPENAI_DEPLOYMENT_NAME",
                "ASK_DEEP_MODEL", "ASK_DATA_DIR"):
        assert f'app_env["{var}"]' in start
    for line in start.splitlines():
        if "print(" in line:
            assert "app_env" not in line and "secrets" not in line


def test_env_vars_match_what_the_app_reads():
    src = (ROOT / "ask" / "config.py").read_text(encoding="utf-8") + (ROOT / "ask" / "llm.py").read_text(encoding="utf-8")
    for var in ("USE_AZURE_OPENAI", "AZURE_OPENAI_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_VERSION",
                "OPENAI_DEPLOYMENT_NAME", "ASK_DEEP_MODEL", "ASK_DATA_DIR"):
        assert var in src, var


def test_start_uses_proxy_safe_streamlit_flags():
    start = _cell_containing("subprocess.Popen")
    for flag in ('"--server.address", "0.0.0.0"', '"--server.headless", "true"',
                 '"--server.enableXsrfProtection", "false"', '"--server.enableCORS", "false"'):
        assert flag in start
    assert "start_new_session=True" in start and "/_stcore/health" in start


def test_proxy_link_uses_notebook_context_workspace_id():
    start = _cell_containing("subprocess.Popen")
    assert "context.workspaceId()" in start and "context.browserHostName()" in start
    assert 'f"https://{browser_host}/driver-proxy/o/{workspace_id}/{cluster_id}/{PORT}/"' in start


def test_rerun_stops_the_previous_app_even_after_a_python_restart():
    start = _cell_containing("subprocess.Popen")
    assert start.index("os.killpg") < start.index("subprocess.Popen(")
    assert "json.dump" in start and "pid_path" in start


class TestReadSecret:
    KEY = "az-" + "Z" * 40

    def _fn(self, store):
        return _functions("def read_secret", {"dbutils": _FakeDbutils(store)})["read_secret"]

    def test_reads_and_trims(self):
        assert self._fn({"OneLab-SecretScope": {"aoai": self.KEY + "\n"}})("OneLab-SecretScope", "aoai") == self.KEY

    def test_blank_key_lists_names_not_values(self):
        with pytest.raises(ValueError, match="Set AZURE_KEY_SECRET.*Keys in 'S': a, b") as err:
            self._fn({"S": {"b": self.KEY, "a": "x"}})("S", "")
        assert self.KEY not in str(err.value)

    def test_missing_key_explains(self):
        with pytest.raises(ValueError, match="Could not read secret 'S/nope'"):
            self._fn({"S": {}})("S", "nope")

    def test_empty_secret(self):
        with pytest.raises(ValueError, match="is empty"):
            self._fn({"S": {"k": "  "}})("S", "k")


def test_context_text_unwraps_scala_options():
    fn = _functions("def context_text")["context_text"]
    assert fn("Some(1234567890)") == "1234567890"
    assert fn("None") == "" and fn("null") == ""
    assert fn(" dbc-1.cloud.databricks.com ") == "dbc-1.cloud.databricks.com"


def test_spark_conf_without_spark_is_blank():
    assert _functions("def spark_conf")["spark_conf"]("spark.databricks.workspaceUrl") == ""


class TestWritableFolder:
    @staticmethod
    def _fn():
        return _functions("def first_writable")["first_writable"]

    def test_skips_unwritable_folders(self, tmp_path):
        blocker = tmp_path / "a_file"
        blocker.write_text("x")                     # a folder can't be created under a file
        good = tmp_path / "ok" / "ask_data"
        assert self._fn()(["", str(blocker / "ask_data"), str(good)]) == str(good)
        assert good.is_dir() and not (good / ".write_test").exists()

    def test_all_unwritable_explains(self, tmp_path):
        blocker = tmp_path / "f"
        blocker.write_text("x")
        with pytest.raises(PermissionError, match="No writable folder.*Tried"):
            self._fn()([str(blocker / "a"), str(blocker / "b")])


# ── log and stop cells ──────────────────────────────────────────────────

def _make_db(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(folder / "chats.db") as con:
        con.execute("CREATE TABLE chats (id TEXT)")
        con.execute("INSERT INTO chats VALUES ('abc')")
    (folder / "preferences.json").write_text('{"tool_name": "Ask"}')


def _launcher_state():
    return _functions("def launcher_state")["launcher_state"]


def test_log_cell_before_start(fake_tmp, capsys):
    exec(_cell_containing("_log = launcher_state"), {"PORT": 8599})
    assert "No log yet" in capsys.readouterr().out


def test_stop_before_start(fake_tmp, capsys):
    exec(_cell_containing("def backup_chats"), {"PORT": 8599, "BACKUP_DIR": "", "launcher_state": _launcher_state()})
    assert "No app started" in capsys.readouterr().out


def test_stop_backs_up_a_consistent_db_and_removes_pid(tmp_path, fake_tmp, capsys):
    data, backup = tmp_path / "data", tmp_path / "vol" / "ask"
    _make_db(data)
    pid_file = data / "streamlit_8599.pid"
    pid_file.write_text("999999999")                        # not a running process
    (fake_tmp / "ask_launcher_8599.json").write_text(json.dumps({
        "data_dir": str(data), "pid_file": str(pid_file), "log_file": str(data / "streamlit_8599.log")}))
    exec(_cell_containing("def backup_chats"),
         {"PORT": 8599, "BACKUP_DIR": str(backup), "launcher_state": _launcher_state()})
    with sqlite3.connect(backup / "chats.db") as con:
        assert con.execute("SELECT id FROM chats").fetchall() == [("abc",)]
    assert (backup / "preferences.json").read_text() == '{"tool_name": "Ask"}'
    assert not pid_file.exists()
    assert "Ask was not running" in capsys.readouterr().out


def test_backup_without_folder_is_a_noop(tmp_path, capsys):
    _functions("def backup_chats")["backup_chats"](str(tmp_path), "")
    assert "No BACKUP_DIR set" in capsys.readouterr().out


# ── app.yaml (manual Databricks Apps deployment) ────────────────────────

class TestDatabricksAppConfig:
    TEXT = (ROOT / "app.yaml").read_text(encoding="utf-8")

    def test_runs_streamlit_app(self):
        assert 'command: ["streamlit", "run", "app.py"]' in self.TEXT

    def test_key_comes_from_app_resource_not_literal(self):
        lines = self.TEXT.splitlines()
        i = next(n for n, ln in enumerate(lines) if "name: OPENAI_API_KEY" in ln)
        assert lines[i + 1].strip() == "valueFrom: openai-api-key"
        assert "sk-" not in self.TEXT

    def test_data_dir_is_writable_scratch(self):
        assert "name: ASK_DATA_DIR" in self.TEXT and "value: /tmp/ask_data" in self.TEXT

    def test_requirements_use_headless_opencv(self):
        reqs = [ln.split("#")[0].strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()]
        assert "opencv-python-headless" in reqs and "opencv-python" not in reqs


@pytest.mark.parametrize("name", ["app.py", "requirements.txt"])
def test_launcher_sits_next_to_the_app(name):
    assert (NOTEBOOK.parent / name).exists()
