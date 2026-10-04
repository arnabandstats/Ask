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


class _Secrets:
    """Fake dbutils.secrets: {scope: {key: value}}."""

    def __init__(self, store):
        self.store = store

    def listScopes(self):
        return [type("S", (), {"name": s})() for s in self.store]

    def list(self, scope):
        return [type("K", (), {"key": k})() for k in self.store[scope]]

    def get(self, scope, key):
        return self.store[scope][key]


class _FakeSpark:
    """mode=None behaves like serverless: cluster settings raise CONFIG_NOT_AVAILABLE."""

    def __init__(self, mode):
        def get(_self, key, *default):
            if key.startswith("spark.databricks.clusterUsageTags.") and mode is None:
                raise RuntimeError("[CONFIG_NOT_AVAILABLE.WITHOUT_SUGGESTION] Configuration "
                                   f"{key} is not available. SQLSTATE: 42K0I")
            if key.endswith("dataSecurityMode"):
                return mode
            if key.endswith("clusterId"):
                return "1004-000000-abcdef"
            return default[0] if default else None
        self.conf = type("C", (), {"get": get})()


class _FakeDbutils:
    def __init__(self, widgets, notebook_path="/Users/me/app/databricks_launcher", secrets=None):
        self.widgets = _Widgets(widgets)
        self.secrets = _Secrets(secrets or {})
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


def _requirements_for(mode, tmp_path):
    mode_file, out_file = tmp_path / "mode.txt", tmp_path / "reqs.txt"
    mode_file.write_text(mode)
    cell = (_cell_containing('_reqs = ["databricks-sdk')
            .replace('"/Workspace" + ', "")
            .replace("/tmp/ask_requirements.txt", str(out_file).replace("\\", "/"))
            .replace("/tmp/ask_launcher_mode.txt", str(mode_file).replace("\\", "/")))
    exec(cell, {"dbutils": _FakeDbutils({}, notebook_path=str(ROOT / "databricks_launcher"))})
    return out_file.read_text().split()


def test_cluster_mode_installs_app_requirements_with_headless_opencv(tmp_path):
    reqs = _requirements_for("cluster", tmp_path)
    assert "opencv-python-headless" in reqs and "opencv-python" not in reqs
    assert {"streamlit>=1.40", "openai>=1.0", "pandas", "pytest", "databricks-sdk>=0.50"} <= set(reqs)
    assert not any(r.startswith("#") or "  " in r for r in reqs)


def test_app_mode_installs_only_the_sdk(tmp_path):
    assert _requirements_for("app", tmp_path) == ["databricks-sdk>=0.50"]


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


@pytest.fixture
def fake_tmp(tmp_path, monkeypatch):
    """Point tempfile.gettempdir() (where the launcher keeps its state file) at a test folder."""
    import tempfile
    d = tmp_path / "systmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


def _make_db(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(folder / "chats.db") as con:
        con.execute("CREATE TABLE chats (id TEXT)")
        con.execute("INSERT INTO chats VALUES ('abc')")
    (folder / "preferences.json").write_text('{"tool_name": "Ask"}')


def test_backup_cell_copies_a_consistent_db(tmp_path, fake_tmp):
    data, backup = tmp_path / "data", tmp_path / "vol" / "ask"
    _make_db(data)
    cell = _cell_containing("def backup_chats")
    exec(cell, {"dbutils": _FakeDbutils({"port": "8599", "data_dir": str(data), "backup_dir": str(backup)})})
    with sqlite3.connect(backup / "chats.db") as con:
        assert con.execute("SELECT id FROM chats").fetchall() == [("abc",)]
    assert (backup / "preferences.json").read_text() == '{"tool_name": "Ask"}'


def test_backup_uses_folder_recorded_by_start_cell(tmp_path, fake_tmp):
    """If Start fell back to another folder (e.g. /tmp), later cells must follow it, not the widget."""
    import json
    actual, backup = tmp_path / "fallback_data", tmp_path / "vol"
    _make_db(actual)
    (fake_tmp / "ask_launcher_8599.json").write_text(json.dumps({
        "data_dir": str(actual), "log_file": str(actual / "streamlit_8599.log"),
        "pid_file": str(actual / "streamlit_8599.pid")}))
    cell = _cell_containing("def backup_chats")
    widgets = {"port": "8599", "data_dir": "/local_disk0/ask_data", "backup_dir": str(backup)}
    exec(cell, {"dbutils": _FakeDbutils(widgets)})
    assert (backup / "chats.db").exists()


def test_backup_without_folder_is_a_noop(tmp_path, fake_tmp, capsys):
    cell = _cell_containing("def backup_chats")
    exec(cell, {"dbutils": _FakeDbutils({"port": "8599", "data_dir": str(tmp_path), "backup_dir": ""})})
    assert "No backup folder set" in capsys.readouterr().out


def test_log_cell_before_start(tmp_path, fake_tmp, capsys):
    cell = _cell_containing("_log = launcher_state()")
    exec(cell, {"dbutils": _FakeDbutils({"port": "8599", "data_dir": str(tmp_path / "none")})})
    assert "No log yet" in capsys.readouterr().out


class TestWritableFolder:
    @staticmethod
    def _fn():
        import ast
        start = _cell_containing("def _first_writable")
        node = next(n for n in ast.parse(start).body
                    if isinstance(n, ast.FunctionDef) and n.name == "_first_writable")
        ns = {"os": __import__("os")}
        exec(compile(ast.Module([node], []), "<cell>", "exec"), ns)
        return ns["_first_writable"]

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

    def test_local_disk_then_tmp_are_candidates(self):
        start = _cell_containing("def _first_writable")
        assert start.index('"/local_disk0/ask_data"') < start.index('"/tmp/ask_data"')
        assert "STATE_FILE" in start and "json.dump" in start


class TestSecretCheck:
    FAKE_KEY = "sk-test-" + "Z" * 40

    def _run(self, store, scope="ask", key="openai-api-key", mode="SINGLE_USER", run_mode="auto",
             mode_file=None):
        widgets = {"secret_scope": scope, "secret_key": key, "run_mode": run_mode}
        ns = {"dbutils": _FakeDbutils(widgets, secrets=store), "spark": _FakeSpark(mode)}
        cell = _cell_containing("def check_api_key_secret")
        if mode_file is None:
            import tempfile
            mode_file = Path(tempfile.mkdtemp()) / "mode.txt"
        cell = cell.replace("/tmp/ask_launcher_mode.txt", str(mode_file).replace("\\", "/"))
        exec(cell, ns)
        return ns

    @pytest.mark.parametrize("cluster_mode,expected", [
        (None, "app"),                                  # serverless
        ("USER_ISOLATION", "app"),                      # Shared
        ("DATA_SECURITY_MODE_STANDARD", "app"),         # Standard
        ("SINGLE_USER", "cluster"),                     # Dedicated
        ("DATA_SECURITY_MODE_DEDICATED", "cluster"),
    ])
    def test_auto_mode_picks_what_works(self, tmp_path, capsys, cluster_mode, expected):
        mode_file = tmp_path / "mode.txt"
        ns = self._run({"ask": {"openai-api-key": self.FAKE_KEY}}, mode=cluster_mode, mode_file=mode_file)
        assert ns["RUN_MODE"] == expected and mode_file.read_text() == expected
        out = capsys.readouterr().out
        assert "API key found" in out                    # secret checked first
        assert ("Databricks App" in out) == (expected == "app")

    def test_forced_app_mode_on_dedicated(self, tmp_path):
        ns = self._run({"ask": {"openai-api-key": self.FAKE_KEY}}, mode="SINGLE_USER",
                       run_mode="databricks_app", mode_file=tmp_path / "m.txt")
        assert ns["RUN_MODE"] == "app"

    @pytest.mark.parametrize("cluster_mode", [None, "USER_ISOLATION"])
    def test_forced_cluster_mode_where_link_is_blocked_explains(self, tmp_path, cluster_mode):
        with pytest.raises(RuntimeError, match="needs a cluster in Dedicated.*'auto' or 'databricks_app'"):
            self._run({"ask": {"openai-api-key": self.FAKE_KEY}}, mode=cluster_mode, run_mode="cluster",
                      mode_file=tmp_path / "m.txt")

    def test_open_cell_handles_serverless(self):
        cell = _cell_containing("driver-proxy/o/")
        ns = {"spark": _FakeSpark(None), "PORT": 8501, "displayHTML": lambda html: None}
        with pytest.raises(RuntimeError, match="serverless has no driver proxy"):
            exec(cell, ns)

    def test_open_cell_shows_link_on_dedicated(self):
        shown = []
        spark = _FakeSpark("SINGLE_USER")
        real_get = spark.conf.get
        spark.conf.get = lambda key, *d: {"spark.databricks.workspaceUrl": "dbc-1.cloud.databricks.com",
                                          "spark.databricks.clusterUsageTags.clusterOwnerOrgId": "123"
                                          }.get(key) or real_get(key, *d)
        exec(_cell_containing("driver-proxy/o/"), {"spark": spark, "PORT": 8501, "displayHTML": shown.append})
        assert "driver-proxy/o/123/1004-000000-abcdef/8501/" in shown[0] and "Shared/Standard" not in shown[0]

    def test_widget_defaults_point_at_ask_scope(self):
        settings = _cell_containing('dbutils.widgets.text("port"')
        assert 'dbutils.widgets.text("secret_scope", "ask"' in settings
        assert 'dbutils.widgets.text("secret_key", "openai-api-key"' in settings

    def test_found_and_value_never_printed(self, capsys):
        self._run({"ask": {"openai-api-key": self.FAKE_KEY}})
        out = capsys.readouterr().out
        assert "API key found in secret 'ask/openai-api-key'" in out
        assert self.FAKE_KEY not in out and "ZZZZ" not in out

    def test_missing_scope_explains_fix(self):
        with pytest.raises(ValueError, match="Secret scope 'ask' not found.*create-scope ask"):
            self._run({"other": {}})

    def test_missing_key_lists_available_names(self):
        with pytest.raises(ValueError, match="Key 'openai-api-key' not found.*Keys there: azure-key"):
            self._run({"ask": {"azure-key": "x"}})

    def test_empty_secret(self):
        with pytest.raises(ValueError, match="is empty"):
            self._run({"ask": {"openai-api-key": "   "}})

    def test_whitespace_is_flagged_and_trimmed_at_start(self, capsys):
        self._run({"ask": {"openai-api-key": self.FAKE_KEY + "\n"}})
        assert "will be trimmed" in capsys.readouterr().out
        assert '.get(scope, key).strip()' in _cell_containing("subprocess.Popen")

    def test_blank_scope_falls_back_to_cluster_key(self, capsys):
        self._run({}, scope="")
        assert "API key defined on the cluster" in capsys.readouterr().out

    def test_check_runs_before_install(self):
        cells = _cells()
        check = next(i for i, c in enumerate(cells) if "def check_api_key_secret" in c)
        install = next(i for i, c in enumerate(cells) if "%pip install" in c)
        assert check < install


class TestDeployDatabricksApp:
    """The app-mode deploy, against a fake of the Databricks SDK's Apps API (real SDK types)."""

    URL = "https://ask-arnab-123.aws.databricksapps.com"

    @pytest.fixture(autouse=True)
    def _sdk(self):
        pytest.importorskip("databricks.sdk")

    @staticmethod
    def _fns():
        import ast
        cell = _cell_containing("def deploy_databricks_app")
        tree = ast.parse(cell)
        keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        ns = {}
        exec(compile(ast.Module(keep, []), "<cell>", "exec"), ns)
        return ns

    def _fake(self, exists=False, state="ACTIVE", result="SUCCEEDED", deploy_error=None, get_error=None):
        from types import SimpleNamespace
        from databricks.sdk.errors import NotFound
        from databricks.sdk.service.apps import AppDeploymentState, ComputeState
        url = self.URL

        class Apps:
            def __init__(self):
                self.calls, self.exists, self.state = [], exists, state

            def get(self, name):
                if get_error:
                    raise get_error
                if not self.exists:
                    raise NotFound(f"App with name {name} does not exist.")
                return SimpleNamespace(url=url, compute_status=SimpleNamespace(state=ComputeState(self.state)))

            def create_and_wait(self, app, timeout):
                self.calls.append(("create", app))
                self.exists = True
                return app

            def update(self, name, app):
                self.calls.append(("update", name, app))

            def start_and_wait(self, name, timeout):
                self.calls.append(("start", name))
                self.state = "ACTIVE"

            def deploy_and_wait(self, app_name, app_deployment, timeout):
                self.calls.append(("deploy", app_name, app_deployment))
                if deploy_error:
                    raise deploy_error
                return SimpleNamespace(status=SimpleNamespace(state=AppDeploymentState(result),
                                                              message="requirements install failed"))

        return SimpleNamespace(apps=Apps())

    def test_creates_app_with_secret_resource_and_deploys_folder(self):
        from databricks.sdk.service.apps import AppDeploymentMode, AppResourceSecretSecretPermission
        w, logs = self._fake(), []
        url = self._fns()["deploy_databricks_app"](w, "ask-arnab", "/Workspace/Users/a/Ask", "ask",
                                                   "openai-api-key", log=logs.append)
        assert url == self.URL
        kinds = [c[0] for c in w.apps.calls]
        assert kinds == ["create", "deploy"]
        app = w.apps.calls[0][1]
        res = app.resources[0]
        assert app.name == "ask-arnab" and res.name == "openai-api-key"          # == app.yaml valueFrom
        assert (res.secret.scope, res.secret.key) == ("ask", "openai-api-key")
        assert res.secret.permission == AppResourceSecretSecretPermission.READ
        dep = w.apps.calls[1][2]
        assert dep.source_code_path == "/Workspace/Users/a/Ask" and dep.mode == AppDeploymentMode.SNAPSHOT
        assert any("Creating app" in m for m in logs)

    def test_existing_app_is_updated_not_recreated(self):
        w = self._fake(exists=True)
        self._fns()["deploy_databricks_app"](w, "ask-arnab", "/src", "ask", "k", log=lambda m: None)
        assert [c[0] for c in w.apps.calls] == ["update", "deploy"]

    def test_stopped_app_is_started_before_deploy(self):
        w = self._fake(exists=True, state="STOPPED")
        self._fns()["deploy_databricks_app"](w, "ask-arnab", "/src", "ask", "k", log=lambda m: None)
        assert [c[0] for c in w.apps.calls] == ["update", "start", "deploy"]

    def test_failed_deployment_points_to_logs(self):
        w = self._fake(result="FAILED")
        with pytest.raises(RuntimeError, match=r"failed: FAILED: requirements install failed.*Logs.*/logz"):
            self._fns()["deploy_databricks_app"](w, "ask-arnab", "/src", "ask", "k", log=lambda m: None)

    def test_sdk_operation_failure_is_explained(self):
        w = self._fake(deploy_error=RuntimeError("OperationFailed: app crashed on start"))
        with pytest.raises(RuntimeError, match="app crashed on start.*Logs"):
            self._fns()["deploy_databricks_app"](w, "ask-arnab", "/src", "ask", "k", log=lambda m: None)

    def test_permission_error_is_not_mistaken_for_missing_app(self):
        from databricks.sdk.errors import PermissionDenied
        w = self._fake(get_error=PermissionDenied("User does not have CAN_MANAGE on app ask-arnab"))
        with pytest.raises(PermissionDenied):
            self._fns()["deploy_databricks_app"](w, "ask-arnab", "/src", "ask", "k", log=lambda m: None)
        assert w.apps.calls == []

    @pytest.mark.parametrize("user,name", [
        ("arnabandstats@gmail.com", "ask-arnabandstats"),
        ("First.Last@corp.com", "ask-first-last"),
        ("a" * 60 + "@x.com", "ask-" + "a" * 26),
        ("___@x.com", "ask"),
    ])
    def test_default_app_name(self, user, name):
        got = self._fns()["default_app_name"](user)
        assert got == name and len(got) <= 30

    def test_cell_deploys_shows_link_and_ends_the_notebook(self, tmp_path, monkeypatch):
        import databricks.sdk as sdk
        mode_file = tmp_path / "mode.txt"
        mode_file.write_text("app")
        fake = self._fake()
        from types import SimpleNamespace
        fake.current_user = SimpleNamespace(me=lambda: SimpleNamespace(user_name="arnabandstats@gmail.com"))
        monkeypatch.setattr(sdk, "WorkspaceClient", lambda: fake)
        exits, shown = [], []
        db = _FakeDbutils({"app_name": "", "secret_scope": "ask", "secret_key": "openai-api-key"},
                          notebook_path="/Users/arnab/Ask/databricks_launcher")
        db.notebook.exit = exits.append
        cell = _cell_containing("def deploy_databricks_app").replace(
            "/tmp/ask_launcher_mode.txt", str(mode_file).replace("\\", "/"))
        exec(cell, {"dbutils": db, "displayHTML": shown.append})
        assert exits == [self.URL] and self.URL in shown[0]
        assert fake.apps.calls[0][1].name == "ask-arnabandstats"
        assert fake.apps.calls[-1][2].source_code_path == "/Workspace/Users/arnab/Ask"

    def test_cell_skips_in_cluster_mode(self, tmp_path, capsys):
        mode_file = tmp_path / "mode.txt"
        mode_file.write_text("cluster")
        cell = _cell_containing("def deploy_databricks_app").replace(
            "/tmp/ask_launcher_mode.txt", str(mode_file).replace("\\", "/"))
        exec(cell, {"dbutils": _FakeDbutils({}), "displayHTML": lambda h: None})
        assert "skipping the Databricks App deployment" in capsys.readouterr().out


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

    def test_env_vars_match_what_the_app_reads(self):
        from ask import config
        src = (ROOT / "ask" / "config.py").read_text(encoding="utf-8") + (ROOT / "ask" / "llm.py").read_text(encoding="utf-8")
        for var in ("OPENAI_API_KEY", "USE_AZURE_OPENAI", "ASK_DATA_DIR"):
            assert var in src, var
        assert config.DATA_DIR is not None


@pytest.mark.parametrize("name", ["app.py", "requirements.txt"])
def test_launcher_sits_next_to_the_app(name):
    assert (NOTEBOOK.parent / name).exists()
