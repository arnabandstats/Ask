"""The Databricks launcher notebook: format, syntax, and what it passes to the app."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "databricks_launcher_cluster.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells() -> list[str]:
    return ["".join(c["source"]) for c in _notebook()["cells"]]


def _start_cell() -> str:
    return next(c for c in _cells() if "subprocess.Popen" in c)


def test_is_a_jupyter_notebook_of_code_cells():
    nb = _notebook()
    assert nb["nbformat"] == 4
    assert [c["cell_type"] for c in nb["cells"]] == ["code"] * 3


def test_pip_installs_requirements_first():
    assert _cells()[0] == "%pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt"


def test_python_cells_compile():
    for cell in _cells()[1:]:
        compile(cell, "<cell>", "exec")


def test_settings():
    start = _start_cell()
    assert "PORT = 8502" in start and 'APP_FILE = "app.py"' in start
    for name in ("AZURE_KEY_SECRET", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT"):
        assert f'{name} = ""' in start                  # nothing secret or org-specific committed


def test_app_gets_azure_settings_and_a_writable_data_dir():
    start = _start_cell()
    assert 'app_env["USE_AZURE_OPENAI"] = "true"' in start
    assert 'scope="OneLab-SecretScope"' in start and "key=AZURE_KEY_SECRET" in start
    for var in ("AZURE_OPENAI_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_VERSION",
                "OPENAI_DEPLOYMENT_NAME"):
        assert f'app_env["{var}"]' in start
    assert 'app_env["ASK_DATA_DIR"] = "/tmp/ask_data"' in start


def test_env_vars_match_what_the_app_reads():
    src = (ROOT / "ask" / "config.py").read_text(encoding="utf-8") + (ROOT / "ask" / "llm.py").read_text(encoding="utf-8")
    for var in ("USE_AZURE_OPENAI", "AZURE_OPENAI_KEY", "AZURE_OPENAI_BASE_URL", "AZURE_OPENAI_VERSION",
                "OPENAI_DEPLOYMENT_NAME", "ASK_DATA_DIR"):
        assert var in src, var


def test_secrets_are_never_printed():
    for cell in _cells():
        for line in cell.splitlines():
            if "print(" in line:
                assert "app_env" not in line and "secrets" not in line


def test_streamlit_command_and_link():
    start = _start_cell()
    for flag in ('"--server.address", "0.0.0.0"', '"--server.port", str(PORT)', '"--server.headless", "true"'):
        assert flag in start
    assert "start_new_session=True" in start
    assert 'f"https://{browser_host}/driver-proxy/o/{workspace_id}/{cluster_id}/{PORT}/"' in start


def test_context_text_unwraps_scala_options():
    import ast
    node = next(n for n in ast.parse(_start_cell()).body
                if isinstance(n, ast.FunctionDef) and n.name == "context_text")
    ns = {}
    exec(compile(ast.Module([node], []), "<cell>", "exec"), ns)
    fn = ns["context_text"]
    assert fn("Some(1234567890)") == "1234567890"
    assert fn("None") == "" and fn("null") == ""


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
