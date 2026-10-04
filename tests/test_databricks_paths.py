"""Loading /Workspace and /Volumes paths through the Databricks API (fake client, real SDK types)."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("databricks.sdk")

from databricks.sdk.errors import NotFound, PermissionDenied  # noqa: E402
from databricks.sdk.service.workspace import Language, ObjectInfo, ObjectType  # noqa: E402

from ask.sources import databricks as dbx  # noqa: E402
from ask.sources.loaders import LoadError, load_path  # noqa: E402

TRAIN = "MIN_ACCURACY = 0.85\n\ndef passes(m):\n    return m['accuracy'] >= MIN_ACCURACY\n"
NOTEBOOK = "# Databricks notebook source\nprint('train')\n"


class FakeWorkspace:
    """/Users/me/MLOps with a file, a notebook, a skipped binary and a .git folder."""

    def __init__(self, stamp=1):
        self.stamp, self.downloads = stamp, []
        self.objects = {
            "/Users/me/MLOps": ObjectInfo(path="/Users/me/MLOps", object_type=ObjectType.DIRECTORY),
            "/Users/me/MLOps/src/train.py": ObjectInfo(path="/Users/me/MLOps/src/train.py",
                                                       object_type=ObjectType.FILE, size=len(TRAIN)),
            "/Users/me/MLOps/notebooks/01_train": ObjectInfo(path="/Users/me/MLOps/notebooks/01_train",
                                                             object_type=ObjectType.NOTEBOOK,
                                                             language=Language.PYTHON),
            "/Users/me/MLOps/logo.png": ObjectInfo(path="/Users/me/MLOps/logo.png", object_type=ObjectType.FILE),
            "/Users/me/MLOps/.git/config": ObjectInfo(path="/Users/me/MLOps/.git/config",
                                                      object_type=ObjectType.FILE),
            "/Users/me/MLOps/README.md": ObjectInfo(path="/Users/me/MLOps/README.md", object_type=ObjectType.FILE),
        }

    def _stamped(self):
        for o in self.objects.values():
            o.modified_at = self.stamp

    def get_status(self, path):
        if path == "/Users/me/secret":
            raise PermissionDenied("User does not have READ on /Users/me/secret")
        if path not in self.objects:
            raise NotFound(f"Path ({path}) doesn't exist.")
        self._stamped()
        return self.objects[path]

    def list(self, path, recursive=False):
        self._stamped()
        return [o for p, o in self.objects.items() if p.startswith(path + "/")]

    def download(self, path, format=None):
        self.downloads.append(path)
        body = {"/Users/me/MLOps/src/train.py": TRAIN, "/Users/me/MLOps/README.md": "# MLOps\nCI/CD demo.\n"}
        return io.BytesIO(body.get(path, "x").encode())

    def export(self, path, format=None):
        self.downloads.append(path)
        return SimpleNamespace(content=base64.b64encode(NOTEBOOK.encode()).decode())


class FakeFiles:
    def __init__(self):
        self.files = {"/Volumes/cat/sch/vol/data/portfolio.csv": b"x,y\n1,0\n2,1\n",
                      "/Volumes/cat/sch/vol/data/notes.bin": b"\x00"}

    def get_metadata(self, path):
        if path not in self.files:
            raise NotFound("not a file")
        return SimpleNamespace(last_modified="t1")

    def list_directory_contents(self, folder):
        return [SimpleNamespace(path=p, name=Path(p).name, is_directory=False, file_size=len(b), last_modified="t1")
                for p, b in self.files.items() if str(Path(p).parent).replace("\\", "/") == folder]

    def download(self, path):
        return SimpleNamespace(contents=io.BytesIO(self.files[path]))


@pytest.fixture
def fake(monkeypatch):
    ws, files = FakeWorkspace(), FakeFiles()
    client = SimpleNamespace(workspace=ws, files=files)
    monkeypatch.setattr(dbx, "_client", client)
    return client


class TestPathRecognition:
    @pytest.mark.parametrize("text", ["/Workspace/Users/a@b.com/MLOps", "/Volumes/c/s/v/f.csv",
                                      "/Users/a@b.com/x", "/Repos/a/b", "/Shared/team", "`/Workspace/Users/x`"])
    def test_databricks_paths(self, text):
        assert dbx.looks_like_path(text)

    @pytest.mark.parametrize("text", ["C:/Users/me/x", "/home/me/x", "dbfs:/mnt/x", "Workspace/x"])
    def test_other_paths(self, text):
        assert not dbx.looks_like_path(text)

    @pytest.mark.parametrize("raw,norm", [("/Users/a/x/", "/Workspace/Users/a/x"),
                                          ("/Workspace/Users/a/x.", "/Workspace/Users/a/x"),
                                          ("\\Workspace\\Users\\a", "/Workspace/Users/a")])
    def test_normalise(self, raw, norm):
        assert dbx.normalise(raw) == norm

    def test_browse_url_has_no_path(self):
        url = "https://dbc-1.cloud.databricks.com/browse/folders/2615473909626896?o=7474"
        assert dbx.url_in(f"{url}\nload this path") == url and dbx.path_from_url(url) is None

    def test_old_style_url_carries_path(self):
        url = "https://dbc-1.cloud.databricks.com/?o=1#workspace/Users/a%40b.com/MLOps"
        assert dbx.path_from_url(url) == "/Workspace/Users/a@b.com/MLOps"


class TestWorkspaceFolder:
    def test_loads_files_and_notebooks_keeps_databricks_path(self, fake):
        [src] = load_path(Path("/Workspace/Users/me/MLOps"))
        assert src.name == "MLOps" and src.path == "/Workspace/Users/me/MLOps" and src.kind == "repo"
        assert set(src.files) == {"src/train.py", "notebooks/01_train.py", "README.md"}
        assert src.files["src/train.py"] == TRAIN and "print('train')" in src.files["notebooks/01_train.py"]
        assert "logo.png" not in str(fake.workspace.downloads) and ".git" not in str(fake.workspace.downloads)

    def test_unchanged_files_are_not_downloaded_again(self, fake):
        load_path(Path("/Workspace/Users/me/MLOps"))
        first = len(fake.workspace.downloads)
        load_path(Path("/Workspace/Users/me/MLOps"))
        assert len(fake.workspace.downloads) == first             # all fresh
        fake.workspace.stamp = 2                                  # everything modified
        load_path(Path("/Workspace/Users/me/MLOps"))
        assert len(fake.workspace.downloads) == 2 * first

    def test_single_file(self, fake):
        [src] = load_path(Path("/Workspace/Users/me/MLOps/src/train.py"))
        assert list(src.files) == ["train.py"] and src.path == "/Workspace/Users/me/MLOps/src/train.py"

    def test_missing_path_explains_sharing(self, fake):
        with pytest.raises(LoadError, match="does not exist, or .* can't see it.*Share"):
            load_path(Path("/Workspace/Users/me/nope"))

    def test_permission_denied(self, fake):
        with pytest.raises(LoadError, match="no access to `/Workspace/Users/me/secret`"):
            load_path(Path("/Workspace/Users/me/secret"))

    def test_app_service_principal_named_in_errors(self, fake, monkeypatch):
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "abc-123")
        with pytest.raises(LoadError, match="service principal \\(client ID abc-123\\)"):
            load_path(Path("/Workspace/Users/me/nope"))


class TestVolumes:
    def test_single_csv(self, fake):
        [src] = load_path(Path("/Volumes/cat/sch/vol/data/portfolio.csv"))
        assert src.kind == "data" and src.df.shape == (2, 2) and src.path == "/Volumes/cat/sch/vol/data/portfolio.csv"

    def test_folder_of_data_files(self, fake):
        [src] = load_path(Path("/Volumes/cat/sch/vol/data"))        # notes.bin is skipped
        assert src.kind == "data" and src.path == "/Volumes/cat/sch/vol/data/portfolio.csv"


class TestWithoutDatabricks:
    def test_no_credentials_gives_clear_message(self, monkeypatch):
        monkeypatch.setattr(dbx, "_client", None)

        class Broken:
            def __init__(self):
                raise ValueError("default auth: cannot configure default credentials")
        import databricks.sdk as sdk
        monkeypatch.setattr(sdk, "WorkspaceClient", Broken)
        with pytest.raises(LoadError, match="can't reach Databricks from here.*databricks auth login"):
            load_path(Path("/Workspace/Users/me/MLOps"))


class TestChat:
    def test_fast_load_of_workspace_path(self, fake):
        from ask.agent.router import _fast_load
        from ask.sources.registry import SourceRegistry
        reg = SourceRegistry()
        turn = _fast_load("/Workspace/Users/me/MLOps load this repo", reg)
        assert turn.content.startswith("Loaded repo **MLOps**: 3 files") and turn.sources_changed
        assert reg.records() == [{"name": "MLOps", "kind": "repo", "path": "/Workspace/Users/me/MLOps"}]

    def test_restore_reloads_from_databricks(self, fake):
        from ask.sources.registry import SourceRegistry
        reg = SourceRegistry()
        assert reg.restore([{"name": "MLOps", "kind": "repo", "path": "/Workspace/Users/me/MLOps"}]) == []
        assert "src/train.py" in reg.sources["MLOps"].files

    def test_browser_link_gets_help_not_path_not_found(self):
        from ask.agent.router import _fast_load
        from ask.sources.registry import SourceRegistry
        turn = _fast_load("https://dbc-1.cloud.databricks.com/browse/folders/26154?o=7\nload this path",
                          SourceRegistry())
        assert "Copy path" in turn.content and "Path not found" not in turn.content

    def test_agent_tool_loads_databricks_path(self, fake, tmp_path):
        import json
        from ask.agent.tools import ToolContext, dispatch
        from ask.sources.registry import SourceRegistry
        ctx = ToolContext(registry=SourceRegistry(), output_dir=tmp_path)
        out = dispatch(ctx, "load_path", json.dumps({"path": "/Workspace/Users/me/MLOps"}))
        assert out.startswith("Loaded [repo] MLOps")
