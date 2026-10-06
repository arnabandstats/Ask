"""Read Databricks workspace folders and Volumes through the Databricks API.

On a classic Databricks cluster `/Workspace/...` and `/Volumes/...` are mounted as normal
folders, so the ordinary loaders read them directly. A **Databricks App** (and a laptop)
has no such mount: there these paths only exist behind the API. This module mirrors the
requested folder or file into a local cache (only the file types the app can read, only
what changed since last time) and hands the local copy to the normal loaders.

Authentication is whatever the Databricks SDK finds: the app's service principal inside a
Databricks App, or the Databricks CLI profile on a laptop.
"""
from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path, PurePosixPath

from ask import config

PREFIXES = ("/Workspace/", "/Volumes/", "/Users/", "/Repos/", "/Shared/")
NOTEBOOK_EXT = {"PYTHON": ".py", "SQL": ".sql", "SCALA": ".scala", "R": ".r"}
_URL = re.compile(r"https?://[^\s]*(?:cloud\.databricks\.com|azuredatabricks\.net|gcp\.databricks\.com)[^\s]*",
                  re.I)


class DatabricksError(Exception):
    """Readable error for the chat (no stack traces)."""


def looks_like_path(text: str) -> bool:
    s = text.strip().strip("`'\"").replace("\\", "/")
    if s.lower().startswith("dbfs:/"):
        return False
    return s.startswith(PREFIXES) or s in {p.rstrip("/") for p in PREFIXES}


def normalise(text: str) -> str:
    """'/Users/a/x' -> '/Workspace/Users/a/x'; trailing punctuation and slashes removed."""
    s = text.strip().strip("`'\"").replace("\\", "/").rstrip(".,;:!?)").rstrip("/")
    if s.startswith(("/Users/", "/Repos/", "/Shared/")):
        s = "/Workspace" + s
    return s


def url_in(text: str) -> str | None:
    m = _URL.search(text)
    return m.group(0) if m else None


def path_from_url(url: str) -> str | None:
    """Old-style links carry the path ('#workspace/Users/...'); new 'browse/folders/<id>' links don't."""
    m = re.search(r"#workspace(/[^?\s]+)", url)
    if m:
        from urllib.parse import unquote
        return normalise("/Workspace" + unquote(m.group(1)))
    return None


URL_HELP = ("That is a browser link: it holds a folder ID, not the path, so it can't be loaded. "
            "In Databricks click ⋮ next to the folder → **Copy path**, then paste the path "
            "(it starts with `/Workspace/...`).")


# ── client ─────────────────────────────────────────────────────────────────

_client = None


def client():
    """A WorkspaceClient, or a DatabricksError explaining why there isn't one."""
    global _client
    if _client is None:
        try:
            from databricks.sdk import WorkspaceClient
        except ImportError as exc:
            raise DatabricksError("Reading Databricks paths needs the `databricks-sdk` package.") from exc
        try:
            w = WorkspaceClient()
            w.config.authenticate()           # fail now, not halfway through a folder
        except Exception as exc:
            raise DatabricksError(
                "This looks like a Databricks path, but the app can't reach Databricks from here. "
                "Run the app on Databricks, or sign in locally with `databricks auth login`. "
                f"({type(exc).__name__}: {str(exc).splitlines()[0][:200]})") from exc
        _client = w
    return _client


def _who() -> str:
    cid = os.getenv("DATABRICKS_CLIENT_ID")
    return (f"the app's service principal (client ID {cid})" if cid
            else "the signed-in Databricks user")


def _access_error(path: str, exc: Exception) -> DatabricksError:
    name = type(exc).__name__
    if name in {"NotFound", "ResourceDoesNotExist"}:
        return DatabricksError(
            f"Databricks says `{path}` does not exist, or {_who()} can't see it. Check the path "
            "(⋮ → Copy path in Databricks). If it is right, share the folder with "
            f"{_who()}: ⋮ → Share (Permissions) → Can Read.")
    if name in {"PermissionDenied", "Unauthenticated"}:
        return DatabricksError(f"{_who().capitalize()} has no access to `{path}`. Share it: "
                               "⋮ → Share (Permissions) → Can Read.")
    return DatabricksError(f"Couldn't read `{path}` from Databricks: {name}: {str(exc)[:300]}")


# ── mirroring ──────────────────────────────────────────────────────────────

def _wanted(name: str) -> bool:
    from ask.sources.loaders import CODE_EXTS, DATA_EXTS, DOC_EXTS, IMAGE_EXTS, NAMED_TEXT_FILES
    ext = PurePosixPath(name).suffix.lower()
    return ext in CODE_EXTS | DOC_EXTS | DATA_EXTS | IMAGE_EXTS or name.lower() in NAMED_TEXT_FILES


def _skipped_dir(rel: str) -> bool:
    from ask.sources.loaders import SKIP_DIRS
    return any(part in SKIP_DIRS or part.startswith(".") for part in PurePosixPath(rel).parts[:-1])


def _cache_root(remote: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._@-]+", "_", remote.strip("/"))
    return config.DATA_DIR / "databricks" / safe


class _Manifest:
    """Remembers each mirrored file's modified time so unchanged files aren't downloaded again."""

    def __init__(self, root: Path):
        self.file = root.parent / (root.name + ".manifest.json")     # beside, not inside, the mirror
        try:
            self.data = json.loads(self.file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def fresh(self, rel: str, stamp, local: Path) -> bool:
        return stamp is not None and self.data.get(rel) == stamp and local.exists()

    def mark(self, rel: str, stamp) -> None:
        self.data[rel] = stamp

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.file.write_text(json.dumps(self.data), encoding="utf-8")


def fetch(path: str) -> Path:
    """Mirror a Databricks workspace/Volume path locally; return the local file or folder."""
    remote = normalise(path)
    if remote.startswith("/Volumes/"):
        return _fetch_volume(remote)
    if remote.startswith("/Workspace/"):
        return _fetch_workspace(remote)
    raise DatabricksError(f"`{path}` is not a /Workspace or /Volumes path.")


def _fetch_workspace(remote: str) -> Path:
    from databricks.sdk.service.workspace import ExportFormat

    w = client()
    api_path = remote[len("/Workspace"):]
    try:
        info = w.workspace.get_status(api_path)
    except Exception as exc:
        raise _access_error(remote, exc) from exc

    root = _cache_root(remote)
    kind = getattr(info.object_type, "value", info.object_type)
    if kind in ("FILE", "NOTEBOOK"):
        entries, base = [info], PurePosixPath(api_path).parent
    elif kind in ("DIRECTORY", "REPO"):
        try:
            entries = [o for o in w.workspace.list(api_path, recursive=True)
                       if getattr(o.object_type, "value", o.object_type) in ("FILE", "NOTEBOOK")]
        except Exception as exc:
            raise _access_error(remote, exc) from exc
        base = PurePosixPath(api_path)
    else:
        raise DatabricksError(f"`{remote}` is a {kind.lower()}, not a folder, file or notebook.")

    from ask.sources.loaders import IMAGE_EXTS
    manifest, kept, last_rel = _Manifest(root), 0, None
    for obj in entries:
        rel = str(PurePosixPath(obj.path).relative_to(base))
        is_nb = getattr(obj.object_type, "value", obj.object_type) == "NOTEBOOK"
        if is_nb:
            lang = getattr(obj.language, "value", obj.language) or "PYTHON"
            rel += NOTEBOOK_EXT.get(lang, ".txt")
        if not _wanted(rel) or _skipped_dir(rel):
            continue
        if kept >= config.MAX_REPO_FILES:
            break
        if obj.size and obj.size > config.MAX_FILE_BYTES and \
                PurePosixPath(rel).suffix.lower() not in {".pdf", ".docx"} | IMAGE_EXTS:
            continue
        kept += 1
        last_rel = rel
        local = root / rel
        if manifest.fresh(rel, obj.modified_at, local):
            continue
        try:
            if is_nb:
                data = base64.b64decode(w.workspace.export(obj.path, format=ExportFormat.SOURCE).content or "")
            else:
                with w.workspace.download(obj.path, format=ExportFormat.AUTO) as fh:
                    data = fh.read()
        except Exception as exc:
            raise _access_error(obj.path, exc) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        manifest.mark(rel, obj.modified_at)
    manifest.save()
    if kept == 0:
        raise DatabricksError(f"No readable code, document or data files in `{remote}`.")
    return root / last_rel if kind in ("FILE", "NOTEBOOK") else root


def _fetch_volume(remote: str) -> Path:
    w = client()
    root = _cache_root(remote)
    manifest = _Manifest(root)

    def download(file_path: str, rel: str, stamp) -> None:
        local = root / rel
        if manifest.fresh(rel, stamp, local):
            return
        try:
            resp = w.files.download(file_path)
            data = resp.contents.read()
        except Exception as exc:
            raise _access_error(file_path, exc) from exc
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        manifest.mark(rel, stamp)

    try:                                          # a single file?
        meta = w.files.get_metadata(remote)
        name = PurePosixPath(remote).name
        if not _wanted(name):
            raise DatabricksError(f"`{name}` is not a file type the app can read.")
        download(remote, name, str(getattr(meta, "last_modified", None)))
        manifest.save()
        return root / name
    except DatabricksError:
        raise
    except Exception:
        pass                                      # not a file: treat as a directory

    kept, stack = 0, [remote]
    try:
        while stack and kept < config.MAX_REPO_FILES:
            folder = stack.pop()
            for entry in w.files.list_directory_contents(folder):
                rel = str(PurePosixPath(entry.path).relative_to(remote))
                if entry.is_directory:
                    if not _skipped_dir(rel + "/x"):
                        stack.append(entry.path)
                    continue
                if not _wanted(entry.name or rel) or _skipped_dir(rel):
                    continue
                if entry.file_size and entry.file_size > config.MAX_FILE_BYTES * 20:
                    continue
                download(entry.path, rel, entry.last_modified)
                kept += 1
    except DatabricksError:
        raise
    except Exception as exc:
        raise _access_error(remote, exc) from exc
    manifest.save()
    if kept == 0:
        raise DatabricksError(f"No readable code, document or data files in `{remote}`.")
    return root
