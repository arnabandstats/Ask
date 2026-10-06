"""User preferences that persist across sessions (ask_data/preferences.json).

Currently: the tool's display name, shown in the browser tab and used by the
assistant to refer to itself; and each user's chat history folder.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from ask import config

DEFAULT_NAME = "Ask"
MAX_NAME_LEN = 40


def _path():
    return config.DATA_DIR / "preferences.json"


def load() -> dict:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def tool_name() -> str:
    name = str(load().get("tool_name") or "").strip()
    return name[:MAX_NAME_LEN] or DEFAULT_NAME


def _save(prefs: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(prefs, indent=2), encoding="utf-8")


def set_tool_name(name: str) -> str:
    """Save a new name (blank resets to the default). Returns the name now in effect."""
    clean = " ".join(str(name).split())[:MAX_NAME_LEN]
    prefs = load()
    prefs["tool_name"] = clean or DEFAULT_NAME
    _save(prefs)
    return prefs["tool_name"]


# ── chat history folder, per user ──

def history_dir(user: str) -> str:
    """The user's chat history folder ('' = the default, next to the app's data)."""
    dirs = load().get("history_dirs")
    return str(dirs.get(user) or "") if isinstance(dirs, dict) else ""


def check_history_dir(path: str) -> Path:
    """The folder as an absolute path, created if needed; ValueError if it can't hold chats."""
    p = Path(path.strip().strip("\"'")).expanduser()
    if not p.is_absolute():
        raise ValueError("Give a full path, e.g. C:\\Users\\me\\chats or /Volumes/team/me/chats.")
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".write_test_{uuid.uuid4().hex[:6]}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise ValueError(f"Can't write to {p}: {exc.strerror or exc}") from None
    return p


def set_history_dir(user: str, path: str) -> str:
    """Save the user's folder (blank resets to the default). Returns the folder now in effect."""
    clean = str(check_history_dir(path)) if path.strip() else ""
    prefs = load()
    dirs = prefs.get("history_dirs") if isinstance(prefs.get("history_dirs"), dict) else {}
    if clean:
        dirs[user] = clean
    else:
        dirs.pop(user, None)
    prefs["history_dirs"] = dirs
    _save(prefs)
    return clean
