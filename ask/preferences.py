"""User preferences that persist across sessions (ask_data/preferences.json).

Currently: the tool's display name, shown in the browser tab and used by the
assistant to refer to itself.
"""
from __future__ import annotations

import json

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


def set_tool_name(name: str) -> str:
    """Save a new name (blank resets to the default). Returns the name now in effect."""
    clean = " ".join(str(name).split())[:MAX_NAME_LEN]
    prefs = load()
    prefs["tool_name"] = clean or DEFAULT_NAME
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _path().write_text(json.dumps(prefs, indent=2), encoding="utf-8")
    return prefs["tool_name"]
