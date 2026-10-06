"""Session state: which chat is open, its messages, its loaded sources, settings."""
from __future__ import annotations

import getpass
import os

import streamlit as st

from ask import config, preferences
from ask.memory import export, store
from ask.sources.registry import SourceRegistry


def current_user() -> str:
    """Who is using this session: the signed-in user's e-mail when a proxy provides it
    (Databricks Apps), else the operating-system user running the app."""
    try:
        h = st.context.headers
        for k in ("X-Forwarded-Email", "X-Forwarded-Preferred-Username", "X-Forwarded-User"):
            if h.get(k):
                return h[k].strip().lower()
    except Exception:
        pass
    try:
        return getpass.getuser().lower()
    except Exception:
        return "local"


def init() -> None:
    ss = st.session_state
    ss.setdefault("user", current_user())
    ss.setdefault("history_dir", preferences.history_dir(ss.user))
    # The store asks for the folder on every call, so dialog/fragment reruns (which skip this
    # function) still use this session's chat database.
    store.set_resolver(lambda: st.session_state.get("history_dir"))
    ss.setdefault("chat_id", None)            # None = new chat, not saved until the first message
    ss.setdefault("messages", [])
    ss.setdefault("registry", SourceRegistry())
    ss.setdefault("notice", None)
    ss.setdefault("opt_deep", False)
    ss.setdefault("opt_model", config.DEFAULT_MODEL)
    ss.setdefault("opt_deep_model", config.DEEP_MODEL)
    ss.setdefault("opt_verify", True)
    ss.setdefault("opt_judge_model", config.JUDGE_MODEL or config.DEFAULT_MODEL)
    ss.setdefault("opt_warehouse", os.getenv("ASK_SQL_WAREHOUSE_ID", ""))


def new_chat() -> None:
    ss = st.session_state
    ss.chat_id, ss.messages, ss.registry, ss.notice = None, [], SourceRegistry(), None


def open_chat(cid: str) -> None:
    chat = store.get_chat(cid)
    if chat is None:
        new_chat()
        return
    ss = st.session_state
    ss.chat_id, ss.messages = cid, chat["messages"]
    reg = SourceRegistry()
    errors = []
    if chat["sources"]:
        with st.spinner("Reloading this chat's sources…"):
            errors = reg.restore(chat["sources"])
    ss.registry = reg
    ss.notice = ("Some sources could not be reloaded:\n- " + "\n- ".join(errors)) if errors else None


def ensure_chat(first_message: str) -> str:
    ss = st.session_state
    if ss.chat_id is None:
        ss.chat_id = store.create_chat(store.title_from(first_message))
    return ss.chat_id


def set_history_dir(path: str) -> str:
    """Switch this user's chat history folder (raises ValueError if unusable) and start fresh
    in it; its existing chats appear in the sidebar."""
    ss = st.session_state
    ss.history_dir = preferences.set_history_dir(ss.user, path)
    new_chat()
    return ss.history_dir


def chat_title(cid: str | None) -> str:
    chat = store.get_chat(cid) if cid else None
    return chat["title"] if chat else "New chat"


def mirror_chat(cid: str | None = None) -> None:
    """Refresh a chat's Markdown copy in the history folder (default: the open chat; best effort)."""
    ss = st.session_state
    cid = cid or ss.chat_id
    chat = store.get_chat(cid) if cid else None
    if chat is None:
        return
    messages = ss.messages if cid == ss.chat_id else chat["messages"]
    try:
        export.mirror(store.folder(), cid, chat["title"], messages, preferences.tool_name())
    except OSError:
        pass


def model() -> str:
    ss = st.session_state
    return (ss.opt_deep_model if ss.opt_deep else ss.opt_model).strip() or config.DEFAULT_MODEL
