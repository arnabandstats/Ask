"""Session state: which chat is open, its messages, its loaded sources, settings."""
from __future__ import annotations

import streamlit as st

from ask import config
from ask.memory import store
from ask.sources.registry import SourceRegistry


def init() -> None:
    ss = st.session_state
    ss.setdefault("chat_id", None)            # None = new chat, not saved until the first message
    ss.setdefault("messages", [])
    ss.setdefault("registry", SourceRegistry())
    ss.setdefault("notice", None)
    ss.setdefault("opt_deep", False)
    ss.setdefault("opt_model", config.DEFAULT_MODEL)
    ss.setdefault("opt_deep_model", config.DEEP_MODEL)
    ss.setdefault("opt_verify", True)


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


def model() -> str:
    ss = st.session_state
    return (ss.opt_deep_model if ss.opt_deep else ss.opt_model).strip() or config.DEFAULT_MODEL
