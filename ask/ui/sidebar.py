"""Sidebar: New chat (+), past chats, and a Settings dialog pinned to the bottom."""
from __future__ import annotations

import streamlit as st

from ask import config, preferences
from ask.analysis.runner import test_catalog
from ask.memory import store
from ask.ui import state


def render() -> None:
    with st.sidebar:
        if st.button("＋", key="new_chat", help="New chat", type="tertiary"):
            state.new_chat()
            st.rerun()

        chats = store.list_chats()
        if chats:
            st.markdown("<div class='ks-label'>Recents</div>", unsafe_allow_html=True)
            active_id = st.session_state.chat_id
            if active_id:
                st.markdown(f"<style>.st-key-chat-{active_id} button {{ background: #e6e3dd !important; "
                            "font-weight: 500; }}</style>", unsafe_allow_html=True)
            for c in chats:
                if st.button(c["title"], key=f"chat-{c['id']}", type="tertiary", width="stretch"):
                    if c["id"] != active_id:
                        state.open_chat(c["id"])
                        st.rerun()

        if st.button("⚙  Settings", key="open_settings", type="tertiary", width="stretch"):
            settings_dialog()


def _bound(widget, label: str, opt: str, **kw):
    wkey = f"w_{opt}"

    def _sync() -> None:
        st.session_state[opt] = st.session_state[wkey]

    return widget(label, value=st.session_state[opt], key=wkey, on_change=_sync, **kw)


@st.dialog("Settings", width="large")
def settings_dialog() -> None:
    ss = st.session_state
    general, sources, tests, chat = st.tabs(["General", "Loaded sources", "Built-in tests", "This chat"])

    with general:
        current = preferences.tool_name()
        n1, n2 = st.columns([4, 1], vertical_alignment="bottom")
        new_name = n1.text_input("Tool name", value=current, max_chars=preferences.MAX_NAME_LEN,
                                 help="Shown in the browser tab, and used by the assistant to refer "
                                      "to itself. Leave blank to reset to "
                                      f"“{preferences.DEFAULT_NAME}”.")
        if n2.button("Save", key="save_name", width="stretch") and new_name.strip() != current:
            preferences.set_tool_name(new_name)
            st.rerun()                     # full rerun: applies the new browser-tab title

        # Widgets use their own keys and copy into opt_* on change: Streamlit drops a
        # widget's state when it isn't rendered, and this dialog is usually closed.
        _bound(st.toggle, "Think deeper", "opt_deep",
               help="Use the stronger (slower) model for every answer in this session.")
        c1, c2 = st.columns(2)
        with c1:
            _bound(st.text_input, "Default model", "opt_model")
        with c2:
            _bound(st.text_input, "Think-deeper model", "opt_deep_model")
        _bound(st.toggle, "Verify citations", "opt_verify",
               help="Check every [file:lines] citation in repo/document answers against the "
                    "lines actually read, and ask the model to fix anything unsupported.")
        st.caption(f"Answering with **{state.model()}**. Chats are saved in `{config.DB_PATH}`.")

    with sources:
        reg = ss.registry
        if not reg.sources:
            st.caption("Nothing loaded. In the chat, type something like "
                       "`load C:/projects/model_x` or `read the data from D:/data/sample.csv`.")
        for name, src in list(reg.sources.items()):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{name}** · {src.kind}  \n{src.summary()}  \n`{src.path}`")
            if c2.button("Unload", key=f"unload-{name}"):
                reg.remove(name)
                if ss.chat_id:
                    store.set_sources(ss.chat_id, reg.records())
                st.rerun()

    with tests:
        st.caption("Ask for these in plain words, e.g. “run the classification tests on this data” "
                   "or “check data quality”.")
        st.markdown("**Data quality** — missing values, validity checks, IQR outliers, "
                    "descriptive statistics, distribution plots")
        for model_type, items in test_catalog().items():
            st.markdown(f"**{model_type}**")
            st.markdown("\n".join(f"- {cat} — {desc}" for cat, desc in items))

    with chat:
        if not ss.chat_id:
            st.caption("This chat isn't saved yet; it's saved when you send the first message.")
        else:
            title = next((c["title"] for c in store.list_chats() if c["id"] == ss.chat_id), "")
            new_title = st.text_input("Title", value=title)
            c1, c2 = st.columns(2)
            if c1.button("Rename", width="stretch") and new_title != title:
                store.rename(ss.chat_id, new_title)
                st.rerun()
            if c2.button("Delete chat", type="primary", width="stretch"):
                store.delete_chat(ss.chat_id)
                state.new_chat()
                st.rerun()
