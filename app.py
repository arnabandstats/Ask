"""Chat with repositories, documents and data. The display name is set in Settings.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from ask import config, preferences

st.set_page_config(page_title=preferences.tool_name(), page_icon=":material/chat_bubble:",
                   layout="centered", initial_sidebar_state="expanded")

from ask.agent import router  # noqa: E402  (after set_page_config)
from ask.memory import store  # noqa: E402
from ask.ui import render, sidebar, state, styles  # noqa: E402

styles.inject()
state.init()
sidebar.render()

ss = st.session_state


def _retitle(cid: str, prompt: str, turn: router.Turn) -> None:
    """A chat opened by a load command is titled after its sources until the first real question."""
    users = [m for m in ss.messages if m["role"] == "user"]
    load_only = all((a.get("meta") or {}).get("kind") == "load"
                    for a in ss.messages if a["role"] == "assistant")
    if turn.meta.get("kind") == "load" and load_only and ss.registry.sources:
        store.rename(cid, "Chat · " + ", ".join(list(ss.registry.sources)[:3]))
    elif turn.meta.get("kind") != "load" and sum(
            1 for a in ss.messages if a["role"] == "assistant"
            and (a.get("meta") or {}).get("kind") != "load") == 1 and len(users) > 1:
        store.rename(cid, store.title_from(prompt))


if ss.notice:
    st.warning(ss.notice)

welcome = st.empty()
if not ss.messages:
    welcome.markdown(
        "<div class='ks-empty'><h2>What are we looking at today?</h2>"
        "<p>Give me a path to a repo, documents or data, then ask anything.</p></div>",
        unsafe_allow_html=True)

title = state.chat_title(ss.chat_id)
for i, m in enumerate(ss.messages):
    render.message(m, i, ss.messages, title)

prompt = st.chat_input("Ask anything, or paste a path to load…")

if prompt:
    welcome.empty()
    cid = state.ensure_chat(prompt)
    user_msg = {"role": "user", "content": prompt, "artifacts": [], "meta": {}}
    ss.messages.append(user_msg)
    store.add_message(cid, "user", prompt)
    render.message(user_msg, len(ss.messages) - 1)

    history = router.history_for_model(ss.messages[:-1])
    with render.assistant_block("pending"):
        indicator = st.empty()
        indicator.markdown(styles.thinking_html("Thinking"), unsafe_allow_html=True)

        def _status(msg: str) -> None:
            indicator.markdown(styles.thinking_html(msg[:140]), unsafe_allow_html=True)

        if ss.opt_warehouse.strip():              # Settings → Validation; read by ask.sources.tables
            os.environ["ASK_SQL_WAREHOUSE_ID"] = ss.opt_warehouse.strip()
        try:
            turn = router.answer(prompt, ss.registry, history, model=state.model(),
                                 verify=ss.opt_verify, output_dir=config.OUTPUT_DIR / cid,
                                 status=_status,
                                 reasoning_effort=config.DEEP_REASONING_EFFORT if ss.opt_deep else None,
                                 judge_model=ss.opt_judge_model.strip() or None)
        except EnvironmentError as exc:
            turn = router.Turn(content=f"**Setup problem:** {exc}")
        except Exception as exc:  # API errors, network, unexpected tool failures
            turn = router.Turn(content=f"Something went wrong: `{type(exc).__name__}: {exc}`")
        indicator.empty()

    ss.messages.append({"role": "assistant", "content": turn.content,
                        "artifacts": turn.artifacts, "meta": turn.meta})
    store.add_message(cid, "assistant", turn.content, turn.artifacts, turn.meta)
    if turn.sources_changed:
        store.set_sources(cid, ss.registry.records())
    _retitle(cid, prompt, turn)
    state.mirror_chat()
    ss.notice = None
    st.rerun()
