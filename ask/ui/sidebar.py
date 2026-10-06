"""Sidebar: a top bar of icons (new chat, notifications, settings) and the past chats."""
from __future__ import annotations

import html

import streamlit as st

from ask import config, preferences
from ask.analysis.runner import test_catalog
from ask.memory import export, store
from ask.ui import state


def render() -> None:
    with st.sidebar:
        # Top bar: three matching icons — new chat, notifications, settings.
        with st.container(key="topbar", horizontal=True, gap="small", vertical_alignment="center"):
            if st.button("", key="new_chat", icon=":material/add:", help="New chat", type="tertiary"):
                state.new_chat()
                st.rerun()
            _notifications()
            if st.button("", key="open_settings", icon=":material/tune:", help="Settings", type="tertiary"):
                settings_dialog()

        chats = store.list_chats()
        active_id = st.session_state.chat_id
        if active_id:
            st.markdown(f"<style>.st-key-chat-{active_id} button {{ background: #e6e3dd !important; "
                        "font-weight: 500; }}</style>", unsafe_allow_html=True)
        pinned = [c for c in chats if c["pinned"]]
        recent = [c for c in chats if not c["pinned"]]
        for label, group in (("Pinned", pinned), ("Recents", recent)):
            if group:
                st.markdown(f"<div class='ks-label'>{label}</div>", unsafe_allow_html=True)
                for c in group:
                    _chat_row(c, active_id)


def _chat_row(c: dict, active_id: str | None) -> None:
    """A past chat: its title (opens it), a pin toggle and a ⋯ menu (rename, delete). The two
    icons show on hover; the pin always shows when pinned."""
    with st.container(key=f"row-{c['id']}", horizontal=True, gap=None, vertical_alignment="center",
                      width="stretch"):
        if st.button(c["title"], key=f"chat-{c['id']}", type="tertiary", width="stretch"):
            if c["id"] != active_id:
                state.open_chat(c["id"])
                st.rerun()
        if st.button("", key=f"pin-{c['id']}", icon=":material/keep:", type="tertiary",
                     help="Unpin" if c["pinned"] else "Pin to the top"):
            store.set_pinned(c["id"], not c["pinned"])
            st.rerun()
        with st.popover("", icon=":material/more_horiz:", type="tertiary", key=f"menu-{c['id']}",
                        help="Save, rename or delete"):
            _chat_menu(c, active_id)
    if c["pinned"]:
        st.markdown(f"<style>.st-key-pin-{c['id']} button {{ opacity: 1 !important; "
                    "color: var(--pinned-icon) !important; } "
                    f".st-key-row-{c['id']} {{ background: var(--pinned); border: 1px solid var(--pinned-line); "
                    "border-radius: 8px; margin-bottom: 3px; }</style>", unsafe_allow_html=True)


def _notifications() -> None:
    """A small bell next to +: notifications such as the cost limit, with a dot when unread."""
    ss = st.session_state
    unread = sum(not n["read"] for n in ss.notifications)
    if unread:
        st.markdown("<style>.st-key-bell button::after { content: ''; position: absolute; top: 3px; "
                    "right: 3px; width: 7px; height: 7px; border-radius: 50%; background: #c0392b; }</style>",
                    unsafe_allow_html=True)
    with st.popover("", icon=":material/notifications:", type="tertiary", key="bell",
                    help=f"{unread} new notification{'s' if unread != 1 else ''}" if unread else "Notifications"):
        with st.container(key="notif_head", horizontal=True, gap=None, vertical_alignment="center"):
            title = "Notifications" if ss.notifications else "No notifications"
            st.markdown(f"<span class='ks-notif-title'>{title}</span>", unsafe_allow_html=True)
            if not ss.notifications:                  # nothing to mark or clear: header only
                return
            if st.button("", key="notif_read", icon=":material/done_all:", type="tertiary", help="Mark all read",
                         disabled=not unread):
                for n in ss.notifications:
                    n["read"] = True
                st.rerun()
            if st.button("", key="notif_clear", icon=":material/delete_sweep:", type="tertiary", help="Clear all"):
                ss.notifications = []
                st.rerun()
        for n in reversed(ss.notifications):
            text = html.escape(n["text"].replace("\\$", "$"))
            st.markdown(f"<div class='ks-notif{'' if not n['read'] else ' read'}'><span>{n['time']}</span>"
                        f"{text}</div>", unsafe_allow_html=True)


def _full_chat_md(folder, cid: str, title: str):
    """The chat as Markdown, built only when the download is clicked. That happens outside the
    session, so the history folder is captured now and read explicitly."""
    def build() -> str:
        chat = store.get_chat(cid, in_folder=folder)
        return export.chat_md(chat["messages"] if chat else [], title, preferences.tool_name())
    return build


def _chat_menu(c: dict, active_id: str | None) -> None:
    """One compact row: title box + ✎ (rename; Enter works too), ⬇ (save full chat), 🗑 (delete)."""
    with st.container(key=f"cmenu-{c['id']}", horizontal=True, gap=None, vertical_alignment="center"):
        with st.form(f"rename-{c['id']}", border=False):     # a form: Enter renames too
            with st.container(horizontal=True, gap=None, vertical_alignment="center"):
                new_title = st.text_input("Rename chat", value=c["title"], max_chars=80,
                                          label_visibility="collapsed", placeholder="Rename chat")
                renamed = st.form_submit_button("", icon=":material/edit:", help="Rename", type="tertiary")
            if renamed and new_title.strip() and new_title.strip() != c["title"]:
                store.rename(c["id"], new_title)
                state.mirror_chat(c["id"])
                st.rerun()
        st.download_button("", _full_chat_md(store.folder(), c["id"], c["title"]),
                           file_name=export.file_name(c["title"]), mime="text/markdown", type="tertiary",
                           icon=":material/download:", key=f"dl-full-{c['id']}", help="Save full chat (.md)",
                           on_click="ignore")
        deleted = st.button("", key=f"del-{c['id']}", icon=":material/delete:", type="tertiary",
                            help="Delete chat")
    if deleted:
        store.delete_chat(c["id"])
        if c["id"] == active_id:
            state.new_chat()
        st.rerun()


def _cost_tab() -> None:
    import pandas as pd
    from ask import usage
    ss = st.session_state
    prices = preferences.prices()
    total, unpriced = state.session_cost()
    rows = ss.usage.rows(prices)
    tokens = sum(r["input"] + r["output"] for r in rows)

    m1, m2, m3 = st.columns(3)
    m1.metric("This session", state.usd(total).replace("\\", ""))
    m2.metric("Tokens", f"{tokens:,}")
    m3.metric("Model calls", f"{sum(r['calls'] for r in rows):,}")
    if ss.cost_limit > 0:
        st.progress(min(total / ss.cost_limit, 1.0),
                    text=f"{state.usd(total)} of your {state.usd(ss.cost_limit)} limit "
                         f"({total / ss.cost_limit:.0%})")
    if rows:
        st.dataframe(pd.DataFrame([{"Model": r["model"], "Calls": r["calls"], "Input tokens": r["input"],
                                    "of which cached": r["cached"], "Output tokens": r["output"],
                                    "Cost (USD)": r["cost"]} for r in rows]),
                     hide_index=True, width="stretch",
                     column_config={"Cost (USD)": st.column_config.NumberColumn(format="$%.4f")})
    else:
        st.caption("No model calls yet in this session.")
    if unpriced:
        st.warning(f"No price set for {', '.join(unpriced)}: add it below to include it in the total.")

    with st.form("cost_limit_form", border=False):
        l1, l2 = st.columns([4, 1], vertical_alignment="bottom")
        limit = l1.number_input("Notify me when this session costs more than (USD, 0 = off)", min_value=0.0,
                                value=float(ss.cost_limit), step=0.1, format="%.3f")
        if l2.form_submit_button("Save", width="stretch"):
            state.set_cost_limit(limit)
            st.rerun()

    with st.expander("Prices (USD per 1M tokens)"):
        table = pd.DataFrame([{"Model": m, "Input": p[0], "Cached input": p[1], "Output": p[2]}
                              for m, p in prices.items()])
        edited = st.data_editor(table, num_rows="dynamic", hide_index=True, width="stretch", key="price_editor")
        if st.button("Save prices", key="save_prices"):
            clean = {str(r["Model"]).strip(): (float(r["Input"] or 0), float(r["Cached input"] or 0),
                                               float(r["Output"] or 0))
                     for _, r in edited.iterrows() if str(r["Model"] or "").strip()}
            preferences.set_prices(clean)
            st.rerun()
        st.caption(usage.PRICES_NOTE + " A model name also matches dated variants that start with it.")
    if st.button("Reset session usage", key="reset_usage", type="tertiary", icon=":material/restart_alt:"):
        ss.usage.reset()
        ss.limit_alerted = False
        st.rerun()


def _bound(widget, label: str, opt: str, **kw):
    wkey = f"w_{opt}"

    def _sync() -> None:
        st.session_state[opt] = st.session_state[wkey]

    return widget(label, value=st.session_state[opt], key=wkey, on_change=_sync, **kw)


@st.dialog("Settings", width="large")
def settings_dialog() -> None:
    ss = st.session_state
    general, validation, cost, sources, tests = st.tabs(["General", "Validation", "Cost", "Loaded sources",
                                                         "Test library"])

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
        _bound(st.toggle, "Guardrail review of every answer", "opt_guard_llm",
               help="One small extra model call per answer checks it for bias, for following an injected "
                    "instruction and for off-topic content, and rewrites it if needed. The rule-based "
                    "guardrails (injection scan, leak check, bias scan) always run.")

        with st.form("history_dir_form", border=False):      # a form: Enter saves too
            h1, h2 = st.columns([4, 1], vertical_alignment="bottom")
            new_dir = h1.text_input("Chat history folder", value=ss.history_dir,
                                    placeholder=str(config.DATA_DIR),
                                    help="Your chats are saved in this folder and loaded from it (a chats.db "
                                         "database plus a readable .md copy of each chat in chats_md). Each "
                                         f"user has their own folder; you are signed in as {ss.user}. Leave "
                                         "blank for the default.")
            saved = h2.form_submit_button("Save", width="stretch")
        if saved and new_dir.strip() != ss.history_dir:
            try:
                state.set_history_dir(new_dir)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        st.caption(f"Answering with **{state.model()}**. Chats are saved in `{store.db_path()}`.")

    with validation:
        _bound(st.text_input, "LLM-judge model", "opt_judge_model",
               help="Model for LLM-judge tests (GenAI groundedness, atomic facts, …). Keep it pinned: "
                    "replies are cached by prompt hash, so a rerun replays the same judgements.")
        _bound(st.text_input, "Databricks SQL warehouse ID", "opt_warehouse",
               help="Used to read Unity Catalog tables when no Spark session is available "
                    "(Databricks App, laptop). SQL Warehouses → your warehouse → Connection details.")
        st.caption(f"Random seed for every test: **{config.VALIDATION_SEED}** (env `ASK_VALIDATION_SEED`). "
                   f"Tables above **{config.MAX_TABLE_ROWS:,}** rows are hash-sampled "
                   "(env `ASK_MAX_TABLE_ROWS`). Every test run is saved in "
                   f"`{config.DATA_DIR / 'validation_runs'}`.")

    with cost:
        _cost_tab()

    with sources:
        reg = ss.registry
        if not reg.sources and not reg.models:
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
        for name, m in list(reg.models.items()):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{name}** · model  \n{m.describe()}")
            if c2.button("Unload", key=f"unload-model-{name}"):
                reg.remove(name)
                if ss.chat_id:
                    store.set_sources(ss.chat_id, reg.records())
                st.rerun()

    with tests:
        st.caption("Ask in plain words, e.g. “run the calibration tests on this PD data”, "
                   "“back-test the VaR”, “check the documentation against the ECB guide”.")
        from ask.validation import core as vcore
        specs = vcore.catalog()
        st.markdown(f"**{len(specs)} deterministic and LLM-judge tests**")
        for key, label in vcore.MODEL_TYPES.items():
            mine = [s for s in specs if key in s.model_types]
            if mine:
                with st.expander(f"{label} · {len(mine)}"):
                    st.markdown("\n".join(f"- **{s.area}** · {s.name} (`{s.id}`)"
                                          + (" · LLM judge" if s.kind == "judge" else "") for s in mine))
        with st.expander("Legacy battery (Excel/HTML report)"):
            for model_type, items in test_catalog().items():
                st.markdown(f"**{model_type}**")
                st.markdown("\n".join(f"- {cat} — {desc}" for cat, desc in items))
