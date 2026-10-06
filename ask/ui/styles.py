"""The app's small stylesheet: hide Streamlit chrome, calm typography, a tidy sidebar."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap');

:root {
  --ink: #16191f;
  --muted: #6b7079;
  --line: #e7e6e3;
  --soft: #f6f5f2;
  --accent: #d9502b;
  --pinned: #eeeae2;          /* pinned chats: a beige a shade darker than the sidebar */
  --pinned-line: #e2dccf;
  --pinned-icon: #8c8273;
}

html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
code, pre, .stCode { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

/* Streamlit chrome */
/* Streamlit chrome: hide the menu / deploy / status bits, but keep the toolbar itself
   because it holds the "open sidebar" button when the sidebar is collapsed. */
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], #MainMenu, footer, [data-testid="stStatusWidget"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 2.6rem; pointer-events: none; }
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] { pointer-events: auto; }

.block-container { max-width: 800px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* Sidebar: compact, full height — "+" top-left, chats fill, Settings pinned to the bottom */
section[data-testid="stSidebar"] { background: var(--soft); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { display: flex; flex-direction: column; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  position: absolute; top: 0.55rem; right: 0.4rem; height: auto; padding: 0; z-index: 2;
}
section[data-testid="stSidebar"] [data-testid="stLogoSpacer"] { display: none; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
  flex: 1; display: flex; flex-direction: column; padding: 0.55rem 0.55rem 0.6rem;
}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div { flex: 1; display: flex; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
  flex: 1; gap: 1px;
}
section[data-testid="stSidebar"] [data-testid="stElementContainer"] { flex-shrink: 0; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
section[data-testid="stSidebar"] .stButton > button {
  justify-content: flex-start; text-align: left; border: none; border-radius: 7px;
  padding: 0.25rem 0.6rem; font-size: 0.85rem; min-height: 1.85rem; width: 100%;
  color: var(--ink); background: transparent;
}
section[data-testid="stSidebar"] .stButton > button:hover { background: #ebe9e4; color: var(--ink); }
section[data-testid="stSidebar"] .stButton > button > div { justify-content: flex-start; width: 100%; }
section[data-testid="stSidebar"] .stButton > button p {
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.85rem;
}
/* Top bar: new chat, notifications and settings as three matching small icons */
section[data-testid="stSidebar"] .st-key-topbar { margin: 0.1rem 0 0.7rem 0.1rem; flex-wrap: nowrap; gap: 0.15rem; }
section[data-testid="stSidebar"] .st-key-topbar > [data-testid="stElementContainer"],
section[data-testid="stSidebar"] .st-key-topbar > div { width: auto !important; flex: 0 0 auto; }
section[data-testid="stSidebar"] :is(.st-key-new_chat, .st-key-bell, .st-key-open_settings) {
  flex: 0 0 auto; width: auto !important;
}
section[data-testid="stSidebar"] :is(.st-key-new_chat, .st-key-bell, .st-key-open_settings) button,
section[data-testid="stSidebar"] :is(.st-key-new_chat, .st-key-bell, .st-key-open_settings) button:focus:not(:active) {
  position: relative; width: 1.7rem; height: 1.7rem; min-height: 1.7rem; padding: 0; border: none;
  border-radius: 7px; justify-content: center; background: transparent; color: var(--pinned-icon);
}
section[data-testid="stSidebar"] :is(.st-key-new_chat, .st-key-bell, .st-key-open_settings) button:hover {
  background: var(--pinned); color: var(--ink);
}
section[data-testid="stSidebar"] :is(.st-key-new_chat, .st-key-bell, .st-key-open_settings) button > div {
  justify-content: center;
}
section[data-testid="stSidebar"] :is(.st-key-new_chat, .st-key-bell, .st-key-open_settings) [data-testid="stIconMaterial"] {
  font-size: 1.15rem;
}
section[data-testid="stSidebar"] .st-key-bell button div[aria-hidden="true"] { display: none; }   /* no chevron */
.ks-label { display: block; font-size: 0.72rem; line-height: 1.2; color: var(--muted);
            padding: 0.15rem 0 0.6rem 0.6rem; }
/* A chat row: title fills the width, the pin toggle appears on hover (always when pinned) */
section[data-testid="stSidebar"] [class*="st-key-row-"] { gap: 0; flex-wrap: nowrap; }
section[data-testid="stSidebar"] [class*="st-key-chat-"] { flex: 1 1 auto; min-width: 0; }
section[data-testid="stSidebar"] [class*="st-key-pin-"] { flex: 0 0 auto; width: auto !important; }
section[data-testid="stSidebar"] [class*="st-key-pin-"] button {
  width: 1.7rem !important; min-height: 1.7rem; padding: 0; justify-content: center;
  color: var(--muted); opacity: 0; transition: opacity 0.12s ease;
}
section[data-testid="stSidebar"] [class*="st-key-pin-"] button > div { justify-content: center; }
section[data-testid="stSidebar"] [class*="st-key-row-"]:hover [class*="st-key-pin-"] button { opacity: 0.85; }
section[data-testid="stSidebar"] [class*="st-key-menu-"] { flex: 0 0 auto; width: auto !important; }
section[data-testid="stSidebar"] [class*="st-key-menu-"] button {
  width: 1.7rem; min-height: 1.7rem; padding: 0; justify-content: center; border: none;
  border-radius: 7px; background: transparent; color: var(--muted); opacity: 0;
  transition: opacity 0.12s ease;
}
section[data-testid="stSidebar"] [class*="st-key-menu-"] button:hover { background: #e4dfd4; }
section[data-testid="stSidebar"] [class*="st-key-menu-"] button div[aria-hidden="true"] { display: none; }
section[data-testid="stSidebar"] [class*="st-key-row-"]:hover [class*="st-key-menu-"] button,
section[data-testid="stSidebar"] [class*="st-key-menu-"] button[aria-expanded="true"] { opacity: 0.85; }
section[data-testid="stSidebar"] .ks-label:not(:first-child) { padding-top: 0.9rem; }

/* The chat ⋯ menu: one row — title box, ✎ rename, ⬇ save full chat, 🗑 delete (icons only) */
[data-testid="stPopoverBody"]:has([class*="st-key-cmenu-"]) { padding: 0.35rem 0.4rem; min-width: 0; width: 17rem; }
[data-testid="stPopoverBody"]:has([class*="st-key-cmenu-"]) [data-testid="stForm"] { padding: 0; }
[class*="st-key-cmenu-"] { flex-wrap: nowrap; gap: 0.1rem; }
[class*="st-key-cmenu-"] > div:first-child { flex: 1 1 auto; min-width: 0; }        /* the rename form */
[class*="st-key-cmenu-"] [data-testid="stElementContainer"]:has(button) { flex: 0 0 auto; width: auto !important; }
[class*="st-key-cmenu-"] input { font-size: 0.82rem; padding: 0.3rem 0.5rem; min-height: 1.9rem; }
[class*="st-key-cmenu-"] [data-testid="stTextInputRootElement"] { min-height: 1.9rem; height: 1.9rem; }
[class*="st-key-cmenu-"] button {
  min-height: 1.9rem; height: 1.9rem; width: 1.9rem; padding: 0; justify-content: center;
  border-radius: 6px; color: var(--muted);
}
[class*="st-key-cmenu-"] button:hover { background: var(--soft); color: var(--ink); }
[class*="st-key-del-"] button:hover { color: #a23b1c; }

/* Notifications popover: compact list with an icon header */
[data-testid="stPopoverBody"]:has(.st-key-notif_head) { padding: 0.45rem 0.6rem 0.55rem; min-width: 0; width: 19rem; }
[data-testid="stPopoverBody"]:has(.st-key-notif_head):not(:has(.ks-notif)) { width: auto; padding: 0.4rem 0.7rem; }
[data-testid="stPopoverBody"]:has(.st-key-notif_head) [data-testid="stVerticalBlock"] { gap: 0.3rem; }
.st-key-notif_head { flex-wrap: nowrap; }
.st-key-notif_head > div:first-child { flex: 1 1 auto; }
.st-key-notif_head [data-testid="stElementContainer"]:has(button) { flex: 0 0 auto; width: auto !important; }
.st-key-notif_head button { min-height: 1.6rem; height: 1.6rem; width: 1.6rem; padding: 0; justify-content: center;
                            border-radius: 6px; color: var(--muted); }
.st-key-notif_head button:hover { background: var(--soft); color: var(--ink); }
.ks-notif-title { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.ks-notif { font-size: 0.8rem; line-height: 1.4; color: var(--ink); padding: 0.35rem 0.45rem; border-radius: 6px;
            background: var(--pinned); }
.ks-notif.read { background: transparent; color: var(--muted); }
.ks-notif span { color: var(--muted); font-size: 0.72rem; margin-right: 0.4rem; }

/* Save menu under each answer: small and quiet until used */
[class*="st-key-save-"] button { color: var(--muted); font-size: 0.78rem; min-height: 1.6rem;
                                 padding: 0.1rem 0.4rem; }
[class*="st-key-save-"] button:hover { color: var(--ink); }
[class*="st-key-save-"] button div[aria-hidden="true"] { display: none; }   /* icon only, no chevron */

/* Empty state */
.ks-empty { text-align: center; margin-top: 26vh; }
.ks-empty h2 { font-weight: 600; font-size: 1.6rem; letter-spacing: -0.02em; margin: 0 0 0.3rem; color: var(--ink); }
.ks-empty p { color: var(--muted); font-size: 0.95rem; margin: 0; }

/* Messages — user: grey bubble on the right; assistant: plain text on the left, no avatars */
[class*="st-key-umsg-"] { margin: 1.4rem 0 0.7rem; }
[class*="st-key-umsg-"] [data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
.ks-user-row { display: flex; justify-content: flex-end; width: 100%; }
.ks-user {
  background: #efede8; color: var(--ink); border-radius: 18px;
  padding: 0.65rem 1.05rem; max-width: 80%; box-sizing: border-box;
  font-size: 1rem; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere;
}
[class*="st-key-amsg-"] { margin: 0.2rem 0 0.8rem; }
[class*="st-key-amsg-"] [data-testid="stMarkdownContainer"] p,
[class*="st-key-amsg-"] [data-testid="stMarkdownContainer"] li { line-height: 1.65; }
[class*="st-key-amsg-"] p code, [class*="st-key-amsg-"] li code {
  color: #3d4a5c; background: var(--soft); border: 1px solid var(--line);
  border-radius: 5px; padding: 0.05rem 0.35rem; font-size: 0.78em;
}
/* Thinking indicator: three dots bouncing in turn, plus the current step */
.ks-thinking { display: flex; align-items: center; gap: 0.7rem; min-height: 2rem; }
.ks-dots { display: inline-flex; gap: 5px; align-items: flex-end; height: 14px; }
.ks-dots span {
  width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
  animation: ks-bounce 1.2s infinite ease-in-out;
}
.ks-dots span:nth-child(2) { animation-delay: 0.15s; background: #e0734f; }
.ks-dots span:nth-child(3) { animation-delay: 0.3s;  background: #e89676; }
@keyframes ks-bounce {
  0%, 60%, 100% { transform: translateY(0);    opacity: 0.45; }
  30%           { transform: translateY(-6px); opacity: 1; }
}
.ks-step { font-size: 0.85rem; color: var(--muted); animation: ks-fade 0.3s ease-out; }
@keyframes ks-fade { from { opacity: 0; } to { opacity: 1; } }

.ks-check { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }
.ks-check.warn { color: #a15c00; }

/* Page: same warm grey as the sidebar, everywhere (incl. the strip behind the input) */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottom"] > div { background: var(--soft) !important; }

/* Chat input: plain white card with a hairline border and a soft shadow */
[data-testid="stChatInput"] > div {
  background: #ffffff !important; border: 1px solid #e3e0da !important; border-radius: 18px !important;
  box-shadow: 0 1px 2px rgba(22, 25, 31, 0.04), 0 4px 16px rgba(22, 25, 31, 0.06);
}
[data-testid="stChatInput"] > div:focus-within { border-color: #cfcac1 !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: var(--ink); }
[data-testid="stChatInputSubmitButton"] { background: #f1efea !important; border-radius: 10px; }
[data-testid="stBottomBlockContainer"] { padding-bottom: 1.4rem; }
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def thinking_html(step: str = "") -> str:
    label = f"<span class='ks-step'>{html.escape(step)}</span>" if step else ""
    return f"<div class='ks-thinking'><span class='ks-dots'><span></span><span></span><span></span></span>{label}</div>"
