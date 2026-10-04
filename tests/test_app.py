"""End-to-end: the real Streamlit app driven with Streamlit's AppTest (fake LLM, temp data dir)."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ask.memory import store
from tests.conftest import calls, say, tool_call

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def at():
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    return app


def _send(app, text):
    app.chat_input[0].set_value(text).run()
    assert not app.exception, app.exception
    return app


def _markdown(app) -> str:
    return "\n".join(m.value for m in app.markdown)


class TestStartup:
    def test_empty_state(self, at):
        assert "What are we looking at today?" in _markdown(at)
        assert len(at.chat_input) == 1

    def test_sidebar_is_minimal(self, at):
        labels = [b.label for b in at.sidebar.button]
        assert labels == ["＋", "⚙  Settings"]          # no chats yet: just New chat + Settings

    def test_browser_tab_uses_tool_name_and_no_logo(self, monkeypatch):
        import streamlit as st
        from ask import preferences
        seen = []
        real = st.set_page_config
        monkeypatch.setattr(st, "set_page_config", lambda **kw: (seen.append(kw), real(**kw)))
        AppTest.from_file(APP, default_timeout=60).run()
        preferences.set_tool_name("Model Lens")
        app = AppTest.from_file(APP, default_timeout=60).run()
        assert not app.exception
        assert [kw["page_title"] for kw in seen] == ["Ask", "Model Lens"]
        assert all(kw["page_icon"] == ":material/chat_bubble:" for kw in seen)

    def test_no_llm_call_at_startup(self, at, fake_llm):
        assert fake_llm.requests == []

    def test_startup_does_not_import_test_engine_libraries(self):
        """Run the app in a fresh interpreter and check the heavy libraries stay unloaded."""
        import subprocess
        import sys
        code = ("import sys; from streamlit.testing.v1 import AppTest; "
                f"a = AppTest.from_file({APP!r}, default_timeout=60); a.run(); "
                "assert not a.exception, a.exception; "
                "print(sorted(m for m in ('shap', 'sklearn', 'cv2', 'statsmodels', 'faiss', 'networkx') "
                "if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip().splitlines()[-1] == "[]"


class TestChatFlow:
    def test_load_command_persists_chat_and_sources(self, at, sample_repo, fake_llm):
        _send(at, f"load {sample_repo}")
        assert fake_llm.requests == []
        assert "model_repo" in at.session_state["registry"].sources
        [chat] = store.list_chats()
        assert chat["title"] == "Chat · model_repo"
        saved = store.get_chat(chat["id"])
        assert saved["sources"][0]["path"] == str(sample_repo)
        assert saved["messages"][1]["content"].startswith("Loaded repo **model_repo**")
        assert any(b.label == "Chat · model_repo" for b in at.sidebar.button)

    def test_question_renders_answer_and_check(self, at, sample_repo, fake_llm):
        _send(at, f"load {sample_repo}")
        fake_llm.script = [calls(tool_call("read_file", file="src/train.py", start_line=7, end_line=8)),
                           say("It compares accuracy to a minimum [src/train.py:L7-8].")]
        _send(at, "how does the gate work?")
        md = _markdown(at)
        assert "`train.py · L7–8`" in md                       # compact citation
        assert "1 citation checked against the source" in md
        assert at.expander[0].label == "Sources"
        [chat] = store.list_chats()
        assert chat["title"] == "how does the gate work?"         # retitled at first real question

    def test_llm_error_is_shown_not_raised(self, at, monkeypatch):
        from ask.agent import llm_call

        def boom(**kw):
            raise RuntimeError("network down")
        monkeypatch.setattr(llm_call, "_responses_create", boom)
        _send(at, "hello")
        assert "Something went wrong: `RuntimeError: network down`" in _markdown(at)

    def test_reopen_chat_restores_messages_and_sources(self, at, sample_repo, fake_llm):
        _send(at, f"load {sample_repo}")
        at.sidebar.button[0].click().run()                       # ＋ new chat
        assert at.session_state["messages"] == [] and not at.session_state["registry"].sources
        chat_btn = next(b for b in at.sidebar.button if b.label == "Chat · model_repo")
        chat_btn.click().run()
        assert not at.exception
        assert len(at.session_state["messages"]) == 2
        assert "model_repo" in at.session_state["registry"].sources

    def test_reopen_with_missing_source_warns(self, at, tmp_path, fake_llm):
        d = tmp_path / "temp_repo"
        d.mkdir()
        (d / "a.py").write_text("x = 1\n")
        _send(at, f"load {d}")
        (d / "a.py").unlink()
        d.rmdir()
        at.sidebar.button[0].click().run()
        next(b for b in at.sidebar.button if b.label.startswith("Chat ·")).click().run()
        assert any("could not be reloaded" in w.value for w in at.warning)

    def test_think_deeper_uses_deep_model_and_effort(self, at, fake_llm):
        at.session_state["opt_deep"] = True
        at.run()
        fake_llm.script = [say("General knowledge: ok.")]
        _send(at, "what is PSI?")
        req = fake_llm.requests[0]
        assert req["model"] == at.session_state["opt_deep_model"] and req["reasoning"]["effort"]
