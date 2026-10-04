"""Model access: Responses API request shape, tool-call round trips, Chat Completions fallback."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ask.agent import llm_call
from ask.agent.llm_call import Conversation
from tests.conftest import calls, say, tool_call

TOOLS = [{"type": "function", "function": {"name": "search", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


class TestHelpers:
    @pytest.mark.parametrize("model,expected", [("gpt-5.6-luna", True), ("gpt-5.5", True), ("o3", True),
                                                ("o4-mini", True), ("gpt-4.1", False), ("gpt-4o", False)])
    def test_is_reasoning_model(self, model, expected):
        assert llm_call.is_reasoning_model(model) is expected

    def test_tool_conversion(self):
        assert llm_call._to_responses_tool(TOOLS[0]) == {
            "type": "function", "name": "search", "description": "d",
            "parameters": {"type": "object", "properties": {}}}


class TestResponsesPath:
    def test_request_shape_non_reasoning(self, fake_llm):
        fake_llm.script = [say("hi")]
        conv = Conversation("SYS", [{"role": "user", "content": "old"},
                                    {"role": "assistant", "content": "older answer"}], "new", "gpt-4.1")
        step = conv.step(TOOLS)
        req = fake_llm.requests[0]
        assert step.text == "hi" and step.calls == []
        assert req["instructions"] == "SYS" and req["temperature"] == 0 and "reasoning" not in req
        assert req["input"][:3] == [{"role": "user", "content": "old"},
                                    {"role": "assistant", "content": "older answer"},
                                    {"role": "user", "content": "new"}]
        assert req["tools"][0]["name"] == "search"

    def test_reasoning_model_gets_effort_not_temperature(self, fake_llm):
        fake_llm.script = [say("ok")]
        Conversation("S", [], "q", "gpt-5.6-luna", reasoning_effort="medium").step(TOOLS)
        req = fake_llm.requests[0]
        assert req["reasoning"] == {"effort": "medium"} and "temperature" not in req

    def test_effort_ignored_for_non_reasoning_model(self, fake_llm):
        fake_llm.script = [say("ok")]
        Conversation("S", [], "q", "gpt-4.1", reasoning_effort="high").step(TOOLS)
        assert "reasoning" not in fake_llm.requests[0]

    def test_tool_round_trip(self, fake_llm):
        fake_llm.script = [calls(tool_call("search", "call_9", query="gate")), say("done")]
        conv = Conversation("S", [], "q", "gpt-4.1")
        step = conv.step(TOOLS)
        assert [(c.id, c.name, json.loads(c.arguments)) for c in step.calls] == [("call_9", "search", {"query": "gate"})]
        conv.add_tool_result(step.calls[0], "RESULT")
        assert conv.step(TOOLS).text == "done"
        second_input = fake_llm.requests[1]["input"]
        assert {"type": "function_call", "call_id": "call_9", "name": "search",
                "arguments": json.dumps({"query": "gate"})} in second_input
        assert second_input[-1] == {"type": "function_call_output", "call_id": "call_9", "output": "RESULT"}

    def test_no_tools_means_no_tools_key(self, fake_llm):
        fake_llm.script = [say("x")]
        Conversation("S", [], "q", "gpt-4.1").step(None)
        assert "tools" not in fake_llm.requests[0]


class _ChatClient:
    """Minimal stand-in for client.chat.completions.create."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _chat_reply(text="", tool_calls=None):
    msg = SimpleNamespace(content=text, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _status_error(cls, status, message):
    import httpx
    resp = httpx.Response(status, request=httpx.Request("POST", "https://x"))
    return cls(message, response=resp, body=None)


class TestFallback:
    def test_404_switches_to_chat_api_for_the_session(self, monkeypatch):
        import openai
        monkeypatch.setattr(llm_call, "_responses_create",
                            lambda **k: (_ for _ in ()).throw(_status_error(openai.NotFoundError, 404, "no")))
        client = _ChatClient([_chat_reply("from chat"), _chat_reply("again")])
        monkeypatch.setattr(llm_call, "get_client", lambda: client)
        conv = Conversation("SYS", [], "q", "gpt-4.1")
        assert conv.step(TOOLS).text == "from chat"
        assert llm_call.api_in_use().startswith("chat.completions")
        assert client.requests[0]["messages"][0] == {"role": "system", "content": "SYS"}
        assert client.requests[0]["tool_choice"] == "auto"
        assert Conversation("S", [], "q2", "gpt-4.1").step(None).text == "again"

    def test_other_errors_propagate(self, monkeypatch):
        import openai
        monkeypatch.setattr(llm_call, "_responses_create",
                            lambda **k: (_ for _ in ()).throw(_status_error(openai.BadRequestError, 400, "bad")))
        with pytest.raises(openai.BadRequestError):
            Conversation("S", [], "q", "gpt-4.1").step(TOOLS)
        assert llm_call.api_in_use() == "responses"

    def test_chat_reasoning_effort_retry(self, monkeypatch):
        import openai
        monkeypatch.setattr(llm_call, "_use_chat_api", True)
        err = _status_error(openai.BadRequestError, 400,
                            "Function tools with reasoning_effort are not supported for gpt-5.6-luna")
        client = _ChatClient([err, _chat_reply("ok"), _chat_reply("ok2")])
        monkeypatch.setattr(llm_call, "get_client", lambda: client)
        assert Conversation("S", [], "q", "gpt-5.6-luna").step(TOOLS).text == "ok"
        assert "reasoning_effort" not in client.requests[0]
        assert client.requests[1]["reasoning_effort"] == "none"
        Conversation("S", [], "q", "gpt-5.6-luna").step(TOOLS)
        assert client.requests[2]["reasoning_effort"] == "none"      # remembered, no failed call

    def test_chat_tool_calls_parsed(self, monkeypatch):
        monkeypatch.setattr(llm_call, "_use_chat_api", True)
        tc = SimpleNamespace(id="t1", function=SimpleNamespace(name="grep", arguments='{"pattern": "x"}'))
        client = _ChatClient([_chat_reply("", [tc])])
        monkeypatch.setattr(llm_call, "get_client", lambda: client)
        conv = Conversation("S", [], "q", "gpt-4.1")
        step = conv.step(TOOLS)
        assert step.calls[0].name == "grep"
        conv.add_tool_result(step.calls[0], "out")
        assert conv.messages[-1] == {"role": "tool", "tool_call_id": "t1", "content": "out"}


class TestRetry:
    def test_retryable_classification(self):
        import openai
        assert llm_call._retryable(_status_error(openai.RateLimitError, 429, "slow down"))
        assert llm_call._retryable(_status_error(openai.InternalServerError, 500, "oops"))
        assert not llm_call._retryable(_status_error(openai.BadRequestError, 400, "bad"))
        assert not llm_call._retryable(_status_error(openai.AuthenticationError, 401, "key"))
        assert not llm_call._retryable(ValueError("x"))
