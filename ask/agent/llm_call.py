"""Model access for the agent. The client always comes from client_create().

Primary path: the OpenAI Responses API, which lets reasoning models (gpt-5.x,
o-series) use function tools WITH reasoning on. If the endpoint doesn't offer
Responses (e.g. an older Azure api-version), we fall back to Chat Completions
once and stay there for the session.

The router talks to a Conversation; it never sees which API is underneath.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ask import usage


_client = None
_lock = threading.Lock()
_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 405, 409, 422}
_use_chat_api = False            # flipped if the Responses API is unavailable
_NEEDS_NO_REASONING: set[str] = set()   # chat-API fallback only: models that reject tools + reasoning


def get_client():
    global _client
    with _lock:
        if _client is None:
            from ask.llm import client_create
            _client = client_create()
        return _client


def _retryable(exc: BaseException) -> bool:
    import openai
    if isinstance(exc, openai.RateLimitError):
        return True
    if getattr(exc, "status_code", None) in _NON_RETRYABLE_STATUS:
        return False
    return isinstance(exc, (openai.APIError, openai.APITimeoutError, openai.APIConnectionError))


_retry = retry(reraise=True, retry=retry_if_exception(_retryable),
               wait=wait_exponential(min=2, max=30), stop=stop_after_attempt(5))


@_retry
def _responses_create(**kw):
    return get_client().responses.create(**kw)


@_retry
def _chat_create(**kw):
    return get_client().chat.completions.create(**kw)


def is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5"))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Step:
    text: str
    calls: list[ToolCall] = field(default_factory=list)


class Conversation:
    """One agent turn's message state, for either API."""

    def __init__(self, system: str, history: list[dict], user: str, model: str,
                 reasoning_effort: str | None = None):
        self.model = model
        self.effort = reasoning_effort if is_reasoning_model(model) else None
        self.system = system
        # Responses-API input items, and the equivalent chat messages (kept in step).
        self.items: list = [*history, {"role": "user", "content": user}]
        self.messages: list[dict] = [{"role": "system", "content": system}, *history,
                                     {"role": "user", "content": user}]

    # ── public API ──
    def step(self, tools: list[dict] | None) -> Step:
        global _use_chat_api
        if not _use_chat_api:
            try:
                return self._step_responses(tools)
            except AttributeError:
                _use_chat_api = True             # SDK without .responses
            except Exception as exc:
                if getattr(exc, "status_code", None) in (404, 405):
                    _use_chat_api = True         # endpoint has no Responses API
                else:
                    raise
        return self._step_chat(tools)

    def add_tool_result(self, call: ToolCall, output: str) -> None:
        self.items.append({"type": "function_call_output", "call_id": call.id, "output": output})
        self.messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    def add_assistant(self, text: str) -> None:
        self.items.append({"role": "assistant", "content": text})
        self.messages.append({"role": "assistant", "content": text})

    def add_user(self, text: str) -> None:
        self.items.append({"role": "user", "content": text})
        self.messages.append({"role": "user", "content": text})

    # ── Responses API ──
    def _step_responses(self, tools: list[dict] | None) -> Step:
        kw: dict = {"model": self.model, "instructions": self.system, "input": self.items}
        if tools:
            kw["tools"] = [_to_responses_tool(t) for t in tools]
        if self.effort:
            kw["reasoning"] = {"effort": self.effort}
        if not is_reasoning_model(self.model):
            kw["temperature"] = 0
        resp = _responses_create(**kw)
        usage.record(self.model, getattr(resp, "usage", None))
        calls = []
        for item in resp.output:
            # Reasoning items and function calls must be sent back on the next request.
            self.items.append(item.model_dump(exclude_none=True))
            if item.type == "function_call":
                calls.append(ToolCall(item.call_id, item.name, item.arguments))
        if calls:
            # Mirror into chat messages so a mid-turn fallback could continue.
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls]})
        return Step(text=(resp.output_text or "").strip(), calls=calls)

    # ── Chat Completions fallback ──
    def _step_chat(self, tools: list[dict] | None) -> Step:
        import openai
        kw: dict = {"model": self.model, "messages": self.messages}
        if tools:
            kw["tools"], kw["tool_choice"] = tools, "auto"
        if not is_reasoning_model(self.model):
            kw["temperature"] = 0
        if tools and self.model in _NEEDS_NO_REASONING:
            kw["reasoning_effort"] = "none"
        try:
            resp = _chat_create(**kw)
        except openai.BadRequestError as exc:
            if not (tools and "reasoning_effort" in str(exc) and "reasoning_effort" not in kw):
                raise
            _NEEDS_NO_REASONING.add(self.model)
            resp = _chat_create(**kw, reasoning_effort="none")
        usage.record(self.model, getattr(resp, "usage", None))
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments) for tc in msg.tool_calls or []]
        if calls:
            self.messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in calls]})
            self.items += [{"type": "function_call", "call_id": c.id, "name": c.name,
                            "arguments": c.arguments} for c in calls]
        return Step(text=(msg.content or "").strip(), calls=calls)


def _to_responses_tool(t: dict) -> dict:
    """Chat-style {"type":"function","function":{...}} -> Responses-style flat tool."""
    f = t.get("function", t)
    return {"type": "function", "name": f["name"], "description": f.get("description", ""),
            "parameters": f.get("parameters", {"type": "object", "properties": {}})}


def api_in_use() -> str:
    return "chat.completions (fallback)" if _use_chat_api else "responses"
