"""The LLM judge used by judge-based tests (GenAI groundedness, atomic facts, ...).

Judge tests are NOT statistical tests: an LLM reads and grades text. To make
them as reproducible as possible:
  - temperature 0 (non-reasoning models) and a pinned model name,
  - every (model, system, user) prompt is hashed and its reply cached on disk,
    so re-running the same test on the same data replays the same replies
    instead of asking the model again,
  - the test result says it is judge-based and records the model.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

from ask import config

_lock = threading.Lock()


def _cache_file() -> Path:
    return config.DATA_DIR / "judge_cache.jsonl"


def _key(model: str, system: str, user: str) -> str:
    return hashlib.sha256(json.dumps([model, system, user]).encode()).hexdigest()


def _load() -> dict[str, str]:
    out = {}
    try:
        with open(_cache_file(), encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    out[rec["k"]] = rec["v"]
                except (ValueError, KeyError):
                    continue
    except OSError:
        pass
    return out


def make_judge(model: str | None = None) -> Callable[[str, str], str]:
    """A (system, user) -> reply function bound to one model, with a disk cache."""
    from ask.agent import llm_call
    model = model or config.JUDGE_MODEL or config.DEFAULT_MODEL
    cache = _load()

    def judge(system: str, user: str) -> str:
        k = _key(model, system, user)
        if k in cache:
            return cache[k]
        conv = llm_call.Conversation(system, [], user, model,
                                     "low" if llm_call.is_reasoning_model(model) else None)
        reply = conv.step(None).text
        with _lock:
            cache[k] = reply
            _cache_file().parent.mkdir(parents=True, exist_ok=True)
            with open(_cache_file(), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"k": k, "model": model, "v": reply}) + "\n")
        return reply

    judge.model = model           # type: ignore[attr-defined]
    return judge
