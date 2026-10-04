"""The configurable tool name."""
from __future__ import annotations

import json

import pytest

from ask import config, preferences


def test_default_name_is_ask():
    assert preferences.tool_name() == "Ask"


def test_set_and_persist(isolated_data_dir):
    assert preferences.set_tool_name("  Model   Lens ") == "Model Lens"
    assert preferences.tool_name() == "Model Lens"
    saved = json.loads((isolated_data_dir / "preferences.json").read_text(encoding="utf-8"))
    assert saved == {"tool_name": "Model Lens"}


def test_blank_resets_to_default():
    preferences.set_tool_name("Something")
    assert preferences.set_tool_name("   ") == "Ask" and preferences.tool_name() == "Ask"


def test_length_is_capped():
    assert len(preferences.set_tool_name("x" * 200)) == preferences.MAX_NAME_LEN


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"tool_name": ""}', '{"tool_name": null}'])
def test_corrupt_or_empty_file_falls_back(isolated_data_dir, content):
    isolated_data_dir.mkdir(parents=True, exist_ok=True)
    (isolated_data_dir / "preferences.json").write_text(content, encoding="utf-8")
    assert preferences.tool_name() == "Ask"


def test_other_preferences_are_kept(isolated_data_dir):
    isolated_data_dir.mkdir(parents=True, exist_ok=True)
    (isolated_data_dir / "preferences.json").write_text('{"other": 1}', encoding="utf-8")
    preferences.set_tool_name("Lens")
    assert preferences.load() == {"other": 1, "tool_name": "Lens"}


def test_name_used_in_system_prompt(repo_registry, tmp_path, fake_llm):
    from ask.agent import router
    from tests.conftest import say
    preferences.set_tool_name("Model Lens")
    fake_llm.script = [say("General knowledge: hi.")]
    router.answer("hi", repo_registry, [], model="gpt-4.1", verify=True, output_dir=tmp_path,
                  status=lambda m: None)
    instructions = fake_llm.requests[0]["instructions"]
    assert instructions.startswith("You are Model Lens,")
