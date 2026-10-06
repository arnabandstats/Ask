"""Turn a user message into an answer.

1. Pure load commands ("load C:/repo") are handled without an LLM call.
2. Everything else goes to one tool-calling agent that picks its own tools
   (repo/doc search, data queries, tests, comparisons, loading).
3. Answers that rely on repo/doc evidence get a deterministic citation check
   and, if it fails, one repair round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ask import config, preferences
from ask.agent import faithfulness, guardrails, llm_call, prompts
from ask.agent.tools import DATA_TOOLS, SCHEMAS, TEXT_TOOLS, ToolContext, dispatch
from ask.sources import databricks
from ask.sources.loaders import LoadError
from ask.sources.paths import find_paths, is_pure_load_command, kind_hint
from ask.sources.registry import SourceRegistry


@dataclass
class Turn:
    content: str
    artifacts: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    sources_changed: bool = False


def _fast_load(text: str, reg: SourceRegistry) -> Turn | None:
    url = databricks.url_in(text)
    if url:
        path = databricks.path_from_url(url)
        if path is None:                         # a browse/folders/<id> link: no path in it
            return Turn(content=databricks.URL_HELP, meta={"kind": "load"})
        text = text.replace(url, path)
    mentions = find_paths(text)
    if not is_pure_load_command(text, mentions):
        return None
    words = text
    for m in mentions:
        words = words.replace(m.raw, " ")
    hint = kind_hint(words)              # from the user's words, never from the path itself
    lines, changed = [], False
    for m in mentions:
        target = m.path
        if target is None and databricks.looks_like_path(m.raw):
            target = Path(databricks.normalise(m.raw))   # fetched through the Databricks API
        if target is None:
            lines.append(f"Path not found: `{m.raw}`. Check the spelling, or that the drive/folder is accessible.")
            continue
        try:
            for s in reg.load(target, hint):
                label = {"repo": "repo", "docs": "documents", "data": "data"}[s.kind]
                lines.append(f"Loaded {label} **{s.name}**: {s.summary()}.")
                changed = True
                if s.skipped:
                    lines.append(f"  Skipped {len(s.skipped)}: " + "; ".join(s.skipped[:3])
                                 + (" …" if len(s.skipped) > 3 else ""))
        except LoadError as exc:
            lines.append(str(exc))
    return Turn(content="\n".join(lines), sources_changed=changed, meta={"kind": "load"})


def _run_loop(conv: llm_call.Conversation, ctx: ToolContext, max_steps: int) -> str:
    for _ in range(max_steps):
        step = conv.step(SCHEMAS)
        if not step.calls:
            return step.text
        for call in step.calls:
            conv.add_tool_result(call, dispatch(ctx, call.name, call.arguments))
    ctx.status("Writing the answer")
    conv.add_user("Stop using tools now and answer with the evidence you have. "
                  "Say clearly what you could not verify.")
    return conv.step(None).text


def _guard_answer(content: str, question: str, conv: llm_call.Conversation, ctx: ToolContext,
                  system: str, model: str, use_llm: bool) -> str:
    """Leak check, bias scan and (optionally) the LLM review; a flagged answer is rewritten once."""
    if guardrails.leaked(content, system):
        ctx.guard.append("An answer that would have revealed internal instructions was withheld.")
        return guardrails.LEAK_REPLY
    issues = [f"generalisation about a group: “{s}”" for s in guardrails.bias_hits(content)]
    if use_llm and content:
        ctx.status("Checking the answer against the guardrails")
        issues += guardrails.llm_check(question, content, model)
    if not issues:
        return content
    ctx.status("Rewriting the answer to meet the guardrails")
    ctx.guard.append("The answer was rewritten by the guardrails: " + "; ".join(issues))
    conv.add_assistant(content)
    conv.add_user(guardrails.REWRITE.format(issues="\n".join(f"- {i}" for i in issues)))
    revised = conv.step(None).text or content
    if guardrails.leaked(revised, system):
        return guardrails.LEAK_REPLY
    return revised


def answer(text: str, reg: SourceRegistry, history: list[dict], *, model: str,
           verify: bool, output_dir: Path, status: Callable[[str], None],
           reasoning_effort: str | None = None, judge_model: str | None = None,
           guard_llm: bool = False) -> Turn:
    fast = _fast_load(text, reg)
    if fast is not None:
        return fast

    ctx = ToolContext(registry=reg, output_dir=output_dir, status=status, judge_model=judge_model)
    system = prompts.SYSTEM.format(name=preferences.tool_name(), sources=reg.describe()) + guardrails.rules()
    hits = guardrails.scan(text)
    if hits:
        system += guardrails.input_notice(hits)
        ctx.guard.append(f"Your message looks like a prompt-injection attempt ({', '.join(hits)}); "
                         "the assistant kept its role and rules.")
    conv = llm_call.Conversation(system,
                                 history[-config.HISTORY_TURNS * 2:], text, model, reasoning_effort)

    status("Thinking")
    content = _run_loop(conv, ctx, config.MAX_AGENT_STEPS)

    meta: dict = {"tools": sorted(ctx.used), "model": model, "api": llm_call.api_in_use()}
    if ctx.runs:
        meta["test_runs"] = list(dict.fromkeys(ctx.runs))
    # With a repo/docs loaded, any answer that isn't about data or explicitly labelled
    # general knowledge must be grounded: an uncited answer goes back for repair.
    text_loaded = bool(reg.of_kind("repo", "docs"))
    grounded = bool(ctx.used & {"search", "grep", "read_file", "overview"}) or (
        text_loaded and not (ctx.used & DATA_TOOLS) and not ctx.sources_changed)
    check_citations = verify and (grounded or ctx.used & TEXT_TOOLS
                                  or faithfulness.CITATION.search(content))

    def problems(rep: faithfulness.Report | None) -> list[str]:
        return ((rep.problems if rep else []) + faithfulness.check_visual_claims(content, ctx.artifacts)
                + faithfulness.check_test_citations(content, ctx.runs)
                + faithfulness.check_findings(content))

    if check_citations:
        status("Checking citations against the source")
    rep = faithfulness.verify(content, reg, ctx.evidence, grounded) if check_citations else None
    issues = problems(rep)
    if issues:
        status("Fixing unsupported statements")
        conv.add_assistant(content)
        conv.add_user(prompts.REPAIR.format(report="\n".join(f"- {p}" for p in issues)))
        revised = _run_loop(conv, ctx, config.MAX_REPAIR_STEPS)
        if revised:
            content = revised
            rep = faithfulness.verify(content, reg, ctx.evidence, grounded) if check_citations else None
        meta["repaired"] = True
    content = _guard_answer(content, text, conv, ctx, system, model, guard_llm)
    if ctx.guard:
        meta["guard"] = list(dict.fromkeys(ctx.guard))
    if rep is not None:
        meta["verification"] = rep.to_meta()
    if ctx.notes:
        meta["notes"] = "\n\n".join(ctx.notes)[-12_000:]
    return Turn(content=content or "(no answer)", artifacts=ctx.artifacts, meta=meta,
                sources_changed=ctx.sources_changed)


def history_for_model(messages: list[dict]) -> list[dict]:
    """Stored chat messages -> API messages. Analysis results are appended to the
    assistant turn so follow-up questions can refer to earlier numbers."""
    out = []
    for m in messages:
        content = m["content"]
        notes = (m.get("meta") or {}).get("notes")
        if m["role"] == "assistant" and notes:
            content += "\n\n[Tool results behind this answer, for reference]\n" + notes[-4000:]
        out.append({"role": m["role"], "content": content})
    return out
