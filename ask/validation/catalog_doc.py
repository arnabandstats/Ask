"""Write docs/TEST_CATALOG.md from the test registry:  python -m ask.validation.catalog_doc"""
from __future__ import annotations

from pathlib import Path

from ask.validation import core

DOC = Path(__file__).resolve().parents[2] / "docs" / "TEST_CATALOG.md"


def markdown() -> str:
    specs = core.catalog()
    n_judge = sum(s.kind == "judge" for s in specs)
    out = ["# Validation test catalog", "",
           f"{len(specs)} tests ({len(specs) - n_judge} deterministic, {n_judge} LLM-judge), generated "
           "from the registry in `ask/validation`. Regenerate with `python -m ask.validation.catalog_doc`.",
           "", "Required inputs are in **bold**, the rest are optional. Run a test with "
           "`run_validation_test` (or as part of `run_validation_suite`); each run is saved with a "
           "citable run_id. `describe_test` gives the full description, H0 and references.", ""]
    groups: dict[str, list[core.TestSpec]] = {}
    for s in specs:
        groups.setdefault(s.id.split(".")[0], []).append(s)
    for prefix in sorted(groups):
        out += [f"## `{prefix}` ({len(groups[prefix])})", "",
                "| Test | Area | Model types | Inputs |", "|---|---|---|---|"]
        for s in sorted(groups[prefix], key=lambda s: s.id):
            ins = ", ".join(f"**{p.name}**" if p.required else p.name for p in s.params)
            judge = " *(LLM judge)*" if s.kind == "judge" else ""
            out.append(f"| `{s.id}` {s.name}{judge} | {s.area} | {', '.join(s.model_types)} | {ins} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown(), encoding="utf-8")
    print(f"Wrote {DOC}")
