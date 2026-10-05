"""Audit trail of test runs: one JSON file per run_id under <data dir>/validation_runs/.

The file holds the parameters, data fingerprint, versions and every number the
test produced, so a [test:<run_id>] citation in an answer can always be traced
back to (and re-run from) its exact inputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ask import config
from ask.validation.core import TestResult

RUN_CITATION = re.compile(r"\[test:\s*([^\]\s]{1,40})\s*\]")     # any id, so mangled ones are caught too


def runs_dir() -> Path:
    return config.DATA_DIR / "validation_runs"


def save(res: TestResult) -> Path | None:
    if res.run_id in {"", "-"}:
        return None
    p = runs_dir() / f"{res.run_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res.to_json(), indent=1, default=str), encoding="utf-8")
    return p


def exists(run_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{12}", run_id)) and (runs_dir() / f"{run_id}.json").exists()


def load(run_id: str) -> dict:
    if not exists(run_id):                     # also rejects anything that isn't a run id (no paths)
        raise KeyError(f"No saved test run {run_id}.")
    p = runs_dir() / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def check_citations(answer: str) -> list[str]:
    """Every [test:<run_id>] in an answer must be a run that actually happened."""
    bad = sorted({m.group(1) for m in RUN_CITATION.finditer(answer) if not exists(m.group(1))})
    return [f"[test:{r}] does not match any test run. Cite only run_ids returned by "
            "run_validation_test / run_validation_suite." for r in bad]
