"""Guard against committing secrets: scan every file git would pick up."""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Real key formats; "sk-" must be followed by a long unbroken token (so prose like
# "risk-management" never matches).
SECRET_PATTERNS = {
    "OpenAI key": re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}"),
    "Databricks token": re.compile(r"\bdapi[0-9a-f]{32}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # an UPPERCASE setting assigned a literal value, e.g. OPENAI_API_KEY=abc... or TOKEN = "abc..."
    "Assigned secret": re.compile(
        r"""(?m)^[ \t]*[A-Z][A-Z0-9_]*(?:API_KEY|_KEY|TOKEN|SECRET|PASSWORD)[ \t]*[:=][ \t]*["']?(?=[A-Za-z0-9_\-./+=]*\d)(?=[A-Za-z0-9_\-./+=]*[A-Za-z])[A-Za-z0-9_\-./+=]{16,}"""),
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".ini", ".json", ".yml", ".yaml", ".cfg", ".env",
                 ".example", ".sh", ".ps1", ".ipynb", ".html", ".css", ".js", ""}


def _gitignore_patterns() -> list[str]:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _ignored(rel: str, patterns: list[str]) -> bool:
    ignored = False
    for pat in patterns:
        negate = pat.startswith("!")
        p = pat[1:] if negate else pat
        p = p.rstrip("/")
        hit = any(fnmatch.fnmatch(part, p) for part in rel.split("/")) or fnmatch.fnmatch(rel, p)
        if hit:
            ignored = not negate
    return ignored


def _committable_files():
    """Files git would commit: asks git when this is a repository (honours .gitignore and
    .git/info/exclude), otherwise falls back to parsing .gitignore."""
    if (ROOT / ".git").exists():
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
        for rel in filter(None, out.split("\0")):
            p = ROOT / rel
            if p.is_file():
                yield rel, p
        return
    patterns = _gitignore_patterns()
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if ".git/" in rel + "/" or not p.is_file() or _ignored(rel, patterns):
            continue
        yield rel, p


def test_env_files_are_ignored():
    patterns = _gitignore_patterns()
    assert _ignored(".env", patterns)
    assert _ignored(".env.local", patterns)
    assert not _ignored(".env.example", patterns)
    for private in ("ask_data/chats.db", "ask_data/text_cache/x.txt",
                    "validator_cache/a.pkl", ".claude/settings.local.json"):
        assert _ignored(private, patterns), private


def test_no_secrets_in_committable_files():
    findings = []
    for rel, path in _committable_files():
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 5_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, rx in SECRET_PATTERNS.items():
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                findings.append(f"{rel}:{line} looks like a {label}")      # value never printed
    assert findings == [], "\n".join(findings)


OLD_NAME = "key" + "stone"          # assembled so this file doesn't contain it either


def test_old_product_name_appears_nowhere():
    """Neither file names nor contents of anything committed may mention the old name."""
    hits = []
    for rel, path in _committable_files():
        if OLD_NAME in rel.lower():
            hits.append(f"{rel}: in the file name")
            continue
        data = path.read_bytes()
        if OLD_NAME.encode() in data.lower():
            line = data[:data.lower().find(OLD_NAME.encode())].count(b"\n") + 1
            hits.append(f"{rel}:{line}")
    assert hits == [], "\n".join(hits)


def test_patterns_catch_real_shapes_and_skip_prose():
    fake_key = "sk-proj-" + "A1b2C3d4" * 6
    assert SECRET_PATTERNS["OpenAI key"].search(f"key = '{fake_key}'")
    assert SECRET_PATTERNS["Assigned secret"].search("OPENAI_API_KEY=" + "x7" * 12)
    assert not SECRET_PATTERNS["OpenAI key"].search("a risk-management framework for AI systems")
    assert not SECRET_PATTERNS["Assigned secret"].search("OPENAI_API_KEY=")
    assert not SECRET_PATTERNS["Assigned secret"].search("OPENAI_API_KEY=<your key here>")
    assert not SECRET_PATTERNS["Assigned secret"].search("        token = match.group(1)")
    assert not SECRET_PATTERNS["Assigned secret"].search('API_KEY = os.getenv("OPENAI_API_KEY")')
    assert SECRET_PATTERNS["Assigned secret"].search('AZURE_OPENAI_KEY = "' + "q9" * 10 + '"')
    assert not SECRET_PATTERNS["Assigned secret"].search('OPTIMISER_RESULT_KEY = "optimiser_result"')
