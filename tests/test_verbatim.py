"""Guard the code reused unchanged from the original single-file app.

client_create(), the deterministic test engine and the data-quality checks were
copied verbatim (only wording in comments/docstrings was neutralised). They are
pinned by SHA-256 so any edit fails here. If a change is ever intended, update the
hash in the same commit and say why.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PINNED = {
    "client_create": "e41303cfef403664fcd6dc2cb4aa3eb14040b7f43c226089088c64022026d82c",
    # Changed deliberately (the_ultimate_validator): seeded the two unseeded noise draws, fixed
    # the uint8 wrap-around in image noise, evaluated surrogate robustness on a hold-out split,
    # and labelled every surrogate-model result as SURROGATE.
    "ask/analysis/test_engine.py": "0afc67373e15e80d8317ec801293164c58a600bb8b3f35761488f53bf0dc08ac",
    "ask/analysis/data_quality.py": "b8bfd4d411e936b1309595a4b01230d123dff23a49940205f7978ae5b42deabc",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def test_client_create_unchanged():
    from ask import llm
    assert _sha(inspect.getsource(llm.client_create).encode()) == PINNED["client_create"]


def test_client_create_signature_and_behaviour(monkeypatch):
    from ask import llm
    assert str(inspect.signature(llm.client_create)) == "()"
    monkeypatch.setattr(llm, "USE_AZURE", False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    client = llm.client_create()
    assert type(client).__name__ == "OpenAI" and client.api_key == "test-key-not-real"
    monkeypatch.delenv("OPENAI_API_KEY")
    try:
        llm.client_create()
        raise AssertionError("expected EnvironmentError")
    except EnvironmentError as exc:
        assert "OPENAI_API_KEY not found" in str(exc)


def test_test_engine_unchanged():
    path = "ask/analysis/test_engine.py"
    assert _sha((ROOT / path).read_bytes()) == PINNED[path]


def test_data_quality_unchanged():
    path = "ask/analysis/data_quality.py"
    assert _sha((ROOT / path).read_bytes()) == PINNED[path]


def test_client_create_is_the_only_client_factory():
    """Every other module must obtain the client through client_create()."""
    offenders = []
    for py in (ROOT / "ask").rglob("*.py"):
        if py.name == "llm.py":
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                if name in {"OpenAI", "AzureOpenAI"}:
                    offenders.append(f"{py.name}:{node.lineno}")
    assert offenders == []


def test_no_embeddings_or_graph_dependencies():
    banned = {"faiss", "networkx", "rank_bm25", "langgraph", "tiktoken"}
    found = []
    for py in list((ROOT / "ask").rglob("*.py")) + [ROOT / "app.py"]:
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found += [(py.name, a.name) for a in node.names if a.name.split(".")[0] in banned]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in banned:
                found.append((py.name, node.module))
    assert found == []
    reqs = (ROOT / "requirements.txt").read_text().lower()
    assert not any(b.replace("_", "-") in reqs or b in reqs for b in banned | {"faiss-cpu"})
