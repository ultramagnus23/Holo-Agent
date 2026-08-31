"""End-to-end test with a REAL self-hosted model (Ollama, qwen2.5:7b-instruct)
driving tool selection -- not ScriptedBackend. This is the piece
test_orchestrator_e2e.py explicitly could not prove: a model actually
choosing which tool to call, not a fixed script.

Skipped automatically if no Ollama server is reachable, so the rest of the
suite doesn't depend on a local model being up.
"""

import pytest

pytest.importorskip("torch")

from holoagent.orchestrator.ollama_backend import ollama_server_running

if not ollama_server_running():
    pytest.skip("no Ollama server reachable at localhost:11434", allow_module_level=True)

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.orchestrator.ollama_backend import OllamaBackend
from holoagent.provenance.store import ProvenanceStore

pytestmark = pytest.mark.slow


def test_real_local_model_selects_and_calls_a_real_tool(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    backend = OllamaBackend()
    orchestrator = Orchestrator(provenance_store=store, model_backend=backend, max_steps=6)

    answer = orchestrator.answer(
        "Check whether D0=5.0 with sigma=0.08 is inside the twin's validated parameter region. "
        "Use the check_validity_region tool."
    )

    assert len(answer.run_ids) >= 1, f"model never called a tool; answer was: {answer.text!r}"
    row = store.get(answer.run_ids[0])
    assert row["tool"] == "check_validity_region", f"model called the wrong tool: {row['tool']}"

    store.close()
