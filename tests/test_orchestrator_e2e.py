"""End-to-end orchestrator loop test against LIVE HoloForge (not mocked).

This is the Phase 1 gate from TRD §8: "one real multi-step question
answered end to end with full provenance." ScriptedBackend stands in for
a real model (no API key available in this environment) but every tool
call underneath is real HoloForge computation — the same T6/T7/T8 path
verified by hand in the smoke tests during development. R6 requires
chaining >= 4 dependent tool calls without losing provenance; this test
chains exactly that many, each depending on the previous run_id existing.
"""

import pytest

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.orchestrator.backend import ModelStep, ScriptedBackend, ToolCall
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore

pytest.importorskip("torch")
pytestmark = pytest.mark.slow


def _build_script() -> list[ModelStep]:
    n_x, dx = 64, 51.2 / 64
    common = dict(n_x=n_x, dx=dx, target_pattern="square_wave", method_id="MIL", seed=0)

    step1 = ModelStep(
        tool_calls=[ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"sigma": 0.08}})],
        final_text=None,
    )
    step2 = ModelStep(
        tool_calls=[
            ToolCall(
                id="c2",
                name="get_convergence_trace",
                arguments={"medium_params": {"sigma": 0.08}, "n_iters": 30, "log_every": 10, **common},
            )
        ],
        final_text=None,
    )
    step3 = ModelStep(
        tool_calls=[
            ToolCall(
                id="c3",
                name="get_sensitivity",
                arguments={
                    "base_medium_params": {"sigma": 0.08},
                    "parameter": "sigma",
                    "perturbation_pct": 0.1,
                    "n_iters": 30,
                    **common,
                },
            )
        ],
        final_text=None,
    )
    step4 = ModelStep(
        tool_calls=[
            ToolCall(
                id="c4",
                name="get_fit_quality",
                arguments={
                    "curve_type": "growth",
                    "K_rad_per_um": 8.98,
                    "xs": [0.1, 0.2, 0.3, 0.5, 0.8],
                    "ys": [0.02, 0.05, 0.09, 0.15, 0.22],
                    "n_starts": 2,
                    "seed": 0,
                },
            )
        ],
        final_text=None,
    )
    step5 = ModelStep(tool_calls=[], final_text="Chained T8 -> T7 -> T6 -> T3 successfully; see provenance chain.")
    return [step1, step2, step3, step4, step5]


def test_orchestrator_chains_four_real_tool_calls_with_provenance(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    backend = ScriptedBackend(_build_script())
    orchestrator = Orchestrator(provenance_store=store, model_backend=backend, max_steps=10)

    answer = orchestrator.answer("Multi-step provenance smoke test (R6).")

    assert len(answer.run_ids) == 4, f"expected 4 chained tool calls, got {len(answer.run_ids)}"
    assert "provenance chain" in answer.text

    # Every run_id is retrievable from the store, and the chain's parent
    # links reconstruct the call order (TRD §4 parent_run_id).
    rows = [store.get(rid) for rid in answer.run_ids]
    assert all(row is not None for row in rows)
    assert rows[0]["parent_run_id"] is None
    for parent, child in zip(rows, rows[1:]):
        assert child["parent_run_id"] == parent["run_id"]

    tools_called = [row["tool"] for row in rows]
    assert tools_called == [
        "check_validity_region",
        "get_convergence_trace",
        "get_sensitivity",
        "get_fit_quality",
    ]
    store.close()
