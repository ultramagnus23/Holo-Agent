"""Regression tests for a real bug found live via Set B (b01/b14): a
single bad tool call -- a still-stubbed tool's NotImplementedError, or a
malformed call missing a required argument -- used to crash
Orchestrator.answer() entirely instead of feeding the failure back into
the loop as a self-describing error. Fixed in orchestrator.py by
broadening the tool-execution except clause beyond ToolError.
"""

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.orchestrator.backend import ModelStep, ScriptedBackend, ToolCall
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore


def test_calling_a_stubbed_tool_does_not_crash_the_loop(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="get_phase_map", arguments={"run_id": "x", "checkpoint": 1})],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="That tool isn't available yet."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("What's the phase map at checkpoint 1?")

    assert answer.escalated is True
    assert answer.run_ids == []  # the failed call never reached the provenance store
    assert "isn't available" in answer.text
    store.close()


def test_malformed_tool_call_missing_required_argument_does_not_crash_the_loop(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        # get_fit_quality requires K_rad_per_um -- omitted here on purpose.
        ModelStep(
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="get_fit_quality",
                    arguments={"curve_type": "growth", "xs": [1, 2], "ys": [1, 2]},
                )
            ],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="I need the spatial frequency K to run that fit."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("Fit this curve.")

    assert answer.escalated is True
    assert answer.run_ids == []
    store.close()
