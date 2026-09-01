"""Tests for Set A's grading harness (grade_question / run_set_a) itself.

Uses ScriptedBackend for fast, deterministic coverage of the grading LOGIC
(tool-selection check, provenance-violation check, error handling) without
needing a live model. Real grading of real Set A questions against a real
model is a separate, slow, opt-in exercise -- see
tests/test_set_a_live_grading.py.
"""

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.eval.set_a_grounded import SetAQuestion, grade_question
from holoagent.orchestrator.backend import ModelStep, ScriptedBackend, ToolCall
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore


def _question(**overrides) -> SetAQuestion:
    defaults = dict(
        id="test_q",
        question="What's the diffraction efficiency at K=8?",
        evidence="test fixture, not a real HoloForge citation",
        expected_tools=["check_validity_region"],
        status="documented_historical",
    )
    defaults.update(overrides)
    return SetAQuestion(**defaults)


def test_grade_question_passes_when_expected_tool_called_and_answer_is_grounded(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"D0": 0.1}})],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="Inside the validated region."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(orchestrator, store, _question())

    assert result.tools_called == ["check_validity_region"]
    assert result.expected_tools_satisfied is True
    assert result.provenance_violations == 0
    assert result.error is None
    store.close()


def test_grade_question_fails_expected_tools_when_wrong_tool_called(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="get_sensitivity", arguments={
                "base_medium_params": {}, "parameter": "sigma", "n_x": 32, "dx": 1.0,
                "target_pattern": "square_wave", "method_id": "MIL", "seed": 0, "n_iters": 5,
            })],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="Sensitivity computed."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(orchestrator, store, _question(expected_tools=["check_validity_region"]))

    assert result.expected_tools_satisfied is False
    store.close()


def test_grade_question_flags_provenance_violation_on_fabricated_number(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"D0": 0.1}})],
            final_text=None,
        ),
        # A number (42.0) that never came from any tool result or the query.
        ModelStep(tool_calls=[], final_text="The efficiency is 42.0 percent."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(orchestrator, store, _question())

    assert result.provenance_violations >= 1
    store.close()


def test_needs_phase_2_tool_question_correctly_fails_expected_tools_when_model_cannot_call_it(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    # Model can't call a tool that doesn't exist -- it just answers directly,
    # which is the correct outcome for a needs_phase_2_tool question, not a bug.
    script = [ModelStep(tool_calls=[], final_text="I don't have a tool for that.")]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(
        orchestrator, store, _question(expected_tools=["get_phase_map"], status="needs_phase_2_tool")
    )

    assert result.expected_tools_satisfied is False
    assert result.tools_called == []
    store.close()


def test_grade_question_handles_backend_errors_without_raising(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    orchestrator = Orchestrator(store, ScriptedBackend([]))  # empty script -> raises on first step

    result = grade_question(orchestrator, store, _question())

    assert result.error is not None
    assert result.provenance_violations == -1
    store.close()
