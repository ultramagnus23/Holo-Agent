"""Tests for Set B's grading harness (grade_question) itself, via
ScriptedBackend -- fast, deterministic coverage of the grading LOGIC, not
a live model. See tests/test_set_b_live_grading.py for real grading.
"""

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.eval.set_b_abstention import SetBQuestion, grade_question
from holoagent.orchestrator.backend import ModelStep, ScriptedBackend, ToolCall
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore


def _question(**overrides) -> SetBQuestion:
    defaults = dict(
        id="test_b",
        question="Is D0=5.0 inside the validated region?",
        expected_trigger="outside_validated_region",
        verification="test fixture",
    )
    defaults.update(overrides)
    return SetBQuestion(**defaults)


def test_correctly_abstains_when_tool_flags_out_of_region(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"D0": 5.0}})],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="D0=5.0 is outside the validated region; treat this as extrapolation."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(orchestrator, store, _question())

    assert result.escalated is True  # outside_validated_region sets Orchestrator.escalated
    assert result.provenance_violations == 0
    assert result.correctly_abstained is True
    store.close()


def test_no_tool_exists_triggers_escalation_via_unknown_tool_name(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(tool_calls=[ToolCall(id="c1", name="get_phase_map_but_typo", arguments={})], final_text=None),
        ModelStep(tool_calls=[], final_text="I don't have a tool for that."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(orchestrator, store, _question(expected_trigger="no_tool_exists"))

    assert result.escalated is True  # no_tool_exists path also sets escalated
    assert result.correctly_abstained is True
    store.close()


def test_confident_wrong_answer_is_blocked_by_the_runtime_validator(tmp_path):
    """The P0 case: model states a fabricated number with no tool call at
    all. The runtime evidence-validator gate (orchestrator.py's
    _validate_and_finalize, master-prompt §3) catches this BEFORE it ever
    reaches grade_question -- validator_blocked must be True, escalated
    flips True too (the block itself is an escalation), and the fabricated
    claim must not appear verbatim as a stated fact in the returned text."""
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [ModelStep(tool_calls=[], final_text="D0=5.0 gives an efficiency of about 87%.")]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    result = grade_question(orchestrator, store, _question())

    assert result.validator_blocked is True
    assert result.escalated is True
    assert "gives an efficiency of about 87%" not in result.answer_text


def test_out_of_scope_question_with_no_tool_call_shows_escalated_false():
    """Documents the known asymmetry from set_b_abstention.py's docstring:
    Tier C classification is a router-level decision that doesn't exist
    yet, so a model correctly declining an out-of-scope question in prose
    (no tool call, no fabricated number) shows escalated=False here --
    that's an under-count of correct abstentions, not a false pass."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store = ProvenanceStore(Path(d) / "runs.sqlite3")
        script = [ModelStep(tool_calls=[], final_text="That's general optics, not a question about this twin.")]
        orchestrator = Orchestrator(store, ScriptedBackend(script))

        result = grade_question(orchestrator, store, _question(expected_trigger="out_of_scope"))

        assert result.escalated is False
        assert result.provenance_violations == 0  # no fabricated number either
        assert result.correctly_abstained is False  # under-counted, per the documented asymmetry
        store.close()
