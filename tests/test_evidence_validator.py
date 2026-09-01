"""Tests for the runtime evidence-validator gate (orchestrator.py's
_validate_and_finalize), the master-prompt §3 "Hard Scientific Safety
Rule": every draft final answer is checked against Set C BEFORE it can
reach the caller, not just audited afterward by an eval harness. This is
the structural difference from a system prompt instruction -- a fabricated
number cannot reach the user even if the model ignores the prompt (see
tests/test_set_b_live_grading.py's documented Brewster-angle finding,
which is exactly a case of the prompt being ignored: the runtime gate
tested here is what makes an ungrounded number "mechanically difficult",
not just discouraged).
"""

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.orchestrator.backend import ModelStep, ScriptedBackend, ToolCall
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore


def test_clean_answer_passes_through_unmodified(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"D0": 0.1}})],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="D0=0.1 is inside the validated region."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("Is D0=0.1 inside the validated region?")

    assert answer.text == "D0=0.1 is inside the validated region."
    assert answer.validator_blocked is False
    store.close()


def test_fabricated_number_is_blocked_not_returned(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [ModelStep(tool_calls=[], final_text="The efficiency is about 73.2%, well within spec.")]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("What's the efficiency?")

    assert answer.validator_blocked is True
    assert answer.escalated is True
    # The block explanation legitimately quotes WHICH token was unsupported
    # (so the user knows what to fix) -- that's not the same as stating it
    # as a fact. What must never appear is the original fabricated claim.
    assert "well within spec" not in answer.text
    assert "no legitimate source" in answer.text
    store.close()


def test_block_is_recorded_to_provenance(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [ModelStep(tool_calls=[], final_text="NRMSE came out to 0.199.")]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("What's the NRMSE?")

    assert len(answer.error_run_ids) == 1
    row = store.get(answer.error_run_ids[0])
    assert row is not None
    assert row["tool"] == "_evidence_validator"
    assert "0.199" in row["diagnostic"]
    store.close()


def test_number_stated_in_the_users_own_query_is_not_blocked(tmp_path):
    """A number the USER supplied as a premise is not model recall -- it
    must not be treated as a fabrication just because no tool echoed it
    back (see docs/set_c_matching_rule.md's provenance-pool rationale)."""
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [ModelStep(tool_calls=[], final_text="Understood -- you specified kappa=2.5 for this run.")]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("Use kappa=2.5 for the next computation.")

    assert answer.validator_blocked is False
    assert answer.text == "Understood -- you specified kappa=2.5 for this run."
    store.close()


def test_real_tool_result_value_is_not_blocked(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"D0": 5.0}})],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="D0=5.0 is outside the validated region (bounds 0.001 to 1.0)."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("Is D0=5.0 valid?")

    assert answer.validator_blocked is False
    store.close()
