"""Regression tests for a real bug found live via Set B (b14): a model
quoting a real tool's own error message verbatim (e.g. "missing 1
required positional argument") false-positived under Set C, because
failed tool calls left no provenance record at all -- only successes were
recorded, violating TRD S4's "every tool invocation writes a record."
Fixed by ProvenanceStore.record_error + Orchestrator recording failures +
Set C grounding numbers found in real recorded error text.
"""

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.eval._transcript import transcript_from_store
from holoagent.eval.set_c_provenance_audit import audit_answer
from holoagent.orchestrator.backend import ModelStep, ScriptedBackend, ToolCall
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore


def test_record_error_persists_a_failed_call(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    run_id = store.record_error(tool="get_fit_quality", inputs={"curve_type": "growth"}, error_text="boom: 42")

    row = store.get(run_id)
    assert row is not None
    assert row["tool"] == "get_fit_quality"
    import json

    assert json.loads(row["outputs_json"]) == {"error": "boom: 42"}
    store.close()


def test_orchestrator_records_failed_calls_as_error_run_ids(tmp_path):
    """Uses a real tool called with a non-numeric value (a proven ToolError
    case -- see tests/test_tool_error_wrapping.py) rather than a stub tool:
    all 12 tools are implemented now (T1/T4/T5/T9 wired since this test was
    first written), so there's no longer a stub to reach for here."""
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    script = [
        ModelStep(
            tool_calls=[
                ToolCall(id="c1", name="check_validity_region", arguments={"medium_params": {"D0": "very_high"}})
            ],
            final_text=None,
        ),
        ModelStep(tool_calls=[], final_text="That value can't be checked."),
    ]
    orchestrator = Orchestrator(store, ScriptedBackend(script))

    answer = orchestrator.answer("is D0=very_high valid?")

    assert answer.run_ids == []
    assert len(answer.error_run_ids) == 1
    row = store.get(answer.error_run_ids[0])
    assert row is not None and "TypeError" in row["diagnostic"]
    store.close()


def test_quoting_a_real_error_message_verbatim_is_grounded_not_flagged(tmp_path):
    """The actual bug: an answer quoting "missing 1 required positional
    argument" from a REAL recorded error must not be flagged, because the
    call genuinely happened and is in provenance now."""
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    error_run_id = store.record_error(
        tool="get_fit_quality",
        inputs={"curve_type": "growth"},
        error_text="TypeError: get_fit_quality() missing 1 required positional argument: 'K_rad_per_um'",
    )
    transcript = transcript_from_store(store, "fit this curve", run_ids=[], error_run_ids=[error_run_id])

    answer_text = (
        "The tool encountered an error: TypeError: get_fit_quality() missing 1 required "
        "positional argument: 'K_rad_per_um'. Please specify K_rad_per_um."
    )
    violations = audit_answer(answer_text, transcript)
    assert violations == []
    store.close()


def test_fabricated_number_alongside_a_real_error_is_still_caught(tmp_path):
    """Grounding real error text must not become a blanket pass -- a
    genuinely fabricated number in the same answer is still flagged."""
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    error_run_id = store.record_error(
        tool="get_fit_quality", inputs={}, error_text="missing 1 required positional argument"
    )
    transcript = transcript_from_store(store, "fit this curve", run_ids=[], error_run_ids=[error_run_id])

    violations = audit_answer("Even so, the NRMSE would probably be about 0.15.", transcript)
    assert len(violations) == 1
    assert violations[0].token == "0.15"
    store.close()
