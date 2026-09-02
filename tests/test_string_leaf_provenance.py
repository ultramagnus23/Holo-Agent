"""Regression tests for a real bug found live while wiring T12
(retrieve_literature): a model quoting a real number verbatim from a
retrieved passage's TEXT (not a structured JSON leaf) false-positived
under Set C, because _walk_numeric_leaves only extracted numbers from
int/float leaves, never from numbers embedded inside string values. Same
class of gap as the error_text fix in test_error_provenance.py, just for
any string field (tool inputs, output value, diagnostic), not only
failure records. Fixed by generalizing _walk_numeric_leaves itself to
extract numbers from string leaves too.
"""

from holoagent.eval._transcript import SessionTranscript, ToolCallRecord
from holoagent.eval.set_c_provenance_audit import audit_answer


def _literature_transcript(passage_text: str) -> SessionTranscript:
    call = ToolCallRecord(
        run_id="r1",
        tool="retrieve_literature",
        inputs={"query": "D0 Bayfol"},
        outputs={
            "value": {
                "passages": [{"file": "x.md", "line_start": 1, "text": passage_text, "score": 3}],
                "count": 1,
            },
            "distance_from_bounds": None,
        },
    )
    return SessionTranscript(query="what is D0 for Bayfol?", calls=[call])


def test_number_quoted_from_retrieved_passage_text_is_grounded():
    transcript = _literature_transcript("D0 is 0.025 um^2/s from Bruder et al.")
    violations = audit_answer("According to the literature, D0=0.025.", transcript)
    assert violations == []


def test_different_fabricated_number_is_still_caught_despite_grounded_passage():
    transcript = _literature_transcript("D0 is 0.025 um^2/s from Bruder et al.")
    violations = audit_answer("According to the literature, D0=0.999.", transcript)
    assert len(violations) == 1
    assert violations[0].token == "0.999"


def test_diagnostic_text_numbers_are_grounded_too():
    """diagnostic wasn't walked into the pool AT ALL before this fix (not
    just string-leaf extraction) -- covers both parts of the generalization."""
    call = ToolCallRecord(
        run_id="r1",
        tool="get_fit_quality",
        inputs={},
        outputs={
            "value": {"nrmse": 0.874},
            "distance_from_bounds": None,
            "diagnostic": "best fit sits at a HoloForge PARAM_BOUNDS edge, kappa capped at 10.0",
        },
    )
    transcript = SessionTranscript(query="fit quality?", calls=[call])

    violations = audit_answer("The fit hit the bound at kappa=10.0.", transcript)
    assert violations == []
