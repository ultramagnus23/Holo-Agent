"""Integrity checks on the Set A question log (holoagent/eval/questions/set_a.jsonl).

Not a grading test (Set A grading itself needs a real ModelBackend run --
see set_a_grounded.run_set_a's NotImplementedError). This just enforces
the structural honesty rules the questions are supposed to follow: every
question cites real evidence, every tool it names actually exists in the
registry, and "reproduced" questions actually carry the numbers they claim
to.
"""

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.eval.set_a_grounded import load_questions
from holoagent.tools.registry import list_tools

_VALID_STATUSES = {"reproduced", "documented_historical", "needs_phase_2_tool"}


def test_set_a_has_questions():
    questions = load_questions()
    assert len(questions) >= 20


def test_every_question_has_ids_and_evidence():
    for q in load_questions():
        assert q.id, "every question needs a stable id"
        assert q.evidence.strip(), f"{q.id}: evidence must not be empty -- an unevidenced entry is not a real question"
        assert q.question.strip()


def test_every_question_status_is_valid():
    for q in load_questions():
        assert q.status in _VALID_STATUSES, f"{q.id}: unknown status {q.status!r}"


def test_every_expected_tool_exists_in_the_registry():
    registered = {spec.name for spec in list_tools()}
    for q in load_questions():
        for tool_name in q.expected_tools:
            assert tool_name in registered, f"{q.id}: expected_tools references unregistered tool {tool_name!r}"


def test_reproduced_questions_carry_real_reference_values():
    for q in load_questions():
        if q.status == "reproduced":
            assert q.reference_values, f"{q.id}: status=reproduced but reference_values is empty"


def test_question_ids_are_unique():
    ids = [q.id for q in load_questions()]
    assert len(ids) == len(set(ids))
