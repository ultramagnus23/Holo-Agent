"""Set A — grounded (TRD §7). Target ~150 real questions asked during
HoloForge Part 2. Graded on: correct tool selection, correct chaining,
numeric match against a hand-computed reference.

Per PRD §7's secondary risk ("agent looks impressive, is never used"), a
fabricated question set defeats the entire point of Set A -- it exists to
measure real usage, not to pass a test written to be passed. Every entry
in questions/set_a.jsonl must trace to `evidence`: a specific file/commit
in the HoloForge repo where the question was genuinely asked and answered
during Part 2 development (a code comment describing a diagnostic finding,
a commit message explaining why a fix was made, a test docstring
documenting a regression). This is what "real" means here, structurally,
not just as a stated intention -- see the first batch, mined directly from
HoloForge's docs/commits/comments, not invented.

`status` on each question tracks how far its grounding actually goes:
  - "reproduced": HoloAgent's own tools were run THIS session against the
    real data/config the original finding used, and reference_values holds
    those fresh numbers (with the run_id that produced them). The strongest
    grounding -- not just "this was once true", but "this is still true,
    verified just now."
  - "documented_historical": a genuine finding from HoloForge's own
    development history (commit/doc), not independently re-run here.
    reference_values may be empty or partial if the original finding
    wasn't a clean scalar (many of these are comparative/qualitative).
  - "needs_phase_2_tool": the question is real, but answering it needs a
    tool that's still a stub (commonly T9 run_parameter_sweep, for
    multi-job ablation/sensitivity studies) -- cannot even be attempted
    against the current registry yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from holoagent.eval._transcript import transcript_from_store
from holoagent.eval.set_c_provenance_audit import audit_answer
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore

_QUESTIONS_PATH = Path(__file__).parent / "questions" / "set_a.jsonl"


@dataclass(frozen=True)
class SetAQuestion:
    id: str
    question: str
    evidence: str  # file:line or commit hash + short quote, proving this was really asked
    expected_tools: list[str]  # tool(s) needed to answer, in order
    status: str  # "reproduced" | "documented_historical" | "needs_phase_2_tool"
    documented_finding: str = ""  # the historical answer, as free text (often not a single scalar)
    reference_values: dict = field(default_factory=dict)  # only populated when status == "reproduced"


def load_questions() -> list[SetAQuestion]:
    if not _QUESTIONS_PATH.exists():
        return []
    questions = []
    for line in _QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(SetAQuestion(**row))
    return questions


@dataclass(frozen=True)
class SetAResult:
    question_id: str
    question_status: str  # carried through from the question (reproduced/documented_historical/needs_phase_2_tool)
    tools_called: list[str]  # in call order
    expected_tools_satisfied: bool  # expected_tools subset of tools_called (order not enforced)
    provenance_violations: int  # from Set C's audit_answer against this session's REAL transcript
    escalated: bool  # Orchestrator's own escalated flag (hit a no-tool/error/out-of-region trigger)
    validator_blocked: bool  # runtime evidence validator refused the model's draft (master-prompt §3)
    answer_text: str
    error: str | None = None


def grade_question(orchestrator: Orchestrator, store: ProvenanceStore, question: SetAQuestion) -> SetAResult:
    """Grades one question against a real orchestrator run.

    What "correct" means here is narrower than the TRD §7 description
    ("numeric match against a hand-computed reference") on purpose: the
    question text alone doesn't specify a seed, n_starts, or grid
    resolution, so the model choosing e.g. a different (but equally valid)
    seed than the one reference_values was computed with will legitimately
    produce a different float -- exact equality would fail runs that did
    nothing wrong. Grading instead checks two things that don't depend on
    the model reading the experimenter's mind:

    1. expected_tools_satisfied -- did the model call the tool(s) the
       question actually needs, at all.
    2. provenance_violations == 0 -- does every numeric claim in the
       model's own final answer trace to a REAL tool result from THIS run
       (Set C's audit, against the real provenance chain this call
       produced, not a synthetic fixture). This is the check that matters
       most: it's R1 exercised end to end, on the model's own words, not
       just on hand-authored test strings (contrast with
       tests/test_set_c_provenance_audit.py's synthetic transcripts and
       tests/test_orchestrator_e2e.py's ScriptedBackend transcript).

    For `needs_phase_2_tool` questions, expected_tools_satisfied will
    correctly come back False (the tool doesn't exist to call) -- the
    right outcome there is Orchestrator.escalated=True with a
    no-tool-exists response (PRD §6 trigger 1), not a failed attempt.
    """
    try:
        answer = orchestrator.answer(question.question)
    except Exception as exc:  # noqa: BLE001 -- a grading harness must not let one bad question kill the run
        return SetAResult(
            question_id=question.id,
            question_status=question.status,
            tools_called=[],
            expected_tools_satisfied=False,
            provenance_violations=-1,
            escalated=False,
            validator_blocked=False,
            answer_text="",
            error=repr(exc),
        )

    tools_called = []
    for run_id in answer.run_ids:
        row = store.get(run_id)
        if row is not None:
            tools_called.append(row["tool"])

    expected_satisfied = (
        set(question.expected_tools).issubset(set(tools_called)) if question.expected_tools else True
    )

    transcript = transcript_from_store(store, question.question, answer.run_ids, answer.error_run_ids)
    violations = audit_answer(answer.text, transcript)

    return SetAResult(
        question_id=question.id,
        question_status=question.status,
        tools_called=tools_called,
        expected_tools_satisfied=expected_satisfied,
        provenance_violations=len(violations),
        escalated=answer.escalated,
        validator_blocked=answer.validator_blocked,
        answer_text=answer.text,
    )


def run_set_a(
    questions: list[SetAQuestion], orchestrator: Orchestrator, store: ProvenanceStore
) -> list[SetAResult]:
    return [grade_question(orchestrator, store, q) for q in questions]
