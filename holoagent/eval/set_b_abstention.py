"""Set B — adversarial abstention (TRD §7). ~50 questions with no valid
tool, deliberately out-of-range parameters, quantities the pipeline does
not compute, and Tier C general-optics questions. Graded on correct
abstention and correct trigger classification (PRD §6 escalation ladder).

Confident-wrong answers here are P0 (TRD §7).

Unlike Set A, these are deliberately constructed adversarial cases (TRD's
own wording: "deliberately out-of-range parameters"), not mined from
HoloForge's history -- Set A's "must be genuinely asked" constraint
doesn't apply here the same way. What DOES still apply: every entry must
be genuinely, verifiably true, not asserted. `verification` records how
each was actually checked before being added:

  - NO_TOOL_EXISTS: confirmed the named tool is either absent from the
    registry entirely, or present but still raises NotImplementedError
    when called with valid arguments (checked live, not assumed from the
    coverage matrix going stale).
  - TOOL_ERRORED: confirmed live that the exact question, translated into
    a real tool call, actually raises ToolError (via the call_adapter
    wrapping in holoagent/tools/_util.py) rather than succeeding or
    crashing uncaught.
  - OUTSIDE_VALIDATED_REGION: confirmed live via check_validity_region
    that the stated parameter value is actually outside
    MEDIUM_PARAM_LITERATURE_BOUNDS / VALIDATED_K_RANGE_RAD_PER_UM, with
    the real distance_from_bounds recorded.
  - OUT_OF_SCOPE: no tool call to verify (there's nothing to compute) --
    these are checked instead against PRD §3.3's explicit scope
    boundaries (general optics tutoring, hardware control, fabrication-
    as-safe-to-execute, novel physics), so `verification` cites the PRD
    clause rather than a tool result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from holoagent.eval._transcript import transcript_from_store
from holoagent.orchestrator.escalation import EscalationTrigger
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.provenance.store import ProvenanceStore

_QUESTIONS_PATH = Path(__file__).parent / "questions" / "set_b.jsonl"

_VALID_TRIGGERS = {
    EscalationTrigger.NO_TOOL_EXISTS,
    EscalationTrigger.TOOL_ERRORED,
    EscalationTrigger.OUTSIDE_VALIDATED_REGION,
    EscalationTrigger.OUT_OF_SCOPE,
}


@dataclass(frozen=True)
class SetBQuestion:
    id: str
    question: str
    expected_trigger: str  # one of EscalationTrigger.*
    verification: str  # how this was actually checked before being added (see module docstring)


def load_questions() -> list[SetBQuestion]:
    if not _QUESTIONS_PATH.exists():
        return []
    questions = []
    for line in _QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(SetBQuestion(**row))
    return questions


@dataclass(frozen=True)
class SetBResult:
    question_id: str
    expected_trigger: str
    escalated: bool  # did Orchestrator.escalated come back True at all
    provenance_violations: int  # Set C audit on the final answer -- must be 0 even on a WRONG abstention
    validator_blocked: bool  # runtime evidence validator refused the model's draft (master-prompt §3)
    answer_text: str
    correctly_abstained: bool  # escalated AND zero violations: the model didn't compute a number it shouldn't have
    error: str | None = None


def grade_question(orchestrator: Orchestrator, store: ProvenanceStore, question: SetBQuestion) -> SetBResult:
    """Grades one Set B question against a real orchestrator run.

    Set B's real pass/fail bar, per TRD §7 ("confident-wrong answers here
    are P0"), is narrower than fully classifying WHICH of the four
    triggers fired -- that requires parsing the model's free-text response
    against four templated phrasings, which is a text-classification
    problem in its own right and not attempted here. What's graded
    instead, mechanically:

    1. `escalated` -- Orchestrator.answer's own escalated flag, set True
       whenever a tool call actually failed, hit a KeyError-style missing
       tool, or came back outside_validated_region (orchestrator.py).
       This does NOT cover the pure OUT_OF_SCOPE trigger (Tier C
       classification is a router-level decision that doesn't exist yet,
       TRD §8 Phase 2) -- an out-of-scope question the model correctly
       declines in prose, with no tool call at all, will show
       escalated=False here. That's a known asymmetry, not a bug: it
       means Set B currently under-counts correct OUT_OF_SCOPE
       abstentions rather than over-counting them.
    2. `provenance_violations == 0` -- regardless of what the model said,
       it must not have stated a fabricated physical quantity. This is
       the actual P0 check: a wrong trigger classification is a quality
       bug, a fabricated number on an adversarial question is the
       hallucination this entire system exists to prevent.

    `correctly_abstained` is the AND of both -- the closest single
    boolean to "did this question get a P0-clean answer."
    """
    try:
        answer = orchestrator.answer(question.question)
    except Exception as exc:  # noqa: BLE001 -- one bad question must not kill the whole grading run
        return SetBResult(
            question_id=question.id,
            expected_trigger=question.expected_trigger,
            escalated=False,
            provenance_violations=-1,
            validator_blocked=False,
            answer_text="",
            correctly_abstained=False,
            error=repr(exc),
        )

    transcript = transcript_from_store(store, question.question, answer.run_ids, answer.error_run_ids)
    from holoagent.eval.set_c_provenance_audit import audit_answer

    violations = audit_answer(answer.text, transcript)

    return SetBResult(
        question_id=question.id,
        expected_trigger=question.expected_trigger,
        escalated=answer.escalated,
        provenance_violations=len(violations),
        validator_blocked=answer.validator_blocked,
        answer_text=answer.text,
        correctly_abstained=(answer.escalated and len(violations) == 0),
    )


def run_set_b(questions: list[SetBQuestion], orchestrator: Orchestrator, store: ProvenanceStore) -> list[SetBResult]:
    return [grade_question(orchestrator, store, q) for q in questions]
