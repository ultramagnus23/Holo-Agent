"""Set A — grounded (TRD §7). Target ~150 real questions asked during
HoloForge Part 2. Graded on: correct tool selection, correct chaining,
numeric match against a hand-computed reference.

Questions live in holoagent/eval/questions/set_a.jsonl (not yet populated —
this is a running log to keep during Part 2 work, per PRD §7 secondary risk
"agent looks impressive, is never used": the eval set IS the real usage
record).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_QUESTIONS_PATH = Path(__file__).parent / "questions" / "set_a.jsonl"


@dataclass(frozen=True)
class SetAQuestion:
    question: str
    expected_tools: list[str]  # expected tool call sequence, in order
    reference_values: dict[str, float]  # hand-computed reference, keyed by quantity name


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


def run_set_a(questions: list[SetAQuestion]) -> None:
    raise NotImplementedError("Set A grading not yet implemented — depends on Orchestrator.answer (Phase 1)")
