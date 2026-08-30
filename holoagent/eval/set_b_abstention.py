"""Set B — adversarial abstention (TRD §7). ~50 questions with no valid
tool, deliberately out-of-range parameters, quantities the pipeline does
not compute, and Tier C general-optics questions. Graded on correct
abstention and correct trigger classification (PRD §6 escalation ladder).

Confident-wrong answers here are P0 (TRD §7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from holoagent.orchestrator.escalation import EscalationTrigger

_QUESTIONS_PATH = Path(__file__).parent / "questions" / "set_b.jsonl"


@dataclass(frozen=True)
class SetBQuestion:
    question: str
    expected_trigger: str  # one of EscalationTrigger.*


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


def run_set_b(questions: list[SetBQuestion]) -> None:
    raise NotImplementedError("Set B grading not yet implemented — depends on Orchestrator.answer (Phase 1)")
