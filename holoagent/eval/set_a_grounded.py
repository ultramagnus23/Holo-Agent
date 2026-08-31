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


def run_set_a(questions: list[SetAQuestion]) -> None:
    raise NotImplementedError(
        "Set A grading not yet implemented -- needs a real ModelBackend driving "
        "Orchestrator.answer() over these questions and comparing tool selection + "
        "numeric output against reference_values. Only 'reproduced' questions have "
        "reference_values to grade against today; 'documented_historical' entries "
        "need their reference computed the same way question 1 was (see git history "
        "for the pattern) before they're gradable; 'needs_phase_2_tool' entries are "
        "not gradable until their tool exists."
    )
