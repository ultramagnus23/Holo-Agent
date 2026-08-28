"""Set C — provenance audit (TRD §7). Automated, runs every build.

Parses every numeric token in every answer from Sets A and B and asserts
each appears in a tool result from that session. Any failure fails the
build. This is the check that enforces R1 / the product's one-line claim
(PRD §2) — Sets A and B measure quality, C measures whether the central
promise holds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMERIC_TOKEN_RE = re.compile(r"-?\d+\.?\d*(?:e[+-]?\d+)?")


@dataclass(frozen=True)
class ProvenanceViolation:
    answer_text: str
    token: str
    context: str


def extract_numeric_tokens(answer_text: str) -> list[str]:
    return _NUMERIC_TOKEN_RE.findall(answer_text)


def audit_answer(answer_text: str, session_tool_result_values: set[str]) -> list[ProvenanceViolation]:
    """Flag every numeric token in answer_text that does not match a value
    present in session_tool_result_values (the set of all numeric values
    returned by tool calls in that session, stringified consistently).

    NOTE: exact string/token matching is a simplification. Formatting
    (rounding for display, unit conversion) needs a defined equivalence
    rule before this is build-blocking — see TRD §7 Set C. Track that
    decision here before wiring this into CI.
    """
    raise NotImplementedError(
        "Set C audit_answer: numeric-token-to-tool-result matching rule "
        "not yet defined (needs: rounding/formatting equivalence policy)"
    )


def run_set_c(answers: list[str], session_tool_results: list[set[str]]) -> list[ProvenanceViolation]:
    violations: list[ProvenanceViolation] = []
    for answer, tool_values in zip(answers, session_tool_results):
        violations.extend(audit_answer(answer, tool_values))
    return violations
