"""Set C — provenance audit (TRD §7). Automated, runs every build.

Parses every numeric token in an answer and asserts each is grounded in
that session's provenance: a tool result, a tool call's own arguments, or
the user's original query. Any token that doesn't match anything is a
ProvenanceViolation. This is the check that enforces R1 (PRD §5) and the
product's one-line claim (PRD §2) — Sets A and B measure quality, C
measures whether the central promise holds.

See docs/set_c_matching_rule.md for the full design rationale (why this
shape, what's deliberately NOT supported, and why). Do not extend the
matching rules here without updating that document — the rules are
deliberately conservative (biased toward false positives, never false
negatives) and each addition is a real decision, not a convenience fix.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from holoagent.eval._transcript import SessionTranscript

# A standalone number: not adjacent to a word character (so digit runs
# embedded in identifiers like run_ids/input_hashes are never extracted),
# optionally with a trailing '%' consumed as part of the token so the
# percent-conversion rule can see it.
_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?%?(?![\w])"
    r"|(?<![\w.])[-+]?\d+(?:[eE][-+]?\d+)?%?(?![\w])"
)

_IDENTIFIER_PREFIX_RE = re.compile(r"(run_id|input_hash)\s*[:=]\s*$", re.IGNORECASE)

_ROUNDING_RELATIVE_ERROR_CAP = 0.03  # 3%: a "rounded" value further off than this is a different number


@dataclass(frozen=True)
class GroundedValue:
    value: float
    source: str  # human-readable provenance tag for violation reports


@dataclass(frozen=True)
class ProvenanceViolation:
    answer_text: str
    token: str
    start: int
    end: int
    context: str  # a window of surrounding text, for a human to eyeball


@dataclass(frozen=True)
class _MatchResult:
    matched: bool
    rule: str | None = None
    source: str | None = None


def _walk_numeric_leaves(obj, path: str = ""):
    if isinstance(obj, bool):
        return  # bool is a subclass of int in Python; not a numeric quantity
    if isinstance(obj, (int, float)):
        yield path or "value", float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numeric_leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_numeric_leaves(v, f"{path}[{i}]")


def extract_numeric_tokens(text: str) -> list[tuple[str, int, int]]:
    """Returns (token_text, start, end) for every standalone numeric token."""
    return [(m.group(0), *m.span()) for m in _TOKEN_RE.finditer(text)]


def build_provenance_pool(transcript: SessionTranscript) -> list[GroundedValue]:
    pool: list[GroundedValue] = []

    for token_text, _, _ in extract_numeric_tokens(transcript.query):
        core = token_text[:-1] if token_text.endswith("%") else token_text
        try:
            pool.append(GroundedValue(float(core), "user_query"))
        except ValueError:
            continue

    for call in transcript.calls:
        tag = f"run_id={call.run_id}"
        for path, v in _walk_numeric_leaves(call.inputs):
            pool.append(GroundedValue(v, f"{tag} input.{path}"))

        if "error" in call.outputs:
            # A failed call (ProvenanceStore.record_error) -- ground any
            # number the model quotes from the REAL error text (e.g. "missing
            # 1 required positional argument"). This is why error_run_ids
            # exist and get recorded at all: without a real provenance record
            # for the failure, quoting its own error message verbatim would
            # false-positive as an ungrounded number (found live via Set B's
            # b14 -- see docs/set_c_matching_rule.md).
            for token_text, _, _ in extract_numeric_tokens(call.outputs["error"]):
                core = token_text[:-1] if token_text.endswith("%") else token_text
                try:
                    pool.append(GroundedValue(float(core), f"{tag} error_text"))
                except ValueError:
                    continue
            continue

        output_value = call.outputs.get("value")
        for path, v in _walk_numeric_leaves(output_value):
            pool.append(GroundedValue(v, f"{tag} output.value.{path}"))

        dfb = call.outputs.get("distance_from_bounds")
        if isinstance(dfb, (int, float)) and not isinstance(dfb, bool):
            pool.append(GroundedValue(float(dfb), f"{tag} output.distance_from_bounds"))

    return pool


def _decimals_in(token_core_text: str) -> int:
    if "." not in token_core_text:
        return 0
    return len(token_core_text.split(".", 1)[1].split("e")[0].split("E")[0])


def _is_identifier_context(text: str, start: int) -> bool:
    prefix = text[max(0, start - 24) : start]
    return bool(_IDENTIFIER_PREFIX_RE.search(prefix))


def _match_token(token_text: str, pool: list[GroundedValue]) -> _MatchResult:
    is_percent = token_text.endswith("%")
    core = token_text[:-1] if is_percent else token_text
    try:
        token_value = float(core)
    except ValueError:
        return _MatchResult(matched=False)
    decimals = _decimals_in(core)

    if is_percent:
        target = token_value / 100.0
        for gv in pool:
            if math.isclose(gv.value, target, rel_tol=1e-9, abs_tol=1e-12):
                return _MatchResult(matched=True, rule="percent_exact", source=gv.source)
        for gv in pool:
            if round(gv.value * 100, decimals) == token_value:
                return _MatchResult(matched=True, rule="percent_rounded", source=gv.source)
        return _MatchResult(matched=False)

    for gv in pool:
        if math.isclose(gv.value, token_value, rel_tol=1e-9, abs_tol=1e-12):
            return _MatchResult(matched=True, rule="exact", source=gv.source)

    for gv in pool:
        if gv.value == 0:
            continue
        if round(gv.value, decimals) != token_value:
            continue
        rel_err = abs(gv.value - token_value) / abs(gv.value)
        if rel_err <= _ROUNDING_RELATIVE_ERROR_CAP:
            return _MatchResult(matched=True, rule="rounded", source=gv.source)

    return _MatchResult(matched=False)


def audit_answer(answer_text: str, transcript: SessionTranscript) -> list[ProvenanceViolation]:
    pool = build_provenance_pool(transcript)
    violations: list[ProvenanceViolation] = []

    for token_text, start, end in extract_numeric_tokens(answer_text):
        if _is_identifier_context(answer_text, start):
            continue
        result = _match_token(token_text, pool)
        if not result.matched:
            violations.append(
                ProvenanceViolation(
                    answer_text=answer_text,
                    token=token_text,
                    start=start,
                    end=end,
                    context=answer_text[max(0, start - 40) : end + 40],
                )
            )

    return violations


def run_set_c(answers: list[str], transcripts: list[SessionTranscript]) -> list[ProvenanceViolation]:
    violations: list[ProvenanceViolation] = []
    for answer, transcript in zip(answers, transcripts):
        violations.extend(audit_answer(answer, transcript))
    return violations
