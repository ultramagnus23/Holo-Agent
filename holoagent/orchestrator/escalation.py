"""The four escalation triggers (PRD §6) as distinct, templated responses —
"None is left to the model's discretion." These strings are inserted into
the message history as the tool result for a failed/blocked call, so the
orchestrator model narrates around a fixed template rather than improvising
its own refusal wording.
"""

from __future__ import annotations


class EscalationTrigger:
    NO_TOOL_EXISTS = "no_tool_exists"
    TOOL_ERRORED = "tool_errored"
    OUTSIDE_VALIDATED_REGION = "outside_validated_region"
    OUT_OF_SCOPE = "out_of_scope"


def no_tool_exists(requested_quantity: str, closest_tool: str | None) -> str:
    if closest_tool:
        return (
            f"No tool computes '{requested_quantity}'. Closest available: {closest_tool}. "
            "Ask the user whether that's an acceptable substitute — do not silently substitute it yourself."
        )
    return f"No tool computes '{requested_quantity}', and no close substitute is registered."


def tool_errored(tool_name: str, diagnostic: str) -> str:
    return (
        f"Tool '{tool_name}' reported a failure, not a value: {diagnostic}. "
        "Report this failure and its diagnostic verbatim. Do not substitute an estimate, "
        "retry with silently altered parameters, or reason your way to a number the tool did not produce."
    )


def outside_validated_region(distance_from_bounds: float, violations: dict) -> str:
    return (
        f"EXTRAPOLATION WARNING: parameters lie outside the validated region "
        f"(distance_from_bounds={distance_from_bounds}). Violations: {violations}. "
        "You may still answer, but this warning must lead the answer, not be buried or omitted."
    )


def out_of_scope(topic_hint: str = "") -> str:
    suffix = f" ({topic_hint})" if topic_hint else ""
    return (
        f"This question is general optics, not a question about this twin{suffix}. "
        "A general-purpose model will serve the user better for this part. "
        "State that plainly; do not attempt to answer it using twin machinery it wasn't built for."
    )
