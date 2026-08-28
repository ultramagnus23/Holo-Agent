"""The tool contract every HoloForge-wrapping tool must satisfy.

Non-negotiable properties (TRD §3.2):
- JSON schema in, structured result out. No free-text returns to the orchestrator.
- No silent defaults. Missing required parameter -> ToolError naming the field.
- Deterministic. Same inputs, same outputs, recorded seed.
- Validity-bounded. Every numeric return carries in_validated_region + distance.
- Self-describing failure. Non-convergence returns a diagnostic, never a
  best-effort estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    value: Any
    units: str
    in_validated_region: bool
    distance_from_bounds: float | None
    run_id: str  # provenance key
    input_hash: str  # reproducibility
    diagnostic: str | None = None  # populated iff the computation degraded


class ToolError(Exception):
    """Raised for a structured, self-describing tool failure.

    Never caught internally to synthesize a fallback value — a ToolError
    reaching the orchestrator IS the correct behavior for trigger 2
    (errored / non-convergent) in the escalation ladder (PRD §6).
    """

    def __init__(self, message: str, *, missing_field: str | None = None):
        super().__init__(message)
        self.missing_field = missing_field


class MissingParameterError(ToolError):
    """No silent defaults (K3 config-reader principle). A missing required
    parameter is a question back to the user, never a guess."""

    def __init__(self, field: str):
        super().__init__(f"missing required parameter: {field}", missing_field=field)
