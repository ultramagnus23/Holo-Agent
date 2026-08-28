"""T4 get_medium_state — Saturation, exposure history, NPDD state (photopolymer twin).

Phase 2 tool (TRD §8). Stub only.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "timestep": {"type": "integer", "description": "Exposure timestep, or -1 for final state"},
    },
    "required": ["run_id", "timestep"],
}


@register(
    name="get_medium_state",
    json_schema=SCHEMA,
    description="Saturation, exposure history, and NPDD state of the photopolymer twin.",
    source_module="photopolymer_twin",
)
def get_medium_state(run_id: str, timestep: int) -> ToolResult:
    raise NotImplementedError("T4 get_medium_state: not yet wired to HoloForge photopolymer twin")
