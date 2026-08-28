"""T1 get_phase_map — Phase/amplitude field at a given checkpoint (optimizer).

Phase 2 tool (TRD §8). Stub only: wire to holoagent.physics.holoforge_adapter
once the HoloForge Part 2 optimizer checkpoint interface is frozen.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Optimizer run to read the checkpoint from"},
        "checkpoint": {"type": "integer", "description": "Iteration/checkpoint index"},
    },
    "required": ["run_id", "checkpoint"],
}


@register(
    name="get_phase_map",
    json_schema=SCHEMA,
    description="Phase/amplitude field at a given optimizer checkpoint.",
    source_module="optimizer",
)
def get_phase_map(run_id: str, checkpoint: int) -> ToolResult:
    raise NotImplementedError("T1 get_phase_map: not yet wired to HoloForge optimizer")
