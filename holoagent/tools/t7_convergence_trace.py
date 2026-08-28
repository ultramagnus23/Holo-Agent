"""T7 get_convergence_trace — Full optimizer trajectory, not just the endpoint.

Phase 1 tool (TRD §8). Stub only: wire to holoagent.physics.holoforge_adapter.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
    },
    "required": ["run_id"],
}


@register(
    name="get_convergence_trace",
    json_schema=SCHEMA,
    description="Full optimizer trajectory (loss/parameters per iteration), not just the endpoint.",
    source_module="optimizer",
)
def get_convergence_trace(run_id: str) -> ToolResult:
    raise NotImplementedError("T7 get_convergence_trace: not yet wired to HoloForge optimizer")
