"""T3 get_fit_quality — NRMSE per region + parameter set (twin validation).

Phase 1 tool (TRD §8). Stub only: wire to holoagent.physics.holoforge_adapter.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "region": {"type": "string", "description": "Named field region, or 'all'"},
    },
    "required": ["run_id", "region"],
}


@register(
    name="get_fit_quality",
    json_schema=SCHEMA,
    description="NRMSE per region for a given twin fit, plus the parameter set used.",
    source_module="twin_validation",
)
def get_fit_quality(run_id: str, region: str) -> ToolResult:
    raise NotImplementedError("T3 get_fit_quality: not yet wired to HoloForge twin validation")
