"""T8 check_validity_region — Inside/outside validated bounds + distance (validity model).

Phase 1 tool (TRD §8) and a differentiating tool (TRD §3.1): what turns
"here is a number" into "here is a number and here is whether you should
trust it." Stub only: wire to holoagent.physics.holoforge_adapter.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "parameters": {
            "type": "object",
            "description": "Full parameter set to check against the twin's validated region",
        },
    },
    "required": ["parameters"],
}


@register(
    name="check_validity_region",
    json_schema=SCHEMA,
    description="Whether a parameter set sits inside the twin's validated region, and how far outside if not.",
    source_module="validity_model",
)
def check_validity_region(parameters: dict) -> ToolResult:
    raise NotImplementedError("T8 check_validity_region: not yet wired to HoloForge validity model")
