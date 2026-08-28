"""T6 get_sensitivity — d(output)/d(parameter), free from the differentiable twin (autograd).

Phase 1 tool (TRD §8) and a differentiating tool (TRD §3.1): nearly free
because the twin is already differentiable, and most simulators cannot
offer this at all. Stub only: wire to holoagent.physics.holoforge_adapter.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "output_quantity": {"type": "string", "description": "e.g. 'diffraction_efficiency'"},
        "with_respect_to": {"type": "string", "description": "Parameter name to differentiate against"},
    },
    "required": ["run_id", "output_quantity", "with_respect_to"],
}


@register(
    name="get_sensitivity",
    json_schema=SCHEMA,
    description="d(output_quantity)/d(with_respect_to), computed via autograd through the differentiable twin.",
    source_module="autograd",
)
def get_sensitivity(run_id: str, output_quantity: str, with_respect_to: str) -> ToolResult:
    raise NotImplementedError("T6 get_sensitivity: not yet wired to HoloForge autograd twin")
