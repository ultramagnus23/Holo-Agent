"""T9 run_parameter_sweep — Batched T1-T7 over a parameter grid (harness).

Phase 2 tool (TRD §8). Per TRD §5.3: a sweep is one batched solver call,
not N calls in a Python loop. Stub only.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "parameter_grid": {
            "type": "object",
            "description": "Mapping of parameter name -> list of values to sweep",
        },
        "outputs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Which T1-T7 quantities to collect per grid point",
        },
    },
    "required": ["parameter_grid", "outputs"],
}


@register(
    name="run_parameter_sweep",
    json_schema=SCHEMA,
    description="Batched sweep over a parameter grid, collecting the requested output quantities per point.",
    source_module="harness",
)
def run_parameter_sweep(parameter_grid: dict, outputs: list[str]) -> ToolResult:
    raise NotImplementedError("T9 run_parameter_sweep: not yet wired to HoloForge experiment harness")
