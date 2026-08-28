"""T2 get_diffraction_efficiency — Efficiency across field (Kogelnik / BPM).

Phase 2 tool (TRD §8). Stub only.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string"},
        "model": {"type": "string", "enum": ["kogelnik", "bpm"]},
    },
    "required": ["run_id", "model"],
}


@register(
    name="get_diffraction_efficiency",
    json_schema=SCHEMA,
    description="Diffraction efficiency across the field, via Kogelnik or BPM.",
    source_module="kogelnik_bpm",
)
def get_diffraction_efficiency(run_id: str, model: str) -> ToolResult:
    raise NotImplementedError("T2 get_diffraction_efficiency: not yet wired to HoloForge diffraction solver")
