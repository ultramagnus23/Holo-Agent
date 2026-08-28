"""T11 compare_runs — Structured diff of two run records (provenance store).

Phase 2 tool (TRD §8). Stub only: wire to holoagent.provenance.store.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id_a": {"type": "string"},
        "run_id_b": {"type": "string"},
    },
    "required": ["run_id_a", "run_id_b"],
}


@register(
    name="compare_runs",
    json_schema=SCHEMA,
    description="Structured diff of inputs, outputs, and validity flags between two run records.",
    source_module="provenance_store",
)
def compare_runs(run_id_a: str, run_id_b: str) -> ToolResult:
    raise NotImplementedError("T11 compare_runs: not yet wired to holoagent.provenance.store")
