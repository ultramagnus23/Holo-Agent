"""T10 query_run_store — Retrieve any past run by ID / filter (provenance store).

Phase 2 tool (TRD §8). Unlike T1-T9 this reads the provenance store, not
HoloForge directly. Stub only: wire to holoagent.provenance.store.
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Exact run to fetch, if known"},
        "filter": {
            "type": "object",
            "description": "Structured filter, e.g. {'tool': 'get_fit_quality', 'nrmse_gt': 0.3}",
        },
    },
    "required": [],
}


@register(
    name="query_run_store",
    json_schema=SCHEMA,
    description="Retrieve past run record(s) by ID or structured filter, from the provenance store.",
    source_module="provenance_store",
)
def query_run_store(run_id: str | None = None, filter: dict | None = None) -> ToolResult:
    raise NotImplementedError("T10 query_run_store: not yet wired to holoagent.provenance.store")
