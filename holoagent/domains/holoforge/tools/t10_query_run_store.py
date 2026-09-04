"""T10 query_run_store — Retrieve any past run by ID / filter (provenance store).

Phase 2 tool (TRD §8). Unlike T1-T9 this reads holoagent's own
ProvenanceStore, not HoloForge -- wired via
holoagent.provenance.context.get_current_store(), the session-scoped
store the active Orchestrator.answer() call set (see that module's
docstring for why this can't be a global default the way
holoagent.domains.holoforge.physics.get_default_adapter() is for HoloForge).

Filtering is intentionally simple, not a general query language: equality
on `tool` and on `failed` (a real vs. errored call, see
ProvenanceStore.record_error), an inclusive `since`/`until` ISO-timestamp
range, and a threshold check (`value_field` + one of `value_gt`/
`value_lt`/`value_eq`) against a single top-level key of the result's
`value` dict (e.g. "nrmse", "final_psnr") -- covers the master-prompt's
own examples ("find all runs where NRMSE > X", "find failed runs", "find
runs between dates") without building a general JSON-path query engine
nobody asked for (TRD §5 streamlining rule / master-prompt §38 "don't
overbuild").
"""

from __future__ import annotations

import json

from holoagent.provenance.context import get_current_store
from holoagent.tools._util import compute_input_hash, new_run_id
from holoagent.tools.contract import ToolError, ToolResult
from holoagent.tools.registry import register

_MAX_RESULTS = 50  # a bounded dump, not a full-table export

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id": {"type": "string", "description": "Exact run to fetch, if known. Mutually exclusive with filter."},
        "filter": {
            "type": "object",
            "description": (
                "Structured filter. Supported keys: tool (str, exact match), failed (bool), "
                "since/until (ISO timestamp strings, inclusive), value_field (str, a top-level "
                "key of the result's value dict, e.g. 'nrmse'), and exactly one of value_gt / "
                "value_lt / value_eq (float) to threshold it."
            ),
        },
    },
    "required": [],
}


def _row_matches(row: dict, filt: dict) -> bool:
    if "tool" in filt and row["tool"] != filt["tool"]:
        return False

    outputs = json.loads(row["outputs_json"])
    is_failed = "error" in outputs
    if "failed" in filt and bool(filt["failed"]) != is_failed:
        return False

    if "since" in filt and row["timestamp"] < filt["since"]:
        return False
    if "until" in filt and row["timestamp"] > filt["until"]:
        return False

    if "value_field" in filt:
        if is_failed:
            return False  # a failed call has no `value` to threshold
        value = outputs.get("value")
        if not isinstance(value, dict) or filt["value_field"] not in value:
            return False
        field_value = value[filt["value_field"]]
        if not isinstance(field_value, (int, float)) or isinstance(field_value, bool):
            return False
        if "value_gt" in filt and not (field_value > filt["value_gt"]):
            return False
        if "value_lt" in filt and not (field_value < filt["value_lt"]):
            return False
        if "value_eq" in filt and field_value != filt["value_eq"]:
            return False

    return True


def _summarize(row: dict) -> dict:
    outputs = json.loads(row["outputs_json"])
    summary = {
        "run_id": row["run_id"],
        "timestamp": row["timestamp"],
        "tool": row["tool"],
        "parent_run_id": row["parent_run_id"],
    }
    if "error" in outputs:
        summary["failed"] = True
        summary["error"] = outputs["error"]
    else:
        summary["failed"] = False
        summary["value"] = outputs.get("value")
        summary["in_validated_region"] = bool(row["in_validated_region"])
    return summary


@register(
    name="query_run_store",
    json_schema=SCHEMA,
    description="Retrieve past run record(s) from THIS session's provenance store, by ID or structured filter.",
    source_module="provenance_store",
)
def query_run_store(run_id: str | None = None, filter: dict | None = None) -> ToolResult:
    if run_id is not None and filter:
        raise ToolError("query_run_store: pass either run_id or filter, not both -- the combination is ambiguous")

    inputs = dict(run_id=run_id, filter=filter)
    input_hash = compute_input_hash(inputs)
    store = get_current_store()

    diagnostic = None
    if run_id is not None:
        row = store.get(run_id)
        results = [_summarize(row)] if row is not None else []
    else:
        filt = filter or {}
        equality_filter = {"tool": filt["tool"]} if "tool" in filt else {}
        rows = store.query(**equality_filter)
        matching = [row for row in rows if _row_matches(row, filt)]
        results = [_summarize(row) for row in matching[:_MAX_RESULTS]]
        if len(matching) > _MAX_RESULTS:
            diagnostic = f"result set truncated to the first {_MAX_RESULTS} of {len(matching)} matches"

    return ToolResult(
        value={"results": results, "count": len(results)},
        units="n/a (structured run records, not a physical quantity)",
        in_validated_region=True,
        distance_from_bounds=None,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
