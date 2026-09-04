"""T11 compare_runs — Structured diff of two run records (provenance store).

Phase 2 tool (TRD §8). Reads holoagent's own ProvenanceStore, same
session-scoped access pattern as T10 -- see
holoagent/provenance/context.py's docstring for why.

Diffs three things: tool inputs, output values (only for top-level keys
that are numeric or string -- nested structures like loss_history are
reported as "differs"/"same" rather than deep-diffed, since a full
recursive diff isn't what "compare these two runs" usually means and
adds a lot of surface for little value), and validity flags. Comparing a
successful run against a failed one is explicitly supported (that's a
real, useful comparison -- "why did this one fail when that one didn't"),
not an error case.
"""

from __future__ import annotations

import json

from holoagent.provenance.context import get_current_store
from holoagent.tools._util import compute_input_hash, new_run_id
from holoagent.tools.contract import ToolError, ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "run_id_a": {"type": "string"},
        "run_id_b": {"type": "string"},
    },
    "required": ["run_id_a", "run_id_b"],
}


def _diff_dict(a: dict, b: dict) -> dict:
    """Shallow diff: for each key present in either dict, report
    (value_a, value_b) if they differ, skip if equal. Nested dict/list
    values that differ are reported as "differs" rather than deep-diffed."""
    diff = {}
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key, "<absent>"), b.get(key, "<absent>")
        if va == vb:
            continue
        if isinstance(va, (dict, list)) or isinstance(vb, (dict, list)):
            diff[key] = "differs (nested structure, not deep-diffed)"
        else:
            diff[key] = {"a": va, "b": vb}
    return diff


@register(
    name="compare_runs",
    json_schema=SCHEMA,
    description="Structured diff of inputs, outputs, and validity flags between two run records in THIS session's provenance store.",
    source_module="provenance_store",
)
def compare_runs(run_id_a: str, run_id_b: str) -> ToolResult:
    inputs = dict(run_id_a=run_id_a, run_id_b=run_id_b)
    input_hash = compute_input_hash(inputs)
    store = get_current_store()

    row_a, row_b = store.get(run_id_a), store.get(run_id_b)
    missing = [rid for rid, row in [(run_id_a, row_a), (run_id_b, row_b)] if row is None]
    if missing:
        raise ToolError(f"compare_runs: run_id(s) not found in the provenance store: {missing}")

    outputs_a, outputs_b = json.loads(row_a["outputs_json"]), json.loads(row_b["outputs_json"])
    failed_a, failed_b = "error" in outputs_a, "error" in outputs_b

    result = {
        "tool_a": row_a["tool"],
        "tool_b": row_b["tool"],
        "same_tool": row_a["tool"] == row_b["tool"],
        "failed_a": failed_a,
        "failed_b": failed_b,
        "input_diff": _diff_dict(json.loads(row_a["inputs_json"]), json.loads(row_b["inputs_json"])),
    }

    if failed_a or failed_b:
        result["error_a"] = outputs_a.get("error")
        result["error_b"] = outputs_b.get("error")
    else:
        value_a = outputs_a.get("value") if isinstance(outputs_a.get("value"), dict) else {}
        value_b = outputs_b.get("value") if isinstance(outputs_b.get("value"), dict) else {}
        result["value_diff"] = _diff_dict(value_a, value_b)
        result["in_validated_region_a"] = bool(row_a["in_validated_region"])
        result["in_validated_region_b"] = bool(row_b["in_validated_region"])
        result["validity_diff"] = row_a["in_validated_region"] != row_b["in_validated_region"]

    return ToolResult(
        value=result,
        units="n/a (structured diff, not a physical quantity)",
        in_validated_region=True,
        distance_from_bounds=None,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=None,
    )
