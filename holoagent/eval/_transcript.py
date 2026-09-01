"""Builds the session record Set C audits against: the user's query plus
every tool call (inputs + outputs) made while answering it. Reads straight
from the ProvenanceStore rows an Orchestrator run already wrote — this is
not a second logging path, it's a view over the one provenance already
maintains (TRD §4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from holoagent.provenance.store import ProvenanceStore


@dataclass(frozen=True)
class ToolCallRecord:
    run_id: str
    tool: str
    inputs: dict
    outputs: dict  # asdict(ToolResult) for a success, or {"error": ...} for a failure (ProvenanceStore.record_error)


@dataclass(frozen=True)
class SessionTranscript:
    query: str
    calls: list[ToolCallRecord]


def _row_to_record(row: dict) -> ToolCallRecord:
    return ToolCallRecord(
        run_id=row["run_id"],
        tool=row["tool"],
        inputs=json.loads(row["inputs_json"]),
        outputs=json.loads(row["outputs_json"]),
    )


def transcript_from_store(
    store: ProvenanceStore,
    query: str,
    run_ids: list[str],
    error_run_ids: list[str] = (),
) -> SessionTranscript:
    """error_run_ids (OrchestratorAnswer.error_run_ids) are included so Set C
    can ground numbers the model quotes from a tool's own real error message
    (e.g. "missing 1 required positional argument") — those calls really
    happened this session and are provenance-recorded (ProvenanceStore.
    record_error), same as a successful call, just with outputs={"error": ...}
    instead of a ToolResult.
    """
    calls = []
    for run_id in [*run_ids, *error_run_ids]:
        row = store.get(run_id)
        if row is None:
            raise KeyError(f"run_id {run_id!r} not found in provenance store — cannot audit a session with a missing record")
        calls.append(_row_to_record(row))
    return SessionTranscript(query=query, calls=calls)
