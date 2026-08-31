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
    outputs: dict  # asdict(ToolResult), as stored


@dataclass(frozen=True)
class SessionTranscript:
    query: str
    calls: list[ToolCallRecord]


def transcript_from_store(store: ProvenanceStore, query: str, run_ids: list[str]) -> SessionTranscript:
    calls = []
    for run_id in run_ids:
        row = store.get(run_id)
        if row is None:
            raise KeyError(f"run_id {run_id!r} not found in provenance store — cannot audit a session with a missing record")
        calls.append(
            ToolCallRecord(
                run_id=row["run_id"],
                tool=row["tool"],
                inputs=json.loads(row["inputs_json"]),
                outputs=json.loads(row["outputs_json"]),
            )
        )
    return SessionTranscript(query=query, calls=calls)
