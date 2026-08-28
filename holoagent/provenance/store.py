"""Addressable, replayable run store (TRD §4). Not a log.

Identical input_hash returns the cached result without re-running the twin
(TRD §5.2) — the dominant efficiency win in a research loop with repeated
near-identical queries.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from holoagent.tools.contract import ToolResult

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class ProvenanceStore:
    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA_PATH.read_text())
        self._conn.commit()

    def record(
        self,
        *,
        tool: str,
        inputs: dict,
        result: ToolResult,
        parent_run_id: str | None = None,
    ) -> str:
        """Persist a tool invocation. Returns the run_id (== result.run_id)."""
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, timestamp, tool, input_hash, inputs_json, outputs_json,
                in_validated_region, distance_from_bounds, diagnostic, parent_run_id
            ) VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.run_id,
                tool,
                result.input_hash,
                json.dumps(inputs),
                json.dumps(asdict(result), default=str),
                int(result.in_validated_region),
                result.distance_from_bounds,
                result.diagnostic,
                parent_run_id,
            ),
        )
        self._conn.commit()
        return result.run_id

    def find_cached(self, input_hash: str) -> dict | None:
        """R5 / TRD §5.2: identical input_hash short-circuits re-computation."""
        row = self._conn.execute(
            "SELECT outputs_json FROM runs WHERE input_hash = ? ORDER BY timestamp DESC LIMIT 1",
            (input_hash,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get(self, run_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        columns = [d[0] for d in self._conn.execute("SELECT * FROM runs LIMIT 0").description]
        return dict(zip(columns, row))

    def query(self, **filters: object) -> list[dict]:
        """e.g. store.query(tool='get_fit_quality') — simple equality filters only."""
        if not filters:
            where, params = "1=1", ()
        else:
            where = " AND ".join(f"{k} = ?" for k in filters)
            params = tuple(filters.values())
        rows = self._conn.execute(f"SELECT * FROM runs WHERE {where}", params).fetchall()
        columns = [d[0] for d in self._conn.execute("SELECT * FROM runs LIMIT 0").description]
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        self._conn.close()
