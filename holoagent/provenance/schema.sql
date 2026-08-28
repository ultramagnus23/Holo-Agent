-- Provenance store schema (TRD §4).
-- SQLite for records; large arrays (phase maps, fields) go on disk, not here —
-- this table stores a path/reference for those, never the raw bytes.

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,          -- ISO 8601 UTC
    tool            TEXT NOT NULL,          -- tool name from the registry
    input_hash      TEXT NOT NULL,          -- reproducibility key (R5)
    inputs_json     TEXT NOT NULL,          -- JSON-serialized tool inputs
    outputs_json    TEXT NOT NULL,          -- JSON-serialized ToolResult (value may be a path for large arrays)
    in_validated_region INTEGER NOT NULL,   -- 0/1
    distance_from_bounds REAL,
    diagnostic      TEXT,
    parent_run_id   TEXT,                   -- reconstructs multi-step chains
    FOREIGN KEY (parent_run_id) REFERENCES runs (run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_tool ON runs (tool);
CREATE INDEX IF NOT EXISTS idx_runs_input_hash ON runs (input_hash);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs (parent_run_id);
