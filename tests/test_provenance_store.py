from holoagent.provenance.store import ProvenanceStore
from holoagent.tools.contract import ToolResult


def _result(run_id: str, input_hash: str) -> ToolResult:
    return ToolResult(
        value=0.87,
        units="dimensionless",
        in_validated_region=True,
        distance_from_bounds=0.0,
        run_id=run_id,
        input_hash=input_hash,
    )


def test_record_and_get_round_trip(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    result = _result("run-1", "hash-1")
    store.record(tool="get_fit_quality", inputs={"run_id": "twin-1", "region": "all"}, result=result)

    row = store.get("run-1")
    assert row is not None
    assert row["tool"] == "get_fit_quality"
    assert row["input_hash"] == "hash-1"
    assert bool(row["in_validated_region"]) is True
    store.close()


def test_find_cached_short_circuits_on_identical_input_hash(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    assert store.find_cached("hash-does-not-exist") is None

    result = _result("run-2", "hash-2")
    store.record(tool="get_sensitivity", inputs={"run_id": "twin-1"}, result=result)

    cached = store.find_cached("hash-2")
    assert cached is not None
    assert cached["run_id"] == "run-2"
    store.close()


def test_query_by_tool(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    store.record(tool="get_fit_quality", inputs={}, result=_result("run-3", "hash-3"))
    store.record(tool="get_sensitivity", inputs={}, result=_result("run-4", "hash-4"))

    rows = store.query(tool="get_fit_quality")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-3"
    store.close()


def test_parent_run_id_reconstructs_chain(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    store.record(tool="get_fit_quality", inputs={}, result=_result("run-5", "hash-5"))
    store.record(
        tool="get_sensitivity",
        inputs={},
        result=_result("run-6", "hash-6"),
        parent_run_id="run-5",
    )

    child = store.get("run-6")
    assert child is not None
    assert child["parent_run_id"] == "run-5"
    store.close()
