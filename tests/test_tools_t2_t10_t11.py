"""Tests for T2 (get_diffraction_efficiency, rescoped to the spatial field
HoloForge actually computes), T10 (query_run_store), T11 (compare_runs).

T2 needs live HoloForge (marked slow). T10/T11 are pure ProvenanceStore
operations and run fast, using holoagent.provenance.context.use_store
directly (mirroring what Orchestrator.answer() does internally) since
they're called outside of a live orchestrator run here.
"""

import math

import pytest

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.provenance.context import get_current_store, use_store
from holoagent.provenance.store import ProvenanceStore
from holoagent.tools.contract import ToolError, ToolResult
from holoagent.tools.registry import get_tool


def test_t10_raises_without_an_active_store():
    with pytest.raises(ToolError):
        get_tool("query_run_store").fn()


def test_t10_by_run_id(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    r1 = ToolResult(value={"nrmse": 0.2}, units="x", in_validated_region=True, distance_from_bounds=None, run_id="r1", input_hash="h1")
    store.record(tool="get_fit_quality", inputs={"K": 8.0}, result=r1)

    with use_store(store):
        result = get_tool("query_run_store").fn(run_id="r1")

    assert result.value["count"] == 1
    assert result.value["results"][0]["run_id"] == "r1"
    store.close()


def test_t10_filter_by_tool_and_value_threshold(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    good = ToolResult(value={"nrmse": 0.2}, units="x", in_validated_region=True, distance_from_bounds=None, run_id="good", input_hash="h1")
    bad = ToolResult(value={"nrmse": 0.6}, units="x", in_validated_region=False, distance_from_bounds=1.0, run_id="bad", input_hash="h2")
    store.record(tool="get_fit_quality", inputs={}, result=good)
    store.record(tool="get_fit_quality", inputs={}, result=bad)
    store.record(tool="get_sensitivity", inputs={}, result=ToolResult(value={"nrmse": 0.9}, units="x", in_validated_region=True, distance_from_bounds=None, run_id="other_tool", input_hash="h3"))

    with use_store(store):
        result = get_tool("query_run_store").fn(filter={"tool": "get_fit_quality", "value_field": "nrmse", "value_gt": 0.3})

    assert result.value["count"] == 1
    assert result.value["results"][0]["run_id"] == "bad"
    store.close()


def test_t10_filter_failed_only(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    store.record(tool="get_fit_quality", inputs={}, result=ToolResult(value={"x": 1.0}, units="x", in_validated_region=True, distance_from_bounds=None, run_id="ok", input_hash="h1"))
    store.record_error(tool="get_fit_quality", inputs={}, error_text="boom")

    with use_store(store):
        result = get_tool("query_run_store").fn(filter={"failed": True})

    assert result.value["count"] == 1
    assert result.value["results"][0]["failed"] is True
    store.close()


def test_t10_rejects_run_id_and_filter_together(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    with use_store(store), pytest.raises(ToolError):
        get_tool("query_run_store").fn(run_id="x", filter={"tool": "y"})
    store.close()


def test_t11_diffs_two_runs(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    a = ToolResult(value={"nrmse": 0.2, "quality": "good"}, units="x", in_validated_region=True, distance_from_bounds=None, run_id="a", input_hash="h1")
    b = ToolResult(value={"nrmse": 0.6, "quality": "poor"}, units="x", in_validated_region=False, distance_from_bounds=1.2, run_id="b", input_hash="h2")
    store.record(tool="get_fit_quality", inputs={"K_rad_per_um": 8.0}, result=a)
    store.record(tool="get_fit_quality", inputs={"K_rad_per_um": 9.0}, result=b)

    with use_store(store):
        result = get_tool("compare_runs").fn(run_id_a="a", run_id_b="b")

    assert result.value["value_diff"]["nrmse"] == {"a": 0.2, "b": 0.6}
    assert result.value["validity_diff"] is True
    store.close()


def test_t11_compares_a_success_against_a_failure(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    ok = ToolResult(value={"nrmse": 0.2}, units="x", in_validated_region=True, distance_from_bounds=None, run_id="ok", input_hash="h1")
    store.record(tool="get_fit_quality", inputs={}, result=ok)
    store.record_error(tool="get_fit_quality", inputs={}, error_text="boom")
    error_run_id = store.query(tool="get_fit_quality")[1]["run_id"]

    with use_store(store):
        result = get_tool("compare_runs").fn(run_id_a="ok", run_id_b=error_run_id)

    assert result.value["failed_a"] is False
    assert result.value["failed_b"] is True
    assert result.value["error_b"] == "boom"
    store.close()


def test_t11_unknown_run_id_raises_tool_error(tmp_path):
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    with use_store(store), pytest.raises(ToolError):
        get_tool("compare_runs").fn(run_id_a="does_not_exist", run_id_b="also_missing")
    store.close()


# ------------------------------------------------------------------ T2 (needs live HoloForge)

pytest.importorskip("torch")
_common = dict(n_x=32, dx=51.2 / 32, target_pattern="square_wave", seed=0)


@pytest.mark.slow
def test_t2_returns_spatial_field_and_scalar_efficiency():
    fn = get_tool("get_diffraction_efficiency").fn
    result = fn(medium_params={}, method_id="GS", **_common)

    assert len(result.value["recon_intensity_field"]) == 32
    assert len(result.value["exposure_field"]) == 32
    assert math.isfinite(result.value["scalar_efficiency"])
    assert 0.0 <= result.value["scalar_efficiency"] <= 1.0 + 1e-6
    assert "no phase" in result.units or "|E|^2" in result.units


@pytest.mark.slow
def test_t2_optimizer_method_populates_iterations_run():
    fn = get_tool("get_diffraction_efficiency").fn
    result = fn(medium_params={}, method_id="MIL", n_iters=15, log_every=5, **_common)

    assert result.value["iterations_run"] > 0
