"""Regression tests for the tool-layer error-wrapping fix: an unexpected
exception from HoloForge itself (not a deliberately-raised ToolError) must
surface as a structured ToolError with a diagnostic, not an unhandled
exception that would crash Orchestrator.answer() entirely. Required for
PRD §6 escalation trigger 2 ("tool errored... report the failure and
diagnostic") to actually work, and for Set B's tool-errored questions to
be gradable.

These don't run real simulations (they fail fast, before any optimizer
loop or NPDD timestep), so despite touching the real HoloForge adapter
they're fast enough for the default suite -- not marked slow.
"""

import pytest

pytest.importorskip("torch")

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.tools.contract import ToolError
from holoagent.tools.registry import get_tool


def test_t3_unknown_curve_type_raises_tool_error_not_key_error():
    with pytest.raises(ToolError) as excinfo:
        get_tool("get_fit_quality").fn(
            curve_type="not_a_real_curve_type", K_rad_per_um=8.0, xs=[1, 2, 3], ys=[1, 2, 3], n_starts=1, seed=0
        )
    assert "not_a_real_curve_type" in str(excinfo.value)


def test_t3_unknown_medium_param_field_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("get_fit_quality").fn(
            curve_type="growth",
            K_rad_per_um=8.0,
            xs=[1, 2, 3],
            ys=[1, 2, 3],
            base_medium_params={"not_a_real_field": 1.0},
            n_starts=1,
            seed=0,
        )


def test_t7_unknown_method_id_raises_tool_error_not_value_error():
    with pytest.raises(ToolError) as excinfo:
        get_tool("get_convergence_trace").fn(
            medium_params={}, n_x=32, dx=1.0, target_pattern="square_wave", method_id="NOT_A_METHOD", seed=0
        )
    assert "NOT_A_METHOD" in str(excinfo.value)


def test_t7_unknown_target_pattern_raises_tool_error():
    with pytest.raises(ToolError) as excinfo:
        get_tool("get_convergence_trace").fn(
            medium_params={}, n_x=32, dx=1.0, target_pattern="not_a_pattern", method_id="GS", seed=0
        )
    assert "not_a_pattern" in str(excinfo.value)


def test_t8_non_numeric_value_raises_tool_error_not_type_error():
    with pytest.raises(ToolError):
        get_tool("check_validity_region").fn(medium_params={"D0": "not_a_number"})


def test_deliberate_missing_parameter_error_still_passes_through_unwrapped():
    """call_adapter must not double-wrap a ToolError that a tool raises on
    purpose (MissingParameterError) -- it should reach the caller exactly
    as raised, not as a generic wrapped ToolError."""
    from holoagent.tools.contract import MissingParameterError

    with pytest.raises(MissingParameterError) as excinfo:
        get_tool("get_sensitivity").fn(
            base_medium_params={},
            parameter="not_a_real_field",
            n_x=32,
            dx=1.0,
            target_pattern="square_wave",
            method_id="MIL",
            seed=0,
            n_iters=5,
        )
    assert excinfo.value.missing_field == "not_a_real_field"
