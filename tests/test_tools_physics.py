"""Per-tool tests against LIVE HoloForge for the four Phase 1 tools
(T3, T6, T7, T8). Marked slow — these run real simulations, not mocks.
See test_orchestrator_e2e.py for the chained multi-tool test (R6).
"""

import math

import pytest

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.tools.contract import MissingParameterError
from holoagent.tools.registry import get_tool

pytest.importorskip("torch")
pytestmark = pytest.mark.slow

_COMMON = dict(n_x=64, dx=51.2 / 64, target_pattern="square_wave", method_id="MIL", seed=0)


def test_t7_convergence_trace_real_run():
    fn = get_tool("get_convergence_trace").fn
    result = fn(medium_params={}, n_iters=30, log_every=10, **_COMMON)
    assert result.value["iterations_run"] > 0
    assert len(result.value["loss_history"]) > 0
    # loss must be monotonically non-increasing at the logged checkpoints
    # for MIL on a fixed seed -- not a tautology, this is what "converging"
    # actually means for this optimizer.
    losses = [loss for _, loss in result.value["loss_history"]]
    assert all(a >= b - 1e-6 for a, b in zip(losses, losses[1:])), losses
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_t7_closed_form_method_has_empty_history():
    fn = get_tool("get_convergence_trace").fn
    result = fn(medium_params={}, method_id="GS", n_x=64, dx=51.2 / 64, target_pattern="square_wave", seed=0)
    assert result.value["iterations_run"] == 0
    assert result.value["loss_history"] == []
    assert result.value["early_stop_reason"] == "closed_form_no_optimization"


def test_t8_in_bounds_and_out_of_bounds():
    fn = get_tool("check_validity_region").fn

    ok = fn(medium_params={"D0": 0.1, "sigma": 0.08}, K_rad_per_um=8.0)
    assert ok.in_validated_region is True
    assert ok.distance_from_bounds is None

    bad = fn(medium_params={"D0": 50.0}, K_rad_per_um=None)
    assert bad.in_validated_region is False
    assert bad.distance_from_bounds is not None
    assert bad.distance_from_bounds > 0
    assert "D0" in bad.value["violations"]
    assert bad.diagnostic is not None


def test_t6_sensitivity_returns_finite_difference_not_autograd():
    fn = get_tool("get_sensitivity").fn
    result = fn(
        base_medium_params={},
        parameter="sigma",
        perturbation_pct=0.1,
        n_iters=25,
        **_COMMON,
    )
    assert result.value["method"] == "finite_difference_central"
    assert "NOT autograd" in result.units
    assert math.isfinite(result.value["d_output_d_param"])


def test_t6_missing_parameter_raises_structured_error():
    fn = get_tool("get_sensitivity").fn
    with pytest.raises(MissingParameterError) as excinfo:
        fn(
            base_medium_params={},
            parameter="not_a_real_field",
            n_iters=5,
            **_COMMON,
        )
    assert excinfo.value.missing_field == "not_a_real_field"


def test_t3_fit_quality_real_fit():
    fn = get_tool("get_fit_quality").fn
    xs = [0.1, 0.2, 0.3, 0.5, 0.8]
    ys = [0.02, 0.05, 0.09, 0.15, 0.22]
    result = fn(curve_type="growth", K_rad_per_um=8.98, xs=xs, ys=ys, n_starts=2, seed=0)
    assert 0.0 <= result.value["nrmse"]
    assert result.value["quality"] in ("good", "moderate", "poor")
    assert result.units.startswith("nrmse")
