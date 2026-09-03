"""Tests for T1 (get_phase_map, rescoped to a deterministic re-run
checkpoint), T4 (get_medium_state), T5 (reconstruct_wavefront, Kogelnik
tier only), T9 (run_parameter_sweep, batched seed axis + looped medium
params). All need live HoloForge -- marked slow.
"""

import math

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.slow

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.tools.registry import get_tool

_common = dict(n_x=32, dx=51.2 / 32, target_pattern="square_wave", seed=0)


# ------------------------------------------------------------------ T1
def test_t1_returns_field_at_checkpoint_no_phase():
    fn = get_tool("get_phase_map").fn
    result = fn(medium_params={}, method_id="MIL", checkpoint=15, **_common)

    assert len(result.value["exposure_field"]) == 32
    assert len(result.value["recon_intensity_field"]) == 32
    assert result.value["phase"] is None
    assert "re-run" in result.diagnostic


def test_t1_is_deterministic_across_repeated_calls():
    fn = get_tool("get_phase_map").fn
    r1 = fn(medium_params={}, method_id="MIL", checkpoint=15, **_common)
    r2 = fn(medium_params={}, method_id="MIL", checkpoint=15, **_common)
    assert r1.value["exposure_field"] == r2.value["exposure_field"]


def test_t1_closed_form_method_zero_iterations():
    fn = get_tool("get_phase_map").fn
    result = fn(medium_params={}, method_id="GS", checkpoint=1, **_common)
    assert result.value["iterations_run"] == 0


# ------------------------------------------------------------------ T4
def test_t4_returns_state_with_no_dye_field():
    fn = get_tool("get_medium_state").fn
    result = fn(medium_params={}, n_x=32, dx=51.2 / 32, K_rad_per_um=8.0, n_steps=20, t_total=8, timestep=19)

    assert len(result.value["N_field"]) == 32
    assert result.value["dye_state_d"] is None
    assert math.isfinite(result.value["dn1_amplitude"])
    assert result.value["resolved_medium_params"]["D0"] == 0.1


def test_t4_rejects_out_of_range_timestep():
    fn = get_tool("get_medium_state").fn
    with pytest.raises(Exception):
        fn(medium_params={}, n_x=32, dx=51.2 / 32, K_rad_per_um=8.0, n_steps=20, t_total=8, timestep=99)


# ------------------------------------------------------------------ T5
def test_t5_efficiency_peaks_at_bragg_match():
    fn = get_tool("reconstruct_wavefront").fn
    at_bragg = fn(dn=3.5e-3, thickness_um=30.0, dtheta_deg=0.0)
    detuned = fn(dn=3.5e-3, thickness_um=30.0, dtheta_deg=3.0)
    assert at_bragg.value["efficiency"] > detuned.value["efficiency"]


def test_t5_flags_out_of_range_dn():
    fn = get_tool("reconstruct_wavefront").fn
    result = fn(dn=0.05, thickness_um=30.0, dtheta_deg=0.0)
    assert result.in_validated_region is False
    assert result.distance_from_bounds is not None


# ------------------------------------------------------------------ T9
def test_t9_batched_seed_sweep_real_batching():
    fn = get_tool("run_parameter_sweep").fn
    result = fn(
        medium_params={}, n_x=32, dx=51.2 / 32, target_pattern="square_wave", method_id="MIL",
        sweep_field="seed", sweep_values=[0, 1, 2], n_iters=15, log_every=5,
    )
    assert result.value["batched"] is True
    assert len(result.value["psnr_per_seed"]) == 3
    assert result.diagnostic is None


def test_t9_medium_param_sweep_is_honestly_looped():
    fn = get_tool("run_parameter_sweep").fn
    result = fn(
        medium_params={}, n_x=32, dx=51.2 / 32, target_pattern="square_wave", method_id="GS",
        sweep_field="kappa", sweep_values=[1.0, 2.0], n_iters=1,
    )
    assert result.value["batched"] is False
    assert len(result.value["results"]) == 2
    assert "no batching capability" in result.diagnostic


def test_t9_non_batchable_method_falls_back_to_loop_for_seed_sweep():
    fn = get_tool("run_parameter_sweep").fn
    result = fn(
        medium_params={}, n_x=32, dx=51.2 / 32, target_pattern="square_wave", method_id="LPC",
        sweep_field="seed", sweep_values=[0, 1], n_iters=1,
    )
    assert result.value["batched"] is False
    assert "no batched implementation" in result.diagnostic
