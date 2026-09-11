"""Validation tests for retrieve_phase_gerchberg_saxton (master-build §28:
every new computational module needs at least one analytic reference
case; master-build Never-Do list: never overclaim what a computation
demonstrated).

Two synthetic ground-truth cases, both run through the actual registered
tool:
1. A smooth, low-frequency phase profile with a deterministic zero-phase
   initial guess -- GS reliably converges to the true field here
   (<0.1% normalized error), the expected/textbook behavior.
2. A more oscillatory phase profile -- GS converges the measured
   constraints to near-zero error (~1e-7) while still landing far
   (~8.6%) from the true field. This is included DELIBERATELY, not
   hidden: it demonstrates the tool's own honestly-scoped validity
   claim (constraint satisfaction) is real and distinct from a claim
   this tool does NOT make (unique true-phase recovery), consistent with
   the master-build Never-Do list.

Fast: pure numpy, no HoloForge/model dependency. Not marked slow.
"""

import numpy as np

import holoagent.domains.waveoptics  # noqa: F401  (registers the waveoptics tools)
from holoagent.tools.registry import get_tool


def _normalized_field_error(g_est, g_true, amp_mask):
    # Global-phase ambiguity is inherent to two-intensity GS (multiplying
    # the whole field by exp(i*theta) doesn't change either measured
    # intensity) -- align it out before comparing, the standard protocol
    # for evaluating phase-retrieval recovery.
    overlap = np.sum(g_est * np.conj(g_true))
    theta = np.angle(overlap)
    g_aligned = g_est * np.exp(-1j * theta)
    num = np.sqrt(np.mean(np.abs(g_aligned[amp_mask] - g_true[amp_mask]) ** 2))
    den = np.sqrt(np.mean(np.abs(g_true[amp_mask]) ** 2))
    return num / den


def test_gerchberg_saxton_recovers_smooth_phase_from_synthetic_ground_truth():
    n_x, dx = 1024, 0.2
    x = (np.arange(n_x) - n_x // 2) * dx
    amp = np.exp(-(x**2) / (2 * 30.0**2))
    phase_true = 0.05 * x + 0.0008 * x**2
    g_true = amp * np.exp(1j * phase_true)

    object_intensity = (amp**2).tolist()
    fourier_intensity = (np.abs(np.fft.fft(g_true)) ** 2).tolist()

    result = get_tool("retrieve_phase_gerchberg_saxton").fn(
        object_intensity=object_intensity, fourier_intensity=fourier_intensity, n_iterations=300
    )
    assert result.in_validated_region is True, result.diagnostic
    assert result.value["fourier_constraint_error"] < 1e-6

    g_est = np.array(result.value["real"]) + 1j * np.array(result.value["imag"])
    err = _normalized_field_error(g_est, g_true, amp > 0.05)
    assert err < 0.01, f"normalized field error vs ground truth: {err}"


def test_gerchberg_saxton_constraint_satisfaction_does_not_guarantee_unique_recovery():
    # Deliberately NOT a "should recover the true phase" test -- this
    # documents a real, found-live limitation: an oscillatory phase
    # profile makes GS converge to a DIFFERENT fixed point that still
    # satisfies both measured intensities almost exactly.
    n_x, dx = 1024, 0.2
    x = (np.arange(n_x) - n_x // 2) * dx
    amp = np.exp(-(x**2) / (2 * 25.0**2))
    phase_true = 0.3 * np.sin(2 * np.pi * x / 40.0) + 0.02 * x
    g_true = amp * np.exp(1j * phase_true)

    object_intensity = (amp**2).tolist()
    fourier_intensity = (np.abs(np.fft.fft(g_true)) ** 2).tolist()

    result = get_tool("retrieve_phase_gerchberg_saxton").fn(
        object_intensity=object_intensity, fourier_intensity=fourier_intensity, n_iterations=300
    )
    # The tool's own claim (constraint satisfaction) IS true here --
    # this is the honest part of what "in_validated_region: True" means.
    assert result.in_validated_region is True, result.diagnostic
    assert result.value["fourier_constraint_error"] < 1e-5

    # But recovery of the TRUE field is NOT guaranteed by that, and here
    # genuinely fails to happen -- the tool never claimed otherwise.
    g_est = np.array(result.value["real"]) + 1j * np.array(result.value["imag"])
    err = _normalized_field_error(g_est, g_true, amp > 0.05)
    assert err > 0.05, (
        f"expected this case to demonstrate non-unique recovery (err > 0.05), got {err} -- "
        "if this now passes with a small error, GS's behavior on this specific synthetic "
        "case may have changed; re-verify the limitation is still real before loosening this test"
    )


def test_gerchberg_saxton_custom_phase_init_is_used():
    n_x, dx = 256, 0.5
    x = (np.arange(n_x) - n_x // 2) * dx
    amp = np.exp(-(x**2) / (2 * 15.0**2))
    g_true = amp * np.exp(1j * 0.02 * x)
    object_intensity = (amp**2).tolist()
    fourier_intensity = (np.abs(np.fft.fft(g_true)) ** 2).tolist()

    result_zero_init = get_tool("retrieve_phase_gerchberg_saxton").fn(
        object_intensity=object_intensity, fourier_intensity=fourier_intensity, n_iterations=50
    )
    result_custom_init = get_tool("retrieve_phase_gerchberg_saxton").fn(
        object_intensity=object_intensity,
        fourier_intensity=fourier_intensity,
        n_iterations=50,
        phase_init=(0.02 * x).tolist(),  # start at the true phase -- should converge immediately
    )
    # Starting at the true answer should reach at least as low a
    # constraint error as starting from zero, given the same iteration
    # budget (a real, checkable ordering, not just "it runs").
    assert result_custom_init.value["fourier_constraint_error"] <= result_zero_init.value["fourier_constraint_error"]
