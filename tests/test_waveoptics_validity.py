"""Tests that the waveoptics domain's validity flags are real and computed
-- not fixed True -- per master-build §19 ("do not allow a model to hide
validity information"). Each test forces a genuine violation and checks
the tool actually flags it, with a real distance_from_bounds.
"""

import numpy as np

import holoagent.domains.waveoptics  # noqa: F401
from holoagent.tools.contract import ToolError
from holoagent.tools.registry import get_tool

import pytest


def test_plane_wave_aliased_tilt_is_flagged():
    # angle_deg large enough at this wavelength/dx to alias the phase ramp
    result = get_tool("create_plane_wave").fn(n_x=64, dx=2.0, wavelength_um=0.5, angle_deg=80.0)
    assert result.in_validated_region is False
    assert result.distance_from_bounds is not None
    assert result.diagnostic is not None


def test_plane_wave_normal_incidence_is_always_valid():
    result = get_tool("create_plane_wave").fn(n_x=64, dx=2.0, wavelength_um=0.5, angle_deg=0.0)
    assert result.in_validated_region is True


def test_gaussian_beam_truncated_by_narrow_grid_is_flagged():
    # grid half-width << 3x waist -- tails truncated
    result = get_tool("create_gaussian_beam").fn(n_x=16, dx=1.0, wavelength_um=0.5, waist_um=50.0)
    assert result.in_validated_region is False
    assert "truncated" in result.diagnostic


def test_gaussian_beam_under_resolved_is_flagged():
    # dx too coarse relative to the waist
    result = get_tool("create_gaussian_beam").fn(n_x=1000, dx=5.0, wavelength_um=0.5, waist_um=3.0)
    assert result.in_validated_region is False
    assert "under-resolved" in result.diagnostic


def test_gaussian_beam_well_sampled_is_valid():
    result = get_tool("create_gaussian_beam").fn(n_x=8192, dx=1.0, wavelength_um=0.5, waist_um=20.0)
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_fraunhofer_too_close_is_flagged_extrapolative():
    plane = get_tool("create_plane_wave").fn(n_x=2048, dx=0.5, wavelength_um=0.5)
    slit = get_tool("apply_aperture").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=0.5, width_um=50.0
    )
    # z far too short for the Fraunhofer approximation to hold
    result = get_tool("propagate_fraunhofer").fn(
        field={"real": slit.value["real"], "imag": slit.value["imag"]}, dx=0.5, wavelength_um=0.5, z_um=10.0
    )
    assert result.in_validated_region is False
    assert result.distance_from_bounds is not None
    assert "EXTRAPOLATIVE" in result.diagnostic


def test_aperture_wider_than_grid_is_flagged():
    plane = get_tool("create_plane_wave").fn(n_x=32, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_aperture").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=1.0, width_um=1000.0
    )
    assert result.in_validated_region is False


def test_field_real_imag_length_mismatch_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("calculate_intensity").fn(field={"real": [1.0, 2.0], "imag": [1.0]})


def test_thin_lens_fast_aperture_violates_paraxial_bound_is_flagged():
    # grid half-width comparable to f -- well outside the paraxial regime
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_thin_lens").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]},
        dx=1.0,
        wavelength_um=0.5,
        focal_length_um=100.0,
    )
    assert result.in_validated_region is False
    assert result.distance_from_bounds is not None
    assert "paraxial" in result.diagnostic


def test_thin_lens_slow_aperture_is_valid():
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_thin_lens").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]},
        dx=1.0,
        wavelength_um=0.5,
        focal_length_um=10000.0,
    )
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_grating_under_sampled_period_is_flagged():
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_grating").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=1.0, period_um=2.0
    )
    assert result.in_validated_region is False
    assert "under-resolved" in result.diagnostic


def test_grating_well_sampled_period_is_valid():
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_grating").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=1.0, period_um=20.0
    )
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_fourier_filter_cutoff_beyond_nyquist_is_flagged_no_op():
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    nyquist = 1.0 / (2.0 * 1.0)
    result = get_tool("apply_fourier_filter").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]},
        dx=1.0,
        cutoff_um_inv=nyquist * 2,
        filter_type="lowpass",
    )
    assert result.in_validated_region is False
    assert "no effect" in result.diagnostic


def test_fourier_filter_cutoff_within_nyquist_is_valid():
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_fourier_filter").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]},
        dx=1.0,
        cutoff_um_inv=0.1,
        filter_type="lowpass",
    )
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_fourier_filter_invalid_filter_type_raises_tool_error():
    plane = get_tool("create_plane_wave").fn(n_x=64, dx=1.0, wavelength_um=0.5)
    with pytest.raises(ToolError):
        get_tool("apply_fourier_filter").fn(
            field={"real": plane.value["real"], "imag": plane.value["imag"]},
            dx=1.0,
            cutoff_um_inv=0.1,
            filter_type="bandpass",
        )


def test_double_slit_exceeding_grid_extent_is_flagged():
    plane = get_tool("create_plane_wave").fn(n_x=64, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_double_slit").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]},
        dx=1.0,
        slit_width_um=5.0,
        slit_spacing_um=100.0,
    )
    assert result.in_validated_region is False
    assert result.distance_from_bounds is not None


def test_double_slit_within_grid_extent_is_valid():
    plane = get_tool("create_plane_wave").fn(n_x=256, dx=1.0, wavelength_um=0.5)
    result = get_tool("apply_double_slit").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]},
        dx=1.0,
        slit_width_um=5.0,
        slit_spacing_um=30.0,
    )
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_incoherent_otf_cutoff_beyond_grid_nyquist_is_flagged():
    # OTF cutoff is 2x the coherent cutoff -- pick a coherent cutoff whose
    # DOUBLE exceeds the grid's Nyquist frequency, even though the
    # coherent cutoff alone would not.
    dx = 1.0
    nyquist = 1.0 / (2.0 * dx)
    result = get_tool("compute_incoherent_psf_otf").fn(n_x=256, dx=dx, cutoff_um_inv=nyquist * 0.6)
    assert result.in_validated_region is False
    assert "OTF's own support" in result.diagnostic


def test_incoherent_otf_cutoff_within_grid_nyquist_is_valid():
    dx = 1.0
    nyquist = 1.0 / (2.0 * dx)
    result = get_tool("compute_incoherent_psf_otf").fn(n_x=256, dx=dx, cutoff_um_inv=nyquist * 0.2)
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_image_through_incoherent_system_cutoff_beyond_nyquist_is_flagged():
    dx = 1.0
    nyquist = 1.0 / (2.0 * dx)
    result = get_tool("image_through_incoherent_system").fn(
        object_intensity=[1.0] * 256, dx=dx, cutoff_um_inv=nyquist * 0.6
    )
    assert result.in_validated_region is False
    assert "aliased" in result.diagnostic


def test_image_through_incoherent_system_cutoff_within_nyquist_is_valid():
    dx = 1.0
    nyquist = 1.0 / (2.0 * dx)
    result = get_tool("image_through_incoherent_system").fn(
        object_intensity=[1.0] * 256, dx=dx, cutoff_um_inv=nyquist * 0.2
    )
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_shot_noise_low_photon_count_is_flagged_noise_dominated():
    result = get_tool("add_shot_noise").fn(image_intensity=[1.0] * 100, photon_scale=1.0, seed=0)
    assert result.in_validated_region is False
    assert "dominates" in result.diagnostic


def test_shot_noise_high_photon_count_is_valid():
    result = get_tool("add_shot_noise").fn(image_intensity=[1.0] * 100, photon_scale=1000.0, seed=0)
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_shot_noise_negative_intensity_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("add_shot_noise").fn(image_intensity=[-1.0, 1.0], photon_scale=10.0)


def test_shot_noise_nonpositive_photon_scale_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("add_shot_noise").fn(image_intensity=[1.0, 1.0], photon_scale=0.0)


def test_read_noise_low_signal_relative_to_noise_is_flagged():
    result = get_tool("add_read_noise").fn(image_intensity=[1.0] * 100, read_noise_std=1.0, seed=0)
    assert result.in_validated_region is False
    assert "dominates" in result.diagnostic


def test_read_noise_high_signal_relative_to_noise_is_valid():
    result = get_tool("add_read_noise").fn(image_intensity=[100.0] * 100, read_noise_std=1.0, seed=0)
    assert result.in_validated_region is True
    assert result.diagnostic is None


def test_read_noise_zero_std_is_trivially_valid():
    result = get_tool("add_read_noise").fn(image_intensity=[1.0] * 10, read_noise_std=0.0, seed=0)
    assert result.in_validated_region is True
    assert result.value["read_snr"] is None
    assert result.value["noisy_intensity"] == [1.0] * 10


def test_read_noise_negative_std_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("add_read_noise").fn(image_intensity=[1.0, 1.0], read_noise_std=-1.0)


def test_gerchberg_saxton_insufficient_iterations_is_flagged_unconverged():
    n_x, dx = 512, 0.2
    x = (np.arange(n_x) - n_x // 2) * dx
    amp = np.exp(-(x**2) / (2 * 20.0**2))
    g_true = amp * np.exp(1j * (0.4 * np.sin(2 * np.pi * x / 15.0)))
    object_intensity = (amp**2).tolist()
    fourier_intensity = (np.abs(np.fft.fft(g_true)) ** 2).tolist()

    result = get_tool("retrieve_phase_gerchberg_saxton").fn(
        object_intensity=object_intensity, fourier_intensity=fourier_intensity, n_iterations=1
    )
    assert result.in_validated_region is False
    assert "constraint error" in result.diagnostic


def test_gerchberg_saxton_length_mismatch_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_gerchberg_saxton").fn(
            object_intensity=[1.0, 1.0, 1.0], fourier_intensity=[1.0, 1.0], n_iterations=10
        )


def test_gerchberg_saxton_negative_intensity_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_gerchberg_saxton").fn(
            object_intensity=[-1.0, 1.0], fourier_intensity=[1.0, 1.0], n_iterations=10
        )


def test_gerchberg_saxton_zero_iterations_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_gerchberg_saxton").fn(
            object_intensity=[1.0, 1.0], fourier_intensity=[1.0, 1.0], n_iterations=0
        )


def test_gerchberg_saxton_zero_restarts_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_gerchberg_saxton").fn(
            object_intensity=[1.0, 1.0], fourier_intensity=[1.0, 1.0], n_iterations=10, n_restarts=0
        )


def test_hio_insufficient_iterations_is_flagged_unconverged():
    n_x, width = 256, 40
    support_start = n_x // 2 - width // 2
    support_mask = [False] * n_x
    for i in range(support_start, support_start + width):
        support_mask[i] = True
    xs = np.arange(width)
    amp = np.exp(-((xs - width / 2) ** 2) / (2 * 8.0**2))
    g_true = np.zeros(n_x, dtype=complex)
    g_true[support_start : support_start + width] = amp * np.exp(1j * 0.4 * np.sin(2 * np.pi * xs / 12.0))
    fourier_intensity = (np.abs(np.fft.fft(g_true)) ** 2).tolist()

    result = get_tool("retrieve_phase_hio").fn(
        fourier_intensity=fourier_intensity, support_mask=support_mask, n_iterations_hio=1, nonnegative=False
    )
    assert result.in_validated_region is False
    assert "constraint error" in result.diagnostic


def test_hio_length_mismatch_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_hio").fn(
            fourier_intensity=[1.0, 1.0, 1.0], support_mask=[True, True], n_iterations_hio=10
        )


def test_hio_all_false_support_mask_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_hio").fn(
            fourier_intensity=[1.0, 1.0], support_mask=[False, False], n_iterations_hio=10
        )


def test_hio_negative_intensity_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_hio").fn(
            fourier_intensity=[-1.0, 1.0], support_mask=[True, False], n_iterations_hio=10
        )


def test_hio_zero_hio_iterations_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_hio").fn(
            fourier_intensity=[1.0, 1.0], support_mask=[True, False], n_iterations_hio=0
        )


def test_hio_beta_out_of_range_raises_tool_error():
    with pytest.raises(ToolError):
        get_tool("retrieve_phase_hio").fn(
            fourier_intensity=[1.0, 1.0], support_mask=[True, False], n_iterations_hio=10, beta=1.5
        )
