"""Tests that the waveoptics domain's validity flags are real and computed
-- not fixed True -- per master-build §19 ("do not allow a model to hide
validity information"). Each test forces a genuine violation and checks
the tool actually flags it, with a real distance_from_bounds.
"""

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
