"""Validation tests for the waveoptics domain's Stage 4 (Fourier optics /
4f imaging system) tool: apply_fourier_filter. Master-build §28: every new
computational module needs at least one analytic reference case.

apply_fourier_filter implements the net field-level effect of an idealized
4f imaging system with a hard-edged spatial-frequency stop at its Fourier
plane. The exact closed-form check: a rect-shaped passband in the
frequency domain is the Fourier transform of a sinc function in the
spatial domain -- so filtering a numerical impulse (delta function)
through the lowpass filter must reproduce the filter's own analytic
impulse response, 2*fc*sinc(2*fc*x). This is an exact Fourier-transform
pair, not an approximation.

Fast: pure numpy, no HoloForge/model dependency. Not marked slow.
"""

import numpy as np

import holoagent.domains.waveoptics  # noqa: F401  (registers the waveoptics tools)
from holoagent.tools.registry import get_tool


def test_lowpass_filter_impulse_response_matches_analytic_sinc():
    n_x, dx = 8192, 0.05
    fc = 2.0  # cutoff spatial frequency, cycles/um

    # numerical delta: single nonzero sample scaled by 1/dx so its discrete
    # sum approximates a continuous impulse of unit area
    real = [0.0] * n_x
    imag = [0.0] * n_x
    real[n_x // 2] = 1.0 / dx

    result = get_tool("apply_fourier_filter").fn(
        field={"real": real, "imag": imag}, dx=dx, cutoff_um_inv=fc, filter_type="lowpass"
    )
    assert result.in_validated_region is True, result.diagnostic

    filtered_real = np.array(result.value["real"])
    x = (np.arange(n_x) - n_x // 2) * dx
    analytic = 2 * fc * np.sinc(2 * fc * x)

    mask = np.abs(x) < 5.0
    max_abs_err = np.max(np.abs(filtered_real[mask] - analytic[mask]))
    assert max_abs_err < 0.05, f"max abs error vs analytic sinc impulse response: {max_abs_err}"


def test_lowpass_and_highpass_are_exact_complements():
    # A hard lowpass + hard highpass at the same cutoff must sum back to
    # the original field exactly (their frequency masks partition all of
    # fx with no overlap and no gap).
    n_x, dx = 512, 0.5
    rng = np.random.default_rng(0)
    real = rng.normal(size=n_x).tolist()
    imag = rng.normal(size=n_x).tolist()

    low = get_tool("apply_fourier_filter").fn(
        field={"real": real, "imag": imag}, dx=dx, cutoff_um_inv=0.4, filter_type="lowpass"
    )
    high = get_tool("apply_fourier_filter").fn(
        field={"real": real, "imag": imag}, dx=dx, cutoff_um_inv=0.4, filter_type="highpass"
    )

    reconstructed_real = np.array(low.value["real"]) + np.array(high.value["real"])
    reconstructed_imag = np.array(low.value["imag"]) + np.array(high.value["imag"])
    assert np.max(np.abs(reconstructed_real - np.array(real))) < 1e-9
    assert np.max(np.abs(reconstructed_imag - np.array(imag))) < 1e-9


def test_lowpass_leaves_a_broadband_gaussian_beam_unchanged_when_cutoff_is_generous():
    beam = get_tool("create_gaussian_beam").fn(n_x=4096, dx=1.0, wavelength_um=0.5, waist_um=20.0)
    # this beam's spectral width is ~1/(pi*w0) ~ 0.016 cycles/um -- a cutoff
    # of 1.0 cycles/um is far above where any of its real energy sits
    result = get_tool("apply_fourier_filter").fn(
        field={"real": beam.value["real"], "imag": beam.value["imag"]},
        dx=1.0,
        cutoff_um_inv=1.0,
        filter_type="lowpass",
    )
    diff = np.max(np.abs(np.array(result.value["real"]) - np.array(beam.value["real"])))
    assert diff < 1e-6
