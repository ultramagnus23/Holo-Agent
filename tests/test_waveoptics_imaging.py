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


def test_incoherent_otf_matches_analytic_triangle_function():
    # Master-build Milestone 5, first slice: the OTF of a diffraction-
    # limited INCOHERENT imaging system with a hard-edged rect pupil is
    # NOT the pupil function itself (that's the coherent amplitude
    # transfer function, already validated above) -- it's the pupil's own
    # normalized autocorrelation, which for a rect pupil of half-width fc
    # has the exact closed form of a triangle function with cutoff 2*fc
    # (Goodman, Fourier Optics ch. 6): OTF(fx) = max(0, 1 - |fx|/(2*fc)).
    n_x, dx = 4096, 0.05
    fc = 2.0

    result = get_tool("compute_incoherent_psf_otf").fn(n_x=n_x, dx=dx, cutoff_um_inv=fc)
    assert result.in_validated_region is True, result.diagnostic
    assert result.value["otf_cutoff_um_inv"] == 2.0 * fc

    fx = np.array(result.value["fx_um_inv"])
    otf = np.array(result.value["otf"])
    analytic = np.maximum(0, 1 - np.abs(fx) / (2 * fc))
    max_abs_err = np.max(np.abs(otf - analytic))
    assert max_abs_err < 0.001, f"max abs error vs analytic triangle OTF: {max_abs_err}"

    # MTF of a real, non-negative OTF is just the OTF itself
    mtf = np.array(result.value["mtf"])
    assert np.allclose(mtf, np.abs(otf))

    # PSF is peaked and centered at x=0 (a real, physical point-spread
    # function, not an arbitrary array)
    x = np.array(result.value["x_um"])
    psf = np.array(result.value["psf_intensity"])
    assert x[np.argmax(psf)] == 0.0
    assert psf.max() == 1.0


def test_incoherent_imaging_resolves_finer_detail_than_coherent_of_same_pupil():
    # The OTF cutoff (2*fc) exceeding the coherent ATF cutoff (fc) is the
    # real physical claim being made here -- incoherent imaging genuinely
    # resolves finer spatial detail than coherent imaging through the same
    # pupil. Checked directly, not just asserted in a docstring.
    fc = 3.0
    result = get_tool("compute_incoherent_psf_otf").fn(n_x=2048, dx=0.05, cutoff_um_inv=fc)
    assert result.value["otf_cutoff_um_inv"] > result.value["coherent_cutoff_um_inv"]
    assert result.value["otf_cutoff_um_inv"] == 2.0 * result.value["coherent_cutoff_um_inv"]
