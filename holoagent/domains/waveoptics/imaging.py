"""Fourier optics / 4f imaging system (master-build §13 Stage 4), plus a
diffraction-limited incoherent imaging system's PSF/OTF/MTF (master-build
Milestone 5's first slice).

A real 4f system (object -> lens(f) -> Fourier plane, f -> lens(f) ->
image plane, total distance 4f) does two things to a coherent field:
(1) at the Fourier plane, the field is (a scaled) Fourier transform of the
object -- already validated for a single lens stage in elements.py's
apply_thin_lens; (2) whatever mask is placed at the Fourier plane
multiplies that spectrum before the second lens transforms it back to an
image. The NET effect on the field, end to end, is exactly:

    image(x) = IFFT( mask(fx) * FFT(object(x)) )

-- i.e. filtering the object's own spatial-frequency spectrum. This module
implements that net effect directly (an idealized 4f system with unit
magnification and no other aberration), rather than re-simulating both
lens stages' intermediate propagation, since the intermediate fields carry
no information the net filtering operation doesn't already capture.

apply_fourier_filter -- hard-edged lowpass/highpass spatial-frequency
filter, i.e. the field a 4f system would produce with an ideal slit
(lowpass) or beam-block (highpass) stop at its Fourier plane.

Validated (tests/test_waveoptics_imaging.py) against the exact Fourier
pair rect(fx/(2*fc)) <-> 2*fc*sinc(2*fc*x): filtering a numerical impulse
(delta function) reproduces the filter's own analytic point-spread
function to <0.05% of its peak. This is not an approximation to check
against -- it is the exact closed-form impulse response of a hard-edged
frequency-domain filter, matching the diffraction-limited-resolution
result standard in Fourier optics (Goodman, ch. 6-7): a coherent imaging
system with spatial-frequency cutoff fc cannot resolve detail finer than
the sinc kernel's own width, ~1/(2*fc).
"""

from __future__ import annotations

import numpy as np


def apply_fourier_filter(field: np.ndarray, dx: float, cutoff_um_inv: float, filter_type: str = "lowpass") -> np.ndarray:
    if filter_type not in ("lowpass", "highpass"):
        raise ValueError(f"filter_type must be 'lowpass' or 'highpass', got {filter_type!r}")
    n_x = len(field)
    fx = np.fft.fftfreq(n_x, d=dx)
    within_cutoff = np.abs(fx) <= cutoff_um_inv
    mask = within_cutoff if filter_type == "lowpass" else ~within_cutoff
    F = np.fft.fft(field)
    return np.fft.ifft(F * mask.astype(complex))


def compute_incoherent_psf_otf(n_x: int, dx: float, cutoff_um_inv: float) -> dict:
    """Diffraction-limited incoherent imaging system with a hard-edged
    circular... no -- 1D rect pupil of coherent cutoff cutoff_um_inv (the
    same quantity apply_fourier_filter's cutoff_um_inv names: the coherent
    ATF's own support is |fx| <= cutoff_um_inv).

    A real, physically distinct computation from the coherent case above:
    incoherent imaging is linear in INTENSITY, not amplitude, so its
    transfer function (the OTF) is not the pupil function itself -- it is
    the pupil function's own normalized autocorrelation (Goodman, Fourier
    Optics ch. 6): OTF(fx) = [P star P](fx) / [P star P](0). For a
    rect pupil of half-width fc, this autocorrelation has an exact
    closed form: a triangle function with TWICE the coherent cutoff,
    OTF(fx) = max(0, 1 - |fx|/(2*fc)) for |fx| < 2*fc. This is the
    standard result that incoherent imaging resolves finer detail (up to
    2*fc) than coherent imaging of the same pupil (up to fc) -- a real,
    checkable physical distinction, not asserted on faith.

    Computed here via the FFT autocorrelation theorem rather than by
    symbolic convolution: OTF = normalized FFT of the INCOHERENT PSF,
    where the incoherent PSF is |coherent PSF|^2 and the coherent PSF is
    IFFT(pupil). Validated (tests/test_waveoptics_imaging.py) against the
    exact triangle-function closed form to <0.001 absolute error.
    """
    # fx and pupil are in numpy's natural (unshifted) FFT order -- fx[0]=0,
    # ascending then wrapping negative past n_x//2 -- matching how
    # apply_fourier_filter and propagation.py build their own frequency
    # grids. coherent_psf = ifft(pupil) comes out in the matching natural
    # spatial order (index 0 = x=0), NOT the centered make_x_axis order
    # used for field arrays elsewhere -- fftshift below re-centers it for
    # a human/caller-facing x axis, verified to exactly reproduce
    # make_x_axis(n_x, dx) (fftshift(fftfreq(n_x, d=1/(n_x*dx))) ==
    # make_x_axis(n_x, dx), checked numerically before writing this).
    fx = np.fft.fftfreq(n_x, d=dx)
    pupil = (np.abs(fx) <= cutoff_um_inv).astype(complex)

    coherent_psf = np.fft.ifft(pupil)
    incoherent_psf = np.abs(coherent_psf) ** 2

    # OTF via the FFT autocorrelation theorem: fft of the incoherent_psf
    # (still in natural order, so its fft is correctly labeled by the same
    # natural-order fx above -- no shift needed here).
    otf_raw = np.fft.fft(incoherent_psf)
    otf = (otf_raw / otf_raw[0]).real
    mtf = np.abs(otf)

    x_um = np.fft.fftshift(np.fft.fftfreq(n_x, d=1.0 / (n_x * dx)))
    psf_shifted = np.fft.fftshift(incoherent_psf)
    peak = psf_shifted.max()
    psf_normalized = (psf_shifted / peak) if peak > 0 else psf_shifted

    fx_order = np.argsort(fx)

    return dict(
        x_um=x_um.tolist(),
        psf_intensity=psf_normalized.tolist(),
        fx_um_inv=fx[fx_order].tolist(),
        otf=otf[fx_order].tolist(),
        mtf=mtf[fx_order].tolist(),
        coherent_cutoff_um_inv=cutoff_um_inv,
        otf_cutoff_um_inv=2.0 * cutoff_um_inv,
    )
