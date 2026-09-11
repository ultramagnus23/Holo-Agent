"""Fourier optics / 4f imaging system (master-build §13 Stage 4).

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
