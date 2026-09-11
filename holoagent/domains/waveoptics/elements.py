"""Optical elements: amplitude/phase masks applied to a field (master-build
§13 Stages 3 and 5). apply_rect_aperture, apply_thin_lens, apply_grating,
apply_double_slit are built; see docs/tool_coverage_matrix.md for what's
still explicitly not built: general/arbitrary phase masks, N>2 multi-slit
arrays, and a true 2D circular aperture (this domain is 1D transverse
fields only, matching the holoforge domain's own convention -- a 1D
"circular" slit of some half-width would NOT reproduce the Airy pattern
the master prompt's §28 names, since that needs real 2D radial symmetry;
rather than build a mislabeled stand-in, that validation case is not
attempted here at all).
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.fields import make_x_axis


def apply_rect_aperture(field: np.ndarray, dx: float, width_um: float) -> np.ndarray:
    """A hard-edged 1D slit of the given width, centered on the grid."""
    n_x = len(field)
    x = make_x_axis(n_x, dx)
    mask = (np.abs(x) <= width_um / 2).astype(complex)
    return field * mask


def apply_thin_lens(field: np.ndarray, dx: float, wavelength_um: float, focal_length_um: float) -> np.ndarray:
    """Ideal thin lens (paraxial approximation): multiplies by the quadratic
    phase transmission t(x) = exp(-i*k*x^2/(2f)). Positive focal_length_um
    is converging.

    Validated (tests/test_waveoptics_propagation.py) via the classic
    Fourier-transforming-lens identity: a plane wave through an aperture
    then this lens, propagated by the exact angular-spectrum method to
    z=f, produces the same first-null position as the aperture's own
    Fraunhofer diffraction pattern (wavelength*f/width) -- the well-known
    paraxial result that a lens's back focal plane holds (a scaled) Fourier
    transform of the field just before it (Goodman, Fourier Optics ch. 5).
    """
    n_x = len(field)
    x = make_x_axis(n_x, dx)
    k = 2 * np.pi / wavelength_um
    lens_phase = np.exp(-1j * k * x**2 / (2 * focal_length_um))
    return field * lens_phase


def apply_double_slit(field: np.ndarray, dx: float, slit_width_um: float, slit_spacing_um: float) -> np.ndarray:
    """Two identical hard-edged 1D slits (Young's double slit), centered on
    the grid, separated center-to-center by slit_spacing_um. Master-build
    §13 Stage 5 ("remaining diffraction cases") -- distinct from both
    apply_rect_aperture (one slit) and apply_grating (a continuous
    sinusoidal transmission, no discrete edges): this is the archetypal
    two-source interference case.

    Validated (tests/test_waveoptics_imaging.py) against the exact
    closed-form Fraunhofer double-slit pattern I(theta) =
    4*sinc^2(a*fx)*cos^2(pi*d*fx) (fx=sin(theta)/wavelength, a=slit width,
    d=spacing) -- the Fourier transform of two identical rect functions
    offset by +-d/2, an exact analytic identity, not an approximation.
    """
    n_x = len(field)
    x = make_x_axis(n_x, dx)
    half_spacing = slit_spacing_um / 2.0
    mask = (np.abs(x - half_spacing) <= slit_width_um / 2) | (np.abs(x + half_spacing) <= slit_width_um / 2)
    return field * mask.astype(complex)


def apply_grating(field: np.ndarray, dx: float, period_um: float) -> np.ndarray:
    """Sinusoidal amplitude diffraction grating: multiplies by
    t(x) = 0.5*(1 + cos(2*pi*x/period)) -- transmission oscillating between
    0 and 1, period period_um. Produces exactly 3 diffraction orders
    (0, +1, -1; a sinusoidal grating has no higher orders), at far-field
    angles sin(theta_m) = m*wavelength/period.

    Validated (tests/test_waveoptics_propagation.py): far-field intensity
    peaks at the analytic +-1 order positions x_m = m*wavelength*z/period,
    with >99% of far-field power in exactly those 3 orders.
    """
    n_x = len(field)
    x = make_x_axis(n_x, dx)
    transmission = 0.5 * (1 + np.cos(2 * np.pi * x / period_um))
    return field * transmission
