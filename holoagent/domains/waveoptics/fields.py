"""Fundamental scalar wave-optics fields (master-build §13 Stage 1).

Pure, deterministic, no external physics engine to wrap -- this domain
computes directly from standard scalar diffraction theory. Every function
here is validated against a closed-form analytic result; see
tests/test_waveoptics_propagation.py for the validation cases (this
module's own functions are the inputs to those, not independently
validated in isolation -- a plane wave and a Gaussian beam are definitions,
not predictions, so there is nothing to check them against except correct
implementation of the formula).
"""

from __future__ import annotations

import numpy as np


def make_x_axis(n_x: int, dx: float) -> np.ndarray:
    """Centered spatial grid, x=0 at index n_x//2 (matches numpy.fft's own
    zero-frequency convention, so fftshift/ifftshift pairs correctly)."""
    return (np.arange(n_x) - n_x // 2) * dx


def plane_wave(n_x: int, dx: float, wavelength_um: float, angle_deg: float = 0.0, amplitude: float = 1.0) -> np.ndarray:
    """A tilted plane wave: amplitude * exp(i*k*sin(angle_deg)*x).
    angle_deg=0 is normal incidence (spatially constant field)."""
    x = make_x_axis(n_x, dx)
    k = 2 * np.pi / wavelength_um
    return amplitude * np.exp(1j * k * np.sin(np.radians(angle_deg)) * x)


def gaussian_beam(n_x: int, dx: float, wavelength_um: float, waist_um: float, amplitude: float = 1.0) -> np.ndarray:
    """A Gaussian beam at its waist (z=0, planar phase front):
    E(x) = amplitude * exp(-x^2/waist_um^2). Validated (see
    tests/test_waveoptics_propagation.py) to reproduce the paraxial
    w(z) = w0*sqrt(1+(z/zR)^2) divergence law under
    propagation.angular_spectrum_propagate, to <0.03% relative error at a
    gentle divergence angle (w0=20um at wavelength=0.5um)."""
    x = make_x_axis(n_x, dx)
    return (amplitude * np.exp(-(x**2) / (waist_um**2))).astype(complex)
