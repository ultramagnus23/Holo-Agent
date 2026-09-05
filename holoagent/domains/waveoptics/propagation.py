"""Scalar free-space propagation (master-build §13 Stage 2). Two methods,
each returning real, computed validity diagnostics alongside the
propagated field -- not a fixed True, per master-build §19 ("do not allow
a model to hide validity information").

angular_spectrum_propagate -- EXACT within scalar diffraction theory (no
paraxial approximation), valid for any z. Standard method: FFT the field,
multiply by the exact propagation transfer function exp(i*kz*z) with
kz=sqrt(k^2-kx^2), IFFT back. Evanescent components (kx>k, where kz would
be imaginary) are dropped -- physically correct (no real propagating wave
carries that energy at any z>0), but lossy for near-field detail finer
than about one wavelength. The tool reports what fraction of the field's
spectral energy was evanescent, so a caller asking about sub-wavelength
structure gets a real number telling them how much was discarded, not
silence.

propagate_fraunhofer -- the far-field limit: the propagated field is
proportional to the input field's own Fourier transform, evaluated at
angle-scaled output coordinates. Only valid once z exceeds the Fraunhofer
distance (~ w^2/wavelength, w a characteristic aperture size) -- computed
here from the INPUT field's own 90%-power half-width (via its intensity
centroid), not taken on faith from a separate parameter the caller could
get wrong.

Validated against closed-form analytic solutions (see
tests/test_waveoptics_propagation.py): angular_spectrum_propagate
reproduces the paraxial Gaussian-beam w(z) law to <0.03% relative error;
propagate_fraunhofer reproduces the exact sinc^2 single-slit diffraction
pattern to <0.01 absolute intensity error (normalized peak=1).
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.fields import make_x_axis


def angular_spectrum_propagate(field: np.ndarray, dx: float, wavelength_um: float, z_um: float, n0: float = 1.0) -> dict:
    n_x = len(field)
    k = 2 * np.pi * n0 / wavelength_um
    fx = np.fft.fftfreq(n_x, d=dx)
    kx = 2 * np.pi * fx
    kz2 = k**2 - kx**2
    propagating = kz2 >= 0
    kz = np.sqrt(np.clip(kz2, 0, None))
    H = np.where(propagating, np.exp(1j * kz * z_um), 0.0)

    F = np.fft.fft(field)
    energy_total = float(np.sum(np.abs(F) ** 2))
    energy_evanescent = float(np.sum(np.abs(F[~propagating]) ** 2))
    evanescent_fraction = (energy_evanescent / energy_total) if energy_total > 0 else 0.0

    propagated = np.fft.ifft(F * H)

    # Sampling adequacy, checked against the field's OWN spectral content,
    # not a blanket worst-case rule. dx <= wavelength/2 is the bound for
    # representing ANY possible propagating spatial frequency up to k
    # without aliasing -- correct for an arbitrary/broadband field, but
    # needlessly conservative for a field (e.g. a gently-diverging
    # Gaussian beam) whose actual energy sits in a narrow band near kx=0.
    # Found live: a w0=20um beam at dx=1.0um, wavelength=0.5um "failed"
    # the blanket rule (nyquist_dx_um=0.25um) while numerically matching
    # its analytic w(z) to 0.01% -- the blanket check was flagging a
    # sampling risk that didn't actually exist for this field. Fixed by
    # checking what fraction of the field's real spectral energy sits in
    # the outer 10% of the grid's sampled frequency range (near the
    # Nyquist edge, where aliasing actually bites), instead.
    nyquist_dx_um = wavelength_um / 2.0
    fx_max = 1.0 / (2.0 * dx)
    near_edge = np.abs(fx) >= 0.9 * fx_max
    energy_near_edge = float(np.sum(np.abs(F[near_edge]) ** 2))
    nyquist_edge_energy_fraction = (energy_near_edge / energy_total) if energy_total > 0 else 0.0

    return dict(
        field=propagated,
        evanescent_energy_fraction=evanescent_fraction,
        nyquist_satisfied=bool(dx <= nyquist_dx_um),  # kept for reference/diagnosis, not the sole gate
        nyquist_dx_um=nyquist_dx_um,
        nyquist_edge_energy_fraction=nyquist_edge_energy_fraction,
    )


def propagate_fraunhofer(field: np.ndarray, dx: float, wavelength_um: float, z_um: float) -> dict:
    n_x = len(field)
    F = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field))) * dx
    fx = np.fft.fftshift(np.fft.fftfreq(n_x, d=dx))
    x_out = fx * wavelength_um * z_um
    k = 2 * np.pi / wavelength_um
    prefactor = np.exp(1j * k * z_um) / (1j * wavelength_um * z_um)
    far_field = prefactor * F

    intensity_in = np.abs(field) ** 2
    total_power = float(np.sum(intensity_in))
    if total_power > 0:
        x_in = make_x_axis(n_x, dx)
        centroid = float(np.sum(x_in * intensity_in) / total_power)
        r = np.abs(x_in - centroid)
        order = np.argsort(r)
        cum = np.cumsum(intensity_in[order])
        idx90 = int(np.searchsorted(cum, 0.9 * total_power))
        half_width_um = float(r[order[min(idx90, n_x - 1)]])
    else:
        half_width_um = 0.0
    fraunhofer_distance_um = (half_width_um**2) / wavelength_um if wavelength_um > 0 else float("inf")

    return dict(
        field=far_field,
        x_out_um=x_out.tolist(),
        aperture_half_width_um=half_width_um,
        fraunhofer_distance_um=fraunhofer_distance_um,
        far_field_valid=bool(z_um >= fraunhofer_distance_um),
    )
