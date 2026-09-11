"""Validation tests for the waveoptics domain (master-build §28: every new
computational module needs at least one analytic reference case). These
are the actual scientific correctness checks for this domain -- not just
"does it run," but "does it match a known closed-form solution."

Two independent validations, each through the real registered tools (not
the underlying physics functions directly), so a bug in the tool wrapper
layer (JSON serialization, schema handling) would be caught too:

1. Fraunhofer diffraction through a single slit reproduces the exact
   analytic sinc^2 intensity pattern.
2. A Gaussian beam propagated by the exact (non-paraxial) angular
   spectrum method reproduces the paraxial w(z) = w0*sqrt(1+(z/zR)^2)
   divergence law, at a gentle divergence angle where the paraxial
   approximation itself is expected to hold well.

Fast: pure numpy, no HoloForge/model dependency, no torch. Not marked
slow.
"""

import numpy as np

import holoagent.domains.waveoptics  # noqa: F401  (registers the waveoptics tools)
from holoagent.tools.registry import get_tool


def test_fraunhofer_single_slit_matches_analytic_sinc_squared():
    wavelength_um = 0.5
    slit_width_um = 50.0
    n_x, dx = 4096, 0.5
    z_um = 2.0e6  # comfortably in the far field for this aperture

    plane = get_tool("create_plane_wave").fn(n_x=n_x, dx=dx, wavelength_um=wavelength_um)
    slit = get_tool("apply_aperture").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=dx, width_um=slit_width_um
    )
    far = get_tool("propagate_fraunhofer").fn(
        field={"real": slit.value["real"], "imag": slit.value["imag"]},
        dx=dx,
        wavelength_um=wavelength_um,
        z_um=z_um,
    )
    intensity = get_tool("calculate_intensity").fn(field=far.value["field"])

    assert far.in_validated_region is True, far.diagnostic

    I_numeric = np.array(intensity.value["intensity"])
    I_numeric = I_numeric / I_numeric.max()
    x_out = np.array(far.value["x_out_um"])

    theta = x_out / z_um
    u = slit_width_um * theta / wavelength_um
    with np.errstate(divide="ignore", invalid="ignore"):
        sinc = np.sin(np.pi * u) / (np.pi * u)
        sinc[u == 0] = 1.0
    I_analytic = sinc**2

    # Central lobes only -- avoids FFT grid-edge artifacts far from the
    # optical axis, which is not what this test is checking.
    mask = np.abs(x_out) < 8 * wavelength_um * z_um / slit_width_um
    max_abs_error = np.abs(I_numeric[mask] - I_analytic[mask]).max()
    assert max_abs_error < 0.02, f"max abs error vs analytic sinc^2: {max_abs_error}"


def test_gaussian_beam_angular_spectrum_matches_paraxial_w_of_z():
    wavelength_um = 0.5
    w0_um = 20.0  # wide enough that the paraxial approximation holds well
    n_x, dx = 8192, 1.0
    zR_um = np.pi * w0_um**2 / wavelength_um

    beam = get_tool("create_gaussian_beam").fn(n_x=n_x, dx=dx, wavelength_um=wavelength_um, waist_um=w0_um)
    assert beam.in_validated_region is True, beam.diagnostic

    x = (np.arange(n_x) - n_x // 2) * dx

    def measure_width(real, imag):
        mag = np.hypot(real, imag)
        peak = mag.max()
        half = mag / peak
        idx = np.where(half[n_x // 2 :] <= 1 / np.e)[0]
        i = idx[0] + n_x // 2
        x0, x1 = x[i - 1], x[i]
        y0, y1 = half[i - 1], half[i]
        return x0 + (1 / np.e - y0) * (x1 - x0) / (y1 - y0)

    w_at_waist = measure_width(beam.value["real"], beam.value["imag"])
    assert abs(w_at_waist - w0_um) / w0_um < 1e-6

    for z_um in (zR_um, 2 * zR_um):
        propagated = get_tool("propagate_angular_spectrum").fn(
            field={"real": beam.value["real"], "imag": beam.value["imag"]},
            dx=dx,
            wavelength_um=wavelength_um,
            z_um=z_um,
        )
        assert propagated.in_validated_region is True, propagated.diagnostic
        assert propagated.value["evanescent_energy_fraction"] < 1e-6

        w_numeric = measure_width(propagated.value["field"]["real"], propagated.value["field"]["imag"])
        w_analytic = w0_um * np.sqrt(1 + (z_um / zR_um) ** 2)
        rel_err = abs(w_numeric - w_analytic) / w_analytic
        assert rel_err < 0.001, f"z={z_um}: w_numeric={w_numeric} w_analytic={w_analytic} rel_err={rel_err}"


def test_thin_lens_focal_plane_matches_fourier_transforming_lens_identity():
    # Classic result (Goodman, Fourier Optics ch. 5): a plane wave through an
    # aperture then an ideal thin lens, propagated exactly to z=f, reproduces
    # the aperture's own far-field diffraction pattern -- specifically, its
    # first-null position wavelength*f/width. This IS the analytic prediction;
    # it isn't assumed, it's the textbook paraxial-lens Fourier property,
    # checked here against the tool's own exact (non-paraxial) propagation.
    wavelength_um = 0.5
    width_um = 40.0
    f_um = 5000.0
    n_x, dx = 8192, 0.25

    plane = get_tool("create_plane_wave").fn(n_x=n_x, dx=dx, wavelength_um=wavelength_um)
    slit = get_tool("apply_aperture").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=dx, width_um=width_um
    )
    lensed = get_tool("apply_thin_lens").fn(
        field={"real": slit.value["real"], "imag": slit.value["imag"]},
        dx=dx,
        wavelength_um=wavelength_um,
        focal_length_um=f_um,
    )
    assert lensed.in_validated_region is True, lensed.diagnostic

    focus = get_tool("propagate_angular_spectrum").fn(
        field={"real": lensed.value["real"], "imag": lensed.value["imag"]},
        dx=dx,
        wavelength_um=wavelength_um,
        z_um=f_um,
    )
    intensity = np.array(get_tool("calculate_intensity").fn(field=focus.value["field"]).value["intensity"])
    intensity /= intensity.max()

    x = (np.arange(n_x) - n_x // 2) * dx
    x_null_predicted = wavelength_um * f_um / width_um

    mask = (x > 0.5 * x_null_predicted) & (x < 1.5 * x_null_predicted)
    idx_in_window = np.where(mask)[0]
    null_idx = idx_in_window[np.argmin(intensity[idx_in_window])]
    rel_err = abs(x[null_idx] - x_null_predicted) / x_null_predicted
    assert rel_err < 0.01, f"first-null at x={x[null_idx]}, predicted {x_null_predicted}, rel_err={rel_err}"

    # Sanity cross-check: a sinc^2's first-null-to-first-null central lobe
    # holds ~90.3% of the total power -- confirms this is genuinely the same
    # functional form as the slit's own Fraunhofer pattern, not coincidence.
    main_lobe_power = np.sum(intensity[np.abs(x) <= x_null_predicted * 1.1])
    assert main_lobe_power / np.sum(intensity) > 0.85


def test_grating_produces_three_orders_at_analytic_angles():
    wavelength_um = 0.5
    period_um = 20.0
    n_x, dx = 8192, 0.5
    z_um = 2.0e6

    plane = get_tool("create_plane_wave").fn(n_x=n_x, dx=dx, wavelength_um=wavelength_um)
    grated = get_tool("apply_grating").fn(
        field={"real": plane.value["real"], "imag": plane.value["imag"]}, dx=dx, period_um=period_um
    )
    assert grated.in_validated_region is True, grated.diagnostic

    far = get_tool("propagate_fraunhofer").fn(
        field={"real": grated.value["real"], "imag": grated.value["imag"]},
        dx=dx,
        wavelength_um=wavelength_um,
        z_um=z_um,
    )
    intensity = np.array(get_tool("calculate_intensity").fn(field=far.value["field"]).value["intensity"])
    x_out = np.array(far.value["x_out_um"])

    expected_x = {m: m * wavelength_um * z_um / period_um for m in (-1, 0, 1)}
    x_out_spacing = wavelength_um * z_um / (n_x * dx)  # native resolution of the x_out grid
    for m, xe in expected_x.items():
        idx = np.argmin(np.abs(x_out - xe))
        window = slice(max(0, idx - 20), idx + 20)
        peak_idx = window.start + np.argmax(intensity[window])
        assert abs(x_out[peak_idx] - xe) < 5 * x_out_spacing, (
            f"order {m}: expected {xe}, found {x_out[peak_idx]}"
        )

    total_power = np.sum(intensity)
    order_power = 0.0
    for xe in expected_x.values():
        idx = np.argmin(np.abs(x_out - xe))
        window = slice(max(0, idx - 30), idx + 30)
        order_power += np.sum(intensity[window])
    assert order_power / total_power > 0.99, "a sinusoidal grating should put >99% of power in exactly 3 orders"
