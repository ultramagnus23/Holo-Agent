"""Validation tests for add_shot_noise (master-build §28: every new
computational module needs at least one analytic reference case). Both
checks here are statistical, run on the tool's own random output -- not
assumed from the Poisson distribution's textbook properties.

Fast: pure numpy, no HoloForge/model dependency. Not marked slow.
"""

import numpy as np

import holoagent.domains.waveoptics  # noqa: F401  (registers the waveoptics tools)
from holoagent.tools.registry import get_tool


def test_shot_noise_variance_equals_mean_for_a_flat_image():
    # Poisson identity: Var[N] = E[N]. Checked across n=20000 i.i.d. pixels
    # of a constant-intensity image (one tool call, many independent
    # Poisson draws) rather than assumed.
    n = 20000
    mean_intensity = 1.0
    photon_scale = 100.0  # mean count per pixel = 100

    result = get_tool("add_shot_noise").fn(
        image_intensity=[mean_intensity] * n, photon_scale=photon_scale, seed=0
    )
    assert result.in_validated_region is True, result.diagnostic
    assert abs(result.value["mean_photon_count"] - 100.0) < 1e-9
    assert abs(result.value["snr"] - 10.0) < 1e-9

    noisy_counts = np.array(result.value["noisy_intensity"]) * photon_scale
    sample_mean = noisy_counts.mean()
    sample_var = noisy_counts.var()
    rel_err = abs(sample_var / sample_mean - 1.0)
    assert rel_err < 0.05, f"Var/mean ratio should be ~1 (Poisson), got {sample_var/sample_mean}"


def test_shot_noise_relative_magnitude_scales_as_inverse_sqrt_photon_count():
    n = 20000
    low = get_tool("add_shot_noise").fn(image_intensity=[1.0] * n, photon_scale=10.0, seed=1)
    high = get_tool("add_shot_noise").fn(image_intensity=[1.0] * n, photon_scale=1000.0, seed=1)

    rel_std_low = np.std(low.value["noisy_intensity"]) / np.mean(low.value["noisy_intensity"])
    rel_std_high = np.std(high.value["noisy_intensity"]) / np.mean(high.value["noisy_intensity"])

    expected_low = 1.0 / np.sqrt(10.0)
    expected_high = 1.0 / np.sqrt(1000.0)
    assert abs(rel_std_low - expected_low) / expected_low < 0.1
    assert abs(rel_std_high - expected_high) / expected_high < 0.1


def test_shot_noise_is_reproducible_given_a_seed():
    a = get_tool("add_shot_noise").fn(image_intensity=[5.0] * 100, photon_scale=20.0, seed=42)
    b = get_tool("add_shot_noise").fn(image_intensity=[5.0] * 100, photon_scale=20.0, seed=42)
    assert a.value["noisy_intensity"] == b.value["noisy_intensity"]


def test_read_noise_has_zero_mean_and_matches_stated_std():
    n = 20000
    read_noise_std = 0.5

    result = get_tool("add_read_noise").fn(
        image_intensity=[10.0] * n, read_noise_std=read_noise_std, seed=0
    )
    assert result.in_validated_region is True, result.diagnostic

    noise = np.array(result.value["noisy_intensity"]) - 10.0
    assert abs(noise.mean()) < 0.02, f"read noise should average to ~0, got mean={noise.mean()}"
    rel_err = abs(noise.std() - read_noise_std) / read_noise_std
    assert rel_err < 0.05, f"read noise std should match read_noise_std, got {noise.std()}"


def test_read_noise_is_reproducible_given_a_seed():
    a = get_tool("add_read_noise").fn(image_intensity=[5.0] * 100, read_noise_std=1.0, seed=42)
    b = get_tool("add_read_noise").fn(image_intensity=[5.0] * 100, read_noise_std=1.0, seed=42)
    assert a.value["noisy_intensity"] == b.value["noisy_intensity"]


def test_combined_shot_and_read_noise_variances_add():
    # Independent noise sources' variances sum: Var[shot+read] =
    # Var[shot] + read_noise_std^2. Checked directly on real tool output
    # -- confirms the two noise processes are genuinely independent in
    # this implementation (not, say, accidentally correlated through a
    # shared RNG state), not merely asserted from probability theory.
    n = 20000
    photon_scale = 100.0
    read_noise_std = 0.03

    shot = get_tool("add_shot_noise").fn(image_intensity=[1.0] * n, photon_scale=photon_scale, seed=1)
    shot_only = np.array(shot.value["noisy_intensity"])
    var_shot = shot_only.var()

    combined = get_tool("add_read_noise").fn(
        image_intensity=shot.value["noisy_intensity"], read_noise_std=read_noise_std, seed=2
    )
    var_combined = np.array(combined.value["noisy_intensity"]).var()

    expected = var_shot + read_noise_std**2
    rel_err = abs(var_combined - expected) / expected
    assert rel_err < 0.05, f"Var[shot+read] should equal Var[shot]+read_std^2: {var_combined} vs {expected}"


def test_shot_noise_full_pipeline_object_to_noisy_image():
    # End-to-end: a real object, imaged through the diffraction-limited
    # incoherent system, then corrupted by real shot noise -- the actual
    # composed pipeline a caller would use, not just the two tools tested
    # in isolation.
    n_x, dx = 2048, 0.1
    fc = 2.0
    x = (np.arange(n_x) - n_x // 2) * dx
    object_intensity = (1.0 + 0.5 * np.exp(-(x**2) / (2 * 5.0**2))).tolist()

    image = get_tool("image_through_incoherent_system").fn(
        object_intensity=object_intensity, dx=dx, cutoff_um_inv=fc
    )
    noisy = get_tool("add_shot_noise").fn(
        image_intensity=image.value["image_intensity"], photon_scale=500.0, seed=7
    )
    assert noisy.in_validated_region is True, noisy.diagnostic
    assert len(noisy.value["noisy_intensity"]) == n_x

    final = get_tool("add_read_noise").fn(
        image_intensity=noisy.value["noisy_intensity"], read_noise_std=0.01, seed=8
    )
    assert final.in_validated_region is True, final.diagnostic
    assert len(final.value["noisy_intensity"]) == n_x
