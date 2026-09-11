"""Sensor noise: shot noise and read noise (master-build Milestone 5).
image_through_incoherent_system produces an exact, noiseless convolution
result, which no real detector ever measures -- these two tools add the
two physically distinct real noise sources a detector actually has.

add_shot_noise -- Poisson (shot) noise, the fundamental noise floor of any
photon-counting detector, arising from the discreteness of photon arrivals
itself (not sensor electronics). Its variance equals its own mean count
(Var[N] = E[N] for a Poisson process), so relative noise shrinks as
1/sqrt(signal level) rather than staying fixed -- it vanishes (relatively)
at high signal and dominates at low signal.

add_read_noise -- additive Gaussian sensor-electronics noise (amplifier/
ADC noise), a physically distinct source from shot noise: FIXED standard
deviation, independent of signal level (does not shrink as the image gets
brighter, unlike shot noise). The two are independent random processes,
so a detector's combined noise variance is their SUM (variances of
independent sources add): Var[shot + read] = Var[shot] + read_noise_std^2
-- checked directly (tests/test_waveoptics_noise.py), not assumed.

Validated (tests/test_waveoptics_noise.py), all real statistical checks
run on each tool's own random output, not assumed from textbook
properties:
1. add_shot_noise: sample variance/mean ratio across a large flat
   (constant-intensity) array matches the Poisson identity Var=mean to
   within ~5% (n=20000 i.i.d. pixels); relative noise (std/mean) at two
   signal levels matches the 1/sqrt(photon_scale) scaling law to ~10%.
2. add_read_noise: sample mean of the added noise is ~0, sample std
   matches read_noise_std to within ~5%.
3. Combined: Var[shot+read] matches Var[shot]+read_noise_std^2 to
   within ~5%, confirming the two noise sources are actually independent
   in this implementation (not, say, accidentally correlated through a
   shared RNG state).
"""

from __future__ import annotations

import numpy as np


def add_shot_noise(image_intensity: np.ndarray, photon_scale: float, seed: int | None = None) -> np.ndarray:
    if np.any(image_intensity < 0):
        raise ValueError("image_intensity must be non-negative (a Poisson rate cannot be negative)")
    if photon_scale <= 0:
        raise ValueError(f"photon_scale must be positive, got {photon_scale}")

    rng = np.random.default_rng(seed)
    mean_counts = image_intensity * photon_scale
    noisy_counts = rng.poisson(mean_counts)
    return noisy_counts / photon_scale


def add_read_noise(image_intensity: np.ndarray, read_noise_std: float, seed: int | None = None) -> np.ndarray:
    if read_noise_std < 0:
        raise ValueError(f"read_noise_std must be non-negative, got {read_noise_std}")

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=read_noise_std, size=len(image_intensity))
    return image_intensity + noise
