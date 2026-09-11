"""Sensor shot noise (master-build Milestone 5: explicitly flagged as
missing after image_through_incoherent_system -- that tool produces an
exact, noiseless convolution result, which no real detector ever measures).

add_shot_noise -- Poisson (shot) noise, the fundamental noise floor of any
photon-counting detector, arising from the discreteness of photon arrivals
itself (not sensor electronics). A physically distinct noise process from
additive Gaussian read noise (not built here; see
docs/tool_coverage_matrix.md): its variance equals its own mean count
(Var[N] = E[N] for a Poisson process), so relative noise shrinks as
1/sqrt(signal level) rather than staying fixed.

Validated (tests/test_waveoptics_imaging.py) two ways, both real
statistical checks run on the tool's own random output, not assumed
from the Poisson distribution's textbook properties:
1. Sample variance/mean ratio across a large flat (constant-intensity)
   array matches the Poisson identity Var=mean to within ~2% (n=20000
   i.i.d. pixels).
2. Relative noise (std/mean) at two different signal levels matches the
   1/sqrt(photon_scale) scaling law to within ~5%.
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
