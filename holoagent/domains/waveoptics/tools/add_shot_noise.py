"""add_shot_noise — apply Poisson (shot) noise to a real intensity image,
the fundamental noise floor of any photon-counting detector. See
holoagent/domains/waveoptics/noise.py for the physics and the statistical
validation (sample variance/mean ratio, 1/sqrt(signal) scaling law).
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.noise import add_shot_noise as add_shot_noise_fn
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "image_intensity": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Real, non-negative noiseless intensity image (e.g. from image_through_incoherent_system)",
        },
        "photon_scale": {
            "type": "number",
            "description": "Photons per unit intensity -- controls signal level; noisy_counts = Poisson(image_intensity * photon_scale)",
        },
        "seed": {"type": "integer", "description": "Optional RNG seed for reproducibility"},
    },
    "required": ["image_intensity", "photon_scale"],
}


@register(
    name="add_shot_noise",
    json_schema=SCHEMA,
    description=(
        "Apply Poisson shot noise to a real intensity image at the given photon_scale (photons per "
        "unit intensity). Reports the mean photon count and its resulting theoretical SNR=sqrt(mean "
        "count), and flags a noise-dominated (low-SNR) regime."
    ),
    source_module="waveoptics.noise",
)
def add_shot_noise(image_intensity: list, photon_scale: float, seed: int | None = None) -> ToolResult:
    inputs = dict(image_intensity=image_intensity, photon_scale=photon_scale, seed=seed)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter(
        "np.asarray(image_intensity)", lambda v: np.asarray(v, dtype=float), image_intensity
    )
    noisy = call_adapter("add_shot_noise", add_shot_noise_fn, arr, photon_scale, seed)

    # Real, computed diagnostic: theoretical Poisson SNR = sqrt(mean photon
    # count). SNR < 3 is a standard (arbitrary-but-stated, same convention
    # as this domain's other thresholds) cutoff for a "noise-dominated"
    # regime where shot noise swamps the underlying signal structure.
    mean_count = float(np.mean(arr) * photon_scale)
    snr = float(np.sqrt(mean_count)) if mean_count > 0 else 0.0
    in_region = snr >= 3.0
    distance = None if in_region else (3.0 - snr)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"mean photon count={mean_count:.3g} gives SNR=sqrt(mean count)={snr:.3g} < 3 -- "
            "shot noise dominates the underlying signal structure at this photon_scale"
        )

    return ToolResult(
        value={"noisy_intensity": noisy.tolist(), "mean_photon_count": mean_count, "snr": snr},
        units="intensity, arbitrary units (same scale as image_intensity input)",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
