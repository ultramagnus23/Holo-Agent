"""add_read_noise — apply additive Gaussian sensor read noise to a real
intensity image. See holoagent/domains/waveoptics/noise.py for the
physics (a fixed-magnitude noise source, distinct from signal-dependent
Poisson shot noise) and the statistical validation, including the
combined-noise variance-addition check against add_shot_noise.
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.noise import add_read_noise as add_read_noise_fn
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "image_intensity": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Real intensity image (e.g. from image_through_incoherent_system or add_shot_noise)",
        },
        "read_noise_std": {
            "type": "number",
            "description": "Standard deviation of additive Gaussian read noise, same units as image_intensity -- fixed, independent of signal level",
        },
        "seed": {"type": "integer", "description": "Optional RNG seed for reproducibility"},
    },
    "required": ["image_intensity", "read_noise_std"],
}


@register(
    name="add_read_noise",
    json_schema=SCHEMA,
    description=(
        "Apply additive Gaussian sensor read noise (fixed std, independent of signal level -- "
        "physically distinct from Poisson shot noise) to a real intensity image. Reports the "
        "resulting mean-signal/read-noise ratio and flags a read-noise-dominated regime."
    ),
    source_module="waveoptics.noise",
)
def add_read_noise(image_intensity: list, read_noise_std: float, seed: int | None = None) -> ToolResult:
    inputs = dict(image_intensity=image_intensity, read_noise_std=read_noise_std, seed=seed)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter(
        "np.asarray(image_intensity)", lambda v: np.asarray(v, dtype=float), image_intensity
    )
    noisy = call_adapter("add_read_noise", add_read_noise_fn, arr, read_noise_std, seed)

    # Real, computed diagnostic: mean-signal/read-noise ratio. Same
    # arbitrary-but-stated 3.0 floor as add_shot_noise's SNR check, for
    # the analogous "noise dominates the signal" regime -- here caused by
    # fixed sensor-electronics noise rather than photon statistics.
    mean_signal = float(np.mean(arr))
    if read_noise_std > 0:
        snr = mean_signal / read_noise_std
        in_region = snr >= 3.0
    else:
        snr = None  # no noise added -- ratio not applicable, trivially valid
        in_region = True
    distance = None if in_region else (3.0 - snr)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"mean signal={mean_signal:.3g} / read_noise_std={read_noise_std:.3g} gives SNR={snr:.3g} "
            "< 3 -- read noise dominates the underlying signal structure"
        )

    return ToolResult(
        value={"noisy_intensity": noisy.tolist(), "mean_signal": mean_signal, "read_snr": snr},
        units="intensity, arbitrary units (same scale as image_intensity input)",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
