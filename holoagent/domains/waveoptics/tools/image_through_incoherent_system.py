"""image_through_incoherent_system — simulate imaging a real intensity
object through the diffraction-limited incoherent system
compute_incoherent_psf_otf characterizes. See
holoagent/domains/waveoptics/imaging.py for the physics (convolution with
the system's own PSF, computed via the already-validated OTF) and the
linearity/MTF-contrast validations.

Unlike the field-based tools, this one's input/output are real intensity
arrays (object_intensity, image_intensity) -- NOT complex field
amplitudes -- since incoherent imaging is a relationship between
intensities, not amplitudes.
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.imaging import image_through_incoherent_system as image_through_incoherent_system_fn
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "object_intensity": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Real, non-negative intensity distribution of the object (not a field amplitude)",
        },
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "cutoff_um_inv": {
            "type": "number",
            "description": "Coherent pupil cutoff spatial frequency, cycles/micron (same as compute_incoherent_psf_otf's)",
        },
    },
    "required": ["object_intensity", "dx", "cutoff_um_inv"],
}


@register(
    name="image_through_incoherent_system",
    json_schema=SCHEMA,
    description=(
        "Simulate imaging a real intensity object through a diffraction-limited incoherent "
        "imaging system (hard-edged 1D rect pupil, coherent cutoff cutoff_um_inv) -- convolves "
        "the object with the system's own PSF via OTF multiplication in the frequency domain."
    ),
    source_module="waveoptics.imaging",
)
def image_through_incoherent_system(object_intensity: list, dx: float, cutoff_um_inv: float) -> ToolResult:
    inputs = dict(object_intensity=object_intensity, dx=dx, cutoff_um_inv=cutoff_um_inv)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter(
        "np.asarray(object_intensity)", lambda v: np.asarray(v, dtype=float), object_intensity
    )
    image = call_adapter(
        "image_through_incoherent_system", image_through_incoherent_system_fn, arr, dx, cutoff_um_inv
    )

    # Same real, computed check as compute_incoherent_psf_otf: the OTF's
    # own support (2x the coherent cutoff) must fit within the grid's
    # Nyquist frequency, or the convolution kernel this tool actually
    # applies is itself aliased.
    n_x = len(arr)
    nyquist_fx = 1.0 / (2.0 * dx)
    otf_cutoff = 2.0 * cutoff_um_inv
    in_region = 0.0 < otf_cutoff < nyquist_fx
    distance = None if in_region else abs(otf_cutoff - nyquist_fx)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"otf_cutoff_um_inv={otf_cutoff} (2x the coherent cutoff) is outside "
            f"(0, Nyquist={nyquist_fx:.4g} cycles/um) for dx={dx}um -- the imaging kernel "
            "this tool actually applies is itself aliased/clipped by the grid"
        )

    return ToolResult(
        value={"image_intensity": image.tolist()},
        units="intensity, arbitrary units (same scale as object_intensity input)",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
