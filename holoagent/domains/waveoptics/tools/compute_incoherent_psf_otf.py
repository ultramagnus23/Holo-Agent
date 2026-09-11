"""compute_incoherent_psf_otf — PSF/OTF/MTF of a diffraction-limited
incoherent imaging system with a hard-edged 1D rect pupil. See
holoagent/domains/waveoptics/imaging.py for the physics (why incoherent
OTF is NOT the coherent pupil function, but its autocorrelation) and the
exact triangle-function validation.

Unlike the other waveoptics tools, this one does not transform an input
field -- it characterizes an imaging SYSTEM (pupil cutoff -> PSF/OTF/MTF),
the standard object of study for resolution/contrast questions, distinct
from propagating any particular object through it.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.imaging import compute_incoherent_psf_otf as compute_incoherent_psf_otf_fn
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "n_x": {"type": "integer", "description": "Grid size (number of samples)"},
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "cutoff_um_inv": {
            "type": "number",
            "description": "Coherent pupil cutoff spatial frequency, cycles/micron (same quantity as apply_fourier_filter's cutoff_um_inv)",
        },
    },
    "required": ["n_x", "dx", "cutoff_um_inv"],
}


@register(
    name="compute_incoherent_psf_otf",
    json_schema=SCHEMA,
    description=(
        "Compute the point-spread function (PSF), optical transfer function (OTF), and "
        "modulation transfer function (MTF) of a diffraction-limited INCOHERENT imaging "
        "system with a hard-edged 1D rect pupil of coherent cutoff cutoff_um_inv. The OTF "
        "cutoff is exactly 2x the coherent (amplitude) cutoff -- incoherent imaging resolves "
        "finer detail than coherent imaging of the same pupil."
    ),
    source_module="waveoptics.imaging",
)
def compute_incoherent_psf_otf(n_x: int, dx: float, cutoff_um_inv: float) -> ToolResult:
    inputs = dict(n_x=n_x, dx=dx, cutoff_um_inv=cutoff_um_inv)
    input_hash = compute_input_hash(inputs)

    result = call_adapter(
        "compute_incoherent_psf_otf", compute_incoherent_psf_otf_fn, n_x, dx, cutoff_um_inv
    )

    # Real, computed check: the OTF's own support extends to 2*cutoff_um_inv
    # (twice the coherent cutoff). If that exceeds the grid's Nyquist
    # frequency 1/(2*dx), the OTF triangle is itself aliased/clipped by the
    # grid -- a genuine sampling-adequacy condition for THIS computation,
    # not the coherent-only check apply_fourier_filter makes.
    nyquist_fx = 1.0 / (2.0 * dx)
    otf_cutoff = result["otf_cutoff_um_inv"]
    in_region = 0.0 < otf_cutoff < nyquist_fx
    distance = None if in_region else abs(otf_cutoff - nyquist_fx)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"otf_cutoff_um_inv={otf_cutoff} (2x the coherent cutoff) is outside "
            f"(0, Nyquist={nyquist_fx:.4g} cycles/um) for dx={dx}um -- the OTF's own support "
            "is aliased/clipped by the grid, not just the coherent pupil's"
        )

    return ToolResult(
        value=result,
        units="PSF: normalized intensity (peak=1); OTF/MTF: dimensionless (DC-normalized)",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
