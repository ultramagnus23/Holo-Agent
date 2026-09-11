"""apply_fourier_filter — idealized 4f-imaging-system spatial-frequency
filter (lowpass/highpass). See holoagent/domains/waveoptics/imaging.py for
the physics, why this reproduces a real 4f system's net effect, and the
exact-Fourier-pair validation.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.imaging import apply_fourier_filter as apply_fourier_filter_fn
from holoagent.domains.waveoptics.serialization import field_from_json, field_to_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "field": {
            "type": "object",
            "properties": {"real": {"type": "array"}, "imag": {"type": "array"}},
        },
        "dx": {"type": "number", "description": "Grid spacing, microns -- must match the field's own grid"},
        "cutoff_um_inv": {"type": "number", "description": "Spatial-frequency cutoff, cycles/micron"},
        "filter_type": {"type": "string", "enum": ["lowpass", "highpass"], "default": "lowpass"},
    },
    "required": ["field", "dx", "cutoff_um_inv"],
}


@register(
    name="apply_fourier_filter",
    json_schema=SCHEMA,
    description=(
        "Apply an idealized hard-edged spatial-frequency filter (lowpass or highpass) at "
        "cutoff_um_inv -- the net field-level effect of a 4f imaging system with a slit stop "
        "or beam block at its Fourier plane. Reports whether the cutoff is below the grid's "
        "own Nyquist frequency (above it, the filter is a physical no-op)."
    ),
    source_module="waveoptics.imaging",
)
def apply_fourier_filter(field: dict, dx: float, cutoff_um_inv: float, filter_type: str = "lowpass") -> ToolResult:
    inputs = dict(field=field, dx=dx, cutoff_um_inv=cutoff_um_inv, filter_type=filter_type)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    filtered = call_adapter(
        "apply_fourier_filter", apply_fourier_filter_fn, arr, dx, cutoff_um_inv, filter_type
    )

    # Real, computed check: the grid can only represent spatial frequencies
    # up to its own Nyquist frequency (1/(2*dx)). A cutoff at or beyond that
    # keeps (lowpass) or removes (highpass) nothing the grid wasn't already
    # excluding -- the filter call is a physical no-op, not an error, but
    # the caller should know it didn't do what they likely intended.
    nyquist_fx = 1.0 / (2.0 * dx)
    in_region = 0.0 < cutoff_um_inv < nyquist_fx
    distance = None if in_region else abs(cutoff_um_inv - nyquist_fx)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"cutoff_um_inv={cutoff_um_inv} is outside (0, Nyquist={nyquist_fx:.4g} cycles/um) for "
            f"dx={dx}um -- the filter has no effect beyond what the grid's own sampling already excludes"
        )

    return ToolResult(
        value=field_to_json(filtered),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
