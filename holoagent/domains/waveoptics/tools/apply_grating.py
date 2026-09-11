"""apply_grating — multiply a field by a sinusoidal amplitude diffraction
grating. See holoagent/domains/waveoptics/elements.py for the physics and
the diffraction-order validation.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.elements import apply_grating as apply_grating_fn
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
        "period_um": {"type": "number", "description": "Grating period, microns"},
    },
    "required": ["field", "dx", "period_um"],
}


@register(
    name="apply_grating",
    json_schema=SCHEMA,
    description=(
        "Multiply a field by a sinusoidal amplitude diffraction grating "
        "t(x)=0.5*(1+cos(2*pi*x/period)) -- produces exactly 3 far-field diffraction "
        "orders (0, +1, -1). Reports whether the grating period is adequately sampled by dx."
    ),
    source_module="waveoptics.elements",
)
def apply_grating(field: dict, dx: float, period_um: float) -> ToolResult:
    inputs = dict(field=field, dx=dx, period_um=period_um)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    grated = call_adapter("apply_grating", apply_grating_fn, arr, dx, period_um)

    # Real, computed sampling check: a sinusoidal grating's transmission function
    # itself needs to be resolved by the grid, independent of any downstream
    # propagation. 4 samples/period is an arbitrary-but-stated minimum to represent
    # a smooth cosine without the transmission mask itself aliasing.
    samples_per_period = period_um / dx if dx > 0 else 0.0
    in_region = samples_per_period >= 4.0
    distance = None if in_region else (4.0 - samples_per_period)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"period_um={period_um} is sampled by only {samples_per_period:.2f} grid points "
            "(need >=4) -- the grating's own transmission mask is under-resolved by dx"
        )

    return ToolResult(
        value=field_to_json(grated),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
