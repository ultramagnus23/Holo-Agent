"""apply_double_slit — multiply a field by two identical hard-edged 1D
slits (Young's double slit). See holoagent/domains/waveoptics/elements.py
for the physics and the exact closed-form interference-pattern validation.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.elements import apply_double_slit as apply_double_slit_fn
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
        "slit_width_um": {"type": "number", "description": "Width of each slit, microns"},
        "slit_spacing_um": {"type": "number", "description": "Center-to-center separation between the two slits, microns"},
    },
    "required": ["field", "dx", "slit_width_um", "slit_spacing_um"],
}


@register(
    name="apply_double_slit",
    json_schema=SCHEMA,
    description=(
        "Multiply a field by two identical hard-edged 1D slits (Young's double slit), "
        "centered on the grid, separated by slit_spacing_um. Produces a sinc^2-enveloped "
        "cos^2 interference pattern in the far field."
    ),
    source_module="waveoptics.elements",
)
def apply_double_slit(field: dict, dx: float, slit_width_um: float, slit_spacing_um: float) -> ToolResult:
    inputs = dict(field=field, dx=dx, slit_width_um=slit_width_um, slit_spacing_um=slit_spacing_um)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    masked = call_adapter(
        "apply_double_slit", apply_double_slit_fn, arr, dx, slit_width_um, slit_spacing_um
    )

    n_x = len(arr)
    grid_width_um = n_x * dx
    total_extent_um = slit_spacing_um + slit_width_um
    in_region = total_extent_um <= grid_width_um
    distance = None if in_region else (total_extent_um - grid_width_um) / grid_width_um
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"slit_spacing_um+slit_width_um={total_extent_um} exceeds the grid's own extent "
            f"({grid_width_um:.4g}um) -- one or both slits are clipped by the grid boundary"
        )

    return ToolResult(
        value=field_to_json(masked),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
