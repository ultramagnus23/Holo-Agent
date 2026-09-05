"""apply_aperture — multiply a field by a hard-edged 1D rectangular
(slit) aperture mask. See holoagent/domains/waveoptics/elements.py for
why this domain does not offer a circular aperture (would need true 2D
fields to mean what "circular" implies; not built).
"""

from __future__ import annotations

from holoagent.domains.waveoptics.elements import apply_rect_aperture
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
        "width_um": {"type": "number", "description": "Full slit width, microns"},
    },
    "required": ["field", "dx", "width_um"],
}


@register(
    name="apply_aperture",
    json_schema=SCHEMA,
    description="Multiply a field by a hard-edged 1D rectangular (slit) aperture, centered on the grid.",
    source_module="waveoptics.elements",
)
def apply_aperture(field: dict, dx: float, width_um: float) -> ToolResult:
    inputs = dict(field=field, dx=dx, width_um=width_um)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    masked = call_adapter("apply_rect_aperture", apply_rect_aperture, arr, dx, width_um)

    n_x = len(arr)
    grid_width_um = n_x * dx
    in_region = width_um <= grid_width_um
    distance = None if in_region else (width_um - grid_width_um) / grid_width_um
    diagnostic = None
    if not in_region:
        diagnostic = f"width_um={width_um} exceeds the grid's own extent ({grid_width_um:.4g}um) -- aperture is clipped by the grid boundary, not by width_um itself"

    return ToolResult(
        value=field_to_json(masked),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
