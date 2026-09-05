"""propagate_fraunhofer — far-field diffraction pattern (single Fourier
transform, valid only once z exceeds the Fraunhofer distance). See
holoagent/domains/waveoptics/propagation.py for the method and its
validation against the analytic sinc^2 single-slit pattern.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.propagation import propagate_fraunhofer as _propagate_fraunhofer
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
        "wavelength_um": {"type": "number"},
        "z_um": {"type": "number", "description": "Propagation distance to the observation plane, microns"},
    },
    "required": ["field", "dx", "wavelength_um", "z_um"],
}


@register(
    name="propagate_fraunhofer",
    json_schema=SCHEMA,
    description=(
        "Far-field diffraction pattern of a field, valid once z_um exceeds the Fraunhofer distance "
        "(computed from the field's own 90%-power width, not assumed)."
    ),
    source_module="waveoptics.propagation",
)
def propagate_fraunhofer(field: dict, dx: float, wavelength_um: float, z_um: float) -> ToolResult:
    inputs = dict(field=field, dx=dx, wavelength_um=wavelength_um, z_um=z_um)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    result = call_adapter("propagate_fraunhofer", _propagate_fraunhofer, arr, dx, wavelength_um, z_um)

    diagnostic = None
    distance = None
    if not result["far_field_valid"]:
        diagnostic = (
            f"z_um={z_um} is below the Fraunhofer distance ({result['fraunhofer_distance_um']:.4g}um, "
            f"estimated from the input field's own 90%-power half-width of {result['aperture_half_width_um']:.4g}um) "
            "-- this is an EXTRAPOLATIVE far-field approximation applied too close to the aperture"
        )
        distance = (result["fraunhofer_distance_um"] - z_um) / result["fraunhofer_distance_um"]

    return ToolResult(
        value={
            "field": field_to_json(result["field"]),
            "x_out_um": result["x_out_um"],
            "fraunhofer_distance_um": result["fraunhofer_distance_um"],
        },
        units="field: complex amplitude, dimensionless; x_out_um: microns",
        in_validated_region=result["far_field_valid"],
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
