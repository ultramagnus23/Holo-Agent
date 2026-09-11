"""apply_thin_lens — multiply a field by an ideal thin lens's paraxial
quadratic phase transmission. See holoagent/domains/waveoptics/elements.py
for the physics and the Fourier-transforming-lens validation.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.elements import apply_thin_lens as apply_thin_lens_fn
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
        "focal_length_um": {"type": "number", "description": "Positive = converging lens, microns"},
    },
    "required": ["field", "dx", "wavelength_um", "focal_length_um"],
}


@register(
    name="apply_thin_lens",
    json_schema=SCHEMA,
    description=(
        "Multiply a field by an ideal thin lens's paraxial quadratic phase transmission "
        "exp(-i*k*x^2/(2f)). Reports how close the aperture's half-width/focal-length ratio "
        "comes to violating the paraxial approximation the lens model itself relies on."
    ),
    source_module="waveoptics.elements",
)
def apply_thin_lens(field: dict, dx: float, wavelength_um: float, focal_length_um: float) -> ToolResult:
    inputs = dict(field=field, dx=dx, wavelength_um=wavelength_um, focal_length_um=focal_length_um)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    lensed = call_adapter(
        "apply_thin_lens", apply_thin_lens_fn, arr, dx, wavelength_um, focal_length_um
    )

    # Paraxial validity: the thin-lens phase exp(-i*k*x^2/(2f)) is the small-angle
    # (sin(theta) ~ tan(theta) ~ theta) approximation to a true spherical wavefront.
    # half_aperture/f is the tangent of the marginal ray angle -- a real, computed
    # measure of how far the field's own extent pushes the approximation, not a
    # blanket assumption that every lens call is paraxial. Threshold 0.3 (~F/1.7,
    # NA~0.3) is an arbitrary-but-stated cutoff, same convention as the 1%
    # thresholds in propagate_angular_spectrum.
    n_x = len(arr)
    half_aperture_um = (n_x * dx) / 2.0
    paraxial_ratio = half_aperture_um / abs(focal_length_um) if focal_length_um != 0 else float("inf")
    in_region = paraxial_ratio <= 0.3
    distance = None if in_region else paraxial_ratio - 0.3
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"grid half-width/focal-length = {paraxial_ratio:.3f} exceeds 0.3 -- the marginal-ray "
            "angle is too large for the thin lens's paraxial quadratic-phase approximation to be "
            "trustworthy (this is a paraxial model, not an exact spherical-wavefront lens)"
        )

    return ToolResult(
        value=field_to_json(lensed),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
