"""create_gaussian_beam — a Gaussian beam at its waist (z=0). See
holoagent/domains/waveoptics/fields.py for the definition and the
propagation validation this feeds into.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.fields import gaussian_beam
from holoagent.domains.waveoptics.serialization import field_to_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "n_x": {"type": "integer"},
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "wavelength_um": {"type": "number"},
        "waist_um": {"type": "number", "description": "1/e amplitude half-width at the waist"},
        "amplitude": {"type": "number", "default": 1.0},
    },
    "required": ["n_x", "dx", "wavelength_um", "waist_um"],
}


@register(
    name="create_gaussian_beam",
    json_schema=SCHEMA,
    description="A Gaussian beam at its waist (z=0, planar phase front).",
    source_module="waveoptics.fields",
)
def create_gaussian_beam(n_x: int, dx: float, wavelength_um: float, waist_um: float, amplitude: float = 1.0) -> ToolResult:
    inputs = dict(n_x=n_x, dx=dx, wavelength_um=wavelength_um, waist_um=waist_um, amplitude=amplitude)
    input_hash = compute_input_hash(inputs)

    field = call_adapter("gaussian_beam", gaussian_beam, n_x, dx, wavelength_um, waist_um, amplitude)

    # Two real, computed sampling checks: the grid must be wide enough not
    # to truncate the beam's tails, and fine enough to resolve its
    # curvature. Both are genuine numerical-validity concerns, not
    # physical ones -- flagged the same way (master-build S19: never hide
    # validity information), since a truncated or under-resolved beam is
    # not the beam the caller asked for.
    grid_half_width_um = (n_x * dx) / 2.0
    width_ratio = grid_half_width_um / waist_um if waist_um > 0 else float("inf")
    resolution_ratio = waist_um / dx if dx > 0 else float("inf")

    width_ok = width_ratio >= 3.0
    resolution_ok = resolution_ratio >= 5.0
    in_region = width_ok and resolution_ok

    distance = None
    diagnostics = []
    if not width_ok:
        distance_width = (3.0 - width_ratio) / 3.0
        diagnostics.append(
            f"grid half-width is only {width_ratio:.2f}x the waist (want >=3x) -- beam tails are truncated"
        )
    else:
        distance_width = 0.0
    if not resolution_ok:
        distance_res = (5.0 - resolution_ratio) / 5.0
        diagnostics.append(
            f"grid spacing is only {resolution_ratio:.2f} samples per waist (want >=5) -- beam curvature under-resolved"
        )
    else:
        distance_res = 0.0
    if not in_region:
        distance = max(distance_width, distance_res)

    return ToolResult(
        value=field_to_json(field),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic="; ".join(diagnostics) if diagnostics else None,
    )
