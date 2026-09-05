"""create_plane_wave — a tilted (or normal-incidence) plane wave field on
a 1D transverse grid. See holoagent/domains/waveoptics/fields.py for the
definition.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.fields import plane_wave
from holoagent.domains.waveoptics.serialization import field_to_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "n_x": {"type": "integer", "description": "Grid points"},
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "wavelength_um": {"type": "number"},
        "angle_deg": {"type": "number", "default": 0.0, "description": "Incidence angle from normal, degrees"},
        "amplitude": {"type": "number", "default": 1.0},
    },
    "required": ["n_x", "dx", "wavelength_um"],
}


@register(
    name="create_plane_wave",
    json_schema=SCHEMA,
    description="A tilted or normal-incidence plane wave on a 1D transverse grid.",
    source_module="waveoptics.fields",
)
def create_plane_wave(n_x: int, dx: float, wavelength_um: float, angle_deg: float = 0.0, amplitude: float = 1.0) -> ToolResult:
    inputs = dict(n_x=n_x, dx=dx, wavelength_um=wavelength_um, angle_deg=angle_deg, amplitude=amplitude)
    input_hash = compute_input_hash(inputs)

    field = call_adapter("plane_wave", plane_wave, n_x, dx, wavelength_um, angle_deg, amplitude)

    # Nyquist check on the phase ramp's spatial frequency -- a real,
    # computed validity bound (master-build S19), not a fixed True.
    import math

    ramp_freq_cycles_per_um = abs(math.sin(math.radians(angle_deg))) / wavelength_um
    nyquist_freq = 1.0 / (2.0 * dx)
    in_region = ramp_freq_cycles_per_um <= nyquist_freq
    distance = None if in_region else (ramp_freq_cycles_per_um - nyquist_freq) / nyquist_freq

    diagnostic = None
    if not in_region:
        diagnostic = (
            f"angle_deg={angle_deg} at wavelength_um={wavelength_um} needs a spatial frequency "
            f"({ramp_freq_cycles_per_um:.4g} cycles/um) above the grid's Nyquist limit "
            f"({nyquist_freq:.4g} cycles/um) -- aliased, not a physically meaningful tilted wave"
        )

    return ToolResult(
        value=field_to_json(field),
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
