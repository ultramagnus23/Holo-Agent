"""T4 get_medium_state — Saturation, exposure history, NPDD state at a
given point during recording.

Phase 2 tool (TRD §8). Wired to HoloForge's real capability, confirmed by
reading holomedia/npdd.py directly: NPDDRecorder.forward(...,
return_history=True) returns (u, N) at every timestep of the recording
simulation in one pass. dn at the requested timestep is derived by
applying NPDDRecorder's own final-step formula (dn_max * tanh(1.5*N)) to
that step's N — the identical formula, evaluated earlier, not new physics.

HONEST GAP, not silently omitted: dye/exposure-depletion state (d) is
tracked internally by NPDDRecorder but NOT included in return_history's
captured tuples (only u, N are). Reconstructing d from the outside would
mean re-deriving its bleaching recurrence (d *= exp(-k_bleach*I*dt) each
step) independently of the frozen class — that crosses from "calling
HoloForge differently" into "writing new physics-adjacent code," out of
scope for a frozen layer 4 (TRD §1). d is therefore not returned here;
callers asking about dye depletion specifically get a tool that can't
answer that part, not a silently wrong answer for it.
"""

from __future__ import annotations

import math

from holoagent.physics import get_default_adapter
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "medium_params": {
            "type": "object",
            "description": "Overrides on HoloForge's MediumParams fields. Omitted fields use HoloForge's own defaults.",
        },
        "n_x": {"type": "integer", "description": "Grid points in x"},
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "K_rad_per_um": {"type": "number", "description": "Spatial frequency of the grating being recorded"},
        "contrast": {"type": "number", "default": 0.9, "description": "Exposure fringe visibility, 0-1"},
        "t_total": {"type": "number", "default": 10.0, "description": "Total exposure duration, seconds"},
        "n_steps": {"type": "integer", "default": 300, "description": "Number of IMEX timesteps"},
        "timestep": {"type": "integer", "description": "0-indexed timestep to report state at (must be < n_steps)"},
    },
    "required": ["medium_params", "n_x", "dx", "K_rad_per_um", "timestep"],
}


@register(
    name="get_medium_state",
    json_schema=SCHEMA,
    description=(
        "Monomer (u) and polymer (N) concentration state and derived index modulation (dn) "
        "at a given timestep during recording. Dye depletion state (d) is not exposed -- see docstring."
    ),
    source_module="npdd (return_history=True)",
)
def get_medium_state(
    medium_params: dict,
    n_x: int,
    dx: float,
    K_rad_per_um: float,
    timestep: int,
    contrast: float = 0.9,
    t_total: float = 10.0,
    n_steps: int = 300,
) -> ToolResult:
    inputs = dict(
        medium_params=medium_params,
        n_x=n_x,
        dx=dx,
        K_rad_per_um=K_rad_per_um,
        contrast=contrast,
        t_total=t_total,
        n_steps=n_steps,
        timestep=timestep,
    )
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    resolved_params = call_adapter(
        "constructing MediumParams", adapter.MediumParams, **(medium_params or {})
    ).__dict__
    result = call_adapter(
        f"get_medium_state_at_timestep(timestep={timestep})",
        adapter.get_medium_state_at_timestep,
        params=resolved_params,
        n_x=n_x,
        dx=dx,
        K_rad_per_um=K_rad_per_um,
        contrast=contrast,
        t_total=t_total,
        n_steps=n_steps,
        timestep=timestep,
    )
    validity = adapter.check_validity_region(resolved_params, K_rad_per_um=K_rad_per_um)

    diagnostic = None
    if not math.isfinite(result["dn1_amplitude"]):
        diagnostic = f"non-finite dn1_amplitude ({result['dn1_amplitude']!r}) — degenerate for these parameters"

    return ToolResult(
        value={
            "u_field": result["u"].tolist(),
            "N_field": result["N"].tolist(),
            "dn_field": result["dn"].tolist(),
            "dn1_amplitude": result["dn1_amplitude"],
            "mean_u": result["mean_u"],
            "mean_N": result["mean_N"],
            "dn_max": result["dn_max"],
            "fraction_of_dn_max": result["fraction_of_dn_max"],
            "real_time_s": result["real_time_s"],
            "dye_state_d": None,
            "resolved_medium_params": resolved_params,
        },
        units="u/N: dimensionless concentration; dn/dn1_amplitude: index modulation (dimensionless); real_time_s: seconds",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
