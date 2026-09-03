"""T5 reconstruct_wavefront — Diffraction efficiency at a specified
reconstruction/viewing angle (Kogelnik analytic tier).

Phase 2 tool (TRD §8). SCOPED, not fully general — confirmed by reading
holomedia/diffraction.py directly: the only place in HoloForge with a
real, tested, wired-in reconstruction-angle parameter is the analytic
Kogelnik path (`kogelnik_de(dtheta=...)`, exercised by
experiments/fit_literature_curves.py's simulate_angular_de). The actual
BPM reconstruction engine (SlabBPM, what run_method/T2/T7 use) has NO
angle parameter at all — always normal incidence. SlabBPM does accept an
`incident=` field that could in principle carry a tilted plane wave, but
no call site anywhere in HoloForge ever does this; wrapping it here would
mean exercising a real but never-validated code path, not a verified
capability. Never-do #10 (never claim a computation was performed if it
wasn't): this tool answers "what's the efficiency at this angular
detuning, for a uniform sinusoidal grating of index modulation dn" — it
does NOT answer "reconstruct the actual off-axis BPM field for an
arbitrary recorded profile."
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
        "dn": {"type": "number", "description": "Recorded index modulation amplitude (dimensionless), e.g. from get_medium_state's dn1_amplitude"},
        "thickness_um": {"type": "number"},
        "wavelength_um": {"type": "number", "default": 0.405},
        "theta_B_deg": {"type": "number", "default": 10.0, "description": "Bragg-matched angle, degrees (HoloForge's own default)"},
        "dtheta_deg": {"type": "number", "description": "Angular detuning from theta_B_deg, degrees -- 0 means at Bragg (nominal reconstruction angle)"},
    },
    "required": ["dn", "thickness_um", "dtheta_deg"],
}


@register(
    name="reconstruct_wavefront",
    json_schema=SCHEMA,
    description=(
        "Diffraction efficiency at a reconstruction angle detuned from Bragg-match, via the analytic "
        "Kogelnik tier ONLY -- not the full BPM reconstruction pipeline, which has no angle parameter."
    ),
    source_module="kogelnik (analytic tier only, not the BPM engine T2/T7 use)",
)
def reconstruct_wavefront(
    dn: float,
    thickness_um: float,
    dtheta_deg: float,
    wavelength_um: float = 0.405,
    theta_B_deg: float = 10.0,
) -> ToolResult:
    inputs = dict(dn=dn, thickness_um=thickness_um, wavelength_um=wavelength_um, theta_B_deg=theta_B_deg, dtheta_deg=dtheta_deg)
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    efficiency = call_adapter(
        "reconstruct_at_angle",
        adapter.reconstruct_at_angle,
        dn=dn,
        thickness_um=thickness_um,
        wavelength_um=wavelength_um,
        theta_B_deg=theta_B_deg,
        dtheta_deg=dtheta_deg,
    )
    # dn/thickness are checked against the same hand-transcribed literature
    # bounds T8 uses (dn_max, thickness) -- the closest real check
    # available, since this tool doesn't take a full MediumParams.
    validity = adapter.check_validity_region({"dn_max": dn, "thickness": thickness_um})

    diagnostic = None
    if not math.isfinite(efficiency):
        diagnostic = f"non-finite efficiency ({efficiency!r}) — check dn/thickness/wavelength magnitudes"

    return ToolResult(
        value={"efficiency": efficiency, "dtheta_deg": dtheta_deg, "theta_B_deg": theta_B_deg},
        units="efficiency: dimensionless (0-1); angles: degrees",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
