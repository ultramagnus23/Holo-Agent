"""T8 check_validity_region — Inside/outside validated bounds + distance.

Phase 1 tool (TRD §8) and a differentiating tool (TRD §3.1). IMPORTANT: no
equivalent function or bounds-checking exists anywhere in HoloForge itself
— confirmed by direct inspection of holomedia/*.py, experiments/*.py, and
configs/media/*.yaml. What this tool checks against is hand-transcribed
from two places in the HoloForge repo, neither of which is code the
pipeline reads or enforces:

  1. MediumParams's docstring ("Literature ranges", holomedia/npdd.py) —
     per-parameter physical plausibility, not an enforced constraint.
  2. docs/validation_standard.md / docs/parameter_provenance.md — the
     K = 1.96-15.71 rad/um range the twin was actually fit and tested
     against, written down by a human after the fact, not computed.

See holoagent/physics/holoforge_adapter.py for the transcribed bound
tables themselves (MEDIUM_PARAM_LITERATURE_BOUNDS,
VALIDATED_K_RANGE_RAD_PER_UM). If those source documents change, this
tool goes stale silently unless someone re-checks it by hand — there is
no automated link back to the HoloForge repo for this specific tool,
unlike T3/T7 which call live code.
"""

from __future__ import annotations

from holoagent.physics import get_default_adapter
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "medium_params": {
            "type": "object",
            "description": "Full or partial MediumParams field values to check (D0, sigma, kappa, gamma, dn_max, k_bleach, alpha_D, shrinkage, thickness). Fields not present in the bound table are ignored, not assumed valid.",
        },
        "K_rad_per_um": {
            "type": "number",
            "description": "Optional spatial frequency to check against the twin's actually-validated K range",
        },
    },
    "required": ["medium_params"],
}


@register(
    name="check_validity_region",
    json_schema=SCHEMA,
    description="Whether a parameter set (and optionally a spatial frequency) sits inside hand-transcribed HoloForge literature/validated bounds, and how far outside if not.",
    source_module="validity_model",
)
def check_validity_region(medium_params: dict, K_rad_per_um: float | None = None) -> ToolResult:
    inputs = dict(medium_params=medium_params, K_rad_per_um=K_rad_per_um)
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    result = call_adapter(
        "check_validity_region", adapter.check_validity_region, medium_params, K_rad_per_um=K_rad_per_um
    )

    violations = {name: {"value": value, "bounds": bounds} for name, (value, bounds) in result["violations"].items()}

    diagnostic = None
    if not result["in_validated_region"]:
        diagnostic = f"{len(violations)} field(s) outside hand-transcribed bounds: {sorted(violations)}"

    return ToolResult(
        value={"violations": violations},
        units="bounds are hand-transcribed from HoloForge docs, not a HoloForge-native validity model — see this file's docstring",
        in_validated_region=result["in_validated_region"],
        distance_from_bounds=result["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
