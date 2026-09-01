"""T6 get_sensitivity — d(output)/d(parameter) w.r.t. a physical medium
parameter.

Phase 1 tool (TRD §8) and (per the TRD) a differentiating tool, on the
premise that it is "nearly free" via autograd because the twin is
differentiable. That premise does NOT hold for physical medium parameters:
NPDDRecorder bakes MediumParams into a non-differentiable buffer (G_hat)
at construction (holomedia/npdd.py), so torch.autograd.grad w.r.t. e.g.
sigma is not available without refactoring NPDDRecorder — out of scope,
since layer 4 is frozen (TRD §1). Autograd IS available in HoloForge, but
only for d(loss)/d(exposure field E) — a different quantity, already used
internally by every optimizer method.

This tool therefore computes sensitivity the way HoloForge's own S2
experiment tier does: one-at-a-time central finite-difference perturbation
(experiments/manifest.py). It is reported as such — units and the returned
"method" field both say "finite_difference_central", never "gradient" or
"autograd" — because presenting a finite-difference estimate as an exact
derivative would be exactly the kind of confident-wrong number this whole
system exists to prevent (PRD §1).

ToolResult.value includes resolved_medium_params -- every MediumParams
field actually used, not just base_medium_params' overrides -- so an
answer naming an unspecified field's HoloForge default has provenance to
point to (see t7_convergence_trace.py's docstring and
docs/set_c_matching_rule.md for why this matters).
"""

from __future__ import annotations

import math

from holoagent.physics import get_default_adapter
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import MissingParameterError, ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "base_medium_params": {
            "type": "object",
            "description": "Overrides on HoloForge's MediumParams fields; omitted fields use HoloForge's own defaults",
        },
        "parameter": {
            "type": "string",
            "description": "Which MediumParams field to perturb, e.g. 'sigma', 'D0', 'kappa'",
        },
        "perturbation_pct": {"type": "number", "default": 0.1, "description": "Fractional perturbation, e.g. 0.1 = +/-10%"},
        "n_x": {"type": "integer"},
        "dx": {"type": "number"},
        "target_pattern": {"type": "string", "enum": ["square_wave", "single_slit"]},
        "target_period_px": {"type": "integer", "default": 8},
        "method_id": {"type": "string", "enum": ["GS", "BSGD", "LPC", "MIL", "ORC", "ORU"]},
        "seed": {"type": "integer"},
        "n_iters": {"type": "integer", "default": 300},
        "lr": {"type": "number", "default": 5e-2},
        "dose_budget": {"type": "number", "default": 1.0},
        "output_quantity": {
            "type": "string",
            "enum": ["psnr", "psnr_maxnorm_legacy", "diffraction_efficiency"],
            "default": "psnr",
        },
    },
    "required": ["base_medium_params", "parameter", "n_x", "dx", "target_pattern", "method_id", "seed"],
}


@register(
    name="get_sensitivity",
    json_schema=SCHEMA,
    description=(
        "d(output)/d(parameter) via one-at-a-time central FINITE-DIFFERENCE perturbation "
        "(not autograd — HoloForge has no differentiable path from physical medium "
        "parameters to this output; see this file's docstring)."
    ),
    source_module="finite_difference (HoloForge S2-style; not autograd)",
)
def get_sensitivity(
    base_medium_params: dict,
    parameter: str,
    n_x: int,
    dx: float,
    target_pattern: str,
    method_id: str,
    seed: int,
    perturbation_pct: float = 0.1,
    target_period_px: int = 8,
    n_iters: int = 300,
    lr: float = 5e-2,
    dose_budget: float = 1.0,
    output_quantity: str = "psnr",
) -> ToolResult:
    inputs = dict(
        base_medium_params=base_medium_params,
        parameter=parameter,
        perturbation_pct=perturbation_pct,
        n_x=n_x,
        dx=dx,
        target_pattern=target_pattern,
        target_period_px=target_period_px,
        method_id=method_id,
        seed=seed,
        n_iters=n_iters,
        lr=lr,
        dose_budget=dose_budget,
        output_quantity=output_quantity,
    )
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    resolved_params = call_adapter(
        "constructing MediumParams", adapter.MediumParams, **(base_medium_params or {})
    ).__dict__
    if parameter not in resolved_params:
        raise MissingParameterError(parameter)

    result = call_adapter(
        f"sensitivity_finite_difference(parameter={parameter!r}, output_quantity={output_quantity!r})",
        adapter.sensitivity_finite_difference,
        base_params=resolved_params,
        parameter=parameter,
        perturbation_pct=perturbation_pct,
        n_x=n_x,
        dx=dx,
        target_pattern=target_pattern,
        target_period_px=target_period_px,
        method_id=method_id,
        seed=seed,
        n_iters=n_iters,
        lr=lr,
        dose_budget=dose_budget,
        output_quantity=output_quantity,
    )
    validity = adapter.check_validity_region(resolved_params)

    diagnostic = None
    non_finite = [
        k for k in ("base_output", "output_plus", "output_minus") if not math.isfinite(result[k])
    ]
    if non_finite:
        diagnostic = f"non-finite output(s) at {non_finite} — derivative estimate is unreliable"

    return ToolResult(
        value={**result, "resolved_medium_params": resolved_params},
        units=f"d({output_quantity})/d({parameter}), finite-difference central estimate (NOT autograd)",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
