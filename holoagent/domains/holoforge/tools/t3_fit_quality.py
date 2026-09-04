"""T3 get_fit_quality — NRMSE per region + parameter set (twin validation).

Phase 1 tool (TRD §8). Wired to HoloForge via
holoagent.domains.holoforge.adapter through
experiments/fit_literature_curves.py::fit_curve.

Requires digitized reference (x, y) data — this tool does not fabricate
or look up literature curves itself; the caller supplies xs/ys (e.g. from
a specific literature-digitization CSV). Quality bucketing (good/moderate/
poor at nrmse < 0.3 / 0.3-0.5 / >= 0.5) mirrors fit_literature_curves.py's
own main(), which applies it ad hoc rather than as part of fit_curve's
return value — replicated here so it's part of the typed result instead
of left to the orchestrator to threshold itself.

ToolResult.value includes resolved_base_medium_params -- every
MediumParams field actually used as the fit's fixed starting point, not
just base_medium_params' overrides (see t7_convergence_trace.py's
docstring and docs/set_c_matching_rule.md for why this matters). Note
thickness_um is a separate argument to fit_curve, not read from
MediumParams.thickness -- resolved_base_medium_params.thickness reflects
the dataclass default/override, which may differ from the thickness_um
actually used in the simulation; both are in the provenance pool, and an
answer should cite thickness_um, not resolved_base_medium_params.thickness,
for what the fit actually used.
"""

from __future__ import annotations

from holoagent.domains.holoforge.physics import get_default_adapter
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "curve_type": {"type": "string", "enum": ["growth", "growth_dn", "angular"]},
        "K_rad_per_um": {"type": "number", "description": "Spatial frequency of the recorded grating, rad/um"},
        "xs": {"type": "array", "items": {"type": "number"}},
        "ys": {"type": "array", "items": {"type": "number"}, "description": "Digitized reference y-values"},
        "base_medium_params": {
            "type": "object",
            "description": "Overrides on HoloForge's MediumParams fields used as the fit's fixed/starting parameter set",
        },
        "thickness_um": {"type": "number", "default": 30.0},
        "n_starts": {"type": "integer", "default": 10},
        "seed": {"type": "integer", "default": 0},
        "second_param": {
            "type": "string",
            "description": "Which parameter besides kappa to fit; defaults to HoloForge's per-curve-type choice if omitted",
        },
    },
    "required": ["curve_type", "K_rad_per_um", "xs", "ys"],
}


@register(
    name="get_fit_quality",
    json_schema=SCHEMA,
    description="NRMSE of a bounded least-squares fit against a digitized reference curve, plus the fitted parameter set.",
    source_module="twin_validation",
)
def get_fit_quality(
    curve_type: str,
    K_rad_per_um: float,
    xs: list,
    ys: list,
    base_medium_params: dict | None = None,
    thickness_um: float = 30.0,
    n_starts: int = 10,
    seed: int = 0,
    second_param: str | None = None,
) -> ToolResult:
    inputs = dict(
        curve_type=curve_type,
        K_rad_per_um=K_rad_per_um,
        xs=xs,
        ys=ys,
        base_medium_params=base_medium_params,
        thickness_um=thickness_um,
        n_starts=n_starts,
        seed=seed,
        second_param=second_param,
    )
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    base_params_obj = call_adapter(
        "constructing MediumParams", adapter.MediumParams, **(base_medium_params or {})
    )
    result = call_adapter(
        f"fit_curve(curve_type={curve_type!r})",
        adapter.fit_curve,
        curve_type=curve_type,
        K=K_rad_per_um,
        xs=xs,
        ys=ys,
        base_params=base_params_obj,
        thickness_um=thickness_um,
        n_starts=n_starts,
        seed=seed,
        second_param=second_param,
    )

    nrmse = result["nrmse"]
    quality = "good" if nrmse < 0.3 else "moderate" if nrmse < 0.5 else "poor"

    fitted_params = {"kappa": result["kappa_fit"], result["second_param"]: result["second_param_fit"]}
    validity = adapter.check_validity_region(fitted_params, K_rad_per_um=K_rad_per_um)

    diagnostic = None
    if not result["converged"]:
        diagnostic = "least_squares did not converge for the best-of-n_starts attempt (result.success == False)"
    elif result["at_bound"]:
        diagnostic = "best fit sits at a HoloForge PARAM_BOUNDS edge — kappa_fit/second_param_fit likely unreliable"

    return ToolResult(
        value={
            "nrmse": nrmse,
            "quality": quality,
            "kappa_fit": result["kappa_fit"],
            "second_param": result["second_param"],
            "second_param_fit": result["second_param_fit"],
            "nrmse_spread": {"min": result["nrmse_min"], "max": result["nrmse_max"], "std": result["nrmse_std"]},
            "n_points": result["n_points"],
            "resolved_base_medium_params": base_params_obj.__dict__,
        },
        units="nrmse: dimensionless (rmse / (y.max() - y.min()))",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
