"""T7 get_convergence_trace — Full optimizer trajectory, not just the endpoint.

Phase 1 tool (TRD §8). Wired to HoloForge via
holoagent.domains.holoforge.adapter through
experiments/methods.py::run_method — the one stable, tested, uniform
dispatch boundary in HoloForge (part2/code/tests/test_method_registry.py).

Schema deviates from the original TRD sketch (which took only a `run_id`):
HoloForge does not persist runs internally, so there is nothing to "look
up" by ID. Every call here executes a fresh simulation; run_id in the
returned ToolResult is assigned by HoloAgent's own provenance store, not
recovered from HoloForge.

loss_history is logged every `log_every` iterations, not every iteration
(HoloForge's own limitation — see holomedia/optimize.py). GS and LPC are
closed-form: they return iterations_run=0 and an empty history, which is
correct, not a failure.

ToolResult.value includes resolved_medium_params — every MediumParams
field actually used, not just the caller's overrides. Without this, an
answer that names a value HoloForge defaulted silently (e.g. "using the
default sigma of 0.08") has no provenance record to point to and gets
wrongly flagged by Set C (docs/set_c_matching_rule.md) as ungrounded, even
though it genuinely drove the computation.
"""

from __future__ import annotations

import math

from holoagent.domains.holoforge.physics import get_default_adapter
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "medium_params": {
            "type": "object",
            "description": "Overrides on HoloForge's MediumParams fields (D0, sigma, kappa, gamma, dn_max, k_bleach, alpha_D, shrinkage, thickness, n0). Omitted fields use HoloForge's own defaults.",
        },
        "n_x": {"type": "integer", "description": "Grid points in x"},
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "target_pattern": {"type": "string", "enum": ["square_wave", "single_slit"]},
        "target_period_px": {"type": "integer", "default": 8},
        "method_id": {"type": "string", "enum": ["GS", "BSGD", "LPC", "MIL", "ORC", "ORU"]},
        "seed": {"type": "integer"},
        "n_iters": {"type": "integer", "default": 800},
        "lr": {"type": "number", "default": 5e-2},
        "dose_budget": {"type": "number", "default": 1.0},
        "log_every": {"type": "integer", "default": 50},
    },
    "required": ["medium_params", "n_x", "dx", "target_pattern", "method_id", "seed"],
}


@register(
    name="get_convergence_trace",
    json_schema=SCHEMA,
    description="Full optimizer trajectory (loss per logged iteration), not just the endpoint. Runs a fresh simulation.",
    source_module="optimizer",
)
def get_convergence_trace(
    medium_params: dict,
    n_x: int,
    dx: float,
    target_pattern: str,
    method_id: str,
    seed: int,
    target_period_px: int = 8,
    n_iters: int = 800,
    lr: float = 5e-2,
    dose_budget: float = 1.0,
    log_every: int = 50,
) -> ToolResult:
    inputs = dict(
        medium_params=medium_params,
        n_x=n_x,
        dx=dx,
        target_pattern=target_pattern,
        target_period_px=target_period_px,
        method_id=method_id,
        seed=seed,
        n_iters=n_iters,
        lr=lr,
        dose_budget=dose_budget,
        log_every=log_every,
    )
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    resolved_params = call_adapter(
        "constructing MediumParams", adapter.MediumParams, **(medium_params or {})
    ).__dict__
    recorder, bpm = call_adapter(
        "build_recorder_and_bpm", adapter.build_recorder_and_bpm, resolved_params, n_x, dx
    )
    target = call_adapter(
        f"build_target(pattern={target_pattern!r})",
        adapter.build_target,
        n_x,
        target_pattern,
        target_period_px,
    )
    result = call_adapter(
        f"run_method(method_id={method_id!r})",
        adapter.run_method,
        method_id=method_id,
        target=target,
        recorder=recorder,
        bpm=bpm,
        seed=seed,
        n_iters=n_iters,
        lr=lr,
        dose_budget=dose_budget,
        log_every=log_every,
    )
    validity = adapter.check_validity_region(resolved_params)

    diagnostic = None
    if not math.isfinite(result["psnr"]):
        diagnostic = f"non-finite psnr ({result['psnr']!r}) — reconstruction likely degenerate for these parameters"

    return ToolResult(
        value={
            "loss_history": result["loss_history"],
            "iterations_run": result["iterations_run"],
            "early_stop_reason": result["early_stop_reason"],
            "final_psnr": result["psnr"],
            "resolved_medium_params": resolved_params,
        },
        units="loss: scale-invariant MSE (si_mse); psnr: dB",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
