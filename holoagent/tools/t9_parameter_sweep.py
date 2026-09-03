"""T9 run_parameter_sweep — Sweep over ONE axis: seed (real batched
execution, for 4 of 7 methods) or a physical medium parameter (a Python
loop, since HoloForge has no batching capability across parameters).

Phase 2 tool (TRD §8). Rescoped from the original spec ("Batched T1-T7
over a parameter grid... a sweep is one batched solver call, not N calls
in a Python loop") after investigating what HoloForge actually supports.
Confirmed by reading holomedia/optimize.py directly: real, verified
batching exists (bit-identical to the unbatched path per HoloForge's own
tests/test_smoke.py), but ONLY over a (target, seed) axis, for 4 of 7
methods (MIL, BSGD, GS, ORC — not LPC, SAT, ORU). Sweeping physical medium
parameters (kappa, D0, sigma, ...) is NOT batched anywhere in HoloForge:
every real sweep script in the repo (experiments/gpu_f2_f3_recovery.py,
experiments/gpu_bpm_wavelength_sweep.py) loops in Python, because those
parameters are compiled into per-instance FFT buffers at construction --
one NPDDRecorder/SlabBPM instance cannot represent multiple parameter
settings simultaneously. This tool does the same: `sweep_field="seed"`
with a batchable method_id gets real batched execution; any other
sweep_field gets an honest Python loop, and the returned `batched` field
says which happened -- never silently presented as batching when it
wasn't.

A full cartesian "arbitrary parameter_grid dict" (the original spec) was
also dropped: sweeping multiple axes at once doesn't map onto either real
capability (batching is single-axis; looping N axes is combinatorial and
belongs to a higher-level orchestration decision, not this tool). Ask for
this tool multiple times, once per axis, if a multi-axis sweep is needed.
"""

from __future__ import annotations

import math

from holoagent.physics import get_default_adapter
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

_BATCHABLE_METHODS = ("MIL", "BSGD", "GS", "ORC")

SCHEMA = {
    "type": "object",
    "properties": {
        "medium_params": {
            "type": "object",
            "description": "Base MediumParams overrides, held fixed across the sweep (except the swept field itself).",
        },
        "n_x": {"type": "integer"},
        "dx": {"type": "number"},
        "target_pattern": {"type": "string", "enum": ["square_wave", "single_slit"]},
        "target_period_px": {"type": "integer", "default": 8},
        "method_id": {"type": "string", "enum": ["GS", "BSGD", "LPC", "MIL", "SAT", "ORC", "ORU"]},
        "sweep_field": {
            "type": "string",
            "description": (
                "'seed' for a real batched (target,seed) sweep (only if method_id is one of "
                f"{_BATCHABLE_METHODS}), or a MediumParams field name (e.g. 'kappa') for a "
                "Python-loop sweep. Not both."
            ),
        },
        "sweep_values": {
            "type": "array",
            "description": "Values to sweep over -- ints for seed, floats for a medium parameter.",
        },
        "fixed_seed": {"type": "integer", "default": 0, "description": "Seed used when sweep_field is not 'seed'"},
        "n_iters": {"type": "integer", "default": 400},
        "lr": {"type": "number", "default": 5e-2},
        "dose_budget": {"type": "number", "default": 1.0},
        "log_every": {"type": "integer", "default": 50},
    },
    "required": ["medium_params", "n_x", "dx", "target_pattern", "method_id", "sweep_field", "sweep_values"],
}


@register(
    name="run_parameter_sweep",
    json_schema=SCHEMA,
    description=(
        "Sweep over seed (real batched execution, methods MIL/BSGD/GS/ORC only) or a medium "
        "parameter (Python loop, HoloForge has no parameter batching). Result says which ran."
    ),
    source_module="optimize (batched fns for target/seed; Python loop for medium params)",
)
def run_parameter_sweep(
    medium_params: dict,
    n_x: int,
    dx: float,
    target_pattern: str,
    method_id: str,
    sweep_field: str,
    sweep_values: list,
    target_period_px: int = 8,
    fixed_seed: int = 0,
    n_iters: int = 400,
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
        sweep_field=sweep_field,
        sweep_values=sweep_values,
        fixed_seed=fixed_seed,
        n_iters=n_iters,
        lr=lr,
        dose_budget=dose_budget,
        log_every=log_every,
    )
    input_hash = compute_input_hash(inputs)

    adapter = get_default_adapter()
    use_batched = sweep_field == "seed" and method_id in _BATCHABLE_METHODS

    if use_batched:
        result = call_adapter(
            f"run_seed_sweep_batched(method_id={method_id!r})",
            adapter.run_seed_sweep_batched,
            method_id=method_id,
            params=medium_params,
            n_x=n_x,
            dx=dx,
            target_pattern=target_pattern,
            target_period_px=target_period_px,
            seeds=sweep_values,
            n_iters=n_iters,
            lr=lr,
            dose_budget=dose_budget,
            log_every=log_every,
        )
        value = {
            "batched": True,
            "sweep_field": "seed",
            "seeds": result["seeds"],
            "psnr_per_seed": result["psnr_per_seed"],
            "diffraction_efficiency_per_seed": result["diffraction_efficiency_per_seed"],
            "loss_history_mean": result["history"],
        }
        finite_check = result["psnr_per_seed"]
    else:
        if sweep_field == "seed":
            diagnostic_note = (
                f"method_id={method_id!r} has no batched implementation "
                f"(only {_BATCHABLE_METHODS} do) -- falling back to a Python loop over seeds"
            )
        else:
            diagnostic_note = (
                f"sweep_field={sweep_field!r} is a medium parameter -- HoloForge has no batching "
                "capability across physical medium parameters, this ran as a Python loop"
            )
        result = call_adapter(
            f"run_medium_param_sweep_looped(sweep_field={sweep_field!r})",
            adapter.run_medium_param_sweep_looped,
            method_id=method_id,
            base_params=medium_params,
            sweep_field=sweep_field,
            sweep_values=sweep_values,
            n_x=n_x,
            dx=dx,
            target_pattern=target_pattern,
            target_period_px=target_period_px,
            fixed_seed=fixed_seed,
            n_iters=n_iters,
            lr=lr,
            dose_budget=dose_budget,
            log_every=log_every,
        )
        value = {"batched": False, "sweep_field": sweep_field, "results": result["results"]}
        finite_check = [r["psnr"] for r in result["results"]]

    diagnostic = None if use_batched else diagnostic_note
    if not all(math.isfinite(x) for x in finite_check):
        non_finite_note = "non-finite psnr in at least one sweep point — degenerate for those parameters"
        diagnostic = f"{diagnostic}; {non_finite_note}" if diagnostic else non_finite_note

    validity = adapter.check_validity_region(dict(medium_params or {}, **({sweep_field: sweep_values[0]} if sweep_field != "seed" else {})))

    return ToolResult(
        value=value,
        units="psnr: dB (scale-invariant); diffraction_efficiency: dimensionless",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
