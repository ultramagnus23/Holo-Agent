"""T1 get_phase_map — Exposure/reconstructed-intensity field at a given
optimizer checkpoint (iteration count), via exact deterministic re-run.

Phase 2 tool (TRD §8). Rescoped from its original spec after investigating
what HoloForge actually supports. Confirmed by reading holomedia/optimize.py
directly: NO optimizer method ever snapshots intermediate field state --
`media_in_the_loop` and every other method overwrite E/dn/recon in-place
each loop pass and only return the final post-loop state. There is no
native checkpoint/resume mechanism at all.

What IS real: HoloForge's seeding is genuinely deterministic
(`_seeded_init_theta` uses a CPU-local `torch.Generator`, explicitly not
CUDA, explicitly for bit-reproducibility — holomedia/optimize.py's own
comment explains why) and every optimizer step is deterministic given that
seed (no dropout/stochastic layers). So calling the SAME method with the
SAME seed and `n_iters=checkpoint` reproduces the exact trajectory up to
that point, bit-for-bit. That's what this tool does — literally the same
underlying call as T2 (`HoloForgeAdapter.run_method_with_field`), just
framed as "the state at iteration k" rather than "the final result of a
run". It is NOT a true checkpoint/snapshot: getting the state at 5
different iteration counts means 5 separate full (re-)optimizations from
scratch, not one run with 5 cheap reads. Expensive, and the tool result
says so via `iterations_run` and a diagnostic when the cost is nontrivial.

No phase is returned — same limitation as T2: this BPM formulation is
real-valued (intensity), not phase-resolved. "get_phase_map" is a legacy
name from the original spec; what's actually returned is the exposure
field (E) and reconstructed intensity (recon) at the requested checkpoint.
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
        "n_x": {"type": "integer"},
        "dx": {"type": "number"},
        "target_pattern": {"type": "string", "enum": ["square_wave", "single_slit"]},
        "target_period_px": {"type": "integer", "default": 8},
        "method_id": {"type": "string", "enum": ["GS", "BSGD", "LPC", "MIL", "SAT", "ORC", "ORU"]},
        "seed": {"type": "integer"},
        "checkpoint": {
            "type": "integer",
            "description": "Iteration count to stop at (n_iters passed through) -- the 'checkpoint'. Re-runs from scratch to this point; not a free lookup.",
        },
        "lr": {"type": "number", "default": 5e-2},
        "dose_budget": {"type": "number", "default": 1.0},
    },
    "required": ["medium_params", "n_x", "dx", "target_pattern", "method_id", "seed", "checkpoint"],
}


@register(
    name="get_phase_map",
    json_schema=SCHEMA,
    description=(
        "Exposure field (E) and reconstructed intensity (recon, no phase) at a given optimizer "
        "iteration checkpoint, via exact deterministic re-run to that point -- not a free snapshot lookup."
    ),
    source_module="optimize (deterministic re-run, no native checkpointing exists)",
)
def get_phase_map(
    medium_params: dict,
    n_x: int,
    dx: float,
    target_pattern: str,
    method_id: str,
    seed: int,
    checkpoint: int,
    target_period_px: int = 8,
    lr: float = 5e-2,
    dose_budget: float = 1.0,
) -> ToolResult:
    inputs = dict(
        medium_params=medium_params,
        n_x=n_x,
        dx=dx,
        target_pattern=target_pattern,
        target_period_px=target_period_px,
        method_id=method_id,
        seed=seed,
        checkpoint=checkpoint,
        lr=lr,
        dose_budget=dose_budget,
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
        f"run_method_with_field(method_id={method_id!r}, checkpoint={checkpoint})",
        adapter.run_method_with_field,
        method_id=method_id,
        target=target,
        recorder=recorder,
        bpm=bpm,
        seed=seed,
        n_iters=checkpoint,
        lr=lr,
        dose_budget=dose_budget,
    )
    validity = adapter.check_validity_region(resolved_params)

    diagnostic = "reconstructed via a full deterministic re-run to this checkpoint, not a stored snapshot lookup"
    if not math.isfinite(result["scalar_efficiency"]):
        diagnostic += "; non-finite result — degenerate for these parameters"

    return ToolResult(
        value={
            "exposure_field": result["E"].tolist(),
            "recon_intensity_field": result["recon"].tolist(),
            "iterations_run": result["iterations_run"],
            "early_stop_reason": result["early_stop_reason"],
            "resolved_medium_params": resolved_params,
            "phase": None,
        },
        units="exposure_field/recon_intensity_field: dimensionless, no phase — see docstring",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
