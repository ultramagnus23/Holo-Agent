"""T2 get_diffraction_efficiency — Reconstructed intensity field + scalar
diffraction efficiency, from a fresh simulation.

Phase 2 tool (TRD §8). Rescoped from its original spec ("efficiency
across the field") after investigating what HoloForge actually computes:
confirmed by reading holomedia/optimize.py and holomedia/diffraction.py
directly — HoloForge has no notion of "efficiency at each spatial point."
The only "diffraction efficiency" that exists anywhere in the codebase is
a single scalar (a masked-sum ratio, holomedia.diffraction_efficiency).
What IS genuinely spatially resolved is the reconstructed intensity field
(SlabBPM.forward's output, real-valued |E|^2) — computed internally by
every method but discarded by experiments/methods.py::run_method before
it returns. T2 exposes that discarded field (via
holoagent.domains.holoforge.adapter.HoloForgeAdapter.run_method_with_field,
which calls the exact same underlying per-method functions run_method
does — not a reimplementation), alongside the same scalar efficiency T7
already reports, so this tool adds real information rather than
duplicating T7.

No phase information is returned — this is intensity (real-valued,
|E|^2), not a complex/phase-resolved field. HoloForge's BPM formulation
does not expose or track phase separately; claiming otherwise would be
claiming a computation that was never performed.

KNOWN SCALING GAP: the field arrays go straight into ToolResult.value,
which the provenance store persists inline as JSON. TRD §4 explicitly
says large fields belong on disk, not in the database ("do not put
megabyte phase maps in the database... on-disk arrays for fields"). Fine
for the small grids (n_x <= ~128) this tool is tested against; a large
n_x run would need on-disk array storage with a path reference in
ToolResult.value instead of inlining the array, which isn't built yet.
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
            "description": "Overrides on HoloForge's MediumParams fields. Omitted fields use HoloForge's own defaults.",
        },
        "n_x": {"type": "integer", "description": "Grid points in x"},
        "dx": {"type": "number", "description": "Grid spacing, microns"},
        "target_pattern": {"type": "string", "enum": ["square_wave", "single_slit"]},
        "target_period_px": {"type": "integer", "default": 8},
        "method_id": {"type": "string", "enum": ["GS", "BSGD", "LPC", "MIL", "SAT", "ORC", "ORU"]},
        "seed": {"type": "integer"},
        "n_iters": {"type": "integer", "default": 800},
        "lr": {"type": "number", "default": 5e-2},
        "dose_budget": {"type": "number", "default": 1.0},
        "log_every": {"type": "integer", "default": 50},
    },
    "required": ["medium_params", "n_x", "dx", "target_pattern", "method_id", "seed"],
}


@register(
    name="get_diffraction_efficiency",
    json_schema=SCHEMA,
    description=(
        "Reconstructed intensity field (spatially resolved, real-valued -- no phase) plus the "
        "scalar diffraction efficiency, from a fresh simulation. Runs a fresh simulation."
    ),
    source_module="bpm (spatial field discarded by run_method, exposed here)",
)
def get_diffraction_efficiency(
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
        f"run_method_with_field(method_id={method_id!r})",
        adapter.run_method_with_field,
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
    if not math.isfinite(result["scalar_efficiency"]):
        diagnostic = (
            f"non-finite scalar_efficiency ({result['scalar_efficiency']!r}) — "
            "reconstruction likely degenerate for these parameters"
        )

    return ToolResult(
        value={
            "recon_intensity_field": result["recon"].tolist(),
            "exposure_field": result["E"].tolist(),
            "scalar_efficiency": result["scalar_efficiency"],
            "iterations_run": result["iterations_run"],
            "early_stop_reason": result["early_stop_reason"],
            "resolved_medium_params": resolved_params,
        },
        units="recon_intensity_field: dimensionless intensity |E|^2, no phase; scalar_efficiency: dimensionless masked-sum ratio",
        in_validated_region=validity["in_validated_region"],
        distance_from_bounds=validity["distance_from_bounds"],
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
