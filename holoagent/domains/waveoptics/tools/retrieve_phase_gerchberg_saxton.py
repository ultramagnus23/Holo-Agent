"""retrieve_phase_gerchberg_saxton — classic two-intensity Gerchberg-Saxton
phase retrieval, with an optional multi-restart mode. See
holoagent/domains/waveoptics/phase_retrieval.py for the algorithm, its
convergence validation, an important honestly-documented limitation
(satisfying the measured constraints does not prove the recovered phase
is unique or "the" true phase), and n_restarts's own real, verified
(though not guaranteed) improvement on that limitation.
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.phase_retrieval import gerchberg_saxton
from holoagent.domains.waveoptics.serialization import field_to_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "object_intensity": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Measured intensity at the object plane",
        },
        "fourier_intensity": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Measured intensity at the Fourier-conjugate (far-field/diffraction) plane, same length as object_intensity",
        },
        "n_iterations": {"type": "integer", "description": "Number of GS iterations"},
        "phase_init": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Optional initial phase guess, radians (default: all zeros). Ignored if n_restarts > 1.",
        },
        "n_restarts": {
            "type": "integer",
            "default": 1,
            "description": (
                "If > 1, run this many independent trials from random initial phases (seeded, "
                "reproducible) and keep the one with the lowest final constraint error -- a real, "
                "verified (not guaranteed) mitigation for GS's non-convex stagnation, at the cost "
                "of n_restarts times the compute. phase_init is ignored when this is > 1."
            ),
        },
        "seed": {"type": "integer", "description": "RNG seed for the random restarts (only used if n_restarts > 1)"},
    },
    "required": ["object_intensity", "fourier_intensity", "n_iterations"],
}


@register(
    name="retrieve_phase_gerchberg_saxton",
    json_schema=SCHEMA,
    description=(
        "Recover a complex field's phase from two measured intensities (object plane and its "
        "Fourier-conjugate plane) via the classic Gerchberg-Saxton iterative algorithm, optionally "
        "with multiple random restarts (n_restarts) to mitigate (not guarantee against) stagnation "
        "at a wrong fixed point. Reports the final Fourier-domain constraint-satisfaction error -- "
        "NOT a guarantee of unique phase recovery, since phase retrieval is a non-convex problem "
        "that can converge to a different fixed point satisfying the same measured intensities."
    ),
    source_module="waveoptics.phase_retrieval",
)
def retrieve_phase_gerchberg_saxton(
    object_intensity: list,
    fourier_intensity: list,
    n_iterations: int,
    phase_init: list | None = None,
    n_restarts: int = 1,
    seed: int | None = None,
) -> ToolResult:
    inputs = dict(
        object_intensity=object_intensity,
        fourier_intensity=fourier_intensity,
        n_iterations=n_iterations,
        phase_init=phase_init,
        n_restarts=n_restarts,
        seed=seed,
    )
    input_hash = compute_input_hash(inputs)

    obj_arr = call_adapter(
        "np.asarray(object_intensity)", lambda v: np.asarray(v, dtype=float), object_intensity
    )
    four_arr = call_adapter(
        "np.asarray(fourier_intensity)", lambda v: np.asarray(v, dtype=float), fourier_intensity
    )
    phase_arr = None
    if phase_init is not None:
        phase_arr = call_adapter("np.asarray(phase_init)", lambda v: np.asarray(v, dtype=float), phase_init)

    result = call_adapter(
        "gerchberg_saxton", gerchberg_saxton, obj_arr, four_arr, n_iterations, phase_arr, n_restarts, seed
    )

    # Real, computed check: does the recovered field actually satisfy the
    # MEASURED constraints (the only thing this tool can verify at
    # inference time -- the true phase is by definition unknown, that's
    # what's being solved for). Threshold 0.01 (1% normalized Fourier-
    # intensity mismatch) is an arbitrary-but-stated convergence cutoff,
    # same convention as this domain's other thresholds. A LOW error here
    # does NOT prove uniqueness -- see this tool's own docstring and
    # phase_retrieval.py's module docstring for a real case where GS
    # converged the constraint error to ~1e-7 while still landing ~8.6%
    # from the true field.
    error = result["fourier_constraint_error"]
    in_region = error < 0.01
    distance = None if in_region else (error - 0.01)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"Fourier-domain constraint error={error:.4g} >= 0.01 after {n_iterations} iterations "
            f"({result['n_restarts_used']} restart(s)) -- the recovered field does not adequately "
            "satisfy the measured Fourier intensity; try more iterations, more restarts, or a "
            "different phase_init"
        )

    return ToolResult(
        value={
            **field_to_json(result["field"]),
            "fourier_constraint_error": error,
            "n_restarts_used": result["n_restarts_used"],
            "best_restart_index": result["best_restart_index"],
        },
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
