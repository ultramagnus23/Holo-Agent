"""retrieve_phase_hio — Fienup Hybrid Input-Output single-pattern
coherent-diffractive-imaging phase retrieval. See
holoagent/domains/waveoptics/phase_retrieval.py for the algorithm, why
it's a genuinely different problem from
retrieve_phase_gerchberg_saxton's two-intensity setup (one measured
intensity + a known support constraint, instead of two measured
intensities), and the found-live distinction between real-non-negative
objects (well-posed, recovers reliably) and general complex objects
(markedly worse-conditioned, even with near-perfect constraint
satisfaction).
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.phase_retrieval import fienup_hio
from holoagent.domains.waveoptics.serialization import field_to_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "fourier_intensity": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Measured far-field/diffraction-pattern intensity",
        },
        "support_mask": {
            "type": "array",
            "items": {"type": "boolean"},
            "description": "Known object support: true where the object may be nonzero, false where it's known to be zero. Same length as fourier_intensity.",
        },
        "n_iterations_hio": {"type": "integer", "description": "Number of Hybrid Input-Output iterations"},
        "n_iterations_er": {
            "type": "integer",
            "default": 0,
            "description": "Optional number of Error-Reduction polish iterations run after HIO (standard practice: HIO for global search, ER to clean up)",
        },
        "beta": {"type": "number", "default": 0.9, "description": "HIO feedback parameter, in (0, 1]"},
        "nonnegative": {
            "type": "boolean",
            "default": True,
            "description": "Also enforce the object is real and non-negative -- the classic well-posed Fienup case. Setting false allows a general complex object but is found to recover markedly less reliably.",
        },
        "seed": {"type": "integer", "description": "RNG seed for the random initial guess within the support"},
    },
    "required": ["fourier_intensity", "support_mask", "n_iterations_hio"],
}


@register(
    name="retrieve_phase_hio",
    json_schema=SCHEMA,
    description=(
        "Recover a complex field from a SINGLE measured far-field intensity plus a known support "
        "constraint, via Fienup's Hybrid Input-Output algorithm (optionally polished with "
        "Error-Reduction). A genuinely different problem from retrieve_phase_gerchberg_saxton "
        "(which needs two measured intensities instead of a support constraint). Reports the final "
        "Fourier-domain constraint-satisfaction error -- not a guarantee of unique recovery, and "
        "recovery is markedly less reliable for a general complex-valued object than for a real, "
        "non-negative one (nonnegative=True, the default)."
    ),
    source_module="waveoptics.phase_retrieval",
)
def retrieve_phase_hio(
    fourier_intensity: list,
    support_mask: list,
    n_iterations_hio: int,
    n_iterations_er: int = 0,
    beta: float = 0.9,
    nonnegative: bool = True,
    seed: int | None = None,
) -> ToolResult:
    inputs = dict(
        fourier_intensity=fourier_intensity,
        support_mask=support_mask,
        n_iterations_hio=n_iterations_hio,
        n_iterations_er=n_iterations_er,
        beta=beta,
        nonnegative=nonnegative,
        seed=seed,
    )
    input_hash = compute_input_hash(inputs)

    four_arr = call_adapter(
        "np.asarray(fourier_intensity)", lambda v: np.asarray(v, dtype=float), fourier_intensity
    )
    support_arr = call_adapter(
        "np.asarray(support_mask)", lambda v: np.asarray(v, dtype=bool), support_mask
    )

    result = call_adapter(
        "fienup_hio",
        fienup_hio,
        four_arr,
        support_arr,
        n_iterations_hio,
        n_iterations_er,
        beta,
        nonnegative,
        seed,
    )

    # Same real, computed check as retrieve_phase_gerchberg_saxton: does
    # the recovered field satisfy the ONE thing checkable at inference
    # time (the measured Fourier intensity). A low error here is markedly
    # less trustworthy as a proxy for "correct recovery" when
    # nonnegative=False -- see this tool's and phase_retrieval.py's own
    # docstrings for the found-live ~17-25% field error even at near-zero
    # constraint error in that regime.
    error = result["fourier_constraint_error"]
    in_region = error < 0.01
    distance = None if in_region else (error - 0.01)
    diagnostic = None
    if not in_region:
        diagnostic = (
            f"Fourier-domain constraint error={error:.4g} >= 0.01 after {n_iterations_hio} HIO + "
            f"{n_iterations_er} ER iterations -- the recovered field does not adequately satisfy the "
            "measured intensity; try more iterations, a different beta, or (if applicable) "
            "nonnegative=True for the better-conditioned real-object case"
        )

    return ToolResult(
        value={**field_to_json(result["field"]), "fourier_constraint_error": error},
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
