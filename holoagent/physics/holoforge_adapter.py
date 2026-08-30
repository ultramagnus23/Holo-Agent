"""Thin adapter over the HoloForge repository (TRD §1, layer 4 — FROZEN).

This module never edits HoloForge. It imports and calls into it. Any bug
found in the physics goes back to the HoloForge repo through its own
process, not a patch here (TRD §1).

The tool modules in holoagent.tools call through this adapter rather than
importing HoloForge directly, so the frozen-dependency boundary has exactly
one crossing point.

IMPORTANT — deviation from the TRD's assumption about T6 (get_sensitivity):
the TRD describes sensitivity as "d(output)/d(parameter) -- free from the
differentiable twin (autograd)". That is true for the exposure field E(x)
(the optimizers already do E.requires_grad_(True) + .backward()), but it
is NOT true for the physical medium parameters (D0, sigma, kappa, ...):
NPDDRecorder bakes MediumParams into a non-differentiable buffer (G_hat)
at construction time (holomedia/npdd.py), so torch.autograd.grad w.r.t.
e.g. sigma is not available without refactoring NPDDRecorder itself, which
is out of scope (layer 4 is frozen). HoloForge's own codebase computes
parameter sensitivity via one-at-a-time finite-difference perturbation
(experiments/manifest.py, tier S2) -- this adapter exposes that, and the
tool that wraps it (T6) reports it honestly as finite-difference, not
autograd.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

DEFAULT_HOLOFORGE_CODE_PATH = Path(
    os.environ.get(
        "HOLOFORGE_CODE_PATH",
        r"C:\Users\chait\Downloads\HoloForge\HoloForge\part2\code",
    )
)

# Hand-transcribed from HoloForge source, since no equivalent exists as
# code/config HoloForge itself reads (see docs/tool_coverage_matrix.md and
# the T8 investigation note in docs/TRD.md). Two independent bound sets:
#
# 1. Per-parameter literature ranges, transcribed from the MediumParams
#    docstring (holomedia/npdd.py, "Literature ranges (see paper Table 1)").
#    These describe what's physically plausible per parameter in isolation.
MEDIUM_PARAM_LITERATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "D0": (1e-3, 1e0),
    "sigma": (0.02, 0.3),
    "kappa": (0.1, 10.0),
    "gamma": (0.5, 1.0),
    "dn_max": (1e-3, 6e-3),
    "k_bleach": (0.0, 1.0),
    "alpha_D": (0.0, 3.0),
    "shrinkage": (0.0, 0.03),
    "thickness": (5.0, 100.0),
}

# 2. The spatial-frequency range the twin was actually validated against
#    (fit to literature curves + tested V1 grid), transcribed from
#    docs/validation_standard.md:24 and docs/parameter_provenance.md:65-66
#    in the HoloForge repo. This is the one the PRD's "validated region"
#    language is really about -- it's a claim about where the fit was
#    checked, not just where parameters are physically plausible.
VALIDATED_K_RANGE_RAD_PER_UM: tuple[float, float] = (1.96, 15.71)


class HoloForgeAdapter:
    """Holds a live import of a HoloForge Part 2 checkout and exposes the
    entry points the tool layer calls through.

    Mirrors the sys.path convention HoloForge's own tests/experiments use
    (see part2/code/tests/test_method_registry.py,
    part2/code/experiments/run_manifest.py): add code/ (for `holomedia`)
    and code/experiments/ (for `methods`, `fit_literature_curves`) to
    sys.path, then import as top-level modules. This is HoloForge's own
    calling convention, not one invented here -- changing it would mean
    diverging from how HoloForge is imported everywhere else in its own
    codebase.
    """

    def __init__(self, holoforge_code_path: str | Path = DEFAULT_HOLOFORGE_CODE_PATH):
        self._code_path = Path(holoforge_code_path)
        if not self._code_path.exists():
            raise FileNotFoundError(
                f"HoloForge repo not found at {self._code_path} — "
                "HoloAgent wraps an existing, frozen HoloForge checkout; "
                "it does not vendor or reimplement the physics. Set "
                "HOLOFORGE_CODE_PATH or pass holoforge_code_path explicitly."
            )

        experiments_path = self._code_path / "experiments"
        for p in (str(self._code_path), str(experiments_path)):
            if p not in sys.path:
                sys.path.insert(0, p)

        import torch

        torch.set_default_dtype(torch.float64)
        self._torch = torch

        from holomedia import MediumParams, NPDDRecorder, SlabBPM

        self.MediumParams = MediumParams
        self.NPDDRecorder = NPDDRecorder
        self.SlabBPM = SlabBPM

        import methods as _holoforge_methods

        self._run_method_fn = _holoforge_methods.run_method

        import fit_literature_curves as _holoforge_fit

        self._fit_curve_fn = _holoforge_fit.fit_curve

    # ------------------------------------------------------------------ helpers
    def build_target(self, n_x: int, pattern: str, period_px: int = 8) -> Any:
        """Canonical target patterns, matching the ones HoloForge's own
        tests build by hand (e.g. tests/test_method_registry.py:63,
        ((x // period_px) % 2)). Kept intentionally small: this is the
        boundary of what a tool call can reasonably specify without
        accepting a raw (n_x,)-length array from the orchestrator, which
        would defeat the point of a typed schema.
        """
        torch = self._torch
        x = torch.arange(n_x)
        if pattern == "square_wave":
            return ((x // period_px) % 2).double()
        if pattern == "single_slit":
            target = torch.zeros(n_x, dtype=torch.float64)
            lo, hi = n_x * 2 // 5, n_x * 3 // 5
            target[lo:hi] = 1.0
            return target
        raise ValueError(f"unknown target pattern {pattern!r}, expected 'square_wave' or 'single_slit'")

    def build_recorder_and_bpm(
        self,
        params: dict,
        n_x: int,
        dx: float,
        t_total: float = 10.0,
        n_steps: int = 300,
        wavelength_um: float = 0.405,
        n_z: int = 32,
        z_recon_um: float = 5.0e4,
    ) -> tuple[Any, Any]:
        medium_params = self.MediumParams(**params)
        recorder = self.NPDDRecorder(n_x, dx, t_total=t_total, n_steps=n_steps, params=medium_params)
        bpm = self.SlabBPM(
            n_x, dx, wavelength_um, medium_params.thickness, n_z=n_z, n0=medium_params.n0, z_recon_um=z_recon_um
        )
        return recorder, bpm

    # ------------------------------------------------------------------ T7 / T3 passthroughs
    def run_method(self, **kwargs: Any) -> dict:
        """Passthrough to experiments/methods.py::run_method — the one
        stable, tested, uniform dispatch boundary in HoloForge (see
        tests/test_method_registry.py)."""
        return self._run_method_fn(**kwargs)

    def fit_curve(self, **kwargs: Any) -> dict:
        """Passthrough to experiments/fit_literature_curves.py::fit_curve."""
        return self._fit_curve_fn(**kwargs)

    # ------------------------------------------------------------------ T8: hand-built, not HoloForge-native
    def check_validity_region(self, params: dict, K_rad_per_um: float | None = None) -> dict:
        """No equivalent exists in HoloForge itself (see module docstring
        and docs/tool_coverage_matrix.md) -- this checks the hand-
        transcribed bounds above, and says so in the result rather than
        presenting a HoloForge-native validity model that doesn't exist.
        """
        violations: dict[str, tuple[float, tuple[float, float]]] = {}
        distances: list[float] = []
        for name, value in params.items():
            bounds = MEDIUM_PARAM_LITERATURE_BOUNDS.get(name)
            if bounds is None:
                continue
            lo, hi = bounds
            if value < lo:
                violations[name] = (value, bounds)
                distances.append((lo - value) / (hi - lo))
            elif value > hi:
                violations[name] = (value, bounds)
                distances.append((value - hi) / (hi - lo))

        if K_rad_per_um is not None:
            k_lo, k_hi = VALIDATED_K_RANGE_RAD_PER_UM
            if K_rad_per_um < k_lo:
                violations["K_rad_per_um"] = (K_rad_per_um, (k_lo, k_hi))
                distances.append((k_lo - K_rad_per_um) / (k_hi - k_lo))
            elif K_rad_per_um > k_hi:
                violations["K_rad_per_um"] = (K_rad_per_um, (k_lo, k_hi))
                distances.append((K_rad_per_um - k_hi) / (k_hi - k_lo))

        in_region = len(violations) == 0
        distance = None if in_region else max(distances)
        return dict(in_validated_region=in_region, violations=violations, distance_from_bounds=distance)

    # ------------------------------------------------------------------ T6: finite-difference, not autograd
    def sensitivity_finite_difference(
        self,
        base_params: dict,
        parameter: str,
        perturbation_pct: float,
        n_x: int,
        dx: float,
        target_pattern: str,
        target_period_px: int,
        method_id: str,
        seed: int,
        n_iters: int,
        lr: float,
        dose_budget: float,
        output_quantity: str,
    ) -> dict:
        """One-at-a-time finite-difference sensitivity, matching HoloForge's
        own S2 tier (experiments/manifest.py) -- NOT autograd. See module
        docstring: real autograd w.r.t. physical medium parameters is not
        available in HoloForge today.
        """
        if parameter not in base_params:
            raise KeyError(f"parameter {parameter!r} not in base_params")

        def run_at(value: float) -> float:
            perturbed = dict(base_params, **{parameter: value})
            recorder, bpm = self.build_recorder_and_bpm(perturbed, n_x, dx)
            target = self.build_target(n_x, target_pattern, target_period_px)
            result = self.run_method(
                method_id=method_id,
                target=target,
                recorder=recorder,
                bpm=bpm,
                seed=seed,
                n_iters=n_iters,
                lr=lr,
                dose_budget=dose_budget,
            )
            if output_quantity not in result:
                raise KeyError(
                    f"output_quantity {output_quantity!r} not in run_method result "
                    f"(available: {sorted(result.keys())})"
                )
            return float(result[output_quantity])

        base_value = base_params[parameter]
        delta = base_value * perturbation_pct
        plus = run_at(base_value + delta)
        minus = run_at(base_value - delta)
        base_output = run_at(base_value)
        d_output_d_param = (plus - minus) / (2 * delta) if delta != 0 else float("nan")

        return dict(
            parameter=parameter,
            base_value=base_value,
            perturbation_pct=perturbation_pct,
            output_quantity=output_quantity,
            base_output=base_output,
            output_plus=plus,
            output_minus=minus,
            d_output_d_param=d_output_d_param,
            method="finite_difference_central",
        )
