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

import math
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

        from holomedia import (
            media_blind_gs,
            media_blind_sgd,
            media_in_the_loop,
            linear_precomp,
            oracle_ideal,
            oracle_unconstrained,
            sat_sgd,
            diffraction_efficiency,
        )

        self._diffraction_efficiency_fn = diffraction_efficiency

        from holomedia import kogelnik_de

        self._kogelnik_de_fn = kogelnik_de

        from holomedia import (
            psnr_si_batch,
            media_in_the_loop_batched,
            media_blind_sgd_batched,
            media_blind_gs_batched,
            oracle_ideal_batched,
        )

        self._psnr_si_batch_fn = psnr_si_batch
        self._batched_method_fns = {
            "MIL": media_in_the_loop_batched,
            "BSGD": media_blind_sgd_batched,
            "GS": media_blind_gs_batched,
            "ORC": oracle_ideal_batched,
        }

        self._method_fns = {
            "GS": media_blind_gs,
            "BSGD": media_blind_sgd,
            "LPC": linear_precomp,
            "MIL": media_in_the_loop,
            "SAT": sat_sgd,
            "ORC": oracle_ideal,
            "ORU": oracle_unconstrained,
        }

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

    def build_grating_exposure(self, n_x: int, dx: float, K_rad_per_um: float, contrast: float = 0.9) -> Any:
        """The canonical sinusoidal-grating exposure pattern used
        throughout HoloForge's own literature-fitting code (e.g.
        fit_literature_curves.py's simulate_growth_dn: "exposure = 1.0 +
        0.9*cos(K*x)") -- the natural "exposure" for T4's recording-state
        questions, distinct from build_target's binary reconstruction
        targets (T4 is about the recording physics itself, not an
        optimizer's target pattern).
        """
        torch = self._torch
        x = torch.arange(n_x, dtype=torch.float64) * dx
        return 1.0 + contrast * torch.cos(K_rad_per_um * x)

    def get_medium_state_at_timestep(
        self,
        params: dict,
        n_x: int,
        dx: float,
        K_rad_per_um: float,
        contrast: float,
        t_total: float,
        n_steps: int,
        timestep: int,
    ) -> dict:
        """Real HoloForge capability: NPDDRecorder.forward(...,
        return_history=True) returns (u, N) at every one of n_steps
        timesteps in one pass (holomedia/npdd.py) -- confirmed by reading
        the class directly, not assumed. dn at each step is derived by
        applying the SAME formula NPDDRecorder itself uses for the final
        step (dn_max * tanh(1.5*N)) to that step's N -- not new physics,
        the identical formula, just evaluated earlier.

        Dye/exposure-depletion state (d) is genuinely NOT exposed by
        return_history (only (u, N) are captured) and is not
        reconstructable from the outside without re-deriving the bleaching
        recurrence, which would cross from "calling HoloForge differently"
        into "writing new physics-adjacent code" -- out of scope for a
        frozen layer 4. This is documented, not silently omitted.
        """
        if not (0 <= timestep < n_steps):
            raise ValueError(f"timestep {timestep} out of range [0, {n_steps})")

        torch = self._torch
        medium_params = self.MediumParams(**params)
        recorder = self.NPDDRecorder(n_x, dx, t_total=t_total, n_steps=n_steps, params=medium_params)
        exposure = self.build_grating_exposure(n_x, dx, K_rad_per_um, contrast)

        _, hist = recorder(exposure, return_history=True)
        u_t, N_t = hist[timestep]
        dn_t = medium_params.dn_max * torch.tanh(1.5 * N_t)
        x = torch.arange(n_x, dtype=torch.float64) * dx
        dn1_amplitude = float((2.0 * torch.mean(dn_t * torch.cos(K_rad_per_um * x))).abs())

        return dict(
            u=u_t,
            N=N_t,
            dn=dn_t,
            dn1_amplitude=dn1_amplitude,
            mean_u=float(u_t.mean()),
            mean_N=float(N_t.mean()),
            dn_max=medium_params.dn_max,
            fraction_of_dn_max=dn1_amplitude / medium_params.dn_max if medium_params.dn_max else float("nan"),
            real_time_s=timestep * (t_total / n_steps),
        )

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

    # ------------------------------------------------------------------ T2: spatial field, not run_method's discarded one
    def run_method_with_field(
        self,
        method_id: str,
        target: Any,
        recorder: Any,
        bpm: Any,
        seed: int,
        n_iters: int = 800,
        lr: float = 5e-2,
        dose_budget: float = 1.0,
        contrast_cap: float | None = None,
        converge_tol: float | None = None,
        log_every: int = 50,
    ) -> dict:
        """Calls the SAME underlying per-method functions
        experiments/methods.py::run_method dispatches to (imported from
        holomedia, not reimplemented) -- but keeps the spatially-resolved
        exposure (E) and reconstructed intensity (recon) fields that
        run_method computes and then discards, returning only a reduced
        scalar diffraction_efficiency. This is the real capability behind
        T2 -- see docs/tool_coverage_matrix.md's note on why T2 was
        rescoped from a nonexistent "spatial efficiency map" (no such
        quantity is computed anywhere in HoloForge) to this, the actual
        spatially-resolved intensity HoloForge does compute.
        """
        if method_id not in self._method_fns:
            raise ValueError(f"unknown method_id {method_id!r}, expected one of {sorted(self._method_fns)}")

        fn = self._method_fns[method_id]
        history: list = []
        iterations_run = n_iters
        early_stop_reason = "n/a"

        if method_id == "GS":
            E, recon = fn(target, recorder, bpm, dose_budget=dose_budget, seed=seed, contrast_cap=contrast_cap)
            iterations_run = 0
            early_stop_reason = "closed_form_no_optimization"
        elif method_id == "LPC":
            E, recon = fn(target, recorder, bpm, dose_budget=dose_budget, contrast_cap=contrast_cap)
            iterations_run = 0
            early_stop_reason = "closed_form_no_optimization"
        elif method_id == "ORU":
            E, recon = fn(target, recorder, bpm, n_iters=n_iters, lr=lr, seed=seed)
            early_stop_reason = "n_iters_exhausted"
        elif method_id == "ORC":
            E, recon = fn(
                target, recorder, bpm, n_iters=n_iters, lr=lr, dose_budget=dose_budget, seed=seed,
                contrast_cap=contrast_cap,
            )
            early_stop_reason = "n_iters_exhausted"
        elif method_id in ("BSGD", "MIL", "SAT"):
            kwargs = dict(
                n_iters=n_iters, lr=lr, dose_budget=dose_budget, seed=seed,
                log_every=log_every, contrast_cap=contrast_cap,
            )
            if method_id == "MIL":
                kwargs["verbose"] = False
            if method_id in ("MIL", "SAT"):
                kwargs["converge_tol"] = converge_tol
            E, recon, history = fn(target, recorder, bpm, **kwargs)
            iterations_run = history[-1][0] if history else 0
            early_stop_reason = (
                "converge_tol" if (converge_tol is not None and iterations_run < n_iters - 1) else "n_iters_exhausted"
            )
        else:
            raise ValueError(f"unhandled method_id {method_id!r} in run_method_with_field dispatch")

        # Same mask convention experiments/methods.py::run_method uses
        # (mask = target > 0.05) so the scalar here is directly comparable
        # to T7's final_psnr-adjacent diffraction_efficiency, not a
        # different definition invented for this tool.
        mask = (target > 0.05).double()
        scalar_efficiency = float(self._diffraction_efficiency_fn(recon, mask))

        return dict(
            method_id=method_id,
            E=E,
            recon=recon,
            scalar_efficiency=scalar_efficiency,
            iterations_run=iterations_run,
            early_stop_reason=early_stop_reason,
            history=history,
        )

    # ------------------------------------------------------------------ T9: real batching only for (target, seed)
    def run_seed_sweep_batched(
        self,
        method_id: str,
        params: dict,
        n_x: int,
        dx: float,
        target_pattern: str,
        target_period_px: int,
        seeds: list[int],
        n_iters: int = 400,
        lr: float = 5e-2,
        dose_budget: float = 1.0,
        log_every: int = 50,
    ) -> dict:
        """Real HoloForge batching: one (B, n_x) Adam trajectory instead of
        B sequential Python-loop calls -- confirmed bit-identical to the
        unbatched path by HoloForge's own tests/test_smoke.py (see
        holomedia/optimize.py's comment above these functions). Only 4 of
        HoloForge's 7 methods have a batched counterpart (MIL, BSGD, GS,
        ORC — not LPC, SAT, ORU), and the batch axis is (target, seed), not
        physical medium parameters (see run_medium_param_sweep_looped for
        why that can't be batched at all).
        """
        if method_id not in self._batched_method_fns:
            raise ValueError(
                f"no batched implementation for method_id {method_id!r}, "
                f"expected one of {sorted(self._batched_method_fns)}"
            )
        torch = self._torch
        fn = self._batched_method_fns[method_id]
        medium_params = self.MediumParams(**params)
        recorder = self.NPDDRecorder(n_x, dx, params=medium_params)
        bpm = self.SlabBPM(n_x, dx, 0.405, medium_params.thickness, n0=medium_params.n0)
        target = self.build_target(n_x, target_pattern, target_period_px)
        batch_size = len(seeds)
        targets = target.unsqueeze(0).expand(batch_size, -1).contiguous()

        history: list = []
        if method_id == "GS":
            E, recon = fn(targets, recorder, bpm, seeds, dose_budget=dose_budget)
        elif method_id in ("BSGD", "ORC"):
            E, recon = fn(targets, recorder, bpm, seeds, n_iters=n_iters, lr=lr, dose_budget=dose_budget)
        else:  # MIL
            E, recon, history = fn(
                targets, recorder, bpm, seeds, n_iters=n_iters, lr=lr,
                dose_budget=dose_budget, log_every=log_every, verbose=False,
            )

        psnr_per_row = self._psnr_si_batch_fn(recon, targets)
        mask = (target > 0.05).double()
        de_per_row = (recon * mask).sum(dim=-1) / (recon.sum(dim=-1) + 1e-12)

        return dict(
            method_id=method_id,
            seeds=list(seeds),
            psnr_per_seed=[float(x) for x in psnr_per_row],
            diffraction_efficiency_per_seed=[float(x) for x in de_per_row],
            history=history,
            batched=True,
        )

    def run_medium_param_sweep_looped(
        self,
        method_id: str,
        base_params: dict,
        sweep_field: str,
        sweep_values: list,
        n_x: int,
        dx: float,
        target_pattern: str,
        target_period_px: int,
        fixed_seed: int,
        n_iters: int,
        lr: float,
        dose_budget: float,
        log_every: int,
    ) -> dict:
        """The honest fallback: HoloForge has NO batching capability across
        physical medium parameters at all -- confirmed by reading every
        real sweep script in the repo (experiments/gpu_f2_f3_recovery.py::
        full(), experiments/gpu_bpm_wavelength_sweep.py::main()), which all
        loop in Python. kappa/D0/sigma/etc. are baked into NPDDRecorder's
        and SlabBPM's per-instance FFT buffers at construction, so one
        instance cannot represent multiple parameter settings at once.
        This executes exactly that same kind of loop, one fresh
        recorder/bpm per value -- presented honestly as a loop (see
        T9's `batched: False` field), not marketed as batching HoloForge
        doesn't have.
        """
        results = []
        for value in sweep_values:
            # sweep_field == "seed" (a non-batchable method falling back to
            # this loop) means sweep the SEED, medium params fixed --
            # "seed" is not a MediumParams field and must never be injected
            # into it. Any other sweep_field is a real MediumParams
            # override, seed held fixed instead. Conflating these two was
            # a real bug: TypeError: MediumParams.__init__() got an
            # unexpected keyword argument 'seed', caught live in testing.
            if sweep_field == "seed":
                params = dict(base_params)
                seed = value
            else:
                params = dict(base_params, **{sweep_field: value})
                seed = fixed_seed
            recorder, bpm = self.build_recorder_and_bpm(params, n_x, dx)
            target = self.build_target(n_x, target_pattern, target_period_px)
            r = self.run_method(
                method_id=method_id, target=target, recorder=recorder, bpm=bpm,
                seed=seed, n_iters=n_iters, lr=lr, dose_budget=dose_budget, log_every=log_every,
            )
            results.append(
                dict(
                    sweep_value=value,
                    psnr=r["psnr"],
                    diffraction_efficiency=r["diffraction_efficiency"],
                    iterations_run=r["iterations_run"],
                )
            )
        return dict(method_id=method_id, sweep_field=sweep_field, results=results, batched=False)

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

    # ------------------------------------------------------------------ T5: Kogelnik tier only, not the real BPM path
    def reconstruct_at_angle(
        self,
        dn: float,
        thickness_um: float,
        wavelength_um: float,
        theta_B_deg: float,
        dtheta_deg: float,
    ) -> float:
        """Diffraction efficiency at a reconstruction angle detuned by
        dtheta_deg from the Bragg-matched angle theta_B_deg, via
        holomedia.kogelnik_de(dtheta=...) -- confirmed by reading
        holomedia/diffraction.py directly to be the ONLY place in
        HoloForge with a real, tested, wired-in angle parameter
        (experiments/fit_literature_curves.py:simulate_angular_de calls it
        exactly this way).

        NOT the real BPM reconstruction pipeline (SlabBPM, what
        run_method/T2/T7 actually use): SlabBPM.forward has no angle
        parameter at all (always normal incidence) -- confirmed by reading
        its signature and every call site in the repo. SlabBPM does accept
        an `incident=` field a caller COULD construct as a tilted plane
        wave, but grep across the whole HoloForge repo finds zero call
        sites ever doing this -- it would mean exercising a real but never
        validated/tested code path, not wrapping an existing verified
        capability. T5 is therefore scoped to this analytic Kogelnik tier
        only; it does not answer "off-axis BPM reconstruction" questions.
        """
        torch = self._torch
        de = self._kogelnik_de_fn(
            torch.tensor(dn, dtype=torch.float64),
            thickness_um,
            wavelength_um,
            theta_B=math.radians(theta_B_deg),
            dtheta=torch.tensor(math.radians(dtheta_deg), dtype=torch.float64),
        )
        return float(de)

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
