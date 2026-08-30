# Tool coverage matrix

Checked-in artefact per TRD §3.1. A quantity is not "done" until all six
columns are true. Update this file whenever a tool's status changes —
it is the source of truth for R2 (PRD §5).

| # | Tool | Phase | computed | exposed | typed | tested | documented | validity-bounded |
|---|---|---|---|---|---|---|---|---|
| T1 | `get_phase_map` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T2 | `get_diffraction_efficiency` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T3 | `get_fit_quality` | 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T4 | `get_medium_state` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T5 | `reconstruct_wavefront` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T6 | `get_sensitivity` | 1 | ✓ (finite-difference, not autograd — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T7 | `get_convergence_trace` | 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T8 | `check_validity_region` | 1 | ✓ (hand-transcribed bounds, not a HoloForge model — see below) | ✓ | ✓ | ✓ | ✓ | n/a (is the validity check itself) |
| T9 | `run_parameter_sweep` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T10 | `query_run_store` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T11 | `compare_runs` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T12 | `retrieve_literature` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |

**computed** — HoloForge actually computes this quantity and the adapter
call into it works.
**exposed** — registered in `holoagent.tools.registry` with a JSON schema.
**typed** — returns `ToolResult`, raises `ToolError`/`MissingParameterError`
per the contract in `holoagent/tools/contract.py`.
**tested** — has a passing test in `tests/` exercising real HoloForge output
(`tests/test_tools_physics.py`, `tests/test_orchestrator_e2e.py`; marked
`slow`, excluded from the default `pytest` run, run explicitly with
`pytest -m slow`).
**documented** — has a docstring stating what it returns, its source
module, and any deviation from the TRD's original description.
**validity-bounded** — `in_validated_region` / `distance_from_bounds` are
populated from a real check, not placeholder values.

## Phase 1 gate — status: MET

TRD §8 Phase 1 gate: "One real multi-step question answered end to end
with full provenance." `tests/test_orchestrator_e2e.py` chains T8 → T7 →
T6 → T3 (4 dependent tool calls, R6's minimum) through a real
`Orchestrator.answer()` loop against live HoloForge, with every call
recorded to `ProvenanceStore` and parent_run_id reconstructing the chain.
The orchestrator's model backend is `ScriptedBackend` (a fixed script, not
a live LLM) — no `ANTHROPIC_API_KEY` is available in this environment, so
the *tool execution and provenance plumbing* is proven end to end, but
*real tool selection by a model* is not yet exercised. That's the actual
remaining Phase 1 gap: swap in `AnthropicBackend` (or any `ModelBackend`)
with a real key and re-run Set A-style questions.

## Two corrections to the TRD surfaced while wiring T6 and T8

**T6 (`get_sensitivity`).** The TRD describes this as "free from the
differentiable twin (autograd)". That's true for d(loss)/d(exposure field
E(x)) — already used internally by every HoloForge optimizer — but NOT
true for d(output)/d(physical medium parameter). `NPDDRecorder` bakes
`MediumParams` into a non-differentiable buffer (`G_hat`) at construction
(`holomedia/npdd.py`), so no autograd path from e.g. `sigma` to any output
exists in HoloForge today, and building one would mean modifying frozen
layer-4 code (out of scope, TRD §1). T6 is implemented as one-at-a-time
central finite-difference perturbation instead — the same method
HoloForge's own S2 experiment tier uses (`experiments/manifest.py`) — and
every `ToolResult` from T6 says `"NOT autograd"` in `units` and
`"method": "finite_difference_central"` in `value`, precisely so the
orchestrator (and Set C) can't present it as an exact derivative.

**T8 (`check_validity_region`).** No validity-region concept exists
anywhere in HoloForge as code or config — confirmed by inspecting
`holomedia/*.py`, `experiments/*.py`, and `configs/media/*.yaml`. What T8
checks is hand-transcribed from two places in the HoloForge repo that are
prose, not enforced constraints: `MediumParams`'s docstring ("Literature
ranges") in `holomedia/npdd.py`, and the K = 1.96–15.71 rad/µm range
written down in `docs/validation_standard.md` / `docs/parameter_provenance.md`.
See `holoagent/physics/holoforge_adapter.py`'s `MEDIUM_PARAM_LITERATURE_BOUNDS`
and `VALIDATED_K_RANGE_RAD_PER_UM`. Unlike T3/T7 (which call live HoloForge
code), T8 has no automated link back to HoloForge — if those source docs
change, T8 goes stale silently until someone re-checks it by hand.
