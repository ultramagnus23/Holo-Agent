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

## Phase 1 gate — status: MET, including model-driven tool selection

TRD §8 Phase 1 gate: "One real multi-step question answered end to end
with full provenance." `tests/test_orchestrator_e2e.py` chains T8 → T7 →
T6 → T3 (4 dependent tool calls, R6's minimum) through a real
`Orchestrator.answer()` loop against live HoloForge, with every call
recorded to `ProvenanceStore` and parent_run_id reconstructing the chain
(that test still uses `ScriptedBackend` — a fixed script — since it's
testing the tool-execution/provenance plumbing specifically, independent
of any model's behavior).

The remaining gap noted in the previous version of this doc — no model had
actually chosen a tool — is closed:
`tests/test_ollama_backend_live.py::test_real_local_model_selects_and_calls_a_real_tool`
runs a real self-hosted model (`qwen2.5:7b-instruct` via Ollama) through
`Orchestrator.answer()`, and the model correctly selects and calls
`check_validity_region` for a validity question, with the resulting run
recorded in the provenance store. See
`holoagent/orchestrator/ollama_backend.py` for the backend and why it's
Ollama/qwen2.5-instruct rather than the TRD's own Phase 1/2 picks (hosted
API, then DeepSeek-V4-Flash) — this project runs self-hosted with no
hosted API key, and the TRD's local-model pick assumed hardware (500GB
disk, implicitly enough VRAM to matter) this dev machine doesn't have (4GB
VRAM laptop GPU).

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

## Set C — status: implemented and tested (was previously a stubbed `NotImplementedError`)

The numeric-token-matching rule (the design decision that blocked Set C —
see `holoagent/eval/set_c_provenance_audit.py`'s old docstring) is decided
and implemented; full rationale in
[docs/set_c_matching_rule.md](set_c_matching_rule.md). Summary: a
provenance pool built from tool outputs + tool call inputs + the user's
own query, matched via exact / rounded-within-3%-relative-error / percent
rules, deliberately biased toward flagging too much (unsupported unit
conversions and derived arithmetic are flagged for human review, not
silently allowed). `tests/test_set_c_provenance_audit.py` (11 fast unit
tests) plus `tests/test_orchestrator_e2e.py` (proves it against a REAL
provenance chain from live HoloForge: a hand-authored answer citing real
run values passes, the same answer with one number swapped for a
fabricated value is caught).

The tool-layer defaults gap this surfaced (HoloForge's own unlogged
`MediumParams` defaults weren't provenance-recorded, so an answer naming
one would false-positive) is now fixed: T3/T6/T7 all include the fully
resolved `MediumParams` in `ToolResult.value`
(`resolved_medium_params`/`resolved_base_medium_params`), not just the
caller's literal overrides. See `docs/set_c_matching_rule.md`.

## Set A grading — status: implemented and run live (was previously a stubbed `NotImplementedError`)

`grade_question`/`run_set_a` in `holoagent/eval/set_a_grounded.py` grades
a real `Orchestrator.answer()` run on two checks that don't require the
model to guess an unstated seed/config: `expected_tools_satisfied` (did it
call the tool(s) the question needs) and `provenance_violations == 0`
(Set C's audit, run against the REAL provenance chain that answer
produced — not a synthetic fixture). `tests/test_set_a_grading.py` covers
the grading logic itself with `ScriptedBackend` (fast, deterministic).

`tests/test_set_a_live_grading.py` runs this against the real self-hosted
model and surfaces a genuine architectural gap rather than hiding it: Set
A's question text is phrased the way the original HoloForge finding
reads (e.g. q01: "For the Bayfol HX growth_dn curve (K=8.98 rad/um),
does fitting {kappa, dn_max}...") — it does NOT carry the digitized (x,y)
curve data `get_fit_quality` actually requires as input, and no tool in
the current registry can look up a named literature curve by source (T12
is still a stub, and even built is scoped for text-passage retrieval, not
structured datasets). Both sides of this gap are tested and pass live
(455s): given the bare question text, the model does not fabricate curve
data or an invented NRMSE (zero provenance violations — it recognizes it
can't answer rather than guessing); given the same finding phrased the
way a researcher who actually has their digitized numbers in hand would
type it, the model correctly calls `get_fit_quality` and produces a fully
grounded answer.

This is a real scoping gap worth closing eventually (a
`get_literature_curve(source_id)`-style lookup tool, or a structured-data
mode for T12), not a Phase 1 blocker — it only affects Set A questions
whose grounding data isn't in the question text itself.
