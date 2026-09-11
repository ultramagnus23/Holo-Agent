# Tool coverage matrix

Checked-in artefact per TRD §3.1. A quantity is not "done" until all six
columns are true. Update this file whenever a tool's status changes —
it is the source of truth for R2 (PRD §5).

| # | Tool | Phase | computed | exposed | typed | tested | documented | validity-bounded |
|---|---|---|---|---|---|---|---|---|
| T1 | `get_phase_map` | 2 | ✓ (rescoped: deterministic re-run, no phase — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T2 | `get_diffraction_efficiency` | 2 | ✓ (rescoped to the real spatial field — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T3 | `get_fit_quality` | 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T4 | `get_medium_state` | 2 | ✓ (no dye state d — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T5 | `reconstruct_wavefront` | 2 | ✓ (Kogelnik tier only, not full BPM — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T6 | `get_sensitivity` | 1 | ✓ (finite-difference, not autograd — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T7 | `get_convergence_trace` | 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| T8 | `check_validity_region` | 1 | ✓ (hand-transcribed bounds, not a HoloForge model — see below) | ✓ | ✓ | ✓ | ✓ | n/a (is the validity check itself) |
| T9 | `run_parameter_sweep` | 2 | ✓ (batched only for target/seed, looped for medium params — see below) | ✓ | ✓ | ✓ | ✓ | ✓ |
| T10 | `query_run_store` | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | n/a (structured records, not a physical quantity) |
| T11 | `compare_runs` | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | n/a (structured diff, not a physical quantity) |
| T12 | `retrieve_literature` | 2 | ✓ (plain keyword search, not embedding/RAG — see below) | ✓ | ✓ | ✓ | ✓ | n/a (returns cited text, not a physical quantity) |

**All 12 tools are now implemented.** Every one carries at least one
honest scope caveat (see the per-tool notes throughout this document) —
none of them do exactly what the original TRD sketch described, because
that sketch was written before anyone checked what HoloForge actually
computes. That gap-finding, not the wiring itself, was most of the work.

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
See `holoagent/domains/holoforge/adapter.py`'s `MEDIUM_PARAM_LITERATURE_BOUNDS`
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

## Runtime evidence validator — status: implemented (Set C is now also a runtime gate, not just an offline audit)

`Orchestrator._validate_and_finalize` runs Set C's `audit_answer` against
every draft final answer before it can be returned to the caller. A
violation blocks the draft, returns a deterministic abstention citing the
unsupported token(s), and records the block to provenance
(`ProvenanceStore.record_validator_block`, `tool="_evidence_validator"`).
This closes the gap between "Set C catches fabrication in eval/CI" and
"a fabricated number can still reach the user in a live session" — found
to be a real, not theoretical, gap: `qwen2.5:7b-instruct` fabricated a
Brewster-angle value live during Set B testing despite a system prompt
explicitly forbidding it (see `tests/test_set_b_live_grading.py`'s
documented xfail). Design is block-and-abstain, not block-and-regenerate
— see `holoagent/orchestrator/orchestrator.py`'s
`_validate_and_finalize` docstring for why retrying isn't the safer
default here. Tests: `tests/test_evidence_validator.py` (5 fast tests:
clean pass-through, fabrication blocked, block recorded to provenance,
user-supplied numbers not blocked, real tool-result numbers not blocked).

## Milestone 2 (master-build prompt): T2, T10, T11 implemented

Before touching any of T1/T2/T4/T5/T9, an Explore agent investigated what
each is actually implementable against, live in the HoloForge source
(not assumed) — full findings are in git history for this commit. Verdict
for all five: **partially supportable** — HoloForge has real capability
in each direction, but every one of the original specs overclaims in some
way. T2 was the cleanest to close this pass; T1/T4/T5/T9 remain open,
each documented below with the concrete constraint HoloForge itself
imposes (see "Milestone 2 continued" further down, once those land).

**T2 (`get_diffraction_efficiency`) — rescoped, not just implemented.**
The original spec ("efficiency across the field") describes a quantity
that doesn't exist anywhere in HoloForge — confirmed by reading
`holomedia/optimize.py` and `holomedia/diffraction.py` directly: the only
"diffraction efficiency" HoloForge computes is a single scalar (a
masked-sum ratio, `holomedia.diffraction_efficiency`). What genuinely IS
spatially resolved is the reconstructed intensity field
(`SlabBPM.forward`'s real-valued `|E|^2` output) — computed internally by
every method but discarded by `experiments/methods.py::run_method` before
it returns (confirmed: its return dict keeps only the scalar). T2 now
exposes that discarded field via
`HoloForgeAdapter.run_method_with_field`, which calls the exact same
underlying per-method functions (`media_blind_gs`, `media_in_the_loop`,
etc., imported from `holomedia` — the same functions
`experiments/methods.py` itself calls) rather than reimplementing
anything. No phase is returned — this BPM formulation doesn't track phase
separately from intensity, and claiming otherwise would be claiming a
computation that was never performed. Known scaling gap, documented not
hidden: the field arrays are inlined into `ToolResult.value` (fine for
the small grids this is tested against, n_x<=128); TRD §4 says large
fields belong on disk with a path reference, which isn't built yet.
Tested live against real HoloForge (`tests/test_tools_t2_t10_t11.py`).

**T10 (`query_run_store`) and T11 (`compare_runs`) — implemented, no
HoloForge dependency.** These read `holoagent`'s own `ProvenanceStore`,
not HoloForge, so no rescoping was needed against physics reality. The
real design problem was different: unlike T3/T6/T7/T8, which reach
HoloForge via a process-wide adapter singleton, T10/T11 need THIS
session's store specifically — a global default would silently answer
against the wrong session's data. Solved with a scoped contextvar
(`holoagent/provenance/context.py`, `use_store`/`get_current_store`),
set by `Orchestrator.answer()` for the duration of each call. T10 supports
equality/threshold filtering (`tool`, `failed`, `since`/`until`,
`value_field` + `value_gt`/`value_lt`/`value_eq`) — deliberately not a
general JSON-path query language, since nothing in the master prompt's
own examples needs one. T11 diffs inputs, output values, and validity
flags between two runs, including comparing a success against a failure
(a real, useful case: "why did this one fail when that one didn't"). Both
fully tested (`tests/test_tools_t2_t10_t11.py`, fast — no HoloForge
compute needed).

Implementing these retired now-stale Set B claims: `b02`
(diffraction-efficiency-field stub), `b06` (query-run-store stub), `b07`
(compare-runs stub) were removed from `set_b.jsonl` rather than left
false — exactly the staleness `test_set_b_questions.py` exists to catch,
and it did.

**T12 (`retrieve_literature`) — implemented as plain keyword search, not
embedding/RAG.** Master-prompt §38 warns against "unnecessary web-search
frameworks"; TRD §6 says "RAG (T12): plain — no heavy framework." For a
corpus this size (~750 lines across ~10 real HoloForge markdown files —
`docs/*.md`, `README.md`, `data/literature/README.md`), a real embedding
index buys nothing a substring search doesn't already get, and adds a
failure mode this project can't afford: a semantic "near match" that
isn't the passage the answer actually relies on.
`holoagent/domains/holoforge/literature_corpus.py` chunks the real files on blank
lines and scores by raw keyword occurrence count; every returned passage
is a verbatim substring of a real file with a `file:line` citation — the
tool never summarizes or paraphrases, so "unsupported text generation"
(never-do #10) can't happen inside it. Removed `b08`
(retrieve-literature-stub) from Set B as stale for the same reason as
the three above.

Implementing T12 surfaced two more real bugs, fixed in the same pass:

1. **Circular import** (`holoagent/provenance/context.py` importing
   `ProvenanceStore` for a type hint created a cycle through
   `holoagent.tools.__init__` → T10 → context → store, mid-init).
   Fragile-by-luck: existing tests happened to avoid it because they all
   imported `holoagent.tools` first. Fixed with a `TYPE_CHECKING` guard —
   the runtime dependency wasn't needed at all.
2. **Set C only grounded numbers in structured JSON leaves, never numbers
   embedded in string values** (a retrieved passage's prose, or a tool's
   `diagnostic` message — which wasn't walked into the pool at all before
   this). Same root cause as the error-text gap from Set B's b14, just
   not limited to failure records. Fixed by generalizing
   `_walk_numeric_leaves` itself; see `docs/set_c_matching_rule.md`.

## Milestone 2 continued: T1, T4, T5, T9 implemented — all 12 tools done

Same discipline as T2/T10/T11/T12: the Explore agent's earlier
verdict for all four was "partially supportable," so each is scoped to
exactly the real HoloForge capability, never past it.

**T4 (`get_medium_state`) — the cleanest of the four.**
`NPDDRecorder.forward(..., return_history=True)` (`holomedia/npdd.py`)
genuinely returns `(u, N)` at every timestep in one pass — confirmed by
reading the class directly. `dn` at a requested timestep is derived by
applying the SAME formula the class uses for its final step
(`dn_max * tanh(1.5*N)`) to that step's `N`, not new physics. Honest gap:
dye-depletion state (`d`) is tracked internally but not included in
`return_history`'s captured tuples — reconstructing it would mean
re-deriving its bleaching recurrence outside the frozen class, which
crosses from "calling HoloForge differently" into "writing new
physics-adjacent code." `ToolResult.value["dye_state_d"]` is explicitly
`None`, not omitted, so a caller knows the gap exists rather than
inferring it from a missing key. Interesting real finding while testing:
the tool reproduces Set A q06's documented non-monotonic-saturation
result directly (dn1 amplitude at an early timestep exceeded a later
one) — the tool wasn't built to demonstrate that, it just did.

**T5 (`reconstruct_wavefront`) — scoped to the Kogelnik analytic tier
only.** `holomedia.kogelnik_de(dtheta=...)` (`holomedia/diffraction.py`)
is the only place in HoloForge with a real, tested, wired-in
reconstruction-angle parameter (exercised by
`experiments/fit_literature_curves.py::simulate_angular_de`). The actual
BPM engine everything else in this registry uses (`SlabBPM`, what
`run_method`/T2/T7 call) has no angle parameter at all — always normal
incidence, confirmed by reading its signature and every call site in the
repo. `SlabBPM` does accept an `incident=` field that could in principle
carry a tilted wavefront, but zero call sites anywhere in HoloForge ever
do this — wrapping it would mean exercising a real but never-validated
path, not a verified capability, so T5 doesn't. Verified live: efficiency
peaks at Bragg match and drops sharply when detuned, as physically
expected.

**T9 (`run_parameter_sweep`) — real batching for one axis, honest
looping for the other.** Confirmed real, bit-identical-to-unbatched
batching exists (`holomedia/optimize.py`'s `_batched` functions,
HoloForge's own `tests/test_smoke.py` verifies the equivalence) — but
only over a `(target, seed)` axis, for 4 of 7 methods (MIL, BSGD, GS,
ORC — not LPC, SAT, ORU). Sweeping physical medium parameters is NOT
batched anywhere in HoloForge: every real sweep script in the repo loops
in Python, because `kappa`/`D0`/`sigma`/etc. are baked into
`NPDDRecorder`'s and `SlabBPM`'s per-instance FFT buffers at
construction — one instance cannot represent multiple parameter settings
at once. T9 does the same: `sweep_field="seed"` with a batchable
`method_id` gets real batched execution; anything else gets an honest
Python loop, and `ToolResult.value["batched"]` says which happened — a
medium-parameter sweep is never presented as batching HoloForge doesn't
have. Bug caught during testing, not after: the loop fallback originally
tried to inject `sweep_field="seed"` as a `MediumParams` override key
when a non-batchable method fell back to it (e.g. LPC), which isn't a
real field — `TypeError: MediumParams.__init__() got an unexpected
keyword argument 'seed'`. Fixed to branch on `sweep_field == "seed"`
explicitly rather than treating every sweep axis the same way.

**T1 (`get_phase_map`) — rescoped hardest of all, name included.** No
optimizer method in HoloForge ever snapshots intermediate field state —
confirmed by reading every method in `holomedia/optimize.py`: each
overwrites `E`/`dn`/`recon` in place every loop pass and returns only the
final post-loop state. There is no native checkpoint/resume mechanism.
What IS real: HoloForge's seeding is genuinely deterministic
(`_seeded_init_theta` uses a CPU-local `torch.Generator`, explicitly not
CUDA, explicitly for bit-reproducibility per its own comment) and every
optimizer step is deterministic given that seed. So calling the same
method with the same seed and `n_iters=checkpoint` reproduces the exact
trajectory up to that point — verified live, not assumed: two identical
calls to T1 produced bit-identical `exposure_field` arrays. This is NOT a
free snapshot lookup: reading 5 different checkpoints means 5 separate
full re-optimizations from scratch. And "phase map" is a misnomer kept
only as the tool's registered name — HoloForge's BPM formulation is
real-valued (intensity), never phase-resolved, so no phase is returned
(`ToolResult.value["phase"]` is explicitly `None`). Under the hood this
calls the exact same `HoloForgeAdapter.run_method_with_field` T2 uses,
just reframed as "state at iteration k" rather than "final result."

Implementing these four retired the last of the stale "stub" claims in
Set B: `b01`/`b03`/`b04`/`b05` (which claimed T1/T4/T5/T9 were stubs)
were removed as false, same pattern as the four before them. Set B is
now 35 questions (was 43) — `_STUB_TOOL_CALLS` in
`tests/test_set_b_questions.py` is now empty because there are no
remaining stub tools to check, kept as an empty dict rather than deleted
in case a future phase adds one.

## Milestone 4: waveoptics domain (general wave optics, not HoloForge)

A second domain plugin, `holoagent/domains/waveoptics/` — independent
scalar wave optics, master-build §12's explicit instruction that
HoloForge must not become a general optics engine. Deliberately narrow
(master-build §38: depth over feature count; §37: do not build
everything at once) — Stage 1 (fields), Stage 2 (propagation), most of
Stage 3 (apertures, thin lens, grating), and the core of Stage 4
(idealized 4f-system spatial filtering), each with a real,
independently-verified analytic reference case per §28.

| Tool | computed | exposed | typed | tested | documented | validity-bounded |
|---|---|---|---|---|---|---|
| `create_plane_wave` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (Nyquist on the phase ramp) |
| `create_gaussian_beam` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (grid width + resolution) |
| `calculate_intensity` | ✓ | ✓ | ✓ | ✓ | ✓ | n/a (deterministic transform) |
| `calculate_phase` | ✓ | ✓ | ✓ | ✓ | ✓ | n/a (deterministic transform) |
| `apply_aperture` | ✓ (1D rect/slit only) | ✓ | ✓ | ✓ | ✓ | ✓ (aperture vs. grid extent) |
| `propagate_angular_spectrum` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (spectral-content-aware, see below) |
| `propagate_fraunhofer` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (real Fraunhofer-distance check) |
| `apply_thin_lens` | ✓ (paraxial, ideal) | ✓ | ✓ | ✓ | ✓ | ✓ (half-aperture/f paraxial-angle check) |
| `apply_grating` | ✓ (sinusoidal amplitude only) | ✓ | ✓ | ✓ | ✓ | ✓ (grating-period sampling check) |
| `apply_fourier_filter` | ✓ (hard-edged lowpass/highpass) | ✓ | ✓ | ✓ | ✓ | ✓ (cutoff vs. grid Nyquist frequency) |

**Five analytic validations, all run through the actual registered tools
(not the underlying physics functions in isolation)** —
`tests/test_waveoptics_propagation.py` and
`tests/test_waveoptics_imaging.py`:

1. A plane wave through a slit aperture, propagated to the far field via
   `propagate_fraunhofer`, reproduces the exact analytic sinc² single-slit
   diffraction pattern to <0.02 absolute intensity error (normalized
   peak=1) — the FT-of-a-rect-is-a-sinc identity has no approximation to
   go wrong, so this is a strong, near-exact check.
2. A Gaussian beam propagated by `propagate_angular_spectrum` (the exact,
   non-paraxial method) reproduces the paraxial `w(z) = w0*sqrt(1+(z/zR)²)`
   divergence law to <0.1% relative error, at a gentle divergence angle
   (w0=20µm, wavelength=0.5µm) where the paraxial approximation itself is
   expected to hold well.
3. A plane wave through a slit aperture and then `apply_thin_lens`,
   propagated exactly to z=f via `propagate_angular_spectrum`, reproduces
   the classic Fourier-transforming-lens identity (Goodman, *Fourier
   Optics* ch. 5): the focal-plane field's first null lands at
   `wavelength*f/width` (the same formula as the aperture's own Fraunhofer
   first null) to <1% relative error, with >85% of the focal-plane power
   in that central lobe (theoretical sinc² value ≈90.3%). This is the
   textbook prediction, not an invented check — it's what distinguishes
   a real Fourier-transforming lens from a tool that merely "applies some
   phase."
4. A plane wave through `apply_grating` (sinusoidal amplitude,
   `t(x)=0.5*(1+cos(2πx/period))`), propagated to the far field, produces
   exactly 3 diffraction orders (0, ±1) at the analytic angles
   `sin(θ_m)=m·wavelength/period`, matched to within 5 grid-resolution
   units, holding >99% of the far-field power — confirming both the
   positions and the (sinusoidal-grating-specific) absence of higher
   orders.
5. `apply_fourier_filter` (Stage 4: the net field-level effect of an
   idealized 4f imaging system with a hard-edged stop at its Fourier
   plane — `image(x) = IFFT(mask(fx)·FFT(object(x)))`) reproduces the
   *exact* closed-form Fourier pair `rect(fx/2fc) ↔ 2·fc·sinc(2·fc·x)`:
   filtering a numerical impulse (delta function) through the lowpass
   filter matches the filter's own analytic point-spread function to
   <0.05% of its peak — this is the textbook diffraction-limited-
   resolution result (Goodman ch. 6–7), not an approximation. Two further
   exact-identity checks: a lowpass + highpass filter at the same cutoff
   sum back to the original field to numerical precision (complementary
   masks partition all spatial frequencies with no gap/overlap), and a
   generous lowpass cutoff leaves a Gaussian beam whose real spectral
   content sits far below that cutoff numerically unchanged.

**A real bug caught by the second validation, not assumed away:** the
first Nyquist-sampling check written for `propagate_angular_spectrum`
used the blanket worst-case rule `dx <= wavelength/2` (correct for
representing *any* possible spatial frequency up to k) — this flagged
the w0=20µm beam at dx=1.0µm as under-sampled (`in_validated_region:
False`) even though it numerically matched its analytic w(z) to 0.01%.
The rule was checking against a frequency band the field's actual
content never touches (a gently-diverging beam's energy sits in a narrow
band near kx=0, not spread to the theoretical maximum). Fixed by
computing what fraction of the field's *real* spectral energy sits near
the grid's Nyquist edge (outer 10% of the sampled frequency range)
instead of a blanket bound — the same principle as the evanescent-energy
check right next to it, just for aliasing risk instead of near-field
loss. Confirmed the fix doesn't just paper over false positives:
`create_plane_wave`'s own (unrelated, single-exact-frequency) Nyquist
check still correctly flags a genuinely aliased tilted plane wave, and
that aliasing is correctly *not* re-detectable downstream by
`propagate_angular_spectrum` once it's already baked into the discrete
samples — confirming each tool validates at the point where it can
actually see the problem, not after the fact.

**Explicitly not built, not a gap left unstated:** a true 2D circular
aperture (this domain is 1D transverse fields only, matching the
holoforge domain's own convention — a 1D slit of some half-width does
NOT reproduce the Airy pattern master-build §28 names, which needs real
2D radial symmetry; rather than build a mislabeled stand-in, that
validation case is not attempted). No general/arbitrary phase masks
(only the specific cases above: ideal thin lens, sinusoidal grating).
`apply_fourier_filter` covers only a hard-edged rect passband/stopband —
no soft/apodized filters, no formal PSF/OTF/MTF tool that reports a
system's resolving power as a first-class quantity (the sinc impulse
response is currently validated only inside a test, not exposed as its
own tool output), no 2D imaging, coherence, or phase retrieval — the
remainder of Stage 4, all of Stage 5, and master-build Milestones 5-8
generally.

This domain is deliberately NOT auto-loaded by `holoagent.tools`'s
backward-compatible default (see that module's docstring) — it's a
second domain now, not the one-domain-era special case. Using it means
an explicit `import holoagent.domains.waveoptics`.
