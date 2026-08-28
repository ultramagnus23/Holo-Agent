# PRD — HoloAgent

**Twin-in-the-loop reasoning agent for computational holography**

Owner: Chaitanya Tripathi
Status: Draft v0.1 — pre-build
Depends on: HoloForge Part 2 repository (differentiable NPDD solver, Kogelnik/BPM diffraction, media-in-the-loop optimizer)

---

## 1. Problem

Frontier language models can discuss holography fluently and are wrong in ways that are hard to detect. They generate plausible diffraction efficiencies, plausible NRMSE values, plausible parameter recommendations — none of which correspond to any computation. In a research setting this is worse than useless, because a confident wrong number costs days before it is caught. The NPDD non-local kernel bug and the saturation-ceiling citation-scoping error in HoloForge Part 2 are both examples of errors that survived precisely because a plausible-looking number did not announce itself as wrong.

Meanwhile, the HoloForge repository already computes the correct answers. It has a validated differentiable twin, a diffraction solver, an optimizer, and a full experiment harness. That capability is reachable only by writing Python and reading arrays.

The gap is not knowledge. It is **access**: there is no interface that lets a researcher ask a question in natural language and get an answer that is computed rather than recalled.

## 2. Product thesis

HoloAgent is a tool-calling agent that treats the HoloForge twin as its source of truth. The language model never produces a physical quantity. It decides *which computations to run*, orchestrates them, and explains the result. Every number in every answer traces to a tool call with a recorded input, output, and provenance record.

The one-line claim:

> Every quantity in an answer is computed, cited, and reproducible — or the system states it cannot compute it.

## 3. Scope

### 3.1 In scope — Tier A: grounded simulation reasoning (the product)

Questions answerable by running or querying the twin. This is where the system is intended to be the best available tool in existence, because it is the only one with access to the simulator.

- Any quantity the HoloForge pipeline computes, individually addressable (see TRD §3 tool registry)
- Multi-step simulation experiments: parameter sweeps, sensitivity analysis, counterfactuals ("what if saturation ceiling rose 15%")
- Interpretation of stored past runs: convergence trajectories, failed fits, parameter regions
- Validity assessment: whether a proposed configuration sits inside or outside the twin's validated region
- Comparison against recorded baselines

### 3.2 In scope — Tier B: grounded literature reasoning (supporting)

Retrieval over a curated corpus: HoloForge Part 1 preprint and Part 2 manuscript, own experiment notes, cited optics literature. Answers must quote or cite retrieved passages. No corpus hit above threshold means no Tier B answer.

### 3.3 Explicitly out of scope

- **General optics tutoring.** Beam propagation basics, laser safety, textbook derivations. The system routes these out with an honest statement rather than competing badly against frontier models.
- **Lab hardware control.** No interface to physical equipment. Simulation only.
- **Fabrication or exposure recommendations presented as safe-to-execute.** The agent may report what the twin predicts; it never issues an instruction to run a physical exposure.
- **Novel physics.** The agent computes with the models implemented in HoloForge. It does not extend, derive, or invent physics.

### 3.4 The never-do list

1. Never emit a numeric physical quantity that did not come from a tool result in the current session.
2. Never interpolate, round, or "approximately" restate a tool output without labelling it as such.
3. Never answer a Tier A question when the required tool is missing, erroring, or out of validated range. Escalate instead.
4. Never present twin output as experimental fact. Simulation results are always labelled as such.
5. Never suppress an uncertainty or validity flag to give a cleaner answer.
6. Never silently substitute a default parameter. Missing input is a question to the user, not a guess. (Direct lift from the K3 config-reader principle: a reader that guesses gives you a different model without telling you.)
7. Never claim reproducibility without a stored run record and hash.

## 4. Users

**Primary — the author, during active research.** Interrogating runs, designing next experiments, catching anomalies. Highest-value use, and the eval set is built from real questions asked during HoloForge work.

**Secondary — a collaborating lab (e.g. Computational Light Lab, UCL).** Someone who knows the physics but not the codebase, and needs to query the twin without reading it.

**Tertiary — reviewers and readers of the paper.** Reproducing a claimed figure by asking for it.

## 5. Core requirements

| # | Requirement | Acceptance test |
|---|---|---|
| R1 | Numbers are computed, not generated | Automated audit: every numeric token in output matches a value present in a tool result from that session. Target: 100%, enforced structurally, not by prompt |
| R2 | Exhaustive tool coverage | Tool coverage matrix (TRD §3) shows every quantity the pipeline computes is exposed, tested, and documented. Target: 100% of enumerated quantities |
| R3 | Honest refusal | On a held-out set of unanswerable / out-of-scope / no-tool-exists questions, correct abstention rate ≥ 95%, confident-wrong rate ≈ 0 |
| R4 | Validity awareness | Any answer involving parameters outside the twin's validated region carries an explicit extrapolation warning |
| R5 | Reproducibility | Every answer carries a run record; re-executing that record reproduces the same numbers bit-for-bit |
| R6 | Multi-tool orchestration | Agent chains ≥ 4 dependent tool calls without losing provenance |
| R7 | Runs on author hardware | Full loop executes on the existing laptop (RTX 3050, ~500 GB disk budget) without cloud dependency for the physics layer |

## 6. Escalation behaviour ("above my pay grade")

This is a **structural** property, not a prompt instruction. Models self-assess confidence badly; the architecture must make the refusal mechanical.

Four escalation triggers, in priority order:

1. **No tool exists** for the required quantity → "This needs a computation the twin does not implement. Closest available: X."
2. **Tool exists but errored or did not converge** → report the failure and the diagnostic, never a fallback estimate.
3. **Parameters outside validated region** → answer *with* a prominent extrapolation flag and the distance from validated bounds; user decides.
4. **Question is Tier C (out of scope)** → decline and say plainly that a general-purpose model is the better tool for it.

Each trigger produces a distinct, templated response. None is left to the model's discretion.

## 7. Success metrics

**Primary** — hallucinated-quantity rate: 0. Any single instance is a P0 bug, not a quality regression.

**Secondary**
- Tool coverage matrix completeness (%)
- Correct abstention rate on the adversarial eval set
- Median tool-calls-per-answer for multi-step questions (measures whether it actually orchestrates or just does lookups)
- Time-to-answer for a standard sweep vs. writing the equivalent script by hand — the honest usefulness test

**Explicit non-metric** — general optics benchmark performance. Not measured, not optimized, not claimed.

## 8. Sequencing and honest risk

This is a **post-submission project.** HoloForge Part 2 has a hard Sep 30 Optics Express deadline and the Sep 10 recording-chemistry cutoff. Build begins after submission. The known pattern of generating projects faster than finishing them is the main risk to this document, and the mitigation is that dated gate, not enthusiasm.

Secondary risks:

| Risk | Mitigation |
|---|---|
| Scope creep toward "answers all optics" | §3.3 and the Tier A/B/C split are load-bearing; Tier C routing is a feature, not a gap |
| Tool wrapping is more work than expected | Phase 1 wraps only 4 tools and ships a working loop before expanding |
| Local model too weak for reliable tool-calling | Fall back to hosted model for orchestration; physics layer stays local either way (TRD §2) |
| Agent looks impressive, is never used | Eval set built from real questions asked during Part 2; if the author does not use it weekly, it failed |

## 9. Positioning

Publishable framing: *LLM-orchestrated differentiable simulation for computational holography, with structural hallucination prevention and calibrated validity bounds.* Sits at the intersection of scientific agents and computational optics — an area with real activity and no crowding at this specific intersection.

It is also a legitimate collaboration artefact: a lab can be handed the agent and query the twin without reading the codebase.
