# HoloAgent

Twin-in-the-loop reasoning agent for computational holography. Wraps the
HoloForge Part 2 differentiable twin so that every number in an answer
traces to a tool call, not model recall. Fully self-hosted: no hosted API
key required or used — orchestration runs on a local model via Ollama.

See [docs/PRD.md](docs/PRD.md) and [docs/TRD.md](docs/TRD.md) for the full
spec, [docs/tool_coverage_matrix.md](docs/tool_coverage_matrix.md) for
per-tool status, and [docs/set_c_matching_rule.md](docs/set_c_matching_rule.md)
for the provenance-audit design.

## Status

- **The Sep 30 2026 / Sep 10 chemistry-cutoff gate in PRD §8 has been
  knowingly overridden** — this build started before HoloForge Part 2
  submission, at explicit user direction, against the PRD's own stated
  risk mitigation. Flagging this here so it isn't silently forgotten.
- **Phase 1 gate (TRD §8): MET, including model-driven tool selection.**
  T3/T6/T7/T8 wired to live HoloForge (not stubs); the orchestrator's
  tool-calling loop chains dependent calls with full provenance (R6); Set C
  proven against a real provenance chain. A real self-hosted model
  (`qwen2.5:7b-instruct` via Ollama) drives the loop end to end —
  `tests/test_ollama_backend_live.py` — not just `ScriptedBackend`.
- **Model backend: self-hosted via Ollama, not a hosted API.** TRD §2's own
  Phase 2 pick (DeepSeek-V4-Flash) and Kimi K3 (already ruled out by the
  TRD itself) both assume hardware this machine doesn't have — a 4GB VRAM
  laptop GPU (RTX 3050 Laptop). See
  `holoagent/orchestrator/ollama_backend.py`'s docstring for the hardware
  reasoning, and the same file for why `qwen2.5-coder:7b` (code-tuned) was
  tried and rejected in favor of `qwen2.5:7b-instruct` (general-instruct) —
  the coder model emitted tool calls as prose JSON instead of using
  Ollama's structured `tool_calls` field; verified directly against
  `/api/chat`, not assumed. `AnthropicBackend` still exists
  (`holoagent/orchestrator/backend.py`) for anyone who wants to compare
  against a hosted model, but is not the default and not used here.
- T1/T2/T4/T5/T9/T10/T11/T12 remain Phase 2 stubs (`NotImplementedError`).
- Router (Tier A/B/C classification) remains a Phase 2 stub — not required
  for the Phase 1 gate.
- **Set A: 51 real questions, mined from HoloForge's own development
  history** (`holoagent/eval/questions/set_a.jsonl`) — not invented. Every
  entry cites `evidence`: a specific commit or doc/comment in the
  HoloForge repo where the question was genuinely asked and answered
  during Part 2 work (dn_max citation-scoping bug, unphysical-oracle
  regression, K-grid realizability bug, RCWA-vs-Kogelnik validity
  boundary, config_hash collision risks, and more — two independent Explore
  passes over commits, `docs/*.md`, `analysis/*.py`, and test docstrings).
  **6 are `status: "reproduced"`** — re-run live this session against real
  digitized literature CSVs and real HoloForge medium configs, with fresh
  run_ids: e.g. q01's post-fix fit (NRMSE 0.286, inside the documented
  0.17-0.29 range) sits alongside q47's *pre-fix* fit on the exact same
  data (NRMSE 0.874, right at the documented 0.68-0.87 range's edge,
  D0_fit blown out to 37.5 — a physically absurd compensation for the
  real culprit, dn_max, not being free); q07 independently fits both
  digitized Bruder series and gets a 2.70x dn_max disagreement, matching
  the historically documented "2.7x" almost exactly. The rest are honestly
  split: `documented_historical` (real findings, not independently re-run)
  or `needs_phase_2_tool` (genuinely need T9/T1/T2/T4/T10/T11/T12, not yet
  wired — most of the S1/S2 ablation and multi-job sweep questions land
  here).
- **Set A grading (`run_set_a`/`grade_question`) is implemented and run
  live against the real self-hosted model**
  (`holoagent/eval/set_a_grounded.py`). Grading deliberately does NOT
  require exact equality against `reference_values` — the question text
  doesn't specify a seed/n_starts, so a model choosing a different (but
  valid) one legitimately gets a different float. It checks two things
  that don't depend on the model reading the experimenter's mind: did it
  call the tool(s) the question needs, and does Set C find zero
  provenance violations in its own final answer (R1, exercised on the
  model's real words, not a synthetic fixture). `tests/test_set_a_live_grading.py`
  found and documents a real architectural gap rather than hiding it: Set
  A's question text (phrased the way the original HoloForge finding reads,
  e.g. "For the Bayfol HX growth_dn curve...") doesn't carry the digitized
  curve data `get_fit_quality` actually needs — no tool exists yet to look
  up a named literature curve (T12 is still a stub, and even built it's
  scoped for text passages, not structured datasets). Two live tests
  confirm the right behavior on both sides of that gap: given the bare
  question text, the real local model does NOT fabricate curve data or a
  fake NRMSE (zero provenance violations); given the same finding phrased
  the way a researcher who actually has their digitized numbers in hand
  would type it, the model correctly calls `get_fit_quality` and produces
  a fully grounded answer. Both passed (455s, live Ollama + live
  HoloForge compute).
- **Set B: 43 adversarial questions across all four escalation triggers**
  (`holoagent/eval/questions/set_b.jsonl`) — 13 no-tool-exists, 6
  tool-errored, 12 outside-validated-region, 12 out-of-scope. Every entry
  is verified, not asserted: `no_tool_exists` claims are checked live
  against the actual registry (a stub really raises `NotImplementedError`
  today); `tool_errored` claims are checked live to actually raise
  `ToolError`; `outside_validated_region` claims are checked live via
  `check_validity_region` with the real `distance_from_bounds` recorded.
  `tests/test_set_b_questions.py` re-checks all of this on every test run,
  so a future tool implementation that makes a claim stale fails loudly
  instead of silently rotting.
- **Set B grading found and fixed two real bugs, and surfaced one real
  model-reliability limitation — live, not in theory:**
  - A single bad tool call (a stub's `NotImplementedError`, or a malformed
    call missing a required argument) used to **crash the entire
    orchestrator loop** instead of escalating gracefully — found via b01
    and b14 live against the real model. Fixed in `orchestrator.py`:
    broadened beyond `ToolError` to catch and gracefully escalate any
    tool-call exception.
  - Fixing that surfaced a second bug: a model quoting a real tool's own
    error text verbatim ("...missing 1 required positional argument...")
    false-positived under Set C, because **failed tool calls left no
    provenance record at all** — a real gap against TRD §4's "every tool
    invocation writes a record," not just a Set C shortcoming. Fixed
    structurally: `ProvenanceStore.record_error` persists failures too;
    see `docs/set_c_matching_rule.md`.
  - Not fixed, because it isn't a bug: `qwen2.5:7b-instruct`, asked a
    general-optics question (Brewster angle), correctly said "that's
    general optics, not this twin" — then computed and stated the actual
    angle anyway ("approximately 56.31 degrees"), unprompted, contradicting
    its own system prompt. A real instruction-following limitation of this
    7B model on the exact rule this system exists to enforce. Recorded as
    `xfail(strict=True)` in `tests/test_set_b_live_grading.py` so it stays
    a visible, tracked finding — TRD §7: "confident-wrong answers here are
    P0." This is the clearest evidence yet that a 7B self-hosted model may
    not be reliable enough for production without either a stronger model
    or a hardened system-prompt/output-filtering layer.
  All of this was found and fixed against the real live model, not
  discovered later — 5 passed, 1 documented xfail, 341s.

## Runtime evidence validator (the master-prompt's "Hard Scientific Safety Rule")

Set C used to be an offline audit only — something Set A/B grading and CI
ran *after the fact*. It's now also a **runtime gate**: every draft final
answer the orchestrator produces is checked against Set C's audit
*before it can reach the caller*
(`Orchestrator._validate_and_finalize`, `holoagent/orchestrator/orchestrator.py`).
If the draft contains a numeric claim not grounded in that session's real
provenance, the orchestrator refuses to return it — deterministically, no
extra model call — and returns a fixed abstention message instead, citing
which token(s) were unsupported. The block itself is provenance-recorded
(`ProvenanceStore.record_validator_block`), so "the validator fired" is
itself an auditable event, not a side effect.

This is a structural fix, not a bigger prompt: a fabricated number cannot
reach the user even when the model ignores its system prompt — which
happened, live, during Set B testing (`qwen2.5:7b-instruct` computed a
Brewster angle unprompted right after correctly declining the question in
prose). Design choice: **block-and-abstain, not block-and-regenerate** —
retrying against the same 7B model that just demonstrated it doesn't
reliably self-correct on this exact rule isn't a safe default, and
abstention is explicitly licensed as an alternative in the spec this was
built against. See `tests/test_evidence_validator.py`.

## Master-build scope note

A later, much larger design document ("MASTER BUILD PROMPT") asked for
HoloAgent to become a general-purpose computational-optics research
platform — a plugin architecture separating agent core from domain tools,
a full general wave-optics/Fourier-optics/imaging/phase-retrieval tool
library validated against analytic reference cases, a claim validator
beyond numeric tokens, research sessions, and automatic hypothesis
testing. That document's own §37 says not to build it all at once, and
this project has followed that literally rather than faking broad
progress. Done so far:

- **Milestone 1** (harden current system) — the runtime evidence-validator
  gate (§3), provenance-recorded failures, real Set A/B/C.
- **Milestone 2** (complete HoloForge tools) — all 12 tools real, each
  rescoped to HoloForge's actual capability with documented caveats.
- **Milestone 3** (generalize architecture) — `holoagent/domains/` plugin
  separation; core no longer knows HoloForge exists.
- **Milestone 4, started** — a second domain,
  `holoagent/domains/waveoptics/` (general scalar wave optics,
  independent of HoloForge per §12's explicit instruction). 9 tools
  (fields, propagation, aperture, thin lens, grating), 4 real analytic
  validations (Fraunhofer sinc² diffraction, Gaussian-beam w(z) divergence
  via the exact angular-spectrum method, the Fourier-transforming-lens
  identity, sinusoidal-grating diffraction orders) — see
  `docs/tool_coverage_matrix.md`'s Milestone 4 section for exactly what's
  covered and what's explicitly not (no 2D circular aperture/Airy
  pattern, no general/arbitrary phase masks, imaging/coherence/
  phase-retrieval yet). Deliberately not auto-loaded into the default
  HoloForge-focused orchestrator session — using it means an explicit
  `import holoagent.domains.waveoptics`.

Milestones 5-8 (imaging, phase retrieval/holography as a general
capability, research knowledge, a full research agent) are not started —
still a multi-month undertaking, not something a handful of sessions
closes out. Treat the "Status" section above as the accurate current
HoloForge-domain state; this note is what's true for the master-build
document's own broader vision specifically.

## Layout

Master-build Milestone 3 (plugin architecture): the generic core no
longer knows anything about HoloForge. Domain-specific tools live under
`domains/`, one self-contained plugin package per domain; the core
imports nothing from there, only the reverse. HoloForge is currently the
only domain — a future one (e.g. general wave optics) would be a sibling
package under `domains/`, without touching anything below.

```
holoagent/
  router/          Tier A/B/C classification (layer 1) -- stub
  orchestrator/     Tool-calling loop, ModelBackend protocol (backend.py),
                    Ollama backend (ollama_backend.py), escalation templates,
                    the runtime evidence-validator gate (layer 2) -- CORE
  tools/            The tool contract (contract.py) and registry (registry.py)
                    every domain's tools register into -- CORE, domain-agnostic
  provenance/       SQLite-backed run store, session-scoped store context -- CORE
  eval/             Set A (grounded) / Set B (abstention) / Set C (provenance
                    audit) -- CORE (eval/questions/*.jsonl is HoloForge-domain
                    data, not core, but hasn't been relocated yet)
  domains/
    holoforge/      The holoforge domain plugin -- ALL 12 tools (T1-T12),
                    real, tested against live HoloForge
      adapter.py     The one crossing point into the frozen HoloForge repo
      physics.py      get_default_adapter() singleton
      literature_corpus.py  Plain keyword search backing T12
      tools/          t1_phase_map.py ... t12_retrieve_literature.py
docs/
  PRD.md, TRD.md, orchestrator_system_prompt.md, tool_coverage_matrix.md,
  set_c_matching_rule.md
tests/
```

## Setup

```bash
pip install -e ".[dev]"
pytest              # fast suite: contract/registry/store/escalation/Set C, no compute
pytest -m slow      # real HoloForge simulations + a real local model (~10 min)
```

Requires a HoloForge Part 2 checkout on disk for anything beyond the fast
suite — defaults to `C:\Users\chait\Downloads\HoloForge\HoloForge\part2\code`,
override with the `HOLOFORGE_CODE_PATH` environment variable.

For the live orchestrator test (`tests/test_ollama_backend_live.py`), a
local Ollama server must be running with `qwen2.5:7b-instruct` pulled:

```bash
ollama serve                      # if not already running
ollama pull qwen2.5:7b-instruct
```

That test is skipped automatically (not failed) if no Ollama server is
reachable at `localhost:11434`, so the rest of the suite doesn't depend on
a local model being up.
