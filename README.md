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
  here). Set A *grading* (`run_set_a`) is still not implemented — it needs
  a real `ModelBackend` run over these questions compared against
  `reference_values`, which only the 6 "reproduced" entries currently
  have. `tests/test_set_a_questions.py` enforces the structural honesty
  rules (every question has real evidence, every named tool actually
  exists in the registry).
- Set B (adversarial abstention, TRD §7) is still empty.

## Layout

```
holoagent/
  router/         Tier A/B/C classification (layer 1) -- stub
  orchestrator/    Tool-calling loop, ModelBackend protocol (backend.py),
                   Ollama backend (ollama_backend.py), escalation templates (layer 2)
  tools/           Typed HoloForge-wrapping tools T1-T12 (layer 3) -- T3/T6/T7/T8 real
  physics/         Adapter to the (unmodified) HoloForge repo (layer 4 boundary)
  provenance/      SQLite-backed run store
  eval/            Set A (grounded) / Set B (abstention) / Set C (provenance audit)
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
