# HoloAgent

Twin-in-the-loop reasoning agent for computational holography. Wraps the
HoloForge Part 2 differentiable twin so that every number in an answer
traces to a tool call, not model recall.

See [docs/PRD.md](docs/PRD.md) and [docs/TRD.md](docs/TRD.md) for the full
spec, and [docs/tool_coverage_matrix.md](docs/tool_coverage_matrix.md) for
per-tool status and two corrections to the TRD found while wiring T6/T8.

## Status

- **The Sep 30 2026 / Sep 10 chemistry-cutoff gate in PRD §8 has been
  knowingly overridden** — this build started before HoloForge Part 2
  submission, at explicit user direction, against the PRD's own stated
  risk mitigation. Flagging this here so it isn't silently forgotten.
- **Phase 1 gate (TRD §8): MET for tool execution and provenance.** T3
  (`get_fit_quality`), T6 (`get_sensitivity`), T7 (`get_convergence_trace`),
  T8 (`check_validity_region`) are wired to live HoloForge
  (`C:\Users\chait\Downloads\HoloForge`, path configurable via
  `HOLOFORGE_CODE_PATH`) and tested against it — not stubs. The
  orchestrator's tool-execution loop (`holoagent/orchestrator/orchestrator.py`)
  chains dependent tool calls with full provenance
  (`tests/test_orchestrator_e2e.py`, chains 4 real calls per R6).
- **Not yet met:** the orchestrator loop has only been exercised with
  `ScriptedBackend` (a fixed script), not a live model — no
  `ANTHROPIC_API_KEY` is configured in this environment. Tool
  execution/provenance is proven; model-driven tool *selection* is not.
  Wire `AnthropicBackend` (`holoagent/orchestrator/backend.py`) with a real
  key, or another `ModelBackend`, to close this.
- T1/T2/T4/T5/T9/T10/T11/T12 remain Phase 2 stubs (`NotImplementedError`).
- Router (Tier A/B/C classification) remains a Phase 2 stub — not required
  for the Phase 1 gate.

## Layout

```
holoagent/
  router/         Tier A/B/C classification (layer 1) -- stub
  orchestrator/    Tool-calling loop, ModelBackend protocol, escalation templates (layer 2)
  tools/           Typed HoloForge-wrapping tools T1-T12 (layer 3) -- T3/T6/T7/T8 real
  physics/         Adapter to the (unmodified) HoloForge repo (layer 4 boundary)
  provenance/      SQLite-backed run store
  eval/            Set A (grounded) / Set B (abstention) / Set C (provenance audit)
docs/
  PRD.md, TRD.md, orchestrator_system_prompt.md, tool_coverage_matrix.md
tests/
```

## Setup

```bash
pip install -e ".[dev]"
pytest              # fast suite: contract/registry/store/escalation, no HoloForge compute
pytest -m slow      # also runs real HoloForge simulations through T3/T6/T7/T8 (~2 min)
```

Requires a HoloForge Part 2 checkout on disk for anything beyond the fast
suite — defaults to `C:\Users\chait\Downloads\HoloForge\HoloForge\part2\code`,
override with the `HOLOFORGE_CODE_PATH` environment variable.

For a live orchestrator run (not `ScriptedBackend`):

```bash
pip install -e ".[dev,anthropic]"
export ANTHROPIC_API_KEY=...   # not committed anywhere, set locally
```
