# HoloAgent

Twin-in-the-loop reasoning agent for computational holography. Wraps the
HoloForge Part 2 differentiable twin so that every number in an answer
traces to a tool call, not model recall.

See [docs/PRD.md](docs/PRD.md) and [docs/TRD.md](docs/TRD.md) for the full
spec. This repo is currently a **structural scaffold, pre-implementation**
(TRD §8 Phase 0/1) — every tool in `holoagent/tools/` raises
`NotImplementedError` until wired to a real HoloForge checkout.

## Status

- Gate: HoloForge Part 2 submission to Optica Express, **Sep 30 2026**.
  Nothing here starts before that (PRD §8).
- Phase 1 target (TRD §8): tools T3 / T6 / T7 / T8 + provenance store +
  hosted orchestrator, one real multi-step question answered end to end
  with full provenance.

## Layout

```
holoagent/
  router/         Tier A/B/C classification (layer 1)
  orchestrator/    Tool-calling LLM loop (layer 2)
  tools/           Typed HoloForge-wrapping tools T1-T12 (layer 3)
  physics/         Adapter to the frozen HoloForge repo (layer 4 boundary)
  provenance/      SQLite-backed run store
  eval/            Set A (grounded) / Set B (abstention) / Set C (provenance audit)
docs/
  PRD.md, TRD.md, orchestrator_system_prompt.md, tool_coverage_matrix.md
tests/
```

## Setup

```bash
pip install -e ".[dev]"
pytest
```

Tests currently cover the parts with real logic: the tool contract, the
tool registry, and the provenance store. The tool stubs themselves are
intentionally untested beyond "does the registry see them" — there is
nothing to test until `holoagent/physics/holoforge_adapter.py` is wired to
a real HoloForge checkout.
