# Tool coverage matrix

Checked-in artefact per TRD §3.1. A quantity is not "done" until all six
columns are true. Update this file whenever a tool's status changes —
it is the source of truth for R2 (PRD §5).

| # | Tool | Phase | computed | exposed | typed | tested | documented | validity-bounded |
|---|---|---|---|---|---|---|---|---|
| T1 | `get_phase_map` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T2 | `get_diffraction_efficiency` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T3 | `get_fit_quality` | 1 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T4 | `get_medium_state` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T5 | `reconstruct_wavefront` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T6 | `get_sensitivity` | 1 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T7 | `get_convergence_trace` | 1 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T8 | `check_validity_region` | 1 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T9 | `run_parameter_sweep` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T10 | `query_run_store` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T11 | `compare_runs` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |
| T12 | `retrieve_literature` | 2 | ✗ | ✓ (stub) | ✓ | ✗ | ✓ | ✗ |

**computed** — HoloForge actually computes this quantity and the adapter
call into it works.
**exposed** — registered in `holoagent.tools.registry` with a JSON schema
(all 12 stubs meet this now).
**typed** — returns `ToolResult`, raises `ToolError`/`MissingParameterError`
per the contract in `holoagent/tools/contract.py` (all 12 stubs meet this
now, modulo the `NotImplementedError` body).
**tested** — has a passing test in `tests/` exercising real HoloForge output,
not just the stub's `NotImplementedError`.
**documented** — has a docstring stating what it returns and its source
module (all 12 stubs meet this now).
**validity-bounded** — `in_validated_region` / `distance_from_bounds` are
populated from the real validity model, not placeholder values.

Phase 1 gate (TRD §8): T3, T6, T7, T8 reach `computed` + `tested` +
`validity-bounded` = true, backed by a real `HoloForgeAdapter`, with one
real multi-step question answered end to end through the provenance store.
