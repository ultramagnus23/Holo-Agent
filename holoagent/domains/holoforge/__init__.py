"""The holoforge domain plugin: computational holography, wrapping the
frozen HoloForge Part 2 differentiable twin.

Importing this package registers T1-T12 with holoagent.tools.registry
(via each tool module's own @register decorator) as a side effect --
that's the entire plugin contract. holoagent's core code (orchestrator,
provenance, router, eval) never imports this package directly; only
holoagent/tools/__init__.py does, for backward compatibility (see that
module's docstring), and application entry points that want this domain
explicitly.

Layout:
  adapter.py            HoloForgeAdapter -- the one crossing point into
                         the frozen HoloForge repo (TRD S1 layer 4).
  physics.py             get_default_adapter() -- the process-wide
                         singleton every tool in tools/ calls through.
  literature_corpus.py   Plain keyword search over HoloForge's own real
                         docs, backing T12.
  tools/                 T1-T12, one module each, each a thin typed
                         wrapper that calls through physics.py's adapter
                         and never imports HoloForge directly itself.
"""

from holoagent.domains.holoforge.tools import (  # noqa: F401  (imported for registration side effects)
    t1_phase_map,
    t2_diffraction_efficiency,
    t3_fit_quality,
    t4_medium_state,
    t5_reconstruct_wavefront,
    t6_sensitivity,
    t7_convergence_trace,
    t8_validity_region,
    t9_parameter_sweep,
    t10_query_run_store,
    t11_compare_runs,
    t12_retrieve_literature,
)
