"""HoloForge-wrapping tools. Import this package to populate the registry.

Per TRD §5 streamlining rule, the registry (holoagent.tools.registry) is the
only public surface the orchestrator sees — no other path into the physics
core is exposed.
"""

from holoagent.tools import (  # noqa: F401  (imported for registration side effects)
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
from holoagent.tools.registry import get_tool, list_tools  # noqa: F401
