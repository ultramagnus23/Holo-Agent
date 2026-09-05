"""The waveoptics domain plugin: general scalar wave optics (master-build
Milestone 4), independent of HoloForge -- master-build §12 is explicit
that HoloForge must not become a general optics engine; this is a
separate domain package, a sibling of domains/holoforge/, not a
dependency of or on it.

Scope, deliberately narrow (master-build §38: depth over feature count;
§37: do not build the whole system at once). Only what's implemented AND
validated against a closed-form analytic solution is registered as a
tool -- see docs/tool_coverage_matrix.md's Milestone 4 section for
exactly what that covers today and what it explicitly does not (no true
2D circular aperture / Airy-pattern case yet -- this domain is 1D
transverse fields only, matching the holoforge domain's own convention;
no lenses, gratings, phase masks, imaging, coherence, or phase retrieval
yet).

Layout:
  fields.py         create_plane_wave / create_gaussian_beam physics
  propagation.py     angular_spectrum_propagate (exact) /
                     propagate_fraunhofer (far-field) physics
  elements.py        apply_aperture (rect only) physics
  serialization.py   JSON <-> complex-array field representation
  tools/             T1-T7 (waveoptics numbering restarts at its own T1;
                     these are NOT the same T-numbers as the holoforge
                     domain's T1-T12 -- tool identity is by registered
                     name, not number, across domains)
"""

from holoagent.domains.waveoptics.tools import (  # noqa: F401  (registration side effects)
    create_plane_wave,
    create_gaussian_beam,
    calculate_intensity,
    calculate_phase,
    apply_aperture,
    propagate_angular_spectrum,
    propagate_fraunhofer,
)
