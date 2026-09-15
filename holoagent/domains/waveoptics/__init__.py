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
no general/arbitrary phase masks, N>2 multi-slit arrays, 2D
imaging/circular pupils, or coherence yet; phase retrieval now covers
both the two-intensity (GS) and single-pattern-plus-support (HIO)
problem formulations).

Layout:
  fields.py         create_plane_wave / create_gaussian_beam physics
  propagation.py     angular_spectrum_propagate (exact) /
                     propagate_fraunhofer (far-field) physics
  elements.py        apply_aperture (rect only) / apply_thin_lens /
                     apply_grating / apply_double_slit physics
  imaging.py          apply_fourier_filter (idealized 4f-system
                     lowpass/highpass spatial-frequency filter) /
                     compute_incoherent_psf_otf (Milestone 5: PSF/OTF/MTF
                     of a diffraction-limited incoherent imaging system) /
                     image_through_incoherent_system (convolve a real
                     object intensity through that system) physics
  noise.py            add_shot_noise (Poisson/shot noise) /
                     add_read_noise (additive Gaussian sensor noise) --
                     the noiseless image_through_incoherent_system output
                     is not what any real detector measures
  phase_retrieval.py  gerchberg_saxton (Milestone 6: classic two-
                     intensity iterative phase retrieval) / fienup_hio
                     (single-pattern-plus-support-constraint phase
                     retrieval, a genuinely different problem) physics
  serialization.py   JSON <-> complex-array field representation
  tools/             T1-T17 (waveoptics numbering restarts at its own T1;
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
    apply_thin_lens,
    apply_grating,
    apply_fourier_filter,
    apply_double_slit,
    compute_incoherent_psf_otf,
    image_through_incoherent_system,
    add_shot_noise,
    add_read_noise,
    retrieve_phase_gerchberg_saxton,
    retrieve_phase_hio,
)
