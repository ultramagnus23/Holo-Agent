"""Optical elements: pure amplitude masks applied to a field (master-build
§13 Stage 3 -- deliberately just the minimal slice needed to validate a
real diffraction pattern this session; see docs/tool_coverage_matrix.md
for what's explicitly not built yet: lenses, phase masks, gratings, and a
true 2D circular aperture (this domain is 1D transverse fields only,
matching the holoforge domain's own convention -- a 1D "circular" slit of
some half-width would NOT reproduce the Airy pattern the master prompt's
§28 names, since that needs real 2D radial symmetry; rather than build a
mislabeled stand-in, that validation case is not attempted here at all).
"""

from __future__ import annotations

import numpy as np

from holoagent.domains.waveoptics.fields import make_x_axis


def apply_rect_aperture(field: np.ndarray, dx: float, width_um: float) -> np.ndarray:
    """A hard-edged 1D slit of the given width, centered on the grid."""
    n_x = len(field)
    x = make_x_axis(n_x, dx)
    mask = (np.abs(x) <= width_um / 2).astype(complex)
    return field * mask
