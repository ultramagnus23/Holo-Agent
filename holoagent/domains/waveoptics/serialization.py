"""JSON <-> complex-array field representation. JSON has no complex
number type, so every field crossing a tool boundary is
{"real": [...], "imag": [...]} -- the same "inline the array" approach
(and the same TRD §4 scaling caveat: fine for the small 1D grids these
tools are tested against, large fields belong on disk with a path
reference, not built yet) already used for T1/T2's spatial fields in the
holoforge domain.
"""

from __future__ import annotations

import numpy as np


def field_to_json(field: np.ndarray) -> dict:
    return {"real": field.real.tolist(), "imag": field.imag.tolist()}


def field_from_json(real: list, imag: list) -> np.ndarray:
    real_arr = np.asarray(real, dtype=float)
    imag_arr = np.asarray(imag, dtype=float)
    if real_arr.shape != imag_arr.shape:
        raise ValueError(f"field real/imag length mismatch: {real_arr.shape} vs {imag_arr.shape}")
    return real_arr + 1j * imag_arr
