"""propagate_angular_spectrum — exact scalar free-space propagation (no
paraxial approximation), for any z. See
holoagent/domains/waveoptics/propagation.py for the method and its
validation against the analytic Gaussian-beam w(z) divergence law.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.propagation import angular_spectrum_propagate
from holoagent.domains.waveoptics.serialization import field_from_json, field_to_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "field": {
            "type": "object",
            "properties": {"real": {"type": "array"}, "imag": {"type": "array"}},
        },
        "dx": {"type": "number", "description": "Grid spacing, microns -- must match the field's own grid"},
        "wavelength_um": {"type": "number"},
        "z_um": {"type": "number", "description": "Propagation distance, microns"},
        "n0": {"type": "number", "default": 1.0, "description": "Background refractive index"},
    },
    "required": ["field", "dx", "wavelength_um", "z_um"],
}


@register(
    name="propagate_angular_spectrum",
    json_schema=SCHEMA,
    description=(
        "Exact scalar free-space propagation by z_um (no paraxial approximation), via the angular "
        "spectrum method. Reports the evanescent-energy fraction discarded and Nyquist sampling status."
    ),
    source_module="waveoptics.propagation",
)
def propagate_angular_spectrum(field: dict, dx: float, wavelength_um: float, z_um: float, n0: float = 1.0) -> ToolResult:
    inputs = dict(field=field, dx=dx, wavelength_um=wavelength_um, z_um=z_um, n0=n0)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    result = call_adapter(
        "angular_spectrum_propagate", angular_spectrum_propagate, arr, dx, wavelength_um, z_um, n0
    )

    # Two independent, real, computed validity signals, both checked
    # against the field's OWN actual spectral content rather than a
    # blanket worst-case rule (see propagation.py's docstring on why the
    # naive dx<=wavelength/2 Nyquist check was replaced -- it flagged a
    # gently-diverging Gaussian beam as under-sampled when it numerically
    # matched its analytic w(z) to 0.01%): near-Nyquist-edge spectral
    # energy (aliasing risk) and evanescent energy (near-field detail
    # discarded), both thresholded at 1% -- an arbitrary-but-stated cutoff
    # for "negligible".
    nyquist_ok = result["nyquist_edge_energy_fraction"] < 0.01
    evanescent_ok = result["evanescent_energy_fraction"] < 0.01
    in_region = nyquist_ok and evanescent_ok

    distance = None
    diagnostics = []
    if not nyquist_ok:
        diagnostics.append(
            f"{result['nyquist_edge_energy_fraction']:.2%} of the field's spectral energy sits near the "
            f"grid's Nyquist edge (dx={dx}um) -- real aliasing risk for THIS field's actual content, "
            "not just the generic wavelength/2 worst-case bound"
        )
    if not evanescent_ok:
        diagnostics.append(
            f"{result['evanescent_energy_fraction']:.2%} of the field's spectral energy was in evanescent "
            "components and was discarded -- near-field sub-wavelength detail was lost, not preserved"
        )
    if not in_region:
        distance = max(result["nyquist_edge_energy_fraction"], result["evanescent_energy_fraction"])

    return ToolResult(
        value={
            "field": field_to_json(result["field"]),
            "evanescent_energy_fraction": result["evanescent_energy_fraction"],
            "nyquist_edge_energy_fraction": result["nyquist_edge_energy_fraction"],
        },
        units="complex field amplitude, dimensionless",
        in_validated_region=in_region,
        distance_from_bounds=distance,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic="; ".join(diagnostics) if diagnostics else None,
    )
