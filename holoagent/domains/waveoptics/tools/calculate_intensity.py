"""calculate_intensity — |field|^2. A deterministic derived quantity
(master-build §23: legitimate deterministic transforms of a field are not
"model invention"), always valid by construction -- there's no bound to
be outside of for a pure elementwise transform of already-provenanced
field data.
"""

from __future__ import annotations

from holoagent.domains.waveoptics.serialization import field_from_json
from holoagent.tools._util import call_adapter, compute_input_hash, new_run_id
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "field": {
            "type": "object",
            "properties": {"real": {"type": "array"}, "imag": {"type": "array"}},
            "description": "A complex field, as returned by another waveoptics tool",
        },
    },
    "required": ["field"],
}


@register(
    name="calculate_intensity",
    json_schema=SCHEMA,
    description="Intensity |field|^2 of a complex field.",
    source_module="waveoptics (deterministic derived quantity)",
)
def calculate_intensity(field: dict) -> ToolResult:
    inputs = dict(field=field)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    intensity = call_adapter("abs(field)**2", lambda a: (a.real**2 + a.imag**2), arr)

    return ToolResult(
        value={"intensity": intensity.tolist()},
        units="dimensionless (|E|^2)",
        in_validated_region=True,
        distance_from_bounds=None,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=None,
    )
