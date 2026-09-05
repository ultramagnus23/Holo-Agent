"""calculate_phase — arg(field), in radians. A deterministic derived
quantity (master-build §23), always valid by construction, same rationale
as calculate_intensity.
"""

from __future__ import annotations

import numpy as np

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
    name="calculate_phase",
    json_schema=SCHEMA,
    description="Phase arg(field), in radians, of a complex field.",
    source_module="waveoptics (deterministic derived quantity)",
)
def calculate_phase(field: dict) -> ToolResult:
    inputs = dict(field=field)
    input_hash = compute_input_hash(inputs)

    arr = call_adapter("field_from_json", field_from_json, field.get("real", []), field.get("imag", []))
    phase = call_adapter("angle(field)", np.angle, arr)

    return ToolResult(
        value={"phase_rad": phase.tolist()},
        units="radians",
        in_validated_region=True,
        distance_from_bounds=None,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=None,
    )
