"""The orchestrator (TRD §1, layer 2): an LLM that selects and sequences
tools and sees tool RESULTS only — never raw model recall for numbers.

This is the architectural invariant the whole system depends on (TRD §1):
the orchestrator's context contains no physical quantity except tool
outputs. The system prompt in docs/orchestrator_system_prompt.md is a
secondary guardrail only; Set C (TRD §7) is what actually enforces this.

Model choice is a pluggable backend (holoagent.orchestrator.backend,
ModelBackend protocol). The default target is self-hosted/local
(holoagent.orchestrator.ollama_backend.OllamaBackend) — not a hosted API
key; see that module's docstring for why (hardware-driven, not the TRD's
original Phase 1/2 split). This module never hardcodes a provider: it
builds a provider-neutral Turn history (backend.Turn) and lets each
backend translate to its own wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from holoagent.orchestrator.backend import (
    AssistantToolCallTurn,
    ModelBackend,
    ToolResultTurn,
    Turn,
    UserTurn,
)
from holoagent.orchestrator.escalation import no_tool_exists, outside_validated_region, tool_errored

# out_of_scope (trigger 4) is a router-level decision (TRD §1 layer 1), not
# something the tool-execution loop below discovers — it belongs in
# holoagent.router once Tier A/B/C classification exists (TRD §8 Phase 2).
from holoagent.provenance.store import ProvenanceStore
from holoagent.tools.contract import ToolError
from holoagent.tools.registry import ToolSpec, get_tool, list_tools

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "docs" / "orchestrator_system_prompt.md"


@dataclass(frozen=True)
class OrchestratorAnswer:
    text: str
    run_ids: list[str] = field(default_factory=list)  # provenance chain, in call order
    escalated: bool = False  # True iff any tool call hit an escalation trigger


class Orchestrator:
    def __init__(
        self,
        provenance_store: ProvenanceStore,
        model_backend: ModelBackend,
        max_steps: int = 12,
    ):
        self._store = provenance_store
        self._model_backend = model_backend
        self._max_steps = max_steps
        self._tools: list[ToolSpec] = list_tools()

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    def _tool_specs(self) -> list[dict[str, Any]]:
        """Provider-neutral tool spec shape (name/description/input_schema).
        Each backend translates this to its own wire format
        (AnthropicBackend, OllamaBackend) since the registry's JSON Schema
        itself is already provider-agnostic.
        """
        return [
            {"name": spec.name, "description": spec.description, "input_schema": spec.json_schema}
            for spec in self._tools
        ]

    def answer(self, query: str) -> OrchestratorAnswer:
        history: list[Turn] = [UserTurn(text=query)]
        tool_specs = self._tool_specs()
        parent_run_id: str | None = None
        run_ids: list[str] = []
        escalated = False

        for _ in range(self._max_steps):
            step = self._model_backend.step(self.system_prompt, history, tool_specs)

            if not step.tool_calls:
                if step.final_text is None:
                    raise RuntimeError("ModelBackend returned neither tool_calls nor final_text")
                return OrchestratorAnswer(text=step.final_text, run_ids=run_ids, escalated=escalated)

            history.append(AssistantToolCallTurn(tool_calls=step.tool_calls))
            tool_results: list[dict[str, str]] = []

            for call in step.tool_calls:
                try:
                    spec = get_tool(call.name)
                except KeyError:
                    escalated = True
                    tool_results.append(
                        {"tool_call_id": call.id, "text": no_tool_exists(call.name, closest_tool=None)}
                    )
                    continue

                try:
                    result = spec.fn(**call.arguments)
                except ToolError as exc:
                    escalated = True
                    tool_results.append({"tool_call_id": call.id, "text": tool_errored(call.name, str(exc))})
                    continue

                run_id = self._store.record(
                    tool=call.name, inputs=call.arguments, result=result, parent_run_id=parent_run_id
                )
                parent_run_id = run_id
                run_ids.append(run_id)

                text = (
                    f"run_id={run_id} value={result.value!r} units={result.units!r} "
                    f"in_validated_region={result.in_validated_region}"
                )
                if result.diagnostic:
                    text += f" diagnostic={result.diagnostic!r}"
                if not result.in_validated_region:
                    escalated = True
                    violation_detail = (
                        result.value.get("violations", result.value)
                        if isinstance(result.value, dict)
                        else result.value
                    )
                    text += "\n" + outside_validated_region(result.distance_from_bounds, violation_detail)
                tool_results.append({"tool_call_id": call.id, "text": text})

            history.append(ToolResultTurn(results=tool_results))

        raise RuntimeError(
            f"Orchestrator.answer: exceeded max_steps={self._max_steps} without a final answer "
            f"(run_ids so far: {run_ids})"
        )
