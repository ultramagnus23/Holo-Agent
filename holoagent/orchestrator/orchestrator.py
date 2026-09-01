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
from holoagent.orchestrator.escalation import (
    evidence_validator_blocked,
    no_tool_exists,
    outside_validated_region,
    tool_errored,
)

# out_of_scope (trigger 4) is a router-level decision (TRD §1 layer 1), not
# something the tool-execution loop below discovers — it belongs in
# holoagent.router once Tier A/B/C classification exists (TRD §8 Phase 2).
from holoagent.eval._transcript import transcript_from_store
from holoagent.eval.set_c_provenance_audit import audit_answer
from holoagent.provenance.store import ProvenanceStore
from holoagent.tools.contract import ToolError
from holoagent.tools.registry import ToolSpec, get_tool, list_tools

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "docs" / "orchestrator_system_prompt.md"


@dataclass(frozen=True)
class OrchestratorAnswer:
    text: str
    run_ids: list[str] = field(default_factory=list)  # successful, result-bearing calls, in call order
    error_run_ids: list[str] = field(default_factory=list)  # failed calls (ProvenanceStore.record_error), in call order
    escalated: bool = False  # True iff any tool call hit an escalation trigger
    validator_blocked: bool = False  # True iff the runtime evidence validator refused the model's draft (master-prompt §3)


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

    def _validate_and_finalize(
        self,
        draft_text: str,
        *,
        query: str,
        run_ids: list[str],
        error_run_ids: list[str],
        escalated: bool,
        parent_run_id: str | None,
    ) -> OrchestratorAnswer:
        """The runtime evidence-validator gate (master-prompt §3: "Hard
        Scientific Safety Rule" -- do not rely solely on the system prompt).
        Every draft final answer is checked against Set C (the same audit
        used offline for Sets A/B/C) BEFORE it can reach the caller. This is
        deliberately block-and-abstain, not block-and-regenerate: retrying
        against the same model risks looping (and this project has already
        measured, live, that a 7B local model does not reliably self-correct
        on this exact rule -- see tests/test_set_b_live_grading.py's
        documented Brewster-angle xfail). Abstention is explicitly permitted
        as an alternative to regeneration in the master prompt's own wording
        ("regenerate or produce an abstention"), and it's the deterministic,
        testable, zero-extra-model-call option.
        """
        transcript = transcript_from_store(self._store, query, run_ids, error_run_ids)
        violations = audit_answer(draft_text, transcript)

        if not violations:
            return OrchestratorAnswer(
                text=draft_text, run_ids=run_ids, error_run_ids=error_run_ids, escalated=escalated
            )

        unsupported_tokens = [v.token for v in violations]
        block_run_id = self._store.record_validator_block(
            draft_text=draft_text, unsupported_tokens=unsupported_tokens, parent_run_id=parent_run_id
        )
        return OrchestratorAnswer(
            text=evidence_validator_blocked(unsupported_tokens, block_run_id),
            run_ids=run_ids,
            error_run_ids=[*error_run_ids, block_run_id],
            escalated=True,
            validator_blocked=True,
        )

    def answer(self, query: str) -> OrchestratorAnswer:
        history: list[Turn] = [UserTurn(text=query)]
        tool_specs = self._tool_specs()
        parent_run_id: str | None = None
        run_ids: list[str] = []
        error_run_ids: list[str] = []
        escalated = False

        for _ in range(self._max_steps):
            step = self._model_backend.step(self.system_prompt, history, tool_specs)

            if not step.tool_calls:
                if step.final_text is None:
                    raise RuntimeError("ModelBackend returned neither tool_calls nor final_text")
                return self._validate_and_finalize(
                    step.final_text,
                    query=query,
                    run_ids=run_ids,
                    error_run_ids=error_run_ids,
                    escalated=escalated,
                    parent_run_id=parent_run_id,
                )

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
                    error_text = str(exc)
                    error_run_id = self._store.record_error(
                        tool=call.name, inputs=call.arguments, error_text=error_text, parent_run_id=parent_run_id
                    )
                    parent_run_id = error_run_id
                    error_run_ids.append(error_run_id)
                    tool_results.append(
                        {"tool_call_id": call.id, "text": f"run_id={error_run_id} " + tool_errored(call.name, error_text)}
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    # Catches everything call_adapter's ToolError-wrapping
                    # doesn't: a still-stubbed tool's own NotImplementedError
                    # (T1/T2/T4/T5/T9-T12 raise this directly, not via
                    # call_adapter), and a malformed call (TypeError for a
                    # missing/unexpected argument -- the model's own mistake,
                    # not HoloForge's). Either way, a single bad tool call
                    # must feed back into the loop as a self-describing
                    # failure, never crash Orchestrator.answer() entirely --
                    # found live via Set B (b01/b14) crashing the whole
                    # session on exactly these two cases.
                    escalated = True
                    error_text = f"{type(exc).__name__}: {exc}"
                    error_run_id = self._store.record_error(
                        tool=call.name, inputs=call.arguments, error_text=error_text, parent_run_id=parent_run_id
                    )
                    parent_run_id = error_run_id
                    error_run_ids.append(error_run_id)
                    tool_results.append(
                        {"tool_call_id": call.id, "text": f"run_id={error_run_id} " + tool_errored(call.name, error_text)}
                    )
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
