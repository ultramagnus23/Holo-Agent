"""The orchestrator (TRD §1, layer 2): an LLM that selects and sequences
tools and sees tool RESULTS only — never raw model recall for numbers.

This is the architectural invariant the whole system depends on (TRD §1):
the orchestrator's context contains no physical quantity except tool
outputs. The system prompt in docs/orchestrator_system_prompt.md is a
secondary guardrail only; Set C (TRD §7) is what actually enforces this.

Model choice is a pluggable backend (TRD §2): hosted API in Phase 1,
llama.cpp/GGUF local in Phase 2+. This module should not hardcode a
provider.
"""

from __future__ import annotations

from pathlib import Path

from holoagent.provenance.store import ProvenanceStore
from holoagent.tools.registry import ToolSpec, list_tools

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent.parent / "docs" / "orchestrator_system_prompt.md"


class EscalationTrigger:
    """The four escalation triggers (PRD §6). Each produces a distinct,
    templated response — never left to model discretion."""

    NO_TOOL_EXISTS = "no_tool_exists"
    TOOL_ERRORED = "tool_errored"
    OUTSIDE_VALIDATED_REGION = "outside_validated_region"
    OUT_OF_SCOPE = "out_of_scope"


class Orchestrator:
    def __init__(self, provenance_store: ProvenanceStore, model_backend: object | None = None):
        self._store = provenance_store
        self._model_backend = model_backend  # pluggable: hosted API (P1) or local llama.cpp (P2+)
        self._tools: list[ToolSpec] = list_tools()

    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    def answer(self, query: str) -> str:
        raise NotImplementedError(
            "Orchestrator.answer: tool-calling loop not yet implemented. "
            "Phase 1 target (TRD §8): one real multi-step question answered "
            "end to end with full provenance, using only T3/T6/T7/T8."
        )
