"""Pluggable model backend (TRD §2: hosted API in Phase 1, local
llama.cpp/GGUF in Phase 2+). The orchestrator loop in orchestrator.py is
written against this Protocol only — it never imports a specific provider
SDK directly, so swapping backends does not touch the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ModelStep:
    """Exactly one of tool_calls / final_text is meaningful:
    tool_calls non-empty -> orchestrator executes them and continues the loop.
    final_text is not None and tool_calls is empty -> the model is done.
    """

    tool_calls: list[ToolCall]
    final_text: str | None


class ModelBackend(Protocol):
    def step(self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]) -> ModelStep: ...


class ScriptedBackend:
    """A deterministic backend that plays back a fixed script of steps,
    ignoring the actual message history. Not a mock of a real model's
    behavior — a fixture for exercising the orchestrator loop (tool
    execution, provenance chaining, escalation handling) without a live
    API key. Real Set A/B grading needs a real backend; this only proves
    the loop's plumbing is correct.
    """

    def __init__(self, script: list[ModelStep]):
        self._script = list(script)
        self._i = 0

    def step(self, messages: list[dict[str, Any]], tool_specs: list[dict[str, Any]]) -> ModelStep:
        if self._i >= len(self._script):
            raise RuntimeError("ScriptedBackend: script exhausted before a final_text step")
        step = self._script[self._i]
        self._i += 1
        return step


def anthropic_backend_available() -> bool:
    import os

    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return "ANTHROPIC_API_KEY" in os.environ


class AnthropicBackend:
    """Hosted-API backend (TRD §2 Phase 1 recommendation). Requires the
    `anthropic` package and ANTHROPIC_API_KEY to be set — neither is
    assumed present; construction fails loudly (no silent fallback) if
    either is missing, per the same no-silent-defaults principle as the
    tool contract (TRD §3.2).
    """

    def __init__(self, model: str = "claude-sonnet-4-5"):
        import os

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicBackend requires the 'anthropic' package (pip install anthropic)"
            ) from exc
        if "ANTHROPIC_API_KEY" not in os.environ:
            raise RuntimeError("AnthropicBackend requires ANTHROPIC_API_KEY to be set")
        self._client = anthropic.Anthropic()
        self._model = model

    def step(self, messages: list[dict], tool_specs: list[dict]) -> ModelStep:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=messages,
            tools=tool_specs,
        )
        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        if tool_calls:
            return ModelStep(tool_calls=tool_calls, final_text=None)
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return ModelStep(tool_calls=[], final_text="".join(text_blocks))
