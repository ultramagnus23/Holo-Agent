"""Pluggable model backend (TRD §2: hosted API in Phase 1, local
llama.cpp/GGUF in Phase 2+). The orchestrator loop in orchestrator.py is
written against this Protocol only, using a provider-neutral Turn history
— it never assumes a specific wire format, so swapping backends does not
touch the loop, and each backend owns its own translation to its actual
API shape.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Union


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


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantToolCallTurn:
    tool_calls: list[ToolCall]


@dataclass(frozen=True)
class ToolResultTurn:
    results: list[dict[str, str]]  # each: {"tool_call_id": str, "text": str}


Turn = Union[UserTurn, AssistantToolCallTurn, ToolResultTurn]


class ModelBackend(Protocol):
    def step(self, system_prompt: str, history: list[Turn], tool_specs: list[dict[str, Any]]) -> ModelStep: ...


class ScriptedBackend:
    """A deterministic backend that plays back a fixed script of steps,
    ignoring the actual message history. Not a mock of a real model's
    behavior — a fixture for exercising the orchestrator loop (tool
    execution, provenance chaining, escalation handling) without a live
    model. Real Set A/B grading needs a real backend; this only proves
    the loop's plumbing is correct.
    """

    def __init__(self, script: list[ModelStep]):
        self._script = list(script)
        self._i = 0

    def step(self, system_prompt: str, history: list[Turn], tool_specs: list[dict[str, Any]]) -> ModelStep:
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
    """Hosted-API backend. Requires the `anthropic` package and
    ANTHROPIC_API_KEY to be set — neither is assumed present; construction
    fails loudly (no silent fallback) if either is missing, per the same
    no-silent-defaults principle as the tool contract (TRD §3.2).

    NOT the default anymore (see holoagent/orchestrator/ollama_backend.py):
    this project runs on a self-hosted free model, not a hosted API key.
    Kept available for anyone who wants to compare against a hosted model.
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

    def _to_anthropic_messages(self, history: list[Turn]) -> list[dict]:
        messages: list[dict] = []
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.text})
            elif isinstance(turn, AssistantToolCallTurn):
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                            for tc in turn.tool_calls
                        ],
                    }
                )
            elif isinstance(turn, ToolResultTurn):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": r["tool_call_id"], "content": r["text"]}
                            for r in turn.results
                        ],
                    }
                )
            else:
                raise TypeError(f"unknown turn type: {type(turn)}")
        return messages

    def step(self, system_prompt: str, history: list[Turn], tool_specs: list[dict]) -> ModelStep:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system_prompt,
            messages=self._to_anthropic_messages(history),
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
