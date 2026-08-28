"""The tool registry: the only public surface the orchestrator sees.

TRD §5 streamlining rule: anything not reachable from this registry is
deleted, not deprecated. Register tools with @register(name, schema) and
they become orchestrator-callable and coverage-matrix-trackable in one step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from holoagent.tools.contract import ToolResult

ToolFn = Callable[..., ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    fn: ToolFn
    json_schema: dict[str, Any]
    description: str
    source_module: str


_REGISTRY: dict[str, ToolSpec] = {}


def register(
    name: str,
    json_schema: dict[str, Any],
    description: str,
    source_module: str,
) -> Callable[[ToolFn], ToolFn]:
    def decorator(fn: ToolFn) -> ToolFn:
        if name in _REGISTRY:
            raise ValueError(f"tool already registered: {name}")
        _REGISTRY[name] = ToolSpec(
            name=name,
            fn=fn,
            json_schema=json_schema,
            description=description,
            source_module=source_module,
        )
        return fn

    return decorator


def get_tool(name: str) -> ToolSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"no such tool: {name}") from exc


def list_tools() -> list[ToolSpec]:
    return list(_REGISTRY.values())
