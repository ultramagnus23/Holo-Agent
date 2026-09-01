"""Shared helpers for tool implementations. Not part of the public registry
surface — internal to holoagent.tools.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def call_adapter(description: str, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Calls an adapter/HoloForge function, wrapping any unexpected
    exception as a ToolError so it reaches the orchestrator as escalation
    trigger 2 (PRD §6: "tool errored... report the failure and diagnostic,
    never a fallback estimate") instead of an unhandled exception crashing
    Orchestrator.answer() entirely. ToolError/MissingParameterError raised
    deliberately by tool code (e.g. a missing required field) pass through
    unchanged -- this only catches what HoloForge itself might raise
    (KeyError on an unknown curve_type, numerical blowups, etc.), which is
    otherwise not "self-describing" at all per TRD §3.2.
    """
    from holoagent.tools.contract import ToolError

    try:
        return fn(*args, **kwargs)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"{description} failed: {type(exc).__name__}: {exc}") from exc


def compute_input_hash(inputs: dict[str, Any]) -> str:
    """Mirrors HoloForge's own config_hash pattern (experiments/manifest.py:
    sha256 of canonicalized sorted-key JSON) so provenance hashes here are
    directly comparable in spirit, if not value, to HoloForge's job hashes.
    """
    canonical = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    return uuid.uuid4().hex
