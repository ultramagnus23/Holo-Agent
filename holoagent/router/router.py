"""Tier A / B / C classification (TRD §1, layer 1). Cheap and local.

Tier A: grounded simulation reasoning -> the tool registry (PRD §3.1).
Tier B: grounded literature reasoning -> T12 retrieve_literature (PRD §3.2).
Tier C: out of scope (general optics, hardware control, novel physics) ->
        escalation trigger 4 (PRD §6), routed out honestly rather than
        answered badly.

This is intentionally NOT an LLM call in the critical numeric path — it
only decides which downstream path to take, never states a physical
quantity itself. A cheap classifier (rules, then a small local model if
rules prove insufficient) is the intended implementation; keep it that way.
"""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    A = "A"  # grounded simulation reasoning
    B = "B"  # grounded literature reasoning
    C = "C"  # out of scope


def classify(query: str) -> Tier:
    raise NotImplementedError(
        "router.classify: Tier A/B/C classification not yet implemented "
        "(TRD §1 layer 1 — build after Phase 1's hosted-orchestrator loop "
        "is working end to end)"
    )
