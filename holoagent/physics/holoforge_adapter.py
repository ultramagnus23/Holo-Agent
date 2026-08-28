"""Thin adapter over the HoloForge repository (TRD §1, layer 4 — FROZEN).

This module never edits HoloForge. It imports and calls into it. Any bug
found in the physics goes back to the HoloForge repo through its own
process, not a patch here (TRD §1).

The tool modules in holoagent.tools call through this adapter rather than
importing HoloForge directly, so the frozen-dependency boundary has exactly
one crossing point.
"""

from __future__ import annotations

from pathlib import Path


class HoloForgeAdapter:
    """Holds a reference to a HoloForge checkout and exposes the entry
    points the tool layer needs (optimizer, diffraction solver, twin
    validation, autograd sensitivity, validity model).

    Deliberately not implemented yet: the HoloForge Part 2 module/function
    boundaries this should call are not finalized until Sep 30 (PRD §8).
    """

    def __init__(self, holoforge_repo_path: str | Path):
        self._repo_path = Path(holoforge_repo_path)
        if not self._repo_path.exists():
            raise FileNotFoundError(
                f"HoloForge repo not found at {self._repo_path} — "
                "HoloAgent wraps an existing, frozen HoloForge checkout; "
                "it does not vendor or reimplement the physics."
            )
        raise NotImplementedError(
            "HoloForgeAdapter: entry points not yet wired — build starts "
            "after HoloForge Part 2 submission (PRD §8, TRD §8 Phase 0 gate)"
        )
