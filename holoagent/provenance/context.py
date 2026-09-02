"""Makes the CURRENT session's ProvenanceStore reachable from inside a pure
tool function, for T10 (query_run_store) and T11 (compare_runs) -- the two
tools whose entire job is reading provenance, not HoloForge.

Every other tool (T3/T6/T7/T8) reaches its backend via
holoagent.physics.get_default_adapter(), a process-wide singleton -- that's
correct there because there's exactly one HoloForge checkout. It would be
WRONG here: each Orchestrator (each test, each session) has its own
ProvenanceStore instance, often backed by a throwaway file. A module-level
"default store" would silently answer T10/T11 queries against the wrong
session's data. Instead, Orchestrator.answer() sets a contextvar for the
duration of the call (see `use_store`), and T10/T11 read it via
`get_current_store()` -- scoped to the actual call, never global state that
outlives it or leaks across sessions.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import TYPE_CHECKING

from holoagent.tools.contract import ToolError

if TYPE_CHECKING:
    # Type-only: a real (runtime) import here creates a circular import --
    # holoagent.provenance.__init__ -> store.py -> tools.contract ->
    # holoagent.tools.__init__ (imports every tool, including T10) -> this
    # module -> back to holoagent.provenance.store, which is still
    # mid-import (ProvenanceStore not yet defined). Found live: any entry
    # point that imports holoagent.provenance or holoagent.eval before
    # holoagent.tools hit this; existing tests happened to avoid it purely
    # by import order (`import holoagent.tools` first), which is exactly
    # the kind of fragile-by-luck bug worth actually fixing.
    from holoagent.provenance.store import ProvenanceStore

_current_store: ContextVar["ProvenanceStore | None"] = ContextVar("current_store", default=None)


@contextlib.contextmanager
def use_store(store: ProvenanceStore):
    token = _current_store.set(store)
    try:
        yield
    finally:
        _current_store.reset(token)


def get_current_store() -> ProvenanceStore:
    store = _current_store.get()
    if store is None:
        raise ToolError(
            "no active provenance store for this call -- T10/T11 must run inside an "
            "Orchestrator.answer() call (or a test using holoagent.provenance.context.use_store)"
        )
    return store
