"""Shared helpers for tool implementations. Not part of the public registry
surface — internal to holoagent.tools.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def compute_input_hash(inputs: dict[str, Any]) -> str:
    """Mirrors HoloForge's own config_hash pattern (experiments/manifest.py:
    sha256 of canonicalized sorted-key JSON) so provenance hashes here are
    directly comparable in spirit, if not value, to HoloForge's job hashes.
    """
    canonical = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def new_run_id() -> str:
    return uuid.uuid4().hex
