"""T12 retrieve_literature — Cited passages from the curated corpus (Tier B, RAG index).

Phase 2 tool (TRD §8). The only tool that supports Tier B (PRD §3.2), not
Tier A. No corpus hit above threshold means no answer, not a lower-confidence
guess. Stub only: wire to a FAISS/Chroma index over the curated corpus
(HoloForge Part 1 preprint, Part 2 manuscript, experiment notes, cited
optics literature).
"""

from __future__ import annotations

from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import register

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 5},
    },
    "required": ["query"],
}


@register(
    name="retrieve_literature",
    json_schema=SCHEMA,
    description="Cited passages from the curated corpus, above a relevance threshold. Empty if none clear.",
    source_module="rag_index",
)
def retrieve_literature(query: str, top_k: int = 5) -> ToolResult:
    raise NotImplementedError("T12 retrieve_literature: not yet wired to the RAG index")
