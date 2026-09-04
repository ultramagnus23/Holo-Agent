"""T12 retrieve_literature — Cited passages from the curated corpus (Tier B).

Phase 2 tool (TRD §8). Backed by holoagent.domains.holoforge.literature_corpus: plain
keyword search over HoloForge's own real markdown docs
(docs/*.md, README.md, REPRODUCE.md, CHANGELOG.md,
data/literature/README.md) -- see that module's docstring for why this is
deliberately not an embedding/RAG index. Every returned passage is a
verbatim substring of a real file with a file:line citation; the tool
never summarizes, paraphrases, or generates text. No corpus hit above
threshold means an empty result, not a lower-confidence guess.
"""

from __future__ import annotations

from holoagent.domains.holoforge.literature_corpus import search
from holoagent.tools._util import compute_input_hash, new_run_id
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
    description="Cited passages (verbatim, with file:line) from HoloForge's own real docs, keyword-matched. Empty if nothing scores.",
    source_module="literature_corpus (plain keyword search, not embedding/RAG)",
)
def retrieve_literature(query: str, top_k: int = 5) -> ToolResult:
    inputs = dict(query=query, top_k=top_k)
    input_hash = compute_input_hash(inputs)

    results = search(query, top_k)
    passages = [
        {"file": passage.file, "line_start": passage.line_start, "text": passage.text, "score": score}
        for passage, score in results
    ]

    diagnostic = None
    if not passages:
        diagnostic = "no corpus passage scored above zero for this query -- no answer, not a guess"

    return ToolResult(
        value={"passages": passages, "count": len(passages)},
        units="n/a (retrieved text passages, not a physical quantity)",
        in_validated_region=True,
        distance_from_bounds=None,
        run_id=new_run_id(),
        input_hash=input_hash,
        diagnostic=diagnostic,
    )
