"""Plain keyword retrieval over HoloForge's own real markdown docs and
literature-provenance notes. Backing T12 (retrieve_literature).

Deliberately NOT embedding/RAG-based: master-prompt §38 explicitly warns
against "unnecessary web-search frameworks" and TRD §6 says "RAG (T12):
plain -- no heavy framework." More importantly, for a corpus this size
(under 1000 lines across ~10 files), a real embedding index buys nothing
a human wouldn't get from grep, and it adds a failure mode this project
can't afford: a semantic "near match" that isn't actually the passage the
answer relies on. Every passage this returns is a verbatim substring of a
real file the LLM never gets to touch or paraphrase -- the tool returns
quotes, not summaries, so "unsupported text generation" (never-do #10 in
the master prompt) can't happen inside this tool at all.

Corpus: docs/*.md, README.md, REPRODUCE.md, CHANGELOG.md, and
data/literature/README.md, all under the HoloForge checkout
(HOLOFORGE_CODE_PATH). Nothing outside that checkout, and nothing
invented -- if a paper's claim isn't written down in one of these real
files, this tool returns zero passages, not a guess.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS_ROOT = Path(
    os.environ.get("HOLOFORGE_CODE_PATH", r"C:\Users\chait\Downloads\HoloForge\HoloForge\part2\code")
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and", "or",
    "with", "this", "that", "does", "do", "did", "what", "how", "why", "which", "it", "its",
    "as", "at", "by", "be", "been", "has", "have", "had", "not", "no", "if", "than", "then",
}

_MIN_KEYWORD_LEN = 3


@dataclass(frozen=True)
class Passage:
    file: str  # path relative to the corpus root, forward-slashed
    line_start: int
    text: str


_cache: list[Passage] | None = None


def _corpus_files(root: Path) -> list[Path]:
    files: list[Path] = []
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        files.extend(sorted(docs_dir.glob("*.md")))
    for name in ("README.md", "REPRODUCE.md", "CHANGELOG.md"):
        p = root / name
        if p.is_file():
            files.append(p)
    lit_readme = root / "data" / "literature" / "README.md"
    if lit_readme.is_file():
        files.append(lit_readme)
    return files


def _chunk_file(path: Path, root: Path) -> list[Passage]:
    """Splits on blank lines -- a markdown paragraph/table-row-group per
    passage. Simple on purpose: this corpus's files are hand-written prose
    with real blank-line paragraph breaks, not machine-generated blobs
    that would need a smarter splitter.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rel = str(path.relative_to(root)).replace("\\", "/")

    passages: list[Passage] = []
    buf: list[str] = []
    start_line = 1
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if buf:
                passages.append(Passage(file=rel, line_start=start_line, text="\n".join(buf).strip()))
                buf = []
            start_line = i + 1
        else:
            if not buf:
                start_line = i
            buf.append(line)
    if buf:
        passages.append(Passage(file=rel, line_start=start_line, text="\n".join(buf).strip()))
    return [p for p in passages if len(p.text) >= 15]


def build_corpus(root: Path | None = None) -> list[Passage]:
    global _cache
    if root is None and _cache is not None:
        return _cache
    use_root = root or DEFAULT_CORPUS_ROOT
    passages: list[Passage] = []
    for f in _corpus_files(use_root):
        passages.extend(_chunk_file(f, use_root))
    if root is None:
        _cache = passages
    return passages


def _keywords(query: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) >= _MIN_KEYWORD_LEN}


def search(query: str, top_k: int, root: Path | None = None) -> list[tuple[Passage, int]]:
    """Ranks passages by raw keyword occurrence count (a simple TF score,
    no IDF -- this corpus is too small for IDF to mean anything). Returns
    only passages with score > 0; an empty result is the honest answer
    when nothing in the real corpus mentions what was asked, not a
    reason to lower the bar and return an unrelated top-k anyway.
    """
    keywords = _keywords(query)
    if not keywords:
        return []
    corpus = build_corpus(root)
    scored = []
    for passage in corpus:
        text_lower = passage.text.lower()
        score = sum(text_lower.count(kw) for kw in keywords)
        if score > 0:
            scored.append((passage, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
