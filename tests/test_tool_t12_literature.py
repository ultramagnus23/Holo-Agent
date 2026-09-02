"""Tests for T12 (retrieve_literature) and its backing keyword-search
corpus (holoagent.tools._literature_corpus). Needs the real HoloForge
checkout's docs/ on disk (same as everything else), but is fast -- plain
keyword scoring over ~750 lines of markdown, no model/compute involved.
"""

import pytest

pytest.importorskip("torch")

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.tools._literature_corpus import build_corpus, search
from holoagent.tools.registry import get_tool


def test_corpus_is_nonempty():
    corpus = build_corpus()
    assert len(corpus) > 10


def test_search_finds_a_real_documented_finding():
    results = search("dn_max citation scoping Bayfol Table 3", top_k=5)
    assert len(results) > 0
    assert any("parameter_provenance.md" in p.file for p, _ in results)


def test_search_returns_nothing_for_genuinely_absent_terms():
    results = search("xyzzyplugh frobnicate qwzxjk", top_k=5)
    assert results == []


def test_search_respects_top_k():
    results = search("the parameter medium recording", top_k=2)
    assert len(results) <= 2


def test_passages_are_verbatim_with_citations():
    """Every passage must be an exact substring of the real source file --
    the whole point of T12 not being generative."""
    results = search("dn_max Bayfol", top_k=1)
    assert results
    passage, _ = results[0]
    from pathlib import Path

    from holoagent.tools._literature_corpus import DEFAULT_CORPUS_ROOT

    full_text = (DEFAULT_CORPUS_ROOT / passage.file).read_text(encoding="utf-8", errors="replace")
    assert passage.text in full_text


def test_t12_tool_returns_grounded_passages():
    result = get_tool("retrieve_literature").fn(query="dn_max citation scoping Bayfol")
    assert result.value["count"] > 0
    assert result.diagnostic is None
    assert all("file" in p and "line_start" in p and "text" in p for p in result.value["passages"])


def test_t12_tool_empty_result_has_diagnostic():
    result = get_tool("retrieve_literature").fn(query="xyzzyplugh frobnicate qwzxjk")
    assert result.value["count"] == 0
    assert result.diagnostic is not None
    assert "no answer, not a guess" in result.diagnostic
