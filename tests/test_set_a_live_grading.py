"""Real Set A grading against a real self-hosted model (Ollama).

Skipped automatically if no Ollama server is reachable, same pattern as
test_ollama_backend_live.py.

IMPORTANT GAP this test surfaces rather than hides: Set A's stored
question text (holoagent/eval/questions/set_a.jsonl) is phrased the way
the original HoloForge finding is described ("For the Bayfol HX growth_dn
curve (K=8.98 rad/um), does fitting {kappa, dn_max}..."), which does NOT
carry the digitized (x, y) curve data get_fit_quality actually requires.
No tool in the current registry can look up a named literature curve --
T12 (retrieve_literature) is still a stub, and even once built it's
scoped for text-passage retrieval, not necessarily structured digitized
datasets. So the bare question text from set_a.jsonl is not, today,
independently answerable by a model with no other context -- it would
have to either hallucinate the data (exactly what this whole system
exists to prevent) or correctly refuse.

This test does NOT paper over that: it grades the bare question text
first (expecting the model to escalate rather than fabricate the curve
data) AND separately grades a data-complete version (as a real researcher
pasting their own digitized numbers into a chat would actually phrase it)
to prove the tool/provenance loop itself works when the data is actually
available. Both outcomes are informative; neither is a failure of the
system to hide.
"""

import pytest

pytest.importorskip("torch")

from holoagent.orchestrator.ollama_backend import ollama_server_running

if not ollama_server_running():
    pytest.skip("no Ollama server reachable at localhost:11434", allow_module_level=True)

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.eval.set_a_grounded import SetAQuestion, grade_question, load_questions
from holoagent.orchestrator.orchestrator import Orchestrator
from holoagent.orchestrator.ollama_backend import OllamaBackend
from holoagent.provenance.store import ProvenanceStore

pytestmark = pytest.mark.slow


def test_bare_question_text_does_not_fabricate_curve_data(tmp_path):
    """q01 as literally stored in set_a.jsonl -- no digitized data included.
    The model has no way to get real xs/ys, so the only acceptable outcomes
    are: it doesn't call get_fit_quality with invented numbers, or if it
    does call a tool, the final answer stays within Set C's provenance
    grounding (no fabricated NRMSE/kappa/dn_max values presented as fact).
    """
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    orchestrator = Orchestrator(store, OllamaBackend(), max_steps=6)
    question = next(q for q in load_questions() if q.id == "q01_dn_max_vs_D0_growth_dn_fit")

    result = grade_question(orchestrator, store, question)

    # The real assertion: whatever the model did, it must not have stated
    # a fabricated numeric result as fact. It's fine if it asks for the
    # data, refuses, or gives a qualitative non-numeric answer.
    assert result.error is None
    assert result.provenance_violations == 0, (
        f"model stated ungrounded number(s) rather than asking for the missing curve data: "
        f"{result.answer_text!r}"
    )
    store.close()


def test_data_complete_question_closes_the_loop(tmp_path):
    """Same finding as q01, but phrased the way a researcher who actually
    has their digitized data in hand would really type it -- proves the
    tool/provenance loop itself works end to end with a real model, real
    HoloForge compute, and real data, once the data gap above is filled in
    by the user rather than the model.
    """
    store = ProvenanceStore(tmp_path / "runs.sqlite3")
    orchestrator = Orchestrator(store, OllamaBackend(), max_steps=6)

    question = SetAQuestion(
        id="q01_data_complete_variant",
        question=(
            "I digitized the Bayfol HX growth_dn curve at K=8.98 rad/um from Bruder et al. 2017 "
            "Figure 3 (dose, dn) pairs: (2.0, 0.0038), (4.0, 0.0140), (5.5, 0.0125), (7.5, 0.0122), "
            "(11.0, 0.0118), (20.0, 0.0105), (40.0, 0.0105), (80.0, 0.0098). Using base medium "
            "parameters D0=0.025, sigma=0.0092, kappa=2.0, gamma=1.0, dn_max=6.0e-3, k_bleach=0.2, "
            "alpha_D=1.0, shrinkage=0.025, thickness=50.0, n0=1.5 and thickness_um=50.0, fit this "
            "curve with dn_max as the free second parameter (alongside kappa) and tell me the NRMSE."
        ),
        evidence="derived from q01 for live-grading purposes -- see this file's module docstring",
        expected_tools=["get_fit_quality"],
        status="documented_historical",
    )

    result = grade_question(orchestrator, store, question)

    assert result.error is None, f"orchestrator errored: {result.error}"
    assert result.expected_tools_satisfied, f"model never called get_fit_quality; tools called: {result.tools_called}"
    assert result.provenance_violations == 0, f"model stated an ungrounded number: {result.answer_text!r}"
    store.close()
