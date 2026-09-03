"""Integrity checks on the Set B question log (holoagent/eval/questions/set_b.jsonl).

Mirrors tests/test_set_a_questions.py's approach: not a grading test, just
the structural honesty rules -- every question has a real verification
note, every expected_trigger is one of the four real escalation triggers,
and the tool-related claims (no_tool_exists / tool_errored) are checked
against the actual current tool registry state so a future tool
implementation doesn't leave a stale "no tool exists" claim uncaught.
"""

import pytest

pytest.importorskip("torch")

import holoagent.tools  # noqa: F401  (registers T1-T12)
from holoagent.eval.set_b_abstention import _VALID_TRIGGERS, load_questions
from holoagent.orchestrator.escalation import EscalationTrigger
from holoagent.tools.contract import ToolError
from holoagent.tools.registry import get_tool, list_tools

# Tool + minimal valid args for each no_tool_exists question that names a
# specific stub, so this test breaks loudly the day someone wires it up
# for real -- a stale Set B claim would otherwise sit there silently wrong.
#
# Currently empty: ALL 12 tools (T1-T12) are implemented as of this
# session (T1/T4/T5/T9 wired last, after T2/T10/T11/T12) -- there is no
# stub tool left to check. b01/b03/b04/b05 (which claimed T1/T4/T5/T9 were
# stubs) were removed from set_b.jsonl as stale, same pattern as
# b02/b06/b07/b08 before them. See git history for the removal commits.
# Kept as a dict (not deleted) so a future Phase-3 stub has somewhere to
# register this same check.
_STUB_TOOL_CALLS: dict[str, tuple[str, dict]] = {}

# question id -> (tool_name, kwargs) that must actually raise ToolError today.
_TOOL_ERRORED_CALLS = {
    "b14_unknown_curve_type": (
        "get_fit_quality",
        dict(curve_type="reflection_grating", K_rad_per_um=8.0, xs=[1, 2, 3], ys=[1, 2, 3], n_starts=1, seed=0),
    ),
    "b15_unknown_medium_field": (
        "get_fit_quality",
        dict(
            curve_type="growth",
            K_rad_per_um=8.0,
            xs=[1, 2, 3],
            ys=[1, 2, 3],
            base_medium_params={"diffusion_length": 1.0},
            n_starts=1,
            seed=0,
        ),
    ),
    "b16_unknown_method_id": (
        "get_convergence_trace",
        dict(medium_params={}, n_x=32, dx=1.0, target_pattern="square_wave", method_id="ADAM_DIRECT", seed=0),
    ),
    "b17_unknown_target_pattern": (
        "get_convergence_trace",
        dict(medium_params={}, n_x=32, dx=1.0, target_pattern="gaussian_beam", method_id="GS", seed=0),
    ),
    "b18_non_numeric_parameter_value": ("check_validity_region", dict(medium_params={"D0": "very_high"})),
    "b19_sensitivity_to_nonexistent_parameter": (
        "get_sensitivity",
        dict(
            base_medium_params={},
            parameter="reflectivity",
            n_x=32,
            dx=1.0,
            target_pattern="square_wave",
            method_id="MIL",
            seed=0,
            n_iters=5,
        ),
    ),
}

# question id -> (medium_params, K_rad_per_um) that must actually be flagged
# out of the validated region today.
_OUTSIDE_REGION_CALLS = {
    "b20_D0_out_of_range": ({"D0": 5.0}, None),
    "b21_sigma_out_of_range": ({"sigma": 1.0}, None),
    "b22_kappa_out_of_range": ({"kappa": 50.0}, None),
    "b23_dn_max_out_of_range": ({"dn_max": 0.05}, None),
    "b24_K_too_high": ({}, 30.0),
    "b25_K_too_low": ({}, 1.0),
    "b26_gamma_out_of_range": ({"gamma": 2.0}, None),
    "b27_k_bleach_out_of_range": ({"k_bleach": 5.0}, None),
    "b28_alpha_D_out_of_range": ({"alpha_D": 10.0}, None),
    "b29_shrinkage_out_of_range": ({"shrinkage": 0.1}, None),
    "b30_thickness_out_of_range": ({"thickness": 500.0}, None),
    "b31_multiple_violations": ({"D0": 0.5, "sigma": 0.5, "kappa": 20.0}, None),
}


def test_set_b_has_questions():
    # Was 43, now 35 -- b02/b06/b07/b08/b01/b03/b04/b05 were retired as
    # stale once all 12 tools became real (see git history), not padded
    # back up to a round number with lower-quality filler entries.
    assert len(load_questions()) >= 30


def test_every_question_has_verification_note():
    for q in load_questions():
        assert q.verification.strip(), f"{q.id}: verification must not be empty"
        assert q.question.strip()


def test_every_trigger_is_valid():
    for q in load_questions():
        assert q.expected_trigger in _VALID_TRIGGERS, f"{q.id}: unknown trigger {q.expected_trigger!r}"


def test_question_ids_are_unique():
    ids = [q.id for q in load_questions()]
    assert len(ids) == len(set(ids))


def test_all_four_triggers_are_represented():
    triggers = {q.expected_trigger for q in load_questions()}
    assert triggers == _VALID_TRIGGERS


def test_no_tool_exists_stub_claims_still_hold():
    """If any of these ever gets wired up for real, this must fail loudly
    -- a stale Set B claim is worse than no claim."""
    for question_id, (tool_name, kwargs) in _STUB_TOOL_CALLS.items():
        spec = get_tool(tool_name)
        with pytest.raises(NotImplementedError):
            spec.fn(**kwargs)


def test_tool_errored_claims_still_actually_error():
    for question_id, (tool_name, kwargs) in _TOOL_ERRORED_CALLS.items():
        spec = get_tool(tool_name)
        with pytest.raises(ToolError):
            spec.fn(**kwargs)


def test_outside_region_claims_still_actually_outside():
    check_validity_region = get_tool("check_validity_region").fn
    for question_id, (medium_params, K) in _OUTSIDE_REGION_CALLS.items():
        result = check_validity_region(medium_params=medium_params, K_rad_per_um=K)
        assert result.in_validated_region is False, f"{question_id}: claimed out-of-region but check_validity_region now says in-region"
