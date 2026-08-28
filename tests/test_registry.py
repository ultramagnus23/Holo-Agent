import pytest

import holoagent.tools as tools  # noqa: F401  (registers all T1-T12 stubs)
from holoagent.tools.contract import ToolResult
from holoagent.tools.registry import get_tool, list_tools, register


def test_all_twelve_tools_registered():
    names = {spec.name for spec in list_tools()}
    expected = {
        "get_phase_map",
        "get_diffraction_efficiency",
        "get_fit_quality",
        "get_medium_state",
        "reconstruct_wavefront",
        "get_sensitivity",
        "get_convergence_trace",
        "check_validity_region",
        "run_parameter_sweep",
        "query_run_store",
        "compare_runs",
        "retrieve_literature",
    }
    assert expected <= names


def test_get_tool_unknown_name_raises():
    with pytest.raises(KeyError):
        get_tool("does_not_exist")


def test_register_duplicate_name_raises():
    def fn(**kwargs) -> ToolResult:
        raise NotImplementedError

    register("dupe_test_tool", {}, "desc", "module")(fn)
    with pytest.raises(ValueError):
        register("dupe_test_tool", {}, "desc", "module")(fn)
