from holoagent.tools.contract import MissingParameterError, ToolError, ToolResult


def test_tool_result_is_frozen():
    result = ToolResult(
        value=0.42,
        units="dimensionless",
        in_validated_region=True,
        distance_from_bounds=None,
        run_id="run-1",
        input_hash="hash-1",
    )
    assert result.value == 0.42
    try:
        result.value = 1.0  # type: ignore[misc]
        assert False, "ToolResult should be frozen"
    except AttributeError:
        pass


def test_missing_parameter_error_carries_field_name():
    err = MissingParameterError("saturation_ceiling")
    assert err.missing_field == "saturation_ceiling"
    assert isinstance(err, ToolError)
    assert "saturation_ceiling" in str(err)
