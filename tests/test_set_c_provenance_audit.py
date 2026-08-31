from holoagent.eval._transcript import SessionTranscript, ToolCallRecord
from holoagent.eval.set_c_provenance_audit import audit_answer, extract_numeric_tokens


def _transcript(**output_value) -> SessionTranscript:
    call = ToolCallRecord(
        run_id="abc",
        tool="get_fit_quality",
        inputs={"K_rad_per_um": 8.98, "seed": 0},
        outputs={"value": output_value, "distance_from_bounds": None},
    )
    return SessionTranscript(query="What's the fit quality at K=8.98?", calls=[call])


def test_exact_match_no_violation():
    t = _transcript(nrmse=0.052, quality="good")
    violations = audit_answer("The NRMSE is 0.052, rated good.", t)
    assert violations == []


def test_rounded_match_within_tolerance_no_violation():
    t = _transcript(final_psnr=7.849105422809464)
    violations = audit_answer("PSNR came out to 7.8 dB.", t)
    assert violations == []


def test_rounded_match_beyond_tolerance_is_flagged():
    # 0.052 rounded to 2 decimals is 0.05, but that's a 3.8% relative error
    # -- above the 3% cap, so it must NOT be silently accepted.
    t = _transcript(nrmse=0.052)
    violations = audit_answer("NRMSE is 0.05.", t)
    assert len(violations) == 1
    assert violations[0].token == "0.05"


def test_percent_conversion_no_violation():
    t = _transcript(nrmse=0.052)
    violations = audit_answer("That's a 5.2% NRMSE.", t)
    assert violations == []


def test_fabricated_number_is_flagged():
    t = _transcript(nrmse=0.052)
    violations = audit_answer("The improvement was 42.0 over baseline.", t)
    assert len(violations) == 1
    assert violations[0].token == "42.0"


def test_tool_call_argument_values_are_grounded():
    t = _transcript(nrmse=0.052)
    violations = audit_answer("Evaluated at K=8.98 rad/um.", t)
    assert violations == []


def test_user_query_values_are_grounded():
    t = _transcript(nrmse=0.052)
    violations = audit_answer("You asked about K=8.98, and NRMSE is 0.052.", t)
    assert violations == []


def test_run_id_digit_runs_are_never_extracted_as_tokens():
    t = _transcript(nrmse=0.052)
    # 'run4x2' etc: digits embedded in an identifier are adjacent to word
    # chars, so the token regex should never even see them as candidates.
    violations = audit_answer("See run_id=e2ffdc6a7e4f433b957a106c7e68c2bf for details.", t)
    assert violations == []


def test_all_digit_hash_excluded_via_prefix_marker():
    t = _transcript(nrmse=0.052)
    # Pathological case: a hash that happens to be all decimal digits would
    # otherwise be extracted as one giant number and (correctly) fail to
    # match anything -- the run_id= prefix marker exists for exactly this.
    violations = audit_answer("See run_id=1234567890123456 for details.", t)
    assert violations == []


def test_loss_history_pairs_are_individually_grounded():
    call = ToolCallRecord(
        run_id="r1",
        tool="get_convergence_trace",
        inputs={"n_iters": 40},
        outputs={"value": {"loss_history": [[0, 0.25], [10, 0.24]]}, "distance_from_bounds": None},
    )
    t = SessionTranscript(query="trace?", calls=[call])
    violations = audit_answer("At iteration 10 the loss was 0.24.", t)
    assert violations == []


def test_extract_numeric_tokens_skips_letters_inside_identifiers():
    tokens = [text for text, _, _ in extract_numeric_tokens("T3 got nrmse=0.052 at K3 grid point 4")]
    assert "0.052" in tokens
    assert "4" in tokens
    assert "3" not in tokens  # from "T3"/"K3" -- never a standalone token
