from c2kv_eval.analysis.compare_history_kv_baselines import (
    _step_rate, _turn_joint, _resident_storage_per_step, summarize_method,
)
import json


def test_missing_reference_is_unknown_not_perfect():
    step = {"turn": 0, "state_drift": False,
            "executed_action_matches_reference": None, "state_matches_reference": None}
    assert _step_rate([step], "state_drift", "state_matches_reference") is None
    assert _turn_joint([{"drift_steps": [step]}]) is None


def test_observed_reference_comparison_still_counts():
    good = {"turn": 0, "executed_action_matches_reference": True, "state_matches_reference": True}
    bad = {"turn": 1, "executed_action_matches_reference": False, "state_matches_reference": True}
    assert _turn_joint([{"drift_steps": [good, bad]}]) == 0.5
    assert _step_rate([good, bad], "executed_action_drift", "executed_action_matches_reference") == 0.5


def test_resident_storage_uses_session_pages_not_repair_counts():
    step = {"kv_memory_report": {"persistent_session_prompt_physical_tokens": 4990,
        "page_size_tokens": 128, "active_raw_repair_tokens": 0, "active_recomputed_raw_tokens": 0}}
    assert _resident_storage_per_step([step]) == 4992
    assert _resident_storage_per_step([{"kv_memory_report": {"active_raw_repair_tokens": 0}}]) is None


def test_output_saturation_diagnostic_handles_nested_counts(tmp_path):
    root = tmp_path / "h2o" / "logs"
    root.mkdir(parents=True)
    (root / "details.jsonl").write_text(json.dumps(
        {"id": "x", "output_token_count": [[4, 8], 2]}) + "\n")
    (root / "summary.json").write_text(json.dumps(
        {"num_examples": 1, "max_completion_tokens": 8}))
    row = summarize_method(tmp_path, "h2o")
    assert row["Output Tokens"] == 14
    assert row["Output Limit Hits"] == 1
    assert row["Output Limit Hit Rate"] == 1 / 3
