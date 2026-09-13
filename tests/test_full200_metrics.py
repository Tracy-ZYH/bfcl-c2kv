from c2kv_eval.adapters.history_step_common import build_step_record
from c2kv_eval.analysis.compare_full200_unified import (
    _history_summary,
    _recovery_summary,
    _step_rate,
)


def test_missing_reference_is_not_empty_action_drift():
    row = build_step_record(
        sample_id="sample",
        turn_idx=0,
        step_idx=0,
        global_step=0,
        candidate_raw_text="<tool_call>example</tool_call>",
        candidate_action=["example()"],
        candidate_status="decoded_action",
        reference_step=None,
        alignment_status="missing_reference",
        executed_action=["example()"],
        state={},
    )

    assert row["candidate_action_matches_reference"] is None
    assert row["candidate_action_drift"] is None
    assert row["executed_action_matches_reference"] is None
    assert row["executed_action_drift"] is None
    assert (
        _step_rate(
            [row], "candidate_action_drift", "candidate_action_matches_reference"
        )
        is None
    )


def test_rollback_history_uses_legacy_nonzero_accounting_fields():
    summary = _history_summary(
        [
            {
                "canonical_full_history_tokens": 0,
                "history_original_tokens": 400,
                "physical_history_kv_tokens": 0,
                "history_effective_tokens": 100,
            }
        ],
        committed_steps=1,
        method="rollback_d2",
        kv_row={},
    )

    assert summary["compression"] == 4.0
    assert summary["coverage"] == 1.0


def test_reference_recovery_rates_are_segment_based():
    segment = {
        "detector_trigger": True,
        "oracle_reference_drift_segment": True,
        "reference_recovery_success": True,
        "rollback_coverage": 0.5,
    }
    summary = _recovery_summary(
        [{"checkpoint_segments": [segment]}],
        [{"rollback_latency_sec": 2.0}],
        "rollback_d2",
    )

    assert summary["trigger_rate"] == 1.0
    assert summary["detector_coverage"] == 1.0
    assert summary["success_rate"] == 1.0
    assert summary["rollback_span_coverage"] == 0.5
