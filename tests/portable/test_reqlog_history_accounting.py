"""History-KV counters must not contaminate C2KV compression metrics."""
from c2kv_eval.portable import reqlog


def test_c2kv_legacy_scheduler_counters_are_not_history_kv_metrics():
    row = {
        "status": "ok",
        "gist_tokens": 25,
        "original_tokens": 100,
        "history_kv_active_tokens": 250,
        "history_kv_full_equivalent_tokens": 10,
        "bytes_per_kv_token": 100,
        "history_tensor_accounting": {
            "before_recovery_bytes": 2500,
            "after_recovery_bytes": 3000,
            "history_ratio_before": 4.0,
            "history_ratio_after": 3.0,
            "scope": "test",
        },
    }
    summary = reqlog.summarize([row])
    assert summary["logical_over_gist"] == 4.0
    assert summary["history_kv_retention_mean"] is None
    assert summary["history_kv_compression_mean"] is None
    assert summary["history_tensor_accounting"]["after_recovery_tokens_mean"] == 30.0


def test_explicit_history_kv_arm_keeps_scheduler_accounting():
    row = {
        "status": "ok",
        "history_kv": {"method": "h2o"},
        "history_kv_active_tokens": 312,
        "history_kv_full_equivalent_tokens": 1000,
    }
    summary = reqlog.summarize([row])
    assert summary["history_kv_retention_mean"] == 0.312
    assert summary["history_kv_compression_mean"] == 1.0 / 0.312


def test_joint_ratios_use_cumulative_counts_not_mean_request_ratios():
    rows = [
        {"status": "ok", "joint_kv_scopes": [{"kind": "tool"}],
         "full_tool_kv": 100, "active_tool_kv": 25,
         "full_history_kv": 20, "active_history_kv": 5},
        {"status": "ok", "joint_kv_scopes": [{"kind": "tool"}],
         "full_tool_kv": 20, "active_tool_kv": 10,
         "full_history_kv": 100, "active_history_kv": 50},
    ]
    summary = reqlog.summarize(rows)
    assert summary["tool_retention"] == 35 / 120
    assert summary["history_retention"] == 55 / 120
    assert summary["joint_retention"] == 90 / 240
    assert summary["joint_compression_ratio"] == 240 / 90
    assert summary["history_kv_active_tokens_mean"] == (5 + 50) / 2
    assert summary["history_kv_retention_mean"] == 55 / 120
    assert abs(summary["history_kv_compression_mean"] - 120 / 55) < 1e-12
