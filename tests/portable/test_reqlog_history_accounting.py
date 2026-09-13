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
