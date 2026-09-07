"""Check privileged selection, placement and native-message isolation."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2kv_eval.portable import proxy
from c2kv_eval.portable.arms import get_arm
from c2kv_eval.portable.backends.sglang import SglangBackend


@pytest.fixture
def planned(monkeypatch):
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "_witness_texts", lambda records: [r["content"] for r in records])
    calls = []

    def repair(messages, arm, counts, tools, out_messages):
        calls.append(arm.repair)
        return {"repair_key_hash": "raw-block", "repair_block_tokens": 4,
                "target_out_index": 0, "current_start_out_index": 2,
                "placement": "append_keep_ledger"}

    monkeypatch.setattr(proxy, "plan_repair", repair)
    return calls


def inputs():
    records = [{"role": "user", "content": text, "out_index": i,
                "record": {"original_seq_len": 4}}
               for i, text in enumerate(["lookup shared", "lookup ticket_xyz", "lookup shared"])]
    out = [{"role": "user", "content": r["content"], "c2kv_key_hash": str(i)}
           for i, r in enumerate(records)]
    oracle = {"kind": "bfcl_gold_turn_v2", "version": 2,
              "task_id": "test_0", "turn": 1,
              "selector": "witness", "values": ["lookup", "ticket_xyz"]}
    return records, out, oracle


def test_witness_uses_rare_value_and_freezes_block(planned):
    records, out, oracle = inputs()
    counts = {"compressed_records": records}
    plan = proxy.plan_gold_repair([], get_arm("c2kv4_gold_witness"), counts, oracle, [], out)
    assert planned == [{"policy": "offset:1", "placement": "append_keep_ledger"}]
    assert counts["gold_recovery"]["selected_index"] == 1
    assert plan["repair_block_tokens"] == 4
    oracle["values"] = ["shared"]
    later_counts = {"compressed_records": records}
    proxy.plan_gold_repair(
        [], get_arm("c2kv4_gold_witness"), later_counts, oracle, [], out)
    assert len(planned) == 1
    assert later_counts["gold_recovery"]["raw_kv_cache_hit"]
    assert later_counts["gold_recovery"]["selected_index"] == 1
    assert later_counts["gold_recovery"]["event_id"] == counts["gold_recovery"]["event_id"]


def test_no_witness_does_not_fall_back_to_gold_text_or_another_block(planned):
    records, out, oracle = inputs()
    oracle["values"] = ["unseen_target"]
    counts = {"compressed_records": records}
    assert proxy.plan_gold_repair([], get_arm("c2kv4_gold_witness"), counts, oracle, [], out) is None
    assert counts["gold_recovery"]["status"] == "no_literal_witness"
    assert planned == []


def test_cached_raw_kv_uses_current_turn_carrier_location(planned):
    records, out, oracle = inputs()
    counts = {"compressed_records": records, "current_start_out_index": 3}
    first = proxy.plan_gold_repair([], get_arm("c2kv4_gold_witness"), counts, oracle, [], out)
    counts["current_start_out_index"] = 5
    second = proxy.plan_gold_repair([], get_arm("c2kv4_gold_witness"), counts, oracle, [], out)
    assert first["current_start_out_index"] == 3
    assert second["current_start_out_index"] == 5
    assert len(planned) == 1
    assert counts["gold_recovery"]["recovery_extract_sec"] == 0


def test_gold_payload_rejected_for_plain_arm(planned):
    records, out, oracle = inputs()
    with pytest.raises(ValueError, match="gold-recovery arm"):
        proxy.plan_gold_repair([], get_arm("c2kv4"), {"compressed_records": records}, oracle, [], out)


def test_v1_and_v2_events_do_not_share_a_frozen_witness(planned):
    records, out, oracle_v2 = inputs()
    oracle_v1 = dict(oracle_v2, kind="bfcl_gold_turn_v1")
    oracle_v1.pop("version")
    counts_v1 = {"compressed_records": records}
    assert proxy.plan_gold_repair(
        [], get_arm("c2kv4_gold_witness"), counts_v1, oracle_v1, [], out)
    oracle_v2["values"] = ["not-present"]
    counts_v2 = {"compressed_records": records}
    assert proxy.plan_gold_repair(
        [], get_arm("c2kv4_gold_witness"), counts_v2, oracle_v2, [], out) is None
    assert counts_v2["gold_recovery"]["status"] == "no_literal_witness"


def test_v3_witness_is_frozen_per_turn_and_isolated_from_v2(planned):
    records, out, oracle_v2 = inputs()
    counts_v2 = {"compressed_records": records}
    assert proxy.plan_gold_repair(
        [], get_arm("c2kv4_gold_witness"), counts_v2, oracle_v2, [], out)

    oracle_v3 = dict(
        oracle_v2,
        kind="bfcl_gold_turn_v3",
        version=3,
        values=["shared"],
    )
    counts_v3 = {"compressed_records": records}
    assert proxy.plan_gold_repair(
        [], get_arm("c2kv4_gold_witness"), counts_v3, oracle_v3, [], out)
    assert len(planned) == 2
    assert counts_v3["gold_recovery"]["oracle_kind"] == "bfcl_gold_turn_v3"
    assert counts_v3["gold_recovery"]["oracle_version"] == 3
    assert (counts_v3["gold_recovery"]["event_id"]
            != counts_v2["gold_recovery"]["event_id"])

    oracle_v3["values"] = ["not-present"]
    later = {"compressed_records": records}
    assert proxy.plan_gold_repair(
        [], get_arm("c2kv4_gold_witness"), later, oracle_v3, [], out)
    assert len(planned) == 2
    assert later["gold_recovery"]["selected_index_at_trigger"] == (
        counts_v3["gold_recovery"]["selected_index_at_trigger"])


def test_native_full_preserves_tools_and_system_absence():
    messages = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
                {"role": "tool", "tool_call_id": "x", "content": "ok"}]
    out, counts = proxy._assemble(messages, get_arm("full_native"))
    assert out == messages
    assert counts["doc_packing"] == "native"


def test_gold_repair_appends_only_kv_carrier(planned):
    records, out, oracle = inputs()
    arm = get_arm("c2kv4_gold_witness")
    plan = proxy.plan_gold_repair([], arm, {"compressed_records": records}, oracle, [], out)
    backend = SglangBackend(None)
    payload = backend.prepare_chat({"messages": out}, arm, plan)
    assert payload["messages"][2] == {
        "role": "user", "content": "", "c2kv_repair_only_key_hashes": ["raw-block"],
        "c2kv_repair_placement": "append_keep_ledger"}
    assert "c2kv_oracle" not in payload


def test_native_hbm_bytes_are_preserved_from_server():
    data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "metadata": {"sglang_runtime": {"bytes_per_kv_token": 8192,
                         "total_gpu_kv_bytes": 24576}}}
    cost = SglangBackend(None).normalize_response(data)["cost"]
    assert cost["bytes_per_kv_token"] == 8192
    assert cost["total_gpu_kv_bytes"] == 24576
