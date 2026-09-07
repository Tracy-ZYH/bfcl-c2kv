from __future__ import annotations

import io
import json

import pytest

from c2kv_eval.portable import proxy as proxy_mod
from c2kv_eval.portable.arms import get_arm
from c2kv_eval.portable.request_recovery import RECOVERY_CONTROL_FIELD


def _response(label: str, prompt: int, completion: int) -> tuple[dict, dict]:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    data = {"choice": label, "usage": usage}
    normalized = {
        "content": label,
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": usage,
        "cost": {},
    }
    return data, normalized


def _compressed_records(*texts: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": text,
            "out_index": index + 1,
            "record": {"original_seq_len": 10},
        }
        for index, text in enumerate(texts)
    ]


@pytest.mark.parametrize(
    ("operation", "placement"),
    [("append", "append_keep_ledger"), ("replace", "in_place")],
)
def test_selected_kv_recovery_retries_once_with_request_local_arm(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    placement: str,
) -> None:
    base_arm = get_arm("c2kv4")
    counts = {"compressed_records": _compressed_records("first", "second")}
    planned = []
    sent = []

    def fake_plan_repair(messages, arm, counts_, tools, out_messages):
        planned.append(arm)
        return {"policy": arm.repair["policy"], "placement": arm.repair["placement"]}

    def fake_send(out_messages, plan, arm):
        sent.append((out_messages, plan, arm))
        return _response("retry", 7, 3)

    monkeypatch.setattr(proxy_mod, "ARM", base_arm)
    monkeypatch.setattr(proxy_mod, "plan_repair", fake_plan_repair)
    initial_data, initial_normalized = _response("initial", 5, 2)
    final_data, final_normalized, plan = proxy_mod.retry_requested_recovery(
        {
            "operation": operation,
            "triggered": True,
            "selector": "index",
            "target_index": 1,
        },
        [{"role": "user", "content": "current"}],
        [{"role": "user", "content": "assembled"}],
        counts,
        base_arm,
        [],
        initial_data,
        initial_normalized,
        fake_send,
    )

    assert len(planned) == len(sent) == 1
    request_arm = sent[0][2]
    assert request_arm is planned[0] and request_arm is not base_arm
    assert request_arm.repair == {
        "policy": "offset:1",
        "placement": placement,
    }
    assert base_arm.repair is None
    assert proxy_mod.ARM is base_arm
    assert plan == request_arm.repair
    assert final_data["choice"] == final_normalized["content"] == "retry"
    assert final_normalized["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "total_tokens": 17,
    }
    assert counts["request_recovery"]["generation_attempts"] == 2


def test_witness_without_literal_match_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        proxy_mod,
        "_witness_texts",
        lambda records: [record["content"] for record in records],
    )
    initial_data, initial_normalized = _response("initial", 4, 1)
    counts = {"compressed_records": _compressed_records("alpha", "most recent")}

    def fail_send(*args, **kwargs):
        pytest.fail("a missing literal witness must not trigger a retry")

    final_data, final_normalized, plan = proxy_mod.retry_requested_recovery(
        {
            "operation": "append",
            "triggered": True,
            "selector": "witness",
            "target_values": ["absent-value"],
        },
        [],
        [],
        counts,
        get_arm("c2kv4"),
        [],
        initial_data,
        initial_normalized,
        fail_send,
    )

    assert final_data is initial_data
    assert final_normalized is initial_normalized
    assert plan is None
    assert counts["request_recovery"]["status"] == "no_literal_witness"
    assert counts["request_recovery"]["selected_index"] is None


def test_retry_full_uses_full_assembly_updates_counts_and_sums_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_arm = get_arm("c2kv4")
    messages = [{"role": "user", "content": "current"}]
    raw_messages = [{"role": "system", "content": "raw"}, *messages]
    retry_counts = {
        "gist_tokens": 0,
        "original_tokens": 0,
        "n_gist_messages": 0,
        "compressed_records": [],
        "doc_packing": "message",
    }
    assembled = []
    sent = []

    def fake_assemble(received_messages, arm):
        assembled.append((received_messages, arm))
        return raw_messages, retry_counts

    def fake_send(out_messages, plan, arm):
        sent.append((out_messages, plan, arm))
        return _response("full retry", 11, 4)

    monkeypatch.setattr(proxy_mod, "_assemble", fake_assemble)
    initial_data, initial_normalized = _response("initial", 6, 2)
    counts = {
        "gist_tokens": 9,
        "original_tokens": 90,
        "n_gist_messages": 2,
        "compressed_records": _compressed_records("old"),
    }
    final_data, final_normalized, plan = proxy_mod.retry_requested_recovery(
        {"operation": "retry_full", "triggered": True},
        messages,
        [{"role": "user", "content": "gist"}],
        counts,
        base_arm,
        [],
        initial_data,
        initial_normalized,
        fake_send,
    )

    assert assembled == [(messages, proxy_mod.FULL_ASSEMBLY)]
    assert sent == [(raw_messages, None, proxy_mod.FULL_ASSEMBLY)]
    assert plan is None
    assert base_arm.repair is None
    assert counts["gist_tokens"] == 0
    assert counts["compressed_records"] == []
    assert counts["request_recovery"]["environment_rollback"] is False
    assert counts["request_recovery"]["raw_kv_source"] == "full_history_prefill"
    assert final_data["choice"] == final_normalized["content"] == "full retry"
    assert final_data["usage"] == final_normalized["usage"] == {
        "prompt_tokens": 17,
        "completion_tokens": 6,
        "total_tokens": 23,
    }


def test_proxy_strips_private_control_from_both_upstream_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = []
    responses = iter([_response("initial", 3, 1), _response("retry", 4, 2)])

    class FakeBackend:
        name = "fake"
        wants_request_context = False

        def prepare_chat(self, payload, arm, plan):
            prepared.append((dict(payload), arm, plan))
            return dict(payload)

        def normalize_response(self, data):
            return data["normalized"]

    def fake_post(path, payload, timeout, retries=2):
        data, normalized = next(responses)
        return {**data, "normalized": normalized}

    counts = {
        "system_raw": 1,
        "history_raw": 0,
        "current_raw": 1,
        "compressed": 1,
        "gist_tokens": 2,
        "original_tokens": 10,
        "history_packed_original_tokens": 10,
        "history_dropped_original_tokens": 0,
        "history_packed_candidate_doc_count": 1,
        "history_retained_fraction": 1.0,
        "n_gist_messages": 1,
        "compressed_records": _compressed_records("history"),
        "doc_packing": "turn",
        "n_docs": 1,
        "dropped_docs": 0,
        "current_start_out_index": 2,
    }

    def fake_plan_repair(messages, arm, counts_, tools, out_messages):
        if arm.repair is None:
            return None
        return dict(arm.repair)

    monkeypatch.setattr(proxy_mod, "ARM", get_arm("c2kv4"))
    monkeypatch.setattr(proxy_mod, "BACKEND", FakeBackend())
    monkeypatch.setattr(proxy_mod, "DEFAULT_RECOVERY_CONTROL", None)
    monkeypatch.setattr(proxy_mod, "QUERY_PROJECTION", None)
    monkeypatch.setattr(proxy_mod, "REQUEST_LOG_PATH", "")
    monkeypatch.setattr(proxy_mod.STATE, "recover", None)
    monkeypatch.setattr(proxy_mod.STATE, "reference_log_path", "")
    monkeypatch.setattr(proxy_mod, "_assemble", lambda messages, arm: (messages, dict(counts)))
    monkeypatch.setattr(proxy_mod, "plan_repair", fake_plan_repair)
    monkeypatch.setattr(proxy_mod, "_repair_frame_check", lambda plan, normalized: None)
    monkeypatch.setattr(proxy_mod, "_post_json", fake_post)

    body = json.dumps(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "question"}],
            RECOVERY_CONTROL_FIELD: {
                "operation": "append",
                "triggered": True,
                "selector": "first",
            },
        }
    ).encode("utf-8")
    handler = proxy_mod.ProxyHandler.__new__(proxy_mod.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    sent = {}
    handler._send_json = lambda code, obj: sent.update(code=code, obj=obj)

    handler.do_POST()

    assert sent["code"] == 200
    assert len(prepared) == 2
    assert all(RECOVERY_CONTROL_FIELD not in payload for payload, _, _ in prepared)
    assert prepared[0][1] is proxy_mod.ARM and prepared[0][2] is None
    assert prepared[1][1] is not proxy_mod.ARM
    assert prepared[1][1].repair == {
        "policy": "offset:0",
        "placement": "append_keep_ledger",
    }
    assert sent["obj"]["choice"] == "retry"
    assert sent["obj"]["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }
