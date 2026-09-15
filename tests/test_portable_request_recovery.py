from __future__ import annotations

from dataclasses import dataclass

import pytest

from c2kv_eval.portable.request_recovery import (
    RECOVERY_CONTROL_FIELD,
    apply_recovery_plan_to_arm,
    plan_request_recovery,
    split_recovery_control,
)


def _records(*texts: str) -> list[dict[str, object]]:
    return [
        {"out_index": index + 3, "decoded": text, "private": f"secret-{index}"}
        for index, text in enumerate(texts)
    ]


def _decode(record: dict[str, object]) -> str:
    return str(record["decoded"])


def test_private_control_is_detached_from_model_wire_and_metadata() -> None:
    payload = {
        "model": "m",
        "messages": [],
        RECOVERY_CONTROL_FIELD: {
            "operation": "append",
            "triggered": True,
            "selector": "witness",
            "target_values": ["private-ticket-123"],
        },
    }
    wire, control = split_recovery_control(payload)
    plan = plan_request_recovery(control, _records("private-ticket-123"), _decode)

    assert RECOVERY_CONTROL_FIELD not in wire
    assert RECOVERY_CONTROL_FIELD in payload
    assert "private-ticket-123" not in repr(plan.metadata())
    assert "secret-0" not in repr(plan.metadata())


def test_trigger_false_does_not_select_or_decode() -> None:
    plan = plan_request_recovery(
        {"operation": "append", "triggered": False},
        _records("ignored"),
        lambda record: pytest.fail("untriggered recovery must not decode records"),
    )

    assert plan.status == "not_triggered"
    assert plan.selected_index is None
    assert not plan.raw_regenerate


@pytest.mark.parametrize("operation", ["append", "replace"])
def test_trigger_with_no_history_is_an_explicit_skip(operation: str) -> None:
    plan = plan_request_recovery(
        {"operation": operation, "triggered": True},
        [],
    )

    assert plan.status == "no_compressed_history"
    assert plan.candidate_count == 0
    assert not plan.selected


def test_witness_no_match_skips_instead_of_falling_back_to_recent() -> None:
    plan = plan_request_recovery(
        {
            "operation": "append",
            "triggered": True,
            "selector": "witness",
            "target_values": ["not-present-anywhere"],
        },
        _records("first", "most recent"),
        _decode,
    )

    assert plan.status == "no_literal_witness"
    assert plan.selected_index is None
    assert not plan.selected


def test_witness_tie_selects_lowest_record_index() -> None:
    plan = plan_request_recovery(
        {
            "operation": "append",
            "triggered": True,
            "selector": "witness",
            "target_values": ["rare-one", "rare-two"],
        },
        _records("rare-one", "rare-two", "unrelated"),
        _decode,
    )

    assert plan.status == "selected"
    assert plan.selected_index == 0
    assert plan.selected_out_index == 3


@pytest.mark.parametrize("target_index", [-1, 2])
def test_index_selector_rejects_out_of_range_target(target_index: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        plan_request_recovery(
            {
                "operation": "replace",
                "triggered": True,
                "selector": "index",
                "target_index": target_index,
            },
            _records("zero", "one"),
        )


@dataclass(frozen=True)
class _Arm:
    name: str = "c2kv"
    repair: object = None
    recover: object = object()
    gold_recovery: object = "witness"


@pytest.mark.parametrize(
    ("operation", "placement"),
    [("append", "append_keep_ledger"), ("replace", "in_place")],
)
def test_selected_operation_derives_exact_repair_placement(
    operation: str,
    placement: str,
) -> None:
    plan = plan_request_recovery(
        {
            "operation": operation,
            "triggered": True,
            "selector": "index",
            "target_index": 1,
        },
        _records("zero", "one"),
    )
    derived = apply_recovery_plan_to_arm(_Arm(), plan)

    assert plan.selected_out_index == 4
    assert derived.repair == {"policy": "offset:1", "placement": placement}
    assert derived.recover is None
    assert derived.gold_recovery is None


def test_w2_selects_two_contiguous_records_and_marks_repair_window() -> None:
    plan = plan_request_recovery(
        {"operation": "replace", "triggered": True, "window": 2},
        _records("zero", "one", "two"),
    )
    derived = apply_recovery_plan_to_arm(_Arm(), plan)

    assert plan.selected_indices == (0, 1)
    assert plan.selected_out_indices == (3, 4)
    assert derived.repair == {
        "policy": "offset:0", "placement": "in_place", "window": 2}


def test_retry_full_requests_raw_regeneration_without_a_repair_target() -> None:
    plan = plan_request_recovery(
        {"operation": "retry_full", "triggered": True},
        _records("unused"),
    )

    assert plan.status == "retry_full"
    assert plan.raw_regenerate
    assert plan.placement is None
    assert plan.selected_index is None
    with pytest.raises(ValueError, match="selected append/replace"):
        apply_recovery_plan_to_arm(_Arm(), plan)


@pytest.mark.parametrize(
    ("control", "message"),
    [
        ({"operation": "unknown", "triggered": True}, "unknown recovery operation"),
        ({"operation": "append", "triggered": True, "selector": "latest"}, "unknown recovery selector"),
        ({"operation": "append", "triggered": True, "surprise": 1}, "unknown recovery control fields"),
        ({"operation": "append", "triggered": 1}, "triggered.*bool"),
        (
            {"operation": "append", "triggered": True, "selector": "witness"},
            "requires 'target_values'",
        ),
        (
            {
                "operation": "append",
                "triggered": True,
                "selector": "witness",
                "target_values": "not-a-list",
            },
            "target_values.*list",
        ),
        (
            {"operation": "replace", "triggered": True, "selector": "index"},
            "requires 'target_index'",
        ),
        (
            {
                "operation": "retry_full",
                "triggered": True,
                "selector": "index",
                "target_index": 0,
            },
            "retry_full does not accept",
        ),
    ],
)
def test_unknown_or_conflicting_controls_fail_loudly(
    control: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        plan_request_recovery(control, _records("one"), _decode)


def test_selected_record_requires_existing_proxy_out_index() -> None:
    with pytest.raises(ValueError, match="integer 'out_index'"):
        plan_request_recovery(
            {"operation": "append", "triggered": True},
            [{"decoded": "missing out index"}],
        )


def test_last_window_selects_recent_documents_and_clips_short_history():
    control = {"operation": "replace", "triggered": True, "selector": "last", "window": 2}
    plan = plan_request_recovery(control, _records("one", "two", "three"), _decode)
    assert plan.selected_indices == (1, 2)
    assert plan_request_recovery(control, _records("one"), _decode).selected_indices == (0,)
