"""Detector-agnostic request recovery planning for portable benchmark runs.

The control object is a private sidecar.  ``split_recovery_control`` removes
it before a request is sent to the model.  This module only plans generation
retry and request-local KV placement; ``retry_full`` is not an environment
rollback.

Witness selection intentionally uses the frozen Witness-IDF core over text
decoded by the caller from the exact compressed records.  A missing literal
witness produces a skip instead of silently repairing the most recent block.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Optional

from .agent.d_witness_core import select_k_star


RECOVERY_CONTROL_FIELD = "c2kv_recovery_control"

_OPERATIONS = {"append", "replace", "retry_full"}
_SELECTORS = {"first", "witness", "index"}
_CONTROL_KEYS = {
    "operation",
    "triggered",
    "selector",
    "target_values",
    "target_index",
}
_PLACEMENTS = {
    "append": "append_keep_ledger",
    "replace": "in_place",
}


@dataclass(frozen=True)
class RequestRecoveryPlan:
    """Validated request-level decision with no private target values."""

    status: str
    operation: str
    triggered: bool
    selector: str
    selected_index: Optional[int]
    selected_out_index: Optional[int]
    placement: Optional[str]
    candidate_count: int
    raw_regenerate: bool = False

    @property
    def selected(self) -> bool:
        return self.status == "selected"

    def metadata(self) -> dict[str, Any]:
        """Return safe selection metadata for logs or response annotations."""

        return {
            "status": self.status,
            "operation": self.operation,
            "triggered": self.triggered,
            "selector": self.selector,
            "selected_index": self.selected_index,
            "selected_out_index": self.selected_out_index,
            "placement": self.placement,
            "candidate_count": self.candidate_count,
            "raw_regenerate": self.raw_regenerate,
        }


def split_recovery_control(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], Any]:
    """Return a wire-safe shallow copy and the detached private control."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    wire_payload = dict(payload)
    control = wire_payload.pop(RECOVERY_CONTROL_FIELD, None)
    return wire_payload, control


def _validated_control(control: Mapping[str, Any]) -> tuple[str, bool, str, Sequence[str], Optional[int]]:
    if not isinstance(control, Mapping):
        raise TypeError("recovery control must be a mapping")
    unknown = set(control) - _CONTROL_KEYS
    if unknown:
        raise ValueError(f"unknown recovery control fields: {sorted(unknown)}")

    operation = control.get("operation")
    if operation not in _OPERATIONS:
        raise ValueError(
            f"unknown recovery operation {operation!r}; expected one of {sorted(_OPERATIONS)}"
        )
    triggered = control.get("triggered")
    if not isinstance(triggered, bool):
        raise TypeError("recovery control 'triggered' must be a bool")

    selector = control.get("selector", "first")
    if selector not in _SELECTORS:
        raise ValueError(
            f"unknown recovery selector {selector!r}; expected one of {sorted(_SELECTORS)}"
        )

    has_values = "target_values" in control
    has_index = "target_index" in control
    target_values: Sequence[str] = ()
    target_index: Optional[int] = None

    if operation == "retry_full":
        if selector != "first" or has_values or has_index:
            raise ValueError("retry_full does not accept a target selector or target fields")
        return operation, triggered, selector, target_values, target_index

    if selector == "witness":
        if not has_values:
            raise ValueError("witness selector requires 'target_values'")
        values = control["target_values"]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise TypeError("recovery control 'target_values' must be a list[str]")
        if has_index:
            raise ValueError("witness selector does not accept 'target_index'")
        target_values = values
    elif selector == "index":
        if not has_index:
            raise ValueError("index selector requires 'target_index'")
        index = control["target_index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("recovery control 'target_index' must be an int")
        if has_values:
            raise ValueError("index selector does not accept 'target_values'")
        target_index = index
    elif has_values or has_index:
        raise ValueError("first selector does not accept target fields")

    return operation, triggered, selector, target_values, target_index


def _selected_out_index(record: Mapping[str, Any], selected_index: int) -> int:
    if not isinstance(record, Mapping):
        raise TypeError(f"compressed record {selected_index} must be a mapping")
    out_index = record.get("out_index")
    if isinstance(out_index, bool) or not isinstance(out_index, int):
        raise ValueError(
            f"compressed record {selected_index} must carry an integer 'out_index'"
        )
    return out_index


def plan_request_recovery(
    control: Mapping[str, Any],
    compressed_records: Sequence[Mapping[str, Any]],
    decode_record_text: Optional[Callable[[Mapping[str, Any]], str]] = None,
) -> RequestRecoveryPlan:
    """Validate a private control and select at most one compressed record."""

    operation, triggered, selector, target_values, target_index = _validated_control(control)
    if isinstance(compressed_records, (str, bytes)) or not isinstance(
        compressed_records, Sequence
    ):
        raise TypeError("compressed_records must be a sequence")
    candidate_count = len(compressed_records)
    placement = _PLACEMENTS.get(operation)

    def decision(
        status: str,
        *,
        selected_index: Optional[int] = None,
        selected_out_index: Optional[int] = None,
        raw_regenerate: bool = False,
    ) -> RequestRecoveryPlan:
        return RequestRecoveryPlan(
            status=status,
            operation=operation,
            triggered=triggered,
            selector=selector,
            selected_index=selected_index,
            selected_out_index=selected_out_index,
            placement=placement,
            candidate_count=candidate_count,
            raw_regenerate=raw_regenerate,
        )

    if not triggered:
        return decision("not_triggered")
    if operation == "retry_full":
        return decision("retry_full", raw_regenerate=True)
    if not compressed_records:
        return decision("no_compressed_history")

    if selector == "first":
        selected_index = 0
    elif selector == "index":
        assert target_index is not None
        if not 0 <= target_index < candidate_count:
            raise ValueError(
                f"target_index {target_index} out of range for {candidate_count} compressed records"
            )
        selected_index = target_index
    else:
        if decode_record_text is None or not callable(decode_record_text):
            raise TypeError("witness selector requires a decode_record_text callback")
        texts: list[str] = []
        for index, record in enumerate(compressed_records):
            if not isinstance(record, Mapping):
                raise TypeError(f"compressed record {index} must be a mapping")
            text = decode_record_text(record)
            if not isinstance(text, str):
                raise TypeError(
                    f"decode_record_text must return str for compressed record {index}"
                )
            texts.append(text)
        witness_index = select_k_star(texts, target_values)
        if witness_index is None:
            return decision("no_literal_witness")
        selected_index = witness_index

    selected_out_index = _selected_out_index(
        compressed_records[selected_index], selected_index
    )
    return decision(
        "selected",
        selected_index=selected_index,
        selected_out_index=selected_out_index,
    )


def apply_recovery_plan_to_arm(arm: Any, plan: RequestRecoveryPlan) -> Any:
    """Derive the one-request repair arm for a selected KV placement plan."""

    if not isinstance(plan, RequestRecoveryPlan):
        raise TypeError("plan must be a RequestRecoveryPlan")
    if not plan.selected or plan.operation not in _PLACEMENTS:
        raise ValueError("only a selected append/replace plan can derive a repair arm")
    assert plan.selected_index is not None and plan.placement is not None
    return replace(
        arm,
        repair={
            "policy": f"offset:{plan.selected_index}",
            "placement": plan.placement,
        },
        recover=None,
        gold_recovery=None,
    )


__all__ = [
    "RECOVERY_CONTROL_FIELD",
    "RequestRecoveryPlan",
    "apply_recovery_plan_to_arm",
    "plan_request_recovery",
    "split_recovery_control",
]
