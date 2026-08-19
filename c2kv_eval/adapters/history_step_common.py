from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    is_empty_execute_response,
)
from bfcl_eval.utils import make_json_serializable


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_action_text(text: Any) -> str:
    if not isinstance(text, str):
        return json_dumps(text)
    return re.sub(r"\s+", "", text)


def stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): stringify_mapping_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [stringify_mapping_keys(item) for item in value]
    if isinstance(value, tuple):
        return [stringify_mapping_keys(item) for item in value]
    return value


def normalize_state(value: Any) -> str:
    return json.dumps(
        stringify_mapping_keys(make_json_serializable(value)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compact_text(text: Any) -> str:
    return " ".join((text or "").split()) if isinstance(text, str) else ""


def reference_result_text(
    reference_result: Sequence[Sequence[Any]],
    turn_idx: int,
    step_idx: int,
) -> str:
    try:
        return str(reference_result[turn_idx][step_idx])
    except Exception:
        return ""


def reference_by_turn_step(
    reference_steps: Sequence[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for global_step, step in enumerate(reference_steps):
        if (
            isinstance(step, dict)
            and step.get("turn") is not None
            and step.get("step") is not None
        ):
            copied = dict(step)
            copied.setdefault("global_step", global_step)
            out[(int(step.get("turn")), int(step.get("step")))] = copied
    return out


def reference_step_for(
    reference_map: dict[tuple[int, int], dict[str, Any]],
    reference_result: Sequence[Sequence[Any]],
    turn_idx: int,
    step_idx: int,
    *,
    fallback_state: Any = None,
) -> tuple[dict[str, Any] | None, str]:
    ref_step = reference_map.get((turn_idx, step_idx))
    if ref_step is not None:
        return ref_step, "matched"
    ref_text = reference_result_text(reference_result, turn_idx, step_idx)
    if ref_text:
        return (
            {
                "turn": turn_idx,
                "step": step_idx,
                "decoded_action": [],
                "assistant_message": {"role": "assistant", "content": ref_text},
                "execution_results": [],
                "state": fallback_state,
                "synthetic_reference_step": True,
            },
            "matched_synthetic_from_result",
        )
    return None, "missing_reference"


@dataclass
class CandidateDecode:
    action: list[str]
    status: str
    decode_error: str | None
    empty_response: bool
    tool_call_parse_success: bool


def decode_candidate(decoder: Any, raw_text: str) -> CandidateDecode:
    empty_response = not bool((raw_text or "").strip())
    try:
        action = decoder.decode_execute(raw_text or "", has_tool_call_tag=False)
    except Exception as exc:
        return CandidateDecode(
            action=[],
            status="decode_error",
            decode_error=str(exc),
            empty_response=empty_response,
            tool_call_parse_success=False,
        )
    if is_empty_execute_response(action):
        if empty_response:
            status = "empty_response"
        elif "<tool_call" in (raw_text or ""):
            status = "invalid_format"
        else:
            status = "empty_action"
        return CandidateDecode(
            action=[],
            status=status,
            decode_error=None,
            empty_response=empty_response,
            tool_call_parse_success=False,
        )
    return CandidateDecode(
        action=list(action),
        status="decoded_action",
        decode_error=None,
        empty_response=False,
        tool_call_parse_success=True,
    )


def action_matches(candidate_action: Any, reference_action: Any) -> bool:
    return normalize_action_text(candidate_action or []) == normalize_action_text(
        reference_action or []
    )


def _first_tool_call(action: Any) -> tuple[str | None, Any]:
    if not isinstance(action, list) or not action:
        return None, None
    first = action[0]
    if isinstance(first, str):
        try:
            first = json.loads(first)
        except Exception:
            return None, None
    if not isinstance(first, dict):
        return None, None
    return first.get("name"), first.get("arguments")


def state_matches(state: Any, reference_state: Any) -> bool | None:
    if reference_state is None:
        return None
    return normalize_state(state) == normalize_state(reference_state)


def serialization_roundtrip(
    decoder: Any,
    raw_text: str,
    executed_action: Any,
) -> dict[str, Any]:
    try:
        decoded_written = decoder.decode_execute(raw_text or "", has_tool_call_tag=False)
        error = None
    except Exception as exc:
        decoded_written = []
        error = str(exc)
    mismatch = not action_matches(decoded_written, executed_action or [])
    return {
        "serialization_mismatch": mismatch,
        "decoded_written_action": decoded_written,
        "serialization_decode_error": error,
        "serialized_raw_text": raw_text,
    }


def build_step_record(
    *,
    sample_id: str,
    turn_idx: int,
    step_idx: int,
    global_step: int,
    candidate_raw_text: str,
    candidate_action: Sequence[str],
    candidate_status: str,
    reference_step: dict[str, Any] | None,
    alignment_status: str,
    executed_action: Sequence[str],
    state: Any,
    decode_error: str | None = None,
    empty_response: bool = False,
    execution_error: str | None = None,
    candidate_assistant_message: dict[str, Any] | None = None,
    executed_assistant_message: dict[str, Any] | None = None,
    execution_results: Sequence[Any] | None = None,
    history_execution_results: Sequence[Any] | None = None,
    oracle_corrected: bool = False,
    oracle_suppressed_extra_action: bool = False,
    response_matches_reference: bool | None = None,
    candidate_response_matches_reference: bool | None = None,
    roundtrip: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_action = (
        list(reference_step.get("decoded_action") or []) if reference_step else None
    )
    reference_state = reference_step.get("state") if reference_step else None
    reference_has_action = bool(
        reference_action and not is_empty_execute_response(reference_action)
    )
    candidate_matches = (
        False
        if candidate_status in {"decode_error", "invalid_format"}
        and reference_has_action
        else action_matches(candidate_action, reference_action or [])
    )
    executed_matches = action_matches(executed_action, reference_action or [])
    state_match = state_matches(state, reference_state)
    tool_name, arguments = _first_tool_call(candidate_action)
    record = {
        "schema_version": 2,
        "id": sample_id,
        "turn": turn_idx,
        "step": step_idx,
        "user_turn": turn_idx,
        "step_in_turn": step_idx,
        "global_step": global_step,
        "candidate_global_step": global_step,
        "reference_global_step": reference_step.get("global_step") if reference_step else None,
        "alignment_status": alignment_status,
        "candidate_raw_text": candidate_raw_text,
        "candidate_action": list(candidate_action or []),
        "candidate_status": candidate_status,
        "has_tool_call": bool(candidate_action) or "<tool_call" in (candidate_raw_text or ""),
        "tool_call_parse_success": candidate_status == "decoded_action",
        "tool_name": tool_name,
        "arguments": arguments,
        "reference_action": reference_action,
        "reference_has_action": reference_has_action,
        "candidate_action_matches_reference": candidate_matches,
        "candidate_action_drift": not candidate_matches,
        "executed_action": list(executed_action or []),
        "executed_action_matches_reference": executed_matches,
        "executed_action_drift": not executed_matches,
        "state": state,
        "reference_state": reference_state,
        "state_matches_reference": state_match,
        "state_drift": state_match is False,
        "decode_error": decode_error,
        "empty_response": empty_response,
        "execution_error": execution_error,
        "execution_success": (
            bool(execution_results)
            and execution_error is None
            and bool(executed_action)
        ),
        "assistant_message": candidate_assistant_message,
        "executed_assistant_message": executed_assistant_message,
        "decoded_action": list(candidate_action or []),
        "action_matches_reference": candidate_matches,
        "execution_results": list(execution_results or []),
        "history_execution_results": list(history_execution_results or []),
        "oracle_corrected": oracle_corrected,
        "oracle_suppressed_extra_action": oracle_suppressed_extra_action,
        "response_matches_reference": response_matches_reference,
        "candidate_response_matches_reference": candidate_response_matches_reference,
    }
    if roundtrip:
        record.update(roundtrip)
    if extra:
        record.update(extra)
    return record


def mark_first_divergence(stats: Any, record: dict[str, Any]) -> None:
    location = {
        "turn": int(record["turn"]),
        "step": int(record["step"]),
        "global_step": int(record["global_step"]),
    }
    if (
        getattr(stats, "first_action_divergence", None) is None
        and record.get("candidate_action_drift")
    ):
        stats.first_action_divergence = dict(location)
    if (
        getattr(stats, "first_state_divergence", None) is None
        and record.get("state_drift")
    ):
        stats.first_state_divergence = dict(location)
