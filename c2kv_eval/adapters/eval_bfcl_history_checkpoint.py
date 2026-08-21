from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mt_utils
from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.utils import (
    load_dataset_entry,
    make_json_serializable,
    sort_file_content_by_id,
    sort_key,
)

from c2kv_eval.adapters.bfcl_history_branch import _assistant_text
from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    HistoryDriftRunner,
    _assistant_history_message,
    _history_units,
    _latest_user_query_index,
    _normalize_action_text,
    _normalize_state,
    _post_json,
    _render_history_unit,
    _state_log,
    _token_count,
    _tool_payload,
)
from c2kv_eval.adapters.history_step_common import (
    action_matches,
    build_step_record,
    decode_candidate,
    mark_first_divergence,
    reference_by_turn_step,
    reference_step_for,
    serialization_roundtrip,
)


CHECKPOINT_MODES = {
    "current_step",
    "first_bad_suffix",
    "oracle_first_bad",
    "recent2",
    "since_checkpoint",
    "whole_segment",
    "full_history",
}

KV_VERIFIERS = {"instant_kv", "cumulative_kv", "kv_divergence"}
ATTRIBUTION_MODES = {"oracle_first_bad", "whole_segment", "heuristic"}
ROLLBACK_BACKENDS = {"message_replay", "kv_restore"}


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        make_json_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_numeric_vector(value: Any, *, max_items: int = 8192) -> list[float]:
    values: list[float] = []

    def visit(obj: Any) -> None:
        if len(values) >= max_items:
            return
        if obj is None:
            return
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            number = float(obj)
            if math.isfinite(number):
                values.append(number)
            return
        if isinstance(obj, dict):
            for key in (
                "hidden_states",
                "last_hidden_state",
                "readout",
                "values",
                "data",
            ):
                if key in obj:
                    visit(obj[key])
                    return
            for item in obj.values():
                visit(item)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)

    visit(value)
    return values


def _cosine_distance(left: list[float], right: list[float]) -> float | None:
    size = min(len(left), len(right))
    if size == 0:
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(size):
        a = float(left[index])
        b = float(right[index])
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    cosine = dot / math.sqrt(left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def _distribution_from_logprobs(logprobs: Any) -> dict[str, float]:
    if not logprobs:
        return {}
    if isinstance(logprobs, dict):
        if isinstance(logprobs.get("content"), list) and logprobs["content"]:
            first = logprobs["content"][0]
            top = first.get("top_logprobs") if isinstance(first, dict) else None
            if isinstance(top, list):
                out = {}
                for item in top:
                    if not isinstance(item, dict):
                        continue
                    token = item.get("token")
                    value = item.get("logprob")
                    if token is not None and isinstance(value, (int, float)):
                        out[str(token)] = float(value)
                return out
        for key in ("top_logprobs", "output_top_logprobs"):
            value = logprobs.get(key)
            if isinstance(value, list):
                return _distribution_from_logprobs(value)
    if isinstance(logprobs, list):
        if not logprobs:
            return {}
        first = logprobs[0]
        if isinstance(first, dict):
            if all(isinstance(v, (int, float)) for v in first.values()):
                return {str(k): float(v) for k, v in first.items()}
            return _distribution_from_logprobs(first)
        if (
            isinstance(first, (list, tuple))
            and len(first) >= 2
            and isinstance(first[1], (int, float))
        ):
            return {str(item[0]): float(item[1]) for item in logprobs if len(item) >= 2}
    return {}


def _entropy_from_log_probs(log_probs: dict[str, float]) -> float | None:
    if not log_probs:
        return None
    probs = [math.exp(value) for value in log_probs.values() if math.isfinite(value)]
    total = sum(probs)
    if total <= 0.0:
        return None
    return -sum((p / total) * math.log(max(p / total, 1e-30)) for p in probs)


def _top1_top2_margin(log_probs: dict[str, float]) -> float | None:
    if len(log_probs) < 2:
        return None
    values = sorted(log_probs.values(), reverse=True)
    return values[0] - values[1]


def _kl_from_log_probs(
    full_log_probs: dict[str, float],
    c2kv_log_probs: dict[str, float],
) -> float | None:
    if not full_log_probs or not c2kv_log_probs:
        return None
    keys = set(full_log_probs) | set(c2kv_log_probs)
    floor = -30.0
    full_weights = {key: math.exp(full_log_probs.get(key, floor)) for key in keys}
    c2kv_weights = {key: math.exp(c2kv_log_probs.get(key, floor)) for key in keys}
    full_total = sum(full_weights.values())
    c2kv_total = sum(c2kv_weights.values())
    if full_total <= 0.0 or c2kv_total <= 0.0:
        return None
    kl = 0.0
    for key in keys:
        p = full_weights[key] / full_total
        q = max(c2kv_weights[key] / c2kv_total, 1e-30)
        if p > 0.0:
            kl += p * math.log(p / q)
    return kl


def _instance_key(model_name: str, test_entry_id: str, class_name: str) -> str:
    return re.sub(r"[-./:]", "_", f"{model_name}_{test_entry_id}_{class_name}_instance")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


class HistoryCheckpointRunner(HistoryDriftRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        drift_args = deepcopy(args)
        drift_args.mode = "history_c2kv4_closed_loop"
        super().__init__(drift_args)
        self.checkpoint_interval = args.checkpoint_interval
        self.requested_verifier = args.verifier
        self.verifier = args.verifier
        self.verify_threshold = args.verify_threshold
        self.recovery_mode = args.recovery_mode
        self.verify_layers = args.verify_layers
        self.online_verify = bool(args.online_verify)
        self.reuse_candidate_readout = bool(args.reuse_candidate_readout)
        self.attribution = args.attribution
        if self.attribution == "auto":
            if self.recovery_mode in {"whole_segment", "since_checkpoint", "full_history"}:
                self.attribution = "whole_segment"
            else:
                self.attribution = "oracle_first_bad"
        self.attribution_safety_margin = int(args.attribution_safety_margin)
        self.rollback_backend = args.rollback_backend
        self._cumulative_divergence = 0.0
        if self.verifier in {"instant_kv", "cumulative_kv"}:
            self.verifier = "kv_divergence"

    def _kv_checkpoint_metadata(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        global_step: int,
        checkpoint_id: int,
    ) -> dict[str, Any]:
        history_units = _history_units(messages)
        return {
            "available": False,
            "checkpoint_id": checkpoint_id,
            "global_step": global_step,
            "message_count": len(messages),
            "history_units": len(history_units),
            "prompt_token_estimate": _token_count(self.tokenizer, messages),
            "cache_handle": None,
            "sequence_length": None,
            "position_metadata": None,
            "c2kv_cache_metadata": [
                {
                    "message_index": index,
                    "c2kv_key_hash": message.get("c2kv_key_hash"),
                }
                for index, message in enumerate(messages)
                if message.get("c2kv_key_hash")
            ],
            "limitation": (
                "OpenAI-compatible SGLang HTTP API does not expose a raw "
                "KV cache handle or restore endpoint for this runner."
            ),
        }

    def _restore_with_backend(
        self,
        *,
        test_entry_id: str,
        snapshot: dict[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
        list[str],
        list[int],
        list[int],
        list[float],
        dict[str, Any],
        list[dict[str, Any]],
        bool,
        dict[str, Any],
    ]:
        start = time.perf_counter()
        backend_info = {
            "rollback_backend": self.rollback_backend,
            "rollback_backend_requested": self.rollback_backend,
            "kv_restore_success": False,
            "kv_restore_fallback": False,
            "kv_restore_fallback_reason": None,
            "kv_reused_tokens": 0,
            "kv_recomputed_tokens": 0,
            "message_replay_prefill_tokens": 0,
            "restore_latency_sec": 0.0,
            "rollback_latency_sec": 0.0,
            "kv_checkpoint_metadata": snapshot.get("kv_checkpoint_metadata"),
        }
        if self.rollback_backend == "kv_restore":
            backend_info.update(
                {
                    "kv_restore_fallback": True,
                    "kv_restore_fallback_reason": (
                        "raw_kv_restore_not_supported_by_sglang_http_api"
                    ),
                }
            )
        (
            messages,
            instances,
            current_turn_response,
            current_turn_inputs,
            current_turn_outputs,
            current_turn_latency,
            turn_log,
            restored_state,
            matches,
        ) = self._restore_snapshot(test_entry_id=test_entry_id, snapshot=snapshot)
        elapsed = time.perf_counter() - start
        backend_info["restore_latency_sec"] = elapsed
        backend_info["rollback_latency_sec"] = elapsed
        backend_info["message_replay_prefill_tokens"] = _token_count(
            self.tokenizer,
            messages,
        )
        return (
            messages,
            instances,
            current_turn_response,
            current_turn_inputs,
            current_turn_outputs,
            current_turn_latency,
            turn_log,
            restored_state,
            matches,
            backend_info,
        )

    @staticmethod
    def _action_text(action: list[str]) -> str:
        return json.dumps(action or [], ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _parse_action_objects(action: list[str]) -> list[dict[str, Any]]:
        out = []
        for item in action or []:
            value: Any = item
            if isinstance(item, str):
                try:
                    value = json.loads(item)
                except Exception:
                    value = item
            if isinstance(value, dict):
                out.append(value)
        return out

    def _argument_grounding_score(
        self,
        *,
        action: list[str],
        messages: Sequence[dict[str, Any]],
    ) -> float:
        action_objects = self._parse_action_objects(action)
        if not action_objects:
            return 1.0 if is_empty_execute_response(action) else 0.0
        recent_text = "\n".join(
            str(message.get("content") or "")
            for message in messages[-12:]
            if message.get("role") in {"user", "tool", "assistant"}
        ).lower()
        values = []
        for obj in action_objects:
            args = obj.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"value": args}
            if isinstance(args, dict):
                for value in args.values():
                    if isinstance(value, (str, int, float)) and str(value).strip():
                        text = str(value).strip().lower()
                        if len(text) >= 3:
                            values.append(text)
        if not values:
            return 1.0
        hits = sum(1 for value in values if value in recent_text)
        return hits / len(values)

    def _repeat_action_score(
        self,
        *,
        action: list[str],
        segment_infos: Sequence[dict[str, Any]],
    ) -> float:
        current = self._action_text(action)
        if current == "[]":
            return 0.0
        previous = [
            self._action_text(info.get("candidate_action") or [])
            for info in segment_infos
        ]
        return 1.0 if current in previous else 0.0

    @staticmethod
    def _observation_anomaly_score(
        *,
        execution_results: Sequence[Any],
        execution_error: str | None,
        candidate_status: str,
        action: list[str],
    ) -> float:
        if candidate_status in {"decode_error", "invalid_format"}:
            return 1.0
        if execution_error:
            return 1.0
        if is_empty_execute_response(action):
            return 0.5
        if not execution_results:
            return 0.5
        text = "\n".join(str(item) for item in execution_results).lower()
        bad_markers = [
            "error",
            "exception",
            "failed",
            "invalid",
            "not found",
            "no result",
            "empty",
        ]
        return 1.0 if any(marker in text for marker in bad_markers) else 0.0

    def _heuristic_attributes(
        self,
        *,
        info: dict[str, Any],
        segment_infos: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        action = info.get("candidate_action") or []
        hard_error = bool(
            info.get("candidate_status") in {"decode_error", "invalid_format"}
            or info.get("execution_error")
        )
        grounding = self._argument_grounding_score(
            action=action,
            messages=info.get("micro_snapshot", {}).get("messages") or [],
        )
        repeat_score = self._repeat_action_score(
            action=action,
            segment_infos=segment_infos,
        )
        observation = self._observation_anomaly_score(
            execution_results=info.get("execution_results") or [],
            execution_error=info.get("execution_error"),
            candidate_status=info.get("candidate_status") or "",
            action=action,
        )
        tool_transition_anomaly = 1.0 if repeat_score >= 1.0 and observation > 0.0 else 0.0
        representation_jump = None
        risk_score = (
            (1.0 if hard_error else 0.0) * 10.0
            + (1.0 - grounding) * 4.0
            + repeat_score * 2.0
            + observation * 3.0
            + tool_transition_anomaly * 2.0
        )
        return {
            "hard_error": hard_error,
            "argument_grounding_score": grounding,
            "argument_grounding_failure": grounding < 0.34,
            "repeat_action_score": repeat_score,
            "tool_transition_anomaly": tool_transition_anomaly,
            "observation_anomaly": observation,
            "representation_jump": representation_jump,
            "risk_score": risk_score,
        }

    def _predict_first_bad_index(
        self,
        *,
        segment_infos: Sequence[dict[str, Any]],
        oracle_first_bad_index: int,
    ) -> tuple[int, dict[str, Any]]:
        if self.attribution == "whole_segment":
            predicted = 0
            reason = "whole_segment"
        elif self.attribution == "oracle_first_bad":
            predicted = oracle_first_bad_index
            reason = "oracle_first_bad"
        elif self.attribution == "heuristic":
            attrs = [info.get("heuristic_attributes") or {} for info in segment_infos]
            hard = [
                index for index, attr in enumerate(attrs) if attr.get("hard_error")
            ]
            grounding = [
                index
                for index, attr in enumerate(attrs)
                if attr.get("argument_grounding_failure")
            ]
            observation = [
                index
                for index, attr in enumerate(attrs)
                if float(attr.get("observation_anomaly") or 0.0) >= 1.0
            ]
            if hard:
                predicted = hard[0]
                reason = "hard_error"
            elif grounding:
                predicted = grounding[0]
                reason = "argument_grounding_failure"
            elif observation:
                predicted = observation[0]
                reason = "observation_anomaly"
            else:
                predicted = max(
                    range(len(segment_infos)),
                    key=lambda idx: float(
                        attrs[idx].get("risk_score") or 0.0
                    ),
                )
                reason = "max_risk_score"
        else:
            raise ValueError(f"Unknown attribution={self.attribution!r}")
        raw_predicted = predicted
        predicted = max(0, predicted - self.attribution_safety_margin)
        return predicted, {
            "attribution": self.attribution,
            "oracle_first_bad_index": oracle_first_bad_index,
            "predicted_first_bad_index": predicted,
            "raw_predicted_first_bad_index": raw_predicted,
            "attribution_reason": reason,
            "attribution_safety_margin": self.attribution_safety_margin,
            "exact_attribution": predicted == oracle_first_bad_index,
            "within1_attribution": abs(predicted - oracle_first_bad_index) <= 1,
            "under_rollback": predicted > oracle_first_bad_index,
            "over_rollback": predicted < oracle_first_bad_index,
            "over_rollback_steps": max(0, oracle_first_bad_index - predicted),
            "predicted_rollback_depth": len(segment_infos) - predicted,
            "oracle_rollback_depth": len(segment_infos) - oracle_first_bad_index,
        }

    def _restore_instances(
        self,
        test_entry_id: str,
        involved_instances: dict[str, Any],
    ) -> None:
        for class_name, instance in involved_instances.items():
            mt_utils.__dict__[
                _instance_key(
                    self.decoder.model_name_underline_replaced,
                    test_entry_id,
                    class_name,
                )
            ] = deepcopy(instance)

    def _reference_maps(self, sample_id: str) -> tuple[dict[tuple[int, int], dict[str, Any]], list[Any]]:
        row = self.reference_by_id.get(sample_id) or {}
        steps = row.get("drift_steps") or []
        return reference_by_turn_step(steps), row.get("result") or []

    def _message_fingerprint(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for index, message in enumerate(messages):
            out.append(
                {
                    "index": index,
                    "role": message.get("role"),
                    "content_hash": _stable_hash(message.get("content")),
                    "content": message.get("content"),
                    "has_c2kv_key_hash": bool(message.get("c2kv_key_hash")),
                    "c2kv_key_hash": message.get("c2kv_key_hash"),
                    "tool_call_hash": _stable_hash(message.get("tool_calls")),
                    "tool_call_id": message.get("tool_call_id"),
                }
            )
        return out

    def _message_diffs(
        self,
        left: Sequence[dict[str, Any]],
        right: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        diffs = []
        max_len = max(len(left), len(right))
        for index in range(max_len):
            if index >= len(left) or index >= len(right):
                diffs.append(
                    {
                        "index": index,
                        "field": "__len__",
                        "static": index < len(left),
                        "checkpoint": index < len(right),
                    }
                )
                continue
            for field in (
                "role",
                "content",
                "tool_calls",
                "tool_call_id",
                "c2kv_key_hash",
            ):
                if left[index].get(field) != right[index].get(field):
                    diffs.append(
                        {
                            "index": index,
                            "field": field,
                            "static": left[index].get(field),
                            "checkpoint": right[index].get(field),
                        }
                    )
        return diffs

    def _maybe_write_recent2_parity(
        self,
        *,
        checkpoint_messages: Sequence[dict[str, Any]],
        recovery_messages: Sequence[dict[str, Any]],
        recovery_debug: dict[str, Any],
    ) -> None:
        if os.environ.get("C2KV_DEBUG_RECENT2_PARITY") != "1":
            return
        output_path = os.environ.get("C2KV_DEBUG_RECENT2_PARITY_PATH")
        if output_path:
            path = Path(output_path)
        else:
            path = Path(self.args.summary_path).parent / "recent2_payload_parity.json"
        old_mode = self.mode
        try:
            probe_stats = DriftStats(
                "recent2_payload_parity",
                "history_recent2_full_rest_c2kv4",
                self.ratio,
            )
            self.mode = "history_recent2_full_rest_c2kv4"
            static_messages = self._build_request_messages(
                checkpoint_messages,
                probe_stats,
            )
        finally:
            self.mode = old_mode
        payload = {
            "static_message_hash": _stable_hash(static_messages),
            "checkpoint_message_hash": _stable_hash(recovery_messages),
            "static_message_count": len(static_messages),
            "checkpoint_message_count": len(recovery_messages),
            "static_history_units": len(_history_units(static_messages)),
            "checkpoint_history_units": len(_history_units(recovery_messages)),
            "static_token_count": _token_count(self.tokenizer, static_messages),
            "checkpoint_token_count": _token_count(self.tokenizer, recovery_messages),
            "recovery_debug": recovery_debug,
            "different_fields": self._message_diffs(
                static_messages,
                recovery_messages,
            ),
            "static_messages": self._message_fingerprint(static_messages),
            "checkpoint_messages": self._message_fingerprint(recovery_messages),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _build_recovery_messages(
        self,
        checkpoint_messages: list[dict[str, Any]],
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        if self.recovery_mode in {
            "current_step",
            "first_bad_suffix",
            "oracle_first_bad",
            "since_checkpoint",
            "whole_segment",
            "full_history",
        }:
            messages = deepcopy(checkpoint_messages)
            return (
                messages,
                _token_count(self.tokenizer, messages),
                {
                    "recovery_prompt_mode": self.recovery_mode,
                    "c2kv_history_units": 0,
                    "full_history_units": len(_history_units(messages)),
                    "current_messages": 0,
                },
            )
        if self.recovery_mode == "recent2":
            latest_query_index = _latest_user_query_index(checkpoint_messages)
            completed = list(checkpoint_messages[:latest_query_index])
            current = deepcopy(list(checkpoint_messages[latest_query_index:]))
            units = _history_units(completed)
            keep_full_from = max(0, len(units) - self.recent_full_units)
            messages: list[dict[str, Any]] = []
            c2kv_units = 0
            full_units = 0
            for unit_index, unit in enumerate(units):
                if unit_index >= keep_full_from:
                    full_tokens = _token_count(self.tokenizer, unit)
                    stats.original_history_tokens += full_tokens
                    stats.effective_history_tokens += full_tokens
                    messages.extend(deepcopy(unit))
                    full_units += 1
                    continue
                text = _render_history_unit(unit)
                full_tokens = _token_count(
                    self.tokenizer,
                    [{"role": "user", "content": text}],
                )
                record = self._extract_history_unit(text, stats)
                stats.original_history_tokens += int(record.original_seq_len or full_tokens)
                if record.success and record.key_hash:
                    stats.effective_history_tokens += int(
                        record.gist_len or record.original_seq_len or full_tokens
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": text,
                            "c2kv_key_hash": record.key_hash,
                        }
                    )
                    c2kv_units += 1
                else:
                    stats.effective_history_tokens += full_tokens
                    messages.append({"role": "user", "content": text})
                    full_units += 1
            messages.extend(current)
            recovery_debug = {
                "recovery_prompt_mode": "recent2",
                "c2kv_history_units": c2kv_units,
                "full_history_units": full_units,
                "current_messages": len(current),
                "completed_units": len(units),
                "recent_full_units": self.recent_full_units,
            }
            self._maybe_write_recent2_parity(
                checkpoint_messages=checkpoint_messages,
                recovery_messages=messages,
                recovery_debug=recovery_debug,
            )
            return (
                messages,
                _token_count(self.tokenizer, messages),
                recovery_debug,
            )
        raise ValueError(f"Unknown recovery mode: {self.recovery_mode}")

    def _verify_oracle(
        self,
        *,
        decoded_action: list[str],
        state_after_step: list[dict[str, Any]],
        ref_step: dict[str, Any] | None,
    ) -> dict[str, Any]:
        reference_action = ref_step.get("decoded_action") if ref_step else None
        candidate_action_matches = action_matches(decoded_action, reference_action or [])
        if ref_step is None:
            state_matches = is_empty_execute_response(decoded_action)
        else:
            state_matches = (
                _normalize_state(state_after_step)
                == _normalize_state(ref_step.get("state"))
            )
        harmful = (not candidate_action_matches) or (state_matches is False)
        return {
            "kv_divergence": None,
            "cumulative_divergence": None,
            "candidate_action_matches_reference": candidate_action_matches,
            "state_matches_reference": state_matches,
            "verify_failed": harmful,
        }

    def _query_with_raw(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stats: DriftStats,
        *,
        max_completion_tokens: int | None = None,
        readout_probe: bool = False,
    ) -> tuple[str, dict[str, Any], float, dict[str, Any], dict[str, Any]]:
        prompt_tokens = _token_count(self.tokenizer, messages)
        max_tokens = max(1, max_completion_tokens or self.max_completion_tokens)
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": self.temperature,
            "max_completion_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if readout_probe:
            payload.update(
                {
                    "logprobs": True,
                    "top_logprobs": 20,
                    "return_hidden_states": True,
                }
            )
        start = time.perf_counter()
        data = _post_json(self.base_url, "/v1/chat/completions", payload, self.timeout)
        elapsed = time.perf_counter() - start
        choice = data.get("choices", [{}])[0] or {}
        message = choice.get("message", {}) or {}
        text = message.get("content") or ""
        tool_calls_text = self._tool_calls_to_text_for_query(message.get("tool_calls"))
        if tool_calls_text:
            text = (text + "\n" + tool_calls_text).strip() if text else tool_calls_text
        usage = data.get("usage") or {}
        stats.chat_calls += 1
        stats.chat_seconds += elapsed
        parsed_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or prompt_tokens),
            "completion_tokens": int(
                usage.get("completion_tokens")
                or len(self.tokenizer.encode(text, add_special_tokens=False))
            ),
        }
        return text, message, elapsed, parsed_usage, data

    @staticmethod
    def _tool_calls_to_text_for_query(tool_calls: Any) -> str:
        from c2kv_eval.adapters.bfcl_history_drift import _tool_calls_to_text

        return _tool_calls_to_text(tool_calls)

    def _readout_payload(
        self,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        choice = (raw.get("choices") or [{}])[0] or {}
        hidden = choice.get("hidden_states")
        source = "choice.hidden_states"
        if hidden is None:
            hidden = raw.get("hidden_states")
            source = "response.hidden_states"
        vector = _extract_numeric_vector(hidden)
        log_probs = _distribution_from_logprobs(choice.get("logprobs"))
        return {
            "vector": vector,
            "log_probs": log_probs,
            "readout_available": bool(vector),
            "readout_source": source if vector else None,
            "readout_dim": len(vector),
            "entropy": _entropy_from_log_probs(log_probs),
            "top1_top2_margin": _top1_top2_margin(log_probs),
        }

    def _verify_kv_divergence(
        self,
        *,
        full_messages: Sequence[dict[str, Any]],
        c2kv_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stats: DriftStats,
        candidate_raw: dict[str, Any] | None = None,
        candidate_usage: dict[str, Any] | None = None,
        candidate_elapsed: float | None = None,
    ) -> dict[str, Any]:
        _, _, full_elapsed, full_usage, full_raw = self._query_with_raw(
            full_messages,
            tools,
            stats,
            max_completion_tokens=2,
            readout_probe=True,
        )
        full_readout = self._readout_payload(full_raw)
        candidate_readout_reused = False
        if self.reuse_candidate_readout and candidate_raw:
            c2kv_readout = self._readout_payload(candidate_raw)
            candidate_readout_reused = bool(c2kv_readout["readout_available"])
        else:
            c2kv_readout = {
                "vector": [],
                "log_probs": {},
                "readout_available": False,
                "readout_source": None,
                "readout_dim": 0,
                "entropy": None,
                "top1_top2_margin": None,
            }
        if candidate_readout_reused:
            c2kv_elapsed = 0.0
            c2kv_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        else:
            _, _, c2kv_elapsed, c2kv_usage, c2kv_raw = self._query_with_raw(
                c2kv_messages,
                tools,
                stats,
                max_completion_tokens=2,
                readout_probe=True,
            )
            c2kv_readout = self._readout_payload(c2kv_raw)
        kv_divergence = _cosine_distance(
            full_readout["vector"],
            c2kv_readout["vector"],
        )
        full_log_probs = full_readout["log_probs"]
        c2kv_log_probs = c2kv_readout["log_probs"]
        logit_kl = _kl_from_log_probs(full_log_probs, c2kv_log_probs)
        if kv_divergence is not None:
            self._cumulative_divergence += kv_divergence
        score = (
            self._cumulative_divergence
            if self.requested_verifier == "cumulative_kv"
            else kv_divergence
        )
        verify_failed = (
            self.online_verify
            and score is not None
            and score >= self.verify_threshold
        )
        return {
            "kv_divergence": kv_divergence,
            "cumulative_divergence": (
                self._cumulative_divergence if kv_divergence is not None else None
            ),
            "logit_kl": logit_kl,
            "entropy": c2kv_readout["entropy"],
            "top1_top2_margin": c2kv_readout["top1_top2_margin"],
            "readout_available": bool(
                full_readout["readout_available"]
                and c2kv_readout["readout_available"]
            ),
            "full_readout_source": full_readout["readout_source"],
            "c2kv_readout_source": c2kv_readout["readout_source"],
            "candidate_readout_reused": candidate_readout_reused,
            "full_readout_dim": full_readout["readout_dim"],
            "c2kv_readout_dim": c2kv_readout["readout_dim"],
            "full_probe_seconds": full_elapsed,
            "c2kv_probe_seconds": c2kv_elapsed,
            "full_probe_prompt_tokens": full_usage["prompt_tokens"],
            "c2kv_probe_prompt_tokens": c2kv_usage["prompt_tokens"],
            "verify_failed": verify_failed,
        }

    def _snapshot(
        self,
        *,
        checkpoint_id: int,
        global_step: int,
        messages: list[dict[str, Any]],
        involved_instances: dict[str, Any],
        current_turn_response: list[str],
        current_turn_inputs: list[int],
        current_turn_outputs: list[int],
        current_turn_latency: list[float],
        turn_log: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "checkpoint_id": checkpoint_id,
            "global_step": global_step,
            "messages": deepcopy(messages),
            "instances": deepcopy(involved_instances),
            "state": _state_log(involved_instances),
            "kv_checkpoint_metadata": self._kv_checkpoint_metadata(
                messages=messages,
                global_step=global_step,
                checkpoint_id=checkpoint_id,
            ),
            "current_turn_response": deepcopy(current_turn_response),
            "current_turn_inputs": deepcopy(current_turn_inputs),
            "current_turn_outputs": deepcopy(current_turn_outputs),
            "current_turn_latency": deepcopy(current_turn_latency),
            "turn_log": deepcopy(turn_log),
        }

    def _restore_snapshot(
        self,
        *,
        test_entry_id: str,
        snapshot: dict[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
        list[str],
        list[int],
        list[int],
        list[float],
        dict[str, Any],
        list[dict[str, Any]],
        bool,
    ]:
        instances = deepcopy(snapshot["instances"])
        self._restore_instances(test_entry_id, instances)
        restored_state = _state_log(instances)
        matches = _normalize_state(restored_state) == _normalize_state(
            snapshot.get("state")
        )
        return (
            deepcopy(snapshot["messages"]),
            instances,
            deepcopy(snapshot["current_turn_response"]),
            deepcopy(snapshot["current_turn_inputs"]),
            deepcopy(snapshot["current_turn_outputs"]),
            deepcopy(snapshot["current_turn_latency"]),
            deepcopy(snapshot["turn_log"]),
            restored_state,
            matches,
        )

    def _execute_action(
        self,
        *,
        action: list[str],
        initial_config: dict[str, Any],
        involved_classes: list[str],
        test_entry_id: str,
        long_context: bool,
    ) -> tuple[list[Any], dict[str, Any] | None, str | None]:
        if is_empty_execute_response(action):
            return [], None, None
        try:
            execution_results, involved_instances = execute_multi_turn_func_call(
                action,
                initial_config,
                involved_classes,
                self.decoder.model_name_underline_replaced,
                test_entry_id,
                long_context=long_context,
                is_evaL_run=False,
            )
            return execution_results, involved_instances, None
        except Exception as exc:
            return [], None, str(exc)

    @staticmethod
    def _oracle_verify_stub(
        *,
        candidate_action_matches: bool,
        state_matches: bool | None,
    ) -> dict[str, Any]:
        harmful = (not candidate_action_matches) or (state_matches is False)
        return {
            "kv_divergence": None,
            "cumulative_divergence": None,
            "candidate_action_matches_reference": candidate_action_matches,
            "state_matches_reference": state_matches,
            "verify_failed": harmful,
            "logit_kl": None,
            "entropy": None,
            "top1_top2_margin": None,
            "readout_available": False,
            "full_readout_source": None,
            "c2kv_readout_source": None,
            "candidate_readout_reused": False,
            "full_readout_dim": 0,
            "c2kv_readout_dim": 0,
            "full_probe_seconds": 0.0,
            "full_probe_prompt_tokens": 0,
            "c2kv_probe_seconds": 0.0,
            "c2kv_probe_prompt_tokens": 0,
        }

    def _checkpoint_step_row(
        self,
        *,
        step_record: dict[str, Any],
        checkpoint_id: int,
        segment_index: int,
        segment_start_step: int,
        segment_length: int,
        verify: dict[str, Any],
        verify_triggered: bool,
        refresh_triggered: bool,
        regenerated_steps: int,
        full_regenerated_tokens: int,
        recovery_debug: dict[str, Any],
        regenerated_same_as_candidate: bool | None,
        history_prompt_tokens: int,
        request_history_tokens: int,
        full_history_tokens: int,
        candidate_readout_reused: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "id": step_record["id"],
            "global_step": step_record["global_step"],
            "candidate_global_step": step_record["candidate_global_step"],
            "reference_global_step": step_record.get("reference_global_step"),
            "alignment_status": step_record.get("alignment_status"),
            "turn": step_record["turn"],
            "step": step_record["step"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_interval": self.checkpoint_interval,
            "segment_index": segment_index,
            "segment_start_step": segment_start_step,
            "segment_length": segment_length,
            "verifier": self.verifier,
            "requested_verifier": self.requested_verifier,
            "verify_threshold": self.verify_threshold,
            "online_verify": self.online_verify,
            "kv_divergence": verify.get("kv_divergence"),
            "cumulative_divergence": verify.get("cumulative_divergence"),
            "logit_kl": verify.get("logit_kl"),
            "entropy": verify.get("entropy"),
            "top1_top2_margin": verify.get("top1_top2_margin"),
            "readout_available": verify.get("readout_available"),
            "full_readout_source": verify.get("full_readout_source"),
            "c2kv_readout_source": verify.get("c2kv_readout_source"),
            "candidate_readout_reused": candidate_readout_reused
            or bool(verify.get("candidate_readout_reused")),
            "full_readout_dim": verify.get("full_readout_dim"),
            "c2kv_readout_dim": verify.get("c2kv_readout_dim"),
            "full_probe_seconds": verify.get("full_probe_seconds"),
            "c2kv_probe_seconds": verify.get("c2kv_probe_seconds"),
            "full_probe_prompt_tokens": verify.get("full_probe_prompt_tokens", 0),
            "c2kv_probe_prompt_tokens": verify.get("c2kv_probe_prompt_tokens", 0),
            "verify_triggered": verify_triggered,
            "refresh_triggered": refresh_triggered,
            "rollback_triggered": refresh_triggered,
            "recovery_mode": self.recovery_mode,
            "regenerated_steps": regenerated_steps,
            "full_regenerated_tokens": full_regenerated_tokens,
            "recovery_prompt_mode": recovery_debug.get("recovery_prompt_mode"),
            "c2kv_history_units": recovery_debug.get("c2kv_history_units", 0),
            "full_history_units": recovery_debug.get("full_history_units", 0),
            "current_messages": recovery_debug.get("current_messages", 0),
            "regenerated_same_as_candidate": regenerated_same_as_candidate,
            "candidate_status": step_record.get("candidate_status"),
            "decode_error": step_record.get("decode_error"),
            "empty_response": step_record.get("empty_response"),
            "candidate_action_drift": step_record.get("candidate_action_drift"),
            "executed_action_drift": step_record.get("executed_action_drift"),
            "state_drift": step_record.get("state_drift"),
            "serialization_mismatch": step_record.get("serialization_mismatch"),
            "executable": not is_empty_execute_response(
                step_record.get("candidate_action") or []
            ),
            "executed_executable": not is_empty_execute_response(
                step_record.get("executed_action") or []
            ),
            "history_prompt_tokens": history_prompt_tokens,
            "request_history_tokens": request_history_tokens,
            "full_history_tokens": full_history_tokens,
        }

    def _make_final_step_record(
        self,
        *,
        spec_info: dict[str, Any],
        executed_text: str,
        executed_message: dict[str, Any],
        executed_action: list[str],
        execution_results: list[Any],
        state_after_step: list[dict[str, Any]],
        execution_error: str | None,
        global_step: int,
        oracle_corrected: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        roundtrip = serialization_roundtrip(
            self.decoder,
            executed_text,
            executed_action,
        )
        return build_step_record(
            sample_id=spec_info["sample_id"],
            turn_idx=spec_info["turn_idx"],
            step_idx=spec_info["step_idx"],
            global_step=global_step,
            candidate_raw_text=spec_info["candidate_text"],
            candidate_action=spec_info["candidate_action"],
            candidate_status=spec_info["candidate_status"],
            reference_step=spec_info["ref_step"],
            alignment_status=spec_info["alignment_status"],
            executed_action=executed_action,
            state=state_after_step,
            decode_error=spec_info["candidate_decode_error"],
            empty_response=spec_info["candidate_empty_response"],
            execution_error=execution_error,
            candidate_assistant_message=spec_info["candidate_assistant"],
            executed_assistant_message=executed_message,
            execution_results=execution_results,
            history_execution_results=execution_results,
            oracle_corrected=oracle_corrected,
            response_matches_reference=None,
            candidate_response_matches_reference=None,
            roundtrip=roundtrip,
            extra=extra,
        )

    def _run_sample_checkpoint_impl_oracle_multistep(
        self,
        test_case: dict[str, Any],
        stats: DriftStats,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        if self.recovery_mode == "recent2" and self.checkpoint_interval != 1:
            raise NotImplementedError(
                "recent2 multi-step rollback is intentionally not compared yet; "
                "use current_step or since_checkpoint for checkpoint_interval > 1."
            )

        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        tools = _tool_payload(test_case["function"])
        long_context = "long_context" in test_category or "composite" in test_category
        ref_map, reference_result = self._reference_maps(test_entry_id)

        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.decoder.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )

        inference_log: list[Any] = []
        initial_state = _state_log(involved_instances)
        if initial_state:
            inference_log.append(initial_state)

        messages: list[dict[str, Any]] = []
        all_model_response: list[list[str]] = []
        input_token_count: list[list[int]] = []
        output_token_count: list[list[int]] = []
        latency: list[list[float]] = []
        checkpoint_steps: list[dict[str, Any]] = []
        checkpoint_segments: list[dict[str, Any]] = []
        drift_steps: list[dict[str, Any]] = []
        verify_count = 0
        refresh_count = 0
        kv_restore_success_count = 0
        kv_restore_fallback_count = 0
        kv_reused_tokens_total = 0
        kv_recomputed_tokens_total = 0
        message_replay_prefill_tokens_total = 0
        rollback_latency_total = 0.0
        restore_latency_total = 0.0
        regenerated_steps_total = 0
        full_regenerated_tokens_total = 0
        checkpoint_id = 0
        force_quit = False

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}
            count = 0

            while True:
                checkpoint_id += 1
                segment_start_step = len(drift_steps)
                segment_start_turn_step = count
                segment_checkpoint = self._snapshot(
                    checkpoint_id=checkpoint_id,
                    global_step=segment_start_step,
                    messages=messages,
                    involved_instances=involved_instances,
                    current_turn_response=current_turn_response,
                    current_turn_inputs=current_turn_inputs,
                    current_turn_outputs=current_turn_outputs,
                    current_turn_latency=current_turn_latency,
                    turn_log=turn_log,
                )
                segment_infos: list[dict[str, Any]] = []
                terminal_after_segment = False

                for segment_index in range(self.checkpoint_interval):
                    micro_snapshot = self._snapshot(
                        checkpoint_id=checkpoint_id,
                        global_step=segment_start_step + len(segment_infos),
                        messages=messages,
                        involved_instances=involved_instances,
                        current_turn_response=current_turn_response,
                        current_turn_inputs=current_turn_inputs,
                        current_turn_outputs=current_turn_outputs,
                        current_turn_latency=current_turn_latency,
                        turn_log=turn_log,
                    )
                    request_messages = self._build_request_messages(messages, stats)
                    (
                        candidate_text,
                        candidate_message,
                        candidate_elapsed,
                        candidate_usage,
                        candidate_raw,
                    ) = self._query_with_raw(
                        request_messages,
                        tools,
                        stats,
                        readout_probe=False,
                    )
                    candidate_assistant = _assistant_history_message(
                        candidate_text,
                        candidate_message.get("tool_calls"),
                    )
                    candidate = decode_candidate(self.decoder, candidate_text)
                    candidate_action = candidate.action
                    messages.append(candidate_assistant)

                    execution_results, next_instances, execution_error = (
                        self._execute_action(
                            action=candidate_action,
                            initial_config=initial_config,
                            involved_classes=involved_classes,
                            test_entry_id=test_entry_id,
                            long_context=long_context,
                        )
                    )
                    if next_instances is not None:
                        involved_instances = next_instances
                    for idx, execution_result in enumerate(execution_results):
                        messages.append(
                            {
                                "role": "tool",
                                "content": execution_result,
                                "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                            }
                        )
                    state_after_candidate = _state_log(involved_instances)
                    ref_step, alignment_status = reference_step_for(
                        ref_map,
                        reference_result,
                        turn_idx,
                        count,
                        fallback_state=state_after_candidate,
                    )
                    reference_action = ref_step.get("decoded_action") if ref_step else None
                    candidate_matches = action_matches(
                        candidate_action,
                        reference_action or [],
                    )
                    if candidate.status in {"decode_error", "invalid_format"} and ref_step:
                        candidate_matches = False
                    reference_state = ref_step.get("state") if ref_step else None
                    state_matches = (
                        None
                        if reference_state is None
                        else _normalize_state(state_after_candidate)
                        == _normalize_state(reference_state)
                    )
                    verify = self._oracle_verify_stub(
                        candidate_action_matches=candidate_matches,
                        state_matches=state_matches,
                    )
                    current_turn_response.append(candidate_text)
                    current_turn_inputs.append(candidate_usage["prompt_tokens"])
                    current_turn_outputs.append(candidate_usage["completion_tokens"])
                    current_turn_latency.append(candidate_elapsed)
                    step_log = [
                        {"role": "assistant", "content": candidate_text},
                        {
                            "role": "handler_log",
                            "content": (
                                "Successfully decoded model response."
                                if candidate.status == "decoded_action"
                                else f"Candidate status: {candidate.status}."
                            ),
                            "model_response_decoded": candidate_action,
                            "candidate_status": candidate.status,
                            "candidate_decode_error": candidate.decode_error,
                        },
                    ]
                    for execution_result in execution_results:
                        step_log.append({"role": "tool", "content": execution_result})
                    turn_log[f"step_{count}"] = step_log

                    segment_infos.append(
                        {
                            "sample_id": test_entry_id,
                            "turn_idx": turn_idx,
                            "step_idx": count,
                            "segment_index": segment_index,
                            "micro_snapshot": micro_snapshot,
                            "request_messages": request_messages,
                            "candidate_text": candidate_text,
                            "candidate_message": candidate_message,
                            "candidate_assistant": candidate_assistant,
                            "candidate_action": candidate_action,
                            "candidate_status": candidate.status,
                            "candidate_decode_error": candidate.decode_error,
                            "candidate_empty_response": candidate.empty_response,
                            "candidate_usage": candidate_usage,
                            "candidate_elapsed": candidate_elapsed,
                            "candidate_raw": candidate_raw,
                            "execution_results": execution_results,
                            "execution_error": execution_error,
                            "state_after_candidate": state_after_candidate,
                            "ref_step": ref_step,
                            "alignment_status": alignment_status,
                            "verify": verify,
                            "terminal": (
                                is_empty_execute_response(candidate_action)
                                or execution_error is not None
                            ),
                            "request_history_tokens": _token_count(
                                self.tokenizer,
                                request_messages,
                            ),
                            "full_history_tokens": _token_count(
                                self.tokenizer,
                                micro_snapshot["messages"],
                            ),
                        }
                    )
                    count += 1
                    segment_infos[-1]["heuristic_attributes"] = (
                        self._heuristic_attributes(
                            info=segment_infos[-1],
                            segment_infos=segment_infos[:-1],
                        )
                    )
                    if is_empty_execute_response(candidate_action):
                        terminal_after_segment = True
                        break
                    if execution_error is not None:
                        terminal_after_segment = True
                        break
                    if count > MAXIMUM_STEP_LIMIT:
                        force_quit = True
                        terminal_after_segment = True
                        step_log.append(
                            {
                                "role": "handler_log",
                                "content": (
                                    "Model has been forced to quit after "
                                    f"{MAXIMUM_STEP_LIMIT} steps."
                                ),
                            }
                        )
                        break

                if not segment_infos:
                    break

                verify_count += 1
                segment_has_drift = any(
                    bool(info["verify"]["verify_failed"]) for info in segment_infos
                )
                speculative_end_state = _state_log(involved_instances)
                speculative_terminal_after_segment = terminal_after_segment
                speculative_force_quit = force_quit
                regenerated_infos: list[dict[str, Any]] = []
                final_records: list[dict[str, Any]] = []
                segment_recovery_tokens = 0
                rollback_steps = 0
                first_bad_index: int | None = None
                predicted_first_bad_index: int | None = None
                attribution_debug: dict[str, Any] = {}
                restored_state: list[dict[str, Any]] | None = None
                rollback_state_matches_checkpoint: bool | None = None
                rollback_backend_info: dict[str, Any] = {
                    "rollback_backend": self.rollback_backend,
                    "rollback_backend_requested": self.rollback_backend,
                    "kv_restore_success": False,
                    "kv_restore_fallback": False,
                    "kv_restore_fallback_reason": None,
                    "kv_reused_tokens": 0,
                    "kv_recomputed_tokens": 0,
                    "message_replay_prefill_tokens": 0,
                    "restore_latency_sec": 0.0,
                    "rollback_latency_sec": 0.0,
                }

                if not segment_has_drift:
                    for info in segment_infos:
                        step_record = self._make_final_step_record(
                            spec_info=info,
                            executed_text=info["candidate_text"],
                            executed_message=info["candidate_assistant"],
                            executed_action=info["candidate_action"],
                            execution_results=info["execution_results"],
                            state_after_step=info["state_after_candidate"],
                            execution_error=info["execution_error"],
                            global_step=len(drift_steps) + len(final_records),
                            oracle_corrected=False,
                            extra={
                                "checkpoint_id": checkpoint_id,
                                "segment_index": info["segment_index"],
                                "segment_committed": True,
                                "refresh_triggered": False,
                                "rollback_triggered": False,
                                "candidate_execution_error": info["execution_error"],
                            },
                        )
                        final_records.append(step_record)
                else:
                    refresh_count += 1
                    terminal_after_segment = False
                    force_quit = False
                    first_bad_index = next(
                        index
                        for index, info in enumerate(segment_infos)
                        if info["verify"]["verify_failed"]
                    )
                    predicted_first_bad_index, attribution_debug = (
                        self._predict_first_bad_index(
                            segment_infos=segment_infos,
                            oracle_first_bad_index=first_bad_index,
                        )
                    )
                    if self.recovery_mode in {
                        "current_step",
                        "first_bad_suffix",
                        "oracle_first_bad",
                    }:
                        kept_infos = segment_infos[:predicted_first_bad_index]
                        for info in kept_infos:
                            step_record = self._make_final_step_record(
                                spec_info=info,
                                executed_text=info["candidate_text"],
                                executed_message=info["candidate_assistant"],
                                executed_action=info["candidate_action"],
                                execution_results=info["execution_results"],
                                state_after_step=info["state_after_candidate"],
                                execution_error=info["execution_error"],
                                global_step=len(drift_steps) + len(final_records),
                                oracle_corrected=False,
                                extra={
                                    "checkpoint_id": checkpoint_id,
                                    "segment_index": info["segment_index"],
                                    "segment_committed": True,
                                    "refresh_triggered": False,
                                    "rollback_triggered": False,
                                    "candidate_execution_error": info["execution_error"],
                                },
                            )
                            final_records.append(step_record)
                        restore_target = segment_infos[predicted_first_bad_index][
                            "micro_snapshot"
                        ]
                        rollback_steps = len(segment_infos) - predicted_first_bad_index
                        if self.recovery_mode == "current_step":
                            target_infos = [segment_infos[predicted_first_bad_index]]
                        else:
                            target_infos = segment_infos[predicted_first_bad_index:]
                    else:
                        if self.attribution == "whole_segment":
                            restore_target = segment_checkpoint
                            rollback_steps = len(segment_infos)
                            target_infos = segment_infos
                        else:
                            kept_infos = segment_infos[:predicted_first_bad_index]
                            for info in kept_infos:
                                step_record = self._make_final_step_record(
                                    spec_info=info,
                                    executed_text=info["candidate_text"],
                                    executed_message=info["candidate_assistant"],
                                    executed_action=info["candidate_action"],
                                    execution_results=info["execution_results"],
                                    state_after_step=info["state_after_candidate"],
                                    execution_error=info["execution_error"],
                                    global_step=len(drift_steps) + len(final_records),
                                    oracle_corrected=False,
                                    extra={
                                        "checkpoint_id": checkpoint_id,
                                        "segment_index": info["segment_index"],
                                        "segment_committed": True,
                                        "refresh_triggered": False,
                                        "rollback_triggered": False,
                                        "candidate_execution_error": info["execution_error"],
                                    },
                                )
                                final_records.append(step_record)
                            restore_target = segment_infos[predicted_first_bad_index][
                                "micro_snapshot"
                            ]
                            rollback_steps = len(segment_infos) - predicted_first_bad_index
                            target_infos = segment_infos[predicted_first_bad_index:]

                    (
                        messages,
                        involved_instances,
                        current_turn_response,
                        current_turn_inputs,
                        current_turn_outputs,
                        current_turn_latency,
                        turn_log,
                        restored_state,
                        rollback_state_matches_checkpoint,
                        rollback_backend_info,
                    ) = self._restore_with_backend(
                        test_entry_id=test_entry_id,
                        snapshot=restore_target,
                    )
                    kv_restore_success_count += int(
                        bool(rollback_backend_info.get("kv_restore_success"))
                    )
                    kv_restore_fallback_count += int(
                        bool(rollback_backend_info.get("kv_restore_fallback"))
                    )
                    kv_reused_tokens_total += int(
                        rollback_backend_info.get("kv_reused_tokens") or 0
                    )
                    kv_recomputed_tokens_total += int(
                        rollback_backend_info.get("kv_recomputed_tokens") or 0
                    )
                    message_replay_prefill_tokens_total += int(
                        rollback_backend_info.get("message_replay_prefill_tokens")
                        or 0
                    )
                    rollback_latency_total += float(
                        rollback_backend_info.get("rollback_latency_sec") or 0.0
                    )
                    restore_latency_total += float(
                        rollback_backend_info.get("restore_latency_sec") or 0.0
                    )
                    if not rollback_state_matches_checkpoint:
                        raise RuntimeError(
                            "rollback_state_mismatch: "
                            f"id={test_entry_id} checkpoint_id={checkpoint_id}"
                        )
                    count = int(target_infos[0]["step_idx"])

                    for info in target_infos:
                        recovery_messages, recovery_local_tokens, recovery_debug = (
                            self._build_recovery_messages(messages, stats)
                        )
                        (
                            recovery_text,
                            recovery_message,
                            recovery_elapsed,
                            recovery_usage,
                        ) = self._query(
                            recovery_messages,
                            tools,
                            stats,
                        )
                        recovery_prompt_tokens = int(
                            recovery_usage.get("prompt_tokens") or recovery_local_tokens
                        )
                        segment_recovery_tokens += recovery_prompt_tokens
                        recovery_assistant = _assistant_history_message(
                            recovery_text,
                            recovery_message.get("tool_calls"),
                        )
                        recovery_candidate = decode_candidate(
                            self.decoder,
                            recovery_text,
                        )
                        executed_action = recovery_candidate.action
                        messages.append(recovery_assistant)
                        execution_results, next_instances, execution_error = (
                            self._execute_action(
                                action=executed_action,
                                initial_config=initial_config,
                                involved_classes=involved_classes,
                                test_entry_id=test_entry_id,
                                long_context=long_context,
                            )
                        )
                        if next_instances is not None:
                            involved_instances = next_instances
                        for idx, execution_result in enumerate(execution_results):
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": execution_result,
                                    "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                                }
                            )
                        state_after_recovery = _state_log(involved_instances)
                        current_turn_response.append(recovery_text)
                        current_turn_inputs.append(recovery_usage["prompt_tokens"])
                        current_turn_outputs.append(recovery_usage["completion_tokens"])
                        current_turn_latency.append(recovery_elapsed)
                        step_log = [
                            {"role": "assistant", "content": recovery_text},
                            {
                                "role": "handler_log",
                                "content": (
                                    "Successfully decoded regenerated model response."
                                    if recovery_candidate.status == "decoded_action"
                                    else (
                                        "Regenerated candidate status: "
                                        f"{recovery_candidate.status}."
                                    )
                                ),
                                "model_response_decoded": executed_action,
                                "candidate_status": info["candidate_status"],
                                "regenerated_status": recovery_candidate.status,
                                "candidate_decode_error": info[
                                    "candidate_decode_error"
                                ],
                            },
                            {
                                "role": "checkpoint_verify",
                                "content": (
                                    "Rolled back speculative C2KV segment and "
                                    "regenerated with high precision history."
                                ),
                                "checkpoint_id": checkpoint_id,
                                "recovery_mode": self.recovery_mode,
                                "segment_index": info["segment_index"],
                                "candidate_text": info["candidate_text"],
                                "candidate_action": info["candidate_action"],
                                "regenerated_text": recovery_text,
                                "regenerated_action": executed_action,
                                "regenerated_same_as_candidate": action_matches(
                                    info["candidate_action"],
                                    executed_action,
                                ),
                                **recovery_debug,
                            },
                        ]
                        for execution_result in execution_results:
                            step_log.append({"role": "tool", "content": execution_result})
                        turn_log[f"step_{count}"] = step_log

                        step_record = self._make_final_step_record(
                            spec_info=info,
                            executed_text=recovery_text,
                            executed_message=recovery_assistant,
                            executed_action=executed_action,
                            execution_results=execution_results,
                            state_after_step=state_after_recovery,
                            execution_error=execution_error,
                            global_step=len(drift_steps) + len(final_records),
                            oracle_corrected=not action_matches(
                                info["candidate_action"],
                                executed_action,
                            ),
                            extra={
                                "checkpoint_id": checkpoint_id,
                                "segment_index": info["segment_index"],
                                "segment_committed": True,
                                "refresh_triggered": True,
                                "rollback_triggered": True,
                                "candidate_execution_error": info["execution_error"],
                                "regenerated_status": recovery_candidate.status,
                                "rollback_backend": rollback_backend_info.get(
                                    "rollback_backend"
                                ),
                                "kv_restore_fallback": rollback_backend_info.get(
                                    "kv_restore_fallback"
                                ),
                            },
                        )
                        final_records.append(step_record)
                        regenerated_infos.append(
                            {
                                "info": info,
                                "record": step_record,
                                "recovery_prompt_tokens": recovery_prompt_tokens,
                                "recovery_local_history_tokens": recovery_local_tokens,
                                "recovery_debug": recovery_debug,
                                "regenerated_same_as_candidate": action_matches(
                                    info["candidate_action"],
                                    executed_action,
                                ),
                            }
                        )
                        count += 1
                        regenerated_steps_total += 1
                        if is_empty_execute_response(executed_action):
                            terminal_after_segment = True
                            break
                        if execution_error is not None:
                            terminal_after_segment = True
                            break
                        if count > MAXIMUM_STEP_LIMIT:
                            force_quit = True
                            terminal_after_segment = True
                            step_log.append(
                                {
                                    "role": "handler_log",
                                    "content": (
                                        "Model has been forced to quit after "
                                        f"{MAXIMUM_STEP_LIMIT} steps."
                                    ),
                                }
                            )
                            break

                    full_regenerated_tokens_total += segment_recovery_tokens

                for index, step_record in enumerate(final_records):
                    drift_steps.append(step_record)
                    mark_first_divergence(stats, step_record)
                    if step_record.get("serialization_mismatch"):
                        stats.errors.append(
                            "serialization mismatch at "
                            f"{test_entry_id} turn={step_record['turn']} "
                            f"step={step_record['step']}"
                        )
                    spec_info = segment_infos[min(index, len(segment_infos) - 1)]
                    regen_match = None
                    step_recovery_tokens = 0
                    step_recovery_debug = {
                        "recovery_prompt_mode": None,
                        "c2kv_history_units": 0,
                        "full_history_units": 0,
                        "current_messages": 0,
                    }
                    if step_record.get("refresh_triggered"):
                        matched_regen = next(
                            (
                                item
                                for item in regenerated_infos
                                if item["record"] is step_record
                            ),
                            None,
                        )
                        if matched_regen:
                            regen_match = matched_regen[
                                "regenerated_same_as_candidate"
                            ]
                            step_recovery_tokens = matched_regen[
                                "recovery_prompt_tokens"
                            ]
                            step_recovery_debug = matched_regen["recovery_debug"]
                    checkpoint_steps.append(
                        self._checkpoint_step_row(
                            step_record=step_record,
                            checkpoint_id=checkpoint_id,
                            segment_index=int(step_record.get("segment_index") or 0),
                            segment_start_step=segment_start_step,
                            segment_length=len(segment_infos),
                            verify=spec_info["verify"],
                            verify_triggered=index == 0,
                            refresh_triggered=bool(
                                step_record.get("refresh_triggered")
                            ),
                            regenerated_steps=int(
                                bool(step_record.get("refresh_triggered"))
                            ),
                            full_regenerated_tokens=step_recovery_tokens,
                            recovery_debug=step_recovery_debug,
                            regenerated_same_as_candidate=regen_match,
                            history_prompt_tokens=spec_info["candidate_usage"][
                                "prompt_tokens"
                            ],
                            request_history_tokens=spec_info[
                                "request_history_tokens"
                            ],
                            full_history_tokens=spec_info["full_history_tokens"],
                        )
                    )
                    checkpoint_steps[-1].update(
                        {
                            "rollback_backend": rollback_backend_info.get(
                                "rollback_backend"
                            ),
                            "attribution": self.attribution,
                            "attribution_safety_margin": self.attribution_safety_margin,
                            "oracle_first_bad_index": first_bad_index,
                            "predicted_first_bad_index": predicted_first_bad_index,
                            "heuristic_attributes": spec_info.get(
                                "heuristic_attributes"
                            ),
                            "hard_error": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("hard_error"),
                            "argument_grounding_score": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("argument_grounding_score"),
                            "repeat_action_score": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("repeat_action_score"),
                            "tool_transition_anomaly": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("tool_transition_anomaly"),
                            "observation_anomaly": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("observation_anomaly"),
                            "representation_jump": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("representation_jump"),
                            "risk_score": (
                                spec_info.get("heuristic_attributes") or {}
                            ).get("risk_score"),
                        }
                    )

                regenerated_end_state = _state_log(involved_instances)
                committed_speculative_tokens = sum(
                    int(info["candidate_usage"].get("prompt_tokens") or 0)
                    for info in segment_infos
                    if not segment_has_drift
                    or (
                        predicted_first_bad_index is not None
                        and int(info["segment_index"]) < predicted_first_bad_index
                    )
                )
                discarded_speculative_tokens = (
                    0
                    if not segment_has_drift
                    else sum(
                        int(info["candidate_usage"].get("prompt_tokens") or 0)
                        for info in segment_infos[
                            predicted_first_bad_index
                            if predicted_first_bad_index is not None
                            else 0 :
                        ]
                    )
                )
                checkpoint_segments.append(
                    {
                        "schema_version": 2,
                        "id": test_entry_id,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_interval": self.checkpoint_interval,
                        "interval": self.checkpoint_interval,
                        "turn": turn_idx,
                        "segment_start_step": segment_start_step,
                        "segment_end_step": len(drift_steps) - 1,
                        "segment_start_turn_step": segment_start_turn_step,
                        "segment_end_turn_step": count - 1,
                        "speculative_steps": len(segment_infos),
                        "segment_length": len(segment_infos),
                        "candidate_actions": [
                            info["candidate_action"] for info in segment_infos
                        ],
                        "reference_actions": [
                            (
                                info["ref_step"].get("decoded_action")
                                if info["ref_step"]
                                else None
                            )
                            for info in segment_infos
                        ],
                        "candidate_action_drift_per_step": [
                            not bool(
                                info["verify"][
                                    "candidate_action_matches_reference"
                                ]
                            )
                            for info in segment_infos
                        ],
                        "candidate_state_drift_per_step": [
                            info["verify"]["state_matches_reference"] is False
                            for info in segment_infos
                        ],
                        "harmful_drift_per_step": [
                            bool(info["verify"]["verify_failed"])
                            for info in segment_infos
                        ],
                        "candidate_drift_per_step": [
                            bool(info["verify"]["verify_failed"])
                            for info in segment_infos
                        ],
                        "segment_candidate_drift_count": sum(
                            1
                            for info in segment_infos
                            if not info["verify"][
                                "candidate_action_matches_reference"
                            ]
                        ),
                        "segment_state_drift_count": sum(
                            1
                            for info in segment_infos
                            if info["verify"]["state_matches_reference"] is False
                        ),
                        "segment_harmful_drift_count": sum(
                            1
                            for info in segment_infos
                            if info["verify"]["verify_failed"]
                        ),
                        "segment_executed_drift_count": sum(
                            1
                            for record in final_records
                            if record.get("executed_action_drift")
                        ),
                        "segment_has_drift": segment_has_drift,
                        "verify_triggered": True,
                        "rollback_triggered": segment_has_drift,
                        "refresh_triggered": segment_has_drift,
                        "checkpoint_state": segment_checkpoint.get("state"),
                        "speculative_end_state": speculative_end_state,
                        "restored_state": restored_state,
                        "regenerated_end_state": regenerated_end_state,
                        "rollback_state_matches_checkpoint": (
                            rollback_state_matches_checkpoint
                        ),
                        "first_bad_index": first_bad_index,
                        "oracle_first_bad_index": first_bad_index,
                        "predicted_first_bad_index": predicted_first_bad_index,
                        "raw_predicted_first_bad_index": attribution_debug.get(
                            "raw_predicted_first_bad_index"
                        ),
                        "attribution": self.attribution,
                        "attribution_reason": attribution_debug.get(
                            "attribution_reason"
                        ),
                        "attribution_safety_margin": self.attribution_safety_margin,
                        "exact_attribution": attribution_debug.get(
                            "exact_attribution"
                        ),
                        "within1_attribution": attribution_debug.get(
                            "within1_attribution"
                        ),
                        "under_rollback": attribution_debug.get("under_rollback"),
                        "over_rollback": attribution_debug.get("over_rollback"),
                        "over_rollback_steps": attribution_debug.get(
                            "over_rollback_steps"
                        ),
                        "predicted_rollback_depth": attribution_debug.get(
                            "predicted_rollback_depth"
                        ),
                        "oracle_rollback_depth": attribution_debug.get(
                            "oracle_rollback_depth"
                        ),
                        "rollback_depth": rollback_steps,
                        "rollback_steps": rollback_steps,
                        "rollback_policy": (
                            "none"
                            if not segment_has_drift
                            else (
                                "first_bad_micro_checkpoint"
                                if self.attribution != "whole_segment"
                                else "segment_checkpoint"
                            )
                        ),
                        "regen_policy": (
                            "none"
                            if not segment_has_drift
                            else (
                                "single_step"
                                if self.recovery_mode == "current_step"
                                else (
                                    "first_bad_suffix"
                                    if self.attribution != "whole_segment"
                                    else "whole_segment"
                                )
                            )
                        ),
                        "regenerated_steps": len(regenerated_infos),
                        "full_regenerated_tokens": segment_recovery_tokens,
                        "rollback_backend": rollback_backend_info.get(
                            "rollback_backend"
                        ),
                        "rollback_backend_requested": rollback_backend_info.get(
                            "rollback_backend_requested"
                        ),
                        "kv_restore_success": rollback_backend_info.get(
                            "kv_restore_success"
                        ),
                        "kv_restore_fallback": rollback_backend_info.get(
                            "kv_restore_fallback"
                        ),
                        "kv_restore_fallback_reason": rollback_backend_info.get(
                            "kv_restore_fallback_reason"
                        ),
                        "kv_reused_tokens": rollback_backend_info.get(
                            "kv_reused_tokens"
                        ),
                        "kv_recomputed_tokens": rollback_backend_info.get(
                            "kv_recomputed_tokens"
                        ),
                        "message_replay_prefill_tokens": rollback_backend_info.get(
                            "message_replay_prefill_tokens"
                        ),
                        "restore_latency_sec": rollback_backend_info.get(
                            "restore_latency_sec"
                        ),
                        "rollback_latency_sec": rollback_backend_info.get(
                            "rollback_latency_sec"
                        ),
                        "speculative_candidate_prompt_tokens": sum(
                            int(info["candidate_usage"].get("prompt_tokens") or 0)
                            for info in segment_infos
                        ),
                        "speculative_candidate_tokens": sum(
                            int(info["candidate_usage"].get("prompt_tokens") or 0)
                            for info in segment_infos
                        ),
                        "discarded_speculative_tokens": discarded_speculative_tokens,
                        "committed_speculative_tokens": committed_speculative_tokens,
                        "speculative_request_history_tokens": sum(
                            int(info.get("request_history_tokens") or 0)
                            for info in segment_infos
                        ),
                        "discarded_speculative_steps": (
                            0
                            if not segment_has_drift
                            else (
                                len(segment_infos)
                                - (predicted_first_bad_index or 0)
                                if predicted_first_bad_index is not None
                                and self.attribution != "whole_segment"
                                else len(segment_infos)
                            )
                        ),
                        "heuristic_attributes_per_step": [
                            info.get("heuristic_attributes") for info in segment_infos
                        ],
                        "final_executed_actions": [
                            record.get("executed_action") for record in final_records
                        ],
                        "executed_drift_per_step": [
                            bool(record.get("executed_action_drift"))
                            for record in final_records
                        ],
                        "state_drift_after_recovery": any(
                            bool(record.get("state_drift"))
                            for record in final_records
                        ),
                        "speculative_terminal_after_segment": (
                            speculative_terminal_after_segment
                        ),
                        "speculative_force_quit": speculative_force_quit,
                        "terminal_after_segment": terminal_after_segment,
                        "recovery_mode": self.recovery_mode,
                    }
                )

                if terminal_after_segment or force_quit:
                    break

            all_model_response.append(current_turn_response)
            input_token_count.append(current_turn_inputs)
            output_token_count.append(current_turn_outputs)
            latency.append(current_turn_latency)
            inference_log.append(turn_log)
            state = _state_log(involved_instances)
            if state:
                inference_log.append(state)
            if force_quit:
                break

        metadata = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
            "inference_log": inference_log,
            "drift_steps": drift_steps,
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_segments": checkpoint_segments,
            "verify_count": verify_count,
            "refresh_count": refresh_count,
            "kv_restore_success": kv_restore_success_count,
            "kv_restore_fallback": kv_restore_fallback_count,
            "kv_reused_tokens": kv_reused_tokens_total,
            "kv_recomputed_tokens": kv_recomputed_tokens_total,
            "message_replay_prefill_tokens": message_replay_prefill_tokens_total,
            "rollback_latency_sec": rollback_latency_total,
            "restore_latency_sec": restore_latency_total,
            "regenerated_steps": regenerated_steps_total,
            "full_regenerated_tokens": full_regenerated_tokens_total,
        }
        return all_model_response, metadata

    def run_sample_checkpoint(self, test_case: dict[str, Any]) -> dict[str, Any]:
        stats = DriftStats(test_case["id"], "history_checkpoint", self.ratio)
        self._cumulative_divergence = 0.0
        try:
            result, metadata = self._run_sample_checkpoint_impl(test_case, stats)
        except Exception as exc:
            result = f"Error during inference: {exc}"
            metadata = {"traceback": traceback.format_exc()}
            stats.errors.append(str(exc))
        metadata["c2kv_checkpoint_metrics"] = {
            **stats.as_dict(),
            "checkpoint_interval": self.checkpoint_interval,
            "verifier": self.verifier,
            "requested_verifier": self.requested_verifier,
            "attribution": self.attribution,
            "attribution_safety_margin": self.attribution_safety_margin,
            "rollback_backend": self.rollback_backend,
            "verify_threshold": self.verify_threshold,
            "verify_layers": self.verify_layers,
            "online_verify": self.online_verify,
            "reuse_candidate_readout": self.reuse_candidate_readout,
            "recovery_mode": self.recovery_mode,
            "verify_count": metadata.get("verify_count", 0),
            "refresh_count": metadata.get("refresh_count", 0),
            "regenerated_steps": metadata.get("regenerated_steps", 0),
            "full_regenerated_tokens": metadata.get("full_regenerated_tokens", 0),
            "kv_restore_success": metadata.get("kv_restore_success", 0),
            "kv_restore_fallback": metadata.get("kv_restore_fallback", 0),
            "kv_reused_tokens": metadata.get("kv_reused_tokens", 0),
            "kv_recomputed_tokens": metadata.get("kv_recomputed_tokens", 0),
            "message_replay_prefill_tokens": metadata.get(
                "message_replay_prefill_tokens",
                0,
            ),
            "rollback_latency_sec": metadata.get("rollback_latency_sec", 0.0),
            "restore_latency_sec": metadata.get("restore_latency_sec", 0.0),
        }
        return {"id": test_case["id"], "result": result, **metadata}

    def _run_sample_checkpoint_impl(
        self,
        test_case: dict[str, Any],
        stats: DriftStats,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        if self.verifier == "oracle":
            return self._run_sample_checkpoint_impl_oracle_multistep(
                test_case,
                stats,
            )
        if self.checkpoint_interval != 1:
            raise NotImplementedError(
                "KV/readout checkpoint verification currently supports "
                "checkpoint_interval=1 only. Run verifier=oracle for true "
                "multi-step rollback with interval=1/2/4."
            )
        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        tools = _tool_payload(test_case["function"])
        long_context = "long_context" in test_category or "composite" in test_category
        reference_by_turn_step, reference_result = self._reference_maps(test_entry_id)

        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.decoder.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )

        inference_log: list[Any] = []
        initial_state = _state_log(involved_instances)
        if initial_state:
            inference_log.append(initial_state)

        messages: list[dict[str, Any]] = []
        all_model_response: list[list[str]] = []
        input_token_count: list[list[int]] = []
        output_token_count: list[list[int]] = []
        latency: list[list[float]] = []
        checkpoint_steps: list[dict[str, Any]] = []
        drift_steps: list[dict[str, Any]] = []
        verify_count = 0
        refresh_count = 0
        regenerated_steps = 0
        full_regenerated_tokens = 0
        checkpoint_id = 0
        force_quit = False

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}

            count = 0
            while True:
                checkpoint_id += 1
                checkpoint_messages = deepcopy(messages)
                checkpoint_instances = deepcopy(involved_instances)
                checkpoint_turn_response = deepcopy(current_turn_response)
                checkpoint_turn_inputs = deepcopy(current_turn_inputs)
                checkpoint_turn_outputs = deepcopy(current_turn_outputs)
                checkpoint_turn_latency = deepcopy(current_turn_latency)
                checkpoint_turn_log = deepcopy(turn_log)

                request_messages = self._build_request_messages(messages, stats)
                (
                    candidate_text,
                    candidate_message,
                    candidate_elapsed,
                    candidate_usage,
                    candidate_raw,
                ) = self._query_with_raw(
                    request_messages,
                    tools,
                    stats,
                    readout_probe=(
                        self.verifier == "kv_divergence"
                        and self.reuse_candidate_readout
                    ),
                )
                candidate_assistant = _assistant_history_message(
                    candidate_text,
                    candidate_message.get("tool_calls"),
                )
                candidate = decode_candidate(self.decoder, candidate_text)
                candidate_action = candidate.action

                messages.append(candidate_assistant)
                candidate_execution_error = None
                if is_empty_execute_response(candidate_action):
                    execution_results = []
                else:
                    try:
                        execution_results, involved_instances = execute_multi_turn_func_call(
                            candidate_action,
                            initial_config,
                            involved_classes,
                            self.decoder.model_name_underline_replaced,
                            test_entry_id,
                            long_context=long_context,
                            is_evaL_run=False,
                        )
                    except Exception as exc:
                        candidate_execution_error = str(exc)
                        execution_results = []
                    for idx, execution_result in enumerate(execution_results):
                        messages.append(
                            {
                                "role": "tool",
                                "content": execution_result,
                                "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                            }
                        )

                state_after_candidate = _state_log(involved_instances)
                ref_step, alignment_status = reference_step_for(
                    reference_by_turn_step,
                    reference_result,
                    turn_idx,
                    count,
                    fallback_state=state_after_candidate,
                )
                if self.verifier == "oracle":
                    verify = self._verify_oracle(
                        decoded_action=candidate_action,
                        state_after_step=state_after_candidate,
                        ref_step=ref_step,
                    )
                    verify.update(
                        {
                            "logit_kl": None,
                            "entropy": None,
                            "top1_top2_margin": None,
                            "readout_available": False,
                            "full_readout_source": None,
                            "c2kv_readout_source": None,
                            "candidate_readout_reused": False,
                            "full_readout_dim": 0,
                            "c2kv_readout_dim": 0,
                            "full_probe_seconds": 0.0,
                            "full_probe_prompt_tokens": 0,
                            "c2kv_probe_seconds": 0.0,
                            "c2kv_probe_prompt_tokens": 0,
                        }
                    )
                else:
                    kv_verify = self._verify_kv_divergence(
                        full_messages=checkpoint_messages,
                        c2kv_messages=request_messages,
                        tools=tools,
                        stats=stats,
                        candidate_raw=candidate_raw,
                        candidate_usage=candidate_usage,
                        candidate_elapsed=candidate_elapsed,
                    )
                    reference_action = ref_step.get("decoded_action") if ref_step else None
                    candidate_action_matches = action_matches(
                        candidate_action,
                        reference_action or [],
                    )
                    if ref_step is None:
                        state_matches = is_empty_execute_response(candidate_action)
                    else:
                        state_matches = (
                            _normalize_state(state_after_candidate)
                            == _normalize_state(ref_step.get("state"))
                        )
                    verify = {
                        **kv_verify,
                        "candidate_action_matches_reference": candidate_action_matches,
                        "state_matches_reference": state_matches,
                    }
                verify_count += 1
                refresh_triggered = bool(verify["verify_failed"])

                if refresh_triggered:
                    refresh_count += 1
                    regenerated_steps += 1
                    self._restore_instances(test_entry_id, checkpoint_instances)
                    messages = deepcopy(checkpoint_messages)
                    involved_instances = deepcopy(checkpoint_instances)
                    current_turn_response = deepcopy(checkpoint_turn_response)
                    current_turn_inputs = deepcopy(checkpoint_turn_inputs)
                    current_turn_outputs = deepcopy(checkpoint_turn_outputs)
                    current_turn_latency = deepcopy(checkpoint_turn_latency)
                    turn_log = deepcopy(checkpoint_turn_log)

                    (
                        recovery_messages,
                        recovery_local_tokens,
                        recovery_debug,
                    ) = self._build_recovery_messages(
                        messages,
                        stats,
                    )
                    text, response_message, elapsed, usage = self._query(
                        recovery_messages,
                        tools,
                        stats,
                    )
                    recovery_prompt_tokens = int(
                        usage.get("prompt_tokens") or recovery_local_tokens
                    )
                    full_regenerated_tokens += recovery_prompt_tokens
                    assistant_history = _assistant_history_message(
                        text,
                        response_message.get("tool_calls"),
                    )
                    recovery_candidate = decode_candidate(self.decoder, text)
                    executed_action = recovery_candidate.action
                    messages.append(assistant_history)
                    executed_text = text
                    executed_elapsed = elapsed
                    executed_usage = usage
                    candidate_debug = {
                        "candidate_text": candidate_text,
                        "candidate_assistant_message": candidate_assistant,
                        "candidate_action": candidate_action,
                        "candidate_status": candidate.status,
                        "regenerated_text": text,
                        "regenerated_action": executed_action,
                        "regenerated_status": recovery_candidate.status,
                        "regenerated_same_as_candidate": (
                            action_matches(candidate_action, executed_action)
                        ),
                        "recovery_local_history_tokens": recovery_local_tokens,
                        **recovery_debug,
                    }
                else:
                    text = candidate_text
                    assistant_history = candidate_assistant
                    executed_action = candidate_action
                    executed_text = candidate_text
                    executed_elapsed = candidate_elapsed
                    executed_usage = candidate_usage
                    candidate_debug = None
                    recovery_debug = {
                        "recovery_prompt_mode": None,
                        "c2kv_history_units": 0,
                        "full_history_units": 0,
                        "current_messages": 0,
                    }

                if refresh_triggered:
                    execution_error = None
                    if is_empty_execute_response(executed_action):
                        execution_results = []
                    else:
                        try:
                            execution_results, involved_instances = execute_multi_turn_func_call(
                                executed_action,
                                initial_config,
                                involved_classes,
                                self.decoder.model_name_underline_replaced,
                                test_entry_id,
                                long_context=long_context,
                                is_evaL_run=False,
                            )
                        except Exception as exc:
                            execution_error = str(exc)
                            execution_results = []
                        for idx, execution_result in enumerate(execution_results):
                            messages.append(
                                {
                                    "role": "tool",
                                    "content": execution_result,
                                    "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                                }
                            )
                else:
                    execution_error = candidate_execution_error

                state_after_step = _state_log(involved_instances)
                reference_action = ref_step.get("decoded_action") if ref_step else None
                executed_action_matches = action_matches(
                    executed_action,
                    reference_action or [],
                )
                if ref_step is None:
                    state_matches = is_empty_execute_response(executed_action)
                else:
                    state_matches = (
                        _normalize_state(state_after_step)
                        == _normalize_state(ref_step.get("state"))
                    )

                current_turn_response.append(executed_text)
                current_turn_inputs.append(executed_usage["prompt_tokens"])
                current_turn_outputs.append(executed_usage["completion_tokens"])
                current_turn_latency.append(executed_elapsed)
                step_log = [
                    {"role": "assistant", "content": executed_text},
                    {
                        "role": "handler_log",
                        "content": (
                            "Successfully decoded model response."
                            if candidate.status == "decoded_action"
                            else f"Candidate status: {candidate.status}."
                        ),
                        "model_response_decoded": executed_action,
                        "candidate_status": candidate.status,
                        "candidate_decode_error": candidate.decode_error,
                    },
                ]
                if candidate_debug is not None:
                    step_log.append(
                        {
                            "role": "checkpoint_verify",
                            "content": "Rolled back speculative C2KV step and regenerated.",
                            **candidate_debug,
                            "recovery_mode": self.recovery_mode,
                            "checkpoint_id": checkpoint_id,
                        }
                    )
                for execution_result in execution_results:
                    step_log.append({"role": "tool", "content": execution_result})
                turn_log[f"step_{count}"] = step_log

                executed_roundtrip = serialization_roundtrip(
                    self.decoder,
                    executed_text,
                    executed_action,
                )
                step_record = build_step_record(
                    sample_id=test_entry_id,
                    turn_idx=turn_idx,
                    step_idx=count,
                    global_step=len(drift_steps),
                    candidate_raw_text=candidate_text,
                    candidate_action=candidate_action,
                    candidate_status=candidate.status,
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=executed_action,
                    state=state_after_step,
                    decode_error=candidate.decode_error,
                    empty_response=candidate.empty_response,
                    execution_error=execution_error,
                    candidate_assistant_message=candidate_assistant,
                    executed_assistant_message=assistant_history,
                    execution_results=execution_results,
                    history_execution_results=execution_results,
                    response_matches_reference=None,
                    candidate_response_matches_reference=None,
                    roundtrip=executed_roundtrip,
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "refresh_triggered": refresh_triggered,
                        "candidate_execution_error": candidate_execution_error,
                    },
                )
                step_record["candidate_action_matches_reference"] = verify[
                    "candidate_action_matches_reference"
                ]
                step_record["candidate_action_drift"] = not verify[
                    "candidate_action_matches_reference"
                ]
                step_record["executed_action_matches_reference"] = executed_action_matches
                step_record["executed_action_drift"] = not executed_action_matches
                step_record["state_matches_reference"] = state_matches
                step_record["state_drift"] = state_matches is False
                drift_steps.append(step_record)
                mark_first_divergence(stats, step_record)
                if executed_roundtrip["serialization_mismatch"]:
                    stats.errors.append(
                        "serialization mismatch at "
                        f"{test_entry_id} turn={turn_idx} step={count}"
                    )
                checkpoint_steps.append(
                    {
                        "id": test_entry_id,
                        "global_step": len(drift_steps) - 1,
                        "candidate_global_step": len(drift_steps) - 1,
                        "reference_global_step": (
                            ref_step.get("global_step") if ref_step else None
                        ),
                        "alignment_status": alignment_status,
                        "turn": turn_idx,
                        "step": count,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_interval": self.checkpoint_interval,
                        "verifier": self.verifier,
                        "requested_verifier": self.requested_verifier,
                        "verify_threshold": self.verify_threshold,
                        "online_verify": self.online_verify,
                        "kv_divergence": verify["kv_divergence"],
                        "cumulative_divergence": verify["cumulative_divergence"],
                        "logit_kl": verify.get("logit_kl"),
                        "entropy": verify.get("entropy"),
                        "top1_top2_margin": verify.get("top1_top2_margin"),
                        "readout_available": verify.get("readout_available"),
                        "full_readout_source": verify.get("full_readout_source"),
                        "c2kv_readout_source": verify.get("c2kv_readout_source"),
                        "candidate_readout_reused": verify.get(
                            "candidate_readout_reused"
                        ),
                        "full_readout_dim": verify.get("full_readout_dim"),
                        "c2kv_readout_dim": verify.get("c2kv_readout_dim"),
                        "full_probe_seconds": verify.get("full_probe_seconds"),
                        "c2kv_probe_seconds": verify.get("c2kv_probe_seconds"),
                        "full_probe_prompt_tokens": verify.get(
                            "full_probe_prompt_tokens"
                        ),
                        "c2kv_probe_prompt_tokens": verify.get(
                            "c2kv_probe_prompt_tokens"
                        ),
                        "verify_triggered": True,
                        "refresh_triggered": refresh_triggered,
                        "recovery_mode": self.recovery_mode,
                        "regenerated_steps": int(refresh_triggered),
                        "full_regenerated_tokens": (
                            recovery_prompt_tokens if refresh_triggered else 0
                        ),
                        "recovery_prompt_mode": recovery_debug.get(
                            "recovery_prompt_mode"
                        ),
                        "c2kv_history_units": recovery_debug.get(
                            "c2kv_history_units", 0
                        ),
                        "full_history_units": recovery_debug.get(
                            "full_history_units", 0
                        ),
                        "current_messages": recovery_debug.get("current_messages", 0),
                        "regenerated_same_as_candidate": (
                            candidate_debug.get("regenerated_same_as_candidate")
                            if candidate_debug
                            else None
                        ),
                        "candidate_status": candidate.status,
                        "decode_error": candidate.decode_error,
                        "empty_response": candidate.empty_response,
                        "candidate_action_drift": step_record[
                            "candidate_action_drift"
                        ],
                        "executed_action_drift": step_record[
                            "executed_action_drift"
                        ],
                        "state_drift": step_record["state_drift"],
                        "serialization_mismatch": executed_roundtrip[
                            "serialization_mismatch"
                        ],
                        "executable": not is_empty_execute_response(candidate_action),
                        "executed_executable": not is_empty_execute_response(
                            executed_action
                        ),
                        "history_prompt_tokens": candidate_usage["prompt_tokens"],
                        "request_history_tokens": _token_count(
                            self.tokenizer,
                            request_messages,
                        ),
                        "full_history_tokens": _token_count(
                            self.tokenizer,
                            checkpoint_messages,
                        ),
                    }
                )

                count += 1
                if is_empty_execute_response(executed_action):
                    break
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            all_model_response.append(current_turn_response)
            input_token_count.append(current_turn_inputs)
            output_token_count.append(current_turn_outputs)
            latency.append(current_turn_latency)
            inference_log.append(turn_log)
            state = _state_log(involved_instances)
            if state:
                inference_log.append(state)
            if force_quit:
                break

        metadata = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
            "inference_log": inference_log,
            "drift_steps": drift_steps,
            "checkpoint_steps": checkpoint_steps,
            "verify_count": verify_count,
            "refresh_count": refresh_count,
            "regenerated_steps": regenerated_steps,
            "full_regenerated_tokens": full_regenerated_tokens,
        }
        return all_model_response, metadata


def run(args: argparse.Namespace) -> None:
    runner = HistoryCheckpointRunner(args)
    entries = load_dataset_entry(args.category)
    entries = [entry for entry in entries if entry["id"].startswith(args.category)]
    entries = sorted(entries, key=sort_key)
    if args.ids_path:
        with open(args.ids_path, encoding="utf-8") as f:
            selected_ids = {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
        entries = [entry for entry in entries if entry["id"] in selected_ids]
    if args.max_examples is not None:
        entries = entries[: args.max_examples]

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    details_rows = []
    metrics_rows = []
    step_rows = []
    segment_rows = []
    for test_case in tqdm(entries, desc=f"history_checkpoint:{args.category}", dynamic_ncols=True):
        row = runner.run_sample_checkpoint(deepcopy(test_case))
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metrics_rows.append(row.get("c2kv_checkpoint_metrics", {}))
        step_rows.extend(row.get("checkpoint_steps") or [])
        segment_rows.extend(row.get("checkpoint_segments") or [])

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metrics_rows)
    _write_jsonl(Path(args.step_metrics_path), step_rows)
    if args.segment_metrics_path:
        _write_jsonl(Path(args.segment_metrics_path), segment_rows)
    summary = {
        "schema_version": 2,
        "category": args.category,
        "num_examples": len(details_rows),
        "compression_ratio": args.compression_ratio,
        "checkpoint_interval": args.checkpoint_interval,
        "verifier": args.verifier,
        "attribution": args.attribution,
        "attribution_safety_margin": args.attribution_safety_margin,
        "rollback_backend": args.rollback_backend,
        "verify_threshold": args.verify_threshold,
        "verify_layers": args.verify_layers,
        "online_verify": args.online_verify,
        "reuse_candidate_readout": args.reuse_candidate_readout,
        "recovery_mode": args.recovery_mode,
        "verify_count": sum(int(row.get("verify_count") or 0) for row in metrics_rows),
        "refresh_count": sum(int(row.get("refresh_count") or 0) for row in metrics_rows),
        "regenerated_steps": sum(int(row.get("regenerated_steps") or 0) for row in metrics_rows),
        "full_regenerated_tokens": sum(int(row.get("full_regenerated_tokens") or 0) for row in metrics_rows),
        "kv_restore_success": sum(int(row.get("kv_restore_success") or 0) for row in metrics_rows),
        "kv_restore_fallback": sum(int(row.get("kv_restore_fallback") or 0) for row in metrics_rows),
        "kv_reused_tokens": sum(int(row.get("kv_reused_tokens") or 0) for row in metrics_rows),
        "kv_recomputed_tokens": sum(int(row.get("kv_recomputed_tokens") or 0) for row in metrics_rows),
        "message_replay_prefill_tokens": sum(int(row.get("message_replay_prefill_tokens") or 0) for row in metrics_rows),
        "segment_count": len(segment_rows),
    }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--reference-details-path", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--step-metrics-path", required=True)
    parser.add_argument("--segment-metrics-path", default="")
    parser.add_argument("--summary-path", required=True)
    parser.add_argument(
        "--compression-ratio",
        "--ratio",
        dest="compression_ratio",
        type=int,
        default=4,
    )
    parser.add_argument("--checkpoint-interval", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument(
        "--verifier",
        choices=["instant_kv", "cumulative_kv", "kv_divergence", "oracle"],
        default="oracle",
    )
    parser.add_argument("--verify-threshold", type=float, default=0.0)
    parser.add_argument(
        "--verify-layers",
        default="25%,50%,75%,last",
        help=(
            "Layer selector reserved for server-side per-layer readout. "
            "The current OpenAI-compatible path uses the returned query readout."
        ),
    )
    parser.add_argument("--online-verify", action="store_true")
    parser.add_argument(
        "--reuse-candidate-readout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--recovery-mode",
        choices=sorted(CHECKPOINT_MODES),
        default="current_step",
    )
    parser.add_argument(
        "--attribution",
        choices=["auto", *sorted(ATTRIBUTION_MODES)],
        default="auto",
    )
    parser.add_argument("--attribution-safety-margin", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--rollback-backend",
        choices=sorted(ROLLBACK_BACKENDS),
        default="message_replay",
    )
    parser.add_argument("--recent-full-units", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    args.ratio = args.compression_ratio
    return args


if __name__ == "__main__":
    run(parse_args())
