from __future__ import annotations

import argparse
import csv
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
RECOVERY_HORIZONS = {"one_step", "suffix", "whole_segment"}
ROLLBACK_BACKENDS = {"message_replay", "kv_restore", "kv_restore_strict"}
ROLLBACK_POLICIES = {"attribution", "whole_segment", "fixed_depth", "rule_depth"}


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


def _iter_token_logprobs(logprobs: Any) -> list[dict[str, Any]]:
    if not logprobs:
        return []
    if isinstance(logprobs, dict):
        for key in ("content", "tokens", "output_tokens"):
            value = logprobs.get(key)
            if isinstance(value, list):
                return [
                    item for item in value if isinstance(item, dict)
                ]
        value = logprobs.get("top_logprobs")
        if isinstance(value, list):
            return _iter_token_logprobs(value)
    if isinstance(logprobs, list):
        return [item for item in logprobs if isinstance(item, dict)]
    return []


def _token_top_logprobs(item: dict[str, Any]) -> dict[str, float]:
    top = item.get("top_logprobs")
    if isinstance(top, list):
        out: dict[str, float] = {}
        for entry in top:
            if not isinstance(entry, dict):
                continue
            token = entry.get("token")
            value = entry.get("logprob")
            if token is not None and isinstance(value, (int, float)):
                out[str(token)] = float(value)
        return out
    if isinstance(top, dict):
        return {
            str(token): float(value)
            for token, value in top.items()
            if isinstance(value, (int, float))
        }
    return {}


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


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
        self.recovery_horizon = args.recovery_horizon
        if self.recovery_horizon == "auto":
            if self.recovery_mode == "current_step":
                self.recovery_horizon = "one_step"
            elif self.recovery_mode in {"whole_segment", "since_checkpoint", "full_history"}:
                self.recovery_horizon = "whole_segment"
            elif self.recovery_mode in {"first_bad_suffix", "oracle_first_bad"}:
                self.recovery_horizon = "suffix"
            else:
                self.recovery_horizon = "suffix"
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
        self.rollback_policy = args.rollback_policy
        self.rollback_depth = int(args.rollback_depth)
        self.rule_detector_threshold = float(args.rule_detector_threshold)
        self.logistic_detector_features_csv = args.logistic_detector_features_csv
        self.logistic_detector_threshold = float(args.logistic_detector_threshold)
        self.logistic_detector_model = None
        self.detector_signal_name = args.detector_signal_name
        self.detector_signal_threshold = float(args.detector_signal_threshold)
        self.rollback_backend = args.rollback_backend
        self.collect_candidate_detector_signals = bool(
            args.collect_candidate_detector_signals
        ) or self.verifier in {"logistic", "feature_signal"}
        self.candidate_logprobs_top_k = int(args.candidate_logprobs_top_k)
        self.candidate_hidden_readout = bool(args.candidate_hidden_readout)
        self.candidate_attention_summary = bool(args.candidate_attention_summary)
        self.enable_recovery_kv_checkpoint = self.rollback_backend in {
            "kv_restore",
            "kv_restore_strict",
        }
        self.kv_restore_strict = self.rollback_backend == "kv_restore_strict"
        self._cumulative_divergence = 0.0
        if self.verifier in {"instant_kv", "cumulative_kv"}:
            self.verifier = "kv_divergence"
        if self.verifier == "logistic":
            self.logistic_detector_model = self._train_logistic_detector(
                self.logistic_detector_features_csv
            )

    def _full_prompt_input_ids(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> list[int]:
        def coerce_input_ids(encoded: Any) -> list[int]:
            if isinstance(encoded, dict):
                encoded = encoded.get("input_ids")
            elif hasattr(encoded, "keys") and "input_ids" in encoded.keys():
                encoded = encoded["input_ids"]
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if (
                isinstance(encoded, list)
                and encoded
                and isinstance(encoded[0], list)
            ):
                if len(encoded) != 1:
                    raise ValueError(
                        "Expected a single rendered chat prompt, got batched input_ids."
                    )
                encoded = encoded[0]
            if not isinstance(encoded, list) or not all(
                isinstance(token_id, int) for token_id in encoded
            ):
                raise TypeError(
                    "tokenizer.apply_chat_template did not return a list[int] "
                    "or an object containing input_ids."
                )
            return list(encoded)

        try:
            return coerce_input_ids(
                self.tokenizer.apply_chat_template(
                    list(messages),
                    tools=list(tools),
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except TypeError:
            try:
                return coerce_input_ids(
                    self.tokenizer.apply_chat_template(
                        list(messages),
                        tools=list(tools),
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
            except Exception as exc:
                if self.enable_recovery_kv_checkpoint:
                    raise RuntimeError(
                        "KV checkpoint requires exact server-equivalent prompt "
                        "tokenization; tokenizer.apply_chat_template failed."
                    ) from exc
                rendered = json.dumps(
                    {"messages": list(messages), "tools": list(tools)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                return list(self.tokenizer.encode(rendered, add_special_tokens=False))
        except Exception as exc:
            if self.enable_recovery_kv_checkpoint:
                raise RuntimeError(
                    "KV checkpoint requires exact server-equivalent prompt "
                    "tokenization; tokenizer.apply_chat_template failed."
                ) from exc
            rendered = json.dumps(
                {"messages": list(messages), "tools": list(tools)},
                ensure_ascii=False,
                sort_keys=True,
            )
            return list(self.tokenizer.encode(rendered, add_special_tokens=False))

    @staticmethod
    def _rank_status_items(response: dict[str, Any]) -> list[dict[str, Any]]:
        status = response.get("status") or {}
        ranks = status.get("ranks")
        if isinstance(ranks, list):
            return [item for item in ranks if isinstance(item, dict)]
        if isinstance(status, dict):
            return [status]
        return [response]

    @classmethod
    def _max_status_int(cls, response: dict[str, Any], key: str) -> int:
        values = []
        for item in cls._rank_status_items(response):
            try:
                values.append(int(item.get(key) or 0))
            except Exception:
                pass
        try:
            values.append(int(response.get(key) or 0))
        except Exception:
            pass
        return max(values) if values else 0

    @staticmethod
    def _generate_prompt_tokens(response: dict[str, Any], fallback: int) -> int:
        if isinstance(response, list):
            response = response[0] if response else {}
        meta = response.get("meta_info") or response.get("meta") or {}
        usage = response.get("usage") or {}
        for source in (usage, meta, response):
            for key in (
                "prompt_tokens",
                "input_tokens",
                "prefill_tokens",
                "num_prompt_tokens",
            ):
                try:
                    value = int(source.get(key) or 0)
                except Exception:
                    value = 0
                if value:
                    return value
        return int(fallback)

    @staticmethod
    def _actual_cached_tokens_from_response(response: dict[str, Any]) -> int | None:
        if isinstance(response, list):
            response = response[0] if response else {}
        usage = response.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
            try:
                return int(prompt_details.get("cached_tokens") or 0)
            except Exception:
                return 0

        sglext = response.get("sglext") or {}
        cached_details = sglext.get("cached_tokens_details") or {}
        if isinstance(cached_details, dict) and cached_details:
            total = 0
            seen = False
            for key in ("device", "host", "storage"):
                if key in cached_details:
                    seen = True
                    try:
                        total += int(cached_details.get(key) or 0)
                    except Exception:
                        pass
            if seen:
                return total

        meta_info = response.get("meta_info") or {}
        if isinstance(meta_info, dict) and "cached_tokens" in meta_info:
            try:
                return int(meta_info.get("cached_tokens") or 0)
            except Exception:
                return 0
        return None

    def _kv_checkpoint_metadata(
        self,
        *,
        test_entry_id: str,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        global_step: int,
        checkpoint_id: int,
        parent_checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        history_units = _history_units(messages)
        prompt_input_ids = self._full_prompt_input_ids(messages, tools)
        prompt_tokens = len(prompt_input_ids)
        sglang_checkpoint_id = (
            f"{test_entry_id}_fullkv_ckpt{checkpoint_id}_step{global_step}_"
            f"{_stable_hash(prompt_input_ids)[:12]}"
        )
        metadata = {
            "available": False,
            "checkpoint_id": checkpoint_id,
            "sglang_checkpoint_id": sglang_checkpoint_id,
            "global_step": global_step,
            "message_count": len(messages),
            "history_units": len(history_units),
            "prompt_token_estimate": _token_count(self.tokenizer, messages),
            "full_prompt_tokens": prompt_tokens,
            "requested_checkpoint_tokens": prompt_tokens,
            "aligned_checkpoint_tokens": 0,
            "cache_handle": None,
            "sequence_length": 0,
            "position_metadata": None,
            "page_size": None,
            "c2kv_cache_metadata": [
                {
                    "message_index": index,
                    "c2kv_key_hash": message.get("c2kv_key_hash"),
                }
                for index, message in enumerate(messages)
                if message.get("c2kv_key_hash")
            ],
        }
        if not self.enable_recovery_kv_checkpoint:
            metadata["limitation"] = "rollback_backend_is_message_replay"
            return metadata
        if prompt_tokens <= 0:
            metadata["limitation"] = "empty_full_prompt"
            return metadata

        started = time.perf_counter()
        try:
            generate_response = _post_json(
                self.base_url,
                "/generate",
                {
                    "input_ids": prompt_input_ids,
                    "sampling_params": {
                        "max_new_tokens": 0,
                        "temperature": 0,
                    },
                },
                self.timeout,
            )
            maintenance_latency = time.perf_counter() - started
            maintenance_prompt_tokens = self._generate_prompt_tokens(
                generate_response,
                prompt_tokens,
            )
            maintenance_cached_tokens = self._actual_cached_tokens_from_response(
                generate_response
            )
            maintenance_cache_report_missing = maintenance_cached_tokens is None
            if maintenance_cached_tokens is None:
                maintenance_cached_tokens = 0
            maintenance_cached_tokens = min(
                maintenance_prompt_tokens,
                max(0, int(maintenance_cached_tokens)),
            )
            maintenance_recomputed_tokens = max(
                maintenance_prompt_tokens - maintenance_cached_tokens,
                0,
            )
            create_response = _post_json(
                self.base_url,
                "/recovery_checkpoint/create",
                {
                    "checkpoint_id": sglang_checkpoint_id,
                    "input_ids": prompt_input_ids,
                    "session_id": test_entry_id,
                    "segment_id": checkpoint_id,
                    "global_step": global_step,
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "tier": "host",
                    "evict_device_after": True,
                    "sync": True,
                },
                self.timeout,
            )
            create_latency = time.perf_counter() - started - maintenance_latency
            checkpoint_tokens = self._max_status_int(
                create_response,
                "checkpoint_tokens",
            )
            if not checkpoint_tokens:
                checkpoint_tokens = self._max_status_int(
                    create_response,
                    "token_count",
                )
            page_size = self._max_status_int(create_response, "page_size") or None
            requested_tokens = (
                self._max_status_int(create_response, "requested_tokens")
                or prompt_tokens
            )
            metadata.update(
                {
                    "available": bool(create_response.get("success")),
                    "cache_handle": sglang_checkpoint_id,
                    "requested_checkpoint_tokens": requested_tokens,
                    "aligned_checkpoint_tokens": checkpoint_tokens,
                    "sequence_length": checkpoint_tokens,
                    "page_size": page_size,
                    "generate_response": make_json_serializable(generate_response),
                    "create_response": make_json_serializable(create_response),
                    "checkpoint_maintenance_logical_prompt_tokens": prompt_tokens,
                    "checkpoint_maintenance_prompt_tokens": maintenance_prompt_tokens,
                    "checkpoint_maintenance_reused_tokens": maintenance_cached_tokens,
                    "checkpoint_maintenance_recomputed_tokens": (
                        maintenance_recomputed_tokens
                    ),
                    "checkpoint_maintenance_cache_report_missing": (
                        maintenance_cache_report_missing
                    ),
                    "checkpoint_create_backup_tokens": self._max_status_int(
                        create_response,
                        "backup_tokens",
                    ),
                    "checkpoint_host_tokens": self._max_status_int(
                        create_response,
                        "host_tokens",
                    ),
                    "checkpoint_device_tokens": self._max_status_int(
                        create_response,
                        "device_tokens",
                    ),
                    "checkpoint_backup_latency_ms": self._max_status_int(
                        create_response,
                        "backup_latency_ms",
                    ),
                    "checkpoint_maintenance_latency_sec": maintenance_latency,
                    "checkpoint_create_latency_sec": create_latency,
                    "limitation": None,
                }
            )
        except Exception as exc:
            metadata.update(
                {
                    "available": False,
                    "limitation": "checkpoint_create_failed",
                    "error": str(exc),
                    "checkpoint_maintenance_latency_sec": (
                        time.perf_counter() - started
                    ),
                }
            )
        return metadata

    @staticmethod
    def _empty_checkpoint_maintenance() -> dict[str, int]:
        return {
            "checkpoint_maintenance_reused_tokens": 0,
            "checkpoint_maintenance_recomputed_tokens": 0,
            "checkpoint_maintenance_logical_prompt_tokens": 0,
            "checkpoint_maintenance_cache_report_missing": 0,
        }

    @classmethod
    def _add_checkpoint_maintenance(
        cls,
        totals: dict[str, int],
        metadata: dict[str, Any] | None,
    ) -> None:
        if not metadata:
            return
        for key in cls._empty_checkpoint_maintenance():
            totals[key] = int(totals.get(key) or 0) + int(metadata.get(key) or 0)

    def _restore_kv_checkpoint_metadata(
        self,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        sglang_checkpoint_id = metadata.get("sglang_checkpoint_id")
        if not metadata.get("available") or not sglang_checkpoint_id:
            return {
                "success": False,
                "fallback_reason": (
                    metadata.get("limitation")
                    or metadata.get("error")
                    or "KV_CHECKPOINT_NOT_AVAILABLE"
                ),
            }
        try:
            response = _post_json(
                self.base_url,
                "/recovery_checkpoint/restore",
                {
                    "checkpoint_id": sglang_checkpoint_id,
                    "sync": True,
                    "pin_device": True,
                },
                self.timeout,
            )
            return make_json_serializable(response)
        except Exception as exc:
            return {
                "success": False,
                "fallback_reason": f"restore_exception:{exc}",
            }

    def _release_kv_checkpoint_metadata(
        self,
        metadata: dict[str, Any] | None,
    ) -> None:
        if not metadata:
            return
        checkpoint_id = metadata.get("sglang_checkpoint_id")
        if not checkpoint_id or metadata.get("released"):
            return
        try:
            response = _post_json(
                self.base_url,
                "/recovery_checkpoint/release",
                {"checkpoint_id": checkpoint_id},
                self.timeout,
            )
            metadata["release_response"] = make_json_serializable(response)
            metadata["released"] = bool(response.get("success"))
        except Exception as exc:
            metadata["release_error"] = str(exc)

    def _release_kv_checkpoint(self, snapshot: dict[str, Any] | None) -> None:
        if not snapshot:
            return
        self._release_kv_checkpoint_metadata(
            snapshot.get("kv_checkpoint_metadata")
        )

    def _assert_kv_checkpoint_available(
        self,
        metadata: dict[str, Any] | None,
        *,
        context: str,
    ) -> None:
        if not self.kv_restore_strict:
            return
        metadata = metadata or {}
        if metadata.get("available") and metadata.get("sglang_checkpoint_id"):
            return
        reason_parts = []
        for key in (
            "error",
            "limitation",
            "fallback_reason",
            "create_response",
            "generate_response",
        ):
            value = metadata.get(key)
            if value is not None:
                if not isinstance(value, str):
                    value = json.dumps(
                        make_json_serializable(value),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                reason_parts.append(f"{key}={value[:1200]}")
        reason = "; ".join(reason_parts) or "KV_CHECKPOINT_CREATE_FAILED"
        raise RuntimeError(f"kv_restore_strict {context} failed: {reason}")

    def _release_kv_snapshots(self, snapshots: Sequence[dict[str, Any] | None]) -> None:
        seen = set()
        for snapshot in snapshots:
            metadata = snapshot.get("kv_checkpoint_metadata") or {}
            checkpoint_id = metadata.get("sglang_checkpoint_id")
            if not checkpoint_id or checkpoint_id in seen:
                continue
            seen.add(checkpoint_id)
            self._release_kv_checkpoint(snapshot)

    def _account_recovery_prompt_work(
        self,
        backend_info: dict[str, Any],
        prompt_tokens: int,
        actual_cached_tokens: int | None,
    ) -> None:
        prompt_tokens = int(prompt_tokens or 0)
        backend_info["recovery_logical_prompt_tokens"] = int(
            backend_info.get("recovery_logical_prompt_tokens") or 0
        ) + prompt_tokens
        if (
            backend_info.get("rollback_backend") in {"kv_restore", "kv_restore_strict"}
            and backend_info.get("kv_restore_success")
            and not backend_info.get("kv_restore_fallback")
        ):
            cached_prefix = int(backend_info.get("_cached_prefix_tokens") or 0)
            expected_reused = min(prompt_tokens, cached_prefix)
            expected_recomputed = max(prompt_tokens - expected_reused, 0)
            backend_info["expected_kv_reused_tokens"] = int(
                backend_info.get("expected_kv_reused_tokens") or 0
            ) + expected_reused
            backend_info["expected_kv_recomputed_tokens"] = int(
                backend_info.get("expected_kv_recomputed_tokens") or 0
            ) + expected_recomputed
            if actual_cached_tokens is None:
                backend_info["actual_cache_report_missing"] = int(
                    backend_info.get("actual_cache_report_missing") or 0
                ) + 1
                actual_cached_tokens = 0
            reused = min(prompt_tokens, max(0, int(actual_cached_tokens)))
            recomputed = max(prompt_tokens - reused, 0)
            backend_info["kv_reused_tokens"] = int(
                backend_info.get("kv_reused_tokens") or 0
            ) + reused
            backend_info["kv_recomputed_tokens"] = int(
                backend_info.get("kv_recomputed_tokens") or 0
            ) + recomputed
            backend_info["_cached_prefix_tokens"] = prompt_tokens
        else:
            backend_info["message_replay_prefill_tokens"] = int(
                backend_info.get("message_replay_prefill_tokens") or 0
            ) + prompt_tokens

    def _restore_with_backend(
        self,
        *,
        test_entry_id: str,
        snapshot: dict[str, Any],
        tools: Sequence[dict[str, Any]],
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
            "expected_kv_reused_tokens": 0,
            "expected_kv_recomputed_tokens": 0,
            "actual_cache_report_missing": 0,
            "message_replay_prefill_tokens": 0,
            "recovery_logical_prompt_tokens": 0,
            "restored_checkpoint_tokens": 0,
            "restore_loaded_from_host_tokens": 0,
            "restore_already_device_tokens": 0,
            "safe_delta_prefill_tokens": 0,
            "safe_delta_reused_tokens": 0,
            "safe_delta_logical_prompt_tokens": 0,
            "_cached_prefix_tokens": 0,
            "restore_latency_sec": 0.0,
            "rollback_latency_sec": 0.0,
            "kv_checkpoint_metadata": snapshot.get("kv_checkpoint_metadata")
            or snapshot.get("kv_anchor_checkpoint_metadata"),
            "logical_snapshot_has_kv_checkpoint": bool(
                snapshot.get("kv_checkpoint_metadata")
            ),
        }
        if self.enable_recovery_kv_checkpoint:
            metadata = (
                snapshot.get("kv_checkpoint_metadata")
                or snapshot.get("kv_anchor_checkpoint_metadata")
                or {}
            )
            sglang_checkpoint_id = metadata.get("sglang_checkpoint_id")
            if metadata.get("available") and sglang_checkpoint_id:
                restore_started = time.perf_counter()
                restore_response = self._restore_kv_checkpoint_metadata(metadata)
                backend_info["restore_latency_sec"] = (
                    time.perf_counter() - restore_started
                )
                if restore_response.get("success"):
                    restored_tokens = int(metadata.get("aligned_checkpoint_tokens") or 0)
                    backend_info.update(
                        {
                            "kv_restore_success": True,
                            "restore_response": make_json_serializable(
                                restore_response
                            ),
                            "restored_checkpoint_tokens": restored_tokens,
                            "restore_loaded_from_host_tokens": self._max_status_int(
                                restore_response,
                                "loaded_from_host_tokens",
                            ),
                            "restore_already_device_tokens": self._max_status_int(
                                restore_response,
                                "already_device_tokens",
                            ),
                            "_cached_prefix_tokens": restored_tokens,
                        }
                    )
                else:
                    reason = (
                        restore_response.get("fallback_reason")
                        or restore_response.get("message")
                        or restore_response.get("error")
                        or "RESTORE_FAILED"
                    )
                    if self.kv_restore_strict:
                        raise RuntimeError(
                            "kv_restore_strict restore failed: "
                            f"{reason}; checkpoint={sglang_checkpoint_id}"
                        )
                    backend_info.update(
                        {
                            "kv_restore_fallback": True,
                            "kv_restore_fallback_reason": reason,
                            "restore_response": make_json_serializable(
                                restore_response
                            ),
                        }
                )
            else:
                reason = (
                    metadata.get("limitation")
                    or metadata.get("error")
                    or "KV_CHECKPOINT_NOT_AVAILABLE"
                )
                if self.kv_restore_strict:
                    raise RuntimeError(
                        "kv_restore_strict restore failed: "
                        f"{reason}; checkpoint={sglang_checkpoint_id or '<missing>'}"
                    )
                backend_info.update(
                    {
                        "kv_restore_fallback": True,
                        "kv_restore_fallback_reason": reason,
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
        if (
            self.enable_recovery_kv_checkpoint
            and backend_info.get("kv_restore_success")
            and not backend_info.get("kv_restore_fallback")
        ):
            prompt_input_ids = self._full_prompt_input_ids(messages, tools)
            prompt_tokens = len(prompt_input_ids)
            restored_tokens = int(backend_info.get("restored_checkpoint_tokens") or 0)
            if prompt_tokens > restored_tokens:
                try:
                    warmup_started = time.perf_counter()
                    warmup_response = _post_json(
                        self.base_url,
                        "/generate",
                        {
                            "input_ids": prompt_input_ids,
                            "sampling_params": {
                                "max_new_tokens": 0,
                                "temperature": 0,
                            },
                        },
                        self.timeout,
                    )
                    warmup_elapsed = time.perf_counter() - warmup_started
                    expected_reused = min(prompt_tokens, restored_tokens)
                    expected_recomputed = max(prompt_tokens - expected_reused, 0)
                    actual_cached = self._actual_cached_tokens_from_response(
                        warmup_response
                    )
                    if actual_cached is None:
                        backend_info["actual_cache_report_missing"] = int(
                            backend_info.get("actual_cache_report_missing") or 0
                        ) + 1
                        actual_cached = 0
                    reused = min(prompt_tokens, max(0, int(actual_cached)))
                    recomputed = max(prompt_tokens - reused, 0)
                    backend_info["safe_delta_logical_prompt_tokens"] = prompt_tokens
                    backend_info["recovery_logical_prompt_tokens"] = int(
                        backend_info.get("recovery_logical_prompt_tokens") or 0
                    ) + prompt_tokens
                    backend_info["safe_delta_reused_tokens"] = reused
                    backend_info["safe_delta_prefill_tokens"] = recomputed
                    backend_info["safe_delta_expected_reused_tokens"] = (
                        expected_reused
                    )
                    backend_info["safe_delta_expected_prefill_tokens"] = (
                        expected_recomputed
                    )
                    backend_info["expected_kv_reused_tokens"] = int(
                        backend_info.get("expected_kv_reused_tokens") or 0
                    ) + expected_reused
                    backend_info["expected_kv_recomputed_tokens"] = int(
                        backend_info.get("expected_kv_recomputed_tokens") or 0
                    ) + expected_recomputed
                    backend_info["kv_reused_tokens"] = int(
                        backend_info.get("kv_reused_tokens") or 0
                    ) + reused
                    backend_info["kv_recomputed_tokens"] = int(
                        backend_info.get("kv_recomputed_tokens") or 0
                    ) + recomputed
                    backend_info["_cached_prefix_tokens"] = prompt_tokens
                    backend_info["safe_delta_warmup_latency_sec"] = warmup_elapsed
                    backend_info["safe_delta_warmup_response"] = (
                        make_json_serializable(warmup_response)
                    )
                except Exception as exc:
                    if self.kv_restore_strict:
                        raise RuntimeError(
                            "kv_restore_strict safe-delta warmup failed: "
                            f"{exc}"
                        ) from exc
                    backend_info.update(
                        {
                            "kv_restore_success": False,
                            "kv_restore_fallback": True,
                            "kv_restore_fallback_reason": (
                                f"safe_delta_warmup_exception:{exc}"
                            ),
                            "_cached_prefix_tokens": 0,
                        }
                    )
            else:
                backend_info["_cached_prefix_tokens"] = prompt_tokens
        elapsed = time.perf_counter() - start
        if not backend_info.get("restore_latency_sec"):
            backend_info["restore_latency_sec"] = elapsed
        backend_info["rollback_latency_sec"] = elapsed
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

    def _heuristic_detector_failed(
        self,
        attrs: dict[str, Any],
    ) -> tuple[bool, str]:
        risk_score = float(attrs.get("risk_score") or 0.0)
        if self.verify_threshold > 0:
            return (
                risk_score >= self.verify_threshold,
                f"risk_score>={self.verify_threshold:g}",
            )
        if attrs.get("hard_error"):
            return True, "hard_error"
        if attrs.get("argument_grounding_failure"):
            return True, "argument_grounding_failure"
        if float(attrs.get("observation_anomaly") or 0.0) >= 1.0:
            return True, "observation_anomaly"
        if float(attrs.get("tool_transition_anomaly") or 0.0) >= 1.0:
            return True, "tool_transition_anomaly"
        return False, "rule_safe"

    def _rule_detector(
        self,
        segment_infos: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        attrs = [info.get("heuristic_attributes") or {} for info in segment_infos]
        hard_trigger = any(bool(attr.get("hard_error")) for attr in attrs)
        grounding_trigger = any(
            bool(attr.get("argument_grounding_failure")) for attr in attrs
        )
        observation_trigger = any(
            float(attr.get("observation_anomaly") or 0.0) >= 1.0
            for attr in attrs
        )
        max_risk = max(
            (float(attr.get("risk_score") or 0.0) for attr in attrs),
            default=0.0,
        )
        risk_trigger = max_risk >= self.rule_detector_threshold
        if hard_trigger:
            reason = "hard_error"
        elif grounding_trigger:
            reason = "argument_grounding"
        elif observation_trigger:
            reason = "observation_anomaly"
        elif risk_trigger:
            reason = "risk_threshold"
        else:
            reason = "none"
        triggered = (
            hard_trigger
            or grounding_trigger
            or observation_trigger
            or risk_trigger
        )
        return {
            "detector": "rule",
            "detector_trigger": triggered,
            "detector_reason": reason,
            "rule_detector_trigger": triggered,
            "rule_detector_max_risk": max_risk,
            "rule_detector_reason": reason,
            "rule_detector_threshold": self.rule_detector_threshold,
        }

    @staticmethod
    def _detector_low_is_bad(name: str) -> bool:
        return any(
            pattern in name
            for pattern in (
                "confidence",
                "probability",
                "logprob",
                "margin",
                "grounding_score",
            )
        )

    @classmethod
    def _detector_score_for_feature(cls, name: str, value: Any) -> float | None:
        numeric = _as_float(value)
        if numeric is None:
            return None
        return -numeric if cls._detector_low_is_bad(name) else numeric

    @staticmethod
    def _detector_aggregate_features(
        step_features: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for features in step_features:
            if not isinstance(features, dict):
                continue
            for key, value in features.items():
                numeric = _as_float(value)
                if numeric is not None:
                    values.setdefault(key, []).append(numeric)
        out: dict[str, float] = {}
        for key, vals in values.items():
            if not vals:
                continue
            out[f"mean_{key}"] = sum(vals) / len(vals)
            out[f"max_{key}"] = max(vals)
            out[f"min_{key}"] = min(vals)
        return out

    def _segment_detector_feature_row(
        self,
        segment_infos: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        merged_steps: list[dict[str, Any]] = []
        for info in segment_infos:
            merged: dict[str, Any] = {}
            for source in (
                info.get("candidate_detector_features") or {},
                info.get("heuristic_attributes") or {},
            ):
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    numeric = _as_float(value)
                    if numeric is not None:
                        merged[key] = numeric
            merged_steps.append(merged)
        features: dict[str, Any] = self._detector_aggregate_features(merged_steps)
        rule = self._rule_detector(segment_infos)
        features.update(
            {
                "rule_detector_trigger": int(
                    bool(rule.get("rule_detector_trigger"))
                ),
                "rule_detector_binary_score": float(
                    bool(rule.get("rule_detector_trigger"))
                ),
                "rule_detector_max_risk": rule.get("rule_detector_max_risk"),
                "rule_detector_threshold": self.rule_detector_threshold,
            }
        )
        return features

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _best_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in sorted(set(scores), reverse=True):
            tp = fp = fn = 0
            for label, score in zip(labels, scores):
                pred = score >= threshold
                if pred and label:
                    tp += 1
                elif pred and not label:
                    fp += 1
                elif not pred and label:
                    fn += 1
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
        return best_threshold

    def _train_logistic_detector(self, features_csv: str) -> dict[str, Any]:
        if not features_csv:
            raise ValueError(
                "verifier=logistic requires --logistic-detector-features-csv"
            )
        with open(features_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        train = [row for row in rows if row.get("split") == "calibration"]
        if not train:
            raise ValueError(
                "logistic detector feature CSV has no calibration split rows: "
                f"{features_csv}"
            )
        excluded = {
            "id",
            "checkpoint_id",
            "turn",
            "segment_start_step",
            "segment_length",
            "segment_harmful",
            "split",
            "rule_detector_reason",
        }
        feature_names = [
            key
            for key in rows[0].keys()
            if key not in excluded
        ]
        usable: list[str] = []
        for name in feature_names:
            vals = [
                self._detector_score_for_feature(name, row.get(name))
                for row in train
            ]
            vals = [value for value in vals if value is not None]
            if len(vals) >= max(4, len(train) // 2):
                usable.append(name)
        if not usable:
            raise ValueError("logistic detector has no usable numeric features")

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for name in usable:
            vals = [
                self._detector_score_for_feature(name, row.get(name))
                for row in train
            ]
            vals = [value for value in vals if value is not None]
            mean = sum(vals) / len(vals)
            var = sum((value - mean) ** 2 for value in vals) / max(len(vals), 1)
            means[name] = mean
            stds[name] = math.sqrt(var) or 1.0

        def vector(row: dict[str, Any]) -> list[float]:
            out = [1.0]
            for name in usable:
                value = self._detector_score_for_feature(name, row.get(name))
                if value is None:
                    value = means[name]
                out.append((value - means[name]) / stds[name])
            return out

        weights = [0.0] * (len(usable) + 1)
        lr = 0.08
        l2 = 0.001
        for _ in range(500):
            grad = [0.0] * len(weights)
            for row in train:
                x = vector(row)
                y = int(float(row["segment_harmful"]))
                p = self._sigmoid(sum(w * xi for w, xi in zip(weights, x)))
                for i, xi in enumerate(x):
                    grad[i] += (p - y) * xi
            for i in range(len(weights)):
                grad[i] /= len(train)
                if i:
                    grad[i] += l2 * weights[i]
                weights[i] -= lr * grad[i]

        train_scores = [
            self._sigmoid(sum(w * xi for w, xi in zip(weights, vector(row))))
            for row in train
        ]
        train_labels = [int(float(row["segment_harmful"])) for row in train]
        threshold = (
            self._best_f1_threshold(train_labels, train_scores)
            if self.logistic_detector_threshold < 0.0
            else self.logistic_detector_threshold
        )
        return {
            "features_csv": features_csv,
            "features": usable,
            "means": means,
            "stds": stds,
            "weights": weights,
            "threshold": threshold,
            "train_rows": len(train),
            "train_episodes": len({row.get("id") for row in train}),
        }

    def _logistic_detector(
        self,
        segment_infos: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.logistic_detector_model is None:
            raise RuntimeError("logistic detector model is not initialized")
        model = self.logistic_detector_model
        row = self._segment_detector_feature_row(segment_infos)
        x = [1.0]
        for name in model["features"]:
            value = self._detector_score_for_feature(name, row.get(name))
            if value is None:
                value = model["means"][name]
            x.append((value - model["means"][name]) / model["stds"][name])
        score = self._sigmoid(
            sum(w * xi for w, xi in zip(model["weights"], x))
        )
        triggered = score >= float(model["threshold"])
        return {
            "detector": "logistic",
            "detector_trigger": triggered,
            "detector_reason": (
                "logistic_score_threshold" if triggered else "logistic_safe"
            ),
            "logistic_detector_score": score,
            "logistic_detector_threshold": model["threshold"],
            "logistic_detector_feature_count": len(model["features"]),
            "logistic_detector_train_rows": model["train_rows"],
            "logistic_detector_train_episodes": model["train_episodes"],
            "rule_detector_trigger": row.get("rule_detector_trigger"),
            "rule_detector_max_risk": row.get("rule_detector_max_risk"),
            "rule_detector_reason": None,
        }

    def _feature_signal_detector(
        self,
        segment_infos: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.detector_signal_name:
            raise RuntimeError(
                "verifier=feature_signal requires --detector-signal-name"
            )
        row = self._segment_detector_feature_row(segment_infos)
        score = self._detector_score_for_feature(
            self.detector_signal_name,
            row.get(self.detector_signal_name),
        )
        triggered = (
            False
            if score is None
            else score >= self.detector_signal_threshold
        )
        return {
            "detector": "feature_signal",
            "detector_trigger": triggered,
            "detector_reason": (
                f"{self.detector_signal_name}>={self.detector_signal_threshold:g}"
                if triggered
                else "feature_signal_safe"
            ),
            "detector_signal_name": self.detector_signal_name,
            "detector_signal_score": score,
            "detector_signal_threshold": self.detector_signal_threshold,
            "rule_detector_trigger": row.get("rule_detector_trigger"),
            "rule_detector_max_risk": row.get("rule_detector_max_risk"),
            "rule_detector_reason": None,
        }

    @staticmethod
    def _oracle_harmful(info: dict[str, Any]) -> bool:
        verify = info.get("verify") or {}
        return (
            not bool(verify.get("candidate_action_matches_reference"))
            or verify.get("state_matches_reference") is False
        )

    @staticmethod
    def _detector_confusion(
        *,
        oracle_segment_unsafe: bool,
        detector_segment_trigger: bool,
    ) -> dict[str, bool]:
        return {
            "detector_tp": detector_segment_trigger and oracle_segment_unsafe,
            "detector_fp": detector_segment_trigger and not oracle_segment_unsafe,
            "detector_tn": (not detector_segment_trigger)
            and (not oracle_segment_unsafe),
            "detector_fn": (not detector_segment_trigger) and oracle_segment_unsafe,
        }

    def _rollback_debug(
        self,
        *,
        segment_infos: Sequence[dict[str, Any]],
        rollback_start_index: int,
        oracle_first_bad_index: int | None,
        reason: str,
        raw_predicted_first_bad_index: int | None = None,
    ) -> dict[str, Any]:
        has_oracle_gt = oracle_first_bad_index is not None
        return {
            "attribution": self.attribution,
            "rollback_policy": self.rollback_policy,
            "oracle_first_bad_index": oracle_first_bad_index,
            "predicted_first_bad_index": rollback_start_index,
            "raw_predicted_first_bad_index": raw_predicted_first_bad_index,
            "attribution_reason": reason,
            "attribution_safety_margin": self.attribution_safety_margin,
            "has_oracle_first_bad": has_oracle_gt,
            "detector_false_positive": not has_oracle_gt,
            "exact_attribution": (
                rollback_start_index == oracle_first_bad_index
                if has_oracle_gt
                else None
            ),
            "within1_attribution": (
                abs(rollback_start_index - oracle_first_bad_index) <= 1
                if has_oracle_gt
                else None
            ),
            "rollback_coverage": (
                rollback_start_index <= oracle_first_bad_index
                if has_oracle_gt
                else None
            ),
            "under_rollback": (
                rollback_start_index > oracle_first_bad_index
                if has_oracle_gt
                else None
            ),
            "over_rollback": (
                rollback_start_index < oracle_first_bad_index
                if has_oracle_gt
                else None
            ),
            "over_rollback_steps": (
                max(0, oracle_first_bad_index - rollback_start_index)
                if has_oracle_gt
                else None
            ),
            "predicted_rollback_depth": len(segment_infos) - rollback_start_index,
            "oracle_rollback_depth": (
                len(segment_infos) - oracle_first_bad_index if has_oracle_gt else None
            ),
        }

    @staticmethod
    def _quantize_rollback_depth(required_depth: int) -> int:
        if required_depth <= 1:
            return 1
        if required_depth <= 2:
            return 2
        return 4

    @staticmethod
    def _strong_suspicious_reason(attrs: dict[str, Any]) -> str | None:
        if attrs.get("hard_error"):
            return "hard_error"
        if attrs.get("argument_grounding_failure"):
            return "argument_grounding_failure"
        if float(attrs.get("observation_anomaly") or 0.0) >= 1.0:
            return "observation_anomaly"
        if float(attrs.get("tool_transition_anomaly") or 0.0) >= 1.0:
            return "tool_transition_anomaly"
        return None

    def _rule_depth_decision(
        self,
        segment_infos: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        segment_len = len(segment_infos)
        attrs = [info.get("heuristic_attributes") or {} for info in segment_infos]
        strong_indices = [
            index
            for index, attr in enumerate(attrs)
            if self._strong_suspicious_reason(attr) is not None
        ]
        hard_front_indices = [
            index
            for index, attr in enumerate(attrs)
            if attr.get("hard_error") and index < (segment_len / 2.0)
        ]
        max_risk_index = max(
            range(segment_len),
            key=lambda idx: float(attrs[idx].get("risk_score") or 0.0),
        )

        earliest_suspicious_index: int | None = None
        if hard_front_indices:
            earliest_suspicious_index = min(hard_front_indices)
            predicted_depth = 4
            reason = "hard_error_front_half"
        elif strong_indices:
            earliest_suspicious_index = min(strong_indices)
            required_depth = segment_len - earliest_suspicious_index
            predicted_depth = self._quantize_rollback_depth(required_depth)
            reason = (
                "strong_signal:"
                + (
                    self._strong_suspicious_reason(
                        attrs[earliest_suspicious_index]
                    )
                    or "unknown"
                )
            )
        else:
            required_depth = segment_len - max_risk_index
            predicted_depth = self._quantize_rollback_depth(required_depth)
            reason = "max_risk_score"

        if len(strong_indices) >= 2:
            predicted_depth = max(predicted_depth, 2)
            reason = reason + "+multi_strong_signal"

        actual_depth = min(predicted_depth, segment_len)
        rollback_start_index = segment_len - actual_depth
        return {
            "predicted_rollback_depth": predicted_depth,
            "actual_rollback_depth": actual_depth,
            "rollback_start_index": rollback_start_index,
            "rule_depth_reason": reason,
            "earliest_suspicious_index": earliest_suspicious_index,
            "max_risk_index": max_risk_index,
            "strong_suspicious_count": len(strong_indices),
            "strong_suspicious_indices": strong_indices,
        }

    def _predict_first_bad_index(
        self,
        *,
        segment_infos: Sequence[dict[str, Any]],
        oracle_first_bad_index: int | None,
    ) -> tuple[int, dict[str, Any]]:
        if self.attribution == "whole_segment":
            predicted = 0
            reason = "whole_segment"
        elif self.attribution == "oracle_first_bad":
            predicted = oracle_first_bad_index
            if predicted is None:
                predicted = 0
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
        has_oracle_gt = oracle_first_bad_index is not None
        return predicted, {
            "attribution": self.attribution,
            "oracle_first_bad_index": oracle_first_bad_index,
            "predicted_first_bad_index": predicted,
            "raw_predicted_first_bad_index": raw_predicted,
            "attribution_reason": reason,
            "attribution_safety_margin": self.attribution_safety_margin,
            "has_oracle_first_bad": has_oracle_gt,
            "detector_false_positive": not has_oracle_gt,
            "exact_attribution": (
                predicted == oracle_first_bad_index if has_oracle_gt else None
            ),
            "within1_attribution": (
                abs(predicted - oracle_first_bad_index) <= 1
                if has_oracle_gt
                else None
            ),
            "under_rollback": (
                predicted > oracle_first_bad_index if has_oracle_gt else None
            ),
            "over_rollback": (
                predicted < oracle_first_bad_index if has_oracle_gt else None
            ),
            "rollback_coverage": (
                predicted <= oracle_first_bad_index if has_oracle_gt else None
            ),
            "over_rollback_steps": (
                max(0, oracle_first_bad_index - predicted) if has_oracle_gt else None
            ),
            "predicted_rollback_depth": len(segment_infos) - predicted,
            "oracle_rollback_depth": (
                len(segment_infos) - oracle_first_bad_index if has_oracle_gt else None
            ),
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
        collect_detector_signals: bool = False,
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
            "return_cached_tokens_details": True,
        }
        if readout_probe or collect_detector_signals:
            payload.update(
                {
                    "logprobs": True,
                    "top_logprobs": self.candidate_logprobs_top_k,
                }
            )
            # Native SGLang endpoints use return_logprob/top_logprobs_num.
            # The OpenAI chat endpoint ignores unknown fields in newer forks but
            # this keeps older local forks from silently dropping the request.
            payload["return_logprob"] = True
            payload["top_logprobs_num"] = self.candidate_logprobs_top_k
            payload["return_text_in_logprobs"] = True
        if readout_probe or (
            collect_detector_signals and self.candidate_hidden_readout
        ):
            payload["return_hidden_states"] = True
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
        cached_tokens = self._actual_cached_tokens_from_response(data)
        prompt_token_count = int(usage.get("prompt_tokens") or prompt_tokens)
        parsed_usage = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": int(
                usage.get("completion_tokens")
                or len(self.tokenizer.encode(text, add_special_tokens=False))
            ),
            "cached_tokens": cached_tokens,
            "recomputed_tokens": (
                max(prompt_token_count - int(cached_tokens), 0)
                if cached_tokens is not None
                else None
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

    def _candidate_detector_features(
        self,
        raw: dict[str, Any],
        previous_readout_vector: list[float] | None,
    ) -> tuple[dict[str, Any], list[float] | None]:
        choice = (raw.get("choices") or [{}])[0] or {}
        token_items = _iter_token_logprobs(choice.get("logprobs"))
        logprobs_source = "choice.logprobs"
        if not token_items:
            meta_info = raw.get("meta_info") or choice.get("meta_info") or {}
            output_token_logprobs = meta_info.get("output_token_logprobs")
            output_top_logprobs = meta_info.get("output_top_logprobs")
            if output_token_logprobs:
                token_items = []
                for index, item in enumerate(output_token_logprobs):
                    entry: dict[str, Any] = {}
                    if isinstance(item, (list, tuple)):
                        if len(item) >= 1:
                            entry["logprob"] = item[0]
                        if len(item) >= 3:
                            entry["token"] = item[2]
                    elif isinstance(item, dict):
                        entry.update(item)
                    if output_top_logprobs and index < len(output_top_logprobs):
                        entry["top_logprobs"] = output_top_logprobs[index]
                    token_items.append(entry)
                logprobs_source = "meta_info.output_token_logprobs"

        token_logprobs: list[float] = []
        top1_probs: list[float] = []
        entropies: list[float] = []
        margins: list[float] = []
        tool_name_logprobs: list[float] = []
        argument_logprobs: list[float] = []
        rendered = ""
        argument_region = False

        for item in token_items:
            value = item.get("logprob")
            token = str(item.get("token") or "")
            rendered += token
            if "arguments" in rendered or '"arguments"' in rendered:
                argument_region = True
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                logprob = float(value)
                token_logprobs.append(logprob)
                if argument_region:
                    argument_logprobs.append(logprob)
                elif "name" in rendered or '"name"' in rendered:
                    tool_name_logprobs.append(logprob)

            top = _token_top_logprobs(item)
            if top:
                top_values = sorted(top.values(), reverse=True)
                if top_values:
                    top1_probs.append(math.exp(top_values[0]))
                entropy = _entropy_from_log_probs(top)
                if entropy is not None:
                    entropies.append(float(entropy))
                margin = _top1_top2_margin(top)
                if margin is not None:
                    margins.append(float(margin))

        nll = None
        if token_logprobs:
            nll = -sum(token_logprobs) / len(token_logprobs)
        ppl = math.exp(min(nll, 50.0)) if nll is not None else None

        readout = self._readout_payload(raw)
        readout_vector = readout.get("vector")
        readout_norm = None
        readout_prev_cosine_distance = None
        if readout_vector:
            readout_norm = math.sqrt(
                sum(float(value) * float(value) for value in readout_vector)
            )
            readout_prev_cosine_distance = _cosine_distance(
                previous_readout_vector,
                readout_vector,
            )

        features = {
            "detector_signal_requested": True,
            "detector_signal_available": bool(token_logprobs),
            "logprobs_source": logprobs_source if token_items else None,
            "generation_token_count": len(token_logprobs),
            "generation_nll": nll,
            "generation_ppl": ppl,
            "mean_top1_probability": _mean(top1_probs),
            "min_top1_probability": min(top1_probs) if top1_probs else None,
            "mean_logprob": _mean(token_logprobs),
            "min_logprob": min(token_logprobs) if token_logprobs else None,
            "mean_entropy": _mean(entropies),
            "max_entropy": max(entropies) if entropies else None,
            "mean_top1_top2_margin": _mean(margins),
            "min_top1_top2_margin": min(margins) if margins else None,
            "tool_name_generation_nll": (
                -_mean(tool_name_logprobs) if tool_name_logprobs else None
            ),
            "argument_generation_nll": (
                -_mean(argument_logprobs) if argument_logprobs else None
            ),
            "readout_available": bool(readout_vector),
            "readout_norm": readout_norm,
            "readout_prev_cosine_distance": readout_prev_cosine_distance,
            "attention_available": False,
            "attention_entropy": None,
            "current_query_attention_mass": None,
            "recent_observation_attention_mass": None,
            "older_history_attention_mass": None,
        }
        return features, readout_vector if readout_vector else previous_readout_vector

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
        test_entry_id: str,
        checkpoint_id: int,
        global_step: int,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        involved_instances: dict[str, Any],
        current_turn_response: list[str],
        current_turn_inputs: list[int],
        current_turn_outputs: list[int],
        current_turn_latency: list[float],
        turn_log: dict[str, Any],
        parent_kv_checkpoint_id: str | None = None,
        create_kv_checkpoint: bool = False,
        kv_anchor_checkpoint_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kv_checkpoint_metadata = None
        if create_kv_checkpoint:
            kv_checkpoint_metadata = self._kv_checkpoint_metadata(
                test_entry_id=test_entry_id,
                messages=messages,
                tools=tools,
                global_step=global_step,
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_kv_checkpoint_id,
            )
        return {
            "checkpoint_id": checkpoint_id,
            "global_step": global_step,
            "messages": deepcopy(messages),
            "instances": deepcopy(involved_instances),
            "state": _state_log(involved_instances),
            "kv_checkpoint_metadata": kv_checkpoint_metadata,
            "kv_anchor_checkpoint_metadata": deepcopy(kv_anchor_checkpoint_metadata),
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
            "oracle_harmful": harmful,
            "detector_trigger": harmful,
            "detector": "oracle",
            "detector_reason": "oracle_harmful" if harmful else "oracle_safe",
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
            "schema_version": 3,
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
            "recovery_horizon": self.recovery_horizon,
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
            "kv_memory_report": step_record.get("kv_memory_report"),
            "kv_runtime_stats": step_record.get("kv_runtime_stats"),
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
        record = build_step_record(
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
        usage_source = extra.get("executed_usage")
        if not isinstance(usage_source, dict):
            usage_source = spec_info.get("candidate_usage") or {}
        if usage_source.get("kv_memory_report") is not None:
            record["kv_memory_report"] = usage_source.get("kv_memory_report")
        if usage_source.get("kv_runtime_stats") is not None:
            record["kv_runtime_stats"] = usage_source.get("kv_runtime_stats")
        return record

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
        expected_kv_reused_tokens_total = 0
        expected_kv_recomputed_tokens_total = 0
        actual_cache_report_missing_total = 0
        message_replay_prefill_tokens_total = 0
        rollback_latency_total = 0.0
        restore_latency_total = 0.0
        regenerated_steps_total = 0
        full_regenerated_tokens_total = 0
        checkpoint_id = 0
        physical_checkpoint_id = 0
        active_full_checkpoint_metadata: dict[str, Any] | None = None
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
                segment_checkpoint_maintenance = (
                    self._empty_checkpoint_maintenance()
                )
                segment_initial_full_checkpoint_metadata = None
                segment_active_full_checkpoint_before = (
                    active_full_checkpoint_metadata.get("sglang_checkpoint_id")
                    if active_full_checkpoint_metadata
                    else None
                )
                if (
                    self.enable_recovery_kv_checkpoint
                    and active_full_checkpoint_metadata is None
                ):
                    physical_checkpoint_id += 1
                    initial_metadata = self._kv_checkpoint_metadata(
                        test_entry_id=test_entry_id,
                        messages=messages,
                        tools=tools,
                        global_step=segment_start_step,
                        checkpoint_id=physical_checkpoint_id,
                        parent_checkpoint_id=None,
                    )
                    self._assert_kv_checkpoint_available(
                        initial_metadata,
                        context="initial checkpoint create",
                    )
                    segment_initial_full_checkpoint_metadata = initial_metadata
                    self._add_checkpoint_maintenance(
                        segment_checkpoint_maintenance,
                        segment_initial_full_checkpoint_metadata,
                    )
                    if initial_metadata.get("available"):
                        active_full_checkpoint_metadata = initial_metadata
                        segment_active_full_checkpoint_before = (
                            active_full_checkpoint_metadata.get("sglang_checkpoint_id")
                        )
                    else:
                        active_full_checkpoint_metadata = None
                        segment_active_full_checkpoint_before = None
                segment_checkpoint = self._snapshot(
                    test_entry_id=test_entry_id,
                    checkpoint_id=checkpoint_id,
                    global_step=segment_start_step,
                    messages=messages,
                    tools=tools,
                    involved_instances=involved_instances,
                    current_turn_response=current_turn_response,
                    current_turn_inputs=current_turn_inputs,
                    current_turn_outputs=current_turn_outputs,
                    current_turn_latency=current_turn_latency,
                    turn_log=turn_log,
                    create_kv_checkpoint=False,
                    kv_anchor_checkpoint_metadata=active_full_checkpoint_metadata,
                )
                segment_infos: list[dict[str, Any]] = []
                terminal_after_segment = False
                previous_candidate_readout_vector: list[float] | None = None

                for segment_index in range(self.checkpoint_interval):
                    micro_snapshot = self._snapshot(
                        test_entry_id=test_entry_id,
                        checkpoint_id=checkpoint_id,
                        global_step=segment_start_step + len(segment_infos),
                        messages=messages,
                        tools=tools,
                        involved_instances=involved_instances,
                        current_turn_response=current_turn_response,
                        current_turn_inputs=current_turn_inputs,
                        current_turn_outputs=current_turn_outputs,
                        current_turn_latency=current_turn_latency,
                        turn_log=turn_log,
                        create_kv_checkpoint=False,
                        kv_anchor_checkpoint_metadata=active_full_checkpoint_metadata,
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
                        collect_detector_signals=(
                            self.collect_candidate_detector_signals
                        ),
                    )
                    candidate_detector_features = {}
                    if self.collect_candidate_detector_signals:
                        (
                            candidate_detector_features,
                            previous_candidate_readout_vector,
                        ) = self._candidate_detector_features(
                            candidate_raw,
                            previous_candidate_readout_vector,
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
                            "candidate_detector_features": (
                                candidate_detector_features
                            ),
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
                    per_step_detector_failed, per_step_detector_reason = (
                        self._heuristic_detector_failed(
                            segment_infos[-1]["heuristic_attributes"]
                        )
                    )
                    segment_infos[-1]["per_step_detector_trigger"] = (
                        per_step_detector_failed
                    )
                    segment_infos[-1]["per_step_detector_reason"] = (
                        per_step_detector_reason
                    )
                    segment_infos[-1]["verify"]["oracle_harmful"] = (
                        self._oracle_harmful(segment_infos[-1])
                    )
                    segment_infos[-1]["verify"]["per_step_detector_trigger"] = (
                        per_step_detector_failed
                    )
                    segment_infos[-1]["verify"]["per_step_detector_reason"] = (
                        per_step_detector_reason
                    )
                    segment_infos[-1]["verify"]["detector_risk_score"] = (
                        segment_infos[-1]["heuristic_attributes"].get("risk_score")
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
                oracle_harmful_per_step = [
                    self._oracle_harmful(info) for info in segment_infos
                ]
                oracle_segment_unsafe = any(oracle_harmful_per_step)
                if self.verifier == "oracle":
                    detector_debug = {
                        "detector": "oracle",
                        "detector_trigger": oracle_segment_unsafe,
                        "detector_reason": (
                            "oracle_segment_unsafe"
                            if oracle_segment_unsafe
                            else "oracle_segment_safe"
                        ),
                        "rule_detector_trigger": None,
                        "rule_detector_max_risk": None,
                        "rule_detector_reason": None,
                    }
                elif self.verifier in {"rule", "heuristic"}:
                    detector_debug = self._rule_detector(segment_infos)
                elif self.verifier == "logistic":
                    detector_debug = self._logistic_detector(segment_infos)
                elif self.verifier == "feature_signal":
                    detector_debug = self._feature_signal_detector(segment_infos)
                else:
                    detector_debug = {
                        "detector": self.verifier,
                        "detector_trigger": any(
                            bool(info["verify"].get("detector_trigger"))
                            for info in segment_infos
                        ),
                        "detector_reason": self.verifier,
                        "rule_detector_trigger": None,
                        "rule_detector_max_risk": None,
                        "rule_detector_reason": None,
                    }
                segment_has_drift = bool(detector_debug.get("detector_trigger"))
                if self.verifier == "oracle":
                    detector_trigger_per_step = list(oracle_harmful_per_step)
                else:
                    detector_trigger_per_step = [
                        bool(info.get("per_step_detector_trigger"))
                        for info in segment_infos
                    ]
                detector_confusion = self._detector_confusion(
                    oracle_segment_unsafe=oracle_segment_unsafe,
                    detector_segment_trigger=segment_has_drift,
                )
                for info in segment_infos:
                    info["verify"]["detector"] = detector_debug.get("detector")
                    info["verify"]["detector_trigger"] = segment_has_drift
                    info["verify"]["detector_reason"] = detector_debug.get(
                        "detector_reason"
                    ) or detector_debug.get("rule_detector_reason")
                speculative_end_state = _state_log(involved_instances)
                speculative_terminal_after_segment = terminal_after_segment
                speculative_force_quit = force_quit
                regenerated_infos: list[dict[str, Any]] = []
                final_records: list[dict[str, Any]] = []
                segment_recovery_tokens = 0
                rollback_steps = 0
                configured_rollback_depth = (
                    self.rollback_depth if self.rollback_policy == "fixed_depth" else None
                )
                actual_rollback_depth = 0
                rule_depth_debug: dict[str, Any] = {}
                rollback_start_index: int | None = None
                rollback_restore_policy = "none"
                first_bad_index: int | None = None
                detector_first_bad_index: int | None = None
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
                    "expected_kv_reused_tokens": 0,
                    "expected_kv_recomputed_tokens": 0,
                    "actual_cache_report_missing": 0,
                    "message_replay_prefill_tokens": 0,
                    "recovery_logical_prompt_tokens": 0,
                    "restored_checkpoint_tokens": 0,
                    "restore_loaded_from_host_tokens": 0,
                    "restore_already_device_tokens": 0,
                    "safe_delta_prefill_tokens": 0,
                    "safe_delta_reused_tokens": 0,
                    "safe_delta_logical_prompt_tokens": 0,
                    "restore_latency_sec": 0.0,
                    "rollback_latency_sec": 0.0,
                    "logical_snapshot_has_kv_checkpoint": False,
                    "kv_checkpoint_metadata": active_full_checkpoint_metadata,
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
                    detector_first_bad_index = next(
                        (
                            index
                            for index, triggered in enumerate(
                                detector_trigger_per_step
                            )
                            if triggered
                        ),
                        None,
                    )
                    first_bad_index = next(
                        (
                            index
                            for index, harmful in enumerate(oracle_harmful_per_step)
                            if harmful
                        ),
                        None,
                    )
                    if self.rollback_policy == "fixed_depth":
                        actual_depth = min(
                            max(self.rollback_depth, 1),
                            len(segment_infos),
                        )
                        rollback_start_index = len(segment_infos) - actual_depth
                        actual_rollback_depth = actual_depth
                        predicted_first_bad_index = rollback_start_index
                        attribution_debug = self._rollback_debug(
                            segment_infos=segment_infos,
                            rollback_start_index=rollback_start_index,
                            oracle_first_bad_index=first_bad_index,
                            reason=f"fixed_depth:{actual_depth}",
                            raw_predicted_first_bad_index=None,
                        )
                    elif self.rollback_policy == "rule_depth":
                        rule_depth_debug = self._rule_depth_decision(segment_infos)
                        rollback_start_index = int(
                            rule_depth_debug["rollback_start_index"]
                        )
                        actual_rollback_depth = int(
                            rule_depth_debug["actual_rollback_depth"]
                        )
                        predicted_first_bad_index = rollback_start_index
                        attribution_debug = self._rollback_debug(
                            segment_infos=segment_infos,
                            rollback_start_index=rollback_start_index,
                            oracle_first_bad_index=first_bad_index,
                            reason=rule_depth_debug["rule_depth_reason"],
                            raw_predicted_first_bad_index=rollback_start_index,
                        )
                        attribution_debug.update(rule_depth_debug)
                    elif (
                        self.rollback_policy == "whole_segment"
                        or self.attribution == "whole_segment"
                        or self.recovery_horizon == "whole_segment"
                    ):
                        rollback_start_index = 0
                        actual_rollback_depth = len(segment_infos)
                        predicted_first_bad_index = 0
                        attribution_debug = self._rollback_debug(
                            segment_infos=segment_infos,
                            rollback_start_index=rollback_start_index,
                            oracle_first_bad_index=first_bad_index,
                            reason="whole_segment",
                            raw_predicted_first_bad_index=0,
                        )
                    else:
                        predicted_first_bad_index, attribution_debug = (
                            self._predict_first_bad_index(
                                segment_infos=segment_infos,
                                oracle_first_bad_index=first_bad_index,
                            )
                        )
                        rollback_start_index = predicted_first_bad_index
                        actual_rollback_depth = len(segment_infos) - rollback_start_index

                    if rollback_start_index == 0:
                        restore_target = segment_checkpoint
                        rollback_restore_policy = "segment_checkpoint"
                        rollback_steps = len(segment_infos)
                        target_infos = segment_infos
                    else:
                        kept_infos = segment_infos[:rollback_start_index]
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
                        restore_target = segment_infos[rollback_start_index][
                            "micro_snapshot"
                        ]
                        rollback_restore_policy = "first_bad_micro_checkpoint"
                        rollback_steps = len(segment_infos) - rollback_start_index
                        if (
                            self.rollback_policy != "fixed_depth"
                            and self.recovery_horizon == "one_step"
                        ):
                            target_infos = [segment_infos[rollback_start_index]]
                        else:
                            target_infos = segment_infos[rollback_start_index:]
                    actual_rollback_depth = rollback_steps

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
                        tools=tools,
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
                            recovery_raw,
                        ) = self._query_with_raw(
                            recovery_messages,
                            tools,
                            stats,
                            readout_probe=False,
                        )
                        recovery_prompt_tokens = int(
                            recovery_usage.get("prompt_tokens") or recovery_local_tokens
                        )
                        self._account_recovery_prompt_work(
                            rollback_backend_info,
                            recovery_prompt_tokens,
                            recovery_usage.get("cached_tokens"),
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
                                "recovery_horizon": self.recovery_horizon,
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
                                "executed_usage": recovery_usage,
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
                    expected_kv_reused_tokens_total += int(
                        rollback_backend_info.get("expected_kv_reused_tokens") or 0
                    )
                    expected_kv_recomputed_tokens_total += int(
                        rollback_backend_info.get("expected_kv_recomputed_tokens")
                        or 0
                    )
                    actual_cache_report_missing_total += int(
                        rollback_backend_info.get("actual_cache_report_missing") or 0
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
                            "rollback_policy": self.rollback_policy,
                            "configured_rollback_depth": configured_rollback_depth,
                            "actual_rollback_depth": actual_rollback_depth,
                            "rollback_start_index": rollback_start_index,
                            "rule_depth_reason": attribution_debug.get(
                                "rule_depth_reason"
                            ),
                            "earliest_suspicious_index": attribution_debug.get(
                                "earliest_suspicious_index"
                            ),
                            "max_risk_index": attribution_debug.get(
                                "max_risk_index"
                            ),
                            "attribution": self.attribution,
                            "attribution_safety_margin": self.attribution_safety_margin,
                            "oracle_harmful": self._oracle_harmful(spec_info),
                            "detector": detector_debug.get("detector"),
                            "detector_trigger": segment_has_drift,
                            "detector_reason": (
                                detector_debug.get("detector_reason")
                                or detector_debug.get("rule_detector_reason")
                            ),
                            "logistic_detector_score": detector_debug.get(
                                "logistic_detector_score"
                            ),
                            "logistic_detector_threshold": detector_debug.get(
                                "logistic_detector_threshold"
                            ),
                            "detector_signal_name": detector_debug.get(
                                "detector_signal_name"
                            ),
                            "detector_signal_score": detector_debug.get(
                                "detector_signal_score"
                            ),
                            "detector_signal_threshold": detector_debug.get(
                                "detector_signal_threshold"
                            ),
                            "logistic_detector_feature_count": detector_debug.get(
                                "logistic_detector_feature_count"
                            ),
                            "logistic_detector_train_rows": detector_debug.get(
                                "logistic_detector_train_rows"
                            ),
                            "logistic_detector_train_episodes": detector_debug.get(
                                "logistic_detector_train_episodes"
                            ),
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
                            "candidate_detector_features": spec_info.get(
                                "candidate_detector_features"
                            ),
                            "generation_nll": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("generation_nll"),
                            "detector_signal_requested": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("detector_signal_requested"),
                            "detector_signal_available": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("detector_signal_available"),
                            "logprobs_source": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("logprobs_source"),
                            "generation_ppl": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("generation_ppl"),
                            "mean_top1_probability": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("mean_top1_probability"),
                            "min_top1_probability": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("min_top1_probability"),
                            "mean_logprob": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("mean_logprob"),
                            "min_logprob": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("min_logprob"),
                            "mean_entropy": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("mean_entropy"),
                            "max_entropy": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("max_entropy"),
                            "mean_top1_top2_margin": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("mean_top1_top2_margin"),
                            "min_top1_top2_margin": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("min_top1_top2_margin"),
                            "readout_norm": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("readout_norm"),
                            "readout_prev_cosine_distance": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("readout_prev_cosine_distance"),
                            "attention_available": (
                                spec_info.get("candidate_detector_features") or {}
                            ).get("attention_available"),
                        }
                    )

                chain_advance_metadata = None
                chain_restore_response = None
                chain_restore_success = None
                chain_restore_fallback = False
                chain_restore_fallback_reason = None
                chain_release_previous = False
                chain_advance_error = None
                previous_active_checkpoint_metadata = active_full_checkpoint_metadata
                if self.enable_recovery_kv_checkpoint:
                    parent_checkpoint_id = (
                        previous_active_checkpoint_metadata.get(
                            "sglang_checkpoint_id"
                        )
                        if previous_active_checkpoint_metadata
                        and previous_active_checkpoint_metadata.get("available")
                        else None
                    )
                    if previous_active_checkpoint_metadata:
                        restore_started = time.perf_counter()
                        chain_restore_response = self._restore_kv_checkpoint_metadata(
                            previous_active_checkpoint_metadata
                        )
                        chain_restore_success = bool(
                            chain_restore_response.get("success")
                        )
                        chain_restore_response["elapsed_sec"] = (
                            time.perf_counter() - restore_started
                        )
                        if not chain_restore_success:
                            chain_restore_fallback = True
                            chain_restore_fallback_reason = (
                                chain_restore_response.get("fallback_reason")
                                or chain_restore_response.get("message")
                                or chain_restore_response.get("error")
                                or "CHAIN_RESTORE_FAILED"
                            )
                            if self.kv_restore_strict:
                                raise RuntimeError(
                                    "kv_restore_strict chain advance restore failed: "
                                    f"{chain_restore_fallback_reason}; "
                                    "refusing hidden full recompute"
                                )
                    try:
                        physical_checkpoint_id += 1
                        chain_advance_metadata = self._kv_checkpoint_metadata(
                            test_entry_id=test_entry_id,
                            messages=messages,
                            tools=tools,
                            global_step=len(drift_steps),
                            checkpoint_id=physical_checkpoint_id,
                            parent_checkpoint_id=parent_checkpoint_id,
                        )
                        self._add_checkpoint_maintenance(
                            segment_checkpoint_maintenance,
                            chain_advance_metadata,
                        )
                        self._assert_kv_checkpoint_available(
                            chain_advance_metadata,
                            context="chain advance checkpoint create",
                        )
                        if chain_advance_metadata.get("available"):
                            active_full_checkpoint_metadata = chain_advance_metadata
                            if previous_active_checkpoint_metadata:
                                self._release_kv_checkpoint_metadata(
                                    previous_active_checkpoint_metadata
                                )
                                chain_release_previous = True
                    except Exception as exc:
                        chain_advance_error = str(exc)
                        if self.kv_restore_strict:
                            raise

                regenerated_end_state = _state_log(involved_instances)
                if segment_has_drift:
                    assert rollback_start_index is not None
                    committed_until_index = rollback_start_index
                else:
                    committed_until_index = len(segment_infos)
                committed_speculative_tokens = sum(
                    int(info["candidate_usage"].get("prompt_tokens") or 0)
                    for info in segment_infos[:committed_until_index]
                )
                discarded_speculative_tokens = (
                    0
                    if not segment_has_drift
                    else sum(
                        int(info["candidate_usage"].get("prompt_tokens") or 0)
                        for info in segment_infos[rollback_start_index:]
                    )
                )
                segment_executed_drift_count = sum(
                    1
                    for record in final_records
                    if record.get("executed_action_drift")
                )
                state_drift_after_recovery = any(
                    bool(record.get("state_drift"))
                    for record in final_records
                )
                segment_recovery_success = bool(
                    segment_has_drift
                    and segment_executed_drift_count == 0
                    and not state_drift_after_recovery
                )
                segment_checkpoint_metadata = (
                    chain_advance_metadata
                    if chain_advance_metadata and chain_advance_metadata.get("available")
                    else segment_initial_full_checkpoint_metadata
                ) or {}
                segment_checkpoint_host_tokens = int(
                    segment_checkpoint_metadata.get("checkpoint_host_tokens") or 0
                )
                segment_checkpoint_device_tokens = int(
                    segment_checkpoint_metadata.get("checkpoint_device_tokens") or 0
                )
                checkpoint_segments.append(
                    {
                        "schema_version": 3,
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
                        "harmful_drift_per_step": oracle_harmful_per_step,
                        "oracle_harmful_drift_per_step": oracle_harmful_per_step,
                        "detector_trigger_per_step": detector_trigger_per_step,
                        "candidate_drift_per_step": oracle_harmful_per_step,
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
                            1 for harmful in oracle_harmful_per_step if harmful
                        ),
                        "segment_detector_trigger_count": sum(
                            1 for triggered in detector_trigger_per_step if triggered
                        ),
                        "segment_executed_drift_count": (
                            segment_executed_drift_count
                        ),
                        "segment_has_drift": segment_has_drift,
                        "oracle_segment_unsafe": oracle_segment_unsafe,
                        "oracle_reference_drift_segment": oracle_segment_unsafe,
                        "detector_trigger": segment_has_drift,
                        "detector": detector_debug.get("detector"),
                        "detector_reason": (
                            detector_debug.get("detector_reason")
                            or detector_debug.get("rule_detector_reason")
                        ),
                        "detector_tp": detector_confusion["detector_tp"],
                        "detector_fp": detector_confusion["detector_fp"],
                        "detector_tn": detector_confusion["detector_tn"],
                        "detector_fn": detector_confusion["detector_fn"],
                        "rule_detector_trigger": detector_debug.get(
                            "rule_detector_trigger"
                        ),
                        "rule_detector_max_risk": detector_debug.get(
                            "rule_detector_max_risk"
                        ),
                        "rule_detector_reason": detector_debug.get(
                            "rule_detector_reason"
                        ),
                        "rule_detector_threshold": detector_debug.get(
                            "rule_detector_threshold"
                        ),
                        "logistic_detector_score": detector_debug.get(
                            "logistic_detector_score"
                        ),
                        "logistic_detector_threshold": detector_debug.get(
                            "logistic_detector_threshold"
                        ),
                        "logistic_detector_feature_count": detector_debug.get(
                            "logistic_detector_feature_count"
                        ),
                        "detector_signal_name": detector_debug.get(
                            "detector_signal_name"
                        ),
                        "detector_signal_score": detector_debug.get(
                            "detector_signal_score"
                        ),
                        "detector_signal_threshold": detector_debug.get(
                            "detector_signal_threshold"
                        ),
                        "logistic_detector_train_rows": detector_debug.get(
                            "logistic_detector_train_rows"
                        ),
                        "logistic_detector_train_episodes": detector_debug.get(
                            "logistic_detector_train_episodes"
                        ),
                        "verify_triggered": True,
                        "rollback_triggered": segment_has_drift,
                        "refresh_triggered": segment_has_drift,
                        "segment_recovery_success": segment_recovery_success,
                        "reference_recovery_success": segment_recovery_success,
                        "checkpoint_state": segment_checkpoint.get("state"),
                        "speculative_end_state": speculative_end_state,
                        "restored_state": restored_state,
                        "regenerated_end_state": regenerated_end_state,
                        "rollback_state_matches_checkpoint": (
                            rollback_state_matches_checkpoint
                        ),
                        "first_bad_index": first_bad_index,
                        "oracle_first_bad_index": first_bad_index,
                        "detector_first_bad_index": detector_first_bad_index,
                        "predicted_first_bad_index": predicted_first_bad_index,
                        "raw_predicted_first_bad_index": attribution_debug.get(
                            "raw_predicted_first_bad_index"
                        ),
                        "attribution": self.attribution,
                        "recovery_horizon": self.recovery_horizon,
                        "attribution_reason": attribution_debug.get(
                            "attribution_reason"
                        ),
                        "attribution_safety_margin": self.attribution_safety_margin,
                        "has_oracle_first_bad": attribution_debug.get(
                            "has_oracle_first_bad"
                        ),
                        "detector_false_positive": attribution_debug.get(
                            "detector_false_positive"
                        ),
                        "exact_attribution": attribution_debug.get(
                            "exact_attribution"
                        ),
                        "within1_attribution": attribution_debug.get(
                            "within1_attribution"
                        ),
                        "under_rollback": attribution_debug.get("under_rollback"),
                        "over_rollback": attribution_debug.get("over_rollback"),
                        "rollback_coverage": attribution_debug.get(
                            "rollback_coverage"
                        ),
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
                        "configured_rollback_depth": configured_rollback_depth,
                        "actual_rollback_depth": actual_rollback_depth,
                        "rollback_start_index": rollback_start_index,
                        "rule_depth_reason": attribution_debug.get(
                            "rule_depth_reason"
                        ),
                        "earliest_suspicious_index": attribution_debug.get(
                            "earliest_suspicious_index"
                        ),
                        "max_risk_index": attribution_debug.get("max_risk_index"),
                        "strong_suspicious_count": attribution_debug.get(
                            "strong_suspicious_count"
                        ),
                        "strong_suspicious_indices": attribution_debug.get(
                            "strong_suspicious_indices"
                        ),
                        "rollback_steps": rollback_steps,
                        "rollback_policy": self.rollback_policy,
                        "rollback_restore_policy": (
                            "none" if not segment_has_drift else rollback_restore_policy
                        ),
                        "regen_policy": (
                            "none"
                            if not segment_has_drift
                            else (
                                "one_step"
                                if self.recovery_horizon == "one_step"
                                else self.recovery_horizon
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
                        "expected_kv_reused_tokens": rollback_backend_info.get(
                            "expected_kv_reused_tokens"
                        ),
                        "expected_kv_recomputed_tokens": rollback_backend_info.get(
                            "expected_kv_recomputed_tokens"
                        ),
                        "actual_cache_report_missing": rollback_backend_info.get(
                            "actual_cache_report_missing"
                        ),
                        "message_replay_prefill_tokens": rollback_backend_info.get(
                            "message_replay_prefill_tokens"
                        ),
                        "recovery_logical_prompt_tokens": rollback_backend_info.get(
                            "recovery_logical_prompt_tokens"
                        ),
                        "restored_checkpoint_tokens": rollback_backend_info.get(
                            "restored_checkpoint_tokens"
                        ),
                        "restore_loaded_from_host_tokens": rollback_backend_info.get(
                            "restore_loaded_from_host_tokens"
                        ),
                        "restore_already_device_tokens": rollback_backend_info.get(
                            "restore_already_device_tokens"
                        ),
                        "safe_delta_prefill_tokens": rollback_backend_info.get(
                            "safe_delta_prefill_tokens"
                        ),
                        "safe_delta_reused_tokens": rollback_backend_info.get(
                            "safe_delta_reused_tokens"
                        ),
                        "safe_delta_logical_prompt_tokens": rollback_backend_info.get(
                            "safe_delta_logical_prompt_tokens"
                        ),
                        "safe_delta_expected_reused_tokens": (
                            rollback_backend_info.get(
                                "safe_delta_expected_reused_tokens"
                            )
                        ),
                        "safe_delta_expected_prefill_tokens": (
                            rollback_backend_info.get(
                                "safe_delta_expected_prefill_tokens"
                            )
                        ),
                        "logical_snapshot_has_kv_checkpoint": (
                            rollback_backend_info.get(
                                "logical_snapshot_has_kv_checkpoint"
                            )
                        ),
                        "kv_checkpoint_metadata": rollback_backend_info.get(
                            "kv_checkpoint_metadata"
                        ),
                        "active_full_checkpoint_before": (
                            segment_active_full_checkpoint_before
                        ),
                        "active_full_checkpoint_after": (
                            active_full_checkpoint_metadata.get(
                                "sglang_checkpoint_id"
                            )
                            if active_full_checkpoint_metadata
                            else None
                        ),
                        "initial_full_checkpoint_metadata": (
                            make_json_serializable(
                                segment_initial_full_checkpoint_metadata
                            )
                            if segment_initial_full_checkpoint_metadata
                            else None
                        ),
                        "full_checkpoint_chain_advanced": bool(
                            chain_advance_metadata
                            and chain_advance_metadata.get("available")
                        ),
                        "full_checkpoint_chain_restore_success": (
                            chain_restore_success
                        ),
                        "full_checkpoint_chain_restore_fallback": (
                            chain_restore_fallback
                        ),
                        "full_checkpoint_chain_restore_fallback_reason": (
                            chain_restore_fallback_reason
                        ),
                        "full_checkpoint_chain_restore_response": (
                            make_json_serializable(chain_restore_response)
                        ),
                        "full_checkpoint_chain_release_previous": (
                            chain_release_previous
                        ),
                        "full_checkpoint_chain_error": chain_advance_error,
                        "full_checkpoint_chain_metadata": (
                            make_json_serializable(chain_advance_metadata)
                        ),
                        "checkpoint_host_tokens": segment_checkpoint_host_tokens,
                        "checkpoint_device_tokens": segment_checkpoint_device_tokens,
                        "checkpoint_maintenance_recomputed_tokens": (
                            segment_checkpoint_maintenance.get(
                                "checkpoint_maintenance_recomputed_tokens"
                            )
                        ),
                        "checkpoint_maintenance_reused_tokens": (
                            segment_checkpoint_maintenance.get(
                                "checkpoint_maintenance_reused_tokens"
                            )
                        ),
                        "checkpoint_maintenance_logical_prompt_tokens": (
                            segment_checkpoint_maintenance.get(
                                "checkpoint_maintenance_logical_prompt_tokens"
                            )
                        ),
                        "checkpoint_maintenance_cache_report_missing": (
                            segment_checkpoint_maintenance.get(
                                "checkpoint_maintenance_cache_report_missing"
                            )
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
                            else len(segment_infos) - int(rollback_start_index or 0)
                        ),
                        "heuristic_attributes_per_step": [
                            info.get("heuristic_attributes") for info in segment_infos
                        ],
                        "candidate_detector_features_per_step": [
                            info.get("candidate_detector_features")
                            for info in segment_infos
                        ],
                        "final_executed_actions": [
                            record.get("executed_action") for record in final_records
                        ],
                        "executed_drift_per_step": [
                            bool(record.get("executed_action_drift"))
                            for record in final_records
                        ],
                        "state_drift_after_recovery": state_drift_after_recovery,
                        "speculative_terminal_after_segment": (
                            speculative_terminal_after_segment
                        ),
                        "speculative_force_quit": speculative_force_quit,
                        "terminal_after_segment": terminal_after_segment,
                        "recovery_mode": self.recovery_mode,
                        "recovery_horizon": self.recovery_horizon,
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

        self._release_kv_checkpoint_metadata(active_full_checkpoint_metadata)

        checkpoint_maintenance_reused_tokens_total = sum(
            int(row.get("checkpoint_maintenance_reused_tokens") or 0)
            for row in checkpoint_segments
        )
        checkpoint_maintenance_recomputed_tokens_total = sum(
            int(row.get("checkpoint_maintenance_recomputed_tokens") or 0)
            for row in checkpoint_segments
        )
        checkpoint_maintenance_logical_prompt_tokens_total = sum(
            int(row.get("checkpoint_maintenance_logical_prompt_tokens") or 0)
            for row in checkpoint_segments
        )
        checkpoint_maintenance_cache_report_missing_total = sum(
            int(row.get("checkpoint_maintenance_cache_report_missing") or 0)
            for row in checkpoint_segments
        )
        checkpoint_host_tokens_total = sum(
            int(row.get("checkpoint_host_tokens") or 0)
            for row in checkpoint_segments
        )
        checkpoint_device_tokens_total = sum(
            int(row.get("checkpoint_device_tokens") or 0)
            for row in checkpoint_segments
        )
        peak_checkpoint_host_tokens = max(
            (int(row.get("checkpoint_host_tokens") or 0) for row in checkpoint_segments),
            default=0,
        )

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
            "expected_kv_reused_tokens": expected_kv_reused_tokens_total,
            "expected_kv_recomputed_tokens": expected_kv_recomputed_tokens_total,
            "actual_cache_report_missing": actual_cache_report_missing_total,
            "message_replay_prefill_tokens": message_replay_prefill_tokens_total,
            "checkpoint_maintenance_reused_tokens": (
                checkpoint_maintenance_reused_tokens_total
            ),
            "checkpoint_maintenance_recomputed_tokens": (
                checkpoint_maintenance_recomputed_tokens_total
            ),
            "checkpoint_maintenance_logical_prompt_tokens": (
                checkpoint_maintenance_logical_prompt_tokens_total
            ),
            "checkpoint_maintenance_cache_report_missing": (
                checkpoint_maintenance_cache_report_missing_total
            ),
            "checkpoint_host_tokens": checkpoint_host_tokens_total,
            "checkpoint_device_tokens": checkpoint_device_tokens_total,
            "peak_checkpoint_host_tokens": peak_checkpoint_host_tokens,
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
            if self.kv_restore_strict:
                raise
            result = f"Error during inference: {exc}"
            metadata = {
                "traceback": traceback.format_exc(),
                "inference_error": True,
            }
            stats.errors.append(str(exc))
        metadata["c2kv_checkpoint_metrics"] = {
            **stats.as_dict(),
            "checkpoint_interval": self.checkpoint_interval,
            "verifier": self.verifier,
            "requested_verifier": self.requested_verifier,
            "attribution": self.attribution,
            "attribution_safety_margin": self.attribution_safety_margin,
        "rollback_policy": self.rollback_policy,
        "rollback_depth": self.rollback_depth,
        "rule_detector_threshold": self.rule_detector_threshold,
        "logistic_detector_features_csv": self.logistic_detector_features_csv,
        "logistic_detector_threshold": (
            self.logistic_detector_model.get("threshold")
            if self.logistic_detector_model
            else self.logistic_detector_threshold
        ),
        "logistic_detector_feature_count": (
            len(self.logistic_detector_model.get("features") or [])
            if self.logistic_detector_model
            else 0
        ),
        "rollback_backend": self.rollback_backend,
            "verify_threshold": self.verify_threshold,
            "verify_layers": self.verify_layers,
            "online_verify": self.online_verify,
            "reuse_candidate_readout": self.reuse_candidate_readout,
            "recovery_mode": self.recovery_mode,
            "recovery_horizon": self.recovery_horizon,
            "verify_count": metadata.get("verify_count", 0),
            "refresh_count": metadata.get("refresh_count", 0),
            "regenerated_steps": metadata.get("regenerated_steps", 0),
            "full_regenerated_tokens": metadata.get("full_regenerated_tokens", 0),
            "kv_restore_success": metadata.get("kv_restore_success", 0),
            "kv_restore_fallback": metadata.get("kv_restore_fallback", 0),
            "kv_reused_tokens": metadata.get("kv_reused_tokens", 0),
            "kv_recomputed_tokens": metadata.get("kv_recomputed_tokens", 0),
            "expected_kv_reused_tokens": metadata.get(
                "expected_kv_reused_tokens",
                0,
            ),
            "expected_kv_recomputed_tokens": metadata.get(
                "expected_kv_recomputed_tokens",
                0,
            ),
            "actual_cache_report_missing": metadata.get(
                "actual_cache_report_missing",
                0,
            ),
            "checkpoint_host_tokens": metadata.get("checkpoint_host_tokens", 0),
            "checkpoint_device_tokens": metadata.get("checkpoint_device_tokens", 0),
            "peak_checkpoint_host_tokens": metadata.get(
                "peak_checkpoint_host_tokens",
                0,
            ),
            "message_replay_prefill_tokens": metadata.get(
                "message_replay_prefill_tokens",
                0,
            ),
            "checkpoint_maintenance_reused_tokens": metadata.get(
                "checkpoint_maintenance_reused_tokens",
                0,
            ),
            "checkpoint_maintenance_recomputed_tokens": metadata.get(
                "checkpoint_maintenance_recomputed_tokens",
                0,
            ),
            "checkpoint_maintenance_logical_prompt_tokens": metadata.get(
                "checkpoint_maintenance_logical_prompt_tokens",
                0,
            ),
            "checkpoint_maintenance_cache_report_missing": metadata.get(
                "checkpoint_maintenance_cache_report_missing",
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
        if self.verifier in {
            "oracle",
            "heuristic",
            "rule",
            "logistic",
            "feature_signal",
        }:
            return self._run_sample_checkpoint_impl_oracle_multistep(
                test_case,
                stats,
            )
        if self.checkpoint_interval != 1:
            raise NotImplementedError(
                "KV/readout checkpoint verification currently supports "
                "checkpoint_interval=1 only. Run verifier=oracle/heuristic/rule/"
                "logistic/feature_signal for true multi-step rollback with "
                "interval=1/2/3/4."
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
                            "recovery_horizon": self.recovery_horizon,
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
                if executed_usage.get("kv_memory_report") is not None:
                    step_record["kv_memory_report"] = executed_usage.get("kv_memory_report")
                if executed_usage.get("kv_runtime_stats") is not None:
                    step_record["kv_runtime_stats"] = executed_usage.get("kv_runtime_stats")
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
                        "recovery_horizon": self.recovery_horizon,
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
                        "kv_memory_report": executed_usage.get("kv_memory_report"),
                        "kv_runtime_stats": executed_usage.get("kv_runtime_stats"),
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
        "rollback_policy": args.rollback_policy,
        "rollback_depth": args.rollback_depth,
        "rule_detector_threshold": args.rule_detector_threshold,
        "detector_signal_name": args.detector_signal_name,
        "detector_signal_threshold": args.detector_signal_threshold,
        "rollback_backend": args.rollback_backend,
        "verify_threshold": args.verify_threshold,
        "verify_layers": args.verify_layers,
        "online_verify": args.online_verify,
        "reuse_candidate_readout": args.reuse_candidate_readout,
        "recovery_mode": args.recovery_mode,
        "recovery_horizon": runner.recovery_horizon,
        "verify_count": sum(int(row.get("verify_count") or 0) for row in metrics_rows),
        "refresh_count": sum(int(row.get("refresh_count") or 0) for row in metrics_rows),
        "regenerated_steps": sum(int(row.get("regenerated_steps") or 0) for row in metrics_rows),
        "full_regenerated_tokens": sum(int(row.get("full_regenerated_tokens") or 0) for row in metrics_rows),
        "kv_restore_success": sum(int(row.get("kv_restore_success") or 0) for row in metrics_rows),
        "kv_restore_fallback": sum(int(row.get("kv_restore_fallback") or 0) for row in metrics_rows),
        "kv_reused_tokens": sum(int(row.get("kv_reused_tokens") or 0) for row in metrics_rows),
        "kv_recomputed_tokens": sum(int(row.get("kv_recomputed_tokens") or 0) for row in metrics_rows),
        "expected_kv_reused_tokens": sum(int(row.get("expected_kv_reused_tokens") or 0) for row in metrics_rows),
        "expected_kv_recomputed_tokens": sum(int(row.get("expected_kv_recomputed_tokens") or 0) for row in metrics_rows),
        "actual_cache_report_missing": sum(int(row.get("actual_cache_report_missing") or 0) for row in metrics_rows),
        "message_replay_prefill_tokens": sum(int(row.get("message_replay_prefill_tokens") or 0) for row in metrics_rows),
        "checkpoint_maintenance_reused_tokens": sum(
            int(row.get("checkpoint_maintenance_reused_tokens") or 0)
            for row in metrics_rows
        ),
        "checkpoint_maintenance_recomputed_tokens": sum(
            int(row.get("checkpoint_maintenance_recomputed_tokens") or 0)
            for row in metrics_rows
        ),
        "checkpoint_maintenance_logical_prompt_tokens": sum(
            int(row.get("checkpoint_maintenance_logical_prompt_tokens") or 0)
            for row in metrics_rows
        ),
        "checkpoint_maintenance_cache_report_missing": sum(
            int(row.get("checkpoint_maintenance_cache_report_missing") or 0)
            for row in metrics_rows
        ),
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
    )
    parser.add_argument(
        "--verifier",
        choices=[
            "instant_kv",
            "cumulative_kv",
            "kv_divergence",
            "oracle",
            "heuristic",
            "rule",
            "logistic",
            "feature_signal",
        ],
        default="oracle",
    )
    parser.add_argument("--verify-threshold", type=float, default=0.0)
    parser.add_argument("--rule-detector-threshold", type=float, default=5.0)
    parser.add_argument(
        "--logistic-detector-features-csv",
        default="",
        help=(
            "detector_features.csv used to train verifier=logistic from "
            "calibration split only."
        ),
    )
    parser.add_argument(
        "--logistic-detector-threshold",
        type=float,
        default=-1.0,
        help=(
            "Logistic detector threshold. Negative means choose best-F1 "
            "threshold on the calibration split."
        ),
    )
    parser.add_argument(
        "--detector-signal-name",
        default="",
        help=(
            "Single aggregated detector feature to use with "
            "verifier=feature_signal, e.g. max_risk_score."
        ),
    )
    parser.add_argument(
        "--detector-signal-threshold",
        type=float,
        default=0.0,
        help=(
            "Score threshold for verifier=feature_signal. Low-is-bad "
            "features are compared after the same sign flip used by the "
            "offline detector benchmark."
        ),
    )
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
        "--collect-candidate-detector-signals",
        action="store_true",
        help=(
            "Request logprobs on the normal C2KV candidate generation call and "
            "record cheap detector features for offline signal benchmarking."
        ),
    )
    parser.add_argument("--candidate-logprobs-top-k", type=int, default=20)
    parser.add_argument(
        "--candidate-hidden-readout",
        action="store_true",
        help="Also request hidden_states/readout on candidate calls if the server supports it.",
    )
    parser.add_argument(
        "--candidate-attention-summary",
        action="store_true",
        help="Reserved for server-side attention summaries; unavailable is logged today.",
    )
    parser.add_argument(
        "--recovery-mode",
        choices=sorted(CHECKPOINT_MODES),
        default="current_step",
    )
    parser.add_argument(
        "--recovery-horizon",
        choices=["auto", *sorted(RECOVERY_HORIZONS)],
        default="auto",
        help=(
            "Orthogonal rollback regeneration horizon. "
            "auto preserves legacy recovery-mode behavior."
        ),
    )
    parser.add_argument(
        "--attribution",
        choices=["auto", *sorted(ATTRIBUTION_MODES)],
        default="auto",
    )
    parser.add_argument("--attribution-safety-margin", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--rollback-policy",
        choices=sorted(ROLLBACK_POLICIES),
        default="attribution",
    )
    parser.add_argument("--rollback-depth", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument(
        "--rollback-backend",
        choices=sorted(ROLLBACK_BACKENDS),
        default="message_replay",
    )
    parser.add_argument(
        "--recovery-checkpoint-page-size",
        type=int,
        default=0,
        help=(
            "Deprecated no-op. Recovery checkpoint page alignment is decided "
            "by the SGLang cache owner."
        ),
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
