from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mt_utils
from bfcl_eval.utils import load_dataset_entry, make_json_serializable, sort_file_content_by_id, sort_key

from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    ExtractRecord,
    HistoryDriftRunner,
    MAXIMUM_STEP_LIMIT,
    _actual_cached_tokens_from_response,
    _assistant_history_message,
    _history_units,
    _kv_runtime_stats_from_response,
    _kv_memory_report_from_response,
    _latest_user_query_index,
    _message_text,
    _post_json,
    _render_history_unit,
    _state_log,
    _token_count,
    _tool_payload,
    _tool_calls_to_text,
    execute_multi_turn_func_call,
    is_empty_execute_response,
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
from c2kv_eval.adapters.eval_bfcl_history_checkpoint import (
    _as_float,
    _entropy_from_log_probs,
    _iter_token_logprobs,
    _mean,
    _token_top_logprobs,
    _top1_top2_margin,
)


REPAIR_ARMS = {
    "full",
    "c2kv",
    "d_sham_mech",
    "hint_only",
    "d_sham_neutral",
    "d_corr",
    "d_corr_w1",
    "d_corr_w2",
    "d_corr_w4",
    "d_corr_w2_hint",
    "d_corr_w2_oracle_location_hint",
    "d_corr_replace_w1",
    "d_corr_replace_w2",
    "d_corr_replace_w4",
    "d_corr_replace_all",
    "append_masked_w2",
    "d_corr_recompute",
    "d_corr_recompute_w2",
    "d_corr_all",
    "raw_all_replace",
    "raw_all_replace_direct",
}

DETECTOR_ARMS = {
    "oracle",
    "combined_logistic",
    "combined_logistic_best_f1",
    "combined_logistic_high_recall",
    "max_risk_score",
    "rule_detector_max_risk",
    "max_observation_anomaly",
    "mean_risk_score",
    "max_hard_error",
    "max_generation_nll",
    "mean_generation_nll",
    "rule_trigger",
    "always_trigger",
    "never_trigger",
}

LOGISTIC_LOGPROB_FEATURE_PATTERNS = (
    "generation_nll",
    "generation_ppl",
    "generation_token_count",
    "logprob",
    "top1_probability",
    "top1_top2_margin",
    "entropy",
    "detector_signal",
    "attention",
    "readout",
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


class KVRepairRunner(HistoryDriftRunner):
    """BFCL history runner for C2KV D-KV repair arms.

    This keeps BFCL tools/system/current turn full. Only completed history units
    are represented as full messages, C2KV gist messages, or gist+repair KV
    messages.
    """

    def __init__(self, args: argparse.Namespace):
        drift_args = deepcopy(args)
        drift_args.mode = (
            "history_full_closed_loop"
            if args.arm == "full"
            else "history_c2kv4_closed_loop"
        )
        super().__init__(drift_args)
        self.arm = args.arm
        self.repair_trigger = args.repair_trigger
        self.detector_arm = getattr(args, "detector_arm", "oracle")
        if self.detector_arm == "combined_logistic":
            self.detector_arm = "combined_logistic_best_f1"
        self.rule_detector_threshold = float(
            getattr(args, "rule_detector_threshold", 5.0)
        )
        self.detector_signal_threshold = float(
            getattr(args, "detector_signal_threshold", 5.0)
        )
        self.candidate_logprobs_top_k = int(
            getattr(args, "candidate_logprobs_top_k", 20)
        )
        self.request_candidate_logprobs = bool(
            getattr(args, "request_candidate_logprobs", False)
        )
        self.collect_candidate_detector_signals = bool(
            getattr(args, "collect_candidate_detector_signals", False)
            or self.detector_arm
            in {
                "combined_logistic_best_f1",
                "combined_logistic_high_recall",
            }
        )
        self.logistic_detector_kfolds = max(
            0,
            int(getattr(args, "logistic_detector_kfolds", 0) or 0),
        )
        self.logistic_detector_feature_set = str(
            getattr(args, "logistic_detector_feature_set", "auto") or "auto"
        )
        self.detector_thresholds = self._load_detector_thresholds(
            getattr(args, "detector_thresholds_json", "")
        )
        self.detector_cv_output_dir = str(
            getattr(args, "detector_cv_output_dir", "") or ""
        )
        self.logistic_detector_model = (
            self._train_logistic_detector(
                getattr(args, "logistic_detector_features_csv", ""),
                threshold_rule=(
                    "high_recall"
                    if self.detector_arm == "combined_logistic_high_recall"
                    else "best_f1"
                ),
            )
            if self.detector_arm
            in {
                "combined_logistic_best_f1",
                "combined_logistic_high_recall",
            }
            else None
        )
        self.checkpoint_interval = max(1, int(args.checkpoint_interval))
        self.require_plan = False
        self.plan = self._load_plan(args.plan_path)
        self.neutral_token_ids = self._load_neutral_tokens(args.neutral_corpus_path)
        self._active_tools: list[dict[str, Any]] = []
        self._repair_enabled_for_current_step = True
        self._repair_target_history_index: int | None = None
        self._repair_window_arg = args.repair_window
        self.repair_extract_source = getattr(args, "repair_extract_source", "auto")
        self._active_oracle_bad_step: int | None = None
        self._last_repair_build_info: dict[str, Any] = {}
        self._last_kv_memory_hint: dict[str, Any] | None = None
        self._previous_readout_vector: list[float] | None = None

    def _use_online_safe_logistic_features(self) -> bool:
        if self.logistic_detector_feature_set == "online_safe":
            return True
        if self.logistic_detector_feature_set == "all":
            return False
        return not self.request_candidate_logprobs

    @staticmethod
    def _episode_fold(sample_id: str, folds: int) -> int:
        import hashlib

        if folds <= 1:
            return 0
        digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % folds

    @staticmethod
    def _episode_bucket(sample_id: str, *, salt: str, buckets: int = 1000) -> int:
        import hashlib

        digest = hashlib.sha256(f"{sample_id}:{salt}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % buckets

    def _inner_train_calibration_split(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        outer_fold: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train: list[dict[str, Any]] = []
        calibration: list[dict[str, Any]] = []
        episode_ids = sorted({str(row.get("id")) for row in rows})
        calibration_ids = {
            sample_id
            for sample_id in episode_ids
            if self._episode_bucket(
                sample_id,
                salt=f"inner_calibration_fold_{outer_fold}",
            )
            < 200
        }
        if not calibration_ids and episode_ids:
            calibration_ids = set(episode_ids[-max(1, len(episode_ids) // 5) :])
        if len(calibration_ids) == len(episode_ids) and len(episode_ids) > 1:
            calibration_ids = set(episode_ids[-max(1, len(episode_ids) // 5) :])
        for row in rows:
            if str(row.get("id")) in calibration_ids:
                calibration.append(row)
            else:
                train.append(row)
        if not train:
            train = list(rows)
        if not calibration:
            calibration = list(rows)
        return train, calibration

    def _logistic_feature_allowed(self, name: str) -> bool:
        if not self._use_online_safe_logistic_features():
            return True
        lowered = name.lower()
        return not any(pattern in lowered for pattern in LOGISTIC_LOGPROB_FEATURE_PATTERNS)

    @staticmethod
    def _load_detector_thresholds(path: str) -> dict[str, Any]:
        if not path:
            return {}
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}

    def _threshold_for_detector(self, detector: str, sample_id: str | None) -> float:
        if self.detector_thresholds:
            fold_thresholds = self.detector_thresholds.get("fold_thresholds")
            if isinstance(fold_thresholds, dict):
                folds = int(self.detector_thresholds.get("folds") or 0)
                fold = (
                    self._episode_fold(str(sample_id), folds)
                    if sample_id is not None and folds > 1
                    else None
                )
                if fold is not None:
                    value = (fold_thresholds.get(str(fold)) or {}).get(detector)
                    if value is not None:
                        return float(value)
            value = self.detector_thresholds.get(detector)
            if value is not None:
                return float(value)
        return float(self.detector_signal_threshold)

    def _repair_extract_source_for(self, effective_arm: str, repair_kind: str) -> str:
        if self.repair_extract_source != "auto":
            return self.repair_extract_source
        if repair_kind == "raw" and effective_arm == "raw_all_replace_direct":
            return "serving_cache"
        return "model_prefill"

    def run_sample(self, test_case: dict[str, Any]) -> dict[str, Any]:
        self._active_tools = _tool_payload(test_case["function"])
        try:
            stats = DriftStats(test_case["id"], self.mode, self.ratio)
            result, metadata = self._run_sample_impl(test_case, stats)
            metrics = stats.as_dict()
            metrics.update(self._repair_metrics(metadata.get("repair_segments") or []))
            metadata["c2kv_drift_metrics"] = metrics
            return {"id": test_case["id"], "result": result, **metadata}
        finally:
            self._active_tools = []

    @staticmethod
    def _repair_metrics(segments: Sequence[dict[str, Any]]) -> dict[str, Any]:
        total = len(segments)
        triggered = [seg for seg in segments if seg.get("repair_triggered")]
        harmful = [seg for seg in segments if seg.get("oracle_segment_harmful")]
        successful = [
            seg for seg in triggered if seg.get("repair_segment_success") is True
        ]
        start_state_correct = [
            seg for seg in triggered if seg.get("segment_start_state_matches_reference")
        ]
        start_state_drifted = [
            seg
            for seg in triggered
            if seg.get("segment_start_state_matches_reference") is False
        ]
        start_state_correct_success = [
            seg for seg in start_state_correct if seg.get("repair_segment_success") is True
        ]
        start_state_drifted_success = [
            seg for seg in start_state_drifted if seg.get("repair_segment_success") is True
        ]
        wrong_to_correct = sum(
            int(seg.get("c2kv_wrong_repair_correct") or 0) for seg in segments
        )
        correct_to_wrong = sum(
            int(seg.get("c2kv_correct_repair_wrong") or 0) for seg in segments
        )
        changed_action = sum(
            int(seg.get("repair_changed_action_count") or 0) for seg in segments
        )
        changed_first = sum(
            int(seg.get("repair_changed_first_token_count") or 0) for seg in segments
        )
        repaired_steps = sum(
            int(seg.get("segment_length") or 0) for seg in triggered
        )
        return {
            "repair_segments": total,
            "oracle_harmful_segments": len(harmful),
            "detector_trigger_count": len(triggered),
            "detector_trigger_rate": len(triggered) / total if total else None,
            "repair_rate": len(triggered) / total if total else None,
            "repair_success_count": len(successful),
            "repair_success_rate": (
                len(successful) / len(triggered) if triggered else None
            ),
            "repair_segment_success_rate": (
                len(successful) / len(harmful) if harmful else None
            ),
            "c2kv_wrong_repair_correct": wrong_to_correct,
            "c2kv_wrong_repair_wrong": sum(
                int(seg.get("c2kv_wrong_repair_wrong") or 0) for seg in segments
            ),
            "c2kv_correct_repair_wrong": correct_to_wrong,
            "net_repair_gain": wrong_to_correct - correct_to_wrong,
            "repair_changed_action_count": changed_action,
            "repair_changed_action_rate": (
                changed_action / repaired_steps if repaired_steps else None
            ),
            "repair_changed_first_token_count": changed_first,
            "repair_changed_first_token_rate": (
                changed_first / repaired_steps if repaired_steps else None
            ),
            "repaired_step_count": repaired_steps,
            "repair_success_when_start_state_correct": (
                len(start_state_correct_success) / len(start_state_correct)
                if start_state_correct
                else None
            ),
            "repair_success_when_start_state_already_drifted": (
                len(start_state_drifted_success) / len(start_state_drifted)
                if start_state_drifted
                else None
            ),
            "speculative_terminal_discarded_count": sum(
                int(bool(seg.get("speculative_terminal_discarded")))
                for seg in segments
            ),
            "detector_tp": sum(int(bool(seg.get("detector_tp"))) for seg in segments),
            "detector_fp": sum(int(bool(seg.get("detector_fp"))) for seg in segments),
            "detector_tn": sum(int(bool(seg.get("detector_tn"))) for seg in segments),
            "detector_fn": sum(int(bool(seg.get("detector_fn"))) for seg in segments),
            "tp_recovery_attempts": sum(
                int(bool(seg.get("detector_tp"))) for seg in segments
            ),
            "tp_recovery_success_count": sum(
                int(
                    bool(seg.get("detector_tp"))
                    and bool(seg.get("repair_segment_success"))
                )
                for seg in segments
            ),
            "fp_recovery_count": sum(
                int(bool(seg.get("detector_fp"))) for seg in segments
            ),
            "fp_recovery_harm_count": sum(
                int(
                    bool(seg.get("detector_fp"))
                    and bool(seg.get("fp_recovery_harm"))
                )
                for seg in segments
            ),
            "false_negative_count": sum(
                int(bool(seg.get("detector_fn"))) for seg in segments
            ),
        }

    @staticmethod
    def _action_text(action: list[str]) -> str:
        return json.dumps(action or [], ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _parse_action_objects(action: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
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
        values: list[str] = []
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
            self._action_text(
                (info.get("step_record") or {}).get("candidate_action") or []
            )
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
        record = info.get("step_record") or {}
        action = record.get("candidate_action") or []
        hard_error = bool(
            record.get("candidate_status") in {"decode_error", "invalid_format"}
            or record.get("execution_error")
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
            execution_results=record.get("execution_results") or [],
            execution_error=record.get("execution_error"),
            candidate_status=record.get("candidate_status") or "",
            action=action,
        )
        tool_transition_anomaly = (
            1.0 if repeat_score >= 1.0 and observation > 0.0 else 0.0
        )
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
            "risk_score": risk_score,
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

    def _rule_detector(self, segment_infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
            "detector": "rule_trigger",
            "detector_trigger": triggered,
            "detector_reason": reason,
            "rule_detector_trigger": triggered,
            "rule_detector_max_risk": max_risk,
            "rule_detector_reason": reason,
            "rule_detector_threshold": self.rule_detector_threshold,
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _threshold_at_high_recall(
        labels: Sequence[int],
        scores: Sequence[float],
        target_recall: float = 0.95,
    ) -> float:
        best: tuple[float, float, float] | None = None
        positives = sum(labels)
        negatives = len(labels) - positives
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
            recall = tp / positives if positives else 0.0
            fpr = fp / negatives if negatives else 0.0
            if recall >= target_recall:
                candidate = (fpr, -recall, threshold)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            return float(best[2])
        return min(scores) if scores else 0.5

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

    def _select_logistic_features(
        self,
        rows: Sequence[dict[str, Any]],
        train: Sequence[dict[str, Any]],
    ) -> list[str]:
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
            if key not in excluded and self._logistic_feature_allowed(key)
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
        return usable

    def _fit_logistic_detector(
        self,
        train: Sequence[dict[str, Any]],
        *,
        usable: Sequence[str],
        threshold_rule: str,
        threshold_rows: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
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

        calibration_rows = list(threshold_rows or train)
        calibration_scores = [
            self._sigmoid(sum(w * xi for w, xi in zip(weights, vector(row))))
            for row in calibration_rows
        ]
        calibration_labels = [
            int(float(row["segment_harmful"])) for row in calibration_rows
        ]
        threshold = (
            self._threshold_at_high_recall(calibration_labels, calibration_scores, 0.95)
            if threshold_rule == "high_recall"
            else self._best_f1_threshold(calibration_labels, calibration_scores)
        )
        return {
            "features": list(usable),
            "means": means,
            "stds": stds,
            "weights": weights,
            "threshold": threshold,
            "threshold_selection_rule": threshold_rule,
            "train_rows": len(train),
            "calibration_rows": len(calibration_rows),
            "train_episode_ids": sorted({str(row.get("id")) for row in train}),
            "calibration_episode_ids": sorted(
                {str(row.get("id")) for row in calibration_rows}
            ),
        }

    def _write_logistic_cv_artifacts(self, model: dict[str, Any]) -> None:
        if not self.detector_cv_output_dir or not model.get("kfold_models"):
            return
        import pickle

        root = Path(self.detector_cv_output_dir)
        root.mkdir(parents=True, exist_ok=True)
        thresholds: dict[str, Any] = {
            "detector": self.detector_arm,
            "kfolds": model.get("kfolds"),
            "feature_set": model.get("feature_set"),
            "fold_thresholds": {},
        }
        for fold, fold_model in model["kfold_models"].items():
            fold_dir = root / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            with open(fold_dir / "logistic_model.pkl", "wb") as f:
                pickle.dump(fold_model, f)
            with open(fold_dir / "scaler.pkl", "wb") as f:
                pickle.dump(
                    {
                        "features": fold_model.get("features"),
                        "means": fold_model.get("means"),
                        "stds": fold_model.get("stds"),
                    },
                    f,
                )
            threshold_path = fold_dir / "thresholds.json"
            existing_thresholds: dict[str, Any] = {}
            if threshold_path.exists():
                try:
                    existing_thresholds = json.loads(
                        threshold_path.read_text(encoding="utf-8")
                    )
                except Exception:
                    existing_thresholds = {}
            fold_threshold = {
                **existing_thresholds,
                "combined_logistic": fold_model.get("threshold"),
                "combined_logistic_best_f1": fold_model.get("threshold"),
                "threshold_selection_rule": fold_model.get(
                    "threshold_selection_rule"
                ),
                "train_episode_ids": fold_model.get("train_episode_ids"),
                "calibration_episode_ids": fold_model.get(
                    "calibration_episode_ids"
                ),
                "test_episode_ids": fold_model.get("fold_test_episode_ids"),
            }
            (fold_dir / "thresholds.json").write_text(
                json.dumps(fold_threshold, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            thresholds["fold_thresholds"][str(fold)] = fold_threshold
        (root / "logistic_cv_thresholds.json").write_text(
            json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _train_logistic_detector(
        self,
        features_csv: str,
        *,
        threshold_rule: str,
    ) -> dict[str, Any]:
        if not features_csv:
            raise ValueError(
                f"detector_arm={self.detector_arm} requires "
                "--logistic-detector-features-csv"
            )
        with open(features_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"logistic detector feature CSV is empty: {features_csv}")
        labeled_rows = [
            row
            for row in rows
            if row.get("id") not in (None, "")
            and row.get("segment_harmful") not in (None, "")
        ]
        if self.logistic_detector_kfolds > 1:
            folds = self.logistic_detector_kfolds
            usable = self._select_logistic_features(labeled_rows, labeled_rows)
            if not usable:
                raise ValueError("logistic detector has no usable numeric features")
            fold_models: dict[str, dict[str, Any]] = {}
            episode_ids = sorted({str(row.get("id")) for row in labeled_rows})
            for fold in range(folds):
                train = [
                    row
                    for row in labeled_rows
                    if self._episode_fold(str(row.get("id")), folds) != fold
                ]
                train_fit, calibration = self._inner_train_calibration_split(
                    train,
                    outer_fold=fold,
                )
                test_episode_ids = [
                    sample_id
                    for sample_id in episode_ids
                    if self._episode_fold(sample_id, folds) == fold
                ]
                if not train:
                    raise ValueError(
                        f"logistic kfold={fold} has no training rows; folds={folds}"
                    )
                model = self._fit_logistic_detector(
                    train_fit,
                    usable=usable,
                    threshold_rule=threshold_rule,
                    threshold_rows=calibration,
                )
                model["fold"] = fold
                model["fold_test_episode_ids"] = test_episode_ids
                fold_models[str(fold)] = model
            root_model = {
                "features_csv": features_csv,
                "features": usable,
                "kfolds": folds,
                "kfold_models": fold_models,
                "threshold_selection_rule": threshold_rule,
                "feature_set": (
                    "online_safe"
                    if self._use_online_safe_logistic_features()
                    else "all"
                ),
                "train_rows": len(labeled_rows),
                "train_episode_ids": episode_ids,
                "test_episode_ids": episode_ids,
            }
            self._write_logistic_cv_artifacts(root_model)
            return root_model

        train = [row for row in labeled_rows if row.get("split") == "calibration"]
        if not train:
            raise ValueError(
                "logistic detector feature CSV has no calibration split rows: "
                f"{features_csv}"
            )
        usable = self._select_logistic_features(labeled_rows, train)
        if not usable:
            raise ValueError("logistic detector has no usable numeric features")
        model = self._fit_logistic_detector(
            train,
            usable=usable,
            threshold_rule=threshold_rule,
        )
        model.update(
            {
                "features_csv": features_csv,
                "feature_set": (
                    "online_safe"
                    if self._use_online_safe_logistic_features()
                    else "all"
                ),
                "test_episode_ids": sorted(
                    {
                        str(row.get("id"))
                        for row in labeled_rows
                        if row.get("split") == "test"
                    }
                ),
            }
        )
        return model

    def _logistic_detector(self, segment_infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if self.logistic_detector_model is None:
            raise RuntimeError("logistic detector model is not initialized")
        root_model = self.logistic_detector_model
        model = root_model
        sample_id = None
        if segment_infos:
            sample_id = (segment_infos[0].get("step_record") or {}).get("id")
        selected_fold = None
        if model.get("kfold_models"):
            folds = int(model.get("kfolds") or 0)
            selected_fold = self._episode_fold(str(sample_id), folds)
            model = model["kfold_models"][str(selected_fold)]
        row = self._segment_detector_feature_row(segment_infos)
        x = [1.0]
        for name in model["features"]:
            value = self._detector_score_for_feature(name, row.get(name))
            if value is None:
                value = model["means"][name]
            x.append((value - model["means"][name]) / model["stds"][name])
        score = self._sigmoid(sum(w * xi for w, xi in zip(model["weights"], x)))
        threshold = float(model["threshold"])
        triggered = score >= threshold
        return {
            "detector": self.detector_arm,
            "detector_trigger": triggered,
            "detector_reason": (
                "logistic_score_threshold" if triggered else "logistic_safe"
            ),
            "detector_threshold": threshold,
            "threshold_selection_rule": model["threshold_selection_rule"],
            "logistic_detector_score": score,
            "logistic_detector_threshold": threshold,
            "logistic_detector_feature_count": len(model["features"]),
            "logistic_detector_features": model["features"],
            "logistic_detector_train_rows": model["train_rows"],
            "detector_train_episode_ids": model["train_episode_ids"],
            "detector_test_episode_ids": (
                model.get("fold_test_episode_ids")
                or root_model.get("test_episode_ids")
                or []
            ),
            "logistic_detector_kfold": selected_fold,
            "logistic_detector_feature_set": root_model.get("feature_set"),
            "detector_evaluation_mode": (
                "episode_kfold_crossfit"
                if selected_fold is not None
                else "stable52_calibration_split"
            ),
        }

    def _feature_signal_detector(
        self,
        segment_infos: Sequence[dict[str, Any]],
        *,
        signal_name: str,
        threshold: float,
    ) -> dict[str, Any]:
        row = self._segment_detector_feature_row(segment_infos)
        score = self._detector_score_for_feature(signal_name, row.get(signal_name))
        triggered = False if score is None else score >= threshold
        return {
            "detector": signal_name,
            "detector_trigger": triggered,
            "detector_reason": (
                f"{signal_name}>={threshold:g}" if triggered else "feature_signal_safe"
            ),
            "detector_threshold": threshold,
            "detector_signal_name": signal_name,
            "detector_signal_score": score,
            "detector_signal_threshold": threshold,
        }

    def _segment_detector(self, segment_infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
        sample_id = None
        if segment_infos:
            sample_id = (segment_infos[0].get("step_record") or {}).get("id")
        if self.detector_arm == "never_trigger":
            return {"detector": "never_trigger", "detector_trigger": False}
        if self.detector_arm == "always_trigger":
            return {"detector": "always_trigger", "detector_trigger": True}
        if self.detector_arm == "oracle":
            triggered = any(
                bool((info.get("step_record") or {}).get("oracle_harmful"))
                for info in segment_infos
            )
            return {"detector": "oracle", "detector_trigger": triggered}
        if self.detector_arm == "rule_trigger":
            return self._rule_detector(segment_infos)
        if self.detector_arm in {
            "combined_logistic_best_f1",
            "combined_logistic_high_recall",
        }:
            return self._logistic_detector(segment_infos)
        if self.detector_arm == "max_risk_score":
            return self._feature_signal_detector(
                segment_infos,
                signal_name="max_risk_score",
                threshold=self._threshold_for_detector("max_risk_score", sample_id),
            )
        scalar_signals = {
            "rule_detector_max_risk": "rule_detector_max_risk",
            "max_observation_anomaly": "max_observation_anomaly",
            "mean_risk_score": "mean_risk_score",
            "max_hard_error": "max_hard_error",
            "max_generation_nll": "max_generation_nll",
            "mean_generation_nll": "mean_generation_nll",
        }
        if self.detector_arm in scalar_signals:
            signal_name = scalar_signals[self.detector_arm]
            return self._feature_signal_detector(
                segment_infos,
                signal_name=signal_name,
                threshold=self._threshold_for_detector(signal_name, sample_id),
            )
        raise RuntimeError(f"Unsupported detector arm: {self.detector_arm}")

    @staticmethod
    def _detector_confusion(
        *,
        oracle_segment_harmful: bool,
        detector_trigger: bool,
    ) -> dict[str, bool]:
        return {
            "detector_tp": detector_trigger and oracle_segment_harmful,
            "detector_fp": detector_trigger and not oracle_segment_harmful,
            "detector_tn": (not detector_trigger) and (not oracle_segment_harmful),
            "detector_fn": (not detector_trigger) and oracle_segment_harmful,
        }

    def _load_plan(self, path: str) -> dict[str, Any]:
        if not path:
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_neutral_tokens(self, path: str) -> list[int]:
        if not path:
            return []
        text = Path(path).read_text(encoding="utf-8")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _restore_instances(
        self,
        test_entry_id: str,
        involved_instances: dict[str, Any],
    ) -> None:
        for class_name, instance in involved_instances.items():
            key = (
                f"{self.decoder.model_name_underline_replaced}_"
                f"{test_entry_id}_{class_name}_instance"
            )
            key = re.sub(r"[-./:]", "_", key)
            mt_utils.__dict__[key] = deepcopy(instance)

    def _snapshot(
        self,
        *,
        messages: list[dict[str, Any]],
        involved_instances: dict[str, Any],
        current_turn_response: list[str],
        current_turn_inputs: list[int],
        current_turn_outputs: list[int],
        current_turn_latency: list[float],
        turn_log: dict[str, Any],
        global_step: int,
    ) -> dict[str, Any]:
        return {
            "messages": deepcopy(messages),
            "instances": deepcopy(involved_instances),
            "state": _state_log(involved_instances),
            "current_turn_response": deepcopy(current_turn_response),
            "current_turn_inputs": deepcopy(current_turn_inputs),
            "current_turn_outputs": deepcopy(current_turn_outputs),
            "current_turn_latency": deepcopy(current_turn_latency),
            "turn_log": deepcopy(turn_log),
            "global_step": global_step,
            "repair_target_history_index": self._latest_compressed_history_index(messages),
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
    ]:
        instances = deepcopy(snapshot["instances"])
        self._restore_instances(test_entry_id, instances)
        self._repair_target_history_index = snapshot.get("repair_target_history_index")
        return (
            deepcopy(snapshot["messages"]),
            instances,
            deepcopy(snapshot["current_turn_response"]),
            deepcopy(snapshot["current_turn_inputs"]),
            deepcopy(snapshot["current_turn_outputs"]),
            deepcopy(snapshot["current_turn_latency"]),
            deepcopy(snapshot["turn_log"]),
        )

    def _latest_compressed_history_index(
        self,
        history_messages: Sequence[dict[str, Any]],
    ) -> int | None:
        latest_query_index = _latest_user_query_index(history_messages)
        completed = list(history_messages[:latest_query_index])
        units = _history_units(completed)
        return len(units) - 1 if units else None

    def _unit_token_ids(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if self.tokenizer.bos_token and rendered.startswith(self.tokenizer.bos_token):
            rendered = rendered[len(self.tokenizer.bos_token) :]
        token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        return list(token_ids)

    @staticmethod
    def _normalize_token_ids(encoded: Any) -> list[int]:
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if isinstance(encoded, dict) and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(x) for x in encoded]

    def _full_prompt_token_ids(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        try:
            encoded = self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(self._active_tools),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "KV repair requires exact server-equivalent chat-template "
                f"tokenization with tools; failed with {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalize_token_ids(encoded)

    def _role_prompt_token_ids(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return self._normalize_token_ids(encoded)

    def _full_history_unit_layout(
        self,
        units: Sequence[Sequence[dict[str, Any]]],
        current_messages: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Locate each role-preserving history unit in the real Full prompt.

        The raw repair slice must come from the same token coordinates used by
        the OpenAI chat endpoint: system/tool template first, the original
        role-preserving completed history, then the current turn and assistant
        generation prompt. This intentionally does not reuse the C2KV
        user-document wrapper, because raw repair KV must be the KV that the
        target span would have in the exact Full context.
        """

        completed_messages = [
            deepcopy(message)
            for unit in units
            for message in unit
        ]
        full_messages = completed_messages + deepcopy(list(current_messages or []))
        full_tokens = self._full_prompt_token_ids(
            full_messages,
            add_generation_prompt=True,
        )
        starts: list[int] = []
        ends: list[int] = []
        prefix_messages: list[dict[str, Any]] = []
        cursor = 0
        for index, unit in enumerate(units):
            if prefix_messages:
                start = len(self._full_prompt_token_ids(prefix_messages))
            else:
                unit_ids = self._role_prompt_token_ids(unit)
                found = -1
                limit = len(full_tokens) - len(unit_ids) + 1
                for pos in range(cursor, max(cursor, limit)):
                    if full_tokens[pos : pos + len(unit_ids)] == unit_ids:
                        found = pos
                        break
                if found < 0:
                    raise RuntimeError(
                        "Cannot locate first role-preserving history unit in "
                        "Full prompt tokenization."
                    )
                start = found
            prefix_messages.extend(deepcopy(list(unit)))
            end = len(self._full_prompt_token_ids(prefix_messages))
            if not (0 <= start < end <= len(full_tokens)):
                raise RuntimeError(
                    "Invalid Full prompt history-unit token span: "
                    f"unit_index={index}, start={start}, end={end}, "
                    f"full_len={len(full_tokens)}"
                )
            if full_tokens[:end] != self._full_prompt_token_ids(prefix_messages):
                raise RuntimeError(
                    "Full prompt prefix tokenization is not prefix-stable for "
                    f"history unit {index}; refusing to build raw repair KV."
                )
            starts.append(start)
            ends.append(end)
            cursor = end
        return full_tokens, starts, ends

    def _extract_repair(
        self,
        *,
        input_ids: list[int],
        span_start: int,
        span_end: int,
        position_offset: int,
        repair_mode: str,
        source_doc_index: int | None,
        stats: DriftStats,
        extract_source: str = "model_prefill",
    ) -> dict[str, Any]:
        def _pad_token_id() -> int:
            for attr in ("eos_token_id", "pad_token_id"):
                value = getattr(self.tokenizer, attr, None)
                if value is not None:
                    return int(value)
            return 0

        def _align_for_serving_cache(ids: list[int], page_size: int) -> list[int]:
            if page_size <= 1:
                return ids
            target_len = ((span_end + page_size - 1) // page_size) * page_size
            if len(ids) >= target_len:
                return ids
            # Padding is appended strictly after the requested repair span. In
            # causal prefill it cannot affect the K/V of [span_start, span_end),
            # but it keeps the containing page resident in the radix cache.
            return ids + [_pad_token_id()] * (target_len - len(ids))

        def _warm_serving_cache(ids: list[int]) -> None:
            nonlocal stats
            warm_start = time.perf_counter()
            warm = _post_json(
                self.base_url,
                "/generate",
                {
                    "input_ids": ids,
                    "sampling_params": {
                        "max_new_tokens": 0,
                        "temperature": 0,
                    },
                },
                self.timeout,
            )
            warm_elapsed = time.perf_counter() - warm_start
            stats.extract_seconds += warm_elapsed
            stats.repair_extract_seconds += warm_elapsed
            stats.extract_calls += 1
            stats.extract_success += 1
            warm_meta = warm.get("meta_info") if isinstance(warm, dict) else {}
            if isinstance(warm_meta, dict):
                warm_prompt_tokens = int(
                    warm_meta.get("prompt_tokens") or len(ids)
                )
                warm_cached_tokens = int(warm_meta.get("cached_tokens") or 0)
                stats.repair_extract_recomputed_tokens += max(
                    warm_prompt_tokens - warm_cached_tokens,
                    0,
                )
            else:
                stats.repair_extract_recomputed_tokens += len(ids)

        if extract_source == "serving_cache":
            _warm_serving_cache(input_ids)

        def _repair_extract_call(ids: list[int]) -> tuple[dict[str, Any], float]:
            start = time.perf_counter()
            result = _post_json(
                self.base_url,
                "/v1/c2kv/repair_extract",
                {
                    "input_ids": ids,
                    "span_start": span_start,
                    "span_end": span_end,
                    "position_offset": position_offset,
                    "repair_mode": repair_mode,
                    "source_doc_index": source_doc_index,
                    "extract_source": extract_source,
                },
                self.timeout,
            )
            return result, time.perf_counter() - start

        repair_extract_calls = 1
        result, elapsed = _repair_extract_call(input_ids)
        if (
            extract_source == "serving_cache"
            and not result.get("success")
            and "PREFIX_NOT_FOUND_IN_SERVING_CACHE" in str(result.get("error") or "")
        ):
            match = re.search(r"page_size=(\d+)", str(result.get("error") or ""))
            if match:
                page_size = int(match.group(1))
                padded_input_ids = _align_for_serving_cache(input_ids, page_size)
                if len(padded_input_ids) > len(input_ids):
                    _warm_serving_cache(padded_input_ids)
                    result, retry_elapsed = _repair_extract_call(padded_input_ids)
                    elapsed += retry_elapsed
                    repair_extract_calls += 1
        stats.extract_seconds += elapsed
        stats.repair_extract_seconds += elapsed
        if extract_source != "serving_cache":
            stats.repair_extract_recomputed_tokens += len(input_ids)
        stats.extract_calls += repair_extract_calls
        if result.get("success"):
            stats.extract_success += 1
        else:
            raise RuntimeError(
                f"repair_extract failed for {repair_mode}: {result.get('error')}"
            )
        return result

    def _query_with_raw(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stats: DriftStats,
        *,
        collect_detector_signals: bool,
    ) -> tuple[str, dict[str, Any], float, dict[str, Any], dict[str, Any]]:
        prompt_tokens = _token_count(self.tokenizer, messages)
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": self.temperature,
            "max_completion_tokens": max(1, self.max_completion_tokens),
            "chat_template_kwargs": {"enable_thinking": False},
            "return_cached_tokens_details": True,
        }
        memory_hint = getattr(self, "_last_kv_memory_hint", None)
        if isinstance(memory_hint, dict):
            payload["c2kv_kv_memory_hint"] = memory_hint
        if collect_detector_signals and self.request_candidate_logprobs:
            payload.update(
                {
                    "logprobs": True,
                    "top_logprobs": self.candidate_logprobs_top_k,
                }
            )
        start = time.perf_counter()
        data = _post_json(self.base_url, "/v1/chat/completions", payload, self.timeout)
        elapsed = time.perf_counter() - start
        choice = (data.get("choices") or [{}])[0] or {}
        message = choice.get("message", {}) or {}
        text = message.get("content") or ""
        tool_calls_text = _tool_calls_to_text(message.get("tool_calls"))
        if tool_calls_text:
            text = (text + "\n" + tool_calls_text).strip() if text else tool_calls_text
        usage = data.get("usage") or {}
        stats.chat_calls += 1
        stats.chat_seconds += elapsed
        usage_prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        cached_tokens = _actual_cached_tokens_from_response(data)
        if cached_tokens is None:
            stats.chat_cache_report_missing += 1
            recomputed_prompt_tokens = 0
        else:
            cached_tokens = min(usage_prompt_tokens, max(0, int(cached_tokens)))
            stats.chat_cached_tokens += cached_tokens
            recomputed_prompt_tokens = max(usage_prompt_tokens - cached_tokens, 0)
            stats.chat_recomputed_prompt_tokens += recomputed_prompt_tokens
        runtime = _kv_runtime_stats_from_response(data)
        kv_memory_report = _kv_memory_report_from_response(data)
        if runtime is None:
            stats.kv_runtime_report_missing += 1
        else:
            try:
                stats.kv_peak_resident_tokens = max(
                    stats.kv_peak_resident_tokens,
                    int(runtime.get("kv_resident_tokens") or 0),
                )
            except Exception:
                stats.kv_runtime_report_missing += 1
        usage_completion_tokens = int(
            usage.get("completion_tokens")
            or len(self.tokenizer.encode(text, add_special_tokens=False))
        )
        stats.chat_prompt_tokens += usage_prompt_tokens
        stats.chat_completion_tokens += usage_completion_tokens
        return text, message, elapsed, {
            "prompt_tokens": usage_prompt_tokens,
            "completion_tokens": usage_completion_tokens,
            "cached_tokens": cached_tokens,
            "recomputed_prompt_tokens": recomputed_prompt_tokens,
            "kv_runtime_stats": runtime,
            "kv_memory_report": kv_memory_report,
        }, data

    def _candidate_detector_features(self, raw: dict[str, Any]) -> dict[str, Any]:
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

        nll = -sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
        ppl = math.exp(min(nll, 50.0)) if nll is not None else None
        return {
            "detector_signal_requested": bool(self.request_candidate_logprobs),
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
        }

    def _plan_for(self, sample_id: str, num_docs: int, doc_lens: list[int]) -> dict[str, Any]:
        plan_root = self.plan.get("per_qid") if isinstance(self.plan, dict) else None
        if not isinstance(plan_root, dict):
            plan_root = self.plan
        plan = plan_root.get(sample_id) or plan_root.get(str(sample_id))
        if plan is None:
            if self._repair_target_history_index is None:
                k_star = num_docs - 1
            else:
                k_star = min(max(0, int(self._repair_target_history_index)), num_docs - 1)
            return {
                "k_star": k_star,
                "span_len": doc_lens[k_star],
                "sham_token_ids": [],
                "source": "online_latest_compressed_history",
            }
        k_star = int(plan.get("k_star"))
        if not (0 <= k_star < num_docs):
            raise RuntimeError(f"k_star out of range for {sample_id}: {k_star} / {num_docs}")
        span_len = int(plan.get("span_len", doc_lens[k_star]))
        if span_len != doc_lens[k_star]:
            raise RuntimeError(
                f"span length mismatch for {sample_id}: plan={span_len}, actual={doc_lens[k_star]}"
            )
        return plan

    def _neutral_ids_for(self, plan: dict[str, Any], span_len: int) -> list[int]:
        sham_ids = list(plan.get("sham_token_ids") or [])
        if sham_ids:
            if len(sham_ids) != span_len:
                raise RuntimeError(
                    f"d_sham_neutral token length mismatch: {len(sham_ids)} != {span_len}"
                )
            return [int(x) for x in sham_ids]
        if not self.neutral_token_ids:
            raise RuntimeError("neutral corpus does not contain any tokens")
        if len(self.neutral_token_ids) >= span_len:
            return self.neutral_token_ids[:span_len]

        repeats = (span_len + len(self.neutral_token_ids) - 1) // len(
            self.neutral_token_ids
        )
        return (self.neutral_token_ids * repeats)[:span_len]

    def _arm_config(self, effective_arm: str) -> dict[str, Any]:
        config = {
            "operation": "append",
            "repair_kind": "none",
            "window": self._repair_window_arg,
            "hint": False,
            "oracle_location_hint": False,
        }
        if effective_arm in {"c2kv", "d_sham_mech"}:
            config["repair_kind"] = "none"
        elif effective_arm == "hint_only":
            config.update({"repair_kind": "none", "hint": True})
        elif effective_arm == "d_sham_neutral":
            config.update({"repair_kind": "neutral", "window": "1"})
        elif effective_arm in {"d_corr", "d_corr_w1"}:
            config.update({"repair_kind": "raw", "window": "1"})
        elif effective_arm in {"d_corr_w2", "d_corr_w2_hint"}:
            config.update({
                "repair_kind": "raw",
                "window": "2",
                "hint": effective_arm.endswith("_hint"),
            })
        elif effective_arm == "d_corr_w2_oracle_location_hint":
            config.update({
                "repair_kind": "raw",
                "window": "2",
                "hint": True,
                "oracle_location_hint": True,
            })
        elif effective_arm == "d_corr_w4":
            config.update({"repair_kind": "raw", "window": "4"})
        elif effective_arm == "d_corr_all":
            config.update({"repair_kind": "raw", "window": "all"})
        elif effective_arm == "d_corr_replace_w1":
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "1",
            })
        elif effective_arm == "d_corr_replace_w2":
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "2",
            })
        elif effective_arm == "append_masked_w2":
            config.update({
                "operation": "append_masked",
                "repair_kind": "raw",
                "window": "2",
            })
        elif effective_arm == "d_corr_replace_w4":
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "4",
            })
        elif effective_arm == "d_corr_replace_all":
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "all",
            })
        elif effective_arm in {"d_corr_recompute", "d_corr_recompute_w2"}:
            config.update({
                "operation": "recompute",
                "repair_kind": "raw",
                "window": "2",
            })
        elif effective_arm in {"raw_all_replace", "raw_all_replace_direct"}:
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "all",
            })
        else:
            raise RuntimeError(f"Unsupported KV repair arm: {effective_arm}")
        return config

    @staticmethod
    def _ordered_roles(unit: Sequence[dict[str, Any]]) -> list[str]:
        roles: list[str] = []
        for message in unit:
            role = str(message.get("role") or "")
            if role and role not in roles:
                roles.append(role)
        return roles

    def _recent_repair_indices(
        self,
        *,
        num_docs: int,
        latest_index: int,
        window: str,
    ) -> list[int]:
        if num_docs <= 0:
            return []
        latest_index = min(max(0, latest_index), num_docs - 1)
        if window == "all":
            return list(range(num_docs))
        try:
            width = int(window)
        except Exception as exc:
            raise RuntimeError(f"Invalid repair window: {window}") from exc
        if width <= 0:
            raise RuntimeError(f"repair window must be positive, got {window}")
        start = max(0, latest_index - width + 1)
        return list(range(start, latest_index + 1))

    def _repair_hint(
        self,
        *,
        target_metadata: Sequence[dict[str, Any]],
        include_oracle_location: bool,
    ) -> str:
        restored = "; ".join(
            (
                f"H{meta['history_index']}: "
                f"{'+'.join(meta.get('roles') or [])}, "
                f"{meta.get('raw_token_count', 0)} raw-KV tokens"
            )
            for meta in target_metadata
        )
        lines = [
            "Recovery note: the current speculative segment was detected as inconsistent with the execution history.",
            (
                "Raw KV memory for history units "
                f"{[meta['history_index'] for meta in target_metadata]} "
                "has been restored and attached to the compressed history."
            ),
            f"Restored units contain: {restored}.",
            (
                "Re-evaluate the current tool call using the restored history, "
                "especially restored assistant/tool observations."
            ),
            "Do not assume the previous candidate action is correct.",
        ]
        if include_oracle_location and self._active_oracle_bad_step is not None:
            lines.append(
                "The inconsistency was first observed at speculative step "
                f"S{self._active_oracle_bad_step}."
            )
        return "\n".join(lines)

    def _build_request_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> list[dict[str, Any]]:
        self._last_repair_build_info = {}
        self._last_kv_memory_hint = None
        latest_query_index = _latest_user_query_index(history_messages)
        completed = list(history_messages[:latest_query_index])
        current = deepcopy(list(history_messages[latest_query_index:]))
        if self.arm == "full":
            full_tokens = _token_count(self.tokenizer, completed)
            stats.original_history_tokens += full_tokens
            stats.effective_history_tokens += full_tokens
            stats.canonical_full_history_tokens += full_tokens
            stats.physical_history_kv_tokens += full_tokens
            self._last_kv_memory_hint = {
                "full_equivalent_history_tokens": full_tokens,
                "active_history_kv_tokens": full_tokens,
                "active_full_raw_tokens": full_tokens,
                "history_scope": "completed_history_only",
                "source": "bfcl_canonical_history_layout",
            }
            return deepcopy(list(history_messages))
        effective_arm = self.arm if self._repair_enabled_for_current_step else "c2kv"

        units = _history_units(completed)
        if not units:
            self._last_kv_memory_hint = {
                "full_equivalent_history_tokens": 0,
                "active_history_kv_tokens": 0,
                "history_scope": "completed_history_only",
                "source": "bfcl_canonical_history_layout",
            }
            return deepcopy(list(history_messages))

        texts = [_render_history_unit(unit) for unit in units]
        doc_ids = [self._unit_token_ids(text) for text in texts]
        full_prompt_ids, starts, ends = self._full_history_unit_layout(
            units,
            current_messages=current,
        )
        doc_lens = [end - start for start, end in zip(starts, ends)]
        canonical_full_history_tokens = sum(doc_lens)
        stats.canonical_full_history_tokens += canonical_full_history_tokens

        sample_id = getattr(stats, "sample_id", "") or getattr(stats, "id", "")
        plan = self._plan_for(sample_id, len(units), doc_lens)
        latest_index = int(plan.get("k_star", (len(units) - 1)))
        config = self._arm_config(effective_arm)
        target_indices = self._recent_repair_indices(
            num_docs=len(units),
            latest_index=latest_index,
            window=str(config["window"]),
        )
        target_set = set(target_indices)
        anchor_index = target_indices[0] if target_indices else latest_index

        if (
            effective_arm
            in {"d_corr_replace_all", "raw_all_replace", "raw_all_replace_direct"}
            and config["operation"] == "replace"
            and config["repair_kind"] == "raw"
            and target_indices == list(range(len(units)))
        ):
            history_start = starts[0]
            history_end = ends[-1]
            history_len = history_end - history_start
            if history_len != canonical_full_history_tokens:
                raise RuntimeError(
                    "raw_all_replace full-history span is not contiguous: "
                    f"span_len={history_len}, "
                    f"canonical_full_history_tokens={canonical_full_history_tokens}"
                )

            repair = self._extract_repair(
                input_ids=(
                    full_prompt_ids
                    if self._repair_extract_source_for(effective_arm, "raw")
                    == "serving_cache"
                    else full_prompt_ids[:history_end]
                ),
                span_start=history_start,
                span_end=history_end,
                position_offset=0,
                repair_mode=effective_arm,
                source_doc_index=None,
                stats=stats,
                extract_source=self._repair_extract_source_for(effective_arm, "raw"),
            )
            token_len = int(repair["token_len"])
            if token_len != history_len:
                raise RuntimeError(
                    "raw_all_replace full-history repair length mismatch: "
                    f"requested={history_len}, injected={token_len}"
                )
            position_start = int(repair.get("position_start", history_start))
            position_end = int(repair.get("position_end", history_end))
            if position_start != history_start or position_end != history_end:
                raise RuntimeError(
                    "raw_all_replace full-history position range mismatch: "
                    f"expected=({history_start}, {history_end}), "
                    f"actual=({position_start}, {position_end})"
                )

            repair_metadata = [
                {
                    "history_index": index,
                    "roles": self._ordered_roles(units[index]),
                    "raw_token_count": doc_lens[index],
                    "absolute_position_start": starts[index],
                    "absolute_position_end": ends[index],
                }
                for index in target_indices
            ]
            stats.original_history_tokens += canonical_full_history_tokens
            stats.effective_history_tokens += token_len
            stats.physical_history_kv_tokens += token_len
            stats.repair_kv_tokens += token_len

            self._last_repair_build_info = {
                "repair_mode": effective_arm,
                "repair_extract_source": repair.get("extract_source") or "",
                "repair_extract_cache_hit_tokens": int(
                    repair.get("cache_hit_tokens") or 0
                ),
                "repair_window": config["window"],
                "repair_operation": config["operation"],
                "uses_oracle_error_location": bool(config["oracle_location_hint"]),
                "repair_target_indices": target_indices,
                "repair_target_roles": [
                    self._ordered_roles(units[index]) for index in target_indices
                ],
                "repair_target_metadata": repair_metadata,
                "repair_tokens_requested": token_len,
                "repair_tokens_injected": token_len,
                "repair_raw_tokens": token_len,
                "repair_physical_tokens": token_len,
                "repair_absolute_position_ranges": [
                    [meta["absolute_position_start"], meta["absolute_position_end"]]
                    for meta in repair_metadata
                ],
                "combined_full_history_repair": True,
                "physical_prefix_len_before": 0,
                "physical_prefix_len_after": token_len,
                "logical_position_before": canonical_full_history_tokens,
                "logical_position_after": canonical_full_history_tokens,
            }
            self._last_kv_memory_hint = {
                "full_equivalent_history_tokens": canonical_full_history_tokens,
                "history_scope": "completed_history_only",
                "source": "bfcl_canonical_history_layout",
            }

            return [
                {
                    "role": "user",
                    "content": "\n".join(texts),
                    "c2kv_repair_only_key_hashes": [repair["key_hash"]],
                },
                *current,
            ]

        gist_records: list[ExtractRecord | None] = []
        messages: list[dict[str, Any]] = []
        repair_keys_by_index: dict[int, list[str]] = {}
        repair_tokens_by_index: dict[int, int] = {}
        repair_metadata: list[dict[str, Any]] = []
        history_layout_debug: list[dict[str, Any]] = []
        repair_tokens = 0
        local_physical_history_tokens = 0

        def should_compress_doc(index: int) -> bool:
            if config["operation"] == "replace" and index in target_set:
                return False
            if config["operation"] == "append_masked" and index in target_set:
                return False
            if config["operation"] == "recompute" and index >= anchor_index:
                return False
            return True

        def build_repair_for_index(index: int) -> None:
            nonlocal repair_tokens
            if config["repair_kind"] == "none":
                return
            span_len = doc_lens[index]
            if config["repair_kind"] == "neutral":
                input_ids = self._neutral_ids_for(plan, span_len)
                span_start = 0
                span_end = span_len
                position_offset = starts[index]
            elif config["repair_kind"] == "raw":
                extract_source = self._repair_extract_source_for(
                    effective_arm,
                    "raw",
                )
                input_ids = (
                    full_prompt_ids
                    if extract_source == "serving_cache"
                    else full_prompt_ids[: ends[index]]
                )
                span_start = starts[index]
                span_end = ends[index]
                position_offset = 0
            else:
                raise RuntimeError(f"Unsupported repair kind: {config['repair_kind']}")
            if config["repair_kind"] != "raw":
                extract_source = self._repair_extract_source_for(
                    effective_arm,
                    str(config["repair_kind"]),
                )

            repair = self._extract_repair(
                input_ids=input_ids,
                span_start=span_start,
                span_end=span_end,
                position_offset=position_offset,
                repair_mode=effective_arm,
                source_doc_index=index,
                stats=stats,
                extract_source=extract_source,
            )
            token_len = int(repair["token_len"])
            expected_len = span_len
            if token_len != expected_len:
                raise RuntimeError(
                    "repair token length mismatch: "
                    f"index={index}, requested={expected_len}, injected={token_len}"
                )
            repair_keys_by_index.setdefault(index, []).append(repair["key_hash"])
            repair_tokens_by_index[index] = repair_tokens_by_index.get(index, 0) + token_len
            repair_tokens += token_len
            position_start = int(repair.get("position_start", starts[index]))
            position_end = int(repair.get("position_end", ends[index]))
            if position_start != starts[index] or position_end != ends[index]:
                raise RuntimeError(
                    "repair position range mismatch: "
                    f"index={index}, expected=({starts[index]}, {ends[index]}), "
                    f"actual=({position_start}, {position_end})"
                )
            repair_metadata.append(
                {
                    "history_index": index,
                    "roles": self._ordered_roles(units[index]),
                    "raw_token_count": token_len,
                    "absolute_position_start": position_start,
                    "absolute_position_end": position_end,
                    "repair_extract_source": repair.get("extract_source") or "",
                    "repair_extract_cache_hit_tokens": int(
                        repair.get("cache_hit_tokens") or 0
                    ),
                }
            )

        if config["operation"] == "append":
            for index in target_indices:
                build_repair_for_index(index)
        elif config["operation"] in {"replace", "append_masked"}:
            for index in target_indices:
                build_repair_for_index(index)
        elif config["operation"] == "recompute":
            build_repair_for_index(anchor_index)

        for index, (unit, text, ids) in enumerate(zip(units, texts, doc_ids)):
            if not should_compress_doc(index):
                stats.original_history_tokens += doc_lens[index]
                if index in repair_keys_by_index:
                    raw_len = repair_tokens_by_index[index]
                    stats.effective_history_tokens += raw_len
                    stats.physical_history_kv_tokens += raw_len
                    stats.repair_kv_tokens += raw_len
                    local_physical_history_tokens += raw_len
                    if config["operation"] == "append_masked":
                        history_layout_debug.append(
                            {
                                "history_index": index,
                                "mode": "raw_append_masked",
                                "roles": self._ordered_roles(unit),
                                "logical_token_range": [starts[index], ends[index]],
                                "absolute_rope_position_range": [starts[index], ends[index]],
                                "raw_tokens": raw_len,
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": text,
                                "c2kv_repair_only_key_hashes": repair_keys_by_index[index],
                            }
                        )
                        history_layout_debug.append(
                            {
                                "history_index": index,
                                "mode": "raw_replace",
                                "roles": self._ordered_roles(unit),
                                "logical_token_range": [starts[index], ends[index]],
                                "absolute_rope_position_range": [starts[index], ends[index]],
                                "raw_tokens": raw_len,
                            }
                        )
                else:
                    full_tokens = _token_count(self.tokenizer, unit)
                    stats.effective_history_tokens += full_tokens
                    stats.physical_history_kv_tokens += doc_lens[index]
                    stats.recomputed_raw_tokens += doc_lens[index]
                    local_physical_history_tokens += doc_lens[index]
                    messages.extend(deepcopy(unit))
                    history_layout_debug.append(
                        {
                            "history_index": index,
                            "mode": (
                                "recomputed_raw"
                                if config["operation"] == "recompute"
                                else "full"
                            ),
                            "roles": self._ordered_roles(unit),
                            "logical_token_range": [starts[index], ends[index]],
                            "absolute_rope_position_range": [starts[index], ends[index]],
                            "raw_tokens": doc_lens[index],
                        }
                    )
                gist_records.append(None)
                continue

            full_tokens = len(ids)
            record = self._extract_history_unit(text, stats)
            stats.original_history_tokens += int(record.original_seq_len or full_tokens)
            if not (record.success and record.key_hash):
                raise RuntimeError(f"C2KV extract failed in arm={self.arm}: {record.error}")
            gist_len = int(record.gist_len or record.original_seq_len or full_tokens)
            stats.effective_history_tokens += gist_len
            stats.physical_history_kv_tokens += gist_len
            stats.c2kv_gist_tokens += gist_len
            local_physical_history_tokens += gist_len
            messages.append(
                {"role": "user", "content": text, "c2kv_key_hash": record.key_hash}
            )
            history_layout_debug.append(
                {
                    "history_index": index,
                    "mode": "gist",
                    "roles": self._ordered_roles(unit),
                    "logical_token_range": [starts[index], ends[index]],
                    "physical_kv_tokens": gist_len,
                    "full_equivalent_tokens": int(record.original_seq_len or full_tokens),
                }
            )
            gist_records.append(record)

        append_repair_keys = [
            key_hash
            for index in target_indices
            for key_hash in repair_keys_by_index.get(index, [])
        ]
        if config["operation"] in {"append", "append_masked"} and append_repair_keys:
            target_message = None
            for message in reversed(messages):
                if message.get("c2kv_key_hash"):
                    target_message = message
                    break
            if target_message is None:
                if config["operation"] == "append_masked":
                    messages.insert(
                        0,
                        {
                            "role": "user",
                            "content": "",
                            "c2kv_repair_only_key_hashes": append_repair_keys,
                        },
                    )
                else:
                    raise RuntimeError(f"Cannot attach append repair keys for arm={effective_arm}")
            else:
                target_message["c2kv_repair_key_hashes"] = append_repair_keys
            if config["operation"] == "append":
                stats.effective_history_tokens += repair_tokens
                stats.physical_history_kv_tokens += repair_tokens
                stats.repair_kv_tokens += repair_tokens
                local_physical_history_tokens += repair_tokens

        if config["operation"] == "recompute" and anchor_index + 1 < len(units):
            recomputed_total = sum(doc_lens[anchor_index + 1 :])
            if recomputed_total <= 0:
                raise RuntimeError(
                    f"{effective_arm} expected downstream recompute tokens."
                )

        physical_before = local_physical_history_tokens
        logical_before = canonical_full_history_tokens
        logical_after = canonical_full_history_tokens
        if repair_tokens and logical_before != logical_after:
            raise RuntimeError("repair unexpectedly changed logical current position")

        if config["hint"] and self._repair_enabled_for_current_step:
            messages.append(
                {
                    "role": "system",
                    "content": self._repair_hint(
                        target_metadata=repair_metadata,
                        include_oracle_location=bool(config["oracle_location_hint"]),
                    ),
                }
            )

        self._last_repair_build_info = {
            "repair_mode": effective_arm,
            "repair_extract_source": self._repair_extract_source_for(
                effective_arm,
                str(config["repair_kind"]),
            ),
            "repair_window": config["window"],
            "repair_operation": config["operation"],
            "uses_oracle_error_location": bool(config["oracle_location_hint"]),
            "repair_target_indices": target_indices,
            "repair_target_roles": [
                self._ordered_roles(units[index]) for index in target_indices
            ],
            "repair_target_metadata": repair_metadata,
            "repair_tokens_requested": repair_tokens,
            "repair_tokens_injected": repair_tokens,
            "repair_raw_tokens": repair_tokens,
            "repair_physical_tokens": repair_tokens,
            "repair_absolute_position_ranges": [
                [meta["absolute_position_start"], meta["absolute_position_end"]]
                for meta in repair_metadata
            ],
            "history_layout": history_layout_debug,
            "physical_prefix_len_before": max(0, physical_before - repair_tokens),
            "physical_prefix_len_after": physical_before,
            "logical_position_before": logical_before,
            "logical_position_after": logical_after,
        }
        if repair_tokens != self._last_repair_build_info["repair_tokens_injected"]:
            raise RuntimeError("repair token accounting invariant failed")

        self._last_kv_memory_hint = {
            "full_equivalent_history_tokens": canonical_full_history_tokens,
            "history_scope": "completed_history_only",
            "source": "bfcl_canonical_history_layout",
        }
        messages.extend(current)
        return messages

    def _oracle_repair_arms(self) -> set[str]:
        return {
            "d_sham_mech",
            "hint_only",
            "d_sham_neutral",
            "d_corr",
            "d_corr_w1",
            "d_corr_w2",
            "d_corr_w4",
            "d_corr_w2_hint",
            "d_corr_w2_oracle_location_hint",
            "d_corr_replace_w1",
            "d_corr_replace_w2",
            "d_corr_replace_w4",
            "d_corr_replace_all",
            "append_masked_w2",
            "d_corr_recompute",
            "d_corr_recompute_w2",
            "d_corr_all",
            "raw_all_replace",
        }

    def _should_repair_candidate(
        self,
        *,
        ref_step: dict[str, Any] | None,
        candidate_action: list[str],
        candidate_status: str,
    ) -> bool:
        if self.repair_trigger != "oracle":
            return self.repair_trigger == "always"
        if ref_step is None:
            return False
        reference_action = ref_step.get("decoded_action") or []
        if candidate_status in {"decode_error", "invalid_format", "empty_response"}:
            return bool(reference_action)
        return not action_matches(candidate_action, reference_action)

    def _run_sample_impl(
        self,
        test_case: dict[str, Any],
        stats: DriftStats,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        if self.arm not in self._oracle_repair_arms():
            self._repair_enabled_for_current_step = True
            return super()._run_sample_impl(test_case, stats)

        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        tools = _tool_payload(test_case["function"])
        long_context = "long_context" in test_category or "composite" in test_category

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
        drift_steps: list[dict[str, Any]] = []
        reference_steps = self._reference_steps(test_entry_id)
        reference_map = reference_by_turn_step(reference_steps)
        reference_result = (self.reference_by_id.get(test_entry_id) or {}).get("result") or []
        force_quit = False
        repair_segments: list[dict[str, Any]] = []

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}

            count = 0

            def run_one_step(
                *,
                step_idx: int,
                global_step: int,
                repair_enabled: bool,
                source_info: dict[str, Any] | None = None,
            ) -> tuple[dict[str, Any], bool]:
                nonlocal messages, involved_instances, force_quit

                state_before_execution = _state_log(involved_instances)
                micro_messages = deepcopy(messages)
                ref_step, alignment_status = reference_step_for(
                    reference_map,
                    reference_result,
                    turn_idx,
                    step_idx,
                    fallback_state=state_before_execution,
                )

                self._repair_enabled_for_current_step = repair_enabled
                request_messages = self._build_request_messages(messages, stats)
                repair_build_info = deepcopy(self._last_repair_build_info)
                raw_response: dict[str, Any] = {}
                if (not repair_enabled) and self.collect_candidate_detector_signals:
                    (
                        text,
                        response_message,
                        elapsed,
                        usage,
                        raw_response,
                    ) = self._query_with_raw(
                        request_messages,
                        tools,
                        stats,
                        collect_detector_signals=True,
                    )
                else:
                    text, response_message, elapsed, usage = self._query(
                        request_messages,
                        tools,
                        stats,
                    )
                decoded = decode_candidate(self.decoder, text)

                assistant_history = _assistant_history_message(
                    text,
                    response_message.get("tool_calls"),
                )
                current_turn_response.append(text)
                current_turn_inputs.append(usage["prompt_tokens"])
                current_turn_outputs.append(usage["completion_tokens"])
                current_turn_latency.append(elapsed)

                step_log: list[dict[str, Any]] = [
                    {"role": "assistant", "content": text},
                    {
                        "role": "c2kv_repair_segment",
                        "repair_enabled": repair_enabled,
                        "repair_mode": self.arm if repair_enabled else "c2kv",
                        "repair_target_history_index": self._repair_target_history_index,
                        "repair_build_info": repair_build_info,
                    },
                ]
                turn_log[f"step_{step_idx}"] = step_log
                if decoded.decode_error:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response.",
                            "error": decoded.decode_error,
                            "model_response_decoded": decoded.action,
                        }
                    )
                else:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded.action,
                        }
                    )

                decoded_to_execute = decoded.action
                should_stop_after_record = False
                if (
                    decoded.status
                    in {"decode_error", "invalid_format", "empty_response"}
                    or is_empty_execute_response(decoded_to_execute)
                ):
                    should_stop_after_record = True

                messages.append(deepcopy(assistant_history))
                execution_error = None
                if is_empty_execute_response(decoded_to_execute):
                    execution_results = []
                else:
                    tool_start = time.perf_counter()
                    try:
                        execution_results, involved_instances = execute_multi_turn_func_call(
                            decoded_to_execute,
                            initial_config,
                            involved_classes,
                            self.decoder.model_name_underline_replaced,
                            test_entry_id,
                            long_context=long_context,
                            is_evaL_run=False,
                        )
                        stats.tool_execution_seconds += time.perf_counter() - tool_start
                    except Exception as exc:
                        stats.tool_execution_seconds += time.perf_counter() - tool_start
                        execution_error = str(exc)
                        execution_results = []
                        should_stop_after_record = True
                for idx, execution_result in enumerate(execution_results):
                    messages.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                            "tool_call_id": f"call_{turn_idx}_{step_idx}_{idx}",
                        }
                    )
                    step_log.append({"role": "tool", "content": execution_result})

                state_after_step = _state_log(involved_instances)
                executed_text = _message_text(assistant_history)
                if assistant_history.get("tool_calls"):
                    tool_call_text = _tool_calls_to_text(assistant_history.get("tool_calls"))
                    executed_text = (
                        (executed_text + "\n" + tool_call_text).strip()
                        if executed_text
                        else tool_call_text
                    )
                roundtrip = serialization_roundtrip(
                    self.decoder,
                    executed_text,
                    decoded_to_execute,
                )

                candidate_raw_text = text
                candidate_action = decoded.action
                candidate_status = decoded.status
                candidate_decode_error = decoded.decode_error
                candidate_empty_response = decoded.empty_response
                if source_info is not None:
                    source_record = source_info["step_record"]
                    candidate_raw_text = source_record.get("candidate_raw_text") or ""
                    candidate_action = list(source_record.get("candidate_action") or [])
                    candidate_status = source_record.get("candidate_status") or "empty_action"
                    candidate_decode_error = source_record.get("decode_error")
                    candidate_empty_response = bool(source_record.get("empty_response"))

                step_record = build_step_record(
                    sample_id=test_entry_id,
                    turn_idx=turn_idx,
                    step_idx=step_idx,
                    global_step=global_step,
                    candidate_raw_text=candidate_raw_text,
                    candidate_action=candidate_action,
                    candidate_status=candidate_status,
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=decoded_to_execute,
                    state=state_after_step,
                    decode_error=candidate_decode_error,
                    empty_response=candidate_empty_response,
                    execution_error=execution_error,
                    candidate_assistant_message=(
                        source_info["assistant_history"]
                        if source_info is not None
                        else assistant_history
                    ),
                    executed_assistant_message=assistant_history,
                    execution_results=execution_results,
                    history_execution_results=execution_results,
                    roundtrip=roundtrip,
                    extra={
                        "repair_triggered": repair_enabled,
                        "repair_arm": self.arm,
                        "repair_mode": self.arm if repair_enabled else "c2kv",
                        "repair_target_history_index": self._repair_target_history_index,
                        "repair_build_info": repair_build_info,
                        "repair_raw_text": text if repair_enabled else None,
                        "repair_action": decoded.action if repair_enabled else None,
                        "repair_status": decoded.status if repair_enabled else None,
                    },
                )
                if usage.get("kv_memory_report") is not None:
                    step_record["kv_memory_report"] = usage.get("kv_memory_report")
                if usage.get("kv_runtime_stats") is not None:
                    step_record["kv_runtime_stats"] = usage.get("kv_runtime_stats")
                step_record["oracle_harmful"] = bool(
                    step_record.get("candidate_action_drift")
                    or step_record.get("state_drift")
                )
                if source_info is not None:
                    step_record["plain_c2kv_raw_text"] = candidate_raw_text
                    step_record["plain_c2kv_action"] = candidate_action
                    step_record["plain_c2kv_status"] = candidate_status
                    step_record["repair_changed_action"] = not action_matches(
                        decoded.action,
                        candidate_action,
                    )
                    candidate_ids = self.tokenizer.encode(
                        candidate_raw_text,
                        add_special_tokens=False,
                    )
                    repaired_ids = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                    step_record["candidate_first_token_id"] = (
                        int(candidate_ids[0]) if candidate_ids else None
                    )
                    step_record["repaired_first_token_id"] = (
                        int(repaired_ids[0]) if repaired_ids else None
                    )
                    step_record["repair_changed_first_token"] = (
                        step_record["candidate_first_token_id"]
                        != step_record["repaired_first_token_id"]
                    )
                    step_record["c2kv_wrong_repair_correct"] = bool(
                        step_record.get("candidate_action_drift")
                        and not step_record.get("executed_action_drift")
                        and not step_record.get("state_drift")
                    )
                    step_record["c2kv_wrong_repair_wrong"] = bool(
                        step_record.get("candidate_action_drift")
                        and (
                            step_record.get("executed_action_drift")
                            or step_record.get("state_drift")
                        )
                    )
                    step_record["c2kv_correct_repair_wrong"] = bool(
                        not step_record.get("candidate_action_drift")
                        and (
                            step_record.get("executed_action_drift")
                            or step_record.get("state_drift")
                        )
                    )
                    if self.arm == "d_sham_mech":
                        expected = (candidate_raw_text or "").strip()
                        actual = (text or "").strip()
                        expected_ids = self.tokenizer.encode(
                            expected,
                            add_special_tokens=False,
                        )
                        actual_ids = self.tokenizer.encode(
                            actual,
                            add_special_tokens=False,
                        )
                        if expected_ids != actual_ids:
                            raise RuntimeError(
                                "d_sham_mech changed generated token ids relative to "
                                "plain C2KV while repair plumbing should be a no-op."
                            )

                if alignment_status == "missing_reference":
                    stats.errors.append(
                        f"missing reference action at turn={turn_idx}, "
                        f"step={step_idx}, candidate_global_step={global_step}"
                    )
                if roundtrip["serialization_mismatch"]:
                    stats.errors.append(
                        f"serialization mismatch at turn={turn_idx}, "
                        f"step={step_idx}, candidate_global_step={global_step}"
                    )

                if should_stop_after_record:
                    return (
                        {
                            "step_record": step_record,
                            "assistant_history": assistant_history,
                            "text": text,
                            "usage": usage,
                            "elapsed": elapsed,
                            "candidate_detector_features": (
                                self._candidate_detector_features(raw_response)
                                if raw_response
                                else {}
                            ),
                            "micro_snapshot": {"messages": micro_messages},
                            "terminal": True,
                        },
                        True,
                    )
                if step_idx + 1 > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": (
                                f"Model has been forced to quit after "
                                f"{MAXIMUM_STEP_LIMIT} steps."
                            ),
                        }
                    )
                    return (
                        {
                            "step_record": step_record,
                            "assistant_history": assistant_history,
                            "text": text,
                            "usage": usage,
                            "elapsed": elapsed,
                            "candidate_detector_features": (
                                self._candidate_detector_features(raw_response)
                                if raw_response
                                else {}
                            ),
                            "micro_snapshot": {"messages": micro_messages},
                            "terminal": True,
                        },
                        True,
                    )
                return (
                    {
                        "step_record": step_record,
                        "assistant_history": assistant_history,
                        "text": text,
                        "usage": usage,
                        "elapsed": elapsed,
                        "candidate_detector_features": (
                            self._candidate_detector_features(raw_response)
                            if raw_response
                            else {}
                        ),
                        "micro_snapshot": {"messages": micro_messages},
                        "terminal": False,
                    },
                    False,
                )

            while True:
                segment_checkpoint = self._snapshot(
                    messages=messages,
                    involved_instances=involved_instances,
                    current_turn_response=current_turn_response,
                    current_turn_inputs=current_turn_inputs,
                    current_turn_outputs=current_turn_outputs,
                    current_turn_latency=current_turn_latency,
                    turn_log=turn_log,
                    global_step=len(drift_steps),
                )
                segment_start_count = count
                segment_infos: list[dict[str, Any]] = []
                speculative_terminal = False
                for _ in range(self.checkpoint_interval):
                    info, terminal = run_one_step(
                        step_idx=count,
                        global_step=len(drift_steps) + len(segment_infos),
                        repair_enabled=False,
                    )
                    info["heuristic_attributes"] = self._heuristic_attributes(
                        info=info,
                        segment_infos=segment_infos,
                    )
                    segment_infos.append(info)
                    if terminal:
                        speculative_terminal = True
                        break
                    count += 1

                if not segment_infos:
                    break

                oracle_segment_harmful = any(
                    bool(info["step_record"].get("oracle_harmful"))
                    for info in segment_infos
                )
                detector_debug = self._segment_detector(segment_infos)
                repair_triggered = bool(detector_debug.get("detector_trigger"))
                if self.repair_trigger == "always":
                    repair_triggered = True
                    detector_debug = {
                        **detector_debug,
                        "detector": "always_trigger",
                        "detector_trigger": True,
                        "detector_reason": "legacy_repair_trigger_always",
                    }
                detector_confusion = self._detector_confusion(
                    oracle_segment_harmful=oracle_segment_harmful,
                    detector_trigger=repair_triggered,
                )
                harmful_step_indices = [
                    idx
                    for idx, info in enumerate(segment_infos)
                    if bool(info["step_record"].get("oracle_harmful"))
                ]
                has_action_drift = any(
                    bool(info["step_record"].get("candidate_action_drift"))
                    for info in segment_infos
                )
                has_state_drift = any(
                    bool(info["step_record"].get("state_drift"))
                    for info in segment_infos
                )
                if has_action_drift and has_state_drift:
                    harmful_reason = "both"
                elif has_action_drift:
                    harmful_reason = "action_drift"
                elif has_state_drift:
                    harmful_reason = "state_drift"
                else:
                    harmful_reason = "none"
                repair_segment = {
                    "sample_id": test_entry_id,
                    "turn": turn_idx,
                    "segment_start_step": segment_start_count,
                    "segment_length": len(segment_infos),
                    "checkpoint_interval": self.checkpoint_interval,
                    "detector_trigger": repair_triggered,
                    "detector_arm": self.detector_arm,
                    "detector": detector_debug.get("detector"),
                    "detector_reason": detector_debug.get("detector_reason"),
                    "detector_threshold": detector_debug.get("detector_threshold"),
                    "detector_signal_name": detector_debug.get("detector_signal_name"),
                    "detector_signal_score": detector_debug.get("detector_signal_score"),
                    "detector_signal_threshold": detector_debug.get(
                        "detector_signal_threshold"
                    ),
                    "rule_detector_trigger": detector_debug.get(
                        "rule_detector_trigger"
                    ),
                    "rule_detector_max_risk": detector_debug.get(
                        "rule_detector_max_risk"
                    ),
                    "rule_detector_reason": detector_debug.get(
                        "rule_detector_reason"
                    ),
                    "logistic_detector_score": detector_debug.get(
                        "logistic_detector_score"
                    ),
                    "logistic_detector_threshold": detector_debug.get(
                        "logistic_detector_threshold"
                    ),
                    "threshold_selection_rule": detector_debug.get(
                        "threshold_selection_rule"
                    ),
                    "detector_evaluation_mode": detector_debug.get(
                        "detector_evaluation_mode"
                    ),
                    "detector_tp": detector_confusion["detector_tp"],
                    "detector_fp": detector_confusion["detector_fp"],
                    "detector_tn": detector_confusion["detector_tn"],
                    "detector_fn": detector_confusion["detector_fn"],
                    "oracle_segment_harmful": oracle_segment_harmful,
                    "oracle_reference_drift_segment": oracle_segment_harmful,
                    "oracle_harmful_reason": harmful_reason,
                    "oracle_reference_drift_reason": harmful_reason,
                    "harmful_step_indices": harmful_step_indices,
                    "repair_triggered": repair_triggered,
                    "repair_trigger_policy": self.repair_trigger,
                    "repair_mode": self.arm if repair_triggered else "c2kv",
                    "repair_target_history_index": segment_checkpoint.get(
                        "repair_target_history_index"
                    ),
                    "candidate_action_drift_per_step": [
                        bool(info["step_record"].get("candidate_action_drift"))
                        for info in segment_infos
                    ],
                    "state_drift_per_step": [
                        bool(info["step_record"].get("state_drift"))
                        for info in segment_infos
                    ],
                    "oracle_harmful_drift_per_step": [
                        bool(info["step_record"].get("oracle_harmful"))
                        for info in segment_infos
                    ],
                    "candidate_detector_features_per_step": [
                        info.get("candidate_detector_features") or {}
                        for info in segment_infos
                    ],
                    "heuristic_attributes_per_step": [
                        info.get("heuristic_attributes") or {}
                        for info in segment_infos
                    ],
                    "speculative_terminal_discarded": False,
                    "repair_segment_success": None,
                    "c2kv_wrong_repair_correct": 0,
                    "c2kv_wrong_repair_wrong": 0,
                    "c2kv_correct_repair_wrong": 0,
                    "repair_changed_action_count": 0,
                    "repair_changed_first_token_count": 0,
                    "candidate_action_correct": not has_action_drift,
                    "candidate_state_correct": not has_state_drift,
                    "repaired_action_correct": None,
                    "repaired_state_correct": None,
                    "segment_start_state_matches_reference": (
                        not any(bool(row.get("state_drift")) for row in drift_steps)
                    ),
                }

                if not repair_triggered:
                    for info in segment_infos:
                        step_record = info["step_record"]
                        mark_first_divergence(stats, step_record)
                        drift_steps.append(step_record)
                    repair_segments.append(repair_segment)
                    if speculative_terminal or force_quit:
                        break
                    continue

                repair_segment["speculative_terminal_discarded"] = speculative_terminal
                self._active_oracle_bad_step = (
                    harmful_step_indices[0] if harmful_step_indices else None
                )
                (
                    messages,
                    involved_instances,
                    current_turn_response,
                    current_turn_inputs,
                    current_turn_outputs,
                    current_turn_latency,
                    turn_log,
                ) = self._restore_snapshot(
                    test_entry_id=test_entry_id,
                    snapshot=segment_checkpoint,
                )
                count = segment_start_count
                repaired_records: list[dict[str, Any]] = []
                repair_terminal = False
                for source_info in segment_infos:
                    info, terminal = run_one_step(
                        step_idx=count,
                        global_step=len(drift_steps),
                        repair_enabled=True,
                        source_info=source_info,
                    )
                    step_record = info["step_record"]
                    step_record["oracle_segment_harmful"] = oracle_segment_harmful
                    step_record["detector_trigger"] = repair_triggered
                    step_record["repair_triggered"] = True
                    mark_first_divergence(stats, step_record)
                    drift_steps.append(step_record)
                    repaired_records.append(step_record)
                    repair_segment["c2kv_wrong_repair_correct"] += int(
                        bool(step_record.get("c2kv_wrong_repair_correct"))
                    )
                    repair_segment["c2kv_wrong_repair_wrong"] += int(
                        bool(step_record.get("c2kv_wrong_repair_wrong"))
                    )
                    repair_segment["c2kv_correct_repair_wrong"] += int(
                        bool(step_record.get("c2kv_correct_repair_wrong"))
                    )
                    repair_segment["repair_changed_action_count"] += int(
                        bool(step_record.get("repair_changed_action"))
                    )
                    repair_segment["repair_changed_first_token_count"] += int(
                        bool(step_record.get("repair_changed_first_token"))
                    )
                    if step_record.get("repair_build_info"):
                        build_info = dict(step_record["repair_build_info"])
                        for key, value in build_info.items():
                            repair_segment.setdefault(key, value)
                    if terminal:
                        repair_terminal = True
                        break
                    count += 1
                repair_segment["repair_segment_success"] = bool(
                    repaired_records
                    and all(
                        not row.get("executed_action_drift")
                        and not row.get("state_drift")
                        for row in repaired_records
                    )
                )
                repair_segment["reference_recovery_success"] = repair_segment[
                    "repair_segment_success"
                ]
                repair_segment["repaired_action_correct"] = bool(
                    repaired_records
                    and all(not row.get("executed_action_drift") for row in repaired_records)
                )
                repair_segment["repaired_state_correct"] = bool(
                    repaired_records
                    and all(not row.get("state_drift") for row in repaired_records)
                )
                repair_segment["tp_recovery_attempt"] = bool(
                    detector_confusion["detector_tp"]
                )
                repair_segment["tp_recovery_success"] = bool(
                    detector_confusion["detector_tp"]
                    and repair_segment["repair_segment_success"]
                )
                repair_segment["fp_recovery_harm"] = bool(
                    detector_confusion["detector_fp"]
                    and (
                        repair_segment["repaired_action_correct"] is False
                        or repair_segment["repaired_state_correct"] is False
                    )
                )
                repair_segment["fp_recovery_still_correct"] = bool(
                    detector_confusion["detector_fp"]
                    and repair_segment["repaired_action_correct"] is True
                    and repair_segment["repaired_state_correct"] is True
                )
                repair_segments.append(repair_segment)
                self._active_oracle_bad_step = None
                if repair_terminal or force_quit:
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
            "repair_segments": repair_segments,
        }
        return all_model_response, metadata


def run(args: argparse.Namespace) -> None:
    runner = KVRepairRunner(args)
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
    metric_rows = []
    for test_case in tqdm(entries, desc=f"kv_repair:{args.arm}", dynamic_ncols=True):
        row = runner.run_sample(deepcopy(test_case))
        row["kv_repair_arm"] = args.arm
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metric_rows.append(row.get("c2kv_drift_metrics", {}))

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metric_rows)
    summary = {
        "arm": args.arm,
        "detector_arm": args.detector_arm,
        "rule_detector_threshold": args.rule_detector_threshold,
        "detector_signal_threshold": args.detector_signal_threshold,
        "logistic_detector_features_csv": args.logistic_detector_features_csv,
        "logistic_detector_kfolds": args.logistic_detector_kfolds,
        "logistic_detector_feature_set": args.logistic_detector_feature_set,
        "request_candidate_logprobs": args.request_candidate_logprobs,
        "category": args.category,
        "num_examples": len(details_rows),
        "errors": sum(
            1
            for row in details_rows
            if str(row.get("result", "")).startswith("Error during inference")
        ),
        "chat_calls": sum(int(row.get("chat_calls") or 0) for row in metric_rows),
        "extract_calls": sum(int(row.get("extract_calls") or 0) for row in metric_rows),
        "extract_success": sum(int(row.get("extract_success") or 0) for row in metric_rows),
        "chat_seconds": sum(float(row.get("chat_seconds") or 0.0) for row in metric_rows),
        "extract_seconds": sum(float(row.get("extract_seconds") or 0.0) for row in metric_rows),
        "c2kv_extract_seconds": sum(float(row.get("c2kv_extract_seconds") or 0.0) for row in metric_rows),
        "repair_extract_seconds": sum(float(row.get("repair_extract_seconds") or 0.0) for row in metric_rows),
        "tool_execution_seconds": sum(float(row.get("tool_execution_seconds") or 0.0) for row in metric_rows),
        "episode_e2e_observed_seconds": sum(
            float(row.get("episode_e2e_observed_seconds") or 0.0)
            for row in metric_rows
        ),
        "chat_prompt_tokens": sum(
            int(row.get("chat_prompt_tokens") or 0) for row in metric_rows
        ),
        "chat_cached_tokens": sum(
            int(row.get("chat_cached_tokens") or 0) for row in metric_rows
        ),
        "chat_recomputed_prompt_tokens": sum(
            int(row.get("chat_recomputed_prompt_tokens") or 0)
            for row in metric_rows
        ),
        "chat_cache_report_missing": sum(
            int(row.get("chat_cache_report_missing") or 0) for row in metric_rows
        ),
        "chat_completion_tokens": sum(
            int(row.get("chat_completion_tokens") or 0) for row in metric_rows
        ),
        "kv_peak_resident_tokens": max(
            [int(row.get("kv_peak_resident_tokens") or 0) for row in metric_rows]
            or [0]
        ),
        "kv_runtime_report_missing": sum(
            int(row.get("kv_runtime_report_missing") or 0) for row in metric_rows
        ),
        "c2kv_extract_recomputed_tokens": sum(
            int(row.get("c2kv_extract_recomputed_tokens") or 0)
            for row in metric_rows
        ),
        "repair_extract_recomputed_tokens": sum(
            int(row.get("repair_extract_recomputed_tokens") or 0)
            for row in metric_rows
        ),
        "total_actual_recomputed_tokens": sum(
            int(row.get("total_actual_recomputed_tokens") or 0)
            for row in metric_rows
        ),
        "history_original_tokens": sum(
            int(row.get("history_original_tokens") or 0) for row in metric_rows
        ),
        "history_effective_tokens": sum(
            int(row.get("history_effective_tokens") or 0) for row in metric_rows
        ),
        "canonical_full_history_tokens": sum(
            int(row.get("canonical_full_history_tokens") or 0) for row in metric_rows
        ),
        "physical_history_kv_tokens": sum(
            int(row.get("physical_history_kv_tokens") or 0) for row in metric_rows
        ),
        "c2kv_gist_tokens": sum(
            int(row.get("c2kv_gist_tokens") or 0) for row in metric_rows
        ),
        "repair_kv_tokens": sum(
            int(row.get("repair_kv_tokens") or 0) for row in metric_rows
        ),
        "recomputed_raw_tokens": sum(
            int(row.get("recomputed_raw_tokens") or 0) for row in metric_rows
        ),
        "repaired_step_count": sum(
            int(row.get("repaired_step_count") or 0) for row in metric_rows
        ),
        "repair_changed_action_count": sum(
            int(row.get("repair_changed_action_count") or 0) for row in metric_rows
        ),
        "repair_changed_first_token_count": sum(
            int(row.get("repair_changed_first_token_count") or 0)
            for row in metric_rows
        ),
        "net_repair_gain": sum(
            int(row.get("net_repair_gain") or 0) for row in metric_rows
        ),
        "detector_tp": sum(int(row.get("detector_tp") or 0) for row in metric_rows),
        "detector_fp": sum(int(row.get("detector_fp") or 0) for row in metric_rows),
        "detector_tn": sum(int(row.get("detector_tn") or 0) for row in metric_rows),
        "detector_fn": sum(int(row.get("detector_fn") or 0) for row in metric_rows),
        "tp_recovery_attempts": sum(
            int(row.get("tp_recovery_attempts") or 0) for row in metric_rows
        ),
        "tp_recovery_success_count": sum(
            int(row.get("tp_recovery_success_count") or 0) for row in metric_rows
        ),
        "fp_recovery_count": sum(
            int(row.get("fp_recovery_count") or 0) for row in metric_rows
        ),
        "fp_recovery_harm_count": sum(
            int(row.get("fp_recovery_harm_count") or 0) for row in metric_rows
        ),
        "false_negative_count": sum(
            int(row.get("false_negative_count") or 0) for row in metric_rows
        ),
    }
    summary["extract_success_rate"] = (
        summary["extract_success"] / summary["extract_calls"]
        if summary["extract_calls"]
        else None
    )
    summary["history_kv_compression"] = (
        summary["canonical_full_history_tokens"] / summary["physical_history_kv_tokens"]
        if summary["physical_history_kv_tokens"]
        else None
    )
    summary["avg_episode_e2e_observed_seconds"] = (
        summary["episode_e2e_observed_seconds"] / summary["num_examples"]
        if summary["num_examples"]
        else None
    )
    total_pred_pos = summary["detector_tp"] + summary["detector_fp"]
    total_actual_pos = summary["detector_tp"] + summary["detector_fn"]
    total_actual_neg = summary["detector_fp"] + summary["detector_tn"]
    precision = (
        summary["detector_tp"] / total_pred_pos if total_pred_pos else None
    )
    recall = summary["detector_tp"] / total_actual_pos if total_actual_pos else None
    summary["detector_precision"] = precision
    summary["detector_recall"] = recall
    summary["detector_f1"] = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    summary["detector_fpr"] = (
        summary["detector_fp"] / total_actual_neg if total_actual_neg else None
    )
    summary["tp_recovery_success_rate"] = (
        summary["tp_recovery_success_count"] / summary["tp_recovery_attempts"]
        if summary["tp_recovery_attempts"]
        else None
    )
    summary["fp_recovery_harm_rate"] = (
        summary["fp_recovery_harm_count"] / summary["fp_recovery_count"]
        if summary["fp_recovery_count"]
        else None
    )
    summary["false_negative_rate"] = (
        summary["false_negative_count"] / total_actual_pos
        if total_actual_pos
        else None
    )
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(REPAIR_ARMS), required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--reference-details-path", default="")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--recent-full-units", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=4,
        help="Number of plain C2KV speculative steps before Oracle segment repair.",
    )
    parser.add_argument(
        "--repair-window",
        default="1",
        choices=["1", "2", "4", "all"],
        help="Recent-W history units to restore for generic repair arms.",
    )
    parser.add_argument(
        "--repair-extract-source",
        default="auto",
        choices=["auto", "model_prefill", "serving_cache"],
        help=(
            "Raw repair KV source. auto keeps the stable model_prefill path "
            "except raw_all_replace_direct, which extracts from the normal "
            "serving radix/KV cache after a /generate warmup. Use "
            "serving_cache explicitly to debug serving-cache raw copy for all "
            "raw repair arms."
        ),
    )
    parser.add_argument("--plan-path", default="")
    parser.add_argument(
        "--repair-trigger",
        choices=["oracle", "always"],
        default="oracle",
        help=(
            "oracle: first query plain C2KV and build repair KV only when the "
            "frozen Full reference says the candidate action drifted. always: "
            "apply the selected repair arm on every step."
        ),
    )
    parser.add_argument(
        "--detector-arm",
        choices=sorted(DETECTOR_ARMS),
        default="oracle",
        help=(
            "Closed-loop segment detector. All detector arms share the same "
            "Replace-W recovery path; oracle labels are still logged for "
            "offline TP/FP/FN accounting."
        ),
    )
    parser.add_argument("--rule-detector-threshold", type=float, default=5.0)
    parser.add_argument(
        "--detector-signal-threshold",
        type=float,
        default=5.0,
        help="Fallback threshold for scalar detector arms after direction normalization.",
    )
    parser.add_argument(
        "--detector-thresholds-json",
        default="",
        help=(
            "Optional fold-wise scalar detector threshold file generated by "
            "prepare_detector_online_5fold.py."
        ),
    )
    parser.add_argument("--candidate-logprobs-top-k", type=int, default=20)
    parser.add_argument(
        "--collect-candidate-detector-signals",
        action="store_true",
        help=(
            "Record cheap detector signals for each candidate step. This no "
            "longer requests token logprobs unless --request-candidate-logprobs "
            "is also set."
        ),
    )
    parser.add_argument(
        "--request-candidate-logprobs",
        action="store_true",
        help=(
            "Ask SGLang to return generation logprobs for detector features. "
            "Keep disabled on current C2KV NPU multi-round path unless that "
            "server path has been validated."
        ),
    )
    parser.add_argument(
        "--logistic-detector-features-csv",
        default="",
        help="Episode-split detector_features.csv for combined logistic detector.",
    )
    parser.add_argument(
        "--logistic-detector-kfolds",
        type=int,
        default=0,
        help=(
            "If >1, train episode-level cross-fit logistic detector models. "
            "Each episode is scored by a model trained on all other folds."
        ),
    )
    parser.add_argument(
        "--logistic-detector-feature-set",
        choices=["auto", "online_safe", "all"],
        default="auto",
        help=(
            "auto uses online-safe non-logprob features unless "
            "--request-candidate-logprobs is set."
        ),
    )
    parser.add_argument(
        "--detector-cv-output-dir",
        default="",
        help="Directory where k-fold logistic model/scaler/threshold artifacts are saved.",
    )
    parser.add_argument(
        "--neutral-corpus-path",
        default="/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_neutral_corpus.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
