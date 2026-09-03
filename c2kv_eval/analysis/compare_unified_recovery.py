from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MIB = 1024 * 1024


DEFAULT_METHODS = (
    ("Full", "full"),
    ("C2KV", "c2kv"),
    ("Rollback D1", "rollback_d1"),
    ("Rollback D2", "rollback_d2"),
    ("Rollback D4", "rollback_d4"),
    ("Replace W1", "replace_w1"),
    ("Replace W2", "replace_w2"),
    ("Replace W4", "replace_w4"),
    ("Replace All", "replace_all"),
    ("Recompute W2", "recompute_w2"),
    ("Append W2", "append_w2"),
    ("Append W2 + Hint", "append_w2_hint"),
    ("Hint Only", "hint_only"),
    ("Append-Masked W2", "append_masked_w2"),
    ("Sham Mech", "sham_mech"),
)

DISPLAY_NAME_BY_DIR = {dirname: label for label, dirname in DEFAULT_METHODS}


CSV_FIELDS = [
    "method",
    "run_dir",
    "bfcl_accuracy",
    "correct_count",
    "num_examples",
    "turn_joint_pass_rate",
    "episode_candidate_action_drift_rate",
    "episode_executed_action_drift_rate",
    "episode_state_drift_rate",
    "step_candidate_action_drift_rate",
    "step_executed_action_drift_rate",
    "step_state_drift_rate",
    "candidate_action_drift_rate",
    "executed_action_drift_rate",
    "state_drift_rate",
    "detector",
    "oracle_harmful_segments",
    "oracle_reference_drift_segments",
    "detector_trigger_count",
    "detector_trigger_rate",
    "recovery_attempt_count",
    "recovery_success_count",
    "recovery_success_rate",
    "reference_recovery_success_rate",
    "total_committed_steps",
    "avg_committed_steps_per_episode",
    "total_candidate_steps",
    "avg_candidate_steps_per_episode",
    "total_regenerated_steps",
    "avg_regenerated_steps_per_episode",
    "total_model_generation_calls",
    "model_calls_per_committed_step",
    "chat_calls",
    "action_wrong_to_correct",
    "action_wrong_to_wrong",
    "action_correct_to_wrong",
    "action_correct_to_correct",
    "action_net_recovery_gain",
    "state_wrong_to_correct",
    "state_wrong_to_wrong",
    "state_correct_to_wrong",
    "state_correct_to_correct",
    "state_net_recovery_gain",
    "avg_full_equivalent_history_kv_tokens_per_step",
    "avg_active_gpu_history_kv_tokens_per_step",
    "online_mean_step_gpu_kv_compression",
    "online_weighted_gpu_history_kv_compression",
    "online_worst_step_gpu_kv_compression",
    "avg_full_history_kv_tokens_per_step",
    "avg_active_history_kv_tokens_per_step",
    "history_kv_compression",
    "mean_step_history_kv_compression",
    "median_step_history_kv_compression",
    "worst_step_history_kv_compression",
    "peak_gpu_kv_mib",
    "peak_main_kv_mib",
    "peak_c2kv_pool_mib",
    "peak_host_checkpoint_kv_mib",
    "actual_weighted_history_kv_compression",
    "actual_mean_step_history_kv_compression",
    "actual_median_step_history_kv_compression",
    "actual_worst_step_history_kv_compression",
    "memory_report_coverage",
    "actual_compression_status",
    "estimated_avg_full_history_tokens_per_step",
    "estimated_avg_active_history_tokens_per_step",
    "estimated_online_weighted_gpu_history_kv_compression",
    "estimated_online_mean_step_gpu_kv_compression",
    "estimated_online_median_step_gpu_kv_compression",
    "estimated_online_worst_step_gpu_kv_compression",
    "estimated_weighted_history_kv_compression",
    "reference_aligned_mean_step_compression",
    "reference_aligned_weighted_compression",
    "reference_aligned_worst_step_compression",
    "reference_aligned_num_steps",
    "reference_aligned_status",
    "host_checkpoint_kv_tokens",
    "avg_host_checkpoint_kv_tokens",
    "peak_host_checkpoint_kv_tokens",
    "avg_host_checkpoint_kv_mib",
    "cumulative_host_checkpoint_token_volume",
    "avg_resident_host_checkpoint_kv_tokens",
    "peak_resident_host_checkpoint_kv_tokens",
    "avg_resident_host_checkpoint_kv_mib",
    "peak_resident_host_checkpoint_kv_mib",
    "host_raw_bank_kv_tokens",
    "peak_host_kv_tokens",
    "raw_kv_bank_implemented",
    "runtime_memory_report_steps",
    "runtime_memory_missing_steps",
    "bytes_per_kv_token",
    "schema_version",
]

V3_CSV_FIELDS = [
    "method",
    "bfcl_accuracy",
    "turn_joint_pass_rate",
    "episode_candidate_action_drift_rate",
    "episode_executed_action_drift_rate",
    "episode_state_drift_rate",
    "step_candidate_action_drift_rate",
    "step_executed_action_drift_rate",
    "step_state_drift_rate",
    "detector",
    "detector_trigger_rate",
    "reference_recovery_success_rate",
    "total_committed_steps",
    "total_candidate_steps",
    "total_regenerated_steps",
    "total_model_generation_calls",
    "model_calls_per_committed_step",
    "avg_full_history_kv_tokens_per_step",
    "avg_active_history_kv_tokens_per_step",
    "history_kv_compression",
    "mean_step_history_kv_compression",
    "worst_step_history_kv_compression",
    "peak_gpu_kv_mib",
    "peak_main_kv_mib",
    "peak_c2kv_pool_mib",
    "peak_host_checkpoint_kv_mib",
    "actual_weighted_history_kv_compression",
    "actual_mean_step_history_kv_compression",
    "actual_worst_step_history_kv_compression",
    "memory_report_coverage",
    "actual_compression_status",
    "estimated_weighted_history_kv_compression",
    "avg_resident_host_checkpoint_kv_tokens",
    "peak_resident_host_checkpoint_kv_tokens",
    "avg_resident_host_checkpoint_kv_mib",
    "peak_resident_host_checkpoint_kv_mib",
    "raw_kv_bank_implemented",
    "schema_version",
]

QUICK_CSV_FIELDS = [
    "Method",
    "BFCL Acc",
    "Turn Joint",
    "Step Executed Action Drift",
    "Step State Drift",
    "Detector Trigger Rate",
    "Recovery Success Rate",
    "Avg Committed Steps / Episode",
    "Regenerated Steps",
    "Model Calls / Committed Step",
    "Avg Active GPU KV Tokens / Step",
    "History KV Compression",
    "Mean Step History KV Compression",
    "Worst Step History KV Compression",
    "Peak GPU KV MiB",
    "Peak Main KV MiB",
    "Peak C2KV Pool MiB",
    "Memory Report Coverage",
    "Peak Host Checkpoint KV MiB",
]

MINIMAL_CSV_FIELDS = [
    "Method",
    "BFCL Acc",
    "Reference Recovery Success",
    "Step Exec Drift",
    "Step State Drift",
    "Model Calls / Committed Step",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_rate(num: float | int | None, den: float | int | None) -> float | None:
    if num is None:
        return None
    return num / den if den else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _find_score(root: Path, category: str) -> dict[str, Any]:
    csv_path = root / "score" / "data_multi_turn.csv"
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            row = rows[0]
            value = row.get("Base") or row.get("Multi Turn Overall Acc")
            if isinstance(value, str) and value.endswith("%"):
                return {"accuracy": float(value[:-1]) / 100.0}
    matches = sorted((root / "score").rglob("*_score.json"))
    for path in matches:
        rows = _load_jsonl(path)
        if rows:
            return rows[0]
    return {}


def _score_row(root: Path, category: str, details: list[dict[str, Any]]) -> tuple[float | None, int | None, int]:
    score = _find_score(root, category)
    total = len(details)
    if not total:
        summary = _load_json(root / "logs" / "summary.json")
        total = _int(summary.get("num_examples") or summary.get("total_samples"))
    accuracy = _num(score.get("accuracy"))
    correct = score.get("correct_count")
    if correct is None and accuracy is not None and total:
        correct = round(accuracy * total)
    return accuracy, (int(correct) if correct is not None else None), total


def _flatten_steps(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for row in details:
        sample_id = str(row.get("id"))
        for step in row.get("drift_steps") or []:
            if isinstance(step, dict):
                step = dict(step)
                step.setdefault("id", sample_id)
                steps.append(step)
    return steps


def _episode_rate(details: list[dict[str, Any]], key: str, false_key: str | None = None) -> float | None:
    total = len(details)
    if not total:
        return None
    bad = 0
    for row in details:
        steps = row.get("drift_steps") or []
        if any(
            isinstance(step, dict)
            and (step.get(key) is True or (false_key and step.get(false_key) is False))
            for step in steps
        ):
            bad += 1
    return bad / total


def _step_rate(steps: list[dict[str, Any]], key: str, false_key: str | None = None) -> float | None:
    if not steps:
        return None
    bad = 0
    for step in steps:
        if step.get(key) is True or (false_key and step.get(false_key) is False):
            bad += 1
    return bad / len(steps)


def _turn_joint(details: list[dict[str, Any]]) -> float | None:
    total = 0
    passed = 0
    for row in details:
        by_turn: dict[int, list[dict[str, Any]]] = {}
        for step in row.get("drift_steps") or []:
            if isinstance(step, dict):
                by_turn.setdefault(_int(step.get("turn")), []).append(step)
        for steps in by_turn.values():
            total += 1
            if all(
                step.get("executed_action_drift") is not True
                and step.get("executed_action_matches_reference") is not False
                and step.get("state_drift") is not True
                and step.get("state_matches_reference") is not False
                for step in steps
            ):
                passed += 1
    return _safe_rate(passed, total)


def _latest_metrics(details: list[dict[str, Any]]) -> dict[str, int | float | None]:
    keys = [
        "chat_calls",
        "canonical_full_history_tokens",
        "physical_history_kv_tokens",
        "c2kv_gist_tokens",
        "repair_kv_tokens",
        "kv_runtime_report_missing",
        "kv_peak_resident_tokens",
        "checkpoint_host_tokens",
        "checkpoint_device_tokens",
        "peak_checkpoint_host_tokens",
    ]
    out: dict[str, int | float] = {}
    for row in details:
        metrics = (
            row.get("c2kv_drift_metrics")
            or row.get("c2kv_checkpoint_metrics")
            or row.get("checkpoint_metrics")
            or {}
        )
        if not isinstance(metrics, dict):
            continue
        for key in keys:
            value = _num(metrics.get(key))
            if value is not None:
                out[key] = out.get(key, 0) + value
    return {key: out.get(key) for key in keys}


def _checkpoint_logs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    logs = root / "logs"
    return (
        _load_jsonl(logs / "checkpoint_steps.jsonl"),
        _load_jsonl(logs / "checkpoint_segments.jsonl"),
        _load_json(logs / "run_summary.json"),
    )


def _segment_value(segment: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in segment:
            return segment.get(key)
    return None


def _segments_from_repair(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for row in details:
        sample_id = str(row.get("id"))
        for segment in row.get("repair_segments") or []:
            if isinstance(segment, dict):
                item = dict(segment)
                item.setdefault("id", sample_id)
                segments.append(item)
    return segments


def _memory_from_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    actual_pairs: list[tuple[float, float]] = []
    estimated_pairs: list[tuple[float, float]] = []
    actual_report_count = 0
    missing = 0
    peak_gpu_bytes: float | None = None
    peak_main_bytes: float | None = None
    peak_c2kv_bytes: float | None = None
    bytes_per_kv_token: float | None = None
    for step in steps:
        report = step.get("kv_memory_report")
        if isinstance(report, dict):
            actual_report_count += 1
            full = _num(report.get("full_equivalent_history_tokens"))
            active = _num(report.get("active_history_kv_tokens"))
            if full is not None and active is not None and active >= 0:
                actual_pairs.append((full, active))
            else:
                missing += 1
            bytes_per_kv_token = bytes_per_kv_token or _num(
                report.get("bytes_per_kv_token")
            )

            current_total = _num(report.get("total_gpu_kv_bytes"))
            peak_total = _num(report.get("peak_total_gpu_kv_bytes"))
            total_value = peak_total if peak_total is not None else current_total
            if total_value is not None:
                peak_gpu_bytes = max(peak_gpu_bytes or 0.0, total_value)

            current_main = _num(report.get("physical_main_kv_bytes"))
            peak_main_slots = _num(report.get("peak_main_paged_kv_slots"))
            main_value = current_main
            if peak_main_slots is not None and bytes_per_kv_token is not None:
                main_value = max(main_value or 0.0, peak_main_slots * bytes_per_kv_token)
            if main_value is not None:
                peak_main_bytes = max(peak_main_bytes or 0.0, main_value)

            current_c2kv = _num(report.get("physical_c2kv_pool_bytes"))
            peak_c2kv_slots = _num(report.get("peak_c2kv_pool_slots"))
            c2kv_value = current_c2kv
            if peak_c2kv_slots is not None and bytes_per_kv_token is not None:
                c2kv_value = max(c2kv_value or 0.0, peak_c2kv_slots * bytes_per_kv_token)
            if c2kv_value is not None:
                peak_c2kv_bytes = max(peak_c2kv_bytes or 0.0, c2kv_value)
        else:
            missing += 1

        repair_info = step.get("repair_build_info")
        if isinstance(repair_info, dict):
            full = _num(repair_info.get("logical_position_before"))
            active = _num(repair_info.get("physical_prefix_len_after"))
            if full is not None and active and active > 0:
                estimated_pairs.append((full, active))
    total = len(steps)

    def summarize_pairs(pairs: list[tuple[float, float]]) -> dict[str, Any]:
        ratios = [
            full / active
            for full, active in pairs
            if full is not None and full > 0 and active is not None and active > 0
        ]
        sum_full = sum(full for full, _ in pairs)
        sum_active = sum(active for _, active in pairs)
        return {
            "avg_full": _safe_rate(sum_full, len(pairs)),
            "avg_active": _safe_rate(sum_active, len(pairs)),
            "weighted": _safe_rate(sum_full, sum_active),
            "mean": _mean(ratios),
            "median": _median(ratios),
            "worst": min(ratios) if ratios else None,
        }

    actual = summarize_pairs(actual_pairs)
    estimated = summarize_pairs(estimated_pairs)
    coverage = _safe_rate(actual_report_count, total)
    status = "complete" if total and actual_report_count == total and missing == 0 else "incomplete"
    return {
        "avg_full_equivalent_history_kv_tokens_per_step": actual["avg_full"],
        "avg_active_gpu_history_kv_tokens_per_step": actual["avg_active"],
        "online_mean_step_gpu_kv_compression": actual["mean"],
        "online_weighted_gpu_history_kv_compression": actual["weighted"],
        "online_worst_step_gpu_kv_compression": actual["worst"],
        "avg_full_history_kv_tokens_per_step": actual["avg_full"],
        "avg_active_history_kv_tokens_per_step": actual["avg_active"],
        "history_kv_compression": actual["weighted"],
        "mean_step_history_kv_compression": actual["mean"],
        "median_step_history_kv_compression": actual["median"],
        "worst_step_history_kv_compression": actual["worst"],
        "actual_weighted_history_kv_compression": actual["weighted"],
        "actual_mean_step_history_kv_compression": actual["mean"],
        "actual_median_step_history_kv_compression": actual["median"],
        "actual_worst_step_history_kv_compression": actual["worst"],
        "peak_gpu_kv_mib": _safe_rate(peak_gpu_bytes, MIB),
        "peak_main_kv_mib": _safe_rate(peak_main_bytes, MIB),
        "peak_c2kv_pool_mib": _safe_rate(peak_c2kv_bytes, MIB),
        "memory_report_coverage": coverage,
        "actual_compression_status": status,
        "estimated_avg_full_history_tokens_per_step": estimated["avg_full"],
        "estimated_avg_active_history_tokens_per_step": estimated["avg_active"],
        "estimated_online_weighted_gpu_history_kv_compression": estimated["weighted"],
        "estimated_online_mean_step_gpu_kv_compression": estimated["mean"],
        "estimated_online_median_step_gpu_kv_compression": estimated["median"],
        "estimated_online_worst_step_gpu_kv_compression": estimated["worst"],
        "estimated_weighted_history_kv_compression": estimated["weighted"],
        "runtime_memory_report_steps": actual_report_count,
        "runtime_memory_missing_steps": missing,
        "bytes_per_kv_token": bytes_per_kv_token,
    }


def _transition_counts(segments: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "action_wrong_to_correct": 0,
        "action_wrong_to_wrong": 0,
        "action_correct_to_wrong": 0,
        "action_correct_to_correct": 0,
        "state_wrong_to_correct": 0,
        "state_wrong_to_wrong": 0,
        "state_correct_to_wrong": 0,
        "state_correct_to_correct": 0,
    }
    for seg in segments:
        before_action = _segment_value(seg, "candidate_action_correct")
        after_action = _segment_value(seg, "repaired_action_correct", "executed_action_correct")
        before_state = _segment_value(seg, "candidate_state_correct")
        after_state = _segment_value(seg, "repaired_state_correct", "executed_state_correct")
        if before_action is False and after_action is True:
            out["action_wrong_to_correct"] += 1
        elif before_action is False and after_action is False:
            out["action_wrong_to_wrong"] += 1
        elif before_action is True and after_action is False:
            out["action_correct_to_wrong"] += 1
        elif before_action is True and after_action is True:
            out["action_correct_to_correct"] += 1
        if before_state is False and after_state is True:
            out["state_wrong_to_correct"] += 1
        elif before_state is False and after_state is False:
            out["state_wrong_to_wrong"] += 1
        elif before_state is True and after_state is False:
            out["state_correct_to_wrong"] += 1
        elif before_state is True and after_state is True:
            out["state_correct_to_correct"] += 1
    out["action_net_recovery_gain"] = (
        out["action_wrong_to_correct"] - out["action_correct_to_wrong"]
    )
    out["state_net_recovery_gain"] = (
        out["state_wrong_to_correct"] - out["state_correct_to_wrong"]
    )
    return out


def summarize_method(run_root: Path, method: str, dirname: str, category: str) -> dict[str, Any]:
    root = run_root / dirname
    details = _load_jsonl(root / "logs" / "details.jsonl")
    checkpoint_steps, checkpoint_segments, checkpoint_summary = _checkpoint_logs(root)
    steps = checkpoint_steps or _flatten_steps(details)
    repair_segments = _segments_from_repair(details)
    segments = checkpoint_segments or repair_segments
    accuracy, correct, total = _score_row(root, category, details)
    total = total or len({str(step.get("id")) for step in steps if step.get("id") is not None})
    committed_steps = len(steps)
    metrics = _latest_metrics(details)

    if checkpoint_segments:
        candidate_steps = sum(_int(seg.get("speculative_steps") or seg.get("segment_length")) for seg in checkpoint_segments)
        regenerated_steps = sum(_int(seg.get("regenerated_steps")) for seg in checkpoint_segments)
    elif repair_segments:
        candidate_steps = sum(_int(seg.get("segment_length") or seg.get("speculative_steps")) for seg in repair_segments)
        regenerated_steps = sum(
            _int(seg.get("repaired_step_count") or seg.get("regenerated_steps") or seg.get("segment_length"))
            for seg in repair_segments
            if seg.get("repair_triggered") or seg.get("detector_trigger")
        )
    else:
        candidate_steps = committed_steps
        regenerated_steps = 0

    oracle_harmful = sum(
        1
        for seg in segments
        if bool(seg.get("oracle_segment_unsafe") or seg.get("oracle_segment_harmful"))
    )
    detector_trigger = sum(
        1
        for seg in segments
        if bool(seg.get("detector_trigger") or seg.get("rollback_triggered") or seg.get("repair_triggered"))
    )
    recovery_attempts = sum(
        1
        for seg in segments
        if bool(seg.get("rollback_triggered") or seg.get("repair_triggered"))
    )
    recovery_success = sum(
        1
        for seg in segments
        if bool(seg.get("segment_recovery_success") or seg.get("repair_segment_success"))
    )
    if checkpoint_segments and recovery_attempts and recovery_success == 0:
        recovery_success = sum(
            1
            for seg in checkpoint_segments
            if bool(seg.get("rollback_triggered"))
            and _int(seg.get("segment_executed_drift_count")) == 0
            and not bool(seg.get("state_drift_after_recovery"))
        )

    memory = _memory_from_steps(steps)

    episode_candidate_drift = _episode_rate(
        details, "candidate_action_drift", "candidate_action_matches_reference"
    )
    episode_executed_drift = _episode_rate(
        details, "executed_action_drift", "executed_action_matches_reference"
    )
    episode_state_drift = _episode_rate(
        details, "state_drift", "state_matches_reference"
    )
    step_candidate_drift = _step_rate(
        steps, "candidate_action_drift", "candidate_action_matches_reference"
    )
    step_executed_drift = _step_rate(
        steps, "executed_action_drift", "executed_action_matches_reference"
    )
    step_state_drift = _step_rate(steps, "state_drift", "state_matches_reference")

    host_values = [
        _num(seg.get("checkpoint_host_tokens"))
        for seg in checkpoint_segments
        if _num(seg.get("checkpoint_host_tokens")) is not None
    ]
    host_values = [value for value in host_values if value is not None]
    avg_host_checkpoint = _mean(host_values)
    peak_host_checkpoint = max(host_values) if host_values else None
    total_host_checkpoint = _num(checkpoint_summary.get("checkpoint_host_tokens"))
    if total_host_checkpoint is None:
        total_host_checkpoint = _num(metrics.get("checkpoint_host_tokens"))
    if peak_host_checkpoint is None:
        peak_host_checkpoint = _num(checkpoint_summary.get("peak_checkpoint_host_tokens"))
    if peak_host_checkpoint is None:
        peak_host_checkpoint = _num(metrics.get("peak_checkpoint_host_tokens"))
    bytes_per_kv_token = _num(memory.get("bytes_per_kv_token"))
    avg_host_checkpoint_mib = (
        avg_host_checkpoint * bytes_per_kv_token / MIB
        if avg_host_checkpoint is not None and bytes_per_kv_token is not None
        else None
    )
    peak_host_checkpoint_mib = (
        peak_host_checkpoint * bytes_per_kv_token / MIB
        if peak_host_checkpoint is not None and bytes_per_kv_token is not None
        else None
    )

    row: dict[str, Any] = {
        "method": method,
        "run_dir": dirname,
        "bfcl_accuracy": accuracy,
        "correct_count": correct,
        "num_examples": total,
        "turn_joint_pass_rate": _turn_joint(details),
        "episode_candidate_action_drift_rate": episode_candidate_drift,
        "episode_executed_action_drift_rate": episode_executed_drift,
        "episode_state_drift_rate": episode_state_drift,
        "step_candidate_action_drift_rate": step_candidate_drift,
        "step_executed_action_drift_rate": step_executed_drift,
        "step_state_drift_rate": step_state_drift,
        "candidate_action_drift_rate": episode_candidate_drift,
        "executed_action_drift_rate": episode_executed_drift,
        "state_drift_rate": episode_state_drift,
        "detector": "oracle" if method not in {"Full", "C2KV"} else "",
        "oracle_harmful_segments": oracle_harmful,
        "oracle_reference_drift_segments": oracle_harmful,
        "detector_trigger_count": detector_trigger,
        "detector_trigger_rate": _safe_rate(detector_trigger, len(segments)),
        "recovery_attempt_count": recovery_attempts,
        "recovery_success_count": recovery_success,
        "recovery_success_rate": _safe_rate(recovery_success, recovery_attempts),
        "reference_recovery_success_rate": _safe_rate(
            recovery_success, recovery_attempts
        ),
        "total_committed_steps": committed_steps,
        "avg_committed_steps_per_episode": _safe_rate(committed_steps, total),
        "total_candidate_steps": candidate_steps,
        "avg_candidate_steps_per_episode": _safe_rate(candidate_steps, total),
        "total_regenerated_steps": regenerated_steps,
        "avg_regenerated_steps_per_episode": _safe_rate(regenerated_steps, total),
        "total_model_generation_calls": candidate_steps + regenerated_steps,
        "model_calls_per_committed_step": _safe_rate(candidate_steps + regenerated_steps, committed_steps),
        "chat_calls": (
            _num(metrics.get("chat_calls"))
            if metrics.get("chat_calls") is not None
            else _num(checkpoint_summary.get("chat_calls"))
        ),
        "reference_aligned_mean_step_compression": None,
        "reference_aligned_weighted_compression": None,
        "reference_aligned_worst_step_compression": None,
        "reference_aligned_num_steps": None,
        "reference_aligned_status": "not_generated",
        "host_checkpoint_kv_tokens": total_host_checkpoint,
        "avg_host_checkpoint_kv_tokens": avg_host_checkpoint,
        "peak_host_checkpoint_kv_tokens": peak_host_checkpoint,
        "avg_host_checkpoint_kv_mib": avg_host_checkpoint_mib,
        "peak_host_checkpoint_kv_mib": peak_host_checkpoint_mib,
        "cumulative_host_checkpoint_token_volume": total_host_checkpoint,
        "avg_resident_host_checkpoint_kv_tokens": avg_host_checkpoint,
        "peak_resident_host_checkpoint_kv_tokens": peak_host_checkpoint,
        "avg_resident_host_checkpoint_kv_mib": avg_host_checkpoint_mib,
        "peak_resident_host_checkpoint_kv_mib": peak_host_checkpoint_mib,
        "host_raw_bank_kv_tokens": 0,
        "peak_host_kv_tokens": peak_host_checkpoint
        or _num(checkpoint_summary.get("peak_host_kv_tokens")),
        "raw_kv_bank_implemented": False,
        "schema_version": 3,
    }
    row.update(memory)
    row.update(_transition_counts(segments))
    return row


def _write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = run_root / "unified_recovery_comparison.csv"
    v3_csv_path = run_root / "unified_recovery_summary_v3.csv"
    json_path = run_root / "unified_recovery_comparison.json"
    md_path = run_root / "unified_recovery_comparison.md"
    quick_csv_path = run_root / "quick_recovery_comparison.csv"
    quick_md_path = run_root / "quick_recovery_comparison.md"
    minimal_csv_path = run_root / "unified_recovery_minimal.csv"
    minimal_md_path = run_root / "unified_recovery_minimal.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(v3_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=V3_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def quick_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "Method": row.get("method"),
            "BFCL Acc": row.get("bfcl_accuracy"),
            "Turn Joint": row.get("turn_joint_pass_rate"),
            "Step Executed Action Drift": row.get("step_executed_action_drift_rate"),
            "Step State Drift": row.get("step_state_drift_rate"),
            "Detector Trigger Rate": row.get("detector_trigger_rate"),
            "Recovery Success Rate": row.get("recovery_success_rate"),
            "Avg Committed Steps / Episode": row.get(
                "avg_committed_steps_per_episode"
            ),
            "Regenerated Steps": row.get("total_regenerated_steps"),
            "Model Calls / Committed Step": row.get(
                "model_calls_per_committed_step"
            ),
            "Avg Active GPU KV Tokens / Step": row.get(
                "avg_active_history_kv_tokens_per_step"
            ),
            "History KV Compression": row.get("history_kv_compression"),
            "Mean Step History KV Compression": row.get(
                "mean_step_history_kv_compression"
            ),
            "Worst Step History KV Compression": row.get(
                "worst_step_history_kv_compression"
            ),
            "Peak GPU KV MiB": row.get("peak_gpu_kv_mib"),
            "Peak Main KV MiB": row.get("peak_main_kv_mib"),
            "Peak C2KV Pool MiB": row.get("peak_c2kv_pool_mib"),
            "Memory Report Coverage": row.get("memory_report_coverage"),
            "Peak Host Checkpoint KV MiB": row.get(
                "peak_resident_host_checkpoint_kv_mib"
            ),
        }

    with open(quick_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=QUICK_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(quick_row(row))

    def minimal_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "Method": row.get("method"),
            "BFCL Acc": row.get("bfcl_accuracy"),
            "Reference Recovery Success": row.get(
                "reference_recovery_success_rate"
            ),
            "Step Exec Drift": row.get("step_executed_action_drift_rate"),
            "Step State Drift": row.get("step_state_drift_rate"),
            "Model Calls / Committed Step": row.get(
                "model_calls_per_committed_step"
            ),
        }

    with open(minimal_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=MINIMAL_CSV_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(minimal_row(row))

    md_cols = [
        "method",
        "bfcl_accuracy",
        "correct_count",
        "executed_action_drift_rate",
        "state_drift_rate",
        "detector_trigger_rate",
        "reference_recovery_success_rate",
        "total_committed_steps",
        "model_calls_per_committed_step",
        "history_kv_compression",
        "mean_step_history_kv_compression",
        "worst_step_history_kv_compression",
        "peak_gpu_kv_mib",
        "peak_host_checkpoint_kv_mib",
        "memory_report_coverage",
        "actual_compression_status",
    ]
    lines = ["# Unified C2KV Recovery Comparison", ""]
    lines.append("| " + " | ".join(md_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(md_cols)) + " |")
    for row in rows:
        vals = []
        for col in md_cols:
            value = row.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif value is None:
                vals.append("-")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    quick_lines = ["# Quick Recovery Comparison", ""]
    quick_lines.append("| " + " | ".join(QUICK_CSV_FIELDS) + " |")
    quick_lines.append("| " + " | ".join(["---"] * len(QUICK_CSV_FIELDS)) + " |")
    for row in rows:
        item = quick_row(row)
        vals = []
        for col in QUICK_CSV_FIELDS:
            value = item.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif value is None:
                vals.append("-")
            else:
                vals.append(str(value))
        quick_lines.append("| " + " | ".join(vals) + " |")
    quick_md_path.write_text("\n".join(quick_lines) + "\n", encoding="utf-8")

    minimal_lines = ["# Unified Recovery Minimal Summary", ""]
    minimal_lines.append("| " + " | ".join(MINIMAL_CSV_FIELDS) + " |")
    minimal_lines.append("| " + " | ".join(["---"] * len(MINIMAL_CSV_FIELDS)) + " |")
    for row in rows:
        item = minimal_row(row)
        vals = []
        for col in MINIMAL_CSV_FIELDS:
            value = item.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif value is None:
                vals.append("-")
            else:
                vals.append(str(value))
        minimal_lines.append("| " + " | ".join(vals) + " |")
    minimal_md_path.write_text("\n".join(minimal_lines) + "\n", encoding="utf-8")


def _repair_transition_diagnosis(run_root: Path, dirname: str) -> dict[str, Any]:
    details = _load_jsonl(run_root / dirname / "logs" / "details.jsonl")
    segments = _segments_from_repair(details)
    counts = {
        "triggered": 0,
        "success": 0,
        "action_wrong_to_correct": 0,
        "action_wrong_to_wrong": 0,
        "action_correct_to_wrong": 0,
        "action_correct_to_correct": 0,
        "state_wrong_to_correct": 0,
        "state_wrong_to_wrong": 0,
        "state_correct_to_wrong": 0,
        "state_correct_to_correct": 0,
    }
    examples: list[dict[str, Any]] = []
    for seg in segments:
        if not bool(seg.get("repair_triggered") or seg.get("detector_trigger")):
            continue
        counts["triggered"] += 1
        if bool(seg.get("repair_segment_success")):
            counts["success"] += 1
        before_action = _segment_value(seg, "candidate_action_correct")
        after_action = _segment_value(
            seg, "repaired_action_correct", "executed_action_correct"
        )
        before_state = _segment_value(seg, "candidate_state_correct")
        after_state = _segment_value(
            seg, "repaired_state_correct", "executed_state_correct"
        )
        if before_action is False and after_action is True:
            counts["action_wrong_to_correct"] += 1
        elif before_action is False and after_action is False:
            counts["action_wrong_to_wrong"] += 1
        elif before_action is True and after_action is False:
            counts["action_correct_to_wrong"] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        key: seg.get(key)
                        for key in (
                            "id",
                            "turn",
                            "segment_start_step",
                            "segment_length",
                            "repair_target_indices",
                            "repair_operation",
                            "candidate_action_correct",
                            "repaired_action_correct",
                            "candidate_state_correct",
                            "repaired_state_correct",
                            "repair_raw_tokens",
                            "physical_prefix_len_before",
                            "physical_prefix_len_after",
                            "logical_position_before",
                            "logical_position_after",
                        )
                    }
                )
        elif before_action is True and after_action is True:
            counts["action_correct_to_correct"] += 1

        if before_state is False and after_state is True:
            counts["state_wrong_to_correct"] += 1
        elif before_state is False and after_state is False:
            counts["state_wrong_to_wrong"] += 1
        elif before_state is True and after_state is False:
            counts["state_correct_to_wrong"] += 1
        elif before_state is True and after_state is True:
            counts["state_correct_to_correct"] += 1
    counts["action_net_recovery_gain"] = (
        counts["action_wrong_to_correct"] - counts["action_correct_to_wrong"]
    )
    counts["state_net_recovery_gain"] = (
        counts["state_wrong_to_correct"] - counts["state_correct_to_wrong"]
    )
    return {"counts": counts, "correct_to_wrong_examples": examples}


def _write_append_w2_diagnosis(run_root: Path) -> None:
    out = {
        "interpretation": (
            "Append W2 keeps target gist KV visible while appending the same "
            "Recent-2 raw KV. If Append has more correct_to_wrong segments than "
            "Replace W2, the likely cause is duplicate representation interference, "
            "not a missing repair path."
        ),
        "append_w2": _repair_transition_diagnosis(run_root, "append_w2"),
        "replace_w2": _repair_transition_diagnosis(run_root, "replace_w2"),
    }
    (run_root / "append_w2_diagnosis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _assert_append_masked_matches_replace(run_root: Path) -> None:
    replace_path = run_root / "replace_w2" / "logs" / "details.jsonl"
    masked_path = run_root / "append_masked_w2" / "logs" / "details.jsonl"
    if not replace_path.exists() or not masked_path.exists():
        return
    replace_rows = {
        str(row.get("id")): row
        for row in _load_jsonl(replace_path)
        if row.get("id") is not None
    }
    masked_rows = {
        str(row.get("id")): row
        for row in _load_jsonl(masked_path)
        if row.get("id") is not None
    }
    mismatches: list[dict[str, Any]] = []
    for sample_id in sorted(set(replace_rows) | set(masked_rows)):
        replace = replace_rows.get(sample_id)
        masked = masked_rows.get(sample_id)
        if replace is None or masked is None:
            mismatches.append(
                {
                    "id": sample_id,
                    "reason": "missing_sample",
                    "replace_present": replace is not None,
                    "append_masked_present": masked is not None,
                }
            )
            continue
        if replace.get("result") != masked.get("result"):
            mismatches.append(
                {
                    "id": sample_id,
                    "reason": "result_mismatch",
                    "replace_result": replace.get("result"),
                    "append_masked_result": masked.get("result"),
                }
            )
    if mismatches:
        mismatch_path = run_root / "append_masked_w2_mismatches.json"
        mismatch_path.write_text(
            json.dumps(mismatches, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Append-Masked W2 must match Replace W2 episode results exactly; "
            f"found {len(mismatches)} mismatches. See {mismatch_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument(
        "--methods",
        default=",".join(f"{label}:{dirname}" for label, dirname in DEFAULT_METHODS),
        help="Comma-separated display:directory entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    rows = []
    for spec in args.methods.split(","):
        if not spec:
            continue
        if ":" in spec:
            label, dirname = spec.split(":", 1)
        else:
            dirname = spec
            label = DISPLAY_NAME_BY_DIR.get(dirname, dirname)
        if not (run_root / dirname).exists():
            rows.append({"method": label, "run_dir": dirname, "schema_version": 2})
            continue
        rows.append(summarize_method(run_root, label, dirname, args.category))
    _assert_append_masked_matches_replace(run_root)
    _write_outputs(run_root, rows)
    _write_append_w2_diagnosis(run_root)
    print(json.dumps({"run_root": str(run_root), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
