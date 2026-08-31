from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


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
    ("Append W2", "append_w2"),
    ("Recompute W2", "recompute_w2"),
    ("Hint Only", "hint_only"),
    ("Sham Mech", "sham_mech"),
)


CSV_FIELDS = [
    "method",
    "run_dir",
    "bfcl_accuracy",
    "correct_count",
    "num_examples",
    "turn_joint_pass_rate",
    "candidate_action_drift_rate",
    "executed_action_drift_rate",
    "state_drift_rate",
    "detector",
    "oracle_harmful_segments",
    "detector_trigger_count",
    "detector_trigger_rate",
    "recovery_attempt_count",
    "recovery_success_count",
    "recovery_success_rate",
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
    "memory_report_coverage",
    "actual_compression_status",
    "estimated_avg_full_history_tokens_per_step",
    "estimated_avg_active_history_tokens_per_step",
    "estimated_online_weighted_gpu_history_kv_compression",
    "estimated_online_mean_step_gpu_kv_compression",
    "estimated_online_median_step_gpu_kv_compression",
    "estimated_online_worst_step_gpu_kv_compression",
    "reference_aligned_mean_step_compression",
    "reference_aligned_weighted_compression",
    "reference_aligned_worst_step_compression",
    "reference_aligned_num_steps",
    "reference_aligned_status",
    "host_checkpoint_kv_tokens",
    "host_raw_bank_kv_tokens",
    "peak_host_kv_tokens",
    "raw_kv_bank_implemented",
    "runtime_memory_report_steps",
    "runtime_memory_missing_steps",
    "schema_version",
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


def _safe_rate(num: float, den: float) -> float | None:
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


def _latest_metrics(details: list[dict[str, Any]]) -> dict[str, int | float]:
    keys = [
        "chat_calls",
        "canonical_full_history_tokens",
        "physical_history_kv_tokens",
        "c2kv_gist_tokens",
        "repair_kv_tokens",
        "kv_runtime_report_missing",
        "kv_peak_resident_tokens",
    ]
    out: dict[str, int | float] = {key: 0 for key in keys}
    for row in details:
        metrics = row.get("c2kv_drift_metrics") or row.get("checkpoint_metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for key in keys:
            out[key] = out.get(key, 0) + (_num(metrics.get(key)) or 0)
    return out


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
    missing = 0
    for step in steps:
        report = step.get("kv_memory_report")
        if isinstance(report, dict):
            full = _num(report.get("full_equivalent_history_tokens"))
            active = _num(
                report.get("active_history_kv_tokens")
                or report.get("resident_history_kv_tokens")
            )
            if full is not None and active and active > 0:
                actual_pairs.append((full, active))
            else:
                missing += 1
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
        ratios = [full / active for full, active in pairs if active > 0]
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
    coverage = _safe_rate(len(actual_pairs), total)
    return {
        "avg_full_equivalent_history_kv_tokens_per_step": actual["avg_full"],
        "avg_active_gpu_history_kv_tokens_per_step": actual["avg_active"],
        "online_mean_step_gpu_kv_compression": actual["mean"],
        "online_weighted_gpu_history_kv_compression": actual["weighted"],
        "online_worst_step_gpu_kv_compression": actual["worst"],
        "memory_report_coverage": coverage,
        "actual_compression_status": (
            "complete" if total and len(actual_pairs) == total else "incomplete"
        ),
        "estimated_avg_full_history_tokens_per_step": estimated["avg_full"],
        "estimated_avg_active_history_tokens_per_step": estimated["avg_active"],
        "estimated_online_weighted_gpu_history_kv_compression": estimated["weighted"],
        "estimated_online_mean_step_gpu_kv_compression": estimated["mean"],
        "estimated_online_median_step_gpu_kv_compression": estimated["median"],
        "estimated_online_worst_step_gpu_kv_compression": estimated["worst"],
        "runtime_memory_report_steps": len(actual_pairs),
        "runtime_memory_missing_steps": missing,
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

    memory = _memory_from_steps(steps)
    if memory["estimated_online_weighted_gpu_history_kv_compression"] is None:
        full_tokens = _num(metrics.get("canonical_full_history_tokens")) or 0
        active_tokens = _num(metrics.get("physical_history_kv_tokens")) or 0
        memory["estimated_avg_full_history_tokens_per_step"] = _safe_rate(full_tokens, committed_steps)
        memory["estimated_avg_active_history_tokens_per_step"] = _safe_rate(active_tokens, committed_steps)
        memory["estimated_online_weighted_gpu_history_kv_compression"] = _safe_rate(full_tokens, active_tokens)

    row: dict[str, Any] = {
        "method": method,
        "run_dir": dirname,
        "bfcl_accuracy": accuracy,
        "correct_count": correct,
        "num_examples": total,
        "turn_joint_pass_rate": _turn_joint(details),
        "candidate_action_drift_rate": _episode_rate(details, "candidate_action_drift", "candidate_action_matches_reference"),
        "executed_action_drift_rate": _episode_rate(details, "executed_action_drift", "executed_action_matches_reference"),
        "state_drift_rate": _episode_rate(details, "state_drift", "state_matches_reference"),
        "detector": "oracle" if method not in {"Full", "C2KV"} else "",
        "oracle_harmful_segments": oracle_harmful,
        "detector_trigger_count": detector_trigger,
        "detector_trigger_rate": _safe_rate(detector_trigger, len(segments)),
        "recovery_attempt_count": recovery_attempts,
        "recovery_success_count": recovery_success,
        "recovery_success_rate": _safe_rate(recovery_success, recovery_attempts),
        "total_committed_steps": committed_steps,
        "avg_committed_steps_per_episode": _safe_rate(committed_steps, total),
        "total_candidate_steps": candidate_steps,
        "avg_candidate_steps_per_episode": _safe_rate(candidate_steps, total),
        "total_regenerated_steps": regenerated_steps,
        "avg_regenerated_steps_per_episode": _safe_rate(regenerated_steps, total),
        "total_model_generation_calls": candidate_steps + regenerated_steps,
        "model_calls_per_committed_step": _safe_rate(candidate_steps + regenerated_steps, committed_steps),
        "chat_calls": _int(metrics.get("chat_calls") or checkpoint_summary.get("chat_calls")),
        "reference_aligned_mean_step_compression": None,
        "reference_aligned_weighted_compression": None,
        "reference_aligned_worst_step_compression": None,
        "reference_aligned_num_steps": None,
        "reference_aligned_status": "not_generated",
        "host_checkpoint_kv_tokens": _int(checkpoint_summary.get("host_checkpoint_kv_tokens")),
        "host_raw_bank_kv_tokens": 0,
        "peak_host_kv_tokens": _int(checkpoint_summary.get("peak_host_kv_tokens")),
        "raw_kv_bank_implemented": False,
        "schema_version": 2,
    }
    row.update(memory)
    row.update(_transition_counts(segments))
    return row


def _write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = run_root / "unified_recovery_comparison.csv"
    json_path = run_root / "unified_recovery_comparison.json"
    md_path = run_root / "unified_recovery_comparison.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_cols = [
        "method",
        "bfcl_accuracy",
        "correct_count",
        "executed_action_drift_rate",
        "state_drift_rate",
        "detector_trigger_rate",
        "recovery_success_rate",
        "total_committed_steps",
        "model_calls_per_committed_step",
        "online_weighted_gpu_history_kv_compression",
        "estimated_online_weighted_gpu_history_kv_compression",
        "memory_report_coverage",
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
            label = dirname = spec
        if not (run_root / dirname).exists():
            rows.append({"method": label, "run_dir": dirname, "schema_version": 2})
            continue
        rows.append(summarize_method(run_root, label, dirname, args.category))
    _write_outputs(run_root, rows)
    print(json.dumps({"run_root": str(run_root), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
