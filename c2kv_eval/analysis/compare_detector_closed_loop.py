from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _fmt(value: Any, *, pct: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if pct:
        return f"{number * 100:.2f}%"
    return f"{number:.4f}"


def _score_files(root: Path) -> list[Path]:
    return sorted((root / "score").rglob("*_score.json"))


def _score_summary(root: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    summary: dict[str, Any] = {}
    valid_by_id: dict[str, bool] = {}
    for path in _score_files(root):
        for row in _read_jsonl(path):
            if "id" in row:
                valid_by_id[str(row["id"])] = bool(row.get("valid"))
            elif "accuracy" in row or "correct_count" in row:
                summary.update(row)
    if not summary and valid_by_id:
        correct = sum(int(v) for v in valid_by_id.values())
        total = len(valid_by_id)
        summary = {
            "accuracy": correct / total if total else None,
            "correct_count": correct,
            "total_count": total,
        }
    return summary, valid_by_id


def _episode_failed_after_fn(details: list[dict[str, Any]], valid_by_id: dict[str, bool]) -> tuple[int, int]:
    episodes_with_fn = 0
    failed_after_fn = 0
    for row in details:
        sample_id = str(row.get("id"))
        segments = row.get("repair_segments") or []
        has_fn = any(bool(seg.get("detector_fn")) for seg in segments)
        if not has_fn:
            continue
        episodes_with_fn += 1
        if valid_by_id.get(sample_id) is False:
            failed_after_fn += 1
    return episodes_with_fn, failed_after_fn


def _transition_counts(segments: list[dict[str, Any]], prefix: str) -> dict[str, int]:
    out = {
        f"{prefix}_wrong_to_correct": 0,
        f"{prefix}_wrong_to_wrong": 0,
        f"{prefix}_correct_to_wrong": 0,
        f"{prefix}_correct_to_correct": 0,
    }
    for seg in segments:
        if not seg.get("repair_triggered"):
            continue
        before = seg.get(f"candidate_{prefix}_correct")
        after = seg.get(f"repaired_{prefix}_correct")
        if before is False and after is True:
            out[f"{prefix}_wrong_to_correct"] += 1
        elif before is False and after is False:
            out[f"{prefix}_wrong_to_wrong"] += 1
        elif before is True and after is False:
            out[f"{prefix}_correct_to_wrong"] += 1
        elif before is True and after is True:
            out[f"{prefix}_correct_to_correct"] += 1
    out[f"{prefix}_net_recovery_gain"] = (
        out[f"{prefix}_wrong_to_correct"] - out[f"{prefix}_correct_to_wrong"]
    )
    return out


def summarize_mode(run_root: Path, mode: str) -> dict[str, Any]:
    root = run_root / mode
    details = _read_jsonl(root / "logs" / "details.jsonl")
    run_summary = _read_json(root / "logs" / "summary.json")
    score, valid_by_id = _score_summary(root)
    segments = [
        seg
        for row in details
        for seg in (row.get("repair_segments") or [])
        if isinstance(seg, dict)
    ]
    steps = [
        step
        for row in details
        for step in (row.get("drift_steps") or [])
        if isinstance(step, dict)
    ]

    total_examples = len(details) or int(score.get("total_count") or 0)
    correct = int(score.get("correct_count") or sum(int(v) for v in valid_by_id.values()))
    accuracy = score.get("accuracy")
    if accuracy is None and total_examples:
        accuracy = correct / total_examples

    turn_joint: dict[tuple[Any, Any], bool] = {}
    for step in steps:
        key = (step.get("id"), step.get("turn"))
        turn_joint.setdefault(key, True)
        if (
            step.get("executed_action_drift") is True
            or step.get("executed_action_matches_reference") is False
            or step.get("state_drift") is True
            or step.get("state_matches_reference") is False
        ):
            turn_joint[key] = False

    executed_drift_ids = {
        row.get("id")
        for row in details
        if any(
            step.get("executed_action_drift") is True
            or step.get("executed_action_matches_reference") is False
            for step in (row.get("drift_steps") or [])
            if isinstance(step, dict)
        )
    }
    state_drift_ids = {
        row.get("id")
        for row in details
        if any(
            step.get("state_drift") is True
            or step.get("state_matches_reference") is False
            for step in (row.get("drift_steps") or [])
            if isinstance(step, dict)
        )
    }

    detector_tp = sum(int(bool(seg.get("detector_tp"))) for seg in segments)
    detector_fp = sum(int(bool(seg.get("detector_fp"))) for seg in segments)
    detector_tn = sum(int(bool(seg.get("detector_tn"))) for seg in segments)
    detector_fn = sum(int(bool(seg.get("detector_fn"))) for seg in segments)
    pred_pos = detector_tp + detector_fp
    actual_pos = detector_tp + detector_fn
    actual_neg = detector_fp + detector_tn
    precision = _rate(detector_tp, pred_pos)
    recall = _rate(detector_tp, actual_pos)
    detector_f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )

    triggered = [seg for seg in segments if seg.get("repair_triggered")]
    tp_attempts = [seg for seg in segments if seg.get("detector_tp")]
    tp_success = [
        seg for seg in tp_attempts if seg.get("repair_segment_success") is True
    ]
    fp_segments = [seg for seg in segments if seg.get("detector_fp")]
    fp_harm = [seg for seg in fp_segments if seg.get("fp_recovery_harm")]
    episodes_with_fn, episodes_failed_after_fn = _episode_failed_after_fn(
        details,
        valid_by_id,
    )

    candidate_steps = sum(int(seg.get("segment_length") or 0) for seg in segments)
    regenerated_steps = sum(
        int(seg.get("segment_length") or 0) for seg in triggered
    )
    committed_steps = len(steps)
    row = {
        "mode": mode,
        "detector": (
            run_summary.get("detector_arm")
            or (segments[0].get("detector_arm") if segments else mode)
        ),
        "threshold": (
            run_summary.get("detector_signal_threshold")
            or run_summary.get("rule_detector_threshold")
            or (segments[0].get("detector_threshold") if segments else None)
        ),
        "bfcl_accuracy": accuracy,
        "correct_count": correct,
        "num_examples": total_examples,
        "turn_joint_pass_rate": _rate(
            sum(int(v) for v in turn_joint.values()),
            len(turn_joint),
        ),
        "executed_action_drift_rate": _rate(len(executed_drift_ids), total_examples),
        "state_drift_rate": _rate(len(state_drift_ids), total_examples),
        "detector_tp": detector_tp,
        "detector_fp": detector_fp,
        "detector_tn": detector_tn,
        "detector_fn": detector_fn,
        "detector_precision": precision,
        "detector_recall": recall,
        "detector_f1": detector_f1,
        "detector_fpr": _rate(detector_fp, actual_neg),
        "detector_trigger_count": pred_pos,
        "detector_trigger_rate": _rate(pred_pos, len(segments)),
        "tp_recovery_attempts": len(tp_attempts),
        "tp_recovery_success_count": len(tp_success),
        "tp_recovery_success_rate": _rate(len(tp_success), len(tp_attempts)),
        "fp_recovery_count": len(fp_segments),
        "fp_recovery_harm_count": len(fp_harm),
        "fp_recovery_harm_rate": _rate(len(fp_harm), len(fp_segments)),
        "false_negative_count": detector_fn,
        "false_negative_rate": _rate(detector_fn, actual_pos),
        "episodes_with_fn": episodes_with_fn,
        "episodes_failed_after_fn": episodes_failed_after_fn,
        "episode_failure_after_first_fn_rate": _rate(
            episodes_failed_after_fn,
            episodes_with_fn,
        ),
        "committed_steps": committed_steps,
        "avg_committed_steps_per_episode": _rate(committed_steps, total_examples),
        "candidate_steps": candidate_steps,
        "regenerated_steps": regenerated_steps,
        "total_model_generation_calls": candidate_steps + regenerated_steps,
        "model_calls_per_committed_step": _rate(
            candidate_steps + regenerated_steps,
            committed_steps,
        ),
        "repair_backend": "replace_w2",
    }
    row.update(_transition_counts(segments, "action"))
    row.update(_transition_counts(segments, "state"))
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "Detector",
        "Threshold",
        "BFCL Acc",
        "Turn Joint",
        "Executed Action Drift",
        "State Drift",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "Trigger Rate",
        "TP Recovery Success Rate",
        "FP Harm Rate",
        "FN Rate",
        "Episode Failure After First FN Rate",
        "Committed Steps",
        "Regenerated Steps",
        "Model Calls / Committed Step",
    ]
    raw_columns = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns)
        writer.writeheader()
        writer.writerows(rows)
    pretty_path = path.with_name(path.stem + "_pretty.csv")
    with open(pretty_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(
                [
                    row.get("detector"),
                    row.get("threshold"),
                    row.get("bfcl_accuracy"),
                    row.get("turn_joint_pass_rate"),
                    row.get("executed_action_drift_rate"),
                    row.get("state_drift_rate"),
                    row.get("detector_precision"),
                    row.get("detector_recall"),
                    row.get("detector_f1"),
                    row.get("detector_fpr"),
                    row.get("detector_trigger_rate"),
                    row.get("tp_recovery_success_rate"),
                    row.get("fp_recovery_harm_rate"),
                    row.get("false_negative_rate"),
                    row.get("episode_failure_after_first_fn_rate"),
                    row.get("committed_steps"),
                    row.get("regenerated_steps"),
                    row.get("model_calls_per_committed_step"),
                ]
            )


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# BFCL stable52 Closed-Loop Detector Benchmark",
        "",
        "| Detector | Threshold | BFCL Acc | Turn Joint | Exec Drift | State Drift | P | R | F1 | FPR | Trigger | TP Recovery | FP Harm | FN | FN Episode Fail | Calls/Step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {detector} | {thr} | {acc} | {joint} | {execd} | {state} | {p} | {r} | {f1} | {fpr} | {trigger} | {tp_recovery} | {fp_harm} | {fn} | {fn_fail} | {calls} |".format(
                detector=row.get("detector"),
                thr=_fmt(row.get("threshold")),
                acc=_fmt(row.get("bfcl_accuracy"), pct=True),
                joint=_fmt(row.get("turn_joint_pass_rate"), pct=True),
                execd=_fmt(row.get("executed_action_drift_rate"), pct=True),
                state=_fmt(row.get("state_drift_rate"), pct=True),
                p=_fmt(row.get("detector_precision"), pct=True),
                r=_fmt(row.get("detector_recall"), pct=True),
                f1=_fmt(row.get("detector_f1"), pct=True),
                fpr=_fmt(row.get("detector_fpr"), pct=True),
                trigger=_fmt(row.get("detector_trigger_rate"), pct=True),
                tp_recovery=_fmt(row.get("tp_recovery_success_rate"), pct=True),
                fp_harm=_fmt(row.get("fp_recovery_harm_rate"), pct=True),
                fn=_fmt(row.get("false_negative_rate"), pct=True),
                fn_fail=_fmt(
                    row.get("episode_failure_after_first_fn_rate"),
                    pct=True,
                ),
                calls=_fmt(row.get("model_calls_per_committed_step")),
            )
        )
    lines.extend(["", "## Auto Readout"])
    by_detector = {str(row.get("detector")): row for row in rows}
    oracle = by_detector.get("oracle")
    logistic = by_detector.get("combined_logistic_best_f1")
    high_recall = by_detector.get("combined_logistic_high_recall")
    max_risk = by_detector.get("max_risk_score")
    rule = by_detector.get("rule_trigger")
    if oracle and logistic:
        lines.append(
            f"- logistic_gap_to_oracle_bfcl = {_fmt((oracle.get('bfcl_accuracy') or 0) - (logistic.get('bfcl_accuracy') or 0), pct=True)}"
        )
    if logistic and max_risk:
        lines.append(
            f"- max_risk_gap_to_logistic_bfcl = {_fmt((logistic.get('bfcl_accuracy') or 0) - (max_risk.get('bfcl_accuracy') or 0), pct=True)}"
        )
    if logistic and rule:
        lines.append(
            f"- rule_gap_to_logistic_bfcl = {_fmt((logistic.get('bfcl_accuracy') or 0) - (rule.get('bfcl_accuracy') or 0), pct=True)}"
        )
    if logistic:
        lines.append(
            f"- logistic_fp_harm_rate = {_fmt(logistic.get('fp_recovery_harm_rate'), pct=True)}"
        )
        lines.append(
            f"- logistic_fn_propagation_rate = {_fmt(logistic.get('episode_failure_after_first_fn_rate'), pct=True)}"
        )
    if logistic and high_recall:
        lines.append(
            f"- high_recall_gain_over_best_f1 = {_fmt((high_recall.get('bfcl_accuracy') or 0) - (logistic.get('bfcl_accuracy') or 0), pct=True)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sanity(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    by_detector = {str(row.get("detector")): row for row in rows}
    oracle = by_detector.get("oracle")
    if oracle:
        if oracle.get("detector_precision") not in (None, 1.0):
            warnings.append("Oracle precision is not 1.0")
        if oracle.get("detector_recall") not in (None, 1.0):
            warnings.append("Oracle recall is not 1.0")
        if oracle.get("detector_fpr") not in (None, 0.0):
            warnings.append("Oracle FPR is not 0.0")
    always = by_detector.get("always_trigger")
    if always:
        if always.get("detector_recall") not in (None, 1.0):
            warnings.append("Always-trigger recall is not 1.0")
        if always.get("detector_fpr") not in (None, 1.0):
            warnings.append("Always-trigger FPR is not 1.0")
    never = by_detector.get("never_trigger")
    if never and (never.get("detector_trigger_rate") or 0.0) != 0.0:
        warnings.append("Never-trigger has non-zero trigger rate")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--modes", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = [
        summarize_mode(run_root, mode)
        for mode in args.modes.split(",")
        if mode.strip()
    ]
    output_csv = run_root / "detector_closed_loop_summary.csv"
    _write_csv(output_csv, rows)
    (run_root / "detector_closed_loop_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_md(run_root / "detector_closed_loop_report.md", rows)
    warnings = _sanity(rows)
    (run_root / "detector_closed_loop_sanity.json").write_text(
        json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "warnings": warnings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
