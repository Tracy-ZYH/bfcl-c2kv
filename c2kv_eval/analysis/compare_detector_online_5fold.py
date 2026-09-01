from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "Detector",
    "BFCL Acc",
    "BFCL Fold Std",
    "Precision",
    "Recall",
    "F1",
    "FPR",
    "Trigger Rate",
    "Recovery Success",
    "Calls / Committed Step",
    "total_correct",
    "total_examples",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "overall_fpr",
    "overall_trigger_rate",
]

DETAIL_FIELDS = [
    "detector",
    "fold",
    "train_episode_count",
    "calibration_episode_count",
    "test_episode_count",
    "threshold",
    "bfcl_accuracy",
    "precision",
    "recall",
    "f1",
    "fpr",
    "trigger_rate",
    "recovery_success",
    "model_calls_per_committed_step",
    "correct_count",
    "num_examples",
    "detector_tp",
    "detector_fp",
    "detector_tn",
    "detector_fn",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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
    return out if math.isfinite(out) else None


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _score_summary(root: Path) -> tuple[float | None, int, int]:
    valid_by_id: dict[str, bool] = {}
    summary: dict[str, Any] = {}
    for path in sorted((root / "score").rglob("*_score.json")):
        for row in _read_jsonl(path):
            if "id" in row:
                valid_by_id[str(row["id"])] = bool(row.get("valid"))
            elif "accuracy" in row or "correct_count" in row:
                summary.update(row)
    total = int(
        summary.get("total_count")
        or summary.get("total_samples")
        or len(valid_by_id)
        or 0
    )
    correct = int(
        summary.get("correct_count")
        or sum(int(value) for value in valid_by_id.values())
        or 0
    )
    accuracy = _num(summary.get("accuracy"))
    if accuracy is None and total:
        accuracy = correct / total
    return accuracy, correct, total


def _summarize_fold(root: Path, detector: str, fold: int) -> dict[str, Any]:
    details = _read_jsonl(root / "logs" / "details.jsonl")
    summary = _read_json(root / "logs" / "summary.json")
    accuracy, correct, total = _score_summary(root)
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
    tp = sum(int(bool(seg.get("detector_tp"))) for seg in segments)
    fp = sum(int(bool(seg.get("detector_fp"))) for seg in segments)
    tn = sum(int(bool(seg.get("detector_tn"))) for seg in segments)
    fn = sum(int(bool(seg.get("detector_fn"))) for seg in segments)
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    triggered = [seg for seg in segments if seg.get("repair_triggered")]
    successful = [seg for seg in triggered if seg.get("repair_segment_success") is True]
    candidate_steps = sum(int(seg.get("segment_length") or 0) for seg in segments)
    regenerated_steps = sum(int(seg.get("segment_length") or 0) for seg in triggered)
    committed_steps = len(steps)
    threshold = (
        (segments[0].get("logistic_detector_threshold") if segments else None)
        or (segments[0].get("detector_threshold") if segments else None)
        or summary.get("detector_signal_threshold")
        or summary.get("rule_detector_threshold")
    )
    return {
        "detector": detector,
        "fold": fold,
        "threshold": threshold,
        "bfcl_accuracy": accuracy,
        "correct_count": correct,
        "num_examples": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _rate(fp, fp + tn),
        "trigger_rate": _rate(tp + fp, len(segments)),
        "recovery_success": _rate(len(successful), len(triggered)),
        "model_calls_per_committed_step": _rate(
            candidate_steps + regenerated_steps,
            committed_steps,
        ),
        "detector_tp": tp,
        "detector_fp": fp,
        "detector_tn": tn,
        "detector_fn": fn,
    }


def _aggregate(detector: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(key: str) -> list[float]:
        out = []
        for row in rows:
            value = _num(row.get(key))
            if value is not None:
                out.append(value)
        return out

    tp = sum(int(row.get("detector_tp") or 0) for row in rows)
    fp = sum(int(row.get("detector_fp") or 0) for row in rows)
    tn = sum(int(row.get("detector_tn") or 0) for row in rows)
    fn = sum(int(row.get("detector_fn") or 0) for row in rows)
    overall_precision = _rate(tp, tp + fp)
    overall_recall = _rate(tp, tp + fn)
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision is not None
        and overall_recall is not None
        and overall_precision + overall_recall
        else None
    )
    correct = sum(int(row.get("correct_count") or 0) for row in rows)
    total = sum(int(row.get("num_examples") or 0) for row in rows)
    return {
        "Detector": detector,
        "BFCL Acc": _rate(correct, total),
        "BFCL Fold Std": _std(vals("bfcl_accuracy")),
        "Precision": _mean(vals("precision")),
        "Recall": _mean(vals("recall")),
        "F1": _mean(vals("f1")),
        "FPR": _mean(vals("fpr")),
        "Trigger Rate": _mean(vals("trigger_rate")),
        "Recovery Success": _mean(vals("recovery_success")),
        "Calls / Committed Step": _mean(vals("model_calls_per_committed_step")),
        "total_correct": correct,
        "total_examples": total,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "overall_fpr": _rate(fp, fp + tn),
        "overall_trigger_rate": _rate(tp + fp, tp + fp + tn + fn),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--detectors", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    detectors = [item.strip() for item in args.detectors.split(",") if item.strip()]
    detail_rows: list[dict[str, Any]] = []
    for detector in detectors:
        for fold in range(args.folds):
            fold_root = run_root / f"detector_{detector}" / f"fold_{fold}"
            row = _summarize_fold(fold_root, detector, fold)
            cv_dir = run_root / "detector_cv" / f"fold_{fold}"
            thresholds = _read_json(cv_dir / "thresholds.json")
            row["train_episode_count"] = len(thresholds.get("train_episode_ids") or [])
            row["calibration_episode_count"] = len(
                thresholds.get("calibration_episode_ids") or []
            )
            row["test_episode_count"] = len(thresholds.get("test_episode_ids") or [])
            detail_rows.append(row)

    summary_rows = [
        _aggregate(detector, [row for row in detail_rows if row["detector"] == detector])
        for detector in detectors
    ]
    summary_rows.sort(
        key=lambda row: (
            -1.0 if row.get("BFCL Acc") is None else -float(row["BFCL Acc"]),
            row["Detector"],
        )
    )
    _write_csv(
        run_root / "detector_online_5fold_details.csv",
        detail_rows,
        DETAIL_FIELDS,
    )
    _write_csv(
        run_root / "detector_online_5fold_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    (run_root / "detector_online_5fold_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_root / "detector_online_5fold_summary.csv")


if __name__ == "__main__":
    main()
