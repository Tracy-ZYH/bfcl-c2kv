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

PARETO_FIELDS = [
    "Detector / Operating Point",
    "Target Trigger Rate",
    "Actual Trigger Rate",
    "BFCL Accuracy",
    "BFCL Fold Std",
    "Correct / 52",
    "Pareto Frontier",
]

BFCL_TRIGGER_SUMMARY_FIELDS = [
    "Detector",
    "BFCL Acc",
    "Trigger Rate",
    "BFCL Fold Std",
    "Correct / 52",
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

LOGISTIC_DIAGNOSTIC_FIELDS = [
    "detector",
    "fold",
    "target_trigger_rate",
    "threshold",
    "train_actual_trigger_rate",
    "train_reference_drift_rate",
    "online_actual_trigger_rate",
    "train_score_count",
    "train_score_num_unique",
    "train_score_min",
    "train_score_p10",
    "train_score_p20",
    "train_score_p30",
    "train_score_p40",
    "train_score_p50",
    "train_score_p60",
    "train_score_p70",
    "train_score_p80",
    "train_score_p90",
    "train_score_max",
    "test_episode_count",
]

PREFERRED_ORDER = [
    "never_trigger",
    "combined_logistic",
    "combined_logistic_rate_10",
    "combined_logistic_rate_20",
    "combined_logistic_rate_30",
    "combined_logistic_rate_40",
    "combined_logistic_rate_50",
    "combined_logistic_rate_60",
    "rule_trigger",
    "always_trigger",
    "oracle",
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


def _detector_label(detector: str) -> str:
    if detector == "never_trigger":
        return "Never Trigger"
    if detector == "always_trigger":
        return "Always Trigger"
    if detector == "oracle":
        return "Oracle"
    if detector == "rule_trigger":
        return "Rule Trigger"
    if detector == "combined_logistic":
        return "Combined Logistic"
    prefix = "combined_logistic_rate_"
    if detector.startswith(prefix):
        return f"Logistic @{detector[len(prefix):]}%"
    if detector == "max_risk_score":
        return "Max Risk Score"
    return detector


def _target_trigger_rate(detector: str) -> float | None:
    prefix = "combined_logistic_rate_"
    if detector.startswith(prefix):
        try:
            return int(detector[len(prefix) :]) / 100.0
        except Exception:
            return None
    if detector == "never_trigger":
        return 0.0
    if detector == "always_trigger":
        return 1.0
    return None


def _is_logistic_rate_detector(detector: str) -> bool:
    return detector.startswith("combined_logistic_rate_")


def _actual_trigger(row: dict[str, Any]) -> float | None:
    value = _num(row.get("overall_trigger_rate"))
    return value if value is not None else _num(row.get("Trigger Rate"))


def _bfcl_acc(row: dict[str, Any]) -> float | None:
    return _num(row.get("BFCL Acc"))


def _preferred_index(detector: str) -> int:
    try:
        return PREFERRED_ORDER.index(detector)
    except ValueError:
        return len(PREFERRED_ORDER)


def _pareto_rows(summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pareto_source: list[dict[str, Any]] = []
    for row in summary_rows:
        detector = str(row.get("Detector") or "")
        acc = _bfcl_acc(row)
        trigger = _actual_trigger(row)
        total = int(row.get("total_examples") or 0)
        correct = int(row.get("total_correct") or 0)
        pareto_source.append(
            {
                "detector": detector,
                "label": _detector_label(detector),
                "target_trigger_rate": _target_trigger_rate(detector),
                "actual_trigger_rate": trigger,
                "bfcl_accuracy": acc,
                "bfcl_fold_std": _num(row.get("BFCL Fold Std")),
                "correct": correct,
                "total": total,
            }
        )

    for point in pareto_source:
        acc = point["bfcl_accuracy"]
        trigger = point["actual_trigger_rate"]
        dominated = False
        if acc is not None and trigger is not None:
            for other in pareto_source:
                if other is point:
                    continue
                other_acc = other["bfcl_accuracy"]
                other_trigger = other["actual_trigger_rate"]
                if other_acc is None or other_trigger is None:
                    continue
                if other_trigger < trigger and other_acc >= acc:
                    dominated = True
                    break
        point["pareto"] = not dominated

    rows = []
    for point in sorted(
        pareto_source,
        key=lambda item: (_preferred_index(item["detector"]), item["detector"]),
    ):
        rows.append(
            {
                "Detector / Operating Point": point["label"],
                "Target Trigger Rate": point["target_trigger_rate"],
                "Actual Trigger Rate": point["actual_trigger_rate"],
                "BFCL Accuracy": point["bfcl_accuracy"],
                "BFCL Fold Std": point["bfcl_fold_std"],
                "Correct / 52": (
                    f"{point['correct']}/{point['total']}" if point["total"] else ""
                ),
                "Pareto Frontier": point["pareto"],
            }
        )

    oracle = next(
        (point for point in pareto_source if point["detector"] == "oracle"),
        None,
    )
    oracle_acc = oracle["bfcl_accuracy"] if oracle else None

    def best_for(frac: float) -> dict[str, Any] | None:
        if oracle_acc is None:
            return None
        target = oracle_acc * frac
        candidates = [
            point
            for point in pareto_source
            if point["bfcl_accuracy"] is not None
            and point["actual_trigger_rate"] is not None
            and point["bfcl_accuracy"] >= target
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item["actual_trigger_rate"],
                -item["bfcl_accuracy"],
                _preferred_index(item["detector"]),
            ),
        )

    frontier = [
        point
        for point in sorted(
            pareto_source,
            key=lambda item: (
                float("inf")
                if item["actual_trigger_rate"] is None
                else item["actual_trigger_rate"],
                -1.0 if item["bfcl_accuracy"] is None else -item["bfcl_accuracy"],
            ),
        )
        if point["pareto"]
    ]
    payload = {
        "oracle_acc": oracle_acc,
        "oracle_70pct_target": oracle_acc * 0.70 if oracle_acc is not None else None,
        "oracle_95pct_target": oracle_acc * 0.95 if oracle_acc is not None else None,
        "best_70pct_oracle_point": best_for(0.70),
        "best_95pct_oracle_point": best_for(0.95),
        "pareto_frontier": frontier,
    }
    return rows, payload


def _bfcl_trigger_summary_rows(
    summary_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(
        summary_rows,
        key=lambda item: (
            _preferred_index(str(item.get("Detector") or "")),
            str(item.get("Detector") or ""),
        ),
    ):
        total = int(row.get("total_examples") or 0)
        correct = int(row.get("total_correct") or 0)
        rows.append(
            {
                "Detector": _detector_label(str(row.get("Detector") or "")),
                "BFCL Acc": row.get("BFCL Acc"),
                "Trigger Rate": _actual_trigger(row),
                "BFCL Fold Std": row.get("BFCL Fold Std"),
                "Correct / 52": f"{correct}/{total}" if total else "",
            }
        )
    return rows


def _sanity_checks(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_detector = {str(row.get("Detector")): row for row in summary_rows}
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    never = by_detector.get("never_trigger")
    if never is not None:
        trigger = _actual_trigger(never)
        add(
            "never_trigger_rate_zero",
            trigger is not None and abs(trigger) < 1e-9,
            f"actual_trigger_rate={trigger}",
        )

    always = by_detector.get("always_trigger")
    if always is not None:
        trigger = _actual_trigger(always)
        add(
            "always_trigger_rate_one",
            trigger is not None and abs(trigger - 1.0) < 1e-9,
            f"actual_trigger_rate={trigger}",
        )

    oracle = by_detector.get("oracle")
    if oracle is not None:
        precision = _num(oracle.get("overall_precision"))
        recall = _num(oracle.get("overall_recall"))
        fpr = _num(oracle.get("overall_fpr"))
        add(
            "oracle_precision_one",
            precision is not None and abs(precision - 1.0) < 1e-9,
            f"overall_precision={precision}",
        )
        add(
            "oracle_recall_one",
            recall is not None and abs(recall - 1.0) < 1e-9,
            f"overall_recall={recall}",
        )
        add(
            "oracle_fpr_zero",
            fpr is not None and abs(fpr) < 1e-9,
            f"overall_fpr={fpr}",
        )

    failed = [check for check in checks if not check["passed"]]
    return {
        "checks": checks,
        "passed": not failed,
        "failed": failed,
    }


def _logistic_diagnostic_row(
    *,
    detector: str,
    fold: int,
    detail_row: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any] | None:
    if not _is_logistic_rate_detector(detector):
        return None
    return {
        "detector": detector,
        "fold": fold,
        "target_trigger_rate": thresholds.get("target_trigger_rate")
        if thresholds.get("target_trigger_rate") is not None
        else _target_trigger_rate(detector),
        "threshold": thresholds.get(detector) or detail_row.get("threshold"),
        "train_actual_trigger_rate": thresholds.get(
            "train_trigger_rate_at_threshold"
        ),
        "train_reference_drift_rate": thresholds.get("train_reference_drift_rate"),
        "online_actual_trigger_rate": detail_row.get("trigger_rate"),
        "train_score_count": thresholds.get("train_score_count"),
        "train_score_num_unique": thresholds.get("train_score_num_unique"),
        "train_score_min": thresholds.get("train_score_min"),
        "train_score_p10": thresholds.get("train_score_p10"),
        "train_score_p20": thresholds.get("train_score_p20"),
        "train_score_p30": thresholds.get("train_score_p30"),
        "train_score_p40": thresholds.get("train_score_p40"),
        "train_score_p50": thresholds.get("train_score_p50"),
        "train_score_p60": thresholds.get("train_score_p60"),
        "train_score_p70": thresholds.get("train_score_p70"),
        "train_score_p80": thresholds.get("train_score_p80"),
        "train_score_p90": thresholds.get("train_score_p90"),
        "train_score_max": thresholds.get("train_score_max"),
        "test_episode_count": detail_row.get("test_episode_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--detectors", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    detectors = [item.strip() for item in args.detectors.split(",") if item.strip()]
    detail_rows: list[dict[str, Any]] = []
    logistic_diagnostics: list[dict[str, Any]] = []
    for detector in detectors:
        for fold in range(args.folds):
            fold_root = run_root / f"detector_{detector}" / f"fold_{fold}"
            row = _summarize_fold(fold_root, detector, fold)
            detector_cv_dir = run_root / "detector_cv" / detector / f"fold_{fold}"
            cv_dir = (
                detector_cv_dir
                if detector_cv_dir.exists()
                else run_root / "detector_cv" / f"fold_{fold}"
            )
            thresholds = _read_json(cv_dir / "thresholds.json")
            row["train_episode_count"] = len(thresholds.get("train_episode_ids") or [])
            row["calibration_episode_count"] = len(
                thresholds.get("calibration_episode_ids") or []
            )
            row["test_episode_count"] = len(thresholds.get("test_episode_ids") or [])
            detail_rows.append(row)
            diagnostic = _logistic_diagnostic_row(
                detector=detector,
                fold=fold,
                detail_row=row,
                thresholds=thresholds,
            )
            if diagnostic is not None:
                logistic_diagnostics.append(diagnostic)

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
    pareto_rows, pareto_payload = _pareto_rows(summary_rows)
    bfcl_trigger_rows = _bfcl_trigger_summary_rows(summary_rows)
    sanity = _sanity_checks(summary_rows)
    if not sanity["passed"]:
        (run_root / "detector_online_5fold_sanity.json").write_text(
            json.dumps(sanity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "detector online 5-fold sanity checks failed: "
            + json.dumps(sanity["failed"], ensure_ascii=False)
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
    _write_csv(
        run_root / "detector_trigger_bfcl_pareto.csv",
        pareto_rows,
        PARETO_FIELDS,
    )
    _write_csv(
        run_root / "detector_bfcl_trigger_summary.csv",
        bfcl_trigger_rows,
        BFCL_TRIGGER_SUMMARY_FIELDS,
    )
    _write_csv(
        run_root / "detector_logistic_trigger_diagnostics.csv",
        logistic_diagnostics,
        LOGISTIC_DIAGNOSTIC_FIELDS,
    )
    (run_root / "detector_online_5fold_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_root / "detector_trigger_bfcl_pareto.json").write_text(
        json.dumps(pareto_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_root / "detector_online_5fold_sanity.json").write_text(
        json.dumps(sanity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(run_root / "detector_online_5fold_summary.csv")
    print(run_root / "detector_trigger_bfcl_pareto.csv")
    print(run_root / "detector_bfcl_trigger_summary.csv")
    print(run_root / "detector_logistic_trigger_diagnostics.csv")


if __name__ == "__main__":
    main()
