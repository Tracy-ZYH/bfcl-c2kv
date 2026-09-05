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
    "Trigger Rate",
    "Calls / Committed Step",
    "Correct / 52",
]

FOLD_FIELDS = [
    "detector",
    "fold",
    "selected_threshold",
    "selected_C",
    "selected_l1_ratio",
    "train_oracle_bfcl",
    "selected_train_bfcl",
    "selected_train_trigger_rate",
    "test_bfcl",
    "test_trigger_rate",
    "calls_per_committed_step",
    "correct_count",
    "num_examples",
    "num_train_segments",
    "num_positive_train",
    "num_test_segments",
    "feature_names",
    "label_fallback",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _num(value: Any) -> float | None:
    if value is None or value == "":
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


def _fold_metrics(root: Path) -> dict[str, Any]:
    accuracy, correct, total = _score_summary(root)
    details = _read_jsonl(root / "logs" / "details.jsonl")
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
    triggered = [seg for seg in segments if seg.get("repair_triggered")]
    candidate_steps = sum(int(seg.get("segment_length") or 0) for seg in segments)
    regenerated_steps = sum(int(seg.get("segment_length") or 0) for seg in triggered)
    threshold = None
    if segments:
        threshold = (
            segments[0].get("logistic_detector_threshold")
            or segments[0].get("detector_threshold")
        )
    return {
        "bfcl_accuracy": accuracy,
        "correct_count": correct,
        "num_examples": total,
        "trigger_rate": _rate(len(triggered), len(segments)),
        "calls_per_committed_step": _rate(candidate_steps + regenerated_steps, len(steps)),
        "threshold": threshold,
        "segments": segments,
    }


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        rank += j - i
        i = j
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _auprc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or sum(labels) == 0:
        return None
    points: list[tuple[float, float]] = []
    positives = sum(labels)
    for threshold in sorted(set(scores), reverse=True):
        tp = fp = 0
        for label, score in zip(labels, scores):
            if score >= threshold:
                if label:
                    tp += 1
                else:
                    fp += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        points.append((recall, precision))
    points.sort()
    area = 0.0
    last_recall = 0.0
    last_precision = 1.0
    for recall, precision in points:
        area += (recall - last_recall) * ((precision + last_precision) / 2.0)
        last_recall = recall
        last_precision = precision
    return area


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    cv_dir = Path(args.cv_dir)
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pooled_labels: list[int] = []
    pooled_scores: list[float] = []

    for detector, label in (
        ("oracle", "Oracle"),
        ("combined_logistic_v2", "New Logistic"),
    ):
        accuracies: list[float] = []
        triggers: list[float] = []
        calls: list[float] = []
        total_correct = 0
        total_examples = 0
        for fold in range(args.folds):
            root = run_root / f"detector_{detector}" / f"fold_{fold}"
            metrics = _fold_metrics(root)
            if metrics["bfcl_accuracy"] is not None:
                accuracies.append(float(metrics["bfcl_accuracy"]))
            if metrics["trigger_rate"] is not None:
                triggers.append(float(metrics["trigger_rate"]))
            if metrics["calls_per_committed_step"] is not None:
                calls.append(float(metrics["calls_per_committed_step"]))
            total_correct += int(metrics["correct_count"] or 0)
            total_examples += int(metrics["num_examples"] or 0)

            model = _read_json(
                cv_dir / f"fold_{fold}" / "combined_logistic_v2_selected_model.json"
            )
            fold_row = {
                "detector": detector,
                "fold": fold,
                "selected_threshold": metrics["threshold"]
                if detector == "combined_logistic_v2"
                else None,
                "selected_C": model.get("selected_C")
                if detector == "combined_logistic_v2"
                else None,
                "selected_l1_ratio": model.get("selected_l1_ratio")
                if detector == "combined_logistic_v2"
                else None,
                "train_oracle_bfcl": model.get("train_oracle_bfcl")
                if detector == "combined_logistic_v2"
                else None,
                "selected_train_bfcl": model.get("selected_train_bfcl")
                if detector == "combined_logistic_v2"
                else None,
                "selected_train_trigger_rate": model.get(
                    "selected_train_trigger_rate"
                )
                if detector == "combined_logistic_v2"
                else None,
                "test_bfcl": metrics["bfcl_accuracy"],
                "test_trigger_rate": metrics["trigger_rate"],
                "calls_per_committed_step": metrics["calls_per_committed_step"],
                "correct_count": metrics["correct_count"],
                "num_examples": metrics["num_examples"],
                "num_train_segments": model.get("train_rows")
                if detector == "combined_logistic_v2"
                else None,
                "num_positive_train": None,
                "num_test_segments": len(metrics["segments"]),
                "feature_names": ",".join(model.get("feature_names") or [])
                if detector == "combined_logistic_v2"
                else None,
                "label_fallback": model.get("label_fallback")
                if detector == "combined_logistic_v2"
                else None,
            }
            fold_rows.append(fold_row)
            if detector == "combined_logistic_v2":
                for seg in metrics["segments"]:
                    score = _num(seg.get("logistic_detector_score"))
                    if score is None:
                        continue
                    pooled_scores.append(score)
                    pooled_labels.append(int(bool(seg.get("oracle_segment_harmful"))))

        summary_rows.append(
            {
                "Detector": label,
                "BFCL Acc": _rate(total_correct, total_examples),
                "BFCL Fold Std": _std(accuracies),
                "Trigger Rate": _mean(triggers),
                "Calls / Committed Step": _mean(calls),
                "Correct / 52": f"{total_correct}/{total_examples}",
            }
        )

    by_name = {row["Detector"]: row for row in summary_rows}
    oracle_acc = _num((by_name.get("Oracle") or {}).get("BFCL Acc"))
    logistic_acc = _num((by_name.get("New Logistic") or {}).get("BFCL Acc"))
    oracle_trigger = _num((by_name.get("Oracle") or {}).get("Trigger Rate"))
    logistic_trigger = _num((by_name.get("New Logistic") or {}).get("Trigger Rate"))
    pooled = {
        "heldout_segments": len(pooled_labels),
        "heldout_positives": sum(pooled_labels),
        "heldout_auroc": _auroc(pooled_labels, pooled_scores),
        "heldout_auprc": _auprc(pooled_labels, pooled_scores),
        "positive_score_mean": _mean(
            [score for score, label in zip(pooled_scores, pooled_labels) if label]
        ),
        "positive_score_std": _std(
            [score for score, label in zip(pooled_scores, pooled_labels) if label]
        ),
        "negative_score_mean": _mean(
            [score for score, label in zip(pooled_scores, pooled_labels) if not label]
        ),
        "negative_score_std": _std(
            [score for score, label in zip(pooled_scores, pooled_labels) if not label]
        ),
        "oracle_minus_logistic_bfcl_gap": (
            oracle_acc - logistic_acc
            if oracle_acc is not None and logistic_acc is not None
            else None
        ),
        "oracle_minus_logistic_trigger_gap": (
            oracle_trigger - logistic_trigger
            if oracle_trigger is not None and logistic_trigger is not None
            else None
        ),
    }

    _write_csv(run_root / "combined_logistic_v2_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(
        run_root / "combined_logistic_v2_fold_diagnostics.csv",
        fold_rows,
        FOLD_FIELDS,
    )
    (run_root / "combined_logistic_v2_pooled_diagnostics.json").write_text(
        json.dumps(pooled, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "| Detector | BFCL Acc | BFCL Fold Std | Trigger Rate | Calls / Step | Correct / 52 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {det} | {acc:.4f} | {std:.4f} | {trig:.4f} | {calls:.4f} | {correct} |".format(
                det=row["Detector"],
                acc=float(row["BFCL Acc"] or 0.0),
                std=float(row["BFCL Fold Std"] or 0.0),
                trig=float(row["Trigger Rate"] or 0.0),
                calls=float(row["Calls / Committed Step"] or 0.0),
                correct=row["Correct / 52"],
            )
        )
    lines.extend(
        [
            "",
            f"Oracle - Logistic BFCL gap: {pooled['oracle_minus_logistic_bfcl_gap']}",
            f"Oracle - Logistic Trigger gap: {pooled['oracle_minus_logistic_trigger_gap']}",
            f"Held-out AUROC: {pooled['heldout_auroc']}",
            f"Held-out AUPRC: {pooled['heldout_auprc']}",
        ]
    )
    (run_root / "combined_logistic_v2_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(run_root / "combined_logistic_v2_summary.csv")
    print(run_root / "combined_logistic_v2_fold_diagnostics.csv")
    print(run_root / "combined_logistic_v2_pooled_diagnostics.json")


if __name__ == "__main__":
    main()
