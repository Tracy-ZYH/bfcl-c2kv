from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PARETO_FIELDS = [
    "Detector",
    "Fold",
    "Threshold",
    "Trigger Rate",
    "BFCL Accuracy",
    "Train Oracle BFCL Accuracy",
    "Oracle 70 Target",
    "Oracle 95 Target",
    "Meets Oracle 70",
    "Meets Oracle 95",
    "Is Pareto",
    "Selected 70",
    "Selected 95",
    "Selected Final",
]

SELECTED_FIELDS = [
    "fold",
    "selected_threshold",
    "selection_rule",
    "train_bfcl_accuracy",
    "train_trigger_rate",
    "train_oracle_bfcl_accuracy",
    "oracle_70_target",
    "oracle_95_target",
    "selected_95_threshold",
    "selected_95_train_bfcl_accuracy",
    "selected_95_train_trigger_rate",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


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


def _trigger_rate(root: Path) -> float | None:
    details = _read_jsonl(root / "logs" / "details.jsonl")
    segments = [
        seg
        for row in details
        for seg in (row.get("repair_segments") or [])
        if isinstance(seg, dict)
    ]
    if not segments:
        summary = _read_json(root / "logs" / "summary.json")
        return _num(summary.get("detector_trigger_rate"))
    triggered = sum(int(bool(seg.get("repair_triggered"))) for seg in segments)
    return triggered / len(segments)


def _is_pareto(point: dict[str, Any], points: list[dict[str, Any]]) -> bool:
    acc = point["bfcl_accuracy"]
    trigger = point["trigger_rate"]
    if acc is None or trigger is None:
        return False
    for other in points:
        if other is point:
            continue
        other_acc = other["bfcl_accuracy"]
        other_trigger = other["trigger_rate"]
        if other_acc is None or other_trigger is None:
            continue
        if other_trigger < trigger and other_acc >= acc:
            return False
    return True


def _select(points: list[dict[str, Any]], target: float | None) -> tuple[dict[str, Any] | None, str]:
    usable = [
        point
        for point in points
        if point.get("bfcl_accuracy") is not None
        and point.get("trigger_rate") is not None
    ]
    if not usable:
        return None, "no_usable_threshold"
    if target is not None:
        candidates = [point for point in usable if point["bfcl_accuracy"] >= target]
        if candidates:
            return min(
                candidates,
                key=lambda point: (
                    point["trigger_rate"],
                    -point["bfcl_accuracy"],
                    point["threshold"],
                ),
            ), "min_trigger_at_oracle_target"
    return max(
        usable,
        key=lambda point: (
            point["bfcl_accuracy"],
            -point["trigger_rate"],
            -point["threshold"],
        ),
    ), "max_bfcl_then_min_trigger"


def _threshold_label(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    thresholds = [
        float(item)
        for item in args.thresholds.split(",")
        if item.strip()
    ]
    fold_thresholds: dict[str, Any] = {}
    selected_rows: list[dict[str, Any]] = []
    pareto_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "folds": args.folds,
        "threshold_selection": (
            "train_closed_loop_min_trigger_at_70pct_train_oracle_bfcl"
        ),
        "fold_thresholds": fold_thresholds,
    }

    for fold in range(args.folds):
        fold_dir = run_root / "logistic_train_sweep" / f"fold_{fold}"
        oracle_root = fold_dir / "oracle"
        oracle_acc, _, _ = _score_summary(oracle_root)
        target70 = oracle_acc * 0.70 if oracle_acc is not None else None
        target95 = oracle_acc * 0.95 if oracle_acc is not None else None
        points: list[dict[str, Any]] = []
        for threshold in thresholds:
            root = fold_dir / f"threshold_{_threshold_label(threshold)}"
            acc, correct, total = _score_summary(root)
            trigger = _trigger_rate(root)
            points.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "bfcl_accuracy": acc,
                    "correct": correct,
                    "total": total,
                    "trigger_rate": trigger,
                }
            )
        selected70, rule70 = _select(points, target70)
        selected95, _ = _select(points, target95)

        for point in points:
            is_pareto = _is_pareto(point, points)
            pareto_rows.append(
                {
                    "Detector": "combined_logistic",
                    "Fold": fold,
                    "Threshold": point["threshold"],
                    "Trigger Rate": point["trigger_rate"],
                    "BFCL Accuracy": point["bfcl_accuracy"],
                    "Train Oracle BFCL Accuracy": oracle_acc,
                    "Oracle 70 Target": target70,
                    "Oracle 95 Target": target95,
                    "Meets Oracle 70": (
                        point["bfcl_accuracy"] >= target70
                        if target70 is not None and point["bfcl_accuracy"] is not None
                        else None
                    ),
                    "Meets Oracle 95": (
                        point["bfcl_accuracy"] >= target95
                        if target95 is not None and point["bfcl_accuracy"] is not None
                        else None
                    ),
                    "Is Pareto": is_pareto,
                    "Selected 70": point is selected70,
                    "Selected 95": point is selected95,
                    "Selected Final": point is selected70,
                }
            )

        selected_threshold = (
            float(selected70["threshold"]) if selected70 is not None else 0.5
        )
        fold_thresholds[str(fold)] = {
            "combined_logistic": selected_threshold,
            "combined_logistic_fixed": selected_threshold,
            "selection_rule": rule70,
            "train_oracle_bfcl_accuracy": oracle_acc,
            "oracle_70_target": target70,
            "oracle_95_target": target95,
            "selected_train_bfcl_accuracy": (
                selected70.get("bfcl_accuracy") if selected70 else None
            ),
            "selected_train_trigger_rate": (
                selected70.get("trigger_rate") if selected70 else None
            ),
            "selected_95_threshold": (
                selected95.get("threshold") if selected95 else None
            ),
            "selected_95_train_bfcl_accuracy": (
                selected95.get("bfcl_accuracy") if selected95 else None
            ),
            "selected_95_train_trigger_rate": (
                selected95.get("trigger_rate") if selected95 else None
            ),
        }
        selected_rows.append(
            {
                "fold": fold,
                "selected_threshold": selected_threshold,
                "selection_rule": rule70,
                "train_bfcl_accuracy": (
                    selected70.get("bfcl_accuracy") if selected70 else None
                ),
                "train_trigger_rate": (
                    selected70.get("trigger_rate") if selected70 else None
                ),
                "train_oracle_bfcl_accuracy": oracle_acc,
                "oracle_70_target": target70,
                "oracle_95_target": target95,
                "selected_95_threshold": (
                    selected95.get("threshold") if selected95 else None
                ),
                "selected_95_train_bfcl_accuracy": (
                    selected95.get("bfcl_accuracy") if selected95 else None
                ),
                "selected_95_train_trigger_rate": (
                    selected95.get("trigger_rate") if selected95 else None
                ),
            }
        )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(run_root / "detector_bfcl_trigger_pareto.csv", pareto_rows, PARETO_FIELDS)
    _write_csv(
        run_root / "detector_bfcl_trigger_selected_thresholds.csv",
        selected_rows,
        SELECTED_FIELDS,
    )
    print(output_json)
    print(run_root / "detector_bfcl_trigger_pareto.csv")


if __name__ == "__main__":
    main()
