from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


PARETO_FIELDS = [
    "fold",
    "threshold",
    "train_closed_loop_bfcl",
    "train_trigger_rate",
    "train_oracle_bfcl",
    "target_bfcl",
    "calls_per_committed_step",
    "is_pareto",
    "pareto_selected",
    "selection_rule",
]

SELECTED_FIELDS = [
    "fold",
    "selected_threshold",
    "selected_C",
    "selected_l1_ratio",
    "train_oracle_bfcl",
    "selected_train_bfcl",
    "selected_train_trigger_rate",
    "target_fraction",
    "target_bfcl",
    "selection_rule",
    "num_train_segments",
    "num_positive_train",
    "num_unique_logistic_scores",
    "score_min",
    "score_max",
    "score_p10",
    "score_p25",
    "score_p50",
    "score_p75",
    "score_p90",
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
    return sum(int(bool(seg.get("repair_triggered"))) for seg in segments) / len(segments)


def _calls_per_step(root: Path) -> float | None:
    details = _read_jsonl(root / "logs" / "details.jsonl")
    committed_steps = sum(
        len(row.get("drift_steps") or [])
        for row in details
        if isinstance(row, dict)
    )
    segments = [
        seg
        for row in details
        for seg in (row.get("repair_segments") or [])
        if isinstance(seg, dict)
    ]
    candidate_steps = sum(int(seg.get("segment_length") or 0) for seg in segments)
    regenerated_steps = sum(
        int(seg.get("segment_length") or 0)
        for seg in segments
        if seg.get("repair_triggered")
    )
    return (candidate_steps + regenerated_steps) / committed_steps if committed_steps else None


def _threshold_label(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _is_pareto(point: dict[str, Any], points: list[dict[str, Any]]) -> bool:
    acc = point.get("train_closed_loop_bfcl")
    trigger = point.get("train_trigger_rate")
    if acc is None or trigger is None:
        return False
    for other in points:
        if other is point:
            continue
        other_acc = other.get("train_closed_loop_bfcl")
        other_trigger = other.get("train_trigger_rate")
        if other_acc is None or other_trigger is None:
            continue
        if other_trigger < trigger and other_acc >= acc:
            return False
    return True


def _select(
    points: list[dict[str, Any]],
    *,
    target_bfcl: float | None,
) -> tuple[dict[str, Any] | None, str]:
    usable = [
        point
        for point in points
        if point.get("train_closed_loop_bfcl") is not None
        and point.get("train_trigger_rate") is not None
    ]
    if not usable:
        return None, "no_usable_threshold"
    if target_bfcl is not None:
        candidates = [
            point for point in usable if point["train_closed_loop_bfcl"] >= target_bfcl
        ]
        if candidates:
            return min(
                candidates,
                key=lambda row: (
                    row["train_trigger_rate"],
                    -row["train_closed_loop_bfcl"],
                    -row["threshold"],
                ),
            ), "min_trigger_at_train_oracle_target"
    return max(
        usable,
        key=lambda row: (
            row["train_closed_loop_bfcl"],
            -row["train_trigger_rate"],
            -row["threshold"],
        ),
    ), "max_bfcl_then_min_trigger"


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
    parser.add_argument("--target-fraction", type=float, default=0.90)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    cv_dir = Path(args.cv_dir)
    fold_thresholds: dict[str, Any] = {}
    pareto_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for fold in range(args.folds):
        fold_cv = cv_dir / f"fold_{fold}"
        model = _read_json(fold_cv / "combined_logistic_v2_model.json")
        threshold_text = (fold_cv / "threshold_candidates.txt").read_text(
            encoding="utf-8"
        )
        thresholds = [
            float(item)
            for item in threshold_text.replace("\n", ",").split(",")
            if item.strip()
        ]
        train_root = run_root / "logistic_v2_train_sweep" / f"fold_{fold}"
        oracle_acc, _, _ = _score_summary(train_root / "oracle")
        target_bfcl = oracle_acc * args.target_fraction if oracle_acc is not None else None

        points: list[dict[str, Any]] = []
        for threshold in thresholds:
            root = train_root / f"threshold_{_threshold_label(threshold)}"
            acc, _, _ = _score_summary(root)
            points.append(
                {
                    "fold": fold,
                    "threshold": threshold,
                    "train_closed_loop_bfcl": acc,
                    "train_trigger_rate": _trigger_rate(root),
                    "train_oracle_bfcl": oracle_acc,
                    "target_bfcl": target_bfcl,
                    "calls_per_committed_step": _calls_per_step(root),
                }
            )

        for point in points:
            point["is_pareto"] = _is_pareto(point, points)
        selected, selection_rule = _select(points, target_bfcl=target_bfcl)
        selected_threshold = float(selected["threshold"]) if selected else 0.5
        for point in points:
            point["pareto_selected"] = bool(point is selected)
            point["selection_rule"] = selection_rule
            pareto_rows.append(point)

        selected_row = {
            "fold": fold,
            "selected_threshold": selected_threshold,
            "selected_C": model.get("selected_C"),
            "selected_l1_ratio": model.get("selected_l1_ratio"),
            "train_oracle_bfcl": oracle_acc,
            "selected_train_bfcl": (
                selected.get("train_closed_loop_bfcl") if selected else None
            ),
            "selected_train_trigger_rate": (
                selected.get("train_trigger_rate") if selected else None
            ),
            "target_fraction": args.target_fraction,
            "target_bfcl": target_bfcl,
            "selection_rule": selection_rule,
            "num_train_segments": model.get("train_rows"),
            "num_positive_train": None,
        }
        diag_csv = cv_dir / "combined_logistic_v2_training_diagnostics.csv"
        if diag_csv.exists():
            with open(diag_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if str(row.get("fold")) == str(fold):
                        for key in (
                            "num_positive_train",
                            "num_unique_logistic_scores",
                            "score_min",
                            "score_max",
                            "score_p10",
                            "score_p25",
                            "score_p50",
                            "score_p75",
                            "score_p90",
                        ):
                            selected_row[key] = row.get(key)
                        break
        selected_rows.append(selected_row)
        updated_model = {
            **model,
            "threshold": selected_threshold,
            "threshold_selection_rule": selection_rule,
            "train_oracle_bfcl": oracle_acc,
            "selected_train_bfcl": selected_row["selected_train_bfcl"],
            "selected_train_trigger_rate": selected_row["selected_train_trigger_rate"],
            "target_fraction": args.target_fraction,
            "target_bfcl": target_bfcl,
        }
        (fold_cv / "combined_logistic_v2_selected_model.json").write_text(
            json.dumps(updated_model, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fold_thresholds[str(fold)] = {
            "combined_logistic_v2": selected_threshold,
            **selected_row,
            "test_episode_ids": model.get("test_episode_ids") or [],
            "train_episode_ids": model.get("train_episode_ids") or [],
        }

    payload = {
        "detector": "combined_logistic_v2",
        "folds": args.folds,
        "target_fraction": args.target_fraction,
        "threshold_selection": "train_closed_loop_bfcl_trigger_pareto",
        "fold_thresholds": fold_thresholds,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        run_root / "combined_logistic_v2_threshold_sweep.csv",
        pareto_rows,
        PARETO_FIELDS,
    )
    _write_csv(
        run_root / "combined_logistic_v2_selected_thresholds.csv",
        selected_rows,
        SELECTED_FIELDS,
    )
    print(output_json)
    print(run_root / "combined_logistic_v2_threshold_sweep.csv")


if __name__ == "__main__":
    main()
