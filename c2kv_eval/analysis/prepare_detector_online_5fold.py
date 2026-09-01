from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCALAR_DETECTORS = [
    "max_risk_score",
    "rule_detector_max_risk",
    "max_observation_anomaly",
    "mean_risk_score",
    "max_hard_error",
    "max_generation_nll",
    "mean_generation_nll",
]


def _episode_fold(sample_id: str, folds: int) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def _episode_bucket(sample_id: str, *, salt: str, buckets: int = 1000) -> int:
    digest = hashlib.sha256(f"{sample_id}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _low_is_bad(name: str) -> bool:
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


def _score(name: str, value: Any) -> float | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return -numeric if _low_is_bad(name) else numeric


def _best_f1_threshold(labels: list[int], scores: list[float]) -> float:
    best_threshold = 0.0
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


def _split_inner(episode_ids: list[str], fold: int) -> tuple[list[str], list[str]]:
    calibration = [
        sample_id
        for sample_id in episode_ids
        if _episode_bucket(sample_id, salt=f"inner_calibration_fold_{fold}") < 200
    ]
    if not calibration and episode_ids:
        calibration = episode_ids[-max(1, len(episode_ids) // 5) :]
    if len(calibration) == len(episode_ids) and len(episode_ids) > 1:
        calibration = episode_ids[-max(1, len(episode_ids) // 5) :]
    calibration_set = set(calibration)
    train = [sample_id for sample_id in episode_ids if sample_id not in calibration_set]
    return train or episode_ids, calibration or episode_ids


def _load_ids(path: Path, max_examples: int) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:max_examples] if max_examples > 0 else ids


def _load_features(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-path", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=52)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = _load_ids(Path(args.ids_path), args.max_examples)
    folds: dict[str, Any] = {
        "folds": args.folds,
        "fold_assignment": {},
        "folds_detail": {},
    }
    for sample_id in ids:
        fold = _episode_fold(sample_id, args.folds)
        folds["fold_assignment"][sample_id] = fold
        folds["folds_detail"].setdefault(str(fold), []).append(sample_id)

    features = [
        row
        for row in _load_features(Path(args.features_csv))
        if str(row.get("id")) in set(ids)
    ]
    thresholds: dict[str, Any] = {
        "folds": args.folds,
        "threshold_policy": "best_f1_on_inner_calibration_episodes",
        "fold_thresholds": {},
        "scalar_detectors": SCALAR_DETECTORS,
    }

    for fold in range(args.folds):
        test_ids = sorted(folds["folds_detail"].get(str(fold), []))
        pool_ids = sorted([sample_id for sample_id in ids if sample_id not in set(test_ids)])
        train_ids, calibration_ids = _split_inner(pool_ids, fold)
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "test_ids.txt").write_text(
            "\n".join(test_ids) + ("\n" if test_ids else ""),
            encoding="utf-8",
        )
        (fold_dir / "train_ids.txt").write_text(
            "\n".join(train_ids) + ("\n" if train_ids else ""),
            encoding="utf-8",
        )
        (fold_dir / "calibration_ids.txt").write_text(
            "\n".join(calibration_ids) + ("\n" if calibration_ids else ""),
            encoding="utf-8",
        )

        fold_thresholds: dict[str, Any] = {
            "test_episode_ids": test_ids,
            "train_episode_ids": train_ids,
            "calibration_episode_ids": calibration_ids,
        }
        calibration_rows = [
            row for row in features if str(row.get("id")) in set(calibration_ids)
        ]
        for detector in SCALAR_DETECTORS:
            labels: list[int] = []
            scores: list[float] = []
            for row in calibration_rows:
                score = _score(detector, row.get(detector))
                if score is None:
                    continue
                labels.append(int(float(row["segment_harmful"])))
                scores.append(score)
            fold_thresholds[detector] = (
                _best_f1_threshold(labels, scores) if scores else 0.0
            )
        thresholds["fold_thresholds"][str(fold)] = fold_thresholds
        (fold_dir / "thresholds.json").write_text(
            json.dumps(fold_thresholds, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    (out_dir / "detector_cv_folds.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "scalar_thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "detector_cv_folds.json")


if __name__ == "__main__":
    main()
