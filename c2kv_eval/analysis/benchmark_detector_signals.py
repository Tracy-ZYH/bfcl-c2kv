#!/usr/bin/env python3
"""Offline detector-signal benchmark for frozen BFCL C2KV segments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _segment_harmful(row: dict[str, Any]) -> bool:
    actions = row.get("candidate_action_drift_per_step") or []
    states = row.get("candidate_state_drift_per_step") or []
    harmful = row.get("oracle_harmful_drift_per_step") or row.get(
        "harmful_drift_per_step"
    ) or []
    max_len = max(len(actions), len(states), len(harmful))
    for index in range(max_len):
        if index < len(actions) and bool(actions[index]):
            return True
        if index < len(states) and bool(states[index]):
            return True
        if index < len(harmful) and bool(harmful[index]):
            return True
    return False


def _merge_step_features(row: dict[str, Any]) -> list[dict[str, float]]:
    detector = row.get("candidate_detector_features_per_step") or []
    heuristic = row.get("heuristic_attributes_per_step") or []
    max_len = max(len(detector), len(heuristic))
    out: list[dict[str, float]] = []
    for index in range(max_len):
        merged: dict[str, float] = {}
        for source in (
            detector[index] if index < len(detector) and isinstance(detector[index], dict) else {},
            heuristic[index] if index < len(heuristic) and isinstance(heuristic[index], dict) else {},
        ):
            for key, value in source.items():
                numeric = _as_float(value)
                if numeric is not None:
                    merged[key] = numeric
        out.append(merged)
    return out


def _aggregate_features(step_features: list[dict[str, float]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for features in step_features:
        for key, value in features.items():
            values.setdefault(key, []).append(value)
    out: dict[str, float] = {}
    for key, vals in values.items():
        if not vals:
            continue
        out[f"mean_{key}"] = sum(vals) / len(vals)
        out[f"max_{key}"] = max(vals)
        out[f"min_{key}"] = min(vals)
    return out


def _as_bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    return None


def _reconstruct_rule_detector(
    step_features: list[dict[str, float]],
    threshold: float,
) -> tuple[bool, float, str]:
    hard_trigger = any(bool(features.get("hard_error")) for features in step_features)
    grounding_trigger = any(
        bool(features.get("argument_grounding_failure")) for features in step_features
    )
    observation_trigger = any(
        float(features.get("observation_anomaly") or 0.0) >= 1.0
        for features in step_features
    )
    max_risk = max(
        (float(features.get("risk_score") or 0.0) for features in step_features),
        default=0.0,
    )
    risk_trigger = max_risk >= threshold
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
    return (
        hard_trigger or grounding_trigger or observation_trigger or risk_trigger,
        max_risk,
        reason,
    )


def _feature_row(row: dict[str, Any], rule_detector_threshold: float) -> dict[str, Any]:
    step_features = _merge_step_features(row)
    features = _aggregate_features(step_features)
    reconstructed_trigger, reconstructed_max_risk, reconstructed_reason = (
        _reconstruct_rule_detector(step_features, rule_detector_threshold)
    )
    rule_trigger = _as_bool_or_none(row.get("rule_detector_trigger"))
    if rule_trigger is None:
        rule_trigger = reconstructed_trigger
    rule_reason = row.get("rule_detector_reason")
    if not rule_reason:
        rule_reason = reconstructed_reason
    features.update(
        {
            "id": row.get("id") or row.get("sample_id"),
            "checkpoint_id": row.get("checkpoint_id"),
            "turn": row.get("turn"),
            "segment_start_step": row.get("segment_start_step"),
            "segment_length": row.get("segment_length")
            or row.get("speculative_steps"),
            "segment_harmful": int(_segment_harmful(row)),
            "rule_detector_trigger": int(rule_trigger),
            "rule_detector_reason": rule_reason,
            "rule_detector_threshold": rule_detector_threshold,
        }
    )
    max_risk = _as_float(row.get("rule_detector_max_risk"))
    if max_risk is not None:
        features["rule_detector_max_risk"] = max_risk
    elif "max_risk_score" in features:
        features["rule_detector_max_risk"] = features["max_risk_score"]
    else:
        features["rule_detector_max_risk"] = reconstructed_max_risk
    features["rule_detector_binary_score"] = float(
        bool(features.get("rule_detector_trigger"))
    )
    return features


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


def _curve_points(labels: list[int], scores: list[float]) -> list[dict[str, float]]:
    thresholds = sorted(set(scores), reverse=True)
    points: list[dict[str, float]] = []
    positives = sum(labels)
    negatives = len(labels) - positives
    for threshold in thresholds:
        tp = fp = tn = fn = 0
        for label, score in zip(labels, scores):
            pred = score >= threshold
            if pred and label:
                tp += 1
            elif pred and not label:
                fp += 1
            elif not pred and label:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        points.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "fpr": fpr,
                "f1": f1,
            }
        )
    return points


def _auprc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or sum(labels) == 0:
        return None
    points = sorted(_curve_points(labels, scores), key=lambda row: row["recall"])
    area = 0.0
    last_recall = 0.0
    last_precision = 1.0
    for point in points:
        recall = point["recall"]
        precision = point["precision"]
        area += (recall - last_recall) * ((precision + last_precision) / 2.0)
        last_recall = recall
        last_precision = precision
    return area


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


def _score_for_feature(name: str, value: Any) -> float | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return -numeric if _low_is_bad(name) else numeric


def _signal_metrics(name: str, labels: list[int], raw_values: list[Any]) -> dict[str, Any] | None:
    filtered: list[tuple[int, float]] = []
    for label, value in zip(labels, raw_values):
        score = _score_for_feature(name, value)
        if score is not None:
            filtered.append((label, score))
    if len(filtered) < 2:
        return None
    y = [item[0] for item in filtered]
    scores = [item[1] for item in filtered]
    points = _curve_points(y, scores)
    best = max(points, key=lambda row: row["f1"]) if points else {}
    return {
        "signal": name,
        "n": len(filtered),
        "positives": sum(y),
        "direction": "low_is_bad" if _low_is_bad(name) else "high_is_bad",
        "auroc": _auroc(y, scores),
        "auprc": _auprc(y, scores),
        "best_threshold": best.get("threshold"),
        "best_f1": best.get("f1"),
        "best_precision": best.get("precision"),
        "best_recall": best.get("recall"),
        "best_fpr": best.get("fpr"),
        "recall_at_fpr_le_0_1": max(
            (row["recall"] for row in points if row["fpr"] <= 0.1),
            default=0.0,
        ),
        "recall_at_fpr_le_0_2": max(
            (row["recall"] for row in points if row["fpr"] <= 0.2),
            default=0.0,
        ),
    }


def _stable_split(sample_id: Any) -> str:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    return "calibration" if int(digest[:8], 16) % 100 < 60 else "test"


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _combined_logistic(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any] | None:
    train = [row for row in rows if row.get("split") == "calibration"]
    test = [row for row in rows if row.get("split") == "test"]
    if len(train) < 8 or len(test) < 4:
        return None
    usable = []
    for name in feature_names:
        vals = [_score_for_feature(name, row.get(name)) for row in train]
        vals = [value for value in vals if value is not None]
        if len(vals) >= max(4, len(train) // 2):
            usable.append(name)
    if not usable:
        return None

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in usable:
        vals = [_score_for_feature(name, row.get(name)) for row in train]
        vals = [value for value in vals if value is not None]
        mean = sum(vals) / len(vals)
        var = sum((value - mean) ** 2 for value in vals) / max(len(vals), 1)
        means[name] = mean
        stds[name] = math.sqrt(var) or 1.0

    def vector(row: dict[str, Any]) -> list[float]:
        out = [1.0]
        for name in usable:
            value = _score_for_feature(name, row.get(name))
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
            y = int(row["segment_harmful"])
            p = _sigmoid(sum(w * xi for w, xi in zip(weights, x)))
            for i, xi in enumerate(x):
                grad[i] += (p - y) * xi
        for i in range(len(weights)):
            grad[i] /= len(train)
            if i:
                grad[i] += l2 * weights[i]
            weights[i] -= lr * grad[i]

    labels = [int(row["segment_harmful"]) for row in test]
    scores = [
        _sigmoid(sum(w * xi for w, xi in zip(weights, vector(row))))
        for row in test
    ]
    result = _signal_metrics("combined_logistic_regression", labels, scores)
    if result is None:
        return None
    result.update(
        {
            "n_calibration": len(train),
            "n_test": len(test),
            "feature_count": len(usable),
            "features": ",".join(usable),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rule-detector-threshold", type=float, default=5.0)
    args = parser.parse_args()

    segments = _read_jsonl(Path(args.segments_path))
    rows = [
        _feature_row(row, args.rule_detector_threshold)
        for row in segments
    ]
    for row in rows:
        row["split"] = _stable_split(row.get("id"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = sorted({key for row in rows for key in row})
    with open(output_dir / "detector_features.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=feature_columns)
        writer.writeheader()
        writer.writerows(rows)

    labels = [int(row["segment_harmful"]) for row in rows]
    metric_rows: list[dict[str, Any]] = []
    candidate_signals = [
        key
        for key in feature_columns
        if key
        not in {
            "id",
            "checkpoint_id",
            "turn",
            "segment_start_step",
            "segment_length",
            "segment_harmful",
            "split",
            "rule_detector_reason",
        }
    ]
    for signal in candidate_signals:
        metrics = _signal_metrics(signal, labels, [row.get(signal) for row in rows])
        if metrics is not None:
            metric_rows.append(metrics)

    combined = _combined_logistic(rows, candidate_signals)
    if combined is not None:
        metric_rows.append(combined)

    metric_columns = sorted({key for row in metric_rows for key in row})
    with open(output_dir / "detector_comparison.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "segments": len(rows),
        "harmful_segments": sum(labels),
        "safe_segments": len(labels) - sum(labels),
        "segments_with_generation_logprobs": sum(
            1 for row in rows if _as_float(row.get("max_generation_token_count"))
        ),
        "segments_with_detector_signal_available": sum(
            1 for row in rows if _as_float(row.get("max_detector_signal_available"))
        ),
        "feature_csv": str(output_dir / "detector_features.csv"),
        "comparison_csv": str(output_dir / "detector_comparison.csv"),
    }
    with open(output_dir / "detector_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
