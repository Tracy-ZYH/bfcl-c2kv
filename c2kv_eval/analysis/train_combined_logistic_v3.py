from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from c2kv_eval.analysis.train_combined_logistic_v2 import (
    _as_float,
    _episode_bucket,
    _episode_fold,
    _inner_split,
    _label,
    _load_ids,
    _matrix,
    _safe_metric,
    _score_for_feature,
    _score_distribution,
    _select_features,
)


def _rows(path: Path, ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("id")) not in ids:
                continue
            label, mode, fallback = _label(row)
            if label is None:
                continue
            row["_label"] = label
            row["_label_mode"] = mode
            row["_label_fallback"] = fallback
            out.append(row)
    return out


def _metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    predicted = [score >= threshold for score in scores]
    tp = sum(int(pred and label) for pred, label in zip(predicted, labels))
    fp = sum(int(pred and not label) for pred, label in zip(predicted, labels))
    tn = sum(int(not pred and not label) for pred, label in zip(predicted, labels))
    fn = sum(int(not pred and label) for pred, label in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "trigger_count": tp + fp,
        "trigger_rate": (tp + fp) / len(labels) if labels else 0.0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tpr": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
    }


def _representatives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    high_pool = [row for row in rows if row["recall"] >= 0.90]
    if high_pool:
        high = max(high_pool, key=lambda r: (r["precision"], -r["fpr"], -r["trigger_rate"]))
    else:
        high = max(rows, key=lambda r: (r["recall"], r["precision"], -r["fpr"]))
    best = max(rows, key=lambda r: (r["f1"], r["precision"], -r["trigger_rate"]))
    positive = [row for row in rows if row["trigger_count"] > 0]
    low = max(positive or rows, key=lambda r: (r["precision"], -r["fpr"], -r["trigger_rate"], r["threshold"]))
    selected: list[dict[str, Any]] = []
    used_thresholds: set[float] = set()
    used_behaviors: set[tuple[int, int, int, int]] = set()
    ranked = {
        "high_recall": sorted(rows, key=lambda r: (r["recall"] >= 0.9, r["recall"], r["precision"], -r["fpr"], -r["trigger_rate"]), reverse=True),
        "best_f1": sorted(rows, key=lambda r: (r["f1"], r["precision"], -r["trigger_rate"], r["threshold"]), reverse=True),
        "low_trigger": sorted(positive or rows, key=lambda r: (r["precision"], -r["fpr"], -r["trigger_rate"], r["threshold"]), reverse=True),
    }
    preferred = {"high_recall": high, "best_f1": best, "low_trigger": low}
    for name in ("high_recall", "best_f1", "low_trigger"):
        candidates = [preferred[name], *ranked[name]]
        candidate = next(
            (
                row for row in candidates
                if row["threshold"] not in used_thresholds
                and (row["tp"], row["fp"], row["tn"], row["fn"]) not in used_behaviors
            ),
            next((row for row in candidates if row["threshold"] not in used_thresholds), preferred[name]),
        )
        used_thresholds.add(candidate["threshold"])
        used_behaviors.add((candidate["tp"], candidate["fp"], candidate["tn"], candidate["fn"]))
        selected.append({**candidate, "threshold_name": name})
    return selected


def _coverage(
    rows: list[dict[str, Any]],
    features: list[str],
    impute: dict[str, float],
    split: str,
    fold: int,
) -> list[dict[str, Any]]:
    output = []
    for name in features:
        values = [_score_for_feature(name, row.get(name)) for row in rows]
        present = [value for value in values if value is not None and math.isfinite(value)]
        output.append({
            "fold": fold,
            "split": split,
            "feature_name": name,
            "rows": len(rows),
            "non_missing": len(present),
            "non_missing_rate": len(present) / len(rows) if rows else None,
            "imputation_rate": 1 - len(present) / len(rows) if rows else None,
            "raw_mean": statistics.mean(present) if present else None,
            "raw_std": statistics.pstdev(present) if len(present) > 1 else 0.0 if present else None,
            "imputation_value": impute[name],
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-path", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=52)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--variant", choices=["legacy", "trained", "rule"], default="legacy")
    args = parser.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids = _load_ids(Path(args.ids_path), args.max_examples)
    all_rows = _rows(Path(args.features_csv), set(ids))
    diagnostics: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    split_payload: dict[str, Any] = {"version": "combined_logistic_v3", "folds": args.folds, "seed": args.seed, "folds_detail": {}}

    for fold in range(args.folds):
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        if args.folds == 1:
            ordered = sorted(ids, key=lambda i: _episode_bucket(i, seed=args.seed, salt="v3_fast_order"))
            test_count = max(2, round(len(ordered) * 0.17))
            calibration_count = max(3, round(len(ordered) * 0.25))
            test_ids = set(ordered[:test_count])
            calibration_ids = set(ordered[test_count:test_count + calibration_count])
            remaining = [i for i in ordered if i not in test_ids and i not in calibration_ids]
        else:
            test_ids = {i for i in ids if _episode_fold(i, args.folds, seed=args.seed) == fold}
            remaining = [i for i in ids if i not in test_ids]
            calibration_ids = {i for i in remaining if _episode_bucket(i, seed=args.seed, salt=f"v3_calibration_{fold}") < 250}
        if not calibration_ids and remaining:
            calibration_ids = set(remaining[-max(1, len(remaining) // 5):])
        model_ids = set(remaining) - calibration_ids
        if not test_ids and len(model_ids) > 2:
            test_ids = {sorted(model_ids)[-1]}; model_ids -= test_ids
        model_rows = [r for r in all_rows if str(r.get("id")) in model_ids]
        cal_rows = [r for r in all_rows if str(r.get("id")) in calibration_ids]
        test_rows = [r for r in all_rows if str(r.get("id")) in test_ids]
        features, unavailable = _select_features(model_rows, max_features=args.max_features)
        if args.variant == "trained":
            # Avoid composite risk + its components and complementary grounding
            # columns appearing together. Means preserve more resolution than max.
            features = ["mean_observation_anomaly", "mean_argument_grounding_failure",
                        "max_hard_error"]
        elif args.variant == "rule":
            features = ["max_risk_score"]
        if args.variant != "legacy":
            unavailable = []
            for row in model_rows + cal_rows:
                if any(_score_for_feature(f, row.get(f)) is None for f in features):
                    raise RuntimeError("detector variant requires complete online-safe features")
        if not features:
            raise RuntimeError(f"fold {fold}: no usable model-train features")

        inner_train_ids, inner_val_ids = _inner_split(sorted(model_ids), seed=args.seed, outer_fold=fold)
        inner_train = [r for r in model_rows if str(r.get("id")) in inner_train_ids]
        inner_val = [r for r in model_rows if str(r.get("id")) in inner_val_ids]
        best = None
        def weights(rows):
            from collections import Counter
            counts = Counter(str(r["id"]) for r in rows)
            return np.asarray([len(rows) / (len(counts) * counts[str(r["id"])]) for r in rows])

        for c_value in (() if args.variant == "rule" else (0.01, 0.1, 1.0, 10.0)):
            # Three preselected features do not need L1 selection. In this
            # small dataset L1 can collapse all coefficients to zero.
            for l1_ratio in ((0.0,) if args.variant == "trained" else (0.0, 0.5, 1.0)):
                train_x, inner_impute = _matrix(inner_train, features)
                val_x, _ = _matrix(inner_val, features, inner_impute)
                train_y = [int(r["_label"]) for r in inner_train]
                val_y = [int(r["_label"]) for r in inner_val]
                if len(set(train_y)) < 2 or not val_y:
                    continue
                inner_scaler = StandardScaler().fit(np.asarray(train_x))
                clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, C=c_value, l1_ratio=l1_ratio, random_state=args.seed + fold)
                clf.fit(inner_scaler.transform(np.asarray(train_x)), np.asarray(train_y),
                        sample_weight=weights(inner_train) if args.variant == "trained" else None)
                scores = clf.predict_proba(inner_scaler.transform(np.asarray(val_x)))[:, 1].tolist()
                rank = (_safe_metric(roc_auc_score, val_y, scores) or -1, _safe_metric(average_precision_score, val_y, scores) or -1)
                if best is None or rank > best["rank"]:
                    best = {"C": c_value, "l1_ratio": l1_ratio, "rank": rank}
        best = best or {"C": 1.0, "l1_ratio": 0.5, "rank": (-1, -1)}

        model_x, impute = _matrix(model_rows, features)
        model_y = [int(r["_label"]) for r in model_rows]
        if len(set(model_y)) < 2 and args.variant != "rule":
            raise RuntimeError(f"fold {fold}: model-train has one class")
        scaler = StandardScaler().fit(np.asarray(model_x))
        clf = LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000, C=best["C"], l1_ratio=best["l1_ratio"], random_state=args.seed + fold)
        cal_x, _ = _matrix(cal_rows, features, impute)
        cal_y = [int(r["_label"]) for r in cal_rows]
        if args.variant == "rule":
            # Fixed score: sigmoid(existing rule risk). No coefficient,
            # normalizer, imputation or feature selection is learned.
            scaler.mean_ = np.zeros(len(features))
            scaler.scale_ = np.ones(len(features))
            impute = {f: 0.0 for f in features}
            clf.coef_ = np.ones((1, len(features)))
            clf.intercept_ = np.zeros(1)
            cal_scores = [1 / (1 + math.exp(-float(row[0]))) for row in cal_x]
            best = {"C": None, "l1_ratio": None, "rank": (None, None)}
        else:
            clf.fit(scaler.transform(np.asarray(model_x)), np.asarray(model_y),
                    sample_weight=weights(model_rows) if args.variant == "trained" else None)
            cal_scores = clf.predict_proba(scaler.transform(np.asarray(cal_x)))[:, 1].tolist() if cal_rows else []
        dense = sorted({round(i / 100, 6) for i in range(1, 100)} | {round(s, 6) for s in cal_scores})
        shadow = [_metrics(cal_y, cal_scores, threshold) for threshold in dense]
        representatives = _representatives(shadow)

        model = {
            "version": "combined_logistic_v3", "fold": fold,
            "detector_variant": args.variant,
            "detector_name": "Fixed Rule Risk Detector" if args.variant == "rule" else "Reference-Drift Logistic Detector",
            "supervised_training": args.variant != "rule",
            "feature_names": features, "feature_unavailable": unavailable,
            "means": dict(zip(features, map(float, scaler.mean_))),
            "scales": dict(zip(features, [float(v) or 1.0 for v in scaler.scale_])),
            "impute_values": impute, "coef": list(map(float, clf.coef_[0])),
            "intercept": float(clf.intercept_[0]), "threshold": 0.5,
            "selected_C": best["C"], "selected_l1_ratio": best["l1_ratio"],
            "inner_cv_auroc": best["rank"][0], "inner_cv_auprc": best["rank"][1],
            "label_mode": all_rows[0]["_label_mode"],
            "model_train_episode_ids": sorted(model_ids),
            "calibration_episode_ids": sorted(calibration_ids),
            "test_episode_ids": sorted(test_ids),
            "train_episode_ids": sorted(model_ids),
            "train_rows": len(model_rows), "feature_source_csv": args.features_csv,
            "threshold_selection_rule": "pending_calibration_closed_loop",
        }
        (fold_dir / "combined_logistic_v3_model.json").write_text(json.dumps(model, indent=2) + "\n")
        for name, values in (("model_train_ids.txt", model_ids), ("calibration_ids.txt", calibration_ids), ("test_ids.txt", test_ids)):
            (fold_dir / name).write_text("\n".join(sorted(values)) + "\n")
        (fold_dir / "threshold_candidates.txt").write_text(",".join(str(r["threshold"]) for r in representatives) + "\n")
        _write_csv(fold_dir / "shadow_threshold_sweep.csv", shadow)
        _write_csv(fold_dir / "shadow_selected_thresholds.csv", representatives)
        coverage.extend(_coverage(model_rows, features, impute, "model_train", fold))
        coverage.extend(_coverage(cal_rows, features, impute, "calibration", fold))
        diagnostics.append({
            "fold": fold, "label_mode": model["label_mode"], "feature_names": ",".join(features),
            "coefficients": ",".join(map(str, model["coef"])), "C": best["C"], "l1_ratio": best["l1_ratio"],
            "model_train_episodes": len(model_ids), "calibration_episodes": len(calibration_ids), "test_episodes": len(test_ids),
            "model_train_segments": len(model_rows), "model_train_positive": sum(model_y),
            "calibration_segments": len(cal_rows), "calibration_positive": sum(cal_y),
            **_score_distribution(cal_scores),
        })
        split_payload["folds_detail"][str(fold)] = {"model_train_ids": sorted(model_ids), "calibration_ids": sorted(calibration_ids), "test_ids": sorted(test_ids)}

    (output / "detector_v3_splits.json").write_text(json.dumps(split_payload, indent=2) + "\n")
    _write_csv(output / "combined_logistic_v3_training_diagnostics.csv", diagnostics)
    _write_csv(output / "feature_coverage_train_calibration.csv", coverage)
    print(output / "detector_v3_splits.json")


if __name__ == "__main__":
    main()
