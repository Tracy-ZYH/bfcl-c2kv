from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
from pathlib import Path
from typing import Any


FEATURE_PRIORITY = [
    "mean_mean_logprob",
    "min_min_logprob",
    "max_tool_name_generation_nll",
    "max_argument_generation_nll",
    "max_generation_nll",
    "max_max_entropy",
    "min_min_top1_top2_margin",
    "min_min_top1_probability",
    "max_risk_score",
    "max_observation_anomaly",
    "max_hard_error",
    "max_argument_grounding_failure",
    "max_repeat_action_score",
    "max_tool_transition_anomaly",
    "min_argument_grounding_score",
    "mean_risk_score",
]

LOW_IS_BAD_PATTERNS = (
    "confidence",
    "probability",
    "logprob",
    "margin",
    "grounding_score",
)


def _episode_fold(sample_id: str, folds: int, *, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds


def _episode_bucket(sample_id: str, *, seed: int, salt: str, buckets: int = 1000) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _low_is_bad(name: str) -> bool:
    lowered = name.lower()
    return any(pattern in lowered for pattern in LOW_IS_BAD_PATTERNS)


def _score_for_feature(name: str, value: Any) -> float | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return -numeric if _low_is_bad(name) else numeric


def _label(row: dict[str, Any]) -> tuple[int | None, str, bool]:
    critical = _as_float(row.get("recovery_critical"))
    if critical is not None:
        return int(bool(critical)), "recovery_critical", False

    reference = _as_float(
        row.get("oracle_reference_drift_segment", row.get("segment_harmful"))
    )
    if reference is None:
        return None, "missing", True
    return int(bool(reference)), "reference_drift", True


def _load_ids(path: Path, max_examples: int) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return ids[:max_examples] if max_examples > 0 else ids


def _load_rows(path: Path, ids: set[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [
            row
            for row in csv.DictReader(f)
            if str(row.get("id")) in ids
        ]
    labeled: list[dict[str, Any]] = []
    for row in rows:
        label, label_mode, fallback = _label(row)
        if label is None:
            continue
        row["_label"] = label
        row["_label_mode"] = label_mode
        row["_label_fallback"] = fallback
        labeled.append(row)
    return labeled


def _feature_coverage(
    rows: list[dict[str, Any]],
    name: str,
) -> tuple[int, list[float]]:
    values = [
        value
        for value in (_score_for_feature(name, row.get(name)) for row in rows)
        if value is not None
    ]
    return len(values), values


def _select_features(rows: list[dict[str, Any]], *, max_features: int) -> tuple[list[str], list[str]]:
    if not rows:
        return [], FEATURE_PRIORITY
    fieldnames = set(rows[0].keys())
    selected: list[str] = []
    unavailable: list[str] = []
    min_count = max(4, len(rows) // 3)
    for name in FEATURE_PRIORITY:
        if name not in fieldnames:
            unavailable.append(name)
            continue
        count, values = _feature_coverage(rows, name)
        if count < min_count or not values:
            unavailable.append(name)
            continue
        if len({round(value, 12) for value in values}) <= 1:
            unavailable.append(name)
            continue
        selected.append(name)
        if len(selected) >= max_features:
            break
    return selected, unavailable


def _matrix(rows: list[dict[str, Any]], features: list[str], means: dict[str, float] | None = None) -> tuple[list[list[float]], dict[str, float]]:
    impute = dict(means or {})
    if means is None:
        for name in features:
            _, values = _feature_coverage(rows, name)
            impute[name] = sum(values) / len(values) if values else 0.0
    x: list[list[float]] = []
    for row in rows:
        vector: list[float] = []
        for name in features:
            value = _score_for_feature(name, row.get(name))
            if value is None:
                value = impute[name]
            vector.append(float(value))
        x.append(vector)
    return x, impute


def _safe_metric(fn: Any, y_true: list[int], scores: list[float]) -> float | None:
    if len(set(y_true)) < 2:
        return None
    try:
        return float(fn(y_true, scores))
    except Exception:
        return None


def _inner_split(episode_ids: list[str], *, seed: int, outer_fold: int) -> tuple[set[str], set[str]]:
    validation = {
        sample_id
        for sample_id in episode_ids
        if _episode_bucket(
            sample_id,
            seed=seed,
            salt=f"combined_logistic_v2_inner_{outer_fold}",
        )
        < 200
    }
    if not validation and episode_ids:
        validation = set(episode_ids[-max(1, len(episode_ids) // 5) :])
    if len(validation) == len(episode_ids) and len(episode_ids) > 1:
        validation = set(episode_ids[-max(1, len(episode_ids) // 5) :])
    train = set(episode_ids) - validation
    return train or set(episode_ids), validation or set(episode_ids)


def _score_distribution(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {
            "num_unique_logistic_scores": 0,
            "score_min": None,
            "score_max": None,
            "score_p10": None,
            "score_p25": None,
            "score_p50": None,
            "score_p75": None,
            "score_p90": None,
        }
    ordered = sorted(scores)

    def pct(q: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
        return float(ordered[idx])

    return {
        "num_unique_logistic_scores": len({round(value, 12) for value in scores}),
        "score_min": float(ordered[0]),
        "score_max": float(ordered[-1]),
        "score_p10": pct(0.10),
        "score_p25": pct(0.25),
        "score_p50": pct(0.50),
        "score_p75": pct(0.75),
        "score_p90": pct(0.90),
    }


def _candidate_thresholds(scores: list[float]) -> list[float]:
    base = [i / 100.0 for i in range(5, 100, 5)]
    if not scores:
        return base
    ordered = sorted(scores)
    quantiles = []
    for i in range(1, 20):
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * i / 20))))
        quantiles.append(float(ordered[idx]))
    thresholds = sorted(
        {
            round(min(0.999999, max(0.000001, value)), 6)
            for value in base + quantiles
        }
    )
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids-path", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=52)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-features", type=int, default=12)
    args = parser.parse_args()

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "combined_logistic_v2 training requires scikit-learn and numpy"
        ) from exc

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = _load_ids(Path(args.ids_path), args.max_examples)
    rows = _load_rows(Path(args.features_csv), set(ids))
    if not rows:
        raise RuntimeError(f"no labeled detector feature rows in {args.features_csv}")

    folds: dict[str, Any] = {
        "folds": args.folds,
        "seed": args.seed,
        "fold_assignment": {},
        "folds_detail": {},
    }
    for sample_id in ids:
        fold = _episode_fold(sample_id, args.folds, seed=args.seed)
        folds["fold_assignment"][sample_id] = fold
        folds["folds_detail"].setdefault(str(fold), []).append(sample_id)

    diagnostics: list[dict[str, Any]] = []
    pooled_test_labels: list[int] = []
    pooled_test_scores: list[float] = []
    for outer_fold in range(args.folds):
        fold_dir = out_dir / f"fold_{outer_fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        test_ids = sorted(folds["folds_detail"].get(str(outer_fold), []))
        outer_train_ids = sorted([sample_id for sample_id in ids if sample_id not in set(test_ids)])
        inner_train_ids, inner_val_ids = _inner_split(
            outer_train_ids,
            seed=args.seed,
            outer_fold=outer_fold,
        )

        outer_train = [row for row in rows if str(row.get("id")) in set(outer_train_ids)]
        inner_train = [row for row in rows if str(row.get("id")) in inner_train_ids]
        inner_val = [row for row in rows if str(row.get("id")) in inner_val_ids]
        outer_test = [row for row in rows if str(row.get("id")) in set(test_ids)]
        features, unavailable = _select_features(outer_train, max_features=args.max_features)
        if not features:
            raise RuntimeError(f"fold {outer_fold} has no usable causal features")

        best: dict[str, Any] | None = None
        for c_value in (0.01, 0.1, 1.0, 10.0):
            for l1_ratio in (0.0, 0.5, 1.0):
                train_x_raw, impute = _matrix(inner_train, features)
                val_x_raw, _ = _matrix(inner_val, features, impute)
                train_y = [int(row["_label"]) for row in inner_train]
                val_y = [int(row["_label"]) for row in inner_val]
                if len(set(train_y)) < 2:
                    continue
                scaler = StandardScaler()
                train_x = scaler.fit_transform(np.asarray(train_x_raw, dtype=float))
                val_x = scaler.transform(np.asarray(val_x_raw, dtype=float))
                clf = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    max_iter=5000,
                    C=c_value,
                    l1_ratio=l1_ratio,
                    random_state=args.seed + outer_fold,
                )
                clf.fit(train_x, np.asarray(train_y, dtype=int))
                val_scores = clf.predict_proba(val_x)[:, 1].tolist() if len(val_y) else []
                auroc = _safe_metric(roc_auc_score, val_y, val_scores)
                auprc = _safe_metric(average_precision_score, val_y, val_scores)
                rank = (
                    -1.0 if auroc is None else auroc,
                    -1.0 if auprc is None else auprc,
                    -len([value for value in clf.coef_[0].tolist() if abs(value) > 1e-9]),
                )
                candidate = {
                    "rank": rank,
                    "C": c_value,
                    "l1_ratio": l1_ratio,
                    "inner_auroc": auroc,
                    "inner_auprc": auprc,
                }
                if best is None or candidate["rank"] > best["rank"]:
                    best = candidate
        if best is None:
            best = {
                "C": 1.0,
                "l1_ratio": 0.5,
                "inner_auroc": None,
                "inner_auprc": None,
                "rank": (-1.0, -1.0, 0),
            }

        outer_x_raw, impute = _matrix(outer_train, features)
        outer_y = [int(row["_label"]) for row in outer_train]
        if len(set(outer_y)) < 2:
            raise RuntimeError(f"fold {outer_fold} outer-train has one class only")
        scaler = StandardScaler()
        outer_x = scaler.fit_transform(np.asarray(outer_x_raw, dtype=float))
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            C=float(best["C"]),
            l1_ratio=float(best["l1_ratio"]),
            random_state=args.seed + outer_fold,
        )
        clf.fit(outer_x, np.asarray(outer_y, dtype=int))
        train_scores = clf.predict_proba(outer_x)[:, 1].tolist()
        test_scores: list[float] = []
        test_y: list[int] = []
        if outer_test:
            test_x_raw, _ = _matrix(outer_test, features, impute)
            test_x = scaler.transform(np.asarray(test_x_raw, dtype=float))
            test_scores = clf.predict_proba(test_x)[:, 1].tolist()
            test_y = [int(row["_label"]) for row in outer_test]
            pooled_test_labels.extend(test_y)
            pooled_test_scores.extend(test_scores)

        model_json = {
            "version": "combined_logistic_v2",
            "fold": outer_fold,
            "feature_names": features,
            "feature_unavailable": unavailable,
            "means": {
                name: float(value)
                for name, value in zip(features, scaler.mean_.tolist())
            },
            "scales": {
                name: float(value) if float(value) != 0.0 else 1.0
                for name, value in zip(features, scaler.scale_.tolist())
            },
            "impute_values": {name: float(value) for name, value in impute.items()},
            "coef": [float(value) for value in clf.coef_[0].tolist()],
            "intercept": float(clf.intercept_[0]),
            "threshold": 0.5,
            "threshold_selection_rule": "pending_train_closed_loop_pareto",
            "selected_C": float(best["C"]),
            "selected_l1_ratio": float(best["l1_ratio"]),
            "inner_cv_metric": "auroc_then_auprc",
            "inner_cv_auroc": best.get("inner_auroc"),
            "inner_cv_auprc": best.get("inner_auprc"),
            "label_mode": rows[0].get("_label_mode") or "reference_drift",
            "label_fallback": any(bool(row.get("_label_fallback")) for row in outer_train),
            "train_rows": len(outer_train),
            "train_episode_ids": outer_train_ids,
            "test_episode_ids": test_ids,
            "feature_source_csv": str(Path(args.features_csv)),
        }
        (fold_dir / "combined_logistic_v2_model.json").write_text(
            json.dumps(model_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with open(fold_dir / "logistic_model.pkl", "wb") as f:
            pickle.dump(clf, f)
        with open(fold_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        thresholds = _candidate_thresholds(train_scores)
        (fold_dir / "threshold_candidates.txt").write_text(
            ",".join(f"{value:.6f}" for value in thresholds) + "\n",
            encoding="utf-8",
        )
        (fold_dir / "test_ids.txt").write_text(
            "\n".join(test_ids) + ("\n" if test_ids else ""),
            encoding="utf-8",
        )
        (fold_dir / "outer_train_ids.txt").write_text(
            "\n".join(outer_train_ids) + ("\n" if outer_train_ids else ""),
            encoding="utf-8",
        )
        diagnostics.append(
            {
                "fold": outer_fold,
                "selected_C": best["C"],
                "selected_l1_ratio": best["l1_ratio"],
                "inner_cv_auroc": best.get("inner_auroc"),
                "inner_cv_auprc": best.get("inner_auprc"),
                "num_train_segments": len(outer_train),
                "num_positive_train": sum(outer_y),
                "num_test_segments": len(outer_test),
                "feature_names": ",".join(features),
                "feature_unavailable": ",".join(unavailable),
                "label_mode": model_json["label_mode"],
                "label_fallback": model_json["label_fallback"],
                "positive_score_mean": (
                    statistics.mean(
                        score for score, label in zip(train_scores, outer_y) if label
                    )
                    if any(outer_y)
                    else None
                ),
                "positive_score_std": (
                    statistics.pstdev(
                        [score for score, label in zip(train_scores, outer_y) if label]
                    )
                    if sum(outer_y) > 1
                    else None
                ),
                "negative_score_mean": (
                    statistics.mean(
                        score for score, label in zip(train_scores, outer_y) if not label
                    )
                    if sum(1 for label in outer_y if not label)
                    else None
                ),
                "negative_score_std": (
                    statistics.pstdev(
                        [score for score, label in zip(train_scores, outer_y) if not label]
                    )
                    if sum(1 for label in outer_y if not label) > 1
                    else None
                ),
                **_score_distribution(train_scores),
            }
        )

    (out_dir / "detector_cv_folds.json").write_text(
        json.dumps(folds, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for row in diagnostics for key in row})
    with open(out_dir / "combined_logistic_v2_training_diagnostics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostics)

    pooled = {
        "heldout_segments": len(pooled_test_labels),
        "heldout_positives": sum(pooled_test_labels),
        "heldout_auroc": _safe_metric(roc_auc_score, pooled_test_labels, pooled_test_scores),
        "heldout_auprc": _safe_metric(average_precision_score, pooled_test_labels, pooled_test_scores),
    }
    (out_dir / "combined_logistic_v2_pooled_offline_diagnostics.json").write_text(
        json.dumps(pooled, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out_dir / "detector_cv_folds.json")


if __name__ == "__main__":
    main()
