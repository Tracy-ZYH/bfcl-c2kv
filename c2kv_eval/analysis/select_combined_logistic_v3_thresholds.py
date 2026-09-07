from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from c2kv_eval.analysis.select_combined_logistic_v2_thresholds import (
    _calls_per_step,
    _read_jsonl,
    _score_summary,
    _trigger_rate,
)


def _label(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_")


def _imputation_rate(root: Path) -> float | None:
    values = [
        float(seg["logistic_detector_imputation_rate"])
        for row in _read_jsonl(root / "logs" / "details.jsonl")
        for seg in (row.get("repair_segments") or [])
        if isinstance(seg, dict) and seg.get("logistic_detector_imputation_rate") is not None
    ]
    return sum(values) / len(values) if values else None


def _segment_diagnostics(root: Path) -> dict[str, Any]:
    details = _read_jsonl(root / "logs" / "details.jsonl")
    segments = [
        seg for row in details for seg in (row.get("repair_segments") or [])
        if isinstance(seg, dict)
    ]
    triggered = [seg for seg in segments if seg.get("repair_triggered")]
    harmful = [seg for seg in segments if seg.get("oracle_reference_drift_segment")]
    summary_path = root / "logs" / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    attempts = summary.get("tp_recovery_attempts")
    successes = summary.get("tp_recovery_success_count")
    recovery_success = (
        float(successes) / float(attempts)
        if attempts not in (None, 0) and successes is not None else None
    )
    runtime_values = [
        float((row.get("c2kv_drift_metrics") or {}).get("episode_e2e_observed_seconds"))
        for row in details
        if (row.get("c2kv_drift_metrics") or {}).get("episode_e2e_observed_seconds") is not None
    ]
    return {
        "online_reference_error_rate": len(harmful) / len(segments) if segments else None,
        "recovery_success": recovery_success,
        "precision": summary.get("detector_precision"),
        "recall": summary.get("detector_recall"),
        "f1": summary.get("detector_f1"),
        "fpr": summary.get("detector_fpr"),
        "runtime_seconds": sum(runtime_values) if runtime_values else None,
    }


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cv-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-fraction", type=float, default=0.90)
    args = parser.parse_args()
    run_root, cv_dir = Path(args.run_root), Path(args.cv_dir)
    output_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    train_coverage = {
        (int(row["fold"]), row["split"], row["feature_name"]): row
        for row in _csv_rows(cv_dir / "feature_coverage_train_calibration.csv")
    }

    for fold in range(args.folds):
        fold_dir = cv_dir / f"fold_{fold}"
        model = json.loads((fold_dir / "combined_logistic_v3_model.json").read_text())
        offline_rows = _csv_rows(fold_dir / "shadow_threshold_sweep.csv")
        online_shadow = json.loads(
            (fold_dir / "online_shadow_score_quantiles.json").read_text()
        ) if (fold_dir / "online_shadow_score_quantiles.json").exists() else {}
        candidate_defs = online_shadow.get("closed_loop_candidates") or []
        if not candidate_defs:
            raise RuntimeError(
                f"missing closed_loop_candidates in {fold_dir / 'online_shadow_score_quantiles.json'}"
            )
        online_root = run_root / "logistic_v3_calibration" / f"fold_{fold}"
        oracle_bfcl, _, _ = _score_summary(online_root / "oracle")
        oracle_trigger_rate = _trigger_rate(online_root / "oracle")
        target = oracle_bfcl * args.target_fraction if oracle_bfcl is not None else None
        candidates: list[dict[str, Any]] = []
        for candidate_def in candidate_defs:
            threshold = float(candidate_def["threshold"])
            shadow = min(
                offline_rows,
                key=lambda row: abs(float(row["threshold"]) - threshold),
            )
            root = online_root / f"threshold_{_label(threshold)}"
            bfcl, _, _ = _score_summary(root)
            segment_diag = _segment_diagnostics(root)
            online_segments = [
                seg for detail in _read_jsonl(root / "logs" / "details.jsonl")
                for seg in (detail.get("repair_segments") or []) if isinstance(seg, dict)
            ]
            for feature_name in model.get("feature_names") or []:
                missing = sum(
                    int(feature_name in (seg.get("logistic_detector_missing_features") or []))
                    for seg in online_segments
                )
                train_row = train_coverage.get((fold, "model_train", feature_name), {})
                calibration_row = train_coverage.get((fold, "calibration", feature_name), {})
                coverage_rows.append({
                    "fold": fold, "threshold_name": candidate_def["threshold_name"],
                    "threshold": threshold, "feature_name": feature_name,
                    "train_non_missing_rate": train_row.get("non_missing_rate"),
                    "calibration_non_missing_rate": calibration_row.get("non_missing_rate"),
                    "online_segments": len(online_segments), "online_missing": missing,
                    "online_non_missing_rate": (
                        1 - missing / len(online_segments) if online_segments else None
                    ),
                    "online_imputation_rate": (
                        missing / len(online_segments) if online_segments else None
                    ),
                    "coefficient": model.get("coef", [])[list(model.get("feature_names") or []).index(feature_name)],
                    "training_imputation_value": (model.get("impute_values") or {}).get(feature_name),
                    "training_mean": (model.get("means") or {}).get(feature_name),
                    "training_std": (model.get("scales") or {}).get(feature_name),
                })
            candidates.append({
                "fold": fold,
                "threshold_name": candidate_def["threshold_name"],
                "target_trigger_rate": candidate_def["target_trigger_rate"],
                "shadow_trigger_rate": candidate_def["shadow_trigger_rate"],
                "threshold": threshold,
                "offline_precision": shadow.get("precision"),
                "offline_recall": shadow.get("recall"),
                "offline_f1": shadow.get("f1"),
                "offline_fpr": shadow.get("fpr"),
                "offline_trigger_rate": shadow.get("trigger_rate"),
                "online_bfcl_acc": bfcl,
                "online_trigger_rate": _trigger_rate(root),
                "actual_online_trigger_rate": _trigger_rate(root),
                "calls_per_committed_step": _calls_per_step(root),
                "feature_imputation_rate": _imputation_rate(root),
                **segment_diag,
                "oracle_bfcl": oracle_bfcl,
                "oracle_calibration_trigger_rate": oracle_trigger_rate,
                "target_bfcl": target,
                "detector_name": "Reference-Drift Logistic Detector",
                "online_score_min": online_shadow.get("score_min"),
                "online_score_p10": online_shadow.get("score_p10"),
                "online_score_p25": online_shadow.get("score_p25"),
                "online_score_p50": online_shadow.get("score_p50"),
                "online_score_p75": online_shadow.get("score_p75"),
                "online_score_p90": online_shadow.get("score_p90"),
                "online_score_max": online_shadow.get("score_max"),
            })
        usable = [r for r in candidates if r["online_bfcl_acc"] is not None and r["online_trigger_rate"] is not None]
        if usable:
            selected = max(
                usable,
                key=lambda row: (
                    row["online_bfcl_acc"],
                    -abs(row["online_trigger_rate"] - oracle_trigger_rate)
                    if oracle_trigger_rate is not None else -row["online_trigger_rate"],
                ),
            )
            rule = "max_calibration_bfcl_then_closest_oracle_calibration_trigger"
        else:
            selected = candidates[0]
            rule = "no_usable_closed_loop_result"
        for row in candidates:
            row["selected"] = row is selected
            row["selection_rule"] = rule
            output_rows.append(row)
        selected_model = {**model, "threshold": selected["threshold"], "threshold_name": selected["threshold_name"], "threshold_selection_rule": rule, "calibration_oracle_bfcl": oracle_bfcl, "calibration_target_bfcl": target}
        (fold_dir / "combined_logistic_v3_selected_model.json").write_text(json.dumps(selected_model, indent=2) + "\n")

        test_root = run_root / "detector_combined_logistic_v3" / f"fold_{fold}"
        if test_root.exists():
            test_bfcl, _, _ = _score_summary(test_root)
            test_diag = _segment_diagnostics(test_root)
            for row in candidates:
                row["heldout_bfcl_acc"] = test_bfcl if row["selected"] else None
                row["heldout_trigger_rate"] = _trigger_rate(test_root) if row["selected"] else None
                row["heldout_feature_imputation_rate"] = _imputation_rate(test_root) if row["selected"] else None
                row["heldout_reference_error_rate"] = test_diag["online_reference_error_rate"] if row["selected"] else None
                row["heldout_recovery_success"] = test_diag["recovery_success"] if row["selected"] else None
                row["heldout_runtime_seconds"] = test_diag["runtime_seconds"] if row["selected"] else None

    fields = sorted({key for row in output_rows for key in row})
    with (run_root / "combined_logistic_v3_threshold_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(output_rows)
    if coverage_rows:
        coverage_fields = sorted({key for row in coverage_rows for key in row})
        with (run_root / "combined_logistic_v3_online_feature_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=coverage_fields)
            writer.writeheader(); writer.writerows(coverage_rows)
    print(run_root / "combined_logistic_v3_threshold_diagnostics.csv")


if __name__ == "__main__":
    main()
