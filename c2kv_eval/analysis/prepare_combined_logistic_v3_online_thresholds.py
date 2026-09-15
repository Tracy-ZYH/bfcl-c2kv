from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path


def _operating_points(scores: list[float]) -> list[dict[str, float | int]]:
    """Enumerate distinct score>=threshold trigger behaviours."""
    points = []
    for threshold in sorted(set(scores)):
        count = sum(score >= threshold for score in scores)
        points.append({
            "threshold": threshold,
            "trigger_count": count,
            "shadow_trigger_rate": count / len(scores),
        })
    points.append({
        "threshold": math.nextafter(max(scores), math.inf),
        "trigger_count": 0,
        "shadow_trigger_rate": 0.0,
    })
    return points


def _nearest(points: list[dict[str, float | int]], target: float) -> dict[str, float | int]:
    return min(
        points,
        key=lambda point: (
            abs(float(point["shadow_trigger_rate"]) - target),
            abs(float(point["shadow_trigger_rate"]) - 0.5),
            float(point["threshold"]),
        ),
    )


def _distinct_candidates(
    points: list[dict[str, float | int]], targets: list[float]
) -> list[dict[str, float | int | str]]:
    """Assign targets to distinct achievable trigger counts with minimum error."""
    unique_points = list({int(point["trigger_count"]): point for point in points}.values())
    if len(unique_points) < len(targets):
        chosen = [_nearest(unique_points, target) for target in targets]
    else:
        best = None
        for assignment in itertools.permutations(unique_points, len(targets)):
            errors = [
                abs(float(point["shadow_trigger_rate"]) - target)
                for target, point in zip(targets, assignment)
            ]
            rank = (sum(errors), max(errors))
            if best is None or rank < best[0]:
                best = (rank, assignment)
        assert best is not None
        chosen = list(best[1])
    return [
        {
            "threshold_name": f"target_{target:.2f}",
            "target_trigger_rate": target,
            **point,
        }
        for target, point in zip(targets, chosen)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-root", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--target-rates", default="0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--closed-loop-rates", default="0.45,0.50,0.55")
    args = parser.parse_args()
    model = json.loads((Path(args.fold_dir) / "combined_logistic_v3_model.json").read_text())
    details = Path(args.shadow_root) / "logs" / "details.jsonl"
    scores = []
    detector_trigger_count = 0
    recovery_count = 0
    with details.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for seg in row.get("repair_segments") or []:
                if model.get("detector_variant") in {"trained", "rule"} and seg.get("logistic_detector_missing_features"):
                    raise RuntimeError(f"online-safe coverage check failed: {seg['logistic_detector_missing_features']}")
                if seg.get("logistic_detector_score") is not None:
                    scores.append(float(seg["logistic_detector_score"]))
                detector_trigger_count += int(bool(seg.get("detector_trigger")))
                recovery_count += int(bool(seg.get("repair_triggered")))
    if not scores:
        raise RuntimeError(f"no logistic_detector_score found in {details}")
    if detector_trigger_count or recovery_count:
        raise RuntimeError(
            "online shadow must not alter trajectories: "
            f"detector_trigger_count={detector_trigger_count}, "
            f"recovery_count={recovery_count}, details={details}"
        )
    scores.sort()
    points = _operating_points(scores)
    target_rates = [float(x) for x in args.target_rates.split(",") if x.strip()]
    quantiles = {}
    for target in target_rates:
        point = _nearest(points, target)
        quantiles[f"target_{target:.2f}"] = {
            "target_trigger_rate": target,
            **point,
        }
    closed_loop_candidates = _distinct_candidates(points, [float(v) for v in args.closed_loop_rates.split(",")])
    # A discrete rule may offer fewer behaviours than target budgets. Never
    # rerun the same output directory or pretend ties meet separate budgets.
    closed_loop_candidates = list({float(item["threshold"]): item for item in closed_loop_candidates}.values())
    fold_dir = Path(args.fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "threshold_candidates.txt").write_text(
        ",".join(
            f'{float(item["threshold"]):.17g}' for item in closed_loop_candidates
        ) + "\n",
        encoding="utf-8",
    )
    payload = {
        "label_mode": "reference_drift",
        "detector_name": json.loads((fold_dir / "combined_logistic_v3_model.json").read_text()).get("detector_name", "Reference-Drift Logistic Detector"),
        "score_count": len(scores),
        "shadow_detector_trigger_count": detector_trigger_count,
        "shadow_recovery_count": recovery_count,
        "score_min": min(scores),
        "score_p10": statistics.quantiles(scores, n=10, method="inclusive")[0] if len(scores) > 1 else scores[0],
        "score_p25": statistics.quantiles(scores, n=4, method="inclusive")[0] if len(scores) > 1 else scores[0],
        "score_p50": statistics.median(scores),
        "score_p75": statistics.quantiles(scores, n=4, method="inclusive")[2] if len(scores) > 1 else scores[0],
        "score_p90": statistics.quantiles(scores, n=10, method="inclusive")[8] if len(scores) > 1 else scores[0],
        "score_max": max(scores),
        "target_thresholds": quantiles,
        "closed_loop_candidates": closed_loop_candidates,
        "note": "Shadow scoring only screens thresholds; it does not replace closed-loop BFCL evaluation.",
    }
    (fold_dir / "online_shadow_score_quantiles.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (fold_dir / "online_shadow_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["score"]); writer.writerows([[x] for x in scores])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
