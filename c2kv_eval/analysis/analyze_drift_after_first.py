from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_MODE = "history_c2kv4_closed_loop"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_ids(path: str) -> set[str] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        }


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _step_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            lookup[(str(row["id"]), int(row["global_step"]))] = row
        except Exception:
            continue
    return lookup


def analyze(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    mode_root = run_root / args.mode
    details = _load_jsonl(mode_root / "logs" / "details.jsonl")
    step_metrics = _load_jsonl(mode_root / "logs" / "step_metrics.jsonl")
    selected_ids = _load_ids(args.ids_path)
    step_by_key = _step_lookup(step_metrics)
    offsets = list(range(args.before, args.after + 1))
    buckets = {
        offset: {
            "relative_step": offset,
            "samples": 0,
            "action_drift": 0,
            "state_drift": 0,
            "turn_failure": 0,
            "turn_failure_known": 0,
            "executable_tool": 0,
            "executable_tool_known": 0,
        }
        for offset in offsets
    }
    aligned_rows: list[dict[str, Any]] = []
    drift_sample_count = 0
    all_action_drift_ids = set()

    for row in details:
        sample_id = str(row.get("id"))
        if selected_ids is not None and sample_id not in selected_ids:
            continue
        metrics = row.get("c2kv_drift_metrics") or {}
        first = metrics.get("first_action_divergence")
        steps = row.get("drift_steps") or []
        if isinstance(steps, list):
            for step in steps:
                if (
                    step.get("candidate_action_drift") is True
                    or step.get("action_matches_reference") is False
                ):
                    all_action_drift_ids.add(sample_id)
                    if not isinstance(first, dict) or first.get("global_step") is None:
                        first = {
                            "turn": step.get("turn"),
                            "step": step.get("step"),
                            "global_step": step.get("global_step"),
                        }
                    break
        if not isinstance(first, dict) or first.get("global_step") is None:
            continue
        drift_step = int(first["global_step"])
        if not isinstance(steps, list):
            continue
        drift_sample_count += 1
        for offset in offsets:
            global_step = drift_step + offset
            if global_step < 0 or global_step >= len(steps):
                continue
            step = steps[global_step]
            metric = step_by_key.get((sample_id, global_step), {})
            action_drift = (
                step.get("candidate_action_drift") is True
                or step.get("action_matches_reference") is False
            )
            state_drift = (
                step.get("state_drift") is True
                or step.get("state_matches_reference") is False
            )
            state_pass = metric.get("state_pass_after_turn")
            response_pass = metric.get("response_pass_after_turn")
            turn_failure = None
            if state_pass is not None and response_pass is not None:
                turn_failure = not (bool(state_pass) and bool(response_pass))
            executable = metric.get("tool_call_parse_success")

            bucket = buckets[offset]
            bucket["samples"] += 1
            bucket["action_drift"] += int(action_drift)
            bucket["state_drift"] += int(state_drift)
            if turn_failure is not None:
                bucket["turn_failure_known"] += 1
                bucket["turn_failure"] += int(turn_failure)
            if executable is not None:
                bucket["executable_tool_known"] += 1
                bucket["executable_tool"] += int(bool(executable))

            aligned_rows.append(
                {
                    "id": sample_id,
                    "first_drift_global_step": drift_step,
                    "relative_step": offset,
                    "global_step": global_step,
                    "turn": step.get("turn"),
                    "step_in_turn": step.get("step"),
                    "action_drift": action_drift,
                    "state_drift": state_drift,
                    "turn_failure": turn_failure,
                    "executable_tool": executable,
                    "decoded_action": step.get("decoded_action"),
                    "reference_action": step.get("reference_action"),
                }
            )

    summary_rows = []
    for offset in offsets:
        bucket = buckets[offset]
        summary_rows.append(
            {
                "relative_step": offset,
                "samples": bucket["samples"],
                "action_drift_rate": _rate(bucket["action_drift"], bucket["samples"]),
                "state_drift_rate": _rate(bucket["state_drift"], bucket["samples"]),
                "turn_failure_rate": _rate(
                    bucket["turn_failure"], bucket["turn_failure_known"]
                ),
                "executable_tool_rate": _rate(
                    bucket["executable_tool"], bucket["executable_tool_known"]
                ),
            }
        )

    output_dir = Path(args.output_dir) if args.output_dir else (
        run_root / "analysis" / f"{args.mode}_after_first_drift"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    rel0_samples = buckets.get(0, {}).get("samples", 0)
    if len(all_action_drift_ids) != rel0_samples:
        warnings.append(
            "Action Drift Samples does not match Relative Step 0 Samples: "
            f"{len(all_action_drift_ids)} vs {rel0_samples}"
        )
    (output_dir / "aligned_steps.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned_rows),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "mode": args.mode,
                "ids_path": args.ids_path or None,
                "drift_sample_count": drift_sample_count,
                "action_drift_sample_count": len(all_action_drift_ids),
                "relative_step_0_samples": rel0_samples,
                "warnings": warnings,
                "rows": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# History C2KV Drift After First Divergence",
        "",
        f"Mode: `{args.mode}`",
        f"Drift samples: {drift_sample_count}",
        f"Action drift sample count: {len(all_action_drift_ids)}",
        f"Relative step 0 samples: {rel0_samples}",
        "",
        "| Relative Step | Samples | Action Drift | State Drift | Turn Failure | Executable Tool |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {rel} | {samples} | {action} | {state} | {turn} | {exec_rate} |".format(
                rel=row["relative_step"],
                samples=row["samples"],
                action=_fmt(row["action_drift_rate"]),
                state=_fmt(row["state_drift_rate"]),
                turn=_fmt(row["turn_failure_rate"]),
                exec_rate=_fmt(row["executable_tool_rate"]),
            )
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "drift_sample_count": drift_sample_count}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--before", type=int, default=-1)
    parser.add_argument("--after", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
