from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from c2kv_eval.analysis.bfcl_task_oracle import DEFAULT_MODEL, evaluate_result_rows
from c2kv_eval.analysis.compare_history_kv_baselines import summarize_method


CSV_FIELDS = [
    "Method",
    "BFCL Accuracy",
    "Correct",
    "Total",
    "Turn Joint",
    "Candidate Action Drift",
    "Executed Action Drift",
    "Reference State Drift",
    "Task Error Rate",
    "Task State Failure Rate",
    "Task Required-Result Failure Rate",
    "Task Trigger Rate",
    "Task Recovery Success",
    "Reference Trigger Rate",
    "Reference Recovery Success",
    "Model Calls / Committed Step",
    "Generation Prefill Tokens / Committed Step",
    "Maintenance Prefill Tokens / Committed Step",
    "Total Prefill Tokens / Committed Step",
    "Avg Full History KV",
    "Avg Active History KV",
    "Estimated Weighted History-KV Retention",
    "Estimated Weighted History-KV Compression",
    "Measured Attention-Visible History-KV Retention",
    "Measured Attention-Visible History-KV Compression",
    "Estimated Idealized History-KV Byte Compression",
    "Auxiliary Repair KV Storage / Committed Step",
    "Memory Report Coverage",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score(method_root: Path, total: int) -> tuple[float | None, int | None]:
    path = method_root / "score" / "data_multi_turn.csv"
    if not path.exists():
        return None, None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    value = rows[0].get("Base") or rows[0].get("Multi Turn Overall Acc")
    if isinstance(value, str) and value.endswith("%"):
        acc = float(value[:-1]) / 100.0
        return acc, round(acc * total)
    try:
        acc = float(value)
    except Exception:
        return None, None
    if acc > 1.0:
        acc /= 100.0
    return acc, round(acc * total)


def _find_details(method_root: Path) -> Path:
    direct = method_root / "logs" / "details.jsonl"
    if direct.exists():
        return direct
    matches = sorted(method_root.rglob("details.jsonl"))
    return matches[0] if matches else direct


def _flatten_steps(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps = []
    for row in details:
        for step in row.get("drift_steps") or []:
            if isinstance(step, dict):
                steps.append(step)
    return steps


def _rate(num: int | float | None, den: int | float | None) -> float | None:
    if num is None:
        return None
    return num / den if den else None


def _step_rate(steps: list[dict[str, Any]], key: str, match_key: str) -> float | None:
    if not steps:
        return None
    bad = sum(
        1
        for step in steps
        if step.get(key) is True or step.get(match_key) is False
    )
    return bad / len(steps)


def _turn_joint_from_reference_steps(details: list[dict[str, Any]]) -> float | None:
    total = 0
    good = 0
    for row in details:
        turns: dict[int, list[dict[str, Any]]] = {}
        for step in row.get("drift_steps") or []:
            if isinstance(step, dict):
                turns.setdefault(int(step.get("turn") or 0), []).append(step)
        for turn_steps in turns.values():
            total += 1
            if all(
                step.get("executed_action_drift") is not True
                and step.get("executed_action_matches_reference") is not False
                and step.get("state_drift") is not True
                and step.get("state_matches_reference") is not False
                for step in turn_steps
            ):
                good += 1
    return _rate(good, total)


def _metric_summary(method_root: Path) -> dict[str, Any]:
    for path in (
        method_root / "logs" / "summary.json",
        method_root / "logs" / "run_summary.json",
    ):
        data = _load_json(path)
        if data:
            return data
    rows = _load_jsonl(method_root / "logs" / "metrics.jsonl")
    if rows:
        return {
            "chat_calls": sum(int(row.get("chat_calls") or 0) for row in rows),
            "repair_extract_recomputed_tokens": sum(
                int(row.get("repair_extract_recomputed_tokens") or 0)
                for row in rows
            ),
            "chat_recomputed_prompt_tokens": sum(
                int(row.get("chat_recomputed_prompt_tokens") or 0) for row in rows
            ),
        }
    return {}


def _kv_baseline_row(run_root: Path, method: str) -> dict[str, Any]:
    try:
        return summarize_method(run_root, method)
    except Exception:
        return {}


def _task_rates(task_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "Task Error Rate": task_summary.get("task_error_rate"),
        "Task State Failure Rate": task_summary.get("task_state_failure_rate"),
        "Task Required-Result Failure Rate": task_summary.get(
            "task_required_result_failure_rate"
        ),
    }


def summarize_one(
    *,
    label: str,
    method: str,
    method_root: Path,
    category: str,
    model: str,
    output_root: Path,
    baseline_run_root: Path | None,
) -> dict[str, Any]:
    details_path = _find_details(method_root)
    details = _load_jsonl(details_path)
    total = len(details)
    task_turns, task_summary = evaluate_result_rows(
        rows=details,
        category=category,
        model=model,
        output_dir=output_root / "task_oracle_by_method" / method,
        mode_name=method,
    )
    steps = _flatten_steps(details)
    acc, correct = _score(method_root, total)
    metrics = _metric_summary(method_root)
    kv_row = _kv_baseline_row(baseline_run_root, method) if baseline_run_root else {}
    reference_trigger_rate = (
        metrics.get("detector_trigger_rate")
        or metrics.get("rollback_rate")
        or metrics.get("repair_rate")
    )
    reference_recovery = (
        metrics.get("reference_recovery_success_rate")
        or metrics.get("repair_success_rate")
        or metrics.get("segment_recovery_success_rate")
    )
    generation_prefill = metrics.get("chat_recomputed_prompt_tokens")
    maintenance_prefill = (
        metrics.get("repair_extract_recomputed_tokens")
        or metrics.get("checkpoint_maintenance_recomputed_tokens")
    )
    committed_steps = len(steps)
    row = {
        "Method": label,
        "BFCL Accuracy": acc if acc is not None else task_summary.get("bfcl_task_accuracy"),
        "Correct": correct if correct is not None else task_summary.get("correct_count"),
        "Total": total,
        "Turn Joint": _turn_joint_from_reference_steps(details),
        "Candidate Action Drift": _step_rate(
            steps, "candidate_action_drift", "candidate_action_matches_reference"
        ),
        "Executed Action Drift": _step_rate(
            steps, "executed_action_drift", "executed_action_matches_reference"
        ),
        "Reference State Drift": _step_rate(steps, "state_drift", "state_matches_reference"),
        **_task_rates(task_summary),
        "Task Trigger Rate": None,
        "Task Recovery Success": None,
        "Reference Trigger Rate": reference_trigger_rate,
        "Reference Recovery Success": reference_recovery,
        "Model Calls / Committed Step": _rate(metrics.get("chat_calls"), committed_steps),
        "Generation Prefill Tokens / Committed Step": _rate(
            generation_prefill, committed_steps
        ),
        "Maintenance Prefill Tokens / Committed Step": _rate(
            maintenance_prefill, committed_steps
        ),
        "Total Prefill Tokens / Committed Step": (
            _rate((generation_prefill or 0) + (maintenance_prefill or 0), committed_steps)
            if generation_prefill is not None or maintenance_prefill is not None
            else None
        ),
        "Avg Full History KV": kv_row.get("Avg Full History KV"),
        "Avg Active History KV": kv_row.get("Avg Active History KV"),
        "Estimated Weighted History-KV Retention": kv_row.get(
            "Estimated Weighted History-KV Retention"
        ),
        "Estimated Weighted History-KV Compression": kv_row.get(
            "Estimated Weighted History-KV Compression"
        ),
        "Measured Attention-Visible History-KV Retention": kv_row.get(
            "Measured Attention-Visible History-KV Retention"
        ),
        "Measured Attention-Visible History-KV Compression": kv_row.get(
            "Measured Attention-Visible History-KV Compression"
        ),
        "Estimated Idealized History-KV Byte Compression": kv_row.get(
            "Estimated History-KV Byte Compression"
        ),
        "Auxiliary Repair KV Storage / Committed Step": kv_row.get(
            "Resident KV Storage / Committed Step"
        ),
        "Memory Report Coverage": kv_row.get("Memory Report Coverage"),
    }
    _write_turn_manifest(output_root, method, details_path, task_turns, task_summary)
    return row


def _write_turn_manifest(
    output_root: Path,
    method: str,
    details_path: Path,
    task_turns: list[dict[str, Any]],
    task_summary: dict[str, Any],
) -> None:
    method_dir = output_root / "task_oracle_by_method" / method
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "source_details_path.txt").write_text(
        str(details_path) + "\n",
        encoding="utf-8",
    )
    (method_dir / "task_summary.json").write_text(
        json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_method_spec(value: str) -> tuple[str, str, Path]:
    label, method, path = value.split(":", 2)
    return label, method, Path(path)


def write_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "unified_full200.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (summary_dir / "unified_full200.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# BFCL multi_turn_base Full-200 Unified Results", ""]
    lines.append("| " + " | ".join(CSV_FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(CSV_FIELDS)) + " |")
    for row in rows:
        values = []
        for field in CSV_FIELDS:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("-")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    (summary_dir / "unified_full200.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    specs = [_parse_method_spec(item) for item in args.method_specs if item]
    rows = [
        summarize_one(
            label=label,
            method=method,
            method_root=path,
            category=args.category,
            model=args.model,
            output_root=output_root,
            baseline_run_root=Path(args.compression_run_root)
            if args.compression_run_root
            else None,
        )
        for label, method, path in specs
    ]
    write_outputs(output_root, rows)
    print(json.dumps({"output": str(output_root / "unified_full200.csv"), "rows": len(rows)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compression-run-root", default="")
    parser.add_argument("method_specs", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
