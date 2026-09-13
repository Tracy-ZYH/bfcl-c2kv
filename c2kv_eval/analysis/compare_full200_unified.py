from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from c2kv_eval.analysis.bfcl_task_oracle import DEFAULT_MODEL, evaluate_result_rows
from c2kv_eval.analysis.compare_history_kv_baselines import summarize_method


MAIN_FIELDS = [
    "Method",
    "BFCL Accuracy",
    "Correct",
    "Total",
    "Task Error Rate",
    "Task State Failure Rate",
    "Task Required-Result Failure Rate",
    "Recovery Target",
    "Recovery Unit",
    "Trigger Count",
    "Eligible Units",
    "Trigger Rate",
    "Detector Coverage",
    "Reference Recovery Success Count",
    "Reference Recovery Success Rate",
    "Reference Recovery Coverage",
    "Rollback Span Coverage",
    "False-Positive Recovery Count",
    "Harmful False-Positive Recovery Count",
    "Harmful False-Positive Recovery Rate",
    "Wrong-Way Step Change Count",
    "Wrong-Way Step Change Rate",
    "Task Recovery Success Rate",
    "Task Recovery Instrumentation",
    "Model Calls / Committed Step",
    "Generation Prefill Tokens / Committed Step",
    "Maintenance Prefill Tokens / Committed Step",
    "Total Prefill Tokens / Committed Step",
    "Prefill Instrumentation",
    "Observed Seconds / Episode",
    "Observed Seconds / Committed Step",
    "Generation Seconds / Model Call",
    "Maintenance Seconds / Committed Step",
    "Recovery Seconds / Trigger",
    "TTFT",
    "Avg Original History Tokens / Committed Step",
    "Avg Active History KV Tokens / Committed Step",
    "Effective History-KV Retention",
    "Effective History-KV Compression",
    "Compression Scope",
    "Estimated Idealized History-KV Byte Compression",
    "Auxiliary Repair KV Tokens / Committed Step",
    "Checkpoint Device KV Tokens / Committed Step",
    "Checkpoint Host KV Tokens / Committed Step",
    "Peak Resident KV Tokens",
    "KV Accounting Coverage",
]

REFERENCE_FIELDS = [
    "Method",
    "Reference Source",
    "Reference-Aligned Steps",
    "Total Steps",
    "Reference Alignment Coverage",
    "Reference Turn Joint",
    "Reference Candidate Action Drift",
    "Reference Executed Action Drift",
    "Reference State Drift",
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
    aligned = [
        step
        for step in steps
        if step.get("alignment_status") != "missing_reference"
        and (
            step.get(key) is not None
            or step.get(match_key) is not None
        )
    ]
    if not aligned:
        return None
    bad = sum(
        1
        for step in aligned
        if step.get(key) is True or step.get(match_key) is False
    )
    return bad / len(aligned)


def _turn_joint_from_reference_steps(details: list[dict[str, Any]]) -> float | None:
    total = 0
    good = 0
    for row in details:
        turns: dict[int, list[dict[str, Any]]] = {}
        for step in row.get("drift_steps") or []:
            if (
                isinstance(step, dict)
                and step.get("alignment_status") != "missing_reference"
                and step.get("executed_action_drift") is not None
            ):
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


def _metric_rows(
    details: list[dict[str, Any]], method_root: Path
) -> list[dict[str, Any]]:
    rows = []
    for row in details:
        metric = row.get("c2kv_checkpoint_metrics") or row.get("c2kv_drift_metrics")
        if isinstance(metric, dict):
            rows.append(metric)
    if rows:
        return rows
    for name in ("metrics.jsonl", "checkpoint_metrics.jsonl"):
        rows = _load_jsonl(method_root / "logs" / name)
        if rows:
            return rows
    return []


def _sum(rows: list[dict[str, Any]], key: str) -> tuple[float, bool]:
    present = any(row.get(key) is not None for row in rows)
    return sum(float(row.get(key) or 0.0) for row in rows), present


def _sum_first(
    rows: list[dict[str, Any]], *keys: str
) -> tuple[float, bool]:
    total = 0.0
    present = False
    for row in rows:
        for key in keys:
            if row.get(key) is not None:
                total += float(row.get(key) or 0.0)
                present = True
                break
    return total, present


def _runtime_summary(
    rows: list[dict[str, Any]], committed_steps: int, method: str
) -> dict[str, Any]:
    chat_calls, calls_present = _sum(rows, "chat_calls")
    generation_prefill, generation_present = _sum(
        rows, "chat_recomputed_prompt_tokens"
    )
    c2kv_work, c2kv_present = _sum(rows, "c2kv_extract_recomputed_tokens")
    repair_work, repair_present = _sum(rows, "repair_extract_recomputed_tokens")
    checkpoint_work, checkpoint_present = _sum(
        rows, "checkpoint_maintenance_recomputed_tokens"
    )
    restore_work, restore_present = _sum(rows, "kv_recomputed_tokens")
    replay_work, replay_present = _sum(rows, "message_replay_prefill_tokens")
    maintenance_prefill = (
        c2kv_work + repair_work + checkpoint_work + restore_work + replay_work
    )
    maintenance_present = any(
        (c2kv_present, repair_present, checkpoint_present, restore_present, replay_present)
    )
    prefill_status = "complete from per-request cache reports"
    if method.startswith("rollback") and generation_prefill == 0 and chat_calls:
        # Checkpoint runs from this schema wrote a zero placeholder even though
        # generation calls occurred. Do not present that placeholder as a
        # measured zero-cost generation path.
        generation_present = False
        prefill_status = (
            "partial: generation prefill unavailable; maintenance includes "
            "C2KV extraction, checkpoint maintenance, and restore recomputation"
        )

    chat_seconds, chat_seconds_present = _sum(rows, "chat_seconds")
    extract_seconds, extract_seconds_present = _sum(rows, "extract_seconds")
    observed_seconds, observed_present = _sum(rows, "episode_e2e_observed_seconds")
    if not observed_present and (chat_seconds_present or extract_seconds_present):
        tool_seconds, _ = _sum(rows, "tool_execution_seconds")
        observed_seconds = chat_seconds + extract_seconds + tool_seconds
        observed_present = True

    return {
        "calls": chat_calls if calls_present else None,
        "generation_prefill": generation_prefill if generation_present else None,
        "maintenance_prefill": maintenance_prefill if maintenance_present else None,
        "total_prefill": (
            generation_prefill + maintenance_prefill
            if generation_present and maintenance_present
            else None
        ),
        "prefill_status": prefill_status,
        "observed_seconds": observed_seconds if observed_present else None,
        "chat_seconds": chat_seconds if chat_seconds_present else None,
        "maintenance_seconds": extract_seconds if extract_seconds_present else None,
        "committed_steps": committed_steps,
    }


def _history_summary(
    rows: list[dict[str, Any]], committed_steps: int, method: str, kv_row: dict[str, Any]
) -> dict[str, Any]:
    original, original_present = _sum(rows, "canonical_full_history_tokens")
    if not original:
        original, original_present = _sum(rows, "history_original_tokens")
    active, active_present = _sum(rows, "physical_history_kv_tokens")
    if not active:
        active, active_present = _sum(rows, "history_effective_tokens")
    repair_kv, repair_present = _sum(rows, "repair_kv_tokens")
    checkpoint_device, checkpoint_device_present = _sum(
        rows, "checkpoint_device_tokens"
    )
    checkpoint_host, checkpoint_host_present = _sum(rows, "checkpoint_host_tokens")
    peaks = []
    accounting_rows = 0
    for row in rows:
        row_original = row.get("canonical_full_history_tokens") or row.get(
            "history_original_tokens"
        )
        row_active = row.get("physical_history_kv_tokens") or row.get(
            "history_effective_tokens"
        )
        if row_original is not None and row_active is not None:
            accounting_rows += 1
        value = row.get("peak_physical_kv_tokens")
        if value is None:
            value = row.get("kv_peak_resident_tokens")
        if value is not None and float(value) > 0:
            peaks.append(float(value))

    if method.startswith("rollback"):
        scope = "candidate history only; checkpoint/recovery KV reported separately"
    elif method.startswith("d_corr"):
        scope = "token-weighted physical history KV including injected repair KV"
    elif method == "kivi":
        scope = "history token count; byte compression is reported separately"
    else:
        scope = "token-weighted physical/selected history KV"
    return {
        "original": original if original_present else None,
        "active": active if active_present else None,
        "retention": _rate(active, original)
        if original_present and active_present
        else None,
        "compression": _rate(original, active)
        if original_present and active_present
        else None,
        "repair_kv": repair_kv if repair_present else None,
        "checkpoint_device": checkpoint_device if checkpoint_device_present else None,
        "checkpoint_host": checkpoint_host if checkpoint_host_present else None,
        "peak": max(peaks) if peaks else None,
        "coverage": _rate(accounting_rows, len(rows)),
        "scope": scope,
        "byte_compression": kv_row.get("Estimated History-KV Byte Compression"),
        "committed_steps": committed_steps,
    }


def _recovery_summary(
    details: list[dict[str, Any]], rows: list[dict[str, Any]], method: str
) -> dict[str, Any]:
    checkpoint_segments = [
        segment
        for row in details
        for segment in (row.get("checkpoint_segments") or [])
        if isinstance(segment, dict)
    ]
    repair_segments = [
        segment
        for row in details
        for segment in (row.get("repair_segments") or [])
        if isinstance(segment, dict)
    ]
    segments = checkpoint_segments or repair_segments
    if not segments:
        return {
            "target": None,
            "unit": None,
            "task_status": "not applicable",
        }

    triggered = [segment for segment in segments if segment.get("detector_trigger") is True]
    harmful = [
        segment
        for segment in segments
        if segment.get("oracle_reference_drift_segment") is True
        or segment.get("segment_has_drift") is True
    ]
    successful = [
        segment
        for segment in triggered
        if segment.get("reference_recovery_success") is True
        or segment.get("segment_recovery_success") is True
        or segment.get("repair_segment_success") is True
    ]
    false_positive = [
        segment
        for segment in triggered
        if not (
            segment.get("oracle_reference_drift_segment") is True
            or segment.get("segment_has_drift") is True
        )
    ]
    harmful_fp = [
        segment for segment in false_positive if segment.get("fp_recovery_harm") is True
    ]
    wrong_way, wrong_way_present = _sum(rows, "c2kv_correct_repair_wrong")
    repaired_steps, repaired_steps_present = _sum(rows, "repaired_step_count")

    rollback_coverage = [
        float(segment["rollback_coverage"])
        for segment in triggered
        if segment.get("rollback_coverage") is not None
    ]
    if checkpoint_segments:
        recovery_seconds, seconds_present = _sum(rows, "rollback_latency_sec")
    else:
        recovery_seconds, seconds_present = _sum(rows, "repair_extract_seconds")
    return {
        "target": "Reference Drift Oracle",
        "unit": "segment",
        "eligible": len(segments),
        "triggers": len(triggered),
        "trigger_rate": _rate(len(triggered), len(segments)),
        "detector_coverage": _rate(len(triggered), len(harmful)),
        "successes": len(successful),
        "success_rate": _rate(len(successful), len(triggered)),
        "recovery_coverage": _rate(len(successful), len(harmful)),
        "rollback_span_coverage": (
            sum(rollback_coverage) / len(rollback_coverage)
            if rollback_coverage
            else None
        ),
        "false_positives": len(false_positive),
        "harmful_fp": len(harmful_fp),
        "harmful_fp_rate": _rate(len(harmful_fp), len(false_positive)),
        "wrong_way": wrong_way if wrong_way_present else None,
        "wrong_way_rate": (
            _rate(wrong_way, repaired_steps)
            if wrong_way_present and repaired_steps_present
            else None
        ),
        "seconds_per_trigger": (
            _rate(recovery_seconds, len(triggered)) if seconds_present else None
        ),
        "task_success": None,
        "task_status": "unavailable: before/after task-valid state was not recorded",
    }


def _reference_analysis(
    label: str,
    comparison_reference: str,
    steps: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> dict[str, Any]:
    aligned = [
        step
        for step in steps
        if step.get("alignment_status") != "missing_reference"
        and step.get("executed_action_drift") is not None
    ]
    reference_source = (
        comparison_reference
        if aligned
        else "none: this run has no aligned trajectory reference"
    )
    return {
        "Method": label,
        "Reference Source": reference_source,
        "Reference-Aligned Steps": len(aligned),
        "Total Steps": len(steps),
        "Reference Alignment Coverage": _rate(len(aligned), len(steps)),
        "Reference Turn Joint": _turn_joint_from_reference_steps(details),
        "Reference Candidate Action Drift": _step_rate(
            steps, "candidate_action_drift", "candidate_action_matches_reference"
        ),
        "Reference Executed Action Drift": _step_rate(
            steps, "executed_action_drift", "executed_action_matches_reference"
        ),
        "Reference State Drift": _step_rate(
            steps, "state_drift", "state_matches_reference"
        ),
    }


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
    task_oracle_root: Path,
    comparison_reference: str,
    baseline_run_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    details_path = _find_details(method_root)
    details = _load_jsonl(details_path)
    total = len(details)
    task_turns, task_summary = evaluate_result_rows(
        rows=details,
        category=category,
        model=model,
        output_dir=task_oracle_root / method,
        mode_name=method,
    )
    steps = _flatten_steps(details)
    acc, correct = _score(method_root, total)
    metric_rows = _metric_rows(details, method_root)
    # A reused Full reference lives outside the new compression run root. Read
    # its KV metrics from that immutable source instead of emitting an empty
    # Full row merely because Full was intentionally not rerun.
    if method == "full":
        kv_row = _kv_baseline_row(method_root.parent, method_root.name)
    else:
        kv_row = _kv_baseline_row(baseline_run_root, method) if baseline_run_root else {}
    committed_steps = len(steps)
    runtime = _runtime_summary(metric_rows, committed_steps, method)
    history = _history_summary(metric_rows, committed_steps, method, kv_row)
    recovery = _recovery_summary(details, metric_rows, method)
    row = {
        "Method": label,
        "BFCL Accuracy": acc if acc is not None else task_summary.get("bfcl_task_accuracy"),
        "Correct": correct if correct is not None else task_summary.get("correct_count"),
        "Total": total,
        **_task_rates(task_summary),
        "Recovery Target": recovery.get("target"),
        "Recovery Unit": recovery.get("unit"),
        "Trigger Count": recovery.get("triggers"),
        "Eligible Units": recovery.get("eligible"),
        "Trigger Rate": recovery.get("trigger_rate"),
        "Detector Coverage": recovery.get("detector_coverage"),
        "Reference Recovery Success Count": recovery.get("successes"),
        "Reference Recovery Success Rate": recovery.get("success_rate"),
        "Reference Recovery Coverage": recovery.get("recovery_coverage"),
        "Rollback Span Coverage": recovery.get("rollback_span_coverage"),
        "False-Positive Recovery Count": recovery.get("false_positives"),
        "Harmful False-Positive Recovery Count": recovery.get("harmful_fp"),
        "Harmful False-Positive Recovery Rate": recovery.get("harmful_fp_rate"),
        "Wrong-Way Step Change Count": recovery.get("wrong_way"),
        "Wrong-Way Step Change Rate": recovery.get("wrong_way_rate"),
        "Task Recovery Success Rate": recovery.get("task_success"),
        "Task Recovery Instrumentation": recovery.get("task_status"),
        "Model Calls / Committed Step": _rate(runtime.get("calls"), committed_steps),
        "Generation Prefill Tokens / Committed Step": _rate(
            runtime.get("generation_prefill"), committed_steps
        ),
        "Maintenance Prefill Tokens / Committed Step": _rate(
            runtime.get("maintenance_prefill"), committed_steps
        ),
        "Total Prefill Tokens / Committed Step": _rate(
            runtime.get("total_prefill"), committed_steps
        ),
        "Prefill Instrumentation": runtime.get("prefill_status"),
        "Observed Seconds / Episode": _rate(runtime.get("observed_seconds"), total),
        "Observed Seconds / Committed Step": _rate(
            runtime.get("observed_seconds"), committed_steps
        ),
        "Generation Seconds / Model Call": _rate(
            runtime.get("chat_seconds"), runtime.get("calls")
        ),
        "Maintenance Seconds / Committed Step": _rate(
            runtime.get("maintenance_seconds"), committed_steps
        ),
        "Recovery Seconds / Trigger": recovery.get("seconds_per_trigger"),
        "TTFT": None,
        "Avg Original History Tokens / Committed Step": _rate(
            history.get("original"), committed_steps
        ),
        "Avg Active History KV Tokens / Committed Step": _rate(
            history.get("active"), committed_steps
        ),
        "Effective History-KV Retention": history.get("retention"),
        "Effective History-KV Compression": history.get("compression"),
        "Compression Scope": history.get("scope"),
        "Estimated Idealized History-KV Byte Compression": history.get(
            "byte_compression"
        ),
        "Auxiliary Repair KV Tokens / Committed Step": _rate(
            history.get("repair_kv"), committed_steps
        ),
        "Checkpoint Device KV Tokens / Committed Step": _rate(
            history.get("checkpoint_device"), committed_steps
        ),
        "Checkpoint Host KV Tokens / Committed Step": _rate(
            history.get("checkpoint_host"), committed_steps
        ),
        "Peak Resident KV Tokens": history.get("peak"),
        "KV Accounting Coverage": history.get("coverage"),
    }
    _write_turn_manifest(task_oracle_root, method, details_path, task_turns, task_summary)
    return row, _reference_analysis(label, comparison_reference, steps, details)


def _write_turn_manifest(
    task_oracle_root: Path,
    method: str,
    details_path: Path,
    task_turns: list[dict[str, Any]],
    task_summary: dict[str, Any],
) -> None:
    method_dir = task_oracle_root / method
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


def _write_table(
    csv_path: Path,
    markdown_path: Path,
    title: str,
    fields: list[str],
    rows: list[dict[str, Any]],
    preamble: list[str] | None = None,
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", ""]
    if preamble:
        lines.extend(preamble)
        lines.append("")
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("-")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_root: Path,
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    output_stem: str,
) -> None:
    summary_dir = output_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{output_stem}.csv"
    markdown_path = summary_dir / f"{output_stem}.md"
    _write_table(
        csv_path,
        markdown_path,
        "BFCL multi_turn_base Full-200: Task Correctness and System Metrics",
        MAIN_FIELDS,
        rows,
        [
            "BFCL and Task-Oracle fields are the primary correctness metrics.",
            "Recovery metrics are explicitly reference-drift based in this run; "
            "task-level before/after recovery success was not recorded.",
            "Timing is observed wall-clock instrumentation, not a synchronized "
            "kernel benchmark; TTFT was not recorded.",
        ],
    )
    (summary_dir / f"{output_stem}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    reference_stem = f"{output_stem}_reference_analysis"
    _write_table(
        summary_dir / f"{reference_stem}.csv",
        summary_dir / f"{reference_stem}.md",
        "BFCL Full-200 Reference-Trajectory Analysis",
        REFERENCE_FIELDS,
        reference_rows,
        [
            "These are trajectory-agreement diagnostics, not BFCL task correctness.",
            "Missing-reference steps are excluded. A reference-free Full run therefore "
            "reports N/A rather than treating tool calls as drift against an empty action.",
        ],
    )
    (summary_dir / f"{reference_stem}.json").write_text(
        json.dumps(reference_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    definitions = {
        "correctness": {
            "primary": [
                "BFCL Accuracy",
                "Task Error Rate",
                "Task State Failure Rate",
                "Task Required-Result Failure Rate",
            ],
            "bfcl_accuracy_unit": "episode",
            "task_oracle_rate_unit": "evaluated turn; it is not 1 - episode BFCL accuracy",
            "reference_analysis_is_not_task_correctness": True,
        },
        "recovery": {
            "target": "Reference Drift Oracle",
            "trigger_rate": "triggered segments / eligible segments",
            "detector_coverage": "triggered segments / reference-harmful segments",
            "success_rate": "reference-recovered triggered segments / triggered segments",
            "recovery_coverage": "reference-recovered segments / reference-harmful segments",
            "task_recovery_success": "unavailable because before/after task-valid state was not recorded",
        },
        "compression": {
            "effective": "sum(original history tokens) / sum(active physical/effective history KV tokens)",
            "rollback_caveat": "candidate-path compression excludes checkpoint/recovery KV; checkpoint device and host storage are separate columns",
        },
        "timing": {
            "observed": "existing wall-clock instrumentation",
            "ttft": "unavailable",
            "cross_path_caveat": "component timers differ by execution path; use observed seconds/episode for the broadest comparison",
        },
        "prefill": {
            "maintenance_components": "C2KV extraction + repair extraction + checkpoint maintenance + restore/message-replay recomputation",
            "rollback_caveat": "generation prefill was stored as a zero placeholder and is reported unavailable, so rollback total prefill is also unavailable",
        },
    }
    (summary_dir / f"{output_stem}_definitions.json").write_text(
        json.dumps(definitions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    task_oracle_root = output_root / f"task_oracle_by_method_{args.output_stem}"
    comparison_reference = args.reference_details_path
    if not comparison_reference:
        config = _load_json(output_root / "experiment_config.json")
        comparison_reference = str(config.get("reference_details_path") or "embedded reference fields")
    specs = [_parse_method_spec(item) for item in args.method_specs if item]
    summarized = [
        summarize_one(
            label=label,
            method=method,
            method_root=path,
            category=args.category,
            model=args.model,
            task_oracle_root=task_oracle_root,
            comparison_reference=comparison_reference,
            baseline_run_root=Path(args.compression_run_root)
            if args.compression_run_root
            else None,
        )
        for label, method, path in specs
    ]
    rows = [item[0] for item in summarized]
    reference_rows = [item[1] for item in summarized]
    write_outputs(output_root, rows, reference_rows, args.output_stem)
    print(
        json.dumps(
            {
                "output": str(output_root / f"{args.output_stem}.csv"),
                "reference_analysis": str(
                    output_root
                    / "summaries"
                    / f"{args.output_stem}_reference_analysis.csv"
                ),
                "rows": len(rows),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--compression-run-root", default="")
    parser.add_argument("--reference-details-path", default="")
    parser.add_argument("--output-stem", default="unified_full200")
    parser.add_argument("method_specs", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
