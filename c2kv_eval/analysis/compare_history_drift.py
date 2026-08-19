from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

from c2kv_eval.analysis.compare_multiturn_modes import (
    DEFAULT_MODEL,
    _analysis_rows,
    _first_failure_buckets,
    _load_prompts_and_answers,
    _rate,
    _result_count,
)


MODES = (
    "history_full_closed_loop",
    "history_c2kv4_teacher_forced",
    "history_c2kv4_closed_loop",
    "history_recent2_full_rest_c2kv4",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _score_header(mode_root: Path, category: str) -> dict[str, Any]:
    path = _find_first(mode_root / "score", f"*_{category}_score.json")
    if path is None:
        return {}
    rows = _load_jsonl(path)
    return rows[0] if rows else {}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _detail_drift_ids(
    details: list[dict[str, Any]],
    key: str,
) -> set[str]:
    ids = set()
    for row in details:
        sample_id = str(row.get("id"))
        for step in row.get("drift_steps") or []:
            if step.get(key) is True:
                ids.add(sample_id)
                break
            if key == "candidate_action_drift" and step.get("action_matches_reference") is False:
                ids.add(sample_id)
                break
            if key == "state_drift" and step.get("state_matches_reference") is False:
                ids.add(sample_id)
                break
    return ids


def _first_action_divergence_turns(
    details: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> list[float]:
    by_id = {
        str(row.get("id")): row
        for row in details
        if row.get("id") is not None
    }
    turns = []
    used_ids = set()
    for row in metrics:
        first = row.get("first_action_divergence")
        if isinstance(first, dict) and first.get("turn") is not None:
            turns.append(float(first["turn"]))
            if row.get("id") is not None:
                used_ids.add(str(row.get("id")))
    for sample_id, row in by_id.items():
        if sample_id in used_ids:
            continue
        for step in row.get("drift_steps") or []:
            if step.get("candidate_action_drift") is True or step.get("action_matches_reference") is False:
                if step.get("turn") is not None:
                    turns.append(float(step["turn"]))
                break
    return turns


def _mode_summary(
    run_root: Path,
    mode: str,
    category: str,
    turn_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_root = run_root / mode
    details = _load_jsonl(mode_root / "logs" / "details.jsonl")
    metrics = _load_jsonl(mode_root / "logs" / "drift_metrics.jsonl")
    score = _score_header(mode_root, category)
    original = sum(int(row.get("history_original_tokens") or 0) for row in metrics)
    effective = sum(int(row.get("history_effective_tokens") or 0) for row in metrics)
    extract_calls = sum(int(row.get("extract_calls") or 0) for row in metrics)
    extract_success = sum(int(row.get("extract_success") or 0) for row in metrics)
    chat_calls = sum(int(row.get("chat_calls") or 0) for row in metrics)
    chat_seconds = sum(float(row.get("chat_seconds") or 0.0) for row in metrics)
    action_diverged_ids = _detail_drift_ids(details, "candidate_action_drift")
    executed_action_diverged_ids = _detail_drift_ids(details, "executed_action_drift")
    state_diverged_ids = _detail_drift_ids(details, "state_drift")
    is_full_mode = mode == "history_full_closed_loop"
    if is_full_mode:
        action_diverged_ids = set()
        executed_action_diverged_ids = set()
        state_diverged_ids = set()
    if not action_diverged_ids:
        action_diverged_ids = {
            str(row.get("id"))
            for row in metrics
            if row.get("first_action_divergence") is not None
        }
    if is_full_mode:
        action_diverged_ids = set()
        executed_action_diverged_ids = set()
    if not state_diverged_ids:
        state_diverged_ids = {
            str(row.get("id"))
            for row in metrics
            if row.get("first_state_divergence") is not None
        }
    if is_full_mode:
        state_diverged_ids = set()
    turn_total = len(turn_rows)
    step_total = len(step_rows)
    parsed_steps = [
        row for row in step_rows if row.get("tool_call_parse_success")
    ]
    valid_tool_steps = [row for row in parsed_steps if row.get("valid_tool")]
    executable_steps = parsed_steps
    successful_exec_steps = [
        row for row in executable_steps if row.get("execution_success")
    ]
    buckets = _first_failure_buckets(turn_rows)
    total_samples = _result_count(mode_root, category)
    return {
        "method": mode,
        "bfcl_accuracy": score.get("accuracy"),
        "correct_count": score.get("correct_count"),
        "valid_samples": score.get("total_count") or total_samples,
        "total_samples": total_samples,
        "history_original_tokens": original,
        "history_effective_tokens": effective,
        "history_compression_ratio": (original / effective if effective else 1.0),
        "extract_calls": extract_calls,
        "extract_success_rate": (
            extract_success / extract_calls if extract_calls else None
        ),
        "average_chat_latency": (chat_seconds / chat_calls if chat_calls else None),
        "samples_with_action_divergence": (
            None if is_full_mode else len(action_diverged_ids)
        ),
        "samples_with_executed_action_divergence": (
            None if is_full_mode else len(executed_action_diverged_ids)
        ),
        "samples_with_state_divergence": None if is_full_mode else len(state_diverged_ids),
        "average_first_action_divergence_turn": _mean(
            [] if is_full_mode else _first_action_divergence_turns(details, metrics)
        ),
        "turn_state_pass_rate": _rate(
            sum(1 for row in turn_rows if row.get("state_pass")),
            turn_total,
        ),
        "turn_response_pass_rate": _rate(
            sum(1 for row in turn_rows if row.get("response_pass")),
            turn_total,
        ),
        "turn_joint_pass_rate": _rate(
            sum(1 for row in turn_rows if row.get("joint_pass")),
            turn_total,
        ),
        "tool_call_rate": _rate(
            sum(1 for row in step_rows if row.get("has_tool_call")),
            step_total,
        ),
        "executable_tool_call_rate": _rate(len(parsed_steps), step_total),
        "valid_tool_rate": _rate(len(valid_tool_steps), len(parsed_steps)),
        "execution_success_rate": _rate(
            len(successful_exec_steps),
            len(executable_steps),
        ),
        "average_steps_per_episode": (
            step_total / total_samples if total_samples else None
        ),
        **buckets,
    }


def _fmt(value: Any, digits: int = 4, none: str = "-") -> str:
    if value is None:
        return none
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(run_root: Path, rows: list[dict[str, Any]]) -> None:
    names = {
        "history_full_closed_loop": "Full Closed Loop",
        "history_c2kv4_teacher_forced": "C2KV-4 Teacher Forced",
        "history_c2kv4_closed_loop": "C2KV-4 Closed Loop",
        "history_recent2_full_rest_c2kv4": "Recent2 Full + Rest C2KV-4",
    }
    lines = [
        "# BFCL Multi-Turn - History Compression Drift",
        "",
        "## Episode / Turn / Step",
        "",
        "| Method | BFCL Acc | Correct | Episodes | Turn State | Turn Response | Turn Joint | Tool Call | Executable Tool Call | Valid Tool | Exec Success | Avg Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc} | {correct} | {episodes} | {state} | {response} | {joint} | {tool_call} | {exec_call} | {valid_tool} | {exec_success} | {avg_steps} |".format(
                method=names.get(row["method"], row["method"]),
                acc=_fmt(row.get("bfcl_accuracy")),
                correct=_fmt(row.get("correct_count"), 0),
                episodes=_fmt(row.get("total_samples"), 0),
                state=_fmt(row.get("turn_state_pass_rate")),
                response=_fmt(row.get("turn_response_pass_rate")),
                joint=_fmt(row.get("turn_joint_pass_rate")),
                tool_call=_fmt(row.get("tool_call_rate")),
                exec_call=_fmt(row.get("executable_tool_call_rate")),
                valid_tool=_fmt(row.get("valid_tool_rate")),
                exec_success=_fmt(row.get("execution_success_rate")),
                avg_steps=_fmt(row.get("average_steps_per_episode")),
            )
        )
    lines.extend(
        [
            "",
            "## History Compression / Drift",
            "",
            "| Method | Hist Compression | Avg Chat s | Extract Success | Candidate Action Drift Samples | Executed Action Drift Samples | State Drift Samples | Avg Action Drift Turn |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {comp}x | {chat} | {extract} | {action_drift} | {exec_action_drift} | {state_drift} | {turn} |".format(
                method=names.get(row["method"], row["method"]),
                comp=_fmt(row.get("history_compression_ratio")),
                chat=_fmt(row.get("average_chat_latency")),
                extract=_fmt(row.get("extract_success_rate")),
                action_drift=_fmt(row.get("samples_with_action_divergence"), 0),
                exec_action_drift=_fmt(
                    row.get("samples_with_executed_action_divergence"),
                    0,
                ),
                state_drift=_fmt(row.get("samples_with_state_divergence"), 0),
                turn=_fmt(row.get("average_first_action_divergence_turn")),
            )
        )
    lines.extend(
        [
            "",
            "## First Failure Turn",
            "",
            "| Method | Turn 1 Fail | Turn 2 Fail | Turn 3 Fail | Turn 4+ Fail |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {t1} | {t2} | {t3} | {t4} |".format(
                method=names.get(row["method"], row["method"]),
                t1=_fmt(row.get("turn_1_fail"), 0),
                t2=_fmt(row.get("turn_2_fail"), 0),
                t3=_fmt(row.get("turn_3_fail"), 0),
                t4=_fmt(row.get("turn_4plus_fail"), 0),
            )
        )
    lines.extend(
        [
            "",
            "Tool definitions, system prompt, and current turn are always full.",
            "State logs are retained in each result entry's `inference_log` for drift inspection.",
            "Step-level rows are written to each mode's `logs/step_metrics.jsonl`; turn-level rows are written to `logs/turn_metrics.jsonl`.",
        ]
    )
    (run_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    config = MODEL_CONFIG_MAPPING[args.model]
    decoder = QwenFCHandler(
        model_name=config.model_name,
        temperature=0,
        registry_name=args.model,
        is_fc_model=config.is_fc_model,
    )
    prompt_by_id, answer_by_id = _load_prompts_and_answers(args.category)
    rows = []
    modes = [
        item.strip()
        for item in (args.modes or ",".join(MODES)).split(",")
        if item.strip()
    ]
    for mode in modes:
        turn_rows, step_rows = _analysis_rows(
            run_root,
            mode,
            args.category,
            decoder,
            prompt_by_id,
            answer_by_id,
        )
        rows.append(_mode_summary(run_root, mode, args.category, turn_rows, step_rows))
    summary = {
        "run_root": str(run_root),
        "category": args.category,
        "methods": rows,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(run_root, rows)
    with open(run_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--modes",
        default=",".join(MODES),
        help="Comma-separated drift mode directories to summarize.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
