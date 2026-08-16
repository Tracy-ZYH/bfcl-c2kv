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
    _load_prompts_and_answers,
    _rate,
    _result_count,
    _score_header,
)


MODES = ("natural", "corrected")


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


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _summary_row(
    *,
    run_root: Path,
    mode: str,
    category: str,
    turn_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_root = run_root / mode
    score = _score_header(mode_root, category)
    total_samples = _result_count(mode_root, category)
    details = _load_jsonl(mode_root / "logs" / "details.jsonl")
    branch_points = sum(
        1
        for row in details
        if (row.get("c2kv_branch_metrics") or {}).get("branch") in {"natural", "corrected"}
    )
    turn_total = len(turn_rows)
    step_total = len(step_rows)
    parsed_steps = [row for row in step_rows if row.get("tool_call_parse_success")]
    valid_tool_steps = [row for row in parsed_steps if row.get("valid_tool")]
    successful_exec_steps = [
        row for row in parsed_steps if row.get("execution_success")
    ]
    return {
        "method": mode,
        "bfcl_accuracy": score.get("accuracy"),
        "correct_count": score.get("correct_count"),
        "total_samples": total_samples,
        "branch_points": branch_points,
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
        "execution_success_rate": _rate(len(successful_exec_steps), len(parsed_steps)),
        "average_steps_per_episode": step_total / total_samples if total_samples else None,
    }


def write_report(run_root: Path, rows: list[dict[str, Any]]) -> None:
    names = {
        "natural": "Natural Drift",
        "corrected": "Correct First Drift",
    }
    lines = [
        "# BFCL History Branch After First Drift",
        "",
        "| Branch | BFCL Acc | Correct | Episodes | Branch Points | Turn State | Turn Response | Turn Joint | Tool Call | Executable Tool Call | Valid Tool | Exec Success | Avg Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc} | {correct} | {episodes} | {branch_points} | {state} | {response} | {joint} | {tool_call} | {exec_call} | {valid_tool} | {exec_success} | {avg_steps} |".format(
                method=names.get(row["method"], row["method"]),
                acc=_fmt(row.get("bfcl_accuracy")),
                correct=_fmt(row.get("correct_count")),
                episodes=_fmt(row.get("total_samples")),
                branch_points=_fmt(row.get("branch_points")),
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
            "Natural Drift executes the first C2KV divergent action.",
            "Correct First Drift replaces only that first divergent action with the Full reference action, then resumes C2KV closed-loop inference.",
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
    for mode in MODES:
        turn_rows, step_rows = _analysis_rows(
            run_root,
            mode,
            args.category,
            decoder,
            prompt_by_id,
            answer_by_id,
        )
        rows.append(
            _summary_row(
                run_root=run_root,
                mode=mode,
                category=args.category,
                turn_rows=turn_rows,
                step_rows=step_rows,
            )
        )
    (run_root / "summary.json").write_text(
        json.dumps(
            {
                "run_root": str(run_root),
                "category": args.category,
                "methods": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with open(run_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(run_root, rows)
    print(json.dumps({"run_root": str(run_root), "methods": rows}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
