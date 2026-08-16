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
    _find_first,
    _load_prompts_and_answers,
    _rate,
    _result_count,
    _score_header,
)


MODES = (
    "c2kv4_oracle_correct1",
    "c2kv4_oracle_correct2",
    "c2kv4_oracle_correct4",
    "c2kv4_oracle_correct_all",
)


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


def _score_ids(mode_root: Path, category: str) -> tuple[set[str], set[str]]:
    result_path = _find_first(mode_root / "result", f"*_{category}_result.json")
    score_path = _find_first(mode_root / "score", f"*_{category}_score.json")
    if result_path is None or score_path is None:
        return set(), set()
    result_ids = {
        str(row["id"])
        for row in _load_jsonl(result_path)
        if row.get("id") is not None
    }
    score_rows = _load_jsonl(score_path)
    invalid_ids = {
        str(row["id"])
        for row in score_rows[1:]
        if row.get("id") is not None and row.get("valid") is False
    }
    explicit_valid = {
        str(row["id"])
        for row in score_rows[1:]
        if row.get("id") is not None and row.get("valid") is True
    }
    if explicit_valid:
        return explicit_valid, invalid_ids
    return result_ids - invalid_ids, invalid_ids


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _is_empty_action(value: Any) -> bool:
    if value is None:
        return True
    if value == [] or value == [""]:
        return True
    return False


def _summary_row(
    *,
    run_root: Path,
    mode: str,
    category: str,
    turn_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_root = run_root / mode
    score = _score_header(mode_root, category)
    total_samples = _result_count(mode_root, category)
    details = _load_jsonl(mode_root / "logs" / "details.jsonl")
    metrics = _load_jsonl(mode_root / "logs" / "oracle_metrics.jsonl")
    correct_ids, _ = _score_ids(mode_root, category)
    turn_total = len(turn_rows)
    correction_by_id = {
        str(row.get("id")): int(row.get("corrections_used") or 0)
        for row in metrics
    }
    corrected_ids = {
        sample_id for sample_id, count in correction_by_id.items() if count > 0
    }
    candidate_action_drift_ids = set()
    executed_action_drift_ids = set()
    extra_executed_action_ids = set()
    state_drift_ids = set()
    for row in details:
        sample_id = str(row.get("id"))
        for step in row.get("drift_steps") or []:
            reference_action = step.get("reference_action")
            decoded_action = step.get("decoded_action")
            executed_action = step.get("executed_action")
            candidate_match = step.get("action_matches_reference")
            if candidate_match is None and reference_action is None:
                candidate_match = _is_empty_action(decoded_action)
            if candidate_match is False:
                candidate_action_drift_ids.add(sample_id)
            executed_match = step.get("executed_action_matches_reference")
            if executed_match is None and reference_action is None:
                executed_match = _is_empty_action(executed_action)
            elif executed_match is None:
                executed_match = step.get("action_matches_reference")
            if executed_match is False:
                executed_action_drift_ids.add(sample_id)
                if reference_action is None and not _is_empty_action(executed_action):
                    extra_executed_action_ids.add(sample_id)
            if step.get("state_matches_reference") is False:
                state_drift_ids.add(sample_id)
    total_corrections = sum(correction_by_id.values())
    return {
        "method": mode,
        "bfcl_accuracy": score.get("accuracy"),
        "correct_count": score.get("correct_count"),
        "total_samples": total_samples,
        "turn_joint_pass_rate": _rate(
            sum(1 for row in turn_rows if row.get("joint_pass")),
            turn_total,
        ),
        "candidate_action_drift_rate": _rate(
            len(candidate_action_drift_ids), total_samples
        ),
        "candidate_action_drift_count": len(candidate_action_drift_ids),
        "executed_action_drift_rate": _rate(
            len(executed_action_drift_ids), total_samples
        ),
        "executed_action_drift_count": len(executed_action_drift_ids),
        "extra_executed_action_count": len(extra_executed_action_ids),
        "extra_executed_action_rate": _rate(
            len(extra_executed_action_ids), total_samples
        ),
        # Backward-compatible alias. For Oracle reports this is candidate drift.
        "action_drift_rate": _rate(len(candidate_action_drift_ids), total_samples),
        "action_drift_count": len(candidate_action_drift_ids),
        "state_drift_rate": _rate(len(state_drift_ids), total_samples),
        "state_drift_count": len(state_drift_ids),
        "average_corrections": total_corrections / total_samples if total_samples else None,
        "total_corrections": total_corrections,
        "corrected_episode_count": len(corrected_ids),
        "recovery_success_rate": _rate(
            len(corrected_ids & correct_ids),
            len(corrected_ids),
        ),
    }


def write_report(run_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# BFCL History C2KV Oracle Recovery",
        "",
        "| Method | BFCL Acc | Correct | Turn Joint | Candidate Action Drift | Executed Action Drift | Extra Executed Action | State Drift | Avg Corrections | Recovery Success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc} | {correct} | {joint} | {cand_action} | {exec_action} | {extra_action} | {state} | {avg_corr} | {recovery} |".format(
                method=row["method"],
                acc=_fmt(row.get("bfcl_accuracy")),
                correct=_fmt(row.get("correct_count")),
                joint=_fmt(row.get("turn_joint_pass_rate")),
                cand_action=_fmt(row.get("candidate_action_drift_rate")),
                exec_action=_fmt(row.get("executed_action_drift_rate")),
                extra_action=_fmt(row.get("extra_executed_action_rate")),
                state=_fmt(row.get("state_drift_rate")),
                avg_corr=_fmt(row.get("average_corrections")),
                recovery=_fmt(row.get("recovery_success_rate")),
            )
        )
    lines.extend(
        [
            "",
            "Candidate Action Drift is `C2KV generated action != Full reference action`, even if the oracle later rejects it.",
            "Executed Action Drift is `actually executed action != Full reference action`, after oracle correction.",
            "Extra Executed Action means the Full reference has no action for this turn/step, but the evaluated trajectory executed one.",
            "State Drift is measured after the executed action.",
            "Recovery Success Rate is BFCL success among episodes where at least one oracle correction was applied.",
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
        turn_rows, _ = _analysis_rows(
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
            )
        )
    summary = {
        "run_root": str(run_root),
        "category": args.category,
        "methods": rows,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with open(run_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(run_root, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
