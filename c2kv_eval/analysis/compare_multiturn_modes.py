from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    response_checker,
    state_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry


MODES = ("full", "c2kv", "hybrid")
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FC"


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


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _score_header(mode_root: Path, category: str) -> dict[str, Any]:
    path = _find_first(mode_root / "score", f"*_{category}_score.json")
    if path is None:
        return {}
    rows = _load_jsonl(path)
    return rows[0] if rows else {}


def _result_count(mode_root: Path, category: str) -> int:
    path = _find_first(mode_root / "result", f"*_{category}_result.json")
    return len(_load_jsonl(path)) if path else 0


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _result_rows(mode_root: Path, category: str) -> list[dict[str, Any]]:
    path = _find_first(mode_root / "result", f"*_{category}_result.json")
    return _load_jsonl(path) if path else []


def _tool_names(prompt_entry: dict[str, Any]) -> set[str]:
    names = set()
    for tool in prompt_entry.get("function", []):
        if isinstance(tool, dict) and tool.get("name"):
            names.add(str(tool["name"]))
    return names


def _extract_json_tool_calls(text: str) -> tuple[bool, list[dict[str, Any]]]:
    if not isinstance(text, str):
        return False, []
    has_tool_call = "<tool_call" in text or '"name"' in text and '"arguments"' in text
    calls = []
    for match in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            value = json.loads(match)
        except Exception:
            continue
        if isinstance(value, dict):
            calls.append(value)
    return has_tool_call, calls


def _execution_ok(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    if content.startswith("Error during execution:"):
        return False
    try:
        value = json.loads(content)
    except Exception:
        return True
    if isinstance(value, dict) and "error" in value:
        return False
    return True


def _step_rows_for_episode(
    row: dict[str, Any],
    prompt_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    valid_tools = _tool_names(prompt_entry)
    logs = row.get("inference_log") or []
    result = row.get("result") or []
    step_rows: list[dict[str, Any]] = []
    global_step = 0
    for user_turn, turn_log in enumerate(logs):
        if not isinstance(turn_log, dict) or "begin_of_turn_query" not in turn_log:
            continue
        for key in sorted(
            (key for key in turn_log if key.startswith("step_")),
            key=lambda item: int(item.split("_", 1)[1]),
        ):
            step_in_turn = int(key.split("_", 1)[1])
            step_log = turn_log.get(key) or []
            assistant = next(
                (
                    item
                    for item in step_log
                    if isinstance(item, dict) and item.get("role") == "assistant"
                ),
                {},
            )
            content = assistant.get("content")
            if content is None:
                try:
                    content = result[user_turn][step_in_turn]
                except Exception:
                    content = ""
            has_tool_call, parsed_calls = _extract_json_tool_calls(content or "")
            tool_name = None
            arguments = None
            if parsed_calls:
                first = parsed_calls[0]
                tool_name = first.get("name")
                arguments = first.get("arguments")
            handler_log = next(
                (
                    item
                    for item in step_log
                    if isinstance(item, dict)
                    and item.get("role") == "handler_log"
                    and "model_response_decoded" in item
                ),
                {},
            )
            decoded = handler_log.get("model_response_decoded")
            parse_success = bool(decoded and not is_empty_execute_response(decoded))
            tool_entries = [
                item
                for item in step_log
                if isinstance(item, dict) and item.get("role") == "tool"
            ]
            execution_success = bool(tool_entries) and all(
                _execution_ok(item.get("content")) for item in tool_entries
            )
            execution_error = None
            for item in tool_entries:
                if not _execution_ok(item.get("content")):
                    execution_error = item.get("content")
                    break
            step_rows.append(
                {
                    "id": row.get("id"),
                    "global_step": global_step,
                    "user_turn": user_turn,
                    "step_in_turn": step_in_turn,
                    "has_tool_call": has_tool_call,
                    "tool_call_parse_success": parse_success,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "valid_tool": bool(tool_name and tool_name in valid_tools),
                    "execution_success": execution_success,
                    "execution_error": execution_error,
                    "state_pass_after_turn": None,
                    "response_pass_after_turn": None,
                }
            )
            global_step += 1
    return step_rows


def _decode_result(
    decoder: QwenFCHandler,
    result: Any,
) -> list[list[list[str]]]:
    decoded: list[list[list[str]]] = []
    if not isinstance(result, list):
        return decoded
    for turn in result:
        turn_rows = []
        if isinstance(turn, list):
            for step_text in turn:
                try:
                    calls = decoder.decode_execute(step_text, has_tool_call_tag=False)
                except Exception:
                    calls = []
                if calls and not is_empty_execute_response(calls):
                    turn_rows.append(calls)
        decoded.append(turn_rows)
    return decoded


def _turn_pass_rows(
    *,
    mode: str,
    decoder: QwenFCHandler,
    row: dict[str, Any],
    prompt_entry: dict[str, Any],
    ground_truth: list[list[str]],
) -> list[dict[str, Any]]:
    decoded = _decode_result(decoder, row.get("result"))
    test_entry_id = row["id"]
    test_category = test_entry_id.rsplit("_", 1)[0]
    model_name = (
        f"c2kv_analysis_{mode}_{test_entry_id}"
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
    )
    long_context = "long_context" in test_category or "composite" in test_category
    all_model_execution_results: list[str] = []
    turn_rows = []
    for turn_index, gt_calls in enumerate(ground_truth):
        step_calls = decoded[turn_index] if turn_index < len(decoded) else []
        model_turn_execution_results = []
        model_turn_execution_results_uncombined = []
        model_instances = {}
        for single_step_calls in step_calls:
            step_results, model_instances = execute_multi_turn_func_call(
                func_call_list=single_step_calls,
                initial_config=prompt_entry["initial_config"],
                involved_classes=prompt_entry["involved_classes"],
                model_name=model_name,
                test_entry_id=test_entry_id,
                long_context=long_context,
                is_evaL_run=True,
            )
            model_turn_execution_results.extend(step_results)
            model_turn_execution_results_uncombined.append(step_results)
        gt_results, gt_instances = execute_multi_turn_func_call(
            func_call_list=gt_calls,
            initial_config=prompt_entry["initial_config"],
            involved_classes=prompt_entry["involved_classes"],
            model_name=model_name + "_ground_truth",
            test_entry_id=test_entry_id,
            long_context=long_context,
            is_evaL_run=True,
        )
        all_model_execution_results.extend(model_turn_execution_results)
        if gt_calls and (not step_calls or is_empty_execute_response(step_calls)):
            state_pass = False
            response_pass = False
        elif not gt_calls:
            # Match BFCL multi_turn_checker: empty-ground-truth turns execute any
            # model calls to keep state progression, then skip state/response checks.
            state_pass = True
            response_pass = True
        else:
            state_pass = bool(model_instances) and state_checker(
                model_instances,
                gt_instances,
            )["valid"]
            response_pass = response_checker(
                all_model_execution_results,
                gt_results,
                turn_index,
            )["valid"]
        turn_rows.append(
            {
                "id": test_entry_id,
                "turn": turn_index,
                "state_pass": state_pass,
                "response_pass": response_pass,
                "joint_pass": state_pass and response_pass,
                "model_execution_result": model_turn_execution_results_uncombined,
                "ground_truth_execution_result": gt_results,
            }
        )
    return turn_rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_prompts_and_answers(category: str) -> tuple[dict[str, Any], dict[str, Any]]:
    prompts = load_dataset_entry(
        category,
        include_prereq=False,
        include_language_specific_hint=False,
    )
    answers = load_ground_truth_entry(category)
    prompt_by_id = {entry["id"]: entry for entry in prompts}
    answer_by_id = {
        prompt["id"]: answer.get("ground_truth", [])
        for prompt, answer in zip(prompts, answers)
    }
    return prompt_by_id, answer_by_id


def _analysis_rows(
    run_root: Path,
    mode: str,
    category: str,
    decoder: QwenFCHandler,
    prompt_by_id: dict[str, Any],
    answer_by_id: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode_root = run_root / mode
    turn_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for row in _result_rows(mode_root, category):
        sample_id = row.get("id")
        prompt_entry = prompt_by_id.get(sample_id)
        ground_truth = answer_by_id.get(sample_id)
        if prompt_entry is None or ground_truth is None:
            continue
        sample_turn_rows = _turn_pass_rows(
            mode=mode,
            decoder=decoder,
            row=row,
            prompt_entry=prompt_entry,
            ground_truth=ground_truth,
        )
        turn_rows.extend(sample_turn_rows)
        sample_step_rows = _step_rows_for_episode(row, prompt_entry)
        turn_by_index = {item["turn"]: item for item in sample_turn_rows}
        for step in sample_step_rows:
            turn = turn_by_index.get(step["user_turn"])
            if turn:
                step["state_pass_after_turn"] = turn["state_pass"]
                step["response_pass_after_turn"] = turn["response_pass"]
        step_rows.extend(sample_step_rows)
    _write_jsonl(mode_root / "logs" / "turn_metrics.jsonl", turn_rows)
    _write_jsonl(mode_root / "logs" / "step_metrics.jsonl", step_rows)
    return turn_rows, step_rows


def _first_failure_buckets(turn_rows: list[dict[str, Any]]) -> dict[str, int]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in turn_rows:
        by_id.setdefault(row["id"], []).append(row)
    buckets = {"turn_1_fail": 0, "turn_2_fail": 0, "turn_3_fail": 0, "turn_4plus_fail": 0}
    for rows in by_id.values():
        for row in sorted(rows, key=lambda item: item["turn"]):
            if not row["joint_pass"]:
                turn = int(row["turn"]) + 1
                if turn <= 1:
                    buckets["turn_1_fail"] += 1
                elif turn == 2:
                    buckets["turn_2_fail"] += 1
                elif turn == 3:
                    buckets["turn_3_fail"] += 1
                else:
                    buckets["turn_4plus_fail"] += 1
                break
    return buckets


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _mode_summary(
    run_root: Path,
    mode: str,
    category: str,
    turn_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    mode_root = run_root / mode
    metrics = _load_jsonl(mode_root / "logs" / "c2kv_metrics.jsonl")
    score = _score_header(mode_root, category)
    original = sum(int(row.get("tool_original_tokens") or 0) for row in metrics)
    effective = sum(int(row.get("tool_effective_tokens") or 0) for row in metrics)
    extract_calls = sum(int(row.get("extract_calls") or 0) for row in metrics)
    extract_success = sum(int(row.get("extract_success") or 0) for row in metrics)
    chat_calls = sum(int(row.get("chat_calls") or 0) for row in metrics)
    chat_seconds = sum(float(row.get("chat_seconds") or 0.0) for row in metrics)
    total_seconds = sum(float(row.get("total_seconds") or 0.0) for row in metrics)
    avg_tools = _mean([float(row.get("num_tools") or 0.0) for row in metrics])
    avg_full_tools = _mean(
        [float(row.get("avg_full_tools")) for row in metrics if row.get("avg_full_tools") is not None]
    )
    avg_compressed_tools = _mean(
        [
            float(row.get("avg_compressed_tools"))
            for row in metrics
            if row.get("avg_compressed_tools") is not None
        ]
    )
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
        "tool_definition_original_tokens": original,
        "tool_definition_effective_tokens": effective,
        "compression_ratio": (original / effective if effective else 1.0),
        "extract_calls": extract_calls,
        "extract_success_rate": (
            extract_success / extract_calls if extract_calls else None
        ),
        "average_number_of_tools": avg_tools,
        "average_chat_latency": (chat_seconds / chat_calls if chat_calls else None),
        "average_total_latency": (total_seconds / len(metrics) if metrics else None),
        "hybrid_top_k": 3 if mode == "hybrid" else None,
        "router_hit_rate": None,
        "average_full_tools": avg_full_tools,
        "average_compressed_tools": avg_compressed_tools,
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
    lines = [
        "# BFCL multi_turn_base - Tool Definition Compression",
        "",
        "## Episode / Turn / Step",
        "",
        "| Method | BFCL Acc | Correct | Episodes | Turn State | Turn Response | Turn Joint | Tool Call | Executable Tool Call | Valid Tool | Exec Success | Avg Steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"full": "Full", "c2kv": "C2KV@4", "hybrid": "Hybrid@4 Top3"}
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
            "## Compression / Latency",
            "",
            "| Method | Compression | Avg Chat s | Avg Total s | Extract Success | Router Hit |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | {comp}x | {chat} | {total} | {extract} | {router} |".format(
                method=names.get(row["method"], row["method"]),
                comp=_fmt(row.get("compression_ratio")),
                chat=_fmt(row.get("average_chat_latency")),
                total=_fmt(row.get("average_total_latency")),
                extract=_fmt(row.get("extract_success_rate")),
                router=_fmt(row.get("router_hit_rate"), none="N/A"),
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
    for mode in MODES:
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
    csv_path = run_root / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
