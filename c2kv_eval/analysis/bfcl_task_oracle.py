from __future__ import annotations

import argparse
import csv
import json
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
from bfcl_eval.utils import (
    load_dataset_entry,
    load_ground_truth_entry,
    make_json_serializable,
    sort_key,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FC"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_first(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _load_prompt_answer_maps(category: str) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _state_snapshot(instances: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, instance in (instances or {}).items():
        out[str(name)] = {
            key: value
            for key, value in vars(instance).items()
            if not key.startswith("_")
        }
    return make_json_serializable(out)


def build_ground_truth_cache(
    *,
    category: str,
    output_root: Path,
    max_examples: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    prompt_by_id, answer_by_id = _load_prompt_answer_maps(category)
    entries = sorted(
        [entry for entry in prompt_by_id.values() if entry["id"].startswith(category)],
        key=sort_key,
    )
    if max_examples is not None:
        entries = entries[:max_examples]

    turn_rows: list[dict[str, Any]] = []
    episode_ids: list[str] = []
    for entry in entries:
        episode_id = entry["id"]
        episode_ids.append(episode_id)
        test_category = episode_id.rsplit("_", 1)[0]
        long_context = "long_context" in test_category or "composite" in test_category
        model_name = (
            f"bfcl_task_gt_{episode_id}"
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        all_gt_results: list[str] = []
        for turn_id, gt_calls in enumerate(answer_by_id.get(episode_id, [])):
            gt_results, gt_instances = execute_multi_turn_func_call(
                func_call_list=gt_calls,
                initial_config=entry["initial_config"],
                involved_classes=entry["involved_classes"],
                model_name=model_name,
                test_entry_id=episode_id,
                long_context=long_context,
                is_evaL_run=True,
            )
            all_gt_results.extend(gt_results)
            turn_rows.append(
                {
                    "episode_id": episode_id,
                    "turn_id": turn_id,
                    "ground_truth_calls": gt_calls,
                    "ground_truth_required_execution_results": gt_results,
                    "ground_truth_all_execution_results": list(all_gt_results),
                    "ground_truth_state_after_turn": _state_snapshot(gt_instances),
                }
            )

    gt_dir = output_root / "ground_truth"
    _write_jsonl(gt_dir / "ground_truth_turns.jsonl", turn_rows)
    (gt_dir / "episode_ids.txt").write_text(
        "\n".join(episode_ids) + "\n",
        encoding="utf-8",
    )
    summary = {
        "category": category,
        "episode_count": len(episode_ids),
        "turn_count": len(turn_rows),
        "episode_ids_path": str(gt_dir / "episode_ids.txt"),
        "turns_path": str(gt_dir / "ground_truth_turns.jsonl"),
    }
    (gt_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return turn_rows, episode_ids


def _decode_result(decoder: QwenFCHandler, result: Any) -> list[list[list[str]]]:
    decoded: list[list[list[str]]] = []
    if not isinstance(result, list):
        return decoded
    for turn in result:
        turn_rows: list[list[str]] = []
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


def evaluate_result_rows(
    *,
    rows: list[dict[str, Any]],
    category: str,
    model: str,
    output_dir: Path,
    mode_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = MODEL_CONFIG_MAPPING[model]
    decoder = QwenFCHandler(
        model_name=config.model_name,
        temperature=0,
        registry_name=model,
        is_fc_model=config.is_fc_model,
    )
    prompt_by_id, answer_by_id = _load_prompt_answer_maps(category)
    turn_rows: list[dict[str, Any]] = []
    passed_episodes = 0
    valid_episode_count = 0
    decoded_step_total = 0
    expected_turn_total = 0

    for row in rows:
        episode_id = row.get("id")
        prompt_entry = prompt_by_id.get(episode_id)
        ground_truth = answer_by_id.get(episode_id)
        if prompt_entry is None or ground_truth is None:
            continue
        valid_episode_count += 1
        decoded = _decode_result(decoder, row.get("result"))
        decoded_step_total += sum(len(turn) for turn in decoded)
        expected_turn_total += len(ground_truth)
        test_category = episode_id.rsplit("_", 1)[0]
        long_context = "long_context" in test_category or "composite" in test_category
        model_name = (
            f"bfcl_task_eval_{mode_name}_{episode_id}"
            .replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
        all_model_results: list[str] = []
        episode_pass = True
        for turn_id, gt_calls in enumerate(ground_truth):
            step_calls = decoded[turn_id] if turn_id < len(decoded) else []
            model_turn_results: list[str] = []
            model_turn_results_uncombined: list[list[str]] = []
            model_instances: dict[str, Any] = {}
            for single_step_calls in step_calls:
                step_results, model_instances = execute_multi_turn_func_call(
                    func_call_list=single_step_calls,
                    initial_config=prompt_entry["initial_config"],
                    involved_classes=prompt_entry["involved_classes"],
                    model_name=model_name,
                    test_entry_id=episode_id,
                    long_context=long_context,
                    is_evaL_run=True,
                )
                model_turn_results.extend(step_results)
                model_turn_results_uncombined.append(step_results)
            gt_results, gt_instances = execute_multi_turn_func_call(
                func_call_list=gt_calls,
                initial_config=prompt_entry["initial_config"],
                involved_classes=prompt_entry["involved_classes"],
                model_name=model_name + "_ground_truth",
                test_entry_id=episode_id,
                long_context=long_context,
                is_evaL_run=True,
            )
            all_model_results.extend(model_turn_results)
            if gt_calls and (not step_calls or is_empty_execute_response(step_calls)):
                state_valid = False
                required_results_valid = False
                error_type = "multi_turn:empty_turn_model_response"
            elif not gt_calls:
                state_valid = True
                required_results_valid = True
                error_type = None
            else:
                state_result = (
                    state_checker(model_instances, gt_instances)
                    if model_instances
                    else {"valid": False, "error_type": "missing_model_instances"}
                )
                response_result = response_checker(
                    all_model_results,
                    gt_results,
                    turn_id,
                )
                state_valid = bool(state_result.get("valid"))
                required_results_valid = bool(response_result.get("valid"))
                error_type = None
                if not state_valid:
                    error_type = state_result.get("error_type")
                elif not required_results_valid:
                    error_type = response_result.get("error_type")
            task_valid = state_valid and required_results_valid
            episode_pass = episode_pass and task_valid
            turn_rows.append(
                {
                    "episode_id": episode_id,
                    "turn_id": turn_id,
                    "mode": mode_name,
                    "task_valid": task_valid,
                    "task_error": not task_valid,
                    "task_state_valid": state_valid,
                    "task_state_failure": not state_valid,
                    "task_required_results_valid": required_results_valid,
                    "task_required_result_failure": not required_results_valid,
                    "error_type": error_type,
                    "model_decoded_calls": step_calls,
                    "model_execution_result": model_turn_results_uncombined,
                    "ground_truth_calls": gt_calls,
                    "ground_truth_execution_result": gt_results,
                }
            )
        if episode_pass:
            passed_episodes += 1

    total_turns = len(turn_rows)
    summary = {
        "mode": mode_name,
        "episodes": valid_episode_count,
        "turns": total_turns,
        "avg_turns": (
            expected_turn_total / valid_episode_count if valid_episode_count else None
        ),
        "total_committed_steps": decoded_step_total,
        "bfcl_task_accuracy": (
            passed_episodes / valid_episode_count if valid_episode_count else None
        ),
        "correct_count": passed_episodes,
        "task_error_rate": (
            sum(1 for row in turn_rows if row["task_error"]) / total_turns
            if total_turns
            else None
        ),
        "task_state_failure_rate": (
            sum(1 for row in turn_rows if row["task_state_failure"]) / total_turns
            if total_turns
            else None
        ),
        "task_required_result_failure_rate": (
            sum(1 for row in turn_rows if row["task_required_result_failure"])
            / total_turns
            if total_turns
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / f"{mode_name}_task_turns.jsonl", turn_rows)
    (output_dir / f"{mode_name}_task_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return turn_rows, summary


def _details_rows(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    return _load_jsonl(Path(path))


def _result_rows_from_dir(path: str, category: str) -> list[dict[str, Any]]:
    if not path:
        return []
    found = _find_first(Path(path), f"*_{category}_result.json")
    return _load_jsonl(found) if found else []


def _drift_by_episode_turn(details: list[dict[str, Any]]) -> dict[tuple[str, int], bool]:
    out: dict[tuple[str, int], bool] = {}
    for row in details:
        episode_id = row.get("id")
        for step in row.get("drift_steps") or []:
            if not isinstance(step, dict):
                continue
            key = (episode_id, int(step.get("turn") or 0))
            out[key] = out.get(key, False) or bool(
                step.get("candidate_action_drift") or step.get("state_drift")
            )
    return out


def quadrant_stats(
    *,
    details: list[dict[str, Any]],
    task_turns: list[dict[str, Any]],
    assume_reference_identity: bool = False,
) -> dict[str, Any]:
    reference_drift = {} if assume_reference_identity else _drift_by_episode_turn(details)
    counts = {
        "reference_drift_0_task_error_0": 0,
        "reference_drift_1_task_error_0": 0,
        "reference_drift_0_task_error_1": 0,
        "reference_drift_1_task_error_1": 0,
    }
    for row in task_turns:
        key = (row["episode_id"], int(row["turn_id"]))
        rd = bool(reference_drift.get(key, False))
        te = bool(row.get("task_error"))
        counts[f"reference_drift_{int(rd)}_task_error_{int(te)}"] += 1
    counts["reference_drift_differs_from_task_error"] = (
        counts["reference_drift_1_task_error_0"]
        + counts["reference_drift_0_task_error_1"]
    )
    counts["turn_count"] = len(task_turns)
    return counts


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    gt_turns, episode_ids = build_ground_truth_cache(
        category=args.category,
        output_root=output_root,
        max_examples=args.max_examples,
    )
    if args.require_full200 and len(episode_ids) != 200:
        raise RuntimeError(f"expected 200 episodes, got {len(episode_ids)}")

    manifest_path = output_root / "experiment_config.json"
    manifest = _load_json(manifest_path)
    manifest.update({
        "category": args.category,
        "max_examples": args.max_examples,
        "episode_count": len(episode_ids),
        "uses_stable52_filter": False,
        "full_success_only": False,
        "ground_truth_cache_path": str(output_root / "ground_truth"),
    })
    summaries: list[dict[str, Any]] = []
    full_rows = _details_rows(args.full_details_path) or _result_rows_from_dir(
        args.full_result_dir,
        args.category,
    )
    if full_rows:
        full_turns, full_summary = evaluate_result_rows(
            rows=full_rows,
            category=args.category,
            model=args.model,
            output_dir=output_root / "full_reference",
            mode_name="full_reference",
        )
        quadrant = quadrant_stats(
            details=full_rows,
            task_turns=full_turns,
            assume_reference_identity=True,
        )
        full_summary["reference_drift_vs_task_error"] = quadrant
        summaries.append(full_summary)
        manifest["full_reference_details_path"] = args.full_details_path
        manifest["full_reference_result_dir"] = args.full_result_dir
        manifest["full_reference_path"] = str(output_root / "full_reference")

    if args.candidate_details_path:
        candidate_rows = _details_rows(args.candidate_details_path)
        task_turns, summary = evaluate_result_rows(
            rows=candidate_rows,
            category=args.category,
            model=args.model,
            output_dir=output_root / "candidate_task_oracle",
            mode_name=args.candidate_name,
        )
        summary["reference_drift_vs_task_error"] = quadrant_stats(
            details=candidate_rows,
            task_turns=task_turns,
        )
        summaries.append(summary)

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if summaries:
        summary_dir = output_root / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        flat_rows = []
        for summary in summaries:
            flat = dict(summary)
            quadrant = flat.pop("reference_drift_vs_task_error", {}) or {}
            flat.update(quadrant)
            flat_rows.append(flat)
        fields = sorted({key for row in flat_rows for key in row})
        with open(summary_dir / "task_oracle_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
        (summary_dir / "task_oracle_summary.json").write_text(
            json.dumps(flat_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output_root": str(output_root), "episodes": len(episode_ids)}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--require-full200", action="store_true")
    parser.add_argument("--full-details-path", default="")
    parser.add_argument("--full-result-dir", default="")
    parser.add_argument("--candidate-details-path", default="")
    parser.add_argument("--candidate-name", default="candidate")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
