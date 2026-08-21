from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mt_utils
from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.utils import (
    load_dataset_entry,
    make_json_serializable,
    sort_file_content_by_id,
    sort_key,
)

from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    HistoryDriftRunner,
    _assistant_history_message,
    _normalize_action_text,
    _normalize_state,
    _state_log,
    _tool_calls_to_text,
    _tool_payload,
)


def _assistant_text(message: dict[str, Any]) -> str:
    text = message.get("content") or ""
    tool_text = _tool_calls_to_text(message.get("tool_calls"))
    if tool_text:
        return (text + "\n" + tool_text).strip() if text else tool_text
    return text


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


def _instance_key(model_name: str, test_entry_id: str, class_name: str) -> str:
    return re.sub(r"[-./:]", "_", f"{model_name}_{test_entry_id}_{class_name}_instance")


class HistoryBranchRunner(HistoryDriftRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        drift_args = deepcopy(args)
        drift_args.mode = "history_c2kv4_closed_loop"
        super().__init__(drift_args)

    def _restore_instances(
        self,
        test_entry_id: str,
        involved_instances: dict[str, Any],
    ) -> None:
        for class_name, instance in involved_instances.items():
            mt_utils.__dict__[
                _instance_key(
                    self.decoder.model_name_underline_replaced,
                    test_entry_id,
                    class_name,
                )
            ] = deepcopy(instance)

    def _execute_from_snapshot(
        self,
        *,
        decoded_to_execute: list[str],
        snapshot_instances: dict[str, Any],
        initial_config: dict[str, Any],
        involved_classes: list[str],
        test_entry_id: str,
        long_context: bool,
    ) -> tuple[list[str], dict[str, Any]]:
        self._restore_instances(test_entry_id, snapshot_instances)
        return execute_multi_turn_func_call(
            decoded_to_execute,
            initial_config,
            involved_classes,
            self.decoder.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )

    def _continue_branch(
        self,
        *,
        branch_name: str,
        test_case: dict[str, Any],
        stats: DriftStats,
        messages: list[dict[str, Any]],
        involved_instances: dict[str, Any],
        all_model_response: list[list[str]],
        input_token_count: list[list[int]],
        output_token_count: list[list[int]],
        latency: list[list[float]],
        inference_log: list[Any],
        drift_steps: list[dict[str, Any]],
        reference_steps: Sequence[dict[str, Any]],
        start_turn_idx: int,
        start_step_idx: int,
    ) -> dict[str, Any]:
        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        tools = _tool_payload(test_case["function"])
        long_context = "long_context" in test_category or "composite" in test_category
        force_quit = False

        for turn_idx in range(start_turn_idx, len(test_case["question"])):
            if turn_idx >= len(all_model_response):
                current_turn_message = deepcopy(test_case["question"][turn_idx])
                messages.extend(current_turn_message)
                all_model_response.append([])
                input_token_count.append([])
                output_token_count.append([])
                latency.append([])
                inference_log.append({"begin_of_turn_query": current_turn_message})

            turn_log = inference_log[-1]
            count = start_step_idx if turn_idx == start_turn_idx else 0
            while True:
                request_messages = self._build_request_messages(messages, stats)
                text, response_message, elapsed, usage = self._query(
                    request_messages,
                    tools,
                    stats,
                )
                assistant_history = _assistant_history_message(
                    text,
                    response_message.get("tool_calls"),
                )
                all_model_response[turn_idx].append(text)
                input_token_count[turn_idx].append(usage["prompt_tokens"])
                output_token_count[turn_idx].append(usage["completion_tokens"])
                latency[turn_idx].append(elapsed)
                step_log = [{"role": "assistant", "content": text}]
                turn_log[f"step_{count}"] = step_log

                try:
                    decoded_prediction = self._decode(text)
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_prediction,
                        }
                    )
                except Exception as exc:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response. Proceed to next turn.",
                            "error": str(exc),
                        }
                    )
                    break

                if is_empty_execute_response(decoded_prediction):
                    break

                ref_index = len(drift_steps)
                ref_step = reference_steps[ref_index] if ref_index < len(reference_steps) else None
                decoded_to_execute = decoded_prediction
                messages.append(assistant_history)
                self._restore_instances(test_entry_id, involved_instances)
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_to_execute,
                    initial_config,
                    involved_classes,
                    self.decoder.model_name_underline_replaced,
                    test_entry_id,
                    long_context=long_context,
                    is_evaL_run=False,
                )
                for idx, execution_result in enumerate(execution_results):
                    messages.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                            "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                        }
                    )
                    step_log.append({"role": "tool", "content": execution_result})

                state_after_step = _state_log(involved_instances)
                drift_steps.append(
                    {
                        "turn": turn_idx,
                        "step": count,
                        "assistant_message": assistant_history,
                        "decoded_action": decoded_prediction,
                        "executed_action": decoded_to_execute,
                        "execution_results": execution_results,
                        "history_execution_results": execution_results,
                        "state": state_after_step,
                        "reference_action": (
                            ref_step.get("decoded_action") if ref_step else None
                        ),
                        "reference_state": ref_step.get("state") if ref_step else None,
                        "action_matches_reference": (
                            None
                            if ref_step is None
                            else _normalize_action_text(decoded_prediction)
                            == _normalize_action_text(ref_step.get("decoded_action") or [])
                        ),
                        "state_matches_reference": (
                            None
                            if ref_step is None
                            else _normalize_state(state_after_step)
                            == _normalize_state(ref_step.get("state"))
                        ),
                    }
                )
                count += 1
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": f"Model has been forced to quit after {MAXIMUM_STEP_LIMIT} steps.",
                        }
                    )
                    break

            state = _state_log(involved_instances)
            if state:
                inference_log.append(state)
            if force_quit:
                break

        return {
            "id": test_entry_id,
            "result": all_model_response,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
            "inference_log": inference_log,
            "drift_steps": drift_steps,
            "c2kv_branch_metrics": {
                **stats.as_dict(),
                "branch": branch_name,
            },
        }

    def _make_branch_at_drift(
        self,
        *,
        branch_name: str,
        test_case: dict[str, Any],
        stats: DriftStats,
        snapshot_messages: list[dict[str, Any]],
        snapshot_instances: dict[str, Any],
        common_all_model_response: list[list[str]],
        common_input_token_count: list[list[int]],
        common_output_token_count: list[list[int]],
        common_latency: list[list[float]],
        common_inference_log: list[Any],
        common_drift_steps: list[dict[str, Any]],
        current_turn_idx: int,
        current_step_idx: int,
        c2kv_text: str,
        c2kv_assistant: dict[str, Any],
        c2kv_decoded: list[str],
        c2kv_usage: dict[str, int],
        c2kv_elapsed: float,
        reference_steps: Sequence[dict[str, Any]],
        reference_step: dict[str, Any],
    ) -> dict[str, Any]:
        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        long_context = "long_context" in test_category or "composite" in test_category
        turn_log = deepcopy(common_inference_log[-1])
        messages = deepcopy(snapshot_messages)
        all_model_response = deepcopy(common_all_model_response)
        input_token_count = deepcopy(common_input_token_count)
        output_token_count = deepcopy(common_output_token_count)
        latency = deepcopy(common_latency)
        inference_log = deepcopy(common_inference_log[:-1]) + [turn_log]
        drift_steps = deepcopy(common_drift_steps)

        if branch_name == "natural":
            assistant_history = deepcopy(c2kv_assistant)
            result_text = c2kv_text
            decoded_to_execute = c2kv_decoded
            decoded_prediction = c2kv_decoded
        else:
            assistant_history = deepcopy(reference_step.get("assistant_message") or {})
            result_text = _assistant_text(assistant_history)
            decoded_to_execute = list(reference_step.get("decoded_action") or [])
            decoded_prediction = c2kv_decoded

        all_model_response[current_turn_idx].append(result_text)
        input_token_count[current_turn_idx].append(c2kv_usage["prompt_tokens"])
        output_token_count[current_turn_idx].append(c2kv_usage["completion_tokens"])
        latency[current_turn_idx].append(c2kv_elapsed)
        step_log = [
            {"role": "assistant", "content": result_text},
            {
                "role": "handler_log",
                "content": "Successfully decoded model response.",
                "model_response_decoded": decoded_to_execute,
            },
        ]
        turn_log[f"step_{current_step_idx}"] = step_log
        messages.append(assistant_history)

        execution_results, involved_instances = self._execute_from_snapshot(
            decoded_to_execute=decoded_to_execute,
            snapshot_instances=snapshot_instances,
            initial_config=initial_config,
            involved_classes=involved_classes,
            test_entry_id=test_entry_id,
            long_context=long_context,
        )
        for idx, execution_result in enumerate(execution_results):
            messages.append(
                {
                    "role": "tool",
                    "content": execution_result,
                    "tool_call_id": f"call_{current_turn_idx}_{current_step_idx}_{idx}",
                }
            )
            step_log.append({"role": "tool", "content": execution_result})

        state_after_step = _state_log(involved_instances)
        drift_steps.append(
            {
                "turn": current_turn_idx,
                "step": current_step_idx,
                "assistant_message": assistant_history,
                "decoded_action": decoded_prediction,
                "executed_action": decoded_to_execute,
                "execution_results": execution_results,
                "history_execution_results": execution_results,
                "state": state_after_step,
                "reference_action": reference_step.get("decoded_action"),
                "reference_state": reference_step.get("state"),
                "action_matches_reference": _normalize_action_text(decoded_prediction)
                == _normalize_action_text(reference_step.get("decoded_action") or []),
                "state_matches_reference": _normalize_state(state_after_step)
                == _normalize_state(reference_step.get("state")),
            }
        )
        stats.first_action_divergence = {
            "turn": current_turn_idx,
            "step": current_step_idx,
            "global_step": len(common_drift_steps),
        }
        if stats.first_state_divergence is None and _normalize_state(state_after_step) != _normalize_state(reference_step.get("state")):
            stats.first_state_divergence = {
                "turn": current_turn_idx,
                "step": current_step_idx,
                "global_step": len(common_drift_steps),
            }

        return self._continue_branch(
            branch_name=branch_name,
            test_case=test_case,
            stats=stats,
            messages=messages,
            involved_instances=involved_instances,
            all_model_response=all_model_response,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            latency=latency,
            inference_log=inference_log,
            drift_steps=drift_steps,
            reference_steps=reference_steps,
            start_turn_idx=current_turn_idx,
            start_step_idx=current_step_idx + 1,
        )

    def run_sample_branches(self, test_case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        natural_stats = DriftStats(test_case["id"], "history_branch_natural", self.ratio)
        corrected_stats = DriftStats(test_case["id"], "history_branch_corrected", self.ratio)
        try:
            return self._run_sample_branches_impl(
                test_case,
                natural_stats,
                corrected_stats,
            )
        except Exception as exc:
            metadata = {
                "traceback": traceback.format_exc(),
                "c2kv_branch_metrics": {
                    "id": test_case["id"],
                    "errors": [str(exc)],
                },
            }
            row = {"id": test_case["id"], "result": f"Error during inference: {exc}", **metadata}
            return deepcopy(row), deepcopy(row)

    def _run_sample_branches_impl(
        self,
        test_case: dict[str, Any],
        natural_stats: DriftStats,
        corrected_stats: DriftStats,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        tools = _tool_payload(test_case["function"])
        long_context = "long_context" in test_category or "composite" in test_category
        reference_steps = self._reference_steps(test_entry_id)

        _, involved_instances = execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.decoder.model_name_underline_replaced,
            test_entry_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        inference_log: list[Any] = []
        initial_state = _state_log(involved_instances)
        if initial_state:
            inference_log.append(initial_state)
        messages: list[dict[str, Any]] = []
        all_model_response: list[list[str]] = []
        input_token_count: list[list[int]] = []
        output_token_count: list[list[int]] = []
        latency: list[list[float]] = []
        drift_steps: list[dict[str, Any]] = []

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            all_model_response.append([])
            input_token_count.append([])
            output_token_count.append([])
            latency.append([])
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}
            inference_log.append(turn_log)
            count = 0
            while True:
                snapshot_messages = deepcopy(messages)
                snapshot_instances = deepcopy(involved_instances)
                request_messages = self._build_request_messages(messages, natural_stats)
                text, response_message, elapsed, usage = self._query(
                    request_messages,
                    tools,
                    natural_stats,
                )
                corrected_stats.chat_calls += 1
                corrected_stats.chat_seconds += elapsed
                assistant_history = _assistant_history_message(
                    text,
                    response_message.get("tool_calls"),
                )
                try:
                    decoded_prediction = self._decode(text)
                except Exception:
                    decoded_prediction = []

                ref_index = len(drift_steps)
                reference_step = (
                    reference_steps[ref_index] if ref_index < len(reference_steps) else None
                )
                is_action_drift = (
                    reference_step is not None
                    and not is_empty_execute_response(decoded_prediction)
                    and _normalize_action_text(decoded_prediction)
                    != _normalize_action_text(reference_step.get("decoded_action") or [])
                )
                if is_action_drift:
                    natural = self._make_branch_at_drift(
                        branch_name="natural",
                        test_case=test_case,
                        stats=natural_stats,
                        snapshot_messages=snapshot_messages,
                        snapshot_instances=snapshot_instances,
                        common_all_model_response=all_model_response,
                        common_input_token_count=input_token_count,
                        common_output_token_count=output_token_count,
                        common_latency=latency,
                        common_inference_log=inference_log,
                        common_drift_steps=drift_steps,
                        current_turn_idx=turn_idx,
                        current_step_idx=count,
                        c2kv_text=text,
                        c2kv_assistant=assistant_history,
                        c2kv_decoded=decoded_prediction,
                        c2kv_usage=usage,
                        c2kv_elapsed=elapsed,
                        reference_steps=reference_steps,
                        reference_step=reference_step,
                    )
                    corrected = self._make_branch_at_drift(
                        branch_name="corrected",
                        test_case=test_case,
                        stats=corrected_stats,
                        snapshot_messages=snapshot_messages,
                        snapshot_instances=snapshot_instances,
                        common_all_model_response=all_model_response,
                        common_input_token_count=input_token_count,
                        common_output_token_count=output_token_count,
                        common_latency=latency,
                        common_inference_log=inference_log,
                        common_drift_steps=drift_steps,
                        current_turn_idx=turn_idx,
                        current_step_idx=count,
                        c2kv_text=text,
                        c2kv_assistant=assistant_history,
                        c2kv_decoded=decoded_prediction,
                        c2kv_usage=usage,
                        c2kv_elapsed=elapsed,
                        reference_steps=reference_steps,
                        reference_step=reference_step,
                    )
                    return natural, corrected

                all_model_response[turn_idx].append(text)
                input_token_count[turn_idx].append(usage["prompt_tokens"])
                output_token_count[turn_idx].append(usage["completion_tokens"])
                latency[turn_idx].append(elapsed)
                step_log = [
                    {"role": "assistant", "content": text},
                    {
                        "role": "handler_log",
                        "content": "Successfully decoded model response.",
                        "model_response_decoded": decoded_prediction,
                    },
                ]
                turn_log[f"step_{count}"] = step_log

                if is_empty_execute_response(decoded_prediction):
                    break
                messages.append(assistant_history)
                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_prediction,
                    initial_config,
                    involved_classes,
                    self.decoder.model_name_underline_replaced,
                    test_entry_id,
                    long_context=long_context,
                    is_evaL_run=False,
                )
                for idx, execution_result in enumerate(execution_results):
                    messages.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                            "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                        }
                    )
                    step_log.append({"role": "tool", "content": execution_result})
                state_after_step = _state_log(involved_instances)
                drift_steps.append(
                    {
                        "turn": turn_idx,
                        "step": count,
                        "assistant_message": assistant_history,
                        "decoded_action": decoded_prediction,
                        "executed_action": decoded_prediction,
                        "execution_results": execution_results,
                        "history_execution_results": execution_results,
                        "state": state_after_step,
                        "reference_action": (
                            reference_step.get("decoded_action") if reference_step else None
                        ),
                        "reference_state": (
                            reference_step.get("state") if reference_step else None
                        ),
                        "action_matches_reference": (
                            None
                            if reference_step is None
                            else _normalize_action_text(decoded_prediction)
                            == _normalize_action_text(reference_step.get("decoded_action") or [])
                        ),
                        "state_matches_reference": (
                            None
                            if reference_step is None
                            else _normalize_state(state_after_step)
                            == _normalize_state(reference_step.get("state"))
                        ),
                    }
                )
                count += 1
                if count > MAXIMUM_STEP_LIMIT:
                    break
            state = _state_log(involved_instances)
            if state:
                inference_log.append(state)

        row = {
            "id": test_entry_id,
            "result": all_model_response,
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
            "inference_log": inference_log,
            "drift_steps": drift_steps,
            "c2kv_branch_metrics": {
                **natural_stats.as_dict(),
                "branch": "no_drift",
            },
        }
        return deepcopy(row), deepcopy(row)


def run(args: argparse.Namespace) -> None:
    runner = HistoryBranchRunner(args)
    entries = load_dataset_entry(args.category)
    entries = [entry for entry in entries if entry["id"].startswith(args.category)]
    entries = sorted(entries, key=sort_key)
    if args.ids_path:
        with open(args.ids_path, encoding="utf-8") as f:
            selected_ids = {
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            }
        entries = [entry for entry in entries if entry["id"] in selected_ids]
    if args.max_examples is not None:
        entries = entries[: args.max_examples]

    natural_result_dir = Path(args.natural_result_dir)
    corrected_result_dir = Path(args.corrected_result_dir)
    natural_result_dir.mkdir(parents=True, exist_ok=True)
    corrected_result_dir.mkdir(parents=True, exist_ok=True)
    natural_rows: list[dict[str, Any]] = []
    corrected_rows: list[dict[str, Any]] = []
    for test_case in tqdm(entries, desc=f"history_branch:{args.category}", dynamic_ncols=True):
        natural, corrected = runner.run_sample_branches(deepcopy(test_case))
        runner.decoder.write(natural, result_dir=natural_result_dir, update_mode=False)
        runner.decoder.write(corrected, result_dir=corrected_result_dir, update_mode=False)
        natural_rows.append(natural)
        corrected_rows.append(corrected)

    for result_dir in (natural_result_dir, corrected_result_dir):
        for result_json in result_dir.rglob("*_result.json"):
            sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.natural_details_path), natural_rows)
    _write_jsonl(Path(args.corrected_details_path), corrected_rows)
    summary = {
        "category": args.category,
        "num_examples": len(natural_rows),
        "natural_result_dir": str(natural_result_dir),
        "corrected_result_dir": str(corrected_result_dir),
        "branch_points": sum(
            1
            for row in natural_rows
            if (row.get("c2kv_branch_metrics") or {}).get("branch") == "natural"
        ),
    }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--reference-details-path", required=True)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--recent-full-units", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--natural-result-dir", required=True)
    parser.add_argument("--corrected-result-dir", required=True)
    parser.add_argument("--natural-details-path", required=True)
    parser.add_argument("--corrected-details-path", required=True)
    parser.add_argument("--summary-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
