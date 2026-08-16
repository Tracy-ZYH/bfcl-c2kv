from __future__ import annotations

import argparse
import json
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

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

from c2kv_eval.adapters.bfcl_history_branch import _assistant_text
from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    HistoryDriftRunner,
    _assistant_history_message,
    _normalize_action_text,
    _normalize_state,
    _state_log,
    _tool_payload,
)


ORACLE_BUDGETS = {
    "c2kv4_oracle_correct1": 1,
    "c2kv4_oracle_correct2": 2,
    "c2kv4_oracle_correct4": 4,
    "c2kv4_oracle_correct_all": None,
}


class HistoryOracleRunner(HistoryDriftRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        drift_args = deepcopy(args)
        drift_args.mode = "history_c2kv4_closed_loop"
        super().__init__(drift_args)
        self.oracle_mode = args.oracle_mode
        self.correction_budget = ORACLE_BUDGETS[self.oracle_mode]

    def _can_correct(self, corrections_used: int) -> bool:
        return self.correction_budget is None or corrections_used < self.correction_budget

    def run_sample_oracle(self, test_case: dict[str, Any]) -> dict[str, Any]:
        stats = DriftStats(test_case["id"], self.oracle_mode, self.ratio)
        try:
            result, metadata = self._run_sample_oracle_impl(test_case, stats)
        except Exception as exc:
            result = f"Error during inference: {exc}"
            metadata = {"traceback": traceback.format_exc()}
            stats.errors.append(str(exc))
        metadata["c2kv_oracle_metrics"] = {
            **stats.as_dict(),
            "oracle_mode": self.oracle_mode,
            "correction_budget": self.correction_budget,
            "corrections_used": metadata.get("corrections_used", 0),
            "correction_events": metadata.get("correction_events", []),
        }
        return {"id": test_case["id"], "result": result, **metadata}

    def _run_sample_oracle_impl(
        self,
        test_case: dict[str, Any],
        stats: DriftStats,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        tools = _tool_payload(test_case["function"])
        long_context = "long_context" in test_category or "composite" in test_category
        reference_steps = self._reference_steps(test_entry_id)
        reference_step_by_turn = {
            (int(step.get("turn")), int(step.get("step"))): step
            for step in reference_steps
            if step.get("turn") is not None and step.get("step") is not None
        }
        reference_row = self.reference_by_id.get(test_entry_id) or {}
        reference_result = reference_row.get("result") or []

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
        correction_events: list[dict[str, Any]] = []
        corrections_used = 0
        force_quit = False

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}

            count = 0
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
                step_log = [{"role": "assistant", "content": text}]
                turn_log[f"step_{count}"] = step_log

                try:
                    decoded_prediction = self._decode(text)
                    handler_log = (
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded_prediction,
                        }
                    )
                    step_log.append(handler_log)
                except Exception as exc:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response. Proceed to next turn.",
                            "error": str(exc),
                        }
                    )
                    current_turn_response.append(text)
                    current_turn_inputs.append(usage["prompt_tokens"])
                    current_turn_outputs.append(usage["completion_tokens"])
                    current_turn_latency.append(elapsed)
                    break

                ref_index = len(drift_steps)
                ref_step = reference_step_by_turn.get((turn_idx, count))
                reference_action = ref_step.get("decoded_action") if ref_step else None
                is_action_drift = (
                    _normalize_action_text(decoded_prediction)
                    != _normalize_action_text(reference_action or [])
                )
                reference_has_action = bool(
                    reference_action and not is_empty_execute_response(reference_action)
                )
                use_oracle = bool(
                    is_action_drift
                    and self._can_correct(corrections_used)
                )

                if is_empty_execute_response(decoded_prediction) and not use_oracle:
                    current_turn_response.append(text)
                    current_turn_inputs.append(usage["prompt_tokens"])
                    current_turn_outputs.append(usage["completion_tokens"])
                    current_turn_latency.append(elapsed)
                    if stats.first_action_divergence is None and is_action_drift:
                        stats.first_action_divergence = {
                            "turn": turn_idx,
                            "step": count,
                            "global_step": ref_index,
                        }
                    break

                if use_oracle:
                    corrections_used += 1
                    correction_events.append(
                        {
                            "global_step": ref_index,
                            "turn": turn_idx,
                            "step": count,
                            "predicted_action": decoded_prediction,
                            "reference_action": reference_action,
                            "suppressed_extra_action": not reference_has_action,
                        }
                    )
                    if reference_has_action:
                        executed_action = list(reference_action)
                        executed_assistant = deepcopy(
                            ref_step.get("assistant_message") or assistant_history
                        )
                        result_text = _assistant_text(executed_assistant)
                    else:
                        executed_action = []
                        try:
                            result_text = str(reference_result[turn_idx][count])
                        except Exception:
                            result_text = ""
                        executed_assistant = {
                            "role": "assistant",
                            "content": result_text,
                        }
                    messages.append(executed_assistant)
                    current_turn_response.append(result_text)
                    step_log[0] = {"role": "assistant", "content": result_text}
                    handler_log["model_response_decoded"] = executed_action
                    step_log.append(
                        {
                            "role": "oracle",
                            "content": "Replaced divergent C2KV action with Full reference action.",
                            "candidate_text": text,
                            "candidate_assistant_message": assistant_history,
                            "candidate_action": decoded_prediction,
                            "executed_action": executed_action,
                            "corrections_used": corrections_used,
                            "suppressed_extra_action": not reference_has_action,
                        }
                    )
                else:
                    executed_action = decoded_prediction
                    messages.append(assistant_history)
                    current_turn_response.append(text)

                current_turn_inputs.append(usage["prompt_tokens"])
                current_turn_outputs.append(usage["completion_tokens"])
                current_turn_latency.append(elapsed)

                if is_empty_execute_response(executed_action):
                    execution_results = []
                else:
                    execution_results, involved_instances = execute_multi_turn_func_call(
                        executed_action,
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
                if stats.first_action_divergence is None and is_action_drift:
                    stats.first_action_divergence = {
                        "turn": turn_idx,
                        "step": count,
                        "global_step": ref_index,
                    }
                if (
                    stats.first_state_divergence is None
                    and ref_step is not None
                    and _normalize_state(state_after_step) != _normalize_state(ref_step.get("state"))
                ):
                    stats.first_state_divergence = {
                        "turn": turn_idx,
                        "step": count,
                        "global_step": ref_index,
                    }

                drift_steps.append(
                    {
                        "turn": turn_idx,
                        "step": count,
                        "assistant_message": assistant_history,
                        "executed_assistant_message": (
                            messages[-1 - len(execution_results)]
                            if use_oracle
                            else assistant_history
                        ),
                        "decoded_action": decoded_prediction,
                        "executed_action": executed_action,
                        "oracle_corrected": use_oracle,
                        "oracle_suppressed_extra_action": (
                            use_oracle and not reference_has_action
                        ),
                        "execution_results": execution_results,
                        "history_execution_results": execution_results,
                        "state": state_after_step,
                        "reference_action": reference_action,
                        "reference_state": ref_step.get("state") if ref_step else None,
                        "action_matches_reference": (
                            is_empty_execute_response(decoded_prediction)
                            if ref_step is None
                            else (
                                _normalize_action_text(decoded_prediction)
                                == _normalize_action_text(reference_action or [])
                            )
                        ),
                        "executed_action_matches_reference": (
                            is_empty_execute_response(executed_action)
                            if ref_step is None
                            else (
                                _normalize_action_text(executed_action)
                                == _normalize_action_text(reference_action or [])
                            )
                        ),
                        "state_matches_reference": (
                            None
                            if ref_step is None
                            else _normalize_state(state_after_step)
                            == _normalize_state(ref_step.get("state"))
                        ),
                    }
                )
                if use_oracle and not reference_has_action:
                    break
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

            all_model_response.append(current_turn_response)
            input_token_count.append(current_turn_inputs)
            output_token_count.append(current_turn_outputs)
            latency.append(current_turn_latency)
            inference_log.append(turn_log)
            state = _state_log(involved_instances)
            if state:
                inference_log.append(state)
            if force_quit:
                break

        metadata = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
            "inference_log": inference_log,
            "drift_steps": drift_steps,
            "corrections_used": corrections_used,
            "correction_events": correction_events,
        }
        return all_model_response, metadata


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    runner = HistoryOracleRunner(args)
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

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    details_rows = []
    metrics_rows = []
    for test_case in tqdm(entries, desc=f"{args.oracle_mode}:{args.category}", dynamic_ncols=True):
        row = runner.run_sample_oracle(deepcopy(test_case))
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metrics_rows.append(row.get("c2kv_oracle_metrics", {}))

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metrics_rows)
    summary = {
        "oracle_mode": args.oracle_mode,
        "category": args.category,
        "num_examples": len(details_rows),
        "corrections_used": sum(int(row.get("corrections_used") or 0) for row in metrics_rows),
        "extract_calls": sum(int(row.get("extract_calls") or 0) for row in metrics_rows),
        "extract_success": sum(int(row.get("extract_success") or 0) for row in metrics_rows),
    }
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-mode", choices=sorted(ORACLE_BUDGETS), required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--reference-details-path", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--recent-full-units", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
