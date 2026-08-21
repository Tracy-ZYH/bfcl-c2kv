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
from c2kv_eval.adapters.history_step_common import (
    action_matches,
    build_step_record,
    decode_candidate,
    mark_first_divergence,
    reference_by_turn_step,
    reference_step_for,
    serialization_roundtrip,
)


ORACLE_BUDGETS = {
    "pure_full_replay": None,
    "c2kv4_oracle_action_mismatch": None,
    "c2kv4_oracle_correct1": 1,
    "c2kv4_oracle_correct2": 2,
    "c2kv4_oracle_correct4": 4,
    "c2kv4_oracle_correct_all": None,
    "c2kv4_oracle_correct_all_strict": None,
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
            if self.oracle_mode == "pure_full_replay":
                result, metadata = self._run_sample_pure_full_replay_impl(
                    test_case,
                    stats,
                )
            else:
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

    def _reference_result_text(
        self,
        reference_result: Sequence[Sequence[Any]],
        turn_idx: int,
        step_idx: int,
    ) -> str:
        try:
            return str(reference_result[turn_idx][step_idx])
        except Exception:
            return ""

    def _reference_action(self, ref_step: dict[str, Any] | None) -> list[str]:
        if not ref_step:
            return []
        return list(ref_step.get("decoded_action") or [])

    def _run_sample_pure_full_replay_impl(
        self,
        test_case: dict[str, Any],
        stats: DriftStats,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        initial_config = test_case.get("initial_config", {})
        involved_classes = test_case["involved_classes"]
        test_entry_id = test_case["id"]
        test_category = test_entry_id.rsplit("_", 1)[0]
        long_context = "long_context" in test_category or "composite" in test_category
        reference_steps = self._reference_steps(test_entry_id)
        reference_step_by_turn = reference_by_turn_step(reference_steps)
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

        all_model_response: list[list[str]] = []
        input_token_count: list[list[int]] = []
        output_token_count: list[list[int]] = []
        latency: list[list[float]] = []
        drift_steps: list[dict[str, Any]] = []

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}
            reference_turn = (
                reference_result[turn_idx]
                if turn_idx < len(reference_result)
                and isinstance(reference_result[turn_idx], list)
                else []
            )

            for count, result_text in enumerate(reference_turn):
                result_text = str(result_text)
                step_log = [{"role": "assistant", "content": result_text}]
                turn_log[f"step_{count}"] = step_log
                decoded_action: list[str] = []
                decode_error: str | None = None
                try:
                    decoded_action = self._decode(result_text)
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded Full reference response.",
                            "model_response_decoded": decoded_action,
                        }
                    )
                except Exception as exc:
                    decode_error = str(exc)
                    stats.errors.append(f"pure_full_replay decode: {decode_error}")
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding Full reference response.",
                            "error": decode_error,
                            "model_response_decoded": decoded_action,
                        }
                    )

                current_turn_response.append(result_text)
                current_turn_inputs.append(0)
                current_turn_outputs.append(0)
                current_turn_latency.append(0.0)

                ref_step, alignment_status = reference_step_for(
                    reference_step_by_turn,
                    reference_result,
                    turn_idx,
                    count,
                    fallback_state=_state_log(involved_instances),
                )
                reference_action = self._reference_action(ref_step)
                execution_error = None
                if decode_error is not None or is_empty_execute_response(decoded_action):
                    execution_results = []
                else:
                    try:
                        execution_results, involved_instances = execute_multi_turn_func_call(
                            decoded_action,
                            initial_config,
                            involved_classes,
                            self.decoder.model_name_underline_replaced,
                            test_entry_id,
                            long_context=long_context,
                            is_evaL_run=False,
                        )
                    except Exception as exc:
                        execution_error = str(exc)
                        execution_results = []
                for execution_result in execution_results:
                    step_log.append({"role": "tool", "content": execution_result})

                state_after_step = _state_log(involved_instances)
                action_match = (
                    False
                    if decode_error is not None
                    else _normalize_action_text(decoded_action)
                    == _normalize_action_text(reference_action)
                )
                state_match = (
                    None
                    if ref_step is None
                    else _normalize_state(state_after_step)
                    == _normalize_state(ref_step.get("state"))
                )
                roundtrip = serialization_roundtrip(
                    self.decoder,
                    result_text,
                    decoded_action,
                )
                step_record = build_step_record(
                    sample_id=test_entry_id,
                    turn_idx=turn_idx,
                    step_idx=count,
                    global_step=len(drift_steps),
                    candidate_raw_text=result_text,
                    candidate_action=decoded_action,
                    candidate_status=(
                        "decode_error"
                        if decode_error is not None
                        else "empty_action"
                        if is_empty_execute_response(decoded_action)
                        else "decoded_action"
                    ),
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=decoded_action,
                    state=state_after_step,
                    decode_error=decode_error,
                    empty_response=not bool(result_text.strip()),
                    execution_error=execution_error,
                    candidate_assistant_message={
                        "role": "assistant",
                        "content": result_text,
                    },
                    executed_assistant_message={
                        "role": "assistant",
                        "content": result_text,
                    },
                    execution_results=execution_results,
                    history_execution_results=execution_results,
                    oracle_corrected=False,
                    response_matches_reference=True,
                    candidate_response_matches_reference=True,
                    roundtrip=roundtrip,
                    extra={"pure_full_replay": True},
                )
                step_record["action_matches_reference"] = action_match
                step_record["candidate_action_matches_reference"] = action_match
                step_record["candidate_action_drift"] = not action_match
                step_record["executed_action_matches_reference"] = action_match
                step_record["executed_action_drift"] = not action_match
                step_record["state_matches_reference"] = state_match
                step_record["state_drift"] = state_match is False
                drift_steps.append(step_record)
                mark_first_divergence(stats, step_record)
                if roundtrip["serialization_mismatch"]:
                    stats.errors.append(
                        "pure_full_replay serialization mismatch at "
                        f"{test_entry_id} turn={turn_idx} step={count}"
                    )
                if execution_error:
                    stats.errors.append(
                        "pure_full_replay execution error at "
                        f"{test_entry_id} turn={turn_idx} step={count}: "
                        f"{execution_error}"
                    )
                    break
                if is_empty_execute_response(decoded_action):
                    break

            all_model_response.append(current_turn_response)
            input_token_count.append(current_turn_inputs)
            output_token_count.append(current_turn_outputs)
            latency.append(current_turn_latency)
            inference_log.append(turn_log)
            state = _state_log(involved_instances)
            if state:
                inference_log.append(state)

        metadata = {
            "input_token_count": input_token_count,
            "output_token_count": output_token_count,
            "latency": latency,
            "inference_log": inference_log,
            "drift_steps": drift_steps,
            "corrections_used": 0,
            "correction_events": [],
        }
        return all_model_response, metadata

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
        reference_step_by_turn = reference_by_turn_step(reference_steps)
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

                candidate = decode_candidate(self.decoder, text)
                decoded_prediction = candidate.action
                decode_error = candidate.decode_error
                handler_log = {
                    "role": "handler_log",
                    "content": (
                        "Successfully decoded model response."
                        if candidate.status == "decoded_action"
                        else f"Candidate status: {candidate.status}."
                    ),
                    "model_response_decoded": decoded_prediction,
                    "candidate_status": candidate.status,
                }
                if decode_error is not None:
                    handler_log["error"] = decode_error
                step_log.append(handler_log)

                ref_index = len(drift_steps)
                ref_step, alignment_status = reference_step_for(
                    reference_step_by_turn,
                    reference_result,
                    turn_idx,
                    count,
                    fallback_state=_state_log(involved_instances),
                )
                reference_action = ref_step.get("decoded_action") if ref_step else None
                reference_has_action = bool(
                    reference_action and not is_empty_execute_response(reference_action)
                )
                reference_text = ""
                if ref_step is not None:
                    reference_text = _assistant_text(
                        ref_step.get("assistant_message") or {}
                    )
                if not reference_text:
                    try:
                        reference_text = str(reference_result[turn_idx][count])
                    except Exception:
                        reference_text = ""
                is_action_drift = (
                    decode_error is not None
                    or not action_matches(decoded_prediction, reference_action or [])
                )
                is_response_drift = bool(
                    ref_step is not None
                    and not reference_has_action
                    and " ".join((text or "").split())
                    != " ".join((reference_text or "").split())
                )
                use_oracle = bool(
                    (is_action_drift or is_response_drift)
                    and self._can_correct(corrections_used)
                    and ref_step is not None
                )
                if self.oracle_mode == "c2kv4_oracle_action_mismatch":
                    use_oracle = bool(
                        decode_error is None
                        and not is_empty_execute_response(decoded_prediction)
                        and is_action_drift
                        and self._can_correct(corrections_used)
                        and ref_step is not None
                    )
                elif self.oracle_mode == "c2kv4_oracle_correct_all_strict":
                    use_oracle = bool(
                        self._can_correct(corrections_used)
                        and ref_step is not None
                    )

                state_after_step = _state_log(involved_instances)
                if (decode_error is not None or is_empty_execute_response(decoded_prediction)) and not use_oracle:
                    current_turn_response.append(text)
                    current_turn_inputs.append(usage["prompt_tokens"])
                    current_turn_outputs.append(usage["completion_tokens"])
                    current_turn_latency.append(elapsed)
                    roundtrip = serialization_roundtrip(self.decoder, text, [])
                    step_record = build_step_record(
                        sample_id=test_entry_id,
                        turn_idx=turn_idx,
                        step_idx=count,
                        global_step=ref_index,
                        candidate_raw_text=text,
                        candidate_action=decoded_prediction,
                        candidate_status=candidate.status,
                        reference_step=ref_step,
                        alignment_status=alignment_status,
                        executed_action=[],
                        state=state_after_step,
                        decode_error=decode_error,
                        empty_response=candidate.empty_response,
                        candidate_assistant_message=assistant_history,
                        executed_assistant_message=assistant_history,
                        execution_results=[],
                        history_execution_results=[],
                        oracle_corrected=False,
                        oracle_suppressed_extra_action=False,
                        response_matches_reference=(
                            None if ref_step is None else not is_response_drift
                        ),
                        candidate_response_matches_reference=(
                            None if ref_step is None else not is_response_drift
                        ),
                        roundtrip=roundtrip,
                        extra={
                            "terminal_failure": (
                                "decode_error"
                                if decode_error is not None
                                else "empty_execute_response"
                            )
                        },
                    )
                    step_record["candidate_action_matches_reference"] = (
                        not is_action_drift
                    )
                    step_record["candidate_action_drift"] = is_action_drift
                    step_record["action_matches_reference"] = not is_action_drift
                    drift_steps.append(step_record)
                    mark_first_divergence(stats, step_record)
                    if roundtrip["serialization_mismatch"]:
                        stats.errors.append(
                            "serialization mismatch at "
                            f"{test_entry_id} turn={turn_idx} step={count}"
                        )
                    break

                if use_oracle:
                    corrections_used += 1
                    correction_events.append(
                        {
                            "global_step": ref_index,
                            "turn": turn_idx,
                            "step": count,
                            "decode_error": decode_error,
                            "predicted_action": decoded_prediction,
                            "reference_action": reference_action,
                            "response_drift": is_response_drift,
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
                        executed_assistant = deepcopy(
                            ref_step.get("assistant_message")
                            or {
                                "role": "assistant",
                                "content": "",
                            }
                        )
                        result_text = _assistant_text(executed_assistant)
                        if not result_text:
                            result_text = reference_text
                            executed_assistant = {
                                "role": "assistant",
                                "content": result_text,
                            }
                    messages.append(executed_assistant)
                    current_turn_response.append(result_text)
                    step_log[0] = {"role": "assistant", "content": result_text}
                    handler_log["model_response_decoded"] = executed_action
                    if decode_error is not None:
                        handler_log["content"] = (
                            "Decode failed for C2KV candidate; oracle supplied "
                            "the Full reference action."
                        )
                    step_log.append(
                        {
                            "role": "oracle",
                            "content": "Replaced divergent C2KV action with Full reference action.",
                            "candidate_text": text,
                            "candidate_assistant_message": assistant_history,
                            "candidate_action": decoded_prediction,
                            "candidate_decode_error": decode_error,
                            "candidate_response_drift": is_response_drift,
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

                execution_error = None
                if is_empty_execute_response(executed_action):
                    execution_results = []
                else:
                    try:
                        execution_results, involved_instances = execute_multi_turn_func_call(
                            executed_action,
                            initial_config,
                            involved_classes,
                            self.decoder.model_name_underline_replaced,
                            test_entry_id,
                            long_context=long_context,
                            is_evaL_run=False,
                        )
                    except Exception as exc:
                        execution_error = str(exc)
                        execution_results = []
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
                executed_assistant_message = (
                    messages[-1 - len(execution_results)]
                    if use_oracle
                    else assistant_history
                )
                executed_text = _assistant_text(executed_assistant_message)
                roundtrip = serialization_roundtrip(
                    self.decoder,
                    executed_text,
                    executed_action,
                )
                step_record = build_step_record(
                    sample_id=test_entry_id,
                    turn_idx=turn_idx,
                    step_idx=count,
                    global_step=ref_index,
                    candidate_raw_text=text,
                    candidate_action=decoded_prediction,
                    candidate_status=candidate.status,
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=executed_action,
                    state=state_after_step,
                    decode_error=decode_error,
                    empty_response=candidate.empty_response,
                    execution_error=execution_error,
                    candidate_assistant_message=assistant_history,
                    executed_assistant_message=executed_assistant_message,
                    execution_results=execution_results,
                    history_execution_results=execution_results,
                    oracle_corrected=use_oracle,
                    oracle_suppressed_extra_action=(use_oracle and not reference_has_action),
                    response_matches_reference=(
                        None
                        if ref_step is None
                        else True
                        if use_oracle
                        else not is_response_drift
                    ),
                    candidate_response_matches_reference=(
                        None if ref_step is None else not is_response_drift
                    ),
                    roundtrip=roundtrip,
                )
                step_record["candidate_action_matches_reference"] = (
                    not is_action_drift
                )
                step_record["candidate_action_drift"] = is_action_drift
                step_record["action_matches_reference"] = not is_action_drift
                step_record["executed_action_matches_reference"] = action_matches(
                    executed_action,
                    reference_action or [],
                )
                step_record["executed_action_drift"] = not step_record[
                    "executed_action_matches_reference"
                ]
                drift_steps.append(step_record)
                mark_first_divergence(stats, step_record)
                if roundtrip["serialization_mismatch"]:
                    stats.errors.append(
                        "serialization mismatch at "
                        f"{test_entry_id} turn={turn_idx} step={count}"
                    )
                if execution_error:
                    break
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
    if args.oracle_mode == "pure_full_replay":
        failures = []
        for row in details_rows:
            sample_id = row.get("id")
            metrics = row.get("c2kv_oracle_metrics") or {}
            for step in row.get("drift_steps") or []:
                if (
                    step.get("executed_action_drift")
                    or step.get("state_drift")
                    or step.get("serialization_mismatch")
                    or step.get("decode_error")
                    or step.get("execution_error")
                    or step.get("alignment_status") == "missing_reference"
                ):
                    failures.append(
                        {
                            "id": sample_id,
                            "turn": step.get("turn"),
                            "step": step.get("step"),
                            "reference_action": step.get("reference_action"),
                            "executed_action": step.get("executed_action"),
                            "reference_state": step.get("reference_state"),
                            "actual_state": step.get("state"),
                            "alignment_status": step.get("alignment_status"),
                            "serialization_mismatch": step.get(
                                "serialization_mismatch"
                            ),
                            "decode_error": step.get("decode_error"),
                            "execution_error": step.get("execution_error"),
                            "bfcl_failure_reason": metrics.get("errors"),
                        }
                    )
        _write_jsonl(
            Path(args.details_path).parent / "reference_replay_failures.jsonl",
            failures,
        )
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
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
