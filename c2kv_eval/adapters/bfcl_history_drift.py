from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.executable_backend_config import (
    OMIT_STATE_INFO_CLASSES,
    STATELESS_CLASSES,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import (
    load_dataset_entry,
    make_json_serializable,
    sort_file_content_by_id,
    sort_key,
)


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FC"
DEFAULT_TOKENIZER_PATH = "/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507"
HTTP = requests.Session()
HTTP.trust_env = False


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _tool_payload(functions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return convert_to_tool(
        list(functions),
        GORILLA_TO_OPENAPI,
        ModelStyle.OPENAI_COMPLETIONS,
    )


def _post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    response = HTTP.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise requests.HTTPError(
            f"{response.status_code} error for {path}: {response.text[:1000]}",
            response=response,
        )
    return response.json()


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _tool_calls_to_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    chunks = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") or call.get("name")
        arguments = function.get("arguments") or call.get("arguments") or {}
        if not name:
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        chunks.append(
            "<tool_call>\n"
            + _json_dumps({"name": name, "arguments": arguments})
            + "\n</tool_call>"
        )
    return "\n".join(chunks)


def _assistant_history_message(text: str, tool_calls: Any) -> dict[str, Any]:
    if isinstance(tool_calls, list) and tool_calls:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": deepcopy(tool_calls),
        }
    return {"role": "assistant", "content": text}


def _render_history_message(message: dict[str, Any]) -> str:
    role = message.get("role", "")
    if role == "assistant" and message.get("tool_calls"):
        body = _tool_calls_to_text(message.get("tool_calls"))
    elif role == "tool":
        body = "<tool_response>\n" + _message_text(message) + "\n</tool_response>"
    else:
        body = _message_text(message)
    return f"<history_message role={role}>\n{body}\n</history_message>"


def _render_history_unit(unit: Sequence[dict[str, Any]]) -> str:
    return "Completed history unit:\n" + "\n".join(
        _render_history_message(message) for message in unit
    )


def _is_real_user_query(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = _message_text(message).strip()
    return not (content.startswith("<tool_response>") and content.endswith("</tool_response>"))


def _latest_user_query_index(messages: Sequence[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if _is_real_user_query(messages[index]):
            return index
    return len(messages)


def _history_units(messages: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    units: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.get("role") == "assistant":
            unit = [message]
            index += 1
            while index < len(messages) and messages[index].get("role") == "tool":
                unit.append(messages[index])
                index += 1
            units.append(unit)
            continue
        units.append([message])
        index += 1
    return units


def _token_count(tokenizer: Any, messages: Iterable[dict[str, Any]]) -> int:
    message_list = list(messages)
    try:
        return len(
            tokenizer.apply_chat_template(
                message_list,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
    except Exception:
        return len(tokenizer.encode(_json_dumps(message_list), add_special_tokens=False))


def _state_log(involved_instances: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for class_name, class_instance in involved_instances.items():
        if class_name in STATELESS_CLASSES or class_name in OMIT_STATE_INFO_CLASSES:
            continue
        class_instance = deepcopy(class_instance)
        rows.append(
            {
                "role": "state_info",
                "class_name": class_name,
                "content": {
                    key: value
                    for key, value in vars(class_instance).items()
                    if not key.startswith("_")
                },
            }
        )
    return rows


def _normalize_action_text(text: Any) -> str:
    if not isinstance(text, str):
        return _json_dumps(text)
    return re.sub(r"\s+", "", text)


def _normalize_state(value: Any) -> str:
    return json.dumps(
        make_json_serializable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


@dataclass
class ExtractRecord:
    success: bool
    key_hash: str | None = None
    gist_len: int | None = None
    original_seq_len: int | None = None
    error: str | None = None


@dataclass
class DriftStats:
    sample_id: str
    mode: str
    ratio: int
    chat_calls: int = 0
    chat_seconds: float = 0.0
    extract_calls: int = 0
    extract_success: int = 0
    extract_seconds: float = 0.0
    original_history_tokens: int = 0
    effective_history_tokens: int = 0
    first_action_divergence: dict[str, int] | None = None
    first_state_divergence: dict[str, int] | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "mode": self.mode,
            "ratio": self.ratio,
            "chat_calls": self.chat_calls,
            "chat_seconds": round(self.chat_seconds, 4),
            "avg_chat_seconds": (
                round(self.chat_seconds / self.chat_calls, 4) if self.chat_calls else None
            ),
            "extract_calls": self.extract_calls,
            "extract_success": self.extract_success,
            "extract_success_rate": (
                self.extract_success / self.extract_calls if self.extract_calls else None
            ),
            "extract_seconds": round(self.extract_seconds, 4),
            "history_original_tokens": self.original_history_tokens,
            "history_effective_tokens": self.effective_history_tokens,
            "history_compression_ratio": (
                self.original_history_tokens / self.effective_history_tokens
                if self.effective_history_tokens
                else 1.0
            ),
            "first_action_divergence": self.first_action_divergence,
            "first_state_divergence": self.first_state_divergence,
            "errors": self.errors,
        }


class HistoryDriftRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = args.mode
        self.base_url = args.base_url.rstrip("/")
        self.ratio = args.ratio
        self.recent_full_units = args.recent_full_units
        self.timeout = args.timeout
        self.temperature = args.temperature
        self.max_completion_tokens = args.max_completion_tokens
        self.model = args.served_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        config = MODEL_CONFIG_MAPPING[args.model]
        self.decoder = QwenFCHandler(
            model_name=config.model_name,
            temperature=args.temperature,
            registry_name=args.model,
            is_fc_model=config.is_fc_model,
        )
        self.decoder.model_name_underline_replaced = (
            config.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
        )
        reference_rows = (
            _load_jsonl(Path(args.reference_details_path))
            if args.reference_details_path
            else []
        )
        self.reference_by_id = {row["id"]: row for row in reference_rows}

    def _extract_history_unit(self, text: str, stats: DriftStats) -> ExtractRecord:
        start = time.perf_counter()
        try:
            result = _post_json(
                self.base_url,
                "/v1/c2kv/extract",
                {
                    "text": text,
                    "compression_ratio": self.ratio,
                    "role": "user",
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                self.timeout,
            )
            record = ExtractRecord(
                success=bool(result.get("success") and result.get("key_hash")),
                key_hash=result.get("key_hash"),
                gist_len=result.get("gist_len"),
                original_seq_len=result.get("original_seq_len"),
                error=result.get("error"),
            )
        except Exception as exc:
            record = ExtractRecord(success=False, error=str(exc))
            stats.errors.append(f"extract: {exc}")
        stats.extract_seconds += time.perf_counter() - start
        stats.extract_calls += 1
        if record.success:
            stats.extract_success += 1
        return record

    def _build_request_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> list[dict[str, Any]]:
        latest_query_index = _latest_user_query_index(history_messages)
        completed = list(history_messages[:latest_query_index])
        current = deepcopy(list(history_messages[latest_query_index:]))
        if self.mode == "history_full_closed_loop":
            full_tokens = _token_count(self.tokenizer, completed)
            stats.original_history_tokens += full_tokens
            stats.effective_history_tokens += full_tokens
            return deepcopy(list(history_messages))

        units = _history_units(completed)
        keep_full_from = len(units)
        if self.mode == "history_recent2_full_rest_c2kv4":
            keep_full_from = max(0, len(units) - self.recent_full_units)

        messages: list[dict[str, Any]] = []
        for unit_index, unit in enumerate(units):
            if self.mode == "history_recent2_full_rest_c2kv4" and unit_index >= keep_full_from:
                full_tokens = _token_count(self.tokenizer, unit)
                stats.original_history_tokens += full_tokens
                stats.effective_history_tokens += full_tokens
                messages.extend(deepcopy(unit))
                continue
            text = _render_history_unit(unit)
            full_tokens = _token_count(self.tokenizer, [{"role": "user", "content": text}])
            record = self._extract_history_unit(text, stats)
            stats.original_history_tokens += int(record.original_seq_len or full_tokens)
            if record.success and record.key_hash:
                stats.effective_history_tokens += int(record.gist_len or record.original_seq_len or full_tokens)
                messages.append(
                    {"role": "user", "content": text, "c2kv_key_hash": record.key_hash}
                )
            else:
                stats.effective_history_tokens += full_tokens
                messages.append({"role": "user", "content": text})
        messages.extend(current)
        return messages

    def _query(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> tuple[str, dict[str, Any], float, dict[str, Any]]:
        prompt_tokens = _token_count(self.tokenizer, messages)
        max_tokens = max(1, self.max_completion_tokens)
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": self.temperature,
            "max_completion_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.perf_counter()
        data = _post_json(self.base_url, "/v1/chat/completions", payload, self.timeout)
        elapsed = time.perf_counter() - start
        message = data.get("choices", [{}])[0].get("message", {}) or {}
        text = message.get("content") or ""
        tool_calls_text = _tool_calls_to_text(message.get("tool_calls"))
        if tool_calls_text:
            text = (text + "\n" + tool_calls_text).strip() if text else tool_calls_text
        usage = data.get("usage") or {}
        stats.chat_calls += 1
        stats.chat_seconds += elapsed
        return text, message, elapsed, {
            "prompt_tokens": int(usage.get("prompt_tokens") or prompt_tokens),
            "completion_tokens": int(
                usage.get("completion_tokens")
                or len(self.tokenizer.encode(text, add_special_tokens=False))
            ),
        }

    def _decode(self, text: str) -> list[str]:
        return self.decoder.decode_execute(text, has_tool_call_tag=False)

    def _reference_steps(self, sample_id: str) -> list[dict[str, Any]]:
        row = self.reference_by_id.get(sample_id) or {}
        steps = row.get("drift_steps")
        return steps if isinstance(steps, list) else []

    def _compare_reference_action(
        self,
        *,
        stats: DriftStats,
        reference_steps: Sequence[dict[str, Any]],
        ref_index: int,
        turn_idx: int,
        step_idx: int,
        decoded_prediction: list[str],
    ) -> dict[str, Any] | None:
        if self.mode == "history_full_closed_loop":
            return None
        if ref_index >= len(reference_steps):
            if stats.first_action_divergence is None:
                stats.first_action_divergence = {
                    "turn": turn_idx,
                    "step": step_idx,
                    "reason": "missing_reference_action",
                }
            stats.errors.append(
                f"missing reference action at global_step={ref_index}, "
                f"turn={turn_idx}, step={step_idx}"
            )
            return None
        ref_step = reference_steps[ref_index]
        reference_action = ref_step.get("decoded_action") or []
        if (
            stats.first_action_divergence is None
            and _normalize_action_text(decoded_prediction)
            != _normalize_action_text(reference_action)
        ):
            stats.first_action_divergence = {
                "turn": turn_idx,
                "step": step_idx,
                "global_step": ref_index,
            }
        return ref_step

    def _compare_reference_state(
        self,
        *,
        stats: DriftStats,
        ref_step: dict[str, Any] | None,
        ref_index: int,
        turn_idx: int,
        step_idx: int,
        state_after_step: list[dict[str, Any]],
    ) -> None:
        if self.mode == "history_full_closed_loop":
            return
        if not ref_step:
            if stats.first_state_divergence is None:
                stats.first_state_divergence = {
                    "turn": turn_idx,
                    "step": step_idx,
                    "global_step": ref_index,
                    "reason": "missing_reference_state",
                }
            return
        reference_state = ref_step.get("state")
        if (
            stats.first_state_divergence is None
            and _normalize_state(state_after_step) != _normalize_state(reference_state)
        ):
            stats.first_state_divergence = {
                "turn": turn_idx,
                "step": step_idx,
                "global_step": ref_index,
            }

    def run_sample(self, test_case: dict[str, Any]) -> dict[str, Any]:
        stats = DriftStats(test_case["id"], self.mode, self.ratio)
        try:
            result, metadata = self._run_sample_impl(test_case, stats)
        except Exception as exc:
            result = f"Error during inference: {exc}"
            metadata = {"traceback": traceback.format_exc()}
            stats.errors.append(str(exc))
        metadata["c2kv_drift_metrics"] = stats.as_dict()
        return {"id": test_case["id"], "result": result, **metadata}

    def _run_sample_impl(
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
        reference_steps = self._reference_steps(test_entry_id)
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
                current_turn_response.append(text)
                current_turn_inputs.append(usage["prompt_tokens"])
                current_turn_outputs.append(usage["completion_tokens"])
                current_turn_latency.append(elapsed)

                step_log = [
                    {"role": "assistant", "content": text},
                ]
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
                ref_step = self._compare_reference_action(
                    stats=stats,
                    reference_steps=reference_steps,
                    ref_index=ref_index,
                    turn_idx=turn_idx,
                    step_idx=count,
                    decoded_prediction=decoded_prediction,
                )
                if self.mode == "history_c2kv4_teacher_forced":
                    if ref_step is None:
                        break
                    decoded_to_execute = ref_step.get("decoded_action") or []
                    messages.append(deepcopy(ref_step.get("assistant_message") or assistant_history))
                    execution_results_for_history = list(
                        ref_step.get("execution_results") or []
                    )
                else:
                    decoded_to_execute = decoded_prediction
                    messages.append(assistant_history)
                    execution_results_for_history = None

                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_to_execute,
                    initial_config,
                    involved_classes,
                    self.decoder.model_name_underline_replaced,
                    test_entry_id,
                    long_context=long_context,
                    is_evaL_run=False,
                )
                history_execution_results = (
                    execution_results_for_history
                    if execution_results_for_history is not None
                    else execution_results
                )
                for idx, execution_result in enumerate(history_execution_results):
                    messages.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                            "tool_call_id": f"call_{turn_idx}_{count}_{idx}",
                        }
                    )
                    step_log.append({"role": "tool", "content": execution_result})

                state_after_step = _state_log(involved_instances)
                self._compare_reference_state(
                    stats=stats,
                    ref_step=ref_step,
                    ref_index=ref_index,
                    turn_idx=turn_idx,
                    step_idx=count,
                    state_after_step=state_after_step,
                )
                drift_steps.append(
                    {
                        "turn": turn_idx,
                        "step": count,
                        "assistant_message": assistant_history,
                        "decoded_action": decoded_prediction,
                        "executed_action": decoded_to_execute,
                        "execution_results": execution_results,
                        "history_execution_results": history_execution_results,
                        "state": state_after_step,
                        "reference_action": (
                            ref_step.get("decoded_action") if ref_step else None
                        ),
                        "reference_state": (
                            ref_step.get("state") if ref_step else None
                        ),
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
        }
        return all_model_response, metadata


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    runner = HistoryDriftRunner(args)
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
    metric_rows = []
    for test_case in tqdm(entries, desc=f"{args.mode}:{args.category}", dynamic_ncols=True):
        row = runner.run_sample(deepcopy(test_case))
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metric_rows.append(row.get("c2kv_drift_metrics", {}))

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metric_rows)
    summary = {
        "mode": args.mode,
        "category": args.category,
        "num_examples": len(details_rows),
        "errors": sum(1 for row in details_rows if str(row.get("result", "")).startswith("Error during inference")),
        "chat_calls": sum(int(row.get("chat_calls") or 0) for row in metric_rows),
        "extract_calls": sum(int(row.get("extract_calls") or 0) for row in metric_rows),
        "extract_success": sum(int(row.get("extract_success") or 0) for row in metric_rows),
    }
    summary["extract_success_rate"] = (
        summary["extract_success"] / summary["extract_calls"]
        if summary["extract_calls"]
        else None
    )
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "history_full_closed_loop",
            "history_c2kv4_teacher_forced",
            "history_c2kv4_closed_loop",
            "history_recent2_full_rest_c2kv4",
        ],
        required=True,
    )
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--ids-path", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--reference-details-path", default="")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--recent-full-units", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
