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

from c2kv_eval.adapters.history_step_common import (
    action_matches,
    build_step_record,
    decode_candidate,
    mark_first_divergence,
    normalize_action_text,
    normalize_state,
    reference_by_turn_step,
    reference_step_for,
    serialization_roundtrip,
    stringify_mapping_keys,
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


def _actual_cached_tokens_from_response(response: dict[str, Any]) -> int | None:
    usage = response.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    if isinstance(prompt_details, dict) and "cached_tokens" in prompt_details:
        try:
            return int(prompt_details.get("cached_tokens") or 0)
        except Exception:
            return 0

    sglext = response.get("sglext") or {}
    cached_details = sglext.get("cached_tokens_details") or {}
    if isinstance(cached_details, dict) and cached_details:
        total = 0
        seen = False
        for key in ("device", "host", "storage"):
            if key in cached_details:
                seen = True
                try:
                    total += int(cached_details.get(key) or 0)
                except Exception:
                    pass
        if seen:
            return total

    meta_info = response.get("meta_info") or {}
    if isinstance(meta_info, dict) and "cached_tokens" in meta_info:
        try:
            return int(meta_info.get("cached_tokens") or 0)
        except Exception:
            return 0
    return None


def _kv_runtime_stats_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
    metadata = response.get("metadata") or {}
    runtime = metadata.get("sglang_runtime")
    if isinstance(runtime, dict):
        return runtime
    meta_info = response.get("meta_info") or {}
    runtime = meta_info.get("kv_runtime_stats") if isinstance(meta_info, dict) else None
    return runtime if isinstance(runtime, dict) else None


def _kv_memory_report_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
    metadata = response.get("metadata") or {}
    report = metadata.get("kv_memory_report")
    if isinstance(report, dict):
        return report
    meta_info = response.get("meta_info") or {}
    report = meta_info.get("kv_memory_report") if isinstance(meta_info, dict) else None
    return report if isinstance(report, dict) else None


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
    return normalize_action_text(text)


def _stringify_mapping_keys(value: Any) -> Any:
    return stringify_mapping_keys(value)


def _normalize_state(value: Any) -> str:
    return normalize_state(value)


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
    c2kv_extract_seconds: float = 0.0
    repair_extract_seconds: float = 0.0
    tool_execution_seconds: float = 0.0
    chat_prompt_tokens: int = 0
    chat_cached_tokens: int = 0
    chat_recomputed_prompt_tokens: int = 0
    chat_cache_report_missing: int = 0
    chat_completion_tokens: int = 0
    kv_peak_resident_tokens: int = 0
    kv_runtime_report_missing: int = 0
    c2kv_extract_recomputed_tokens: int = 0
    repair_extract_recomputed_tokens: int = 0
    original_history_tokens: int = 0
    effective_history_tokens: int = 0
    canonical_full_history_tokens: int = 0
    physical_history_kv_tokens: int = 0
    c2kv_gist_tokens: int = 0
    repair_kv_tokens: int = 0
    recomputed_raw_tokens: int = 0
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
            "c2kv_extract_seconds": round(self.c2kv_extract_seconds, 4),
            "repair_extract_seconds": round(self.repair_extract_seconds, 4),
            "tool_execution_seconds": round(self.tool_execution_seconds, 4),
            "episode_e2e_observed_seconds": round(
                self.extract_seconds + self.chat_seconds + self.tool_execution_seconds,
                4,
            ),
            "chat_prompt_tokens": self.chat_prompt_tokens,
            "chat_cached_tokens": self.chat_cached_tokens,
            "chat_recomputed_prompt_tokens": self.chat_recomputed_prompt_tokens,
            "chat_cache_report_missing": self.chat_cache_report_missing,
            "chat_completion_tokens": self.chat_completion_tokens,
            "kv_peak_resident_tokens": self.kv_peak_resident_tokens,
            "kv_runtime_report_missing": self.kv_runtime_report_missing,
            "c2kv_extract_recomputed_tokens": self.c2kv_extract_recomputed_tokens,
            "repair_extract_recomputed_tokens": self.repair_extract_recomputed_tokens,
            "query_prefill_tokens": self.chat_recomputed_prompt_tokens,
            "decode_tokens": self.chat_completion_tokens,
            "total_actual_recomputed_tokens": (
                self.c2kv_extract_recomputed_tokens
                + self.repair_extract_recomputed_tokens
                + self.chat_recomputed_prompt_tokens
                + self.chat_completion_tokens
            ),
            "history_original_tokens": self.original_history_tokens,
            "history_effective_tokens": self.effective_history_tokens,
            "canonical_full_history_tokens": self.canonical_full_history_tokens,
            "physical_history_kv_tokens": self.physical_history_kv_tokens,
            "peak_physical_kv_tokens": (
                self.kv_peak_resident_tokens if self.kv_peak_resident_tokens else None
            ),
            "c2kv_gist_tokens": self.c2kv_gist_tokens,
            "repair_kv_tokens": self.repair_kv_tokens,
            "recomputed_raw_tokens": self.recomputed_raw_tokens,
            "canonical_history_kv_compression": (
                self.canonical_full_history_tokens / self.physical_history_kv_tokens
                if self.physical_history_kv_tokens
                else 1.0
            ),
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
        elapsed = time.perf_counter() - start
        stats.extract_seconds += elapsed
        stats.c2kv_extract_seconds += elapsed
        stats.extract_calls += 1
        if record.success:
            stats.extract_success += 1
            stats.c2kv_extract_recomputed_tokens += int(
                record.original_seq_len
                or len(self.tokenizer.encode(text, add_special_tokens=False))
            )
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
            "return_cached_tokens_details": True,
        }
        memory_hint = getattr(self, "_last_kv_memory_hint", None)
        if isinstance(memory_hint, dict):
            payload["c2kv_kv_memory_hint"] = memory_hint
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
        usage_prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        cached_tokens = _actual_cached_tokens_from_response(data)
        if cached_tokens is None:
            stats.chat_cache_report_missing += 1
            recomputed_prompt_tokens = 0
        else:
            cached_tokens = min(usage_prompt_tokens, max(0, int(cached_tokens)))
            stats.chat_cached_tokens += cached_tokens
            recomputed_prompt_tokens = max(usage_prompt_tokens - cached_tokens, 0)
            stats.chat_recomputed_prompt_tokens += recomputed_prompt_tokens
        runtime = _kv_runtime_stats_from_response(data)
        kv_memory_report = _kv_memory_report_from_response(data)
        if runtime is None:
            stats.kv_runtime_report_missing += 1
        else:
            try:
                stats.kv_peak_resident_tokens = max(
                    stats.kv_peak_resident_tokens,
                    int(runtime.get("kv_resident_tokens") or 0),
                )
            except Exception:
                stats.kv_runtime_report_missing += 1
        usage_completion_tokens = int(
                usage.get("completion_tokens")
                or len(self.tokenizer.encode(text, add_special_tokens=False))
        )
        stats.chat_prompt_tokens += usage_prompt_tokens
        stats.chat_completion_tokens += usage_completion_tokens
        return text, message, elapsed, {
            "prompt_tokens": usage_prompt_tokens,
            "completion_tokens": usage_completion_tokens,
            "cached_tokens": cached_tokens,
            "recomputed_prompt_tokens": recomputed_prompt_tokens,
            "kv_runtime_stats": runtime,
            "kv_memory_report": kv_memory_report,
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
        reference_map = reference_by_turn_step(reference_steps)
        reference_result = (self.reference_by_id.get(test_entry_id) or {}).get("result") or []
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

                candidate = decode_candidate(self.decoder, text)
                if candidate.decode_error:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response.",
                            "error": candidate.decode_error,
                            "model_response_decoded": candidate.action,
                        }
                    )
                else:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": candidate.action,
                        }
                    )

                ref_index = len(drift_steps)
                state_before_execution = _state_log(involved_instances)
                ref_step, alignment_status = reference_step_for(
                    reference_map,
                    reference_result,
                    turn_idx,
                    count,
                    fallback_state=state_before_execution,
                )
                reference_action = ref_step.get("decoded_action") if ref_step else None

                should_stop_after_record = False
                if self.mode == "history_c2kv4_teacher_forced":
                    if ref_step is None:
                        decoded_to_execute = []
                        assistant_for_history = assistant_history
                        execution_results_for_history = []
                        should_stop_after_record = True
                    else:
                        decoded_to_execute = list(reference_action or [])
                        assistant_for_history = deepcopy(
                            ref_step.get("assistant_message") or assistant_history
                        )
                        execution_results_for_history = list(
                            ref_step.get("execution_results") or []
                        )
                        if is_empty_execute_response(decoded_to_execute):
                            should_stop_after_record = True
                else:
                    decoded_to_execute = candidate.action
                    assistant_for_history = assistant_history
                    execution_results_for_history = None
                    if (
                        candidate.status in {"decode_error", "invalid_format", "empty_response"}
                        or is_empty_execute_response(decoded_to_execute)
                    ):
                        should_stop_after_record = True

                messages.append(deepcopy(assistant_for_history))
                execution_error = None
                if is_empty_execute_response(decoded_to_execute):
                    execution_results = []
                else:
                    try:
                        tool_start = time.perf_counter()
                        execution_results, involved_instances = execute_multi_turn_func_call(
                            decoded_to_execute,
                            initial_config,
                            involved_classes,
                            self.decoder.model_name_underline_replaced,
                            test_entry_id,
                            long_context=long_context,
                            is_evaL_run=False,
                        )
                        stats.tool_execution_seconds += time.perf_counter() - tool_start
                    except Exception as exc:
                        stats.tool_execution_seconds += time.perf_counter() - tool_start
                        execution_error = str(exc)
                        execution_results = []
                        should_stop_after_record = True
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
                executed_text = _message_text(assistant_for_history)
                if assistant_for_history.get("tool_calls"):
                    tool_call_text = _tool_calls_to_text(assistant_for_history.get("tool_calls"))
                    executed_text = (
                        (executed_text + "\n" + tool_call_text).strip()
                        if executed_text
                        else tool_call_text
                    )
                roundtrip = serialization_roundtrip(
                    self.decoder,
                    executed_text,
                    decoded_to_execute,
                )
                step_record = build_step_record(
                    sample_id=test_entry_id,
                    turn_idx=turn_idx,
                    step_idx=count,
                    global_step=ref_index,
                    candidate_raw_text=text,
                    candidate_action=candidate.action,
                    candidate_status=candidate.status,
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=decoded_to_execute,
                    state=state_after_step,
                    decode_error=candidate.decode_error,
                    empty_response=candidate.empty_response,
                    execution_error=execution_error,
                    candidate_assistant_message=assistant_history,
                    executed_assistant_message=assistant_for_history,
                    execution_results=execution_results,
                    history_execution_results=history_execution_results,
                    roundtrip=roundtrip,
                    extra=(
                        {
                            "repair_build_info": deepcopy(
                                getattr(self, "_last_repair_build_info", {})
                            )
                        }
                        if getattr(self, "_last_repair_build_info", None)
                        else None
                    ),
                )
                if usage.get("kv_memory_report") is not None:
                    step_record["kv_memory_report"] = usage.get("kv_memory_report")
                if usage.get("kv_runtime_stats") is not None:
                    step_record["kv_runtime_stats"] = usage.get("kv_runtime_stats")
                mark_first_divergence(stats, step_record)
                if alignment_status == "missing_reference":
                    stats.errors.append(
                        f"missing reference action at turn={turn_idx}, step={count}, "
                        f"candidate_global_step={ref_index}"
                    )
                drift_steps.append(step_record)
                if roundtrip["serialization_mismatch"]:
                    stats.errors.append(
                        f"serialization mismatch at turn={turn_idx}, step={count}, "
                        f"candidate_global_step={ref_index}"
                    )
                if should_stop_after_record:
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
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
