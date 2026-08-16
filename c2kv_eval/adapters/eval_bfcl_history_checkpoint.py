from __future__ import annotations

import argparse
import json
import math
import re
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

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

from c2kv_eval.adapters.bfcl_history_branch import _assistant_text
from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    HistoryDriftRunner,
    _assistant_history_message,
    _history_units,
    _latest_user_query_index,
    _normalize_action_text,
    _normalize_state,
    _post_json,
    _render_history_unit,
    _state_log,
    _token_count,
    _tool_payload,
)


CHECKPOINT_MODES = {
    "current_step",
    "recent2",
    "since_checkpoint",
    "full_history",
}

KV_VERIFIERS = {"instant_kv", "cumulative_kv", "kv_divergence"}


def _extract_numeric_vector(value: Any, *, max_items: int = 8192) -> list[float]:
    values: list[float] = []

    def visit(obj: Any) -> None:
        if len(values) >= max_items:
            return
        if obj is None:
            return
        if isinstance(obj, bool):
            return
        if isinstance(obj, (int, float)):
            number = float(obj)
            if math.isfinite(number):
                values.append(number)
            return
        if isinstance(obj, dict):
            for key in (
                "hidden_states",
                "last_hidden_state",
                "readout",
                "values",
                "data",
            ):
                if key in obj:
                    visit(obj[key])
                    return
            for item in obj.values():
                visit(item)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)

    visit(value)
    return values


def _cosine_distance(left: list[float], right: list[float]) -> float | None:
    size = min(len(left), len(right))
    if size == 0:
        return None
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for index in range(size):
        a = float(left[index])
        b = float(right[index])
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    cosine = dot / math.sqrt(left_norm * right_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def _distribution_from_logprobs(logprobs: Any) -> dict[str, float]:
    if not logprobs:
        return {}
    if isinstance(logprobs, dict):
        if isinstance(logprobs.get("content"), list) and logprobs["content"]:
            first = logprobs["content"][0]
            top = first.get("top_logprobs") if isinstance(first, dict) else None
            if isinstance(top, list):
                out = {}
                for item in top:
                    if not isinstance(item, dict):
                        continue
                    token = item.get("token")
                    value = item.get("logprob")
                    if token is not None and isinstance(value, (int, float)):
                        out[str(token)] = float(value)
                return out
        for key in ("top_logprobs", "output_top_logprobs"):
            value = logprobs.get(key)
            if isinstance(value, list):
                return _distribution_from_logprobs(value)
    if isinstance(logprobs, list):
        if not logprobs:
            return {}
        first = logprobs[0]
        if isinstance(first, dict):
            if all(isinstance(v, (int, float)) for v in first.values()):
                return {str(k): float(v) for k, v in first.items()}
            return _distribution_from_logprobs(first)
        if (
            isinstance(first, (list, tuple))
            and len(first) >= 2
            and isinstance(first[1], (int, float))
        ):
            return {str(item[0]): float(item[1]) for item in logprobs if len(item) >= 2}
    return {}


def _entropy_from_log_probs(log_probs: dict[str, float]) -> float | None:
    if not log_probs:
        return None
    probs = [math.exp(value) for value in log_probs.values() if math.isfinite(value)]
    total = sum(probs)
    if total <= 0.0:
        return None
    return -sum((p / total) * math.log(max(p / total, 1e-30)) for p in probs)


def _top1_top2_margin(log_probs: dict[str, float]) -> float | None:
    if len(log_probs) < 2:
        return None
    values = sorted(log_probs.values(), reverse=True)
    return values[0] - values[1]


def _kl_from_log_probs(
    full_log_probs: dict[str, float],
    c2kv_log_probs: dict[str, float],
) -> float | None:
    if not full_log_probs or not c2kv_log_probs:
        return None
    keys = set(full_log_probs) | set(c2kv_log_probs)
    floor = -30.0
    full_weights = {key: math.exp(full_log_probs.get(key, floor)) for key in keys}
    c2kv_weights = {key: math.exp(c2kv_log_probs.get(key, floor)) for key in keys}
    full_total = sum(full_weights.values())
    c2kv_total = sum(c2kv_weights.values())
    if full_total <= 0.0 or c2kv_total <= 0.0:
        return None
    kl = 0.0
    for key in keys:
        p = full_weights[key] / full_total
        q = max(c2kv_weights[key] / c2kv_total, 1e-30)
        if p > 0.0:
            kl += p * math.log(p / q)
    return kl


def _instance_key(model_name: str, test_entry_id: str, class_name: str) -> str:
    return re.sub(r"[-./:]", "_", f"{model_name}_{test_entry_id}_{class_name}_instance")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


class HistoryCheckpointRunner(HistoryDriftRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        drift_args = deepcopy(args)
        drift_args.mode = "history_c2kv4_closed_loop"
        super().__init__(drift_args)
        self.checkpoint_interval = args.checkpoint_interval
        self.verifier = args.verifier
        self.verify_threshold = args.verify_threshold
        self.recovery_mode = args.recovery_mode
        self.verify_layers = args.verify_layers
        self.online_verify = bool(args.online_verify)
        self._cumulative_divergence = 0.0
        if self.verifier in {"instant_kv", "cumulative_kv"}:
            self.verifier = "kv_divergence"
        if self.checkpoint_interval != 1:
            raise NotImplementedError(
                "The first checkpoint framework version supports oracle + "
                "checkpoint_interval=1. The CLI keeps 1/2/4 for the planned "
                "speculative multi-step extension."
            )

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

    def _reference_maps(self, sample_id: str) -> tuple[dict[tuple[int, int], dict[str, Any]], list[Any]]:
        row = self.reference_by_id.get(sample_id) or {}
        steps = row.get("drift_steps") or []
        by_turn_step = {
            (int(step.get("turn")), int(step.get("step"))): step
            for step in steps
            if isinstance(step, dict)
            and step.get("turn") is not None
            and step.get("step") is not None
        }
        return by_turn_step, row.get("result") or []

    def _build_recovery_messages(
        self,
        checkpoint_messages: list[dict[str, Any]],
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
        if self.recovery_mode in {"current_step", "since_checkpoint", "full_history"}:
            messages = deepcopy(checkpoint_messages)
            return (
                messages,
                _token_count(self.tokenizer, messages),
                {
                    "recovery_prompt_mode": self.recovery_mode,
                    "c2kv_history_units": 0,
                    "full_history_units": len(_history_units(messages)),
                    "current_messages": 0,
                },
            )
        if self.recovery_mode == "recent2":
            latest_query_index = _latest_user_query_index(checkpoint_messages)
            completed = list(checkpoint_messages[:latest_query_index])
            current = deepcopy(list(checkpoint_messages[latest_query_index:]))
            units = _history_units(completed)
            keep_full_from = max(0, len(units) - self.recent_full_units)
            messages: list[dict[str, Any]] = []
            c2kv_units = 0
            full_units = 0
            for unit_index, unit in enumerate(units):
                if unit_index >= keep_full_from:
                    full_tokens = _token_count(self.tokenizer, unit)
                    stats.original_history_tokens += full_tokens
                    stats.effective_history_tokens += full_tokens
                    messages.extend(deepcopy(unit))
                    full_units += 1
                    continue
                text = _render_history_unit(unit)
                full_tokens = _token_count(
                    self.tokenizer,
                    [{"role": "user", "content": text}],
                )
                record = self._extract_history_unit(text, stats)
                stats.original_history_tokens += int(record.original_seq_len or full_tokens)
                if record.success and record.key_hash:
                    stats.effective_history_tokens += int(
                        record.gist_len or record.original_seq_len or full_tokens
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": text,
                            "c2kv_key_hash": record.key_hash,
                        }
                    )
                    c2kv_units += 1
                else:
                    stats.effective_history_tokens += full_tokens
                    messages.append({"role": "user", "content": text})
                    full_units += 1
            messages.extend(current)
            return (
                messages,
                _token_count(self.tokenizer, messages),
                {
                    "recovery_prompt_mode": "recent2",
                    "c2kv_history_units": c2kv_units,
                    "full_history_units": full_units,
                    "current_messages": len(current),
                    "completed_units": len(units),
                    "recent_full_units": self.recent_full_units,
                },
            )
        raise ValueError(f"Unknown recovery mode: {self.recovery_mode}")

    def _verify_oracle(
        self,
        *,
        decoded_action: list[str],
        state_after_step: list[dict[str, Any]],
        ref_step: dict[str, Any] | None,
    ) -> dict[str, Any]:
        reference_action = ref_step.get("decoded_action") if ref_step else None
        candidate_action_matches = (
            _normalize_action_text(decoded_action)
            == _normalize_action_text(reference_action or [])
        )
        if ref_step is None:
            state_matches = is_empty_execute_response(decoded_action)
        else:
            state_matches = (
                _normalize_state(state_after_step)
                == _normalize_state(ref_step.get("state"))
            )
        harmful = not candidate_action_matches
        return {
            "kv_divergence": None,
            "cumulative_divergence": None,
            "candidate_action_matches_reference": candidate_action_matches,
            "state_matches_reference": state_matches,
            "verify_failed": harmful,
        }

    def _query_with_raw(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stats: DriftStats,
        *,
        max_completion_tokens: int | None = None,
        readout_probe: bool = False,
    ) -> tuple[str, dict[str, Any], float, dict[str, Any], dict[str, Any]]:
        prompt_tokens = _token_count(self.tokenizer, messages)
        max_tokens = max(1, max_completion_tokens or self.max_completion_tokens)
        payload = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            "temperature": self.temperature,
            "max_completion_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if readout_probe:
            payload.update(
                {
                    "logprobs": True,
                    "top_logprobs": 20,
                    "return_hidden_states": True,
                }
            )
        start = time.perf_counter()
        data = _post_json(self.base_url, "/v1/chat/completions", payload, self.timeout)
        elapsed = time.perf_counter() - start
        choice = data.get("choices", [{}])[0] or {}
        message = choice.get("message", {}) or {}
        text = message.get("content") or ""
        tool_calls_text = self._tool_calls_to_text_for_query(message.get("tool_calls"))
        if tool_calls_text:
            text = (text + "\n" + tool_calls_text).strip() if text else tool_calls_text
        usage = data.get("usage") or {}
        stats.chat_calls += 1
        stats.chat_seconds += elapsed
        parsed_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens") or prompt_tokens),
            "completion_tokens": int(
                usage.get("completion_tokens")
                or len(self.tokenizer.encode(text, add_special_tokens=False))
            ),
        }
        return text, message, elapsed, parsed_usage, data

    @staticmethod
    def _tool_calls_to_text_for_query(tool_calls: Any) -> str:
        from c2kv_eval.adapters.bfcl_history_drift import _tool_calls_to_text

        return _tool_calls_to_text(tool_calls)

    def _readout_payload(
        self,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        choice = (raw.get("choices") or [{}])[0] or {}
        hidden = choice.get("hidden_states")
        source = "choice.hidden_states"
        if hidden is None:
            hidden = raw.get("hidden_states")
            source = "response.hidden_states"
        vector = _extract_numeric_vector(hidden)
        log_probs = _distribution_from_logprobs(choice.get("logprobs"))
        return {
            "vector": vector,
            "log_probs": log_probs,
            "readout_available": bool(vector),
            "readout_source": source if vector else None,
            "readout_dim": len(vector),
            "entropy": _entropy_from_log_probs(log_probs),
            "top1_top2_margin": _top1_top2_margin(log_probs),
        }

    def _verify_kv_divergence(
        self,
        *,
        full_messages: Sequence[dict[str, Any]],
        c2kv_messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> dict[str, Any]:
        _, _, full_elapsed, full_usage, full_raw = self._query_with_raw(
            full_messages,
            tools,
            stats,
            max_completion_tokens=2,
            readout_probe=True,
        )
        _, _, c2kv_elapsed, c2kv_usage, c2kv_raw = self._query_with_raw(
            c2kv_messages,
            tools,
            stats,
            max_completion_tokens=2,
            readout_probe=True,
        )
        full_readout = self._readout_payload(full_raw)
        c2kv_readout = self._readout_payload(c2kv_raw)
        kv_divergence = _cosine_distance(
            full_readout["vector"],
            c2kv_readout["vector"],
        )
        full_log_probs = full_readout["log_probs"]
        c2kv_log_probs = c2kv_readout["log_probs"]
        logit_kl = _kl_from_log_probs(full_log_probs, c2kv_log_probs)
        if kv_divergence is not None:
            self._cumulative_divergence += kv_divergence
        score = (
            self._cumulative_divergence
            if self.verifier == "cumulative_kv"
            else kv_divergence
        )
        verify_failed = (
            self.online_verify
            and score is not None
            and score >= self.verify_threshold
        )
        return {
            "kv_divergence": kv_divergence,
            "cumulative_divergence": (
                self._cumulative_divergence if kv_divergence is not None else None
            ),
            "logit_kl": logit_kl,
            "entropy": c2kv_readout["entropy"],
            "top1_top2_margin": c2kv_readout["top1_top2_margin"],
            "readout_available": bool(
                full_readout["readout_available"]
                and c2kv_readout["readout_available"]
            ),
            "full_readout_source": full_readout["readout_source"],
            "c2kv_readout_source": c2kv_readout["readout_source"],
            "full_readout_dim": full_readout["readout_dim"],
            "c2kv_readout_dim": c2kv_readout["readout_dim"],
            "full_probe_seconds": full_elapsed,
            "c2kv_probe_seconds": c2kv_elapsed,
            "full_probe_prompt_tokens": full_usage["prompt_tokens"],
            "c2kv_probe_prompt_tokens": c2kv_usage["prompt_tokens"],
            "verify_failed": verify_failed,
        }

    def run_sample_checkpoint(self, test_case: dict[str, Any]) -> dict[str, Any]:
        stats = DriftStats(test_case["id"], "history_checkpoint", self.ratio)
        self._cumulative_divergence = 0.0
        try:
            result, metadata = self._run_sample_checkpoint_impl(test_case, stats)
        except Exception as exc:
            result = f"Error during inference: {exc}"
            metadata = {"traceback": traceback.format_exc()}
            stats.errors.append(str(exc))
        metadata["c2kv_checkpoint_metrics"] = {
            **stats.as_dict(),
            "checkpoint_interval": self.checkpoint_interval,
            "verifier": self.verifier,
            "verify_threshold": self.verify_threshold,
            "verify_layers": self.verify_layers,
            "online_verify": self.online_verify,
            "recovery_mode": self.recovery_mode,
            "verify_count": metadata.get("verify_count", 0),
            "refresh_count": metadata.get("refresh_count", 0),
            "regenerated_steps": metadata.get("regenerated_steps", 0),
            "full_regenerated_tokens": metadata.get("full_regenerated_tokens", 0),
        }
        return {"id": test_case["id"], "result": result, **metadata}

    def _run_sample_checkpoint_impl(
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
        reference_by_turn_step, reference_result = self._reference_maps(test_entry_id)

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
        checkpoint_steps: list[dict[str, Any]] = []
        drift_steps: list[dict[str, Any]] = []
        verify_count = 0
        refresh_count = 0
        regenerated_steps = 0
        full_regenerated_tokens = 0
        checkpoint_id = 0
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
                checkpoint_id += 1
                checkpoint_messages = deepcopy(messages)
                checkpoint_instances = deepcopy(involved_instances)
                checkpoint_turn_response = deepcopy(current_turn_response)
                checkpoint_turn_inputs = deepcopy(current_turn_inputs)
                checkpoint_turn_outputs = deepcopy(current_turn_outputs)
                checkpoint_turn_latency = deepcopy(current_turn_latency)
                checkpoint_turn_log = deepcopy(turn_log)

                request_messages = self._build_request_messages(messages, stats)
                (
                    candidate_text,
                    candidate_message,
                    candidate_elapsed,
                    candidate_usage,
                    candidate_raw,
                ) = self._query_with_raw(
                    request_messages,
                    tools,
                    stats,
                )
                candidate_assistant = _assistant_history_message(
                    candidate_text,
                    candidate_message.get("tool_calls"),
                )
                try:
                    candidate_action = self._decode(candidate_text)
                except Exception:
                    candidate_action = []

                messages.append(candidate_assistant)
                if is_empty_execute_response(candidate_action):
                    execution_results = []
                else:
                    execution_results, involved_instances = execute_multi_turn_func_call(
                        candidate_action,
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

                state_after_candidate = _state_log(involved_instances)
                ref_step = reference_by_turn_step.get((turn_idx, count))
                if self.verifier == "oracle":
                    verify = self._verify_oracle(
                        decoded_action=candidate_action,
                        state_after_step=state_after_candidate,
                        ref_step=ref_step,
                    )
                    verify.update(
                        {
                            "logit_kl": None,
                            "entropy": None,
                            "top1_top2_margin": None,
                            "readout_available": False,
                            "full_readout_source": None,
                            "c2kv_readout_source": None,
                            "full_readout_dim": 0,
                            "c2kv_readout_dim": 0,
                            "full_probe_seconds": 0.0,
                            "full_probe_prompt_tokens": 0,
                            "c2kv_probe_seconds": 0.0,
                            "c2kv_probe_prompt_tokens": 0,
                        }
                    )
                else:
                    kv_verify = self._verify_kv_divergence(
                        full_messages=checkpoint_messages,
                        c2kv_messages=request_messages,
                        tools=tools,
                        stats=stats,
                    )
                    reference_action = ref_step.get("decoded_action") if ref_step else None
                    candidate_action_matches = (
                        _normalize_action_text(candidate_action)
                        == _normalize_action_text(reference_action or [])
                    )
                    if ref_step is None:
                        state_matches = is_empty_execute_response(candidate_action)
                    else:
                        state_matches = (
                            _normalize_state(state_after_candidate)
                            == _normalize_state(ref_step.get("state"))
                        )
                    verify = {
                        **kv_verify,
                        "candidate_action_matches_reference": candidate_action_matches,
                        "state_matches_reference": state_matches,
                    }
                verify_count += 1
                refresh_triggered = bool(verify["verify_failed"])

                if refresh_triggered:
                    refresh_count += 1
                    regenerated_steps += 1
                    self._restore_instances(test_entry_id, checkpoint_instances)
                    messages = deepcopy(checkpoint_messages)
                    involved_instances = deepcopy(checkpoint_instances)
                    current_turn_response = deepcopy(checkpoint_turn_response)
                    current_turn_inputs = deepcopy(checkpoint_turn_inputs)
                    current_turn_outputs = deepcopy(checkpoint_turn_outputs)
                    current_turn_latency = deepcopy(checkpoint_turn_latency)
                    turn_log = deepcopy(checkpoint_turn_log)

                    (
                        recovery_messages,
                        recovery_prompt_tokens,
                        recovery_debug,
                    ) = self._build_recovery_messages(
                        messages,
                        stats,
                    )
                    full_regenerated_tokens += recovery_prompt_tokens
                    text, response_message, elapsed, usage = self._query(
                        recovery_messages,
                        tools,
                        stats,
                    )
                    assistant_history = _assistant_history_message(
                        text,
                        response_message.get("tool_calls"),
                    )
                    try:
                        executed_action = self._decode(text)
                    except Exception:
                        executed_action = []
                    messages.append(assistant_history)
                    executed_text = text
                    executed_elapsed = elapsed
                    executed_usage = usage
                    candidate_debug = {
                        "candidate_text": candidate_text,
                        "candidate_assistant_message": candidate_assistant,
                        "candidate_action": candidate_action,
                        "regenerated_text": text,
                        "regenerated_action": executed_action,
                        "regenerated_same_as_candidate": (
                            _normalize_action_text(candidate_action)
                            == _normalize_action_text(executed_action)
                        ),
                        **recovery_debug,
                    }
                else:
                    text = candidate_text
                    assistant_history = candidate_assistant
                    executed_action = candidate_action
                    executed_text = candidate_text
                    executed_elapsed = candidate_elapsed
                    executed_usage = candidate_usage
                    candidate_debug = None
                    recovery_debug = {
                        "recovery_prompt_mode": None,
                        "c2kv_history_units": 0,
                        "full_history_units": 0,
                        "current_messages": 0,
                    }

                if refresh_triggered:
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

                state_after_step = _state_log(involved_instances)
                reference_action = ref_step.get("decoded_action") if ref_step else None
                executed_action_matches = (
                    _normalize_action_text(executed_action)
                    == _normalize_action_text(reference_action or [])
                )
                if ref_step is None:
                    state_matches = is_empty_execute_response(executed_action)
                else:
                    state_matches = (
                        _normalize_state(state_after_step)
                        == _normalize_state(ref_step.get("state"))
                    )

                current_turn_response.append(executed_text)
                current_turn_inputs.append(executed_usage["prompt_tokens"])
                current_turn_outputs.append(executed_usage["completion_tokens"])
                current_turn_latency.append(executed_elapsed)
                step_log = [
                    {"role": "assistant", "content": executed_text},
                    {
                        "role": "handler_log",
                        "content": "Successfully decoded model response.",
                        "model_response_decoded": executed_action,
                    },
                ]
                if candidate_debug is not None:
                    step_log.append(
                        {
                            "role": "checkpoint_verify",
                            "content": "Rolled back speculative C2KV step and regenerated.",
                            **candidate_debug,
                            "recovery_mode": self.recovery_mode,
                            "checkpoint_id": checkpoint_id,
                        }
                    )
                for execution_result in execution_results:
                    step_log.append({"role": "tool", "content": execution_result})
                turn_log[f"step_{count}"] = step_log

                drift_steps.append(
                    {
                        "turn": turn_idx,
                        "step": count,
                        "assistant_message": (
                            candidate_assistant if not refresh_triggered else candidate_assistant
                        ),
                        "executed_assistant_message": assistant_history,
                        "decoded_action": candidate_action,
                        "executed_action": executed_action,
                        "execution_results": execution_results,
                        "history_execution_results": execution_results,
                        "state": state_after_step,
                        "reference_action": reference_action,
                        "reference_state": ref_step.get("state") if ref_step else None,
                        "action_matches_reference": verify[
                            "candidate_action_matches_reference"
                        ],
                        "executed_action_matches_reference": executed_action_matches,
                        "state_matches_reference": state_matches,
                    }
                )
                checkpoint_steps.append(
                    {
                        "id": test_entry_id,
                        "global_step": len(drift_steps) - 1,
                        "turn": turn_idx,
                        "step": count,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_interval": self.checkpoint_interval,
                        "verifier": self.verifier,
                        "verify_threshold": self.verify_threshold,
                        "online_verify": self.online_verify,
                        "kv_divergence": verify["kv_divergence"],
                        "cumulative_divergence": verify["cumulative_divergence"],
                        "logit_kl": verify.get("logit_kl"),
                        "entropy": verify.get("entropy"),
                        "top1_top2_margin": verify.get("top1_top2_margin"),
                        "readout_available": verify.get("readout_available"),
                        "full_readout_source": verify.get("full_readout_source"),
                        "c2kv_readout_source": verify.get("c2kv_readout_source"),
                        "full_readout_dim": verify.get("full_readout_dim"),
                        "c2kv_readout_dim": verify.get("c2kv_readout_dim"),
                        "full_probe_seconds": verify.get("full_probe_seconds"),
                        "c2kv_probe_seconds": verify.get("c2kv_probe_seconds"),
                        "full_probe_prompt_tokens": verify.get(
                            "full_probe_prompt_tokens"
                        ),
                        "c2kv_probe_prompt_tokens": verify.get(
                            "c2kv_probe_prompt_tokens"
                        ),
                        "verify_triggered": True,
                        "refresh_triggered": refresh_triggered,
                        "recovery_mode": self.recovery_mode,
                        "regenerated_steps": int(refresh_triggered),
                        "full_regenerated_tokens": (
                            recovery_prompt_tokens if refresh_triggered else 0
                        ),
                        "recovery_prompt_mode": recovery_debug.get(
                            "recovery_prompt_mode"
                        ),
                        "c2kv_history_units": recovery_debug.get(
                            "c2kv_history_units", 0
                        ),
                        "full_history_units": recovery_debug.get(
                            "full_history_units", 0
                        ),
                        "current_messages": recovery_debug.get("current_messages", 0),
                        "regenerated_same_as_candidate": (
                            candidate_debug.get("regenerated_same_as_candidate")
                            if candidate_debug
                            else None
                        ),
                        "candidate_action_drift": not verify[
                            "candidate_action_matches_reference"
                        ],
                        "executed_action_drift": not executed_action_matches,
                        "state_drift": not state_matches,
                        "executable": not is_empty_execute_response(candidate_action),
                        "executed_executable": not is_empty_execute_response(
                            executed_action
                        ),
                        "history_prompt_tokens": candidate_usage["prompt_tokens"],
                        "request_history_tokens": _token_count(
                            self.tokenizer,
                            request_messages,
                        ),
                        "full_history_tokens": _token_count(
                            self.tokenizer,
                            checkpoint_messages,
                        ),
                    }
                )

                count += 1
                if is_empty_execute_response(executed_action):
                    break
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
            "checkpoint_steps": checkpoint_steps,
            "verify_count": verify_count,
            "refresh_count": refresh_count,
            "regenerated_steps": regenerated_steps,
            "full_regenerated_tokens": full_regenerated_tokens,
        }
        return all_model_response, metadata


def run(args: argparse.Namespace) -> None:
    runner = HistoryCheckpointRunner(args)
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
    step_rows = []
    for test_case in tqdm(entries, desc=f"history_checkpoint:{args.category}", dynamic_ncols=True):
        row = runner.run_sample_checkpoint(deepcopy(test_case))
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metrics_rows.append(row.get("c2kv_checkpoint_metrics", {}))
        step_rows.extend(row.get("checkpoint_steps") or [])

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metrics_rows)
    _write_jsonl(Path(args.step_metrics_path), step_rows)
    summary = {
        "category": args.category,
        "num_examples": len(details_rows),
        "compression_ratio": args.compression_ratio,
        "checkpoint_interval": args.checkpoint_interval,
        "verifier": args.verifier,
        "verify_threshold": args.verify_threshold,
        "verify_layers": args.verify_layers,
        "online_verify": args.online_verify,
        "recovery_mode": args.recovery_mode,
        "verify_count": sum(int(row.get("verify_count") or 0) for row in metrics_rows),
        "refresh_count": sum(int(row.get("refresh_count") or 0) for row in metrics_rows),
        "regenerated_steps": sum(int(row.get("regenerated_steps") or 0) for row in metrics_rows),
        "full_regenerated_tokens": sum(int(row.get("full_regenerated_tokens") or 0) for row in metrics_rows),
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
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--step-metrics-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument(
        "--compression-ratio",
        "--ratio",
        dest="compression_ratio",
        type=int,
        default=4,
    )
    parser.add_argument("--checkpoint-interval", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument(
        "--verifier",
        choices=["instant_kv", "cumulative_kv", "kv_divergence", "oracle"],
        default="oracle",
    )
    parser.add_argument("--verify-threshold", type=float, default=0.0)
    parser.add_argument(
        "--verify-layers",
        default="25%,50%,75%,last",
        help=(
            "Layer selector reserved for server-side per-layer readout. "
            "The current OpenAI-compatible path uses the returned query readout."
        ),
    )
    parser.add_argument("--online-verify", action="store_true")
    parser.add_argument(
        "--recovery-mode",
        choices=sorted(CHECKPOINT_MODES),
        default="current_step",
    )
    parser.add_argument("--recent-full-units", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.001)
    args = parser.parse_args()
    args.ratio = args.compression_ratio
    return args


if __name__ == "__main__":
    run(parse_args())
