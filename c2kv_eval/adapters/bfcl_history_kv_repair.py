from __future__ import annotations

import argparse
import json
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mt_utils
from bfcl_eval.utils import load_dataset_entry, make_json_serializable, sort_file_content_by_id, sort_key

from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    ExtractRecord,
    HistoryDriftRunner,
    MAXIMUM_STEP_LIMIT,
    _assistant_history_message,
    _history_units,
    _latest_user_query_index,
    _message_text,
    _post_json,
    _render_history_unit,
    _state_log,
    _token_count,
    _tool_payload,
    _tool_calls_to_text,
    execute_multi_turn_func_call,
    is_empty_execute_response,
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


REPAIR_ARMS = {
    "full",
    "c2kv",
    "d_sham_mech",
    "hint_only",
    "d_sham_neutral",
    "d_corr",
    "d_corr_w1",
    "d_corr_w2",
    "d_corr_w4",
    "d_corr_w2_hint",
    "d_corr_w2_oracle_location_hint",
    "d_corr_replace_w2",
    "d_corr_recompute",
    "d_corr_recompute_w2",
    "d_corr_all",
    "raw_all_replace",
    "raw_all_replace_direct",
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


class KVRepairRunner(HistoryDriftRunner):
    """BFCL history runner for C2KV D-KV repair arms.

    This keeps BFCL tools/system/current turn full. Only completed history units
    are represented as full messages, C2KV gist messages, or gist+repair KV
    messages.
    """

    def __init__(self, args: argparse.Namespace):
        drift_args = deepcopy(args)
        drift_args.mode = (
            "history_full_closed_loop"
            if args.arm == "full"
            else "history_c2kv4_closed_loop"
        )
        super().__init__(drift_args)
        self.arm = args.arm
        self.repair_trigger = args.repair_trigger
        self.checkpoint_interval = max(1, int(args.checkpoint_interval))
        self.require_plan = False
        self.plan = self._load_plan(args.plan_path)
        self.neutral_token_ids = self._load_neutral_tokens(args.neutral_corpus_path)
        self._active_tools: list[dict[str, Any]] = []
        self._repair_enabled_for_current_step = True
        self._repair_target_history_index: int | None = None
        self._repair_window_arg = args.repair_window
        self._active_oracle_bad_step: int | None = None
        self._last_repair_build_info: dict[str, Any] = {}

    def run_sample(self, test_case: dict[str, Any]) -> dict[str, Any]:
        self._active_tools = _tool_payload(test_case["function"])
        try:
            stats = DriftStats(test_case["id"], self.mode, self.ratio)
            result, metadata = self._run_sample_impl(test_case, stats)
            metrics = stats.as_dict()
            metrics.update(self._repair_metrics(metadata.get("repair_segments") or []))
            metadata["c2kv_drift_metrics"] = metrics
            return {"id": test_case["id"], "result": result, **metadata}
        finally:
            self._active_tools = []

    @staticmethod
    def _repair_metrics(segments: Sequence[dict[str, Any]]) -> dict[str, Any]:
        total = len(segments)
        triggered = [seg for seg in segments if seg.get("repair_triggered")]
        harmful = [seg for seg in segments if seg.get("oracle_segment_harmful")]
        successful = [
            seg for seg in triggered if seg.get("repair_segment_success") is True
        ]
        start_state_correct = [
            seg for seg in triggered if seg.get("segment_start_state_matches_reference")
        ]
        start_state_drifted = [
            seg
            for seg in triggered
            if seg.get("segment_start_state_matches_reference") is False
        ]
        start_state_correct_success = [
            seg for seg in start_state_correct if seg.get("repair_segment_success") is True
        ]
        start_state_drifted_success = [
            seg for seg in start_state_drifted if seg.get("repair_segment_success") is True
        ]
        wrong_to_correct = sum(
            int(seg.get("c2kv_wrong_repair_correct") or 0) for seg in segments
        )
        correct_to_wrong = sum(
            int(seg.get("c2kv_correct_repair_wrong") or 0) for seg in segments
        )
        changed_action = sum(
            int(seg.get("repair_changed_action_count") or 0) for seg in segments
        )
        changed_first = sum(
            int(seg.get("repair_changed_first_token_count") or 0) for seg in segments
        )
        repaired_steps = sum(
            int(seg.get("segment_length") or 0) for seg in triggered
        )
        return {
            "repair_segments": total,
            "oracle_harmful_segments": len(harmful),
            "detector_trigger_count": len(triggered),
            "detector_trigger_rate": len(triggered) / total if total else None,
            "repair_rate": len(triggered) / total if total else None,
            "repair_success_count": len(successful),
            "repair_success_rate": (
                len(successful) / len(triggered) if triggered else None
            ),
            "repair_segment_success_rate": (
                len(successful) / len(harmful) if harmful else None
            ),
            "c2kv_wrong_repair_correct": wrong_to_correct,
            "c2kv_wrong_repair_wrong": sum(
                int(seg.get("c2kv_wrong_repair_wrong") or 0) for seg in segments
            ),
            "c2kv_correct_repair_wrong": correct_to_wrong,
            "net_repair_gain": wrong_to_correct - correct_to_wrong,
            "repair_changed_action_count": changed_action,
            "repair_changed_action_rate": (
                changed_action / repaired_steps if repaired_steps else None
            ),
            "repair_changed_first_token_count": changed_first,
            "repair_changed_first_token_rate": (
                changed_first / repaired_steps if repaired_steps else None
            ),
            "repaired_step_count": repaired_steps,
            "repair_success_when_start_state_correct": (
                len(start_state_correct_success) / len(start_state_correct)
                if start_state_correct
                else None
            ),
            "repair_success_when_start_state_already_drifted": (
                len(start_state_drifted_success) / len(start_state_drifted)
                if start_state_drifted
                else None
            ),
            "speculative_terminal_discarded_count": sum(
                int(bool(seg.get("speculative_terminal_discarded")))
                for seg in segments
            ),
        }

    def _load_plan(self, path: str) -> dict[str, Any]:
        if not path:
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_neutral_tokens(self, path: str) -> list[int]:
        if not path:
            return []
        text = Path(path).read_text(encoding="utf-8")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _restore_instances(
        self,
        test_entry_id: str,
        involved_instances: dict[str, Any],
    ) -> None:
        for class_name, instance in involved_instances.items():
            key = (
                f"{self.decoder.model_name_underline_replaced}_"
                f"{test_entry_id}_{class_name}_instance"
            )
            key = re.sub(r"[-./:]", "_", key)
            mt_utils.__dict__[key] = deepcopy(instance)

    def _snapshot(
        self,
        *,
        messages: list[dict[str, Any]],
        involved_instances: dict[str, Any],
        current_turn_response: list[str],
        current_turn_inputs: list[int],
        current_turn_outputs: list[int],
        current_turn_latency: list[float],
        turn_log: dict[str, Any],
        global_step: int,
    ) -> dict[str, Any]:
        return {
            "messages": deepcopy(messages),
            "instances": deepcopy(involved_instances),
            "state": _state_log(involved_instances),
            "current_turn_response": deepcopy(current_turn_response),
            "current_turn_inputs": deepcopy(current_turn_inputs),
            "current_turn_outputs": deepcopy(current_turn_outputs),
            "current_turn_latency": deepcopy(current_turn_latency),
            "turn_log": deepcopy(turn_log),
            "global_step": global_step,
            "repair_target_history_index": self._latest_compressed_history_index(messages),
        }

    def _restore_snapshot(
        self,
        *,
        test_entry_id: str,
        snapshot: dict[str, Any],
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any],
        list[str],
        list[int],
        list[int],
        list[float],
        dict[str, Any],
    ]:
        instances = deepcopy(snapshot["instances"])
        self._restore_instances(test_entry_id, instances)
        self._repair_target_history_index = snapshot.get("repair_target_history_index")
        return (
            deepcopy(snapshot["messages"]),
            instances,
            deepcopy(snapshot["current_turn_response"]),
            deepcopy(snapshot["current_turn_inputs"]),
            deepcopy(snapshot["current_turn_outputs"]),
            deepcopy(snapshot["current_turn_latency"]),
            deepcopy(snapshot["turn_log"]),
        )

    def _latest_compressed_history_index(
        self,
        history_messages: Sequence[dict[str, Any]],
    ) -> int | None:
        latest_query_index = _latest_user_query_index(history_messages)
        completed = list(history_messages[:latest_query_index])
        units = _history_units(completed)
        return len(units) - 1 if units else None

    def _unit_token_ids(self, text: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if self.tokenizer.bos_token and rendered.startswith(self.tokenizer.bos_token):
            rendered = rendered[len(self.tokenizer.bos_token) :]
        token_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        return list(token_ids)

    @staticmethod
    def _normalize_token_ids(encoded: Any) -> list[int]:
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if isinstance(encoded, dict) and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(x) for x in encoded]

    def _full_prompt_token_ids(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        try:
            encoded = self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(self._active_tools),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "KV repair requires exact server-equivalent chat-template "
                f"tokenization with tools; failed with {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalize_token_ids(encoded)

    def _role_prompt_token_ids(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> list[int]:
        encoded = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return self._normalize_token_ids(encoded)

    def _full_history_unit_layout(
        self,
        units: Sequence[Sequence[dict[str, Any]]],
        current_messages: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """Locate each role-preserving history unit in the real Full prompt.

        The raw repair slice must come from the same token coordinates used by
        the OpenAI chat endpoint: system/tool template first, the original
        role-preserving completed history, then the current turn and assistant
        generation prompt. This intentionally does not reuse the C2KV
        user-document wrapper, because raw repair KV must be the KV that the
        target span would have in the exact Full context.
        """

        completed_messages = [
            deepcopy(message)
            for unit in units
            for message in unit
        ]
        full_messages = completed_messages + deepcopy(list(current_messages or []))
        full_tokens = self._full_prompt_token_ids(
            full_messages,
            add_generation_prompt=True,
        )
        starts: list[int] = []
        ends: list[int] = []
        prefix_messages: list[dict[str, Any]] = []
        cursor = 0
        for index, unit in enumerate(units):
            if prefix_messages:
                start = len(self._full_prompt_token_ids(prefix_messages))
            else:
                unit_ids = self._role_prompt_token_ids(unit)
                found = -1
                limit = len(full_tokens) - len(unit_ids) + 1
                for pos in range(cursor, max(cursor, limit)):
                    if full_tokens[pos : pos + len(unit_ids)] == unit_ids:
                        found = pos
                        break
                if found < 0:
                    raise RuntimeError(
                        "Cannot locate first role-preserving history unit in "
                        "Full prompt tokenization."
                    )
                start = found
            prefix_messages.extend(deepcopy(list(unit)))
            end = len(self._full_prompt_token_ids(prefix_messages))
            if not (0 <= start < end <= len(full_tokens)):
                raise RuntimeError(
                    "Invalid Full prompt history-unit token span: "
                    f"unit_index={index}, start={start}, end={end}, "
                    f"full_len={len(full_tokens)}"
                )
            if full_tokens[:end] != self._full_prompt_token_ids(prefix_messages):
                raise RuntimeError(
                    "Full prompt prefix tokenization is not prefix-stable for "
                    f"history unit {index}; refusing to build raw repair KV."
                )
            starts.append(start)
            ends.append(end)
            cursor = end
        return full_tokens, starts, ends

    def _extract_repair(
        self,
        *,
        input_ids: list[int],
        span_start: int,
        span_end: int,
        position_offset: int,
        repair_mode: str,
        source_doc_index: int | None,
        stats: DriftStats,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        result = _post_json(
            self.base_url,
            "/v1/c2kv/repair_extract",
            {
                "input_ids": input_ids,
                "span_start": span_start,
                "span_end": span_end,
                "position_offset": position_offset,
                "repair_mode": repair_mode,
                "source_doc_index": source_doc_index,
            },
            self.timeout,
        )
        elapsed = time.perf_counter() - start
        stats.extract_seconds += elapsed
        stats.repair_extract_seconds += elapsed
        stats.repair_extract_recomputed_tokens += len(input_ids)
        stats.extract_calls += 1
        if result.get("success"):
            stats.extract_success += 1
        else:
            raise RuntimeError(
                f"repair_extract failed for {repair_mode}: {result.get('error')}"
            )
        return result

    def _plan_for(self, sample_id: str, num_docs: int, doc_lens: list[int]) -> dict[str, Any]:
        plan_root = self.plan.get("per_qid") if isinstance(self.plan, dict) else None
        if not isinstance(plan_root, dict):
            plan_root = self.plan
        plan = plan_root.get(sample_id) or plan_root.get(str(sample_id))
        if plan is None:
            if self._repair_target_history_index is None:
                k_star = num_docs - 1
            else:
                k_star = min(max(0, int(self._repair_target_history_index)), num_docs - 1)
            return {
                "k_star": k_star,
                "span_len": doc_lens[k_star],
                "sham_token_ids": [],
                "source": "online_latest_compressed_history",
            }
        k_star = int(plan.get("k_star"))
        if not (0 <= k_star < num_docs):
            raise RuntimeError(f"k_star out of range for {sample_id}: {k_star} / {num_docs}")
        span_len = int(plan.get("span_len", doc_lens[k_star]))
        if span_len != doc_lens[k_star]:
            raise RuntimeError(
                f"span length mismatch for {sample_id}: plan={span_len}, actual={doc_lens[k_star]}"
            )
        return plan

    def _neutral_ids_for(self, plan: dict[str, Any], span_len: int) -> list[int]:
        sham_ids = list(plan.get("sham_token_ids") or [])
        if sham_ids:
            if len(sham_ids) != span_len:
                raise RuntimeError(
                    f"d_sham_neutral token length mismatch: {len(sham_ids)} != {span_len}"
                )
            return [int(x) for x in sham_ids]
        if not self.neutral_token_ids:
            raise RuntimeError("neutral corpus does not contain any tokens")
        if len(self.neutral_token_ids) >= span_len:
            return self.neutral_token_ids[:span_len]

        repeats = (span_len + len(self.neutral_token_ids) - 1) // len(
            self.neutral_token_ids
        )
        return (self.neutral_token_ids * repeats)[:span_len]

    def _arm_config(self, effective_arm: str) -> dict[str, Any]:
        config = {
            "operation": "append",
            "repair_kind": "none",
            "window": self._repair_window_arg,
            "hint": False,
            "oracle_location_hint": False,
        }
        if effective_arm in {"c2kv", "d_sham_mech"}:
            config["repair_kind"] = "none"
        elif effective_arm == "hint_only":
            config.update({"repair_kind": "none", "hint": True})
        elif effective_arm == "d_sham_neutral":
            config.update({"repair_kind": "neutral", "window": "1"})
        elif effective_arm in {"d_corr", "d_corr_w1"}:
            config.update({"repair_kind": "raw", "window": "1"})
        elif effective_arm in {"d_corr_w2", "d_corr_w2_hint"}:
            config.update({
                "repair_kind": "raw",
                "window": "2",
                "hint": effective_arm.endswith("_hint"),
            })
        elif effective_arm == "d_corr_w2_oracle_location_hint":
            config.update({
                "repair_kind": "raw",
                "window": "2",
                "hint": True,
                "oracle_location_hint": True,
            })
        elif effective_arm == "d_corr_w4":
            config.update({"repair_kind": "raw", "window": "4"})
        elif effective_arm == "d_corr_all":
            config.update({"repair_kind": "raw", "window": "all"})
        elif effective_arm == "d_corr_replace_w2":
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "2",
            })
        elif effective_arm in {"d_corr_recompute", "d_corr_recompute_w2"}:
            config.update({
                "operation": "recompute",
                "repair_kind": "raw",
                "window": "2",
            })
        elif effective_arm in {"raw_all_replace", "raw_all_replace_direct"}:
            config.update({
                "operation": "replace",
                "repair_kind": "raw",
                "window": "all",
            })
        else:
            raise RuntimeError(f"Unsupported KV repair arm: {effective_arm}")
        return config

    @staticmethod
    def _ordered_roles(unit: Sequence[dict[str, Any]]) -> list[str]:
        roles: list[str] = []
        for message in unit:
            role = str(message.get("role") or "")
            if role and role not in roles:
                roles.append(role)
        return roles

    def _recent_repair_indices(
        self,
        *,
        num_docs: int,
        latest_index: int,
        window: str,
    ) -> list[int]:
        if num_docs <= 0:
            return []
        latest_index = min(max(0, latest_index), num_docs - 1)
        if window == "all":
            return list(range(num_docs))
        try:
            width = int(window)
        except Exception as exc:
            raise RuntimeError(f"Invalid repair window: {window}") from exc
        if width <= 0:
            raise RuntimeError(f"repair window must be positive, got {window}")
        start = max(0, latest_index - width + 1)
        return list(range(start, latest_index + 1))

    def _repair_hint(
        self,
        *,
        target_metadata: Sequence[dict[str, Any]],
        include_oracle_location: bool,
    ) -> str:
        restored = "; ".join(
            (
                f"H{meta['history_index']}: "
                f"{'+'.join(meta.get('roles') or [])}, "
                f"{meta.get('raw_token_count', 0)} raw-KV tokens"
            )
            for meta in target_metadata
        )
        lines = [
            "Recovery note: the current speculative segment was detected as inconsistent with the execution history.",
            (
                "Raw KV memory for history units "
                f"{[meta['history_index'] for meta in target_metadata]} "
                "has been restored and attached to the compressed history."
            ),
            f"Restored units contain: {restored}.",
            (
                "Re-evaluate the current tool call using the restored history, "
                "especially restored assistant/tool observations."
            ),
            "Do not assume the previous candidate action is correct.",
        ]
        if include_oracle_location and self._active_oracle_bad_step is not None:
            lines.append(
                "The inconsistency was first observed at speculative step "
                f"S{self._active_oracle_bad_step}."
            )
        return "\n".join(lines)

    def _build_request_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> list[dict[str, Any]]:
        self._last_repair_build_info = {}
        latest_query_index = _latest_user_query_index(history_messages)
        completed = list(history_messages[:latest_query_index])
        current = deepcopy(list(history_messages[latest_query_index:]))
        if self.arm == "full":
            full_tokens = _token_count(self.tokenizer, completed)
            stats.original_history_tokens += full_tokens
            stats.effective_history_tokens += full_tokens
            stats.canonical_full_history_tokens += full_tokens
            stats.physical_history_kv_tokens += full_tokens
            return deepcopy(list(history_messages))
        effective_arm = self.arm if self._repair_enabled_for_current_step else "c2kv"

        units = _history_units(completed)
        if not units:
            return deepcopy(list(history_messages))

        texts = [_render_history_unit(unit) for unit in units]
        doc_ids = [self._unit_token_ids(text) for text in texts]
        full_prompt_ids, starts, ends = self._full_history_unit_layout(
            units,
            current_messages=current,
        )
        doc_lens = [end - start for start, end in zip(starts, ends)]
        canonical_full_history_tokens = sum(doc_lens)
        stats.canonical_full_history_tokens += canonical_full_history_tokens

        sample_id = getattr(stats, "sample_id", "") or getattr(stats, "id", "")
        plan = self._plan_for(sample_id, len(units), doc_lens)
        latest_index = int(plan.get("k_star", (len(units) - 1)))
        config = self._arm_config(effective_arm)
        target_indices = self._recent_repair_indices(
            num_docs=len(units),
            latest_index=latest_index,
            window=str(config["window"]),
        )
        target_set = set(target_indices)
        anchor_index = target_indices[0] if target_indices else latest_index

        if (
            effective_arm in {"raw_all_replace", "raw_all_replace_direct"}
            and config["operation"] == "replace"
            and config["repair_kind"] == "raw"
            and target_indices == list(range(len(units)))
        ):
            history_start = starts[0]
            history_end = ends[-1]
            history_len = history_end - history_start
            if history_len != canonical_full_history_tokens:
                raise RuntimeError(
                    "raw_all_replace full-history span is not contiguous: "
                    f"span_len={history_len}, "
                    f"canonical_full_history_tokens={canonical_full_history_tokens}"
                )

            repair = self._extract_repair(
                input_ids=full_prompt_ids[:history_end],
                span_start=history_start,
                span_end=history_end,
                position_offset=0,
                repair_mode=effective_arm,
                source_doc_index=None,
                stats=stats,
            )
            token_len = int(repair["token_len"])
            if token_len != history_len:
                raise RuntimeError(
                    "raw_all_replace full-history repair length mismatch: "
                    f"requested={history_len}, injected={token_len}"
                )
            position_start = int(repair.get("position_start", history_start))
            position_end = int(repair.get("position_end", history_end))
            if position_start != history_start or position_end != history_end:
                raise RuntimeError(
                    "raw_all_replace full-history position range mismatch: "
                    f"expected=({history_start}, {history_end}), "
                    f"actual=({position_start}, {position_end})"
                )

            repair_metadata = [
                {
                    "history_index": index,
                    "roles": self._ordered_roles(units[index]),
                    "raw_token_count": doc_lens[index],
                    "absolute_position_start": starts[index],
                    "absolute_position_end": ends[index],
                }
                for index in target_indices
            ]
            stats.original_history_tokens += canonical_full_history_tokens
            stats.effective_history_tokens += token_len
            stats.physical_history_kv_tokens += token_len
            stats.repair_kv_tokens += token_len

            self._last_repair_build_info = {
                "repair_mode": effective_arm,
                "repair_window": config["window"],
                "repair_operation": config["operation"],
                "uses_oracle_error_location": bool(config["oracle_location_hint"]),
                "repair_target_indices": target_indices,
                "repair_target_roles": [
                    self._ordered_roles(units[index]) for index in target_indices
                ],
                "repair_target_metadata": repair_metadata,
                "repair_tokens_requested": token_len,
                "repair_tokens_injected": token_len,
                "repair_raw_tokens": token_len,
                "repair_physical_tokens": token_len,
                "repair_absolute_position_ranges": [
                    [meta["absolute_position_start"], meta["absolute_position_end"]]
                    for meta in repair_metadata
                ],
                "combined_full_history_repair": True,
                "physical_prefix_len_before": 0,
                "physical_prefix_len_after": token_len,
                "logical_position_before": canonical_full_history_tokens,
                "logical_position_after": canonical_full_history_tokens,
            }

            return [
                {
                    "role": "user",
                    "content": "\n".join(texts),
                    "c2kv_repair_only_key_hashes": [repair["key_hash"]],
                },
                *current,
            ]

        gist_records: list[ExtractRecord | None] = []
        messages: list[dict[str, Any]] = []
        repair_keys_by_index: dict[int, list[str]] = {}
        repair_tokens_by_index: dict[int, int] = {}
        repair_metadata: list[dict[str, Any]] = []
        repair_tokens = 0
        local_physical_history_tokens = 0

        def should_compress_doc(index: int) -> bool:
            if config["operation"] == "replace" and index in target_set:
                return False
            if config["operation"] == "recompute" and index >= anchor_index:
                return False
            return True

        def build_repair_for_index(index: int) -> None:
            nonlocal repair_tokens
            if config["repair_kind"] == "none":
                return
            span_len = doc_lens[index]
            if config["repair_kind"] == "neutral":
                input_ids = self._neutral_ids_for(plan, span_len)
                span_start = 0
                span_end = span_len
                position_offset = starts[index]
            elif config["repair_kind"] == "raw":
                input_ids = full_prompt_ids[: ends[index]]
                span_start = starts[index]
                span_end = ends[index]
                position_offset = 0
            else:
                raise RuntimeError(f"Unsupported repair kind: {config['repair_kind']}")

            repair = self._extract_repair(
                input_ids=input_ids,
                span_start=span_start,
                span_end=span_end,
                position_offset=position_offset,
                repair_mode=effective_arm,
                source_doc_index=index,
                stats=stats,
            )
            token_len = int(repair["token_len"])
            expected_len = span_len
            if token_len != expected_len:
                raise RuntimeError(
                    "repair token length mismatch: "
                    f"index={index}, requested={expected_len}, injected={token_len}"
                )
            repair_keys_by_index.setdefault(index, []).append(repair["key_hash"])
            repair_tokens_by_index[index] = repair_tokens_by_index.get(index, 0) + token_len
            repair_tokens += token_len
            position_start = int(repair.get("position_start", starts[index]))
            position_end = int(repair.get("position_end", ends[index]))
            if position_start != starts[index] or position_end != ends[index]:
                raise RuntimeError(
                    "repair position range mismatch: "
                    f"index={index}, expected=({starts[index]}, {ends[index]}), "
                    f"actual=({position_start}, {position_end})"
                )
            repair_metadata.append(
                {
                    "history_index": index,
                    "roles": self._ordered_roles(units[index]),
                    "raw_token_count": token_len,
                    "absolute_position_start": position_start,
                    "absolute_position_end": position_end,
                }
            )

        if config["operation"] == "append":
            for index in target_indices:
                build_repair_for_index(index)
        elif config["operation"] == "replace":
            for index in target_indices:
                build_repair_for_index(index)
        elif config["operation"] == "recompute":
            build_repair_for_index(anchor_index)

        for index, (unit, text, ids) in enumerate(zip(units, texts, doc_ids)):
            if not should_compress_doc(index):
                stats.original_history_tokens += doc_lens[index]
                if index in repair_keys_by_index:
                    raw_len = repair_tokens_by_index[index]
                    stats.effective_history_tokens += raw_len
                    stats.physical_history_kv_tokens += raw_len
                    stats.repair_kv_tokens += raw_len
                    local_physical_history_tokens += raw_len
                    messages.append(
                        {
                            "role": "user",
                            "content": text,
                            "c2kv_repair_only_key_hashes": repair_keys_by_index[index],
                        }
                    )
                else:
                    full_tokens = _token_count(self.tokenizer, unit)
                    stats.effective_history_tokens += full_tokens
                    stats.physical_history_kv_tokens += doc_lens[index]
                    stats.recomputed_raw_tokens += doc_lens[index]
                    local_physical_history_tokens += doc_lens[index]
                    messages.extend(deepcopy(unit))
                gist_records.append(None)
                continue

            full_tokens = len(ids)
            record = self._extract_history_unit(text, stats)
            stats.original_history_tokens += int(record.original_seq_len or full_tokens)
            if not (record.success and record.key_hash):
                raise RuntimeError(f"C2KV extract failed in arm={self.arm}: {record.error}")
            gist_len = int(record.gist_len or record.original_seq_len or full_tokens)
            stats.effective_history_tokens += gist_len
            stats.physical_history_kv_tokens += gist_len
            stats.c2kv_gist_tokens += gist_len
            local_physical_history_tokens += gist_len
            messages.append(
                {"role": "user", "content": text, "c2kv_key_hash": record.key_hash}
            )
            gist_records.append(record)

        append_repair_keys = [
            key_hash
            for index in target_indices
            for key_hash in repair_keys_by_index.get(index, [])
        ]
        if config["operation"] == "append" and append_repair_keys:
            target_message = None
            for message in reversed(messages):
                if message.get("c2kv_key_hash"):
                    target_message = message
                    break
            if target_message is None:
                raise RuntimeError(f"Cannot attach append repair keys for arm={effective_arm}")
            target_message["c2kv_repair_key_hashes"] = append_repair_keys
            stats.effective_history_tokens += repair_tokens
            stats.physical_history_kv_tokens += repair_tokens
            stats.repair_kv_tokens += repair_tokens
            local_physical_history_tokens += repair_tokens

        if config["operation"] == "recompute" and anchor_index + 1 < len(units):
            recomputed_total = sum(doc_lens[anchor_index + 1 :])
            if recomputed_total <= 0:
                raise RuntimeError(
                    f"{effective_arm} expected downstream recompute tokens."
                )

        physical_before = local_physical_history_tokens
        logical_before = canonical_full_history_tokens
        logical_after = canonical_full_history_tokens
        if repair_tokens and logical_before != logical_after:
            raise RuntimeError("repair unexpectedly changed logical current position")

        if config["hint"] and self._repair_enabled_for_current_step:
            messages.append(
                {
                    "role": "system",
                    "content": self._repair_hint(
                        target_metadata=repair_metadata,
                        include_oracle_location=bool(config["oracle_location_hint"]),
                    ),
                }
            )

        self._last_repair_build_info = {
            "repair_mode": effective_arm,
            "repair_window": config["window"],
            "repair_operation": config["operation"],
            "uses_oracle_error_location": bool(config["oracle_location_hint"]),
            "repair_target_indices": target_indices,
            "repair_target_roles": [
                self._ordered_roles(units[index]) for index in target_indices
            ],
            "repair_target_metadata": repair_metadata,
            "repair_tokens_requested": repair_tokens,
            "repair_tokens_injected": repair_tokens,
            "repair_raw_tokens": repair_tokens,
            "repair_physical_tokens": repair_tokens,
            "repair_absolute_position_ranges": [
                [meta["absolute_position_start"], meta["absolute_position_end"]]
                for meta in repair_metadata
            ],
            "physical_prefix_len_before": max(0, physical_before - repair_tokens),
            "physical_prefix_len_after": physical_before,
            "logical_position_before": logical_before,
            "logical_position_after": logical_after,
        }
        if repair_tokens != self._last_repair_build_info["repair_tokens_injected"]:
            raise RuntimeError("repair token accounting invariant failed")

        messages.extend(current)
        return messages

    def _oracle_repair_arms(self) -> set[str]:
        return {
            "d_sham_mech",
            "hint_only",
            "d_sham_neutral",
            "d_corr",
            "d_corr_w1",
            "d_corr_w2",
            "d_corr_w4",
            "d_corr_w2_hint",
            "d_corr_w2_oracle_location_hint",
            "d_corr_replace_w2",
            "d_corr_recompute",
            "d_corr_recompute_w2",
            "d_corr_all",
            "raw_all_replace",
        }

    def _should_repair_candidate(
        self,
        *,
        ref_step: dict[str, Any] | None,
        candidate_action: list[str],
        candidate_status: str,
    ) -> bool:
        if self.repair_trigger != "oracle":
            return self.repair_trigger == "always"
        if ref_step is None:
            return False
        reference_action = ref_step.get("decoded_action") or []
        if candidate_status in {"decode_error", "invalid_format", "empty_response"}:
            return bool(reference_action)
        return not action_matches(candidate_action, reference_action)

    def _run_sample_impl(
        self,
        test_case: dict[str, Any],
        stats: DriftStats,
    ) -> tuple[list[list[str]], dict[str, Any]]:
        if self.arm not in self._oracle_repair_arms():
            self._repair_enabled_for_current_step = True
            return super()._run_sample_impl(test_case, stats)

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
        repair_segments: list[dict[str, Any]] = []

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}

            count = 0

            def run_one_step(
                *,
                step_idx: int,
                global_step: int,
                repair_enabled: bool,
                source_info: dict[str, Any] | None = None,
            ) -> tuple[dict[str, Any], bool]:
                nonlocal messages, involved_instances, force_quit

                state_before_execution = _state_log(involved_instances)
                ref_step, alignment_status = reference_step_for(
                    reference_map,
                    reference_result,
                    turn_idx,
                    step_idx,
                    fallback_state=state_before_execution,
                )

                self._repair_enabled_for_current_step = repair_enabled
                request_messages = self._build_request_messages(messages, stats)
                repair_build_info = deepcopy(self._last_repair_build_info)
                text, response_message, elapsed, usage = self._query(
                    request_messages,
                    tools,
                    stats,
                )
                decoded = decode_candidate(self.decoder, text)

                assistant_history = _assistant_history_message(
                    text,
                    response_message.get("tool_calls"),
                )
                current_turn_response.append(text)
                current_turn_inputs.append(usage["prompt_tokens"])
                current_turn_outputs.append(usage["completion_tokens"])
                current_turn_latency.append(elapsed)

                step_log: list[dict[str, Any]] = [
                    {"role": "assistant", "content": text},
                    {
                        "role": "c2kv_repair_segment",
                        "repair_enabled": repair_enabled,
                        "repair_mode": self.arm if repair_enabled else "c2kv",
                        "repair_target_history_index": self._repair_target_history_index,
                        "repair_build_info": repair_build_info,
                    },
                ]
                turn_log[f"step_{step_idx}"] = step_log
                if decoded.decode_error:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response.",
                            "error": decoded.decode_error,
                            "model_response_decoded": decoded.action,
                        }
                    )
                else:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": decoded.action,
                        }
                    )

                decoded_to_execute = decoded.action
                should_stop_after_record = False
                if (
                    decoded.status
                    in {"decode_error", "invalid_format", "empty_response"}
                    or is_empty_execute_response(decoded_to_execute)
                ):
                    should_stop_after_record = True

                messages.append(deepcopy(assistant_history))
                execution_error = None
                if is_empty_execute_response(decoded_to_execute):
                    execution_results = []
                else:
                    tool_start = time.perf_counter()
                    try:
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
                for idx, execution_result in enumerate(execution_results):
                    messages.append(
                        {
                            "role": "tool",
                            "content": execution_result,
                            "tool_call_id": f"call_{turn_idx}_{step_idx}_{idx}",
                        }
                    )
                    step_log.append({"role": "tool", "content": execution_result})

                state_after_step = _state_log(involved_instances)
                executed_text = _message_text(assistant_history)
                if assistant_history.get("tool_calls"):
                    tool_call_text = _tool_calls_to_text(assistant_history.get("tool_calls"))
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

                candidate_raw_text = text
                candidate_action = decoded.action
                candidate_status = decoded.status
                candidate_decode_error = decoded.decode_error
                candidate_empty_response = decoded.empty_response
                if source_info is not None:
                    source_record = source_info["step_record"]
                    candidate_raw_text = source_record.get("candidate_raw_text") or ""
                    candidate_action = list(source_record.get("candidate_action") or [])
                    candidate_status = source_record.get("candidate_status") or "empty_action"
                    candidate_decode_error = source_record.get("decode_error")
                    candidate_empty_response = bool(source_record.get("empty_response"))

                step_record = build_step_record(
                    sample_id=test_entry_id,
                    turn_idx=turn_idx,
                    step_idx=step_idx,
                    global_step=global_step,
                    candidate_raw_text=candidate_raw_text,
                    candidate_action=candidate_action,
                    candidate_status=candidate_status,
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=decoded_to_execute,
                    state=state_after_step,
                    decode_error=candidate_decode_error,
                    empty_response=candidate_empty_response,
                    execution_error=execution_error,
                    candidate_assistant_message=(
                        source_info["assistant_history"]
                        if source_info is not None
                        else assistant_history
                    ),
                    executed_assistant_message=assistant_history,
                    execution_results=execution_results,
                    history_execution_results=execution_results,
                    roundtrip=roundtrip,
                    extra={
                        "repair_triggered": repair_enabled,
                        "repair_arm": self.arm,
                        "repair_mode": self.arm if repair_enabled else "c2kv",
                        "repair_target_history_index": self._repair_target_history_index,
                        "repair_build_info": repair_build_info,
                        "repair_raw_text": text if repair_enabled else None,
                        "repair_action": decoded.action if repair_enabled else None,
                        "repair_status": decoded.status if repair_enabled else None,
                    },
                )
                step_record["oracle_harmful"] = bool(
                    step_record.get("candidate_action_drift")
                    or step_record.get("state_drift")
                )
                if source_info is not None:
                    step_record["plain_c2kv_raw_text"] = candidate_raw_text
                    step_record["plain_c2kv_action"] = candidate_action
                    step_record["plain_c2kv_status"] = candidate_status
                    step_record["repair_changed_action"] = not action_matches(
                        decoded.action,
                        candidate_action,
                    )
                    candidate_ids = self.tokenizer.encode(
                        candidate_raw_text,
                        add_special_tokens=False,
                    )
                    repaired_ids = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,
                    )
                    step_record["candidate_first_token_id"] = (
                        int(candidate_ids[0]) if candidate_ids else None
                    )
                    step_record["repaired_first_token_id"] = (
                        int(repaired_ids[0]) if repaired_ids else None
                    )
                    step_record["repair_changed_first_token"] = (
                        step_record["candidate_first_token_id"]
                        != step_record["repaired_first_token_id"]
                    )
                    step_record["c2kv_wrong_repair_correct"] = bool(
                        step_record.get("candidate_action_drift")
                        and not step_record.get("executed_action_drift")
                        and not step_record.get("state_drift")
                    )
                    step_record["c2kv_wrong_repair_wrong"] = bool(
                        step_record.get("candidate_action_drift")
                        and (
                            step_record.get("executed_action_drift")
                            or step_record.get("state_drift")
                        )
                    )
                    step_record["c2kv_correct_repair_wrong"] = bool(
                        not step_record.get("candidate_action_drift")
                        and (
                            step_record.get("executed_action_drift")
                            or step_record.get("state_drift")
                        )
                    )
                    if self.arm == "d_sham_mech":
                        expected = (candidate_raw_text or "").strip()
                        actual = (text or "").strip()
                        expected_ids = self.tokenizer.encode(
                            expected,
                            add_special_tokens=False,
                        )
                        actual_ids = self.tokenizer.encode(
                            actual,
                            add_special_tokens=False,
                        )
                        if expected_ids != actual_ids:
                            raise RuntimeError(
                                "d_sham_mech changed generated token ids relative to "
                                "plain C2KV while repair plumbing should be a no-op."
                            )

                if alignment_status == "missing_reference":
                    stats.errors.append(
                        f"missing reference action at turn={turn_idx}, "
                        f"step={step_idx}, candidate_global_step={global_step}"
                    )
                if roundtrip["serialization_mismatch"]:
                    stats.errors.append(
                        f"serialization mismatch at turn={turn_idx}, "
                        f"step={step_idx}, candidate_global_step={global_step}"
                    )

                if should_stop_after_record:
                    return (
                        {
                            "step_record": step_record,
                            "assistant_history": assistant_history,
                            "text": text,
                            "usage": usage,
                            "elapsed": elapsed,
                            "terminal": True,
                        },
                        True,
                    )
                if step_idx + 1 > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": (
                                f"Model has been forced to quit after "
                                f"{MAXIMUM_STEP_LIMIT} steps."
                            ),
                        }
                    )
                    return (
                        {
                            "step_record": step_record,
                            "assistant_history": assistant_history,
                            "text": text,
                            "usage": usage,
                            "elapsed": elapsed,
                            "terminal": True,
                        },
                        True,
                    )
                return (
                    {
                        "step_record": step_record,
                        "assistant_history": assistant_history,
                        "text": text,
                        "usage": usage,
                        "elapsed": elapsed,
                        "terminal": False,
                    },
                    False,
                )

            while True:
                segment_checkpoint = self._snapshot(
                    messages=messages,
                    involved_instances=involved_instances,
                    current_turn_response=current_turn_response,
                    current_turn_inputs=current_turn_inputs,
                    current_turn_outputs=current_turn_outputs,
                    current_turn_latency=current_turn_latency,
                    turn_log=turn_log,
                    global_step=len(drift_steps),
                )
                segment_start_count = count
                segment_infos: list[dict[str, Any]] = []
                speculative_terminal = False
                for _ in range(self.checkpoint_interval):
                    info, terminal = run_one_step(
                        step_idx=count,
                        global_step=len(drift_steps) + len(segment_infos),
                        repair_enabled=False,
                    )
                    segment_infos.append(info)
                    if terminal:
                        speculative_terminal = True
                        break
                    count += 1

                if not segment_infos:
                    break

                oracle_segment_harmful = any(
                    bool(info["step_record"].get("oracle_harmful"))
                    for info in segment_infos
                )
                repair_triggered = (
                    oracle_segment_harmful
                    if self.repair_trigger == "oracle"
                    else self.repair_trigger == "always"
                )
                harmful_step_indices = [
                    idx
                    for idx, info in enumerate(segment_infos)
                    if bool(info["step_record"].get("oracle_harmful"))
                ]
                has_action_drift = any(
                    bool(info["step_record"].get("candidate_action_drift"))
                    for info in segment_infos
                )
                has_state_drift = any(
                    bool(info["step_record"].get("state_drift"))
                    for info in segment_infos
                )
                if has_action_drift and has_state_drift:
                    harmful_reason = "both"
                elif has_action_drift:
                    harmful_reason = "action_drift"
                elif has_state_drift:
                    harmful_reason = "state_drift"
                else:
                    harmful_reason = "none"
                repair_segment = {
                    "sample_id": test_entry_id,
                    "turn": turn_idx,
                    "segment_start_step": segment_start_count,
                    "segment_length": len(segment_infos),
                    "checkpoint_interval": self.checkpoint_interval,
                    "detector_trigger": repair_triggered,
                    "oracle_segment_harmful": oracle_segment_harmful,
                    "oracle_harmful_reason": harmful_reason,
                    "harmful_step_indices": harmful_step_indices,
                    "repair_triggered": repair_triggered,
                    "repair_trigger_policy": self.repair_trigger,
                    "repair_mode": self.arm if repair_triggered else "c2kv",
                    "repair_target_history_index": segment_checkpoint.get(
                        "repair_target_history_index"
                    ),
                    "candidate_action_drift_per_step": [
                        bool(info["step_record"].get("candidate_action_drift"))
                        for info in segment_infos
                    ],
                    "state_drift_per_step": [
                        bool(info["step_record"].get("state_drift"))
                        for info in segment_infos
                    ],
                    "speculative_terminal_discarded": False,
                    "repair_segment_success": None,
                    "c2kv_wrong_repair_correct": 0,
                    "c2kv_wrong_repair_wrong": 0,
                    "c2kv_correct_repair_wrong": 0,
                    "repair_changed_action_count": 0,
                    "repair_changed_first_token_count": 0,
                    "candidate_action_correct": not has_action_drift,
                    "candidate_state_correct": not has_state_drift,
                    "repaired_action_correct": None,
                    "repaired_state_correct": None,
                    "segment_start_state_matches_reference": (
                        not any(bool(row.get("state_drift")) for row in drift_steps)
                    ),
                }

                if not repair_triggered:
                    for info in segment_infos:
                        step_record = info["step_record"]
                        mark_first_divergence(stats, step_record)
                        drift_steps.append(step_record)
                    repair_segments.append(repair_segment)
                    if speculative_terminal or force_quit:
                        break
                    continue

                repair_segment["speculative_terminal_discarded"] = speculative_terminal
                self._active_oracle_bad_step = (
                    harmful_step_indices[0] if harmful_step_indices else None
                )
                (
                    messages,
                    involved_instances,
                    current_turn_response,
                    current_turn_inputs,
                    current_turn_outputs,
                    current_turn_latency,
                    turn_log,
                ) = self._restore_snapshot(
                    test_entry_id=test_entry_id,
                    snapshot=segment_checkpoint,
                )
                count = segment_start_count
                repaired_records: list[dict[str, Any]] = []
                repair_terminal = False
                for source_info in segment_infos:
                    info, terminal = run_one_step(
                        step_idx=count,
                        global_step=len(drift_steps),
                        repair_enabled=True,
                        source_info=source_info,
                    )
                    step_record = info["step_record"]
                    step_record["oracle_segment_harmful"] = oracle_segment_harmful
                    step_record["detector_trigger"] = repair_triggered
                    step_record["repair_triggered"] = True
                    mark_first_divergence(stats, step_record)
                    drift_steps.append(step_record)
                    repaired_records.append(step_record)
                    repair_segment["c2kv_wrong_repair_correct"] += int(
                        bool(step_record.get("c2kv_wrong_repair_correct"))
                    )
                    repair_segment["c2kv_wrong_repair_wrong"] += int(
                        bool(step_record.get("c2kv_wrong_repair_wrong"))
                    )
                    repair_segment["c2kv_correct_repair_wrong"] += int(
                        bool(step_record.get("c2kv_correct_repair_wrong"))
                    )
                    repair_segment["repair_changed_action_count"] += int(
                        bool(step_record.get("repair_changed_action"))
                    )
                    repair_segment["repair_changed_first_token_count"] += int(
                        bool(step_record.get("repair_changed_first_token"))
                    )
                    if step_record.get("repair_build_info"):
                        build_info = dict(step_record["repair_build_info"])
                        for key, value in build_info.items():
                            repair_segment.setdefault(key, value)
                    if terminal:
                        repair_terminal = True
                        break
                    count += 1
                repair_segment["repair_segment_success"] = bool(
                    repaired_records
                    and all(
                        not row.get("executed_action_drift")
                        and not row.get("state_drift")
                        for row in repaired_records
                    )
                )
                repair_segment["repaired_action_correct"] = bool(
                    repaired_records
                    and all(not row.get("executed_action_drift") for row in repaired_records)
                )
                repair_segment["repaired_state_correct"] = bool(
                    repaired_records
                    and all(not row.get("state_drift") for row in repaired_records)
                )
                repair_segments.append(repair_segment)
                self._active_oracle_bad_step = None
                if repair_terminal or force_quit:
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
            "repair_segments": repair_segments,
        }
        return all_model_response, metadata


def run(args: argparse.Namespace) -> None:
    runner = KVRepairRunner(args)
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
    for test_case in tqdm(entries, desc=f"kv_repair:{args.arm}", dynamic_ncols=True):
        row = runner.run_sample(deepcopy(test_case))
        row["kv_repair_arm"] = args.arm
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metric_rows.append(row.get("c2kv_drift_metrics", {}))

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metric_rows)
    summary = {
        "arm": args.arm,
        "category": args.category,
        "num_examples": len(details_rows),
        "errors": sum(
            1
            for row in details_rows
            if str(row.get("result", "")).startswith("Error during inference")
        ),
        "chat_calls": sum(int(row.get("chat_calls") or 0) for row in metric_rows),
        "extract_calls": sum(int(row.get("extract_calls") or 0) for row in metric_rows),
        "extract_success": sum(int(row.get("extract_success") or 0) for row in metric_rows),
        "chat_seconds": sum(float(row.get("chat_seconds") or 0.0) for row in metric_rows),
        "extract_seconds": sum(float(row.get("extract_seconds") or 0.0) for row in metric_rows),
        "c2kv_extract_seconds": sum(float(row.get("c2kv_extract_seconds") or 0.0) for row in metric_rows),
        "repair_extract_seconds": sum(float(row.get("repair_extract_seconds") or 0.0) for row in metric_rows),
        "tool_execution_seconds": sum(float(row.get("tool_execution_seconds") or 0.0) for row in metric_rows),
        "episode_e2e_observed_seconds": sum(
            float(row.get("episode_e2e_observed_seconds") or 0.0)
            for row in metric_rows
        ),
        "chat_prompt_tokens": sum(
            int(row.get("chat_prompt_tokens") or 0) for row in metric_rows
        ),
        "chat_cached_tokens": sum(
            int(row.get("chat_cached_tokens") or 0) for row in metric_rows
        ),
        "chat_recomputed_prompt_tokens": sum(
            int(row.get("chat_recomputed_prompt_tokens") or 0)
            for row in metric_rows
        ),
        "chat_cache_report_missing": sum(
            int(row.get("chat_cache_report_missing") or 0) for row in metric_rows
        ),
        "chat_completion_tokens": sum(
            int(row.get("chat_completion_tokens") or 0) for row in metric_rows
        ),
        "kv_peak_resident_tokens": max(
            [int(row.get("kv_peak_resident_tokens") or 0) for row in metric_rows]
            or [0]
        ),
        "kv_runtime_report_missing": sum(
            int(row.get("kv_runtime_report_missing") or 0) for row in metric_rows
        ),
        "c2kv_extract_recomputed_tokens": sum(
            int(row.get("c2kv_extract_recomputed_tokens") or 0)
            for row in metric_rows
        ),
        "repair_extract_recomputed_tokens": sum(
            int(row.get("repair_extract_recomputed_tokens") or 0)
            for row in metric_rows
        ),
        "total_actual_recomputed_tokens": sum(
            int(row.get("total_actual_recomputed_tokens") or 0)
            for row in metric_rows
        ),
        "history_original_tokens": sum(
            int(row.get("history_original_tokens") or 0) for row in metric_rows
        ),
        "history_effective_tokens": sum(
            int(row.get("history_effective_tokens") or 0) for row in metric_rows
        ),
        "canonical_full_history_tokens": sum(
            int(row.get("canonical_full_history_tokens") or 0) for row in metric_rows
        ),
        "physical_history_kv_tokens": sum(
            int(row.get("physical_history_kv_tokens") or 0) for row in metric_rows
        ),
        "c2kv_gist_tokens": sum(
            int(row.get("c2kv_gist_tokens") or 0) for row in metric_rows
        ),
        "repair_kv_tokens": sum(
            int(row.get("repair_kv_tokens") or 0) for row in metric_rows
        ),
        "recomputed_raw_tokens": sum(
            int(row.get("recomputed_raw_tokens") or 0) for row in metric_rows
        ),
        "repaired_step_count": sum(
            int(row.get("repaired_step_count") or 0) for row in metric_rows
        ),
        "repair_changed_action_count": sum(
            int(row.get("repair_changed_action_count") or 0) for row in metric_rows
        ),
        "repair_changed_first_token_count": sum(
            int(row.get("repair_changed_first_token_count") or 0)
            for row in metric_rows
        ),
        "net_repair_gain": sum(
            int(row.get("net_repair_gain") or 0) for row in metric_rows
        ),
    }
    summary["extract_success_rate"] = (
        summary["extract_success"] / summary["extract_calls"]
        if summary["extract_calls"]
        else None
    )
    summary["history_kv_compression"] = (
        summary["canonical_full_history_tokens"] / summary["physical_history_kv_tokens"]
        if summary["physical_history_kv_tokens"]
        else None
    )
    summary["avg_episode_e2e_observed_seconds"] = (
        summary["episode_e2e_observed_seconds"] / summary["num_examples"]
        if summary["num_examples"]
        else None
    )
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(REPAIR_ARMS), required=True)
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
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=4,
        help="Number of plain C2KV speculative steps before Oracle segment repair.",
    )
    parser.add_argument(
        "--repair-window",
        default="1",
        choices=["1", "2", "4", "all"],
        help="Recent-W history units to restore for generic repair arms.",
    )
    parser.add_argument("--plan-path", default="")
    parser.add_argument(
        "--repair-trigger",
        choices=["oracle", "always"],
        default="oracle",
        help=(
            "oracle: first query plain C2KV and build repair KV only when the "
            "frozen Full reference says the candidate action drifted. always: "
            "apply the selected repair arm on every step."
        ),
    )
    parser.add_argument(
        "--neutral-corpus-path",
        default="/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_neutral_corpus.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
