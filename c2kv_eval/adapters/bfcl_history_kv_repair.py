from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

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
    "d_sham_neutral",
    "d_corr",
    "d_corr_recompute",
    "d_corr_all",
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
        self.require_plan = False
        self.plan = self._load_plan(args.plan_path)
        self.neutral_token_ids = self._load_neutral_tokens(args.neutral_corpus_path)
        self._active_tools: list[dict[str, Any]] = []
        self._repair_enabled_for_current_step = True

    def run_sample(self, test_case: dict[str, Any]) -> dict[str, Any]:
        self._active_tools = _tool_payload(test_case["function"])
        try:
            return super().run_sample(test_case)
        finally:
            self._active_tools = []

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
    ) -> list[int]:
        try:
            encoded = self.tokenizer.apply_chat_template(
                list(messages),
                tools=list(self._active_tools),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "KV repair requires exact server-equivalent chat-template "
                f"tokenization with tools; failed with {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalize_token_ids(encoded)

    def _full_history_doc_layout(
        self,
        texts: Sequence[str],
        doc_ids: Sequence[list[int]],
    ) -> tuple[list[int], list[int], list[int]]:
        """Locate each rendered history document in the real Full prompt.

        The raw repair slice must come from the same token coordinates used by
        the OpenAI chat endpoint: system/tool template first, then H0..Hk.
        """

        full_messages = [{"role": "user", "content": text} for text in texts]
        full_tokens = self._full_prompt_token_ids(full_messages)
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for index, ids in enumerate(doc_ids):
            found = -1
            limit = len(full_tokens) - len(ids) + 1
            for pos in range(cursor, max(cursor, limit)):
                if full_tokens[pos : pos + len(ids)] == ids:
                    found = pos
                    break
            if found < 0:
                raise RuntimeError(
                    "Cannot locate history doc in Full prompt tokenization: "
                    f"doc_index={index}, doc_len={len(ids)}, cursor={cursor}, "
                    f"full_len={len(full_tokens)}"
                )
            starts.append(found)
            ends.append(found + len(ids))
            cursor = found + len(ids)
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
            k_star = (num_docs - 1) // 2
            return {
                "k_star": k_star,
                "span_len": doc_lens[k_star],
                "sham_token_ids": [],
                "source": "online_median_doc",
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
        if len(self.neutral_token_ids) < span_len:
            raise RuntimeError("neutral corpus does not have enough tokens")
        return self.neutral_token_ids[:span_len]

    def _build_request_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> list[dict[str, Any]]:
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
        full_prompt_ids, starts, ends = self._full_history_doc_layout(texts, doc_ids)
        doc_lens = [end - start for start, end in zip(starts, ends)]
        canonical_full_history_tokens = ends[-1] - starts[0] if starts else 0
        stats.canonical_full_history_tokens += canonical_full_history_tokens

        sample_id = getattr(stats, "sample_id", "") or getattr(stats, "id", "")
        plan = self._plan_for(sample_id, len(units), doc_lens)
        k_star = int(plan.get("k_star", (len(units) - 1) // 2))

        gist_records: list[ExtractRecord | None] = []
        messages: list[dict[str, Any]] = []

        def should_compress_doc(index: int) -> bool:
            if effective_arm == "d_corr_recompute" and index > k_star:
                return False
            return True

        for index, (unit, text, ids) in enumerate(zip(units, texts, doc_ids)):
            if not should_compress_doc(index):
                full_tokens = _token_count(self.tokenizer, unit)
                stats.original_history_tokens += full_tokens
                stats.effective_history_tokens += full_tokens
                stats.physical_history_kv_tokens += doc_lens[index]
                stats.recomputed_raw_tokens += doc_lens[index]
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
            messages.append(
                {"role": "user", "content": text, "c2kv_key_hash": record.key_hash}
            )
            gist_records.append(record)

        repair_keys: list[str] = []
        repair_tokens = 0
        if effective_arm in {"c2kv", "d_sham_mech"}:
            repair_keys = []
        elif effective_arm == "d_sham_neutral":
            span_len = doc_lens[k_star]
            neutral_ids = self._neutral_ids_for(plan, span_len)
            repair = self._extract_repair(
                input_ids=neutral_ids,
                span_start=0,
                span_end=span_len,
                position_offset=starts[k_star],
                repair_mode=effective_arm,
                source_doc_index=k_star,
                stats=stats,
            )
            repair_keys.append(repair["key_hash"])
            repair_tokens += int(repair["token_len"])
        elif effective_arm in {"d_corr", "d_corr_recompute"}:
            prefix_ids = full_prompt_ids[: ends[k_star]]
            repair = self._extract_repair(
                input_ids=prefix_ids,
                span_start=starts[k_star],
                span_end=ends[k_star],
                position_offset=0,
                repair_mode=effective_arm,
                source_doc_index=k_star,
                stats=stats,
            )
            repair_keys.append(repair["key_hash"])
            repair_tokens += int(repair["token_len"])
        elif effective_arm == "d_corr_all":
            for index, ids in enumerate(doc_ids):
                prefix_ids = full_prompt_ids[: ends[index]]
                repair = self._extract_repair(
                    input_ids=prefix_ids,
                    span_start=starts[index],
                    span_end=ends[index],
                    position_offset=0,
                    repair_mode=effective_arm,
                    source_doc_index=index,
                    stats=stats,
                )
                repair_keys.append(repair["key_hash"])
                repair_tokens += int(repair["token_len"])

        if repair_keys:
            attach_index = k_star if effective_arm == "d_corr_recompute" else len(messages) - 1
            compressed_seen = -1
            target_message = None
            for message in messages:
                if message.get("c2kv_key_hash"):
                    compressed_seen += 1
                    if compressed_seen == attach_index or effective_arm != "d_corr_recompute":
                        target_message = message
            if target_message is None:
                raise RuntimeError(f"Cannot attach repair keys for arm={effective_arm}")
            target_message["c2kv_repair_key_hashes"] = repair_keys
            stats.effective_history_tokens += repair_tokens
            stats.physical_history_kv_tokens += repair_tokens
            stats.repair_kv_tokens += repair_tokens

        messages.extend(current)
        return messages

    def _oracle_repair_arms(self) -> set[str]:
        return {
            "d_sham_neutral",
            "d_corr",
            "d_corr_recompute",
            "d_corr_all",
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

        for turn_idx, current_turn_message in enumerate(test_case["question"]):
            messages.extend(deepcopy(current_turn_message))
            current_turn_response: list[str] = []
            current_turn_inputs: list[int] = []
            current_turn_outputs: list[int] = []
            current_turn_latency: list[float] = []
            turn_log: dict[str, Any] = {"begin_of_turn_query": current_turn_message}

            count = 0
            while True:
                state_before_execution = _state_log(involved_instances)
                ref_index = len(drift_steps)
                ref_step, alignment_status = reference_step_for(
                    reference_map,
                    reference_result,
                    turn_idx,
                    count,
                    fallback_state=state_before_execution,
                )

                self._repair_enabled_for_current_step = False
                request_messages = self._build_request_messages(messages, stats)
                plain_text, plain_response_message, plain_elapsed, plain_usage = self._query(
                    request_messages,
                    tools,
                    stats,
                )
                plain_candidate = decode_candidate(self.decoder, plain_text)
                repair_triggered = self._should_repair_candidate(
                    ref_step=ref_step,
                    candidate_action=plain_candidate.action,
                    candidate_status=plain_candidate.status,
                )

                text = plain_text
                response_message = plain_response_message
                elapsed = plain_elapsed
                usage = plain_usage
                candidate = plain_candidate
                repaired_candidate = None
                repair_text = None

                if repair_triggered:
                    self._repair_enabled_for_current_step = True
                    repair_request_messages = self._build_request_messages(messages, stats)
                    repair_text, response_message, repair_elapsed, repair_usage = self._query(
                        repair_request_messages,
                        tools,
                        stats,
                    )
                    repaired_candidate = decode_candidate(self.decoder, repair_text)
                    text = repair_text
                    elapsed = plain_elapsed + repair_elapsed
                    usage = {
                        "prompt_tokens": plain_usage["prompt_tokens"]
                        + repair_usage["prompt_tokens"],
                        "completion_tokens": plain_usage["completion_tokens"]
                        + repair_usage["completion_tokens"],
                    }

                final_candidate = repaired_candidate or plain_candidate
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
                    {
                        "role": "c2kv_oracle_repair",
                        "oracle_repair_triggered": repair_triggered,
                        "plain_candidate_status": plain_candidate.status,
                        "plain_candidate_action": plain_candidate.action,
                        "repair_candidate_action": (
                            repaired_candidate.action if repaired_candidate else None
                        ),
                    },
                ]
                if repair_text is not None:
                    step_log[-1]["repair_text"] = repair_text
                turn_log[f"step_{count}"] = step_log

                if final_candidate.decode_error:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Error decoding the model response.",
                            "error": final_candidate.decode_error,
                            "model_response_decoded": final_candidate.action,
                        }
                    )
                else:
                    step_log.append(
                        {
                            "role": "handler_log",
                            "content": "Successfully decoded model response.",
                            "model_response_decoded": final_candidate.action,
                        }
                    )

                decoded_to_execute = final_candidate.action
                assistant_for_history = assistant_history
                execution_results_for_history = None
                should_stop_after_record = False
                if (
                    final_candidate.status
                    in {"decode_error", "invalid_format", "empty_response"}
                    or is_empty_execute_response(decoded_to_execute)
                ):
                    should_stop_after_record = True

                messages.append(deepcopy(assistant_for_history))
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
                    candidate_raw_text=plain_text,
                    candidate_action=plain_candidate.action,
                    candidate_status=plain_candidate.status,
                    reference_step=ref_step,
                    alignment_status=alignment_status,
                    executed_action=decoded_to_execute,
                    state=state_after_step,
                    decode_error=plain_candidate.decode_error,
                    empty_response=plain_candidate.empty_response,
                    execution_error=execution_error,
                    candidate_assistant_message=_assistant_history_message(
                        plain_text,
                        plain_response_message.get("tool_calls"),
                    ),
                    executed_assistant_message=assistant_for_history,
                    execution_results=execution_results,
                    history_execution_results=history_execution_results,
                    roundtrip=roundtrip,
                )
                step_record["oracle_repair_triggered"] = repair_triggered
                step_record["repair_trigger"] = self.repair_trigger
                step_record["repair_arm"] = self.arm
                step_record["repair_raw_text"] = repair_text
                step_record["repair_action"] = (
                    repaired_candidate.action if repaired_candidate else None
                )
                step_record["repair_status"] = (
                    repaired_candidate.status if repaired_candidate else None
                )
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
        "chat_completion_tokens": sum(
            int(row.get("chat_completion_tokens") or 0) for row in metric_rows
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
