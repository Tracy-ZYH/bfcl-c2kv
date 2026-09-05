from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm

from bfcl_eval.utils import load_dataset_entry, make_json_serializable, sort_file_content_by_id, sort_key

from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_MODEL_ID,
    DEFAULT_TOKENIZER_PATH,
    DriftStats,
    ExtractRecord,
    HistoryDriftRunner,
    HTTP,
    _history_units,
    _json_dumps,
    _latest_user_query_index,
    _post_json,
    _render_history_unit,
    _token_count,
    _tool_payload,
)


HISTORY_KV_METHODS = {
    "full",
    "c2kv",
    "streamingllm",
    "h2o",
    "snapkv",
    "snapkv_persistent",
    "snapkv_refresh",
    "pyramidkv",
    "kivi",
}

RUNTIME_EVICTION_METHODS = {
    "streamingllm",
    "h2o",
    "snapkv",
    "snapkv_persistent",
    "pyramidkv",
}

UNSUPPORTED_RUNTIME_METHODS: set[str] = set()
ATTENTION_SCORE_PHYSICAL_METHODS = {"h2o", "snapkv", "snapkv_persistent", "pyramidkv"}


class RuntimeEvictionUnsupported(RuntimeError):
    pass


@dataclass
class HistoryKVDecision:
    method: str
    raw_history_tokens: int
    active_history_kv_tokens: int
    retained_units: list[int]
    dropped_units: list[int]
    runtime_eviction_required: bool = False
    runtime_eviction_available: bool = False
    notes: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        compression = (
            self.raw_history_tokens / self.active_history_kv_tokens
            if self.active_history_kv_tokens
            else None
        )
        return {
            "history_kv_method": self.method,
            "history_raw_tokens": self.raw_history_tokens,
            "history_active_kv_tokens": self.active_history_kv_tokens,
            "history_retention_ratio": (
                self.active_history_kv_tokens / self.raw_history_tokens
                if self.raw_history_tokens
                else None
            ),
            "history_kv_compression": compression,
            "retained_history_units": self.retained_units,
            "dropped_history_units": self.dropped_units,
            "runtime_eviction_required": self.runtime_eviction_required,
            "runtime_eviction_available": self.runtime_eviction_available,
            "notes": self.notes or [],
        }


class HistoryKVCompressor:
    """Build BFCL request messages for history-only KV compression baselines.

    This class deliberately keeps System/Tools/Current turn outside the
    compression decision. Only completed history units before the current user
    query are compressed or dropped.
    """

    def __init__(self, runner: "HistoryKVBaselineRunner") -> None:
        self.runner = runner

    @property
    def method(self) -> str:
        return self.runner.history_kv_method

    def build(
        self,
        history_messages: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], HistoryKVDecision]:
        self.runner._last_runtime_history_kv_extract = None
        self.runner._last_physical_history_kv_eviction = None
        latest_query_index = _latest_user_query_index(history_messages)
        completed = list(history_messages[:latest_query_index])
        current = deepcopy(list(history_messages[latest_query_index:]))
        units = _history_units(completed)
        raw_tokens_by_unit = [_token_count(self.runner.tokenizer, unit) for unit in units]
        unit_raw_history_tokens = sum(raw_tokens_by_unit)
        canonical_history_tokens = self._canonical_history_token_count(completed)

        if self.method == "full":
            stats.original_history_tokens += canonical_history_tokens
            stats.effective_history_tokens += canonical_history_tokens
            stats.canonical_full_history_tokens += canonical_history_tokens
            stats.physical_history_kv_tokens += canonical_history_tokens
            return deepcopy(list(history_messages)), HistoryKVDecision(
                method=self.method,
                raw_history_tokens=canonical_history_tokens,
                active_history_kv_tokens=canonical_history_tokens,
                retained_units=list(range(len(units))),
                dropped_units=[],
            )

        if self.method == "c2kv":
            messages, active_tokens = self._build_c2kv_units(units, stats)
            stats.original_history_tokens += canonical_history_tokens
            stats.canonical_full_history_tokens += canonical_history_tokens
            messages.extend(current)
            return messages, HistoryKVDecision(
                method=self.method,
                raw_history_tokens=canonical_history_tokens,
                active_history_kv_tokens=active_tokens,
                retained_units=list(range(len(units))),
                dropped_units=[],
            )

        if self.method in {
            "streamingllm",
            "h2o",
            "snapkv",
            "snapkv_persistent",
            "pyramidkv",
            "kivi",
        }:
            if self.runner.runtime_history_kv_backend == "physical_eviction":
                # Retention=1 is a correctness identity test. Do not route it
                # through the multi-round eviction scheduler: even a no-op
                # round changes request bookkeeping and cannot prove that the
                # compressor preserves the ordinary Full serving path.
                target_tokens = self._budget(canonical_history_tokens)
                if target_tokens >= canonical_history_tokens:
                    stats.original_history_tokens += canonical_history_tokens
                    stats.canonical_full_history_tokens += canonical_history_tokens
                    stats.effective_history_tokens += canonical_history_tokens
                    stats.physical_history_kv_tokens += canonical_history_tokens
                    return deepcopy(list(history_messages)), HistoryKVDecision(
                        method=self.method,
                        raw_history_tokens=canonical_history_tokens,
                        active_history_kv_tokens=canonical_history_tokens,
                        retained_units=list(range(len(units))),
                        dropped_units=[],
                        runtime_eviction_required=False,
                        runtime_eviction_available=True,
                        notes=[
                            "retention=1 identity path: ordinary Full serving; "
                            "no physical eviction requested"
                        ],
                    )
                messages, active_tokens = self._build_physical_eviction_history_kv(
                    completed, current, canonical_history_tokens, stats
                )
                return messages, HistoryKVDecision(
                    method=self.method,
                    raw_history_tokens=canonical_history_tokens,
                    active_history_kv_tokens=active_tokens,
                    retained_units=[],
                    dropped_units=[],
                    runtime_eviction_required=True,
                    runtime_eviction_available=True,
                    notes=[
                        "physical history-KV baseline: SGLang full-prefills "
                        "completed history, physically compacts surviving KV "
                        "slots at the history/current boundary, and frees "
                        "dropped slots before current-query prefill"
                    ],
                )
            messages, active_tokens = self._build_runtime_history_kv(
                completed, current, canonical_history_tokens, stats
            )
            return messages, HistoryKVDecision(
                method=self.method,
                raw_history_tokens=canonical_history_tokens,
                active_history_kv_tokens=active_tokens,
                retained_units=[],
                dropped_units=[],
                runtime_eviction_required=True,
                runtime_eviction_available=True,
                notes=[
                    "runtime history-KV baseline: SGLang extracted full-causal "
                    "history KV, selected surviving token slots, stored them in "
                    "C2KVPool, and injected the compressed entry at generation"
                ],
            )

        if self.method == "snapkv_refresh" and self.runner.allow_client_fallback:
            messages, active_tokens, retained, dropped = self._build_refresh_proxy(
                units, raw_tokens_by_unit, current, stats
            )
            return messages, HistoryKVDecision(
                method=self.method,
                raw_history_tokens=unit_raw_history_tokens,
                active_history_kv_tokens=active_tokens,
                retained_units=retained,
                dropped_units=dropped,
                notes=[
                    "diagnostic client fallback: full textual history is "
                    "reselected each call; this is not SnapKV-persistent"
                ],
            )

        raise ValueError(f"unsupported history_kv_method={self.method!r}")

    def _budget(self, raw_history_tokens: int) -> int:
        if raw_history_tokens <= 0:
            return 0
        if self.runner.history_kv_retention_ratio is not None:
            return max(1, math.ceil(raw_history_tokens * self.runner.history_kv_retention_ratio))
        if self.runner.history_kv_target_compression:
            return max(1, math.ceil(raw_history_tokens / self.runner.history_kv_target_compression))
        return raw_history_tokens

    def _build_c2kv_units(
        self,
        units: Sequence[Sequence[dict[str, Any]]],
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int]:
        messages: list[dict[str, Any]] = []
        active_tokens = 0
        for unit in units:
            text = _render_history_unit(unit)
            full_tokens = _token_count(self.runner.tokenizer, [{"role": "user", "content": text}])
            record = self.runner._extract_history_unit(text, stats)
            if record.success and record.key_hash:
                gist_len = int(record.gist_len or record.original_seq_len or full_tokens)
                stats.effective_history_tokens += gist_len
                stats.physical_history_kv_tokens += gist_len
                stats.c2kv_gist_tokens += gist_len
                active_tokens += gist_len
                messages.append({"role": "user", "content": text, "c2kv_key_hash": record.key_hash})
            else:
                stats.effective_history_tokens += full_tokens
                stats.physical_history_kv_tokens += full_tokens
                active_tokens += full_tokens
                messages.append({"role": "user", "content": text})
        return messages, active_tokens

    def _build_recent_full_units(
        self,
        units: Sequence[Sequence[dict[str, Any]]],
        raw_tokens_by_unit: Sequence[int],
        current: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int, list[int], list[int]]:
        raw_history_tokens = sum(raw_tokens_by_unit)
        budget = self._budget(raw_history_tokens)
        retained: list[int] = []
        active = 0
        for index in range(len(units) - 1, -1, -1):
            unit_tokens = int(raw_tokens_by_unit[index])
            if active and active + unit_tokens > budget:
                break
            retained.append(index)
            active += unit_tokens
            if active >= budget:
                break
        retained = sorted(retained)
        retained_set = set(retained)
        dropped = [index for index in range(len(units)) if index not in retained_set]
        messages: list[dict[str, Any]] = []
        for index, unit in enumerate(units):
            full_tokens = int(raw_tokens_by_unit[index])
            stats.original_history_tokens += full_tokens
            stats.canonical_full_history_tokens += full_tokens
            if index in retained_set:
                stats.effective_history_tokens += full_tokens
                stats.physical_history_kv_tokens += full_tokens
                messages.extend(deepcopy(list(unit)))
        messages.extend(deepcopy(list(current)))
        return messages, active, retained, dropped

    def _build_refresh_proxy(
        self,
        units: Sequence[Sequence[dict[str, Any]]],
        raw_tokens_by_unit: Sequence[int],
        current: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int, list[int], list[int]]:
        # A low-risk diagnostic proxy for SnapKV-refresh until runtime
        # attention-score based selection is implemented in SGLang.
        return self._build_recent_full_units(units, raw_tokens_by_unit, current, stats)

    @staticmethod
    def _normalize_token_ids(encoded: Any) -> list[int]:
        if isinstance(encoded, Mapping) or (
            hasattr(encoded, "keys") and "input_ids" in set(encoded.keys())
        ):
            if "input_ids" not in encoded:
                raise RuntimeError(
                    "chat template tokenization returned a dict without input_ids"
                )
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if isinstance(encoded, Mapping) or (
            hasattr(encoded, "keys") and "input_ids" in set(encoded.keys())
        ):
            encoded = encoded["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        if encoded and isinstance(encoded[0], Mapping) and "input_ids" in encoded[0]:
            encoded = encoded[0]["input_ids"]
        return [int(x) for x in encoded]

    def _canonical_history_token_count(
        self,
        completed: Sequence[dict[str, Any]],
    ) -> int:
        if not completed:
            return 0
        tools = getattr(self.runner, "_active_tools", [])
        full_ids = self._chat_template_ids(completed, tools=tools)
        history_ids = self._chat_template_ids(completed, tools=None)
        span_start = self._find_subsequence(full_ids, history_ids)
        if span_start < 0:
            raise RuntimeError(
                "Cannot isolate completed-history token span inside the "
                "tools+history prompt. Refusing to mix history-token accounting "
                "with tool/system prompt tokens."
            )
        return len(history_ids)

    def _history_span_in_full_prompt(
        self,
        completed: Sequence[dict[str, Any]],
    ) -> tuple[list[int], int, int]:
        tools = getattr(self.runner, "_active_tools", [])
        full_ids = self._chat_template_ids(completed, tools=tools)
        history_ids = self._chat_template_ids(completed, tools=None)
        span_start = self._find_subsequence(full_ids, history_ids)
        if span_start < 0:
            raise RuntimeError(
                "Cannot isolate completed-history token span inside the "
                "tools+history prompt. Refusing physical history KV eviction."
            )
        return full_ids, span_start, span_start + len(history_ids)

    def _chat_template_ids(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None,
    ) -> list[int]:
        try:
            encoded = self.runner.tokenizer.apply_chat_template(
                list(messages),
                tools=list(tools or []),
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Runtime history-KV baselines require exact chat-template "
                f"tokenization with tools; failed with {type(exc).__name__}: {exc}"
            ) from exc
        return self._normalize_token_ids(encoded)

    @staticmethod
    def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle:
            return 0
        limit = len(haystack) - len(needle) + 1
        for start in range(max(0, limit)):
            if list(haystack[start : start + len(needle)]) == list(needle):
                return start
        return -1

    def _build_runtime_history_kv(
        self,
        completed: Sequence[dict[str, Any]],
        current: Sequence[dict[str, Any]],
        raw_history_tokens: int,
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int]:
        if not completed or raw_history_tokens <= 0:
            return deepcopy(list(current)), 0
        method = "snapkv_persistent" if self.method == "snapkv" else self.method
        tools = getattr(self.runner, "_active_tools", [])
        full_ids = self._chat_template_ids(completed, tools=tools)
        history_ids = self._chat_template_ids(completed, tools=None)
        span_start = self._find_subsequence(full_ids, history_ids)
        if span_start < 0:
            raise RuntimeError(
                "Cannot isolate completed-history token span inside the "
                "tools+history prompt. Refusing to compress tools/system tokens "
                "as history."
            )
        span_end = span_start + len(history_ids)
        span_tokens = span_end - span_start
        if span_tokens <= 0:
            return deepcopy(list(current)), 0
        target_tokens = self._budget(span_tokens)
        if method == "kivi":
            target_tokens = span_tokens
        repair_mode = f"history_kv_{method}"
        payload = {
            "input_ids": full_ids,
            "span_start": span_start,
            "span_end": span_end,
            "position_offset": 0,
            "repair_mode": repair_mode,
            "raw_kv_position_mode": "rotated",
            "extract_source": "model_prefill",
            "history_kv_method": method,
            "history_kv_target_tokens": target_tokens,
            "history_kv_recent_window": self.runner.history_kv_recent_window,
            "history_kv_kernel_size": self.runner.history_kv_kernel_size,
            "history_kv_pooling": self.runner.history_kv_pooling,
            "history_kv_h2o_recent_fraction": self.runner.history_kv_h2o_recent_fraction,
        }
        start = time.perf_counter()
        result = _post_json(
            self.runner.base_url,
            "/v1/c2kv/repair_extract",
            payload,
            self.runner.timeout,
        )
        elapsed = time.perf_counter() - start
        stats.extract_calls += 1
        stats.repair_extract_seconds += elapsed
        stats.extract_seconds += elapsed
        stats.repair_extract_recomputed_tokens += len(full_ids)
        stats.canonical_full_history_tokens += span_tokens
        stats.original_history_tokens += span_tokens
        if not result.get("success") or not result.get("key_hash"):
            raise RuntimeError(
                f"runtime history KV extract failed for {method}: {result.get('error')}"
            )
        stats.extract_success += 1
        selected = int(result.get("selected_token_count") or result.get("token_len") or 0)
        stats.effective_history_tokens += selected
        stats.physical_history_kv_tokens += selected
        stats.repair_kv_tokens += selected
        carrier = {
            "role": "user",
            "content": f"[runtime {method} compressed history kv]",
            "c2kv_repair_only_key_hashes": [result["key_hash"]],
            "c2kv_use_gist_projection": False,
        }
        self.runner._last_runtime_history_kv_extract = {
            "history_kv_method": method,
            "repair_mode": repair_mode,
            "span_start": span_start,
            "span_end": span_end,
            "full_prompt_tokens": len(full_ids),
            "history_span_tokens": span_tokens,
            "target_tokens": target_tokens,
            "selected_token_count": selected,
            "selected_relative_indices": result.get("selected_relative_indices"),
            "key_hash": result["key_hash"],
            "extract_seconds": elapsed,
        }
        if method == "kivi":
            residual = min(span_tokens, self.runner.kivi_residual_length)
            quantized = max(0, span_tokens - residual)
            bit_retention = (
                (quantized * self.runner.kivi_bits + residual * 16)
                / float(span_tokens * 16)
                if span_tokens
                else None
            )
            self.runner._last_runtime_history_kv_extract.update(
                {
                    "kivi_bits": self.runner.kivi_bits,
                    "kivi_group_size": self.runner.kivi_group_size,
                    "kivi_residual_length": self.runner.kivi_residual_length,
                    "estimated_history_kv_byte_retention": bit_retention,
                    "estimated_history_kv_byte_compression": (
                        1.0 / bit_retention if bit_retention else None
                    ),
                }
            )
        return [carrier, *deepcopy(list(current))], selected

    def _build_physical_eviction_history_kv(
        self,
        completed: Sequence[dict[str, Any]],
        current: Sequence[dict[str, Any]],
        canonical_history_tokens: int,
        stats: DriftStats,
    ) -> tuple[list[dict[str, Any]], int]:
        if not completed or canonical_history_tokens <= 0:
            return deepcopy(list(current)), 0
        method = "snapkv_persistent" if self.method == "snapkv" else self.method
        target_tokens = self._budget(canonical_history_tokens)

        stats.original_history_tokens += canonical_history_tokens
        stats.canonical_full_history_tokens += canonical_history_tokens
        stats.effective_history_tokens += target_tokens
        stats.physical_history_kv_tokens += target_tokens
        self.runner._last_runtime_history_kv_extract = {
            "history_kv_method": method,
            "extract_source": "physical_eviction",
            "history_span_tokens": canonical_history_tokens,
            "target_tokens": target_tokens,
        }
        self.runner._last_physical_history_kv_eviction = {
            "method": method,
            # Token offsets must be resolved by SGLang after its own OpenAI
            # chat-template rendering. The local tokenizer's output is useful
            # for accounting but is not a safe coordinate system for a server
            # request (tool-template variants may differ).
            "history_message_count": len(completed),
            "target_tokens": target_tokens,
            "retention_ratio": self.runner.history_kv_retention_ratio,
            "history_kv_recent_window": self.runner.history_kv_recent_window,
            "history_kv_kernel_size": self.runner.history_kv_kernel_size,
            "history_kv_pooling": self.runner.history_kv_pooling,
            "history_kv_h2o_recent_fraction": self.runner.history_kv_h2o_recent_fraction,
            "persistent_session": bool(
                self.runner.persistent_history_kv_session
            ),
        }
        return deepcopy(list(completed)) + deepcopy(list(current)), target_tokens


class HistoryKVBaselineRunner(HistoryDriftRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        args = deepcopy(args)
        args.mode = f"history_{args.history_kv_method}_closed_loop"
        super().__init__(args)
        self.history_kv_method = args.history_kv_method
        self.history_kv_retention_ratio = args.history_kv_retention_ratio
        self.history_kv_target_compression = args.history_kv_target_compression
        self.allow_client_fallback = args.allow_client_fallback
        self.strict_runtime_eviction = args.strict_runtime_eviction
        self.history_kv_recent_window = args.history_kv_recent_window
        self.history_kv_kernel_size = args.history_kv_kernel_size
        self.history_kv_pooling = args.history_kv_pooling
        self.history_kv_h2o_recent_fraction = args.history_kv_h2o_recent_fraction
        self.history_kv_pyramid_budget_scale = args.history_kv_pyramid_budget_scale
        self.kivi_bits = args.kivi_bits
        self.kivi_group_size = args.kivi_group_size
        self.kivi_residual_length = args.kivi_residual_length
        self.runtime_history_kv_backend = args.runtime_history_kv_backend
        if self.runtime_history_kv_backend != "repair_extract":
            raise RuntimeEvictionUnsupported(
                "physical history-KV eviction is disabled for the baseline "
                "benchmark; use repair_extract selection and reinjection"
            )
        self.persistent_history_kv_session = bool(
            args.persistent_history_kv_session
        )
        self._history_kv_compressor = HistoryKVCompressor(self)
        self._active_tools: list[dict[str, Any]] = []
        self._last_runtime_history_kv_extract: dict[str, Any] | None = None
        self._last_physical_history_kv_eviction: dict[str, Any] | None = None
        self._persistent_history_session_id: str | None = None

    def _persistent_session_enabled(self) -> bool:
        return bool(
            self.persistent_history_kv_session
            and self.runtime_history_kv_backend == "physical_eviction"
            and self.history_kv_method in RUNTIME_EVICTION_METHODS
        )

    def _open_persistent_history_session(self, sample_id: str) -> None:
        session_id = f"bfcl-history-{sample_id}-{uuid.uuid4().hex}"
        response = HTTP.post(
            f"{self.base_url.rstrip('/')}/open_session",
            json={
                "capacity_of_str_len": 0,
                "session_id": session_id,
                "streaming": True,
                "timeout": float(self.timeout),
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                "PERSISTENT_HISTORY_SESSION_OPEN_FAILED: " + response.text[:1000]
            )
        returned = response.json()
        if returned != session_id:
            raise RuntimeError(
                "PERSISTENT_HISTORY_SESSION_OPEN_INVALID_RESPONSE: "
                f"expected={session_id!r}, got={returned!r}"
            )
        self._persistent_history_session_id = session_id

    def _close_persistent_history_session(self) -> None:
        session_id = self._persistent_history_session_id
        self._persistent_history_session_id = None
        if not session_id:
            return
        response = HTTP.post(
            f"{self.base_url.rstrip('/')}/close_session",
            json={"session_id": session_id},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                "PERSISTENT_HISTORY_SESSION_CLOSE_FAILED: " + response.text[:1000]
            )

    def run_sample(self, test_case: dict[str, Any]) -> dict[str, Any]:
        stats = DriftStats(test_case["id"], self.mode, self.ratio)
        self._active_tools = _tool_payload(test_case["function"])
        if self._persistent_session_enabled():
            self._open_persistent_history_session(test_case["id"])
        try:
            result, metadata = self._run_sample_impl(test_case, stats)
        except Exception as exc:
            if self.strict_runtime_eviction and self.history_kv_method in RUNTIME_EVICTION_METHODS:
                raise
            result = f"Error during inference: {exc}"
            metadata = {"traceback": traceback.format_exc()}
            stats.errors.append(str(exc))
        finally:
            self._close_persistent_history_session()
            self._active_tools = []
        metadata["c2kv_drift_metrics"] = stats.as_dict()
        return {"id": test_case["id"], "result": result, **metadata}

    def _extract_history_unit(self, text: str, stats: DriftStats) -> ExtractRecord:
        return super()._extract_history_unit(text, stats)

    def _build_request_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
        stats: DriftStats,
    ) -> list[dict[str, Any]]:
        messages, decision = self._history_kv_compressor.build(history_messages, stats)
        self._last_history_kv_decision = decision.as_dict()
        physical_eviction = getattr(self, "_last_physical_history_kv_eviction", None)
        self._last_kv_memory_hint = {
            "full_equivalent_history_tokens": decision.raw_history_tokens,
            "active_history_kv_tokens": decision.active_history_kv_tokens,
            "active_full_raw_tokens": (
                decision.active_history_kv_tokens
                if self.history_kv_method in {"full", "snapkv_refresh"}
                else 0
            ),
            "active_c2kv_gist_tokens": (
                decision.active_history_kv_tokens if self.history_kv_method == "c2kv" else 0
            ),
            "history_kv_method": self.history_kv_method,
            "estimated": True,
        }
        if isinstance(physical_eviction, dict):
            self._last_kv_memory_hint["history_kv_eviction"] = deepcopy(physical_eviction)
        if self._persistent_history_session_id:
            self._last_kv_memory_hint["persistent_history_session"] = {
                "enabled": True,
            }
        runtime_extract = getattr(self, "_last_runtime_history_kv_extract", None)
        if isinstance(runtime_extract, dict):
            self._last_history_kv_decision["runtime_extract"] = deepcopy(runtime_extract)
            # The SGLang injection path adds the actual attention-visible
            # repair tokens into kv_memory_report.  Pre-filling active tokens
            # here double-counts runtime reinjection as both a client estimate
            # and a server-observed active KV entry.
            self._last_kv_memory_hint["active_history_kv_tokens"] = 0
            self._last_kv_memory_hint["active_full_raw_tokens"] = 0
            self._last_kv_memory_hint["active_c2kv_gist_tokens"] = 0
            self._last_kv_memory_hint["active_raw_repair_tokens"] = 0
            self._last_kv_memory_hint["estimated"] = False
        return messages


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_serializable(row), ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    if args.history_kv_method in UNSUPPORTED_RUNTIME_METHODS and args.strict_runtime_eviction:
        raise RuntimeEvictionUnsupported(
            f"{args.history_kv_method} is intentionally blocked in strict mode: "
            "the current runtime does not support layer-wise PyramidKV entries."
        )

    runner = HistoryKVBaselineRunner(args)
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
    for test_case in tqdm(
        entries,
        desc=f"{args.history_kv_method}:{args.category}",
        dynamic_ncols=True,
    ):
        row = runner.run_sample(deepcopy(test_case))
        runner.decoder.write(row, result_dir=result_dir, update_mode=False)
        details_rows.append(row)
        metric_rows.append(row.get("c2kv_drift_metrics", {}))

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(Path(args.details_path), details_rows)
    _write_jsonl(Path(args.metrics_path), metric_rows)
    summary = {
        "history_kv_method": args.history_kv_method,
        "category": args.category,
        "num_examples": len(details_rows),
        "strict_runtime_eviction": args.strict_runtime_eviction,
        "allow_client_fallback": args.allow_client_fallback,
        "history_kv_retention_ratio": args.history_kv_retention_ratio,
        "history_kv_target_compression": args.history_kv_target_compression,
        "history_kv_pyramid_budget_scale": args.history_kv_pyramid_budget_scale,
        "kivi_bits": args.kivi_bits,
        "kivi_group_size": args.kivi_group_size,
        "kivi_residual_length": args.kivi_residual_length,
        "runtime_history_kv_backend": args.runtime_history_kv_backend,
        "errors": sum(
            1
            for row in details_rows
            if str(row.get("result", "")).startswith("Error during inference")
        ),
        "chat_calls": sum(int(row.get("chat_calls") or 0) for row in metric_rows),
        "extract_calls": sum(int(row.get("extract_calls") or 0) for row in metric_rows),
        "extract_success": sum(int(row.get("extract_success") or 0) for row in metric_rows),
        "canonical_full_history_tokens": sum(
            int(row.get("canonical_full_history_tokens") or 0) for row in metric_rows
        ),
        "physical_history_kv_tokens": sum(
            int(row.get("physical_history_kv_tokens") or 0) for row in metric_rows
        ),
    }
    summary["extract_success_rate"] = (
        summary["extract_success"] / summary["extract_calls"]
        if summary["extract_calls"]
        else None
    )
    summary["estimated_weighted_history_kv_compression"] = (
        summary["canonical_full_history_tokens"] / summary["physical_history_kv_tokens"]
        if summary["physical_history_kv_tokens"]
        else None
    )
    summary["estimated_weighted_history_kv_retention_ratio"] = (
        summary["physical_history_kv_tokens"] / summary["canonical_full_history_tokens"]
        if summary["canonical_full_history_tokens"]
        else None
    )
    Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-kv-method", choices=sorted(HISTORY_KV_METHODS), required=True)
    parser.add_argument("--history-kv-retention-ratio", type=float, default=0.312)
    parser.add_argument("--history-kv-target-compression", type=float, default=0.0)
    parser.add_argument("--history-kv-recent-window", type=int, default=64)
    parser.add_argument("--history-kv-kernel-size", type=int, default=5)
    parser.add_argument("--history-kv-pooling", default="avgpool")
    parser.add_argument("--history-kv-h2o-recent-fraction", type=float, default=0.5)
    parser.add_argument(
        "--history-kv-pyramid-budget-scale",
        type=float,
        default=float(os.environ.get("PYRAMIDKV_BUDGET_SCALE", "0.66")),
        help=(
            "Internal PyramidKV per-layer budget scale used before union. "
            "The default calibrates the union-based approximation to about "
            "0.31-0.32 retained history tokens when the shared target is 0.312."
        ),
    )
    parser.add_argument("--kivi-bits", type=int, default=2)
    parser.add_argument("--kivi-group-size", type=int, default=32)
    parser.add_argument("--kivi-residual-length", type=int, default=32)
    parser.add_argument(
        "--runtime-history-kv-backend",
        choices=["repair_extract", "physical_eviction"],
        default="repair_extract",
    )
    parser.add_argument("--strict-runtime-eviction", action="store_true")
    parser.add_argument("--persistent-history-kv-session", action="store_true")
    parser.add_argument("--allow-client-fallback", action="store_true")
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
