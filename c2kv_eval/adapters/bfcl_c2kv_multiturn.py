from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import requests
from overrides import override
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.utils import load_dataset_entry, sort_file_content_by_id, sort_key


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FC"
DEFAULT_TOKENIZER_PATH = "/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507"
TOOL_CALL_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "Function signatures are provided in the preceding tool-definition messages. "
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)


HTTP = requests.Session()
HTTP.trust_env = False


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _as_tool_list(tool_definition: Any) -> list[dict[str, Any]]:
    if isinstance(tool_definition, str):
        try:
            tool_definition = json.loads(tool_definition)
        except json.JSONDecodeError:
            return []
    if isinstance(tool_definition, dict):
        if isinstance(tool_definition.get("tools"), list):
            tool_definition = tool_definition["tools"]
        elif isinstance(tool_definition.get("functions"), list):
            tool_definition = tool_definition["functions"]
        else:
            tool_definition = [tool_definition]
    if not isinstance(tool_definition, list):
        return []
    return [item for item in tool_definition if isinstance(item, dict)]


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    return str(
        function.get("name")
        or tool.get("name")
        or tool.get("tool_name")
        or tool.get("function_name")
        or ""
    )


def _tool_search_text(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    fields = [
        _tool_name(tool),
        function.get("description", ""),
        tool.get("description", ""),
        function.get("parameters", ""),
        tool.get("parameters", ""),
        tool.get("input_schema", ""),
        tool.get("schema", ""),
    ]
    return " ".join(
        item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        for item in fields
        if item
    )


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _query_text(messages: Sequence[dict[str, Any]], scope: str) -> str:
    if scope == "all":
        return "\n".join(_message_text(message) for message in messages)
    for message in reversed(messages):
        if message.get("role") == "user":
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _rank_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _rank_tools(tools: Sequence[dict[str, Any]], query: str) -> list[int]:
    query_tokens = set(_rank_tokens(query))
    if not query_tokens:
        return list(range(len(tools)))
    scored = []
    for index, tool in enumerate(tools):
        name_tokens = set(_rank_tokens(_tool_name(tool)))
        text_tokens = set(_rank_tokens(_tool_search_text(tool)))
        score = 4.0 * len(query_tokens & name_tokens) + float(
            len(query_tokens & text_tokens)
        )
        scored.append((-score, index))
    scored.sort()
    return [index for _, index in scored]


def _render_tool_definition(tools: Sequence[dict[str, Any]]) -> str:
    return json.dumps(list(tools), ensure_ascii=False, separators=(",", ":"))


def _tool_documents(tools: Sequence[dict[str, Any]], document_mode: str) -> list[str]:
    if document_mode == "per_tool":
        return ["Tool definition:\n" + _render_tool_definition([tool]) for tool in tools]
    raise ValueError(
        "BFCL C2KV evaluation only supports per-tool document messages; "
        f"got tool_document_mode={document_mode!r}."
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
            + json.dumps(
                {"name": name, "arguments": arguments},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n</tool_call>"
        )
    return "\n".join(chunks)


@dataclass
class ExtractRecord:
    doc_idx: int
    success: bool
    key_hash: str | None = None
    gist_len: int | None = None
    original_seq_len: int | None = None
    error: str | None = None


@dataclass
class SampleStats:
    sample_id: str
    mode: str
    ratio: int
    hybrid_top_k: int
    extract_calls: int = 0
    extract_success: int = 0
    extract_seconds: float = 0.0
    total_original_tool_tokens: int = 0
    total_effective_tool_tokens: int = 0
    chat_calls: int = 0
    chat_seconds: float = 0.0
    total_seconds: float = 0.0
    num_tools: int = 0
    avg_full_tools_accum: int = 0
    avg_compressed_tools_accum: int = 0
    top_tool_names_by_call: list[list[str]] = field(default_factory=list)
    extract_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        ratio = (
            self.total_original_tool_tokens / self.total_effective_tool_tokens
            if self.total_effective_tool_tokens
            else 1.0
        )
        return {
            "id": self.sample_id,
            "mode": self.mode,
            "ratio": self.ratio,
            "hybrid_top_k": self.hybrid_top_k if self.mode == "hybrid" else None,
            "num_tools": self.num_tools,
            "extract_calls": self.extract_calls,
            "extract_success": self.extract_success,
            "extract_success_rate": (
                self.extract_success / self.extract_calls
                if self.extract_calls
                else None
            ),
            "extract_seconds": round(self.extract_seconds, 4),
            "chat_calls": self.chat_calls,
            "chat_seconds": round(self.chat_seconds, 4),
            "avg_chat_seconds": (
                round(self.chat_seconds / self.chat_calls, 4)
                if self.chat_calls
                else None
            ),
            "total_seconds": round(self.total_seconds, 4),
            "tool_original_tokens": self.total_original_tool_tokens,
            "tool_effective_tokens": self.total_effective_tool_tokens,
            "compression_ratio": round(ratio, 4),
            "avg_full_tools": (
                self.avg_full_tools_accum / self.chat_calls if self.chat_calls else None
            ),
            "avg_compressed_tools": (
                self.avg_compressed_tools_accum / self.chat_calls
                if self.chat_calls
                else None
            ),
            "top_tool_names_by_call": self.top_tool_names_by_call,
            "extract_records": self.extract_records,
            "errors": self.errors,
        }


class C2KVQwenFCHandler(QwenFCHandler):
    def __init__(
        self,
        *,
        mode: str,
        base_url: str,
        served_model_name: str,
        ratio: int,
        hybrid_top_k: int,
        router_scope: str,
        document_mode: str,
        tokenizer_path: str,
        request_timeout: int,
        max_completion_tokens: int,
        temperature: float,
    ) -> None:
        super().__init__(
            model_name=served_model_name,
            temperature=temperature,
            registry_name=served_model_name,
            is_fc_model=True,
        )
        self.mode = mode
        self.base_url = base_url.rstrip("/")
        self.served_model_name = served_model_name
        self.ratio = ratio
        self.hybrid_top_k = hybrid_top_k
        self.router_scope = router_scope
        if document_mode != "per_tool":
            raise ValueError(
                "BFCL C2KV evaluation uses a unified per-tool document interface; "
                f"got tool_document_mode={document_mode!r}."
            )
        self.document_mode = document_mode
        self.request_timeout = request_timeout
        self.max_completion_tokens = max_completion_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        try:
            config = AutoConfig.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            self.max_context_length = int(getattr(config, "max_position_embeddings", 32768))
        except Exception:
            self.max_context_length = 32768
        self.model_path_or_id = served_model_name
        self._tools: list[dict[str, Any]] = []
        self._docs: list[str] = []
        self._extract_records: list[ExtractRecord] = []
        self._current_stats: SampleStats | None = None
        self._current_turn_query = ""
        self._sample_start = 0.0

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        self._tools = _as_tool_list(test_entry["function"])
        self._docs = _tool_documents(self._tools, self.document_mode)
        self._extract_records = []
        self._current_turn_query = ""
        self._sample_start = time.perf_counter()
        self._current_stats = SampleStats(
            sample_id=test_entry["id"],
            mode=self.mode,
            ratio=self.ratio,
            hybrid_top_k=self.hybrid_top_k,
            num_tools=len(self._tools),
        )
        if self.mode in {"c2kv", "hybrid"}:
            self._ensure_extract_cache()
        return {"message": [], "function": self._tools}

    def _set_current_turn_query(self, turn_messages: Sequence[dict[str, Any]]) -> None:
        for message in turn_messages:
            if message.get("role") == "user":
                content = _message_text(message)
                if (
                    not content.startswith("<tool_response>")
                    and not content.endswith("</tool_response>")
                ):
                    self._current_turn_query = content
                    return
        self._current_turn_query = _query_text(turn_messages, self.router_scope)

    def _ensure_extract_cache(self) -> None:
        if self._extract_records:
            return
        assert self._current_stats is not None
        for doc_idx, doc in enumerate(self._docs):
            start = time.perf_counter()
            try:
                result = _post_json(
                    self.base_url,
                    "/v1/c2kv/extract",
                    {
                        "text": doc,
                        "compression_ratio": self.ratio,
                        "role": "user",
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    self.request_timeout,
                )
                success = bool(result.get("success") and result.get("key_hash"))
                record = ExtractRecord(
                    doc_idx=doc_idx,
                    success=success,
                    key_hash=result.get("key_hash"),
                    gist_len=result.get("gist_len"),
                    original_seq_len=result.get("original_seq_len"),
                    error=result.get("error"),
                )
            except Exception as exc:
                record = ExtractRecord(doc_idx=doc_idx, success=False, error=str(exc))
                self._current_stats.errors.append(f"extract[{doc_idx}]: {exc}")
            self._current_stats.extract_seconds += time.perf_counter() - start
            self._current_stats.extract_calls += 1
            if record.success:
                self._current_stats.extract_success += 1
            self._current_stats.extract_records.append(record.__dict__.copy())
            self._extract_records.append(record)

    def _full_doc_messages(
        self,
    ) -> tuple[list[dict[str, Any]], int, int, int, int, list[str]]:
        docs = self._docs
        tokens = sum(
            _token_count(self.tokenizer, [{"role": "user", "content": doc}])
            for doc in docs
        )
        return (
            [{"role": "user", "content": doc} for doc in docs],
            tokens,
            tokens,
            len(self._tools),
            0,
            [],
        )

    def _c2kv_doc_messages(
        self,
    ) -> tuple[list[dict[str, Any]], int, int, int, int, list[str]]:
        self._ensure_extract_cache()
        messages = []
        original = 0
        effective = 0
        for doc, record in zip(self._docs, self._extract_records):
            if record.success and record.key_hash:
                messages.append(
                    {"role": "user", "content": doc, "c2kv_key_hash": record.key_hash}
                )
                original += int(record.original_seq_len or 0)
                effective += int(record.gist_len or record.original_seq_len or 0)
            else:
                messages.append({"role": "user", "content": doc})
                fallback_tokens = _token_count(self.tokenizer, [{"role": "user", "content": doc}])
                original += fallback_tokens
                effective += fallback_tokens
        return messages, original, effective, 0, len(messages), []

    def _hybrid_doc_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int, int, int, list[str]]:
        self._ensure_extract_cache()
        query = self._current_turn_query or _query_text(history_messages, self.router_scope)
        ranked = _rank_tools(self._tools, query)
        top_indices = set(ranked[: max(0, self.hybrid_top_k)])
        top_tools = [tool for index, tool in enumerate(self._tools) if index in top_indices]
        top_names = [_tool_name(tool) for tool in top_tools]
        messages = []
        original = 0
        effective = 0
        compressed_count = 0
        for index, (doc, record) in enumerate(zip(self._docs, self._extract_records)):
            if index in top_indices:
                messages.append({"role": "user", "content": doc})
                full_tokens = _token_count(
                    self.tokenizer, [{"role": "user", "content": doc}]
                )
                original += full_tokens
                effective += full_tokens
                continue
            compressed_count += 1
            if record.success and record.key_hash:
                messages.append(
                    {"role": "user", "content": doc, "c2kv_key_hash": record.key_hash}
                )
                original += int(record.original_seq_len or 0)
                effective += int(record.gist_len or record.original_seq_len or 0)
            else:
                messages.append({"role": "user", "content": doc})
                fallback_tokens = _token_count(self.tokenizer, [{"role": "user", "content": doc}])
                original += fallback_tokens
                effective += fallback_tokens
        return messages, original, effective, len(top_tools), compressed_count, top_names

    def _build_messages(
        self,
        history_messages: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int, int, int, int, list[str]]:
        if self.mode == "full":
            doc_messages, original, effective, full_tools, compressed_tools, top_names = (
                self._full_doc_messages()
            )
        elif self.mode == "c2kv":
            doc_messages, original, effective, full_tools, compressed_tools, top_names = (
                self._c2kv_doc_messages()
            )
        elif self.mode == "hybrid":
            doc_messages, original, effective, full_tools, compressed_tools, top_names = (
                self._hybrid_doc_messages(history_messages)
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        messages: list[dict[str, Any]] = []
        messages.append({"role": "system", "content": TOOL_CALL_SYSTEM_PROMPT})
        messages.extend(doc_messages)
        messages.extend(deepcopy(list(history_messages)))
        return messages, original, effective, full_tools, compressed_tools, top_names

    @override
    def _query_prompting(self, inference_data: dict):
        assert self._current_stats is not None
        history_messages = inference_data["message"]
        messages, original, effective, full_tools, compressed_tools, top_names = (
            self._build_messages(history_messages)
        )
        inference_data["inference_input_log"] = {
            "mode": self.mode,
            "messages": messages,
            "request_tools": [],
            "num_tools": len(self._tools),
            "full_tools": full_tools,
            "compressed_tools": compressed_tools,
            "top_tool_names": top_names,
        }
        prompt_tokens = _token_count(self.tokenizer, messages)
        max_tokens = min(self.max_completion_tokens, 4096)
        if self.max_context_length > 0 and prompt_tokens + 2 < self.max_context_length:
            max_tokens = min(max_tokens, self.max_context_length - prompt_tokens - 2)
        max_tokens = max(1, max_tokens)
        payload = {
            "model": self.served_model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.perf_counter()
        data = _post_json(self.base_url, "/v1/chat/completions", payload, self.request_timeout)
        elapsed = time.perf_counter() - start
        message = data.get("choices", [{}])[0].get("message", {}) or {}
        text = message.get("content") or ""
        if not text and message.get("tool_calls"):
            text = _tool_calls_to_text(message.get("tool_calls"))
        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        output_tokens = int(
            usage.get("completion_tokens")
            or len(self.tokenizer.encode(text, add_special_tokens=False))
        )
        self._current_stats.chat_calls += 1
        self._current_stats.chat_seconds += elapsed
        self._current_stats.total_original_tool_tokens += original
        self._current_stats.total_effective_tool_tokens += effective
        self._current_stats.avg_full_tools_accum += full_tools
        self._current_stats.avg_compressed_tools_accum += compressed_tools
        if top_names:
            self._current_stats.top_tool_names_by_call.append(top_names)
        return (
            SimpleNamespace(
                choices=[SimpleNamespace(text=text)],
                usage=SimpleNamespace(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                ),
            ),
            elapsed,
        )

    @override
    def add_first_turn_message_prompting(
        self,
        inference_data: dict,
        first_turn_message: list[dict],
    ) -> dict:
        self._set_current_turn_query(first_turn_message)
        return super().add_first_turn_message_prompting(
            inference_data,
            first_turn_message,
        )

    @override
    def _add_next_turn_user_message_prompting(
        self,
        inference_data: dict,
        user_message: list[dict],
    ) -> dict:
        self._set_current_turn_query(user_message)
        return super()._add_next_turn_user_message_prompting(
            inference_data,
            user_message,
        )

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        data = super()._parse_query_response_prompting(api_response)
        # Keep the multi-turn history template-neutral. SGLang's HF chat template
        # may reject OpenAI-style assistant.tool_calls on follow-up requests.
        data["model_responses_message_for_chat_history"] = {
            "role": "assistant",
            "content": data["model_responses"],
        }
        return data

    @override
    def _add_assistant_message_prompting(
        self,
        inference_data: dict,
        model_response_data: dict,
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        if not execution_results:
            return inference_data
        content = "\n".join(
            f"<tool_response>\n{execution_result}\n</tool_response>"
            for execution_result in execution_results
        )
        inference_data["message"].append({"role": "user", "content": content})
        return inference_data

    def finish_sample_stats(self) -> dict[str, Any]:
        if self._current_stats is None:
            return {}
        self._current_stats.total_seconds = time.perf_counter() - self._sample_start
        return self._current_stats.as_dict()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_one(handler: C2KVQwenFCHandler, test_case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        result, metadata = handler.inference(
            deepcopy(test_case),
            include_input_log=args.include_input_log,
            exclude_state_log=args.exclude_state_log,
        )
    except Exception as exc:
        result = f"Error during inference: {exc}"
        metadata = {"traceback": traceback.format_exc()}
    c2kv_metrics = handler.finish_sample_stats()
    metadata["c2kv_metrics"] = c2kv_metrics
    return {"id": test_case["id"], "result": result, **metadata}


def run(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    config = MODEL_CONFIG_MAPPING[args.model]
    handler = C2KVQwenFCHandler(
        mode=args.mode,
        base_url=args.base_url,
        served_model_name=args.served_model_name,
        ratio=args.ratio,
        hybrid_top_k=args.hybrid_top_k,
        router_scope=args.router_scope,
        document_mode=args.tool_document_mode,
        tokenizer_path=args.tokenizer_path,
        request_timeout=args.timeout,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )
    handler.registry_name = args.model
    handler.registry_dir_name = args.model.replace("/", "_")
    handler.is_fc_model = config.is_fc_model
    handler.model_name = config.model_name
    handler.model_name_underline_replaced = (
        config.model_name.replace("/", "_").replace("-", "_").replace(".", "_")
    )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    details_path = Path(args.details_path)
    entries = load_dataset_entry(args.category)
    entries = [entry for entry in entries if entry["id"].startswith(args.category)]
    entries = sorted(entries, key=sort_key)
    if args.max_examples is not None:
        entries = entries[: args.max_examples]

    rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    for test_case in tqdm(entries, desc=f"{args.mode}:{args.category}", dynamic_ncols=True):
        row = _run_one(handler, test_case, args)
        handler.write(row, result_dir=result_dir, update_mode=False)
        rows.append(row)
        metrics_rows.append(row.get("c2kv_metrics", {}))

    for result_json in result_dir.rglob("*_result.json"):
        sort_file_content_by_id(result_json)
    _write_jsonl(details_path, rows)
    _write_jsonl(Path(args.metrics_path), metrics_rows)

    summary = {
        "mode": args.mode,
        "category": args.category,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "base_url": args.base_url,
        "num_examples": len(rows),
        "errors": sum(1 for row in rows if str(row.get("result", "")).startswith("Error during inference")),
        "extract_calls": sum(int(row.get("extract_calls") or 0) for row in metrics_rows),
        "extract_success": sum(int(row.get("extract_success") or 0) for row in metrics_rows),
        "avg_tools": (
            sum(int(row.get("num_tools") or 0) for row in metrics_rows) / len(metrics_rows)
            if metrics_rows
            else 0.0
        ),
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
    parser.add_argument("--mode", choices=["full", "c2kv", "hybrid"], required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--details-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    parser.add_argument("--router-scope", choices=["last_user", "all"], default="last_user")
    parser.add_argument("--tool-document-mode", choices=["per_tool", "full"], default="per_tool")
    parser.add_argument("--timeout", type=int, default=72000)
    parser.add_argument("--max-completion-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-input-log", action="store_true")
    parser.add_argument("--exclude-state-log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
