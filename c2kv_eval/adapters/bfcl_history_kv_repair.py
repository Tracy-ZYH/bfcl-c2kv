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
    _history_units,
    _latest_user_query_index,
    _post_json,
    _render_history_unit,
    _token_count,
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
        self.require_plan = bool(args.plan_path)
        self.plan = self._load_plan(args.plan_path)
        self.neutral_token_ids = self._load_neutral_tokens(args.neutral_corpus_path)

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
        stats.extract_seconds += time.perf_counter() - start
        stats.extract_calls += 1
        if result.get("success"):
            stats.extract_success += 1
        else:
            raise RuntimeError(
                f"repair_extract failed for {repair_mode}: {result.get('error')}"
            )
        return result

    def _plan_for(self, sample_id: str, num_docs: int, doc_lens: list[int]) -> dict[str, Any]:
        plan = self.plan.get(sample_id) or self.plan.get(str(sample_id))
        if plan is None:
            if self.require_plan and self.arm in {
                "d_sham_neutral",
                "d_corr",
                "d_corr_recompute",
                "d_corr_all",
            }:
                raise RuntimeError(f"Missing D-KV repair plan for qid={sample_id}")
            k_star = (num_docs - 1) // 2
            return {
                "k_star": k_star,
                "span_len": doc_lens[k_star],
                "sham_token_ids": [],
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
            return deepcopy(list(history_messages))

        units = _history_units(completed)
        if not units:
            return deepcopy(list(history_messages))

        texts = [_render_history_unit(unit) for unit in units]
        doc_ids = [self._unit_token_ids(text) for text in texts]
        doc_lens = [len(ids) for ids in doc_ids]
        starts = []
        cursor = 0
        for length in doc_lens:
            starts.append(cursor)
            cursor += length

        sample_id = getattr(stats, "sample_id", "") or getattr(stats, "id", "")
        plan = self._plan_for(sample_id, len(units), doc_lens)
        k_star = int(plan.get("k_star", (len(units) - 1) // 2))

        gist_records: list[ExtractRecord | None] = []
        messages: list[dict[str, Any]] = []

        def should_compress_doc(index: int) -> bool:
            if self.arm == "d_corr_recompute" and index > k_star:
                return False
            return True

        for index, (unit, text, ids) in enumerate(zip(units, texts, doc_ids)):
            if not should_compress_doc(index):
                full_tokens = _token_count(self.tokenizer, unit)
                stats.original_history_tokens += full_tokens
                stats.effective_history_tokens += full_tokens
                messages.extend(deepcopy(unit))
                gist_records.append(None)
                continue

            full_tokens = len(ids)
            record = self._extract_history_unit(text, stats)
            stats.original_history_tokens += int(record.original_seq_len or full_tokens)
            if not (record.success and record.key_hash):
                raise RuntimeError(f"C2KV extract failed in arm={self.arm}: {record.error}")
            stats.effective_history_tokens += int(
                record.gist_len or record.original_seq_len or full_tokens
            )
            messages.append(
                {"role": "user", "content": text, "c2kv_key_hash": record.key_hash}
            )
            gist_records.append(record)

        repair_keys: list[str] = []
        repair_tokens = 0
        if self.arm == "d_sham_mech":
            repair_keys = []
        elif self.arm == "d_sham_neutral":
            span_len = doc_lens[k_star]
            neutral_ids = self._neutral_ids_for(plan, span_len)
            repair = self._extract_repair(
                input_ids=neutral_ids,
                span_start=0,
                span_end=span_len,
                position_offset=starts[k_star],
                repair_mode=self.arm,
                source_doc_index=k_star,
                stats=stats,
            )
            repair_keys.append(repair["key_hash"])
            repair_tokens += int(repair["token_len"])
        elif self.arm in {"d_corr", "d_corr_recompute"}:
            prefix_ids = [token for ids in doc_ids[: k_star + 1] for token in ids]
            repair = self._extract_repair(
                input_ids=prefix_ids,
                span_start=starts[k_star],
                span_end=starts[k_star] + doc_lens[k_star],
                position_offset=0,
                repair_mode=self.arm,
                source_doc_index=k_star,
                stats=stats,
            )
            repair_keys.append(repair["key_hash"])
            repair_tokens += int(repair["token_len"])
        elif self.arm == "d_corr_all":
            for index, ids in enumerate(doc_ids):
                prefix_ids = [token for part in doc_ids[: index + 1] for token in part]
                repair = self._extract_repair(
                    input_ids=prefix_ids,
                    span_start=starts[index],
                    span_end=starts[index] + doc_lens[index],
                    position_offset=0,
                    repair_mode=self.arm,
                    source_doc_index=index,
                    stats=stats,
                )
                repair_keys.append(repair["key_hash"])
                repair_tokens += int(repair["token_len"])

        if repair_keys:
            attach_index = k_star if self.arm == "d_corr_recompute" else len(messages) - 1
            compressed_seen = -1
            target_message = None
            for message in messages:
                if message.get("c2kv_key_hash"):
                    compressed_seen += 1
                    if compressed_seen == attach_index or self.arm != "d_corr_recompute":
                        target_message = message
            if target_message is None:
                raise RuntimeError(f"Cannot attach repair keys for arm={self.arm}")
            target_message["c2kv_repair_key_hashes"] = repair_keys
            stats.effective_history_tokens += repair_tokens

        messages.extend(current)
        return messages


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
        "history_original_tokens": sum(
            int(row.get("history_original_tokens") or 0) for row in metric_rows
        ),
        "history_effective_tokens": sum(
            int(row.get("history_effective_tokens") or 0) for row in metric_rows
        ),
    }
    summary["extract_success_rate"] = (
        summary["extract_success"] / summary["extract_calls"]
        if summary["extract_calls"]
        else None
    )
    summary["history_kv_compression"] = (
        summary["history_original_tokens"] / summary["history_effective_tokens"]
        if summary["history_effective_tokens"]
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
        "--neutral-corpus-path",
        default="/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_neutral_corpus.txt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
