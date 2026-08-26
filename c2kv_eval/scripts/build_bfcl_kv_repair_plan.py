from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from bfcl_eval.utils import load_dataset_entry, sort_key

from c2kv_eval.adapters.bfcl_history_drift import (
    DEFAULT_TOKENIZER_PATH,
    _history_units,
    _render_history_unit,
    _tool_payload,
)
from c2kv_eval.adapters.bfcl_history_kv_repair import KVRepairRunner


def _load_share_plan_module(path: str) -> Any:
    module_path = Path(path)
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("d_sham_plan", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import d_sham_plan module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_ids(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]


def build_doc_table(
    *,
    category: str,
    ids: Sequence[str],
    tokenizer_path: str,
) -> dict[str, Any]:
    runner_args = argparse.Namespace(
        arm="c2kv",
        plan_path="",
        neutral_corpus_path="",
        mode="history_c2kv4_closed_loop",
        ratio=4,
        compression_ratio=4,
        base_url="http://127.0.0.1:1",
        timeout=1,
        model="Qwen/Qwen3-4B-Instruct-2507-FC",
        served_model_name="Qwen/Qwen3-4B-Instruct-2507-FC",
        tokenizer_path=tokenizer_path,
        reference_details_path="",
        recent_full_units=2,
        max_completion_tokens=1,
        temperature=0.0,
        repair_trigger="oracle",
    )
    runner = KVRepairRunner(runner_args)
    selected = set(ids)
    entries = [
        entry
        for entry in sorted(load_dataset_entry(category), key=sort_key)
        if entry["id"] in selected
    ]
    per_qid: dict[str, Any] = {}
    missing = sorted(selected - {entry["id"] for entry in entries})
    for entry in entries:
        messages: list[dict[str, Any]] = []
        for current_turn_message in entry["question"]:
            messages.extend(current_turn_message)
        # The plan is history-unit based. Use all frozen conversation units in
        # the episode so every repair step can use the same per-qid target.
        units = _history_units(messages)
        if not units:
            per_qid[entry["id"]] = {"n_docs": 0, "doc_lens": []}
            continue
        runner._active_tools = _tool_payload(entry["function"])
        texts = [_render_history_unit(unit) for unit in units]
        doc_ids = [runner._unit_token_ids(text) for text in texts]
        _, starts, ends = runner._full_history_doc_layout(texts, doc_ids)
        doc_lens = [end - start for start, end in zip(starts, ends)]
        per_qid[entry["id"]] = {
            "session_id": entry["id"],
            "n_docs": len(doc_lens),
            "doc_lens": doc_lens,
        }
    total_tokens = sum(
        sum(int(x) for x in row.get("doc_lens", []))
        for row in per_qid.values()
    )
    return {
        "description": "BFCL history-unit doc lengths for D-KV repair plan.",
        "category": category,
        "n_qids": len(per_qid),
        "missing_qids": missing,
        "doc_table_total_tokens": total_tokens,
        "per_qid": per_qid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--ids-path", required=True)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--share-plan-module",
        default="/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_sham_plan.py",
    )
    parser.add_argument(
        "--neutral-corpus",
        default="/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_neutral_corpus.txt",
    )
    parser.add_argument(
        "--out",
        default="/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_sham_plan.json",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()

    ids = _read_ids(args.ids_path)
    doc_table = build_doc_table(
        category=args.category,
        ids=ids,
        tokenizer_path=args.tokenizer_path,
    )
    module = _load_share_plan_module(args.share_plan_module)
    tokenizer = module._load_tokenizer(args.tokenizer_path)
    corpus_text = Path(args.neutral_corpus).read_text(encoding="utf-8")
    corpus_ids = list(tokenizer(corpus_text, add_special_tokens=False)["input_ids"])
    plan = module.build_plan(
        doc_table["per_qid"],
        ids,
        corpus_ids,
        lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=False),
        seed=args.seed,
        header={
            "description": (
                "BFCL stable subset D-KV repair plan. Docs are completed "
                "history units rendered with the BFCL/SGLang history adapter."
            ),
            "category": args.category,
            "ids_path": args.ids_path,
            "neutral_corpus": args.neutral_corpus,
            "neutral_corpus_tokens": len(corpus_ids),
            "doc_table_total_tokens": doc_table["doc_table_total_tokens"],
            "plan_build_tokenization_tokens": (
                doc_table["doc_table_total_tokens"] + len(corpus_ids)
            ),
            "tokenizer": args.tokenizer_path,
            "doc_table": doc_table,
        },
    )
    if plan.get("missing_qids") or plan.get("degenerate_qids"):
        raise RuntimeError(
            "Generated incomplete repair plan: "
            f"missing={plan.get('missing_qids')}, "
            f"degenerate={plan.get('degenerate_qids')}"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    print(
        json.dumps(
            {
                "n_qids": plan.get("n_qids"),
                "typed_tokens_total": (plan.get("budget") or {}).get(
                    "typed_tokens_total"
                ),
                "sham_tokens_total": (plan.get("budget") or {}).get(
                    "sham_tokens_total"
                ),
                "budget_gate_passed": (plan.get("budget") or {}).get(
                    "gate_passed"
                ),
                "neutrality_gate_passed": (plan.get("neutrality") or {}).get(
                    "gate_passed"
                ),
                "doc_table_total_tokens": plan.get("doc_table_total_tokens"),
                "neutral_corpus_tokens": plan.get("neutral_corpus_tokens"),
                "plan_build_tokenization_tokens": plan.get(
                    "plan_build_tokenization_tokens"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
