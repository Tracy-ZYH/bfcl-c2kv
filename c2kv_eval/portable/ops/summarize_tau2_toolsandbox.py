#!/usr/bin/env python3
"""Build one compact comparison table from portable tau2/ToolSandbox runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METHODS = {
    "full": ("Full", "full", None),
    "c2kv4": ("C2KV-4x", "c2kv4", None),
    "c2kv8": ("C2KV-8x", "c2kv", None),
    "append_first": ("Append-First", "c2kv4", "append"),
    "replace_first": ("Replace-First", "c2kv4", "replace"),
    "full_kv_retry": ("Full-KV Retry", "c2kv4", "retry_full"),
    "streamingllm_r312": (
        "StreamingLLM-r312", "history_kv_streamingllm_r312", None),
    "h2o_r312": ("H2O-r312", "history_kv_h2o_r312", None),
    "snapkv_r312": ("SnapKV-r312", "history_kv_snapkv_r312", None),
    "pyramidkv_r312": (
        "PyramidKV-r312", "history_kv_pyramidkv_r312", None),
}


def _mean(values: list[Any]) -> float | None:
    picked = [float(value) for value in values
              if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(picked) / len(picked) if picked else None


def _request_rows(path: str) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("status") == "ok":
            rows.append(value)
    return rows


def _legacy_toolsandbox_diagnostics(summary_path: Path,
                                    task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Backfill terminal diagnostics for summaries written before this fix."""
    from c2kv_eval.portable.adapters.toolsandbox_adapter import (
        _execution_diagnostics,
    )

    native = sorted(summary_path.parent.glob("agent_*/result_summary.json"))
    if not native:
        return {}
    diagnostics = [_execution_diagnostics(native[-1], str(row.get("task_id")))
                   for row in task_rows]
    if not diagnostics:
        return {}
    normal = [item.get("normal_termination") for item in diagnostics]
    tool_rates = [item.get("tool_execution_success_rate") for item in diagnostics]
    return {
        "normal_termination_rate": sum(value is True for value in normal) / len(normal),
        "premature_termination_count": sum(value is not True for value in normal),
        "tool_execution_success_rate": _mean(tool_rates),
        "termination_counts": {
            "normal": sum(value is True for value in normal),
            "max_messages_or_unknown": sum(value is not True for value in normal),
        },
    }


def build_row(summary_path: Path, root: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    method_key = summary_path.parent.name
    label, expected_arm, operation = METHODS.get(
        method_key, (method_key, str(summary.get("arm")), None))
    request_summary = summary.get("request_log_summary") or {}
    request_rows = _request_rows(str(summary.get("request_log") or ""))
    task_rows = summary.get("task_rows") or []
    benchmark = summary.get("benchmark")

    normal_rate = summary.get("normal_termination_rate")
    premature = summary.get("premature_termination_count")
    termination_counts = summary.get("termination_counts") or {}
    tool_execution_success = summary.get("tool_execution_success_rate")
    if normal_rate is None and benchmark == "tau2" and task_rows:
        normal = [row.get("termination") in {"agent_stop", "user_stop"}
                  for row in task_rows]
        normal_rate = sum(normal) / len(normal)
        premature = len(normal) - sum(normal)
        for row in task_rows:
            key = str(row.get("termination") or "unknown")
            termination_counts[key] = termination_counts.get(key, 0) + 1
    elif normal_rate is None and benchmark == "toolsandbox":
        legacy = _legacy_toolsandbox_diagnostics(summary_path, task_rows)
        normal_rate = legacy.get("normal_termination_rate")
        premature = legacy.get("premature_termination_count")
        termination_counts = legacy.get("termination_counts") or {}
        tool_execution_success = legacy.get("tool_execution_success_rate")

    tensor = request_summary.get("history_tensor_accounting") or {}
    if method_key.startswith(("streamingllm_", "h2o_", "snapkv_", "pyramidkv_")):
        active = request_summary.get("history_kv_active_tokens_mean")
        retention = request_summary.get("history_kv_retention_mean")
        compression = request_summary.get("history_kv_compression_mean")
        compression_scope = "history-KV selection"
    elif method_key in {"full", "full_kv_retry"}:
        # retry_full's committed response is regenerated with full history KV.
        active, retention, compression = None, 1.0, 1.0
        compression_scope = "full history KV"
    else:
        # C2KV and append/replace must not consume the scheduler's legacy
        # history_kv_* counters (those can pair cumulative active tokens with
        # the first document's full-equivalent count).  Recovery rows use the
        # post-recovery tensor ratio; plain C2KV uses logical/original:gist.
        ratio = (tensor.get("history_ratio_after_mean")
                 if operation in {"append", "replace"}
                 else request_summary.get("logical_over_gist"))
        compression = float(ratio) if isinstance(ratio, (int, float)) and ratio > 0 else None
        retention = 1.0 / compression if compression else None
        active = (tensor.get("after_recovery_tokens_mean")
                  if operation in {"append", "replace"}
                  else _mean([row.get("gist_tokens") for row in request_rows
                              if (row.get("gist_tokens") or 0) > 0]))
        compression_scope = ("post-recovery history KV tensor payload"
                             if operation in {"append", "replace"}
                             else "logical original history / C2KV gist")

    native = ({
        "protocol_legal_rate": summary.get("protocol_legal_rate"),
        "protocol_tool_pool_size": summary.get("protocol_tool_pool_size"),
    } if summary.get("benchmark") == "tau2" else {
        "milestone_similarity_mean": summary.get("milestone_similarity_mean"),
        "minefield_similarity_mean": summary.get("minefield_similarity_mean"),
        "turn_count_mean": summary.get("turn_count_mean"),
    })
    scope = ("request-local generation/KV retry; no tool/environment rollback"
             if operation else None)
    quality_ok = normal_rate == 1.0 and request_summary.get("n_error", 0) == 0
    return {
        "phase": summary_path.relative_to(root).parts[0],
        "benchmark": summary.get("benchmark"),
        "method": label,
        "portable_arm": summary.get("arm"),
        "expected_arm": expected_arm,
        "recovery_operation": operation,
        "num_cases": summary.get("n"),
        "official_metric_kind": (summary.get("official_metric_kind") or
                                 ("binary_task_success" if benchmark == "tau2"
                                  else "continuous_scenario_similarity")),
        "task_success_rate": ((summary.get("official_success_rate")
                               if summary.get("official_success_rate") is not None
                               else summary.get("semantic_score"))
                              if benchmark == "tau2" else None),
        "official_score": summary.get("semantic_score"),
        "normal_termination_rate": normal_rate,
        "premature_termination_count": premature,
        "termination_counts_json": json.dumps(
            termination_counts, ensure_ascii=False,
            separators=(",", ":")),
        "smoke_quality_ok": quality_ok,
        "turn_success": None,
        "protocol_legal_rate": summary.get("protocol_legal_rate"),
        "tool_execution_success_rate": tool_execution_success,
        "recovery_trigger_rate": request_summary.get(
            "request_recovery_trigger_rate"),
        "recovery_success_rate": request_summary.get(
            "request_recovery_success_rate"),
        "model_calls_per_step": request_summary.get(
            "model_calls_per_proxy_request"),
        "history_kv_retention": retention,
        "history_kv_compression": compression,
        "compression_scope": compression_scope,
        "active_kv_tokens": active,
        "runtime": summary.get("runner_adapter_wall_sec"),
        "recovery_scope": scope,
        "native_metrics_json": json.dumps(native, ensure_ascii=False,
                                            separators=(",", ":")),
        "summary_path": str(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--require-normal-termination", action="store_true",
                        help="fail after writing CSV unless every row ended normally")
    args = parser.parse_args()
    paths = sorted(args.root.glob("*/*/*/summary_*.json"))
    rows = [build_row(path, args.root) for path in paths]
    if not rows:
        raise SystemExit(f"no portable summaries found under {args.root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    if args.require_normal_termination:
        failed = [f"{row['benchmark']}/{row['method']}"
                  for row in rows if row["smoke_quality_ok"] is not True]
        if failed:
            raise SystemExit(
                "smoke quality gate failed (non-normal termination or request error): "
                + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
