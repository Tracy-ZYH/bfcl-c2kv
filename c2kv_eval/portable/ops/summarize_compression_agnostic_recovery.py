#!/usr/bin/env python3
"""Summarize the controlled W2 compression-agnostic recovery smoke matrix."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


_POOL_USAGE_RE = re.compile(r"c2kv usage:\s*([0-9]+(?:\.[0-9]+)?)")


def _write_pool_usage(root: Path, out: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "logs").glob("server_*.log")):
        values = [float(value) for value in _POOL_USAGE_RE.findall(
            path.read_text(encoding="utf-8", errors="replace"))]
        if not values:
            continue
        name = path.name.removeprefix("server_").removesuffix(".log")
        benchmark = name.split("_card", 1)[0]
        rows.append({
            "benchmark": benchmark,
            "samples": len(values),
            "max_c2kv_pool_usage": max(values),
            "last_c2kv_pool_usage": values[-1],
            "server_log": str(path),
        })
    if rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(path: Path, root: Path) -> dict[str, Any]:
    summary = _load(path)
    request = summary.get("request_log_summary") or {}
    tensor = request.get("history_tensor_accounting") or {}
    method = path.parent.name
    arm = str(summary.get("arm") or "")
    recovery = "compression_only"
    if "_replace_w2" in method:
        recovery = "replace_w2"
    elif "_append_w2" in method:
        recovery = "append_w2"
    backend = "c2kv" if arm == "c2kv4" else arm.removeprefix("history_kv_").removesuffix("_r25")
    if backend == "snapkv":
        backend = "snapkv_persistent"
    active = request.get("history_kv_active_tokens_mean")
    compression = request.get("history_kv_compression_mean")
    before = request.get("before_recovery_active_tokens_mean")
    after = request.get("after_recovery_active_tokens_mean")
    if backend == "c2kv":
        if recovery == "compression_only":
            compression = request.get("logical_over_gist")
            active = request.get("gist_tokens_mean")
        else:
            compression = tensor.get("history_ratio_after_mean")
            before = tensor.get("before_recovery_tokens_mean")
            after = tensor.get("after_recovery_tokens_mean")
            active = after if after is not None else None
    return {
        "benchmark": summary.get("benchmark"),
        "method": method,
        "portable_arm": arm,
        "compression_backend": backend,
        "retention": 0.25,
        "nominal_compression_ratio": 4.0,
        "recovery_mode": recovery,
        "status": "completed",
        "official_metric_kind": summary.get("official_metric_kind"),
        "official_task_score": summary.get("semantic_score"),
        "num_cases": summary.get("n"),
        "normal_termination_rate": summary.get("normal_termination_rate"),
        "tool_call_count": summary.get("tool_execution_count"),
        "tool_call_failure_count": summary.get("tool_execution_failure_count"),
        "tool_call_success": summary.get("tool_execution_success_rate"),
        "request_errors": request.get("n_error"),
        "recovery_trigger_rate": request.get("request_recovery_trigger_rate"),
        "recovery_success_rate": request.get("request_recovery_success_rate"),
        "active_history_kv_tokens": active,
        "history_kv_compression": compression,
        "restored_raw_kv_tokens": request.get("restored_raw_kv_tokens_mean"),
        "before_recovery_active_tokens": before,
        "after_recovery_active_tokens": after,
        "recovered_segment_size": request.get("recovered_segment_size_mean"),
        "duplicate_raw_kv_tokens": request.get("duplicate_raw_kv_tokens_total"),
        "model_calls_per_step": request.get("model_calls_per_proxy_request"),
        "runtime_sec": summary.get("runner_adapter_wall_sec"),
        "summary_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", default="smoke")
    parser.add_argument("--expected-completed", type=int, default=22)
    parser.add_argument("--expected-cases", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-infrastructure-clean", action="store_true",
                        help="require complete, error-free requests and valid KV recovery accounting; do not require every benchmark dialogue to terminate normally")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--max-c2kv-pool-usage", type=float, default=None,
                        help="fail a quality gate if a server log exceeds this C2KV pool high-water mark")
    args = parser.parse_args()
    rows = [_row(path, args.root) for path in sorted(
        args.root.glob(f"{args.phase}/*/*/summary_*.json"))]
    unsupported = args.root / "manifests" / "unsupported.tsv"
    if unsupported.exists():
        present_benchmarks = {row["benchmark"] for row in rows}
        with unsupported.open(encoding="utf-8") as handle:
            for item in csv.DictReader(handle, delimiter="\t"):
                for benchmark in ("tau2", "toolsandbox"):
                    if benchmark not in present_benchmarks:
                        continue
                    rows.append({
                        "benchmark": benchmark, "method": item["method"],
                        "portable_arm": item["portable_arm"],
                        "compression_backend": item["compression_backend"],
                        "retention": 0.25, "nominal_compression_ratio": 4.0,
                        "recovery_mode": item["recovery_mode"],
                        "status": "unsupported", "official_metric_kind": None,
                        "official_task_score": None, "num_cases": 0,
                        "normal_termination_rate": None,
                        "tool_call_count": None,
                        "tool_call_failure_count": None,
                        "tool_call_success": None, "request_errors": None,
                        "recovery_trigger_rate": None,
                        "recovery_success_rate": None,
                        "active_history_kv_tokens": None,
                        "history_kv_compression": None,
                        "restored_raw_kv_tokens": None,
                        "before_recovery_active_tokens": None,
                        "after_recovery_active_tokens": None,
                        "recovered_segment_size": None,
                        "duplicate_raw_kv_tokens": None,
                        "model_calls_per_step": None, "runtime_sec": None,
                        "summary_path": item["reason"],
                    })
    if not rows:
        raise SystemExit(f"no smoke summaries found below {args.root}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    pool_rows = _write_pool_usage(args.root, args.out.with_name("pool_usage.csv"))
    print(f"wrote {len(rows)} rows to {args.out}")
    if (args.require_complete or args.require_infrastructure_clean
            or args.require_clean):
        completed = [row for row in rows if row["status"] == "completed"]
        failures = []
        if len(completed) != args.expected_completed:
            failures.append(
                f"expected {args.expected_completed} completed cells, "
                f"found {len(completed)}")
        for row in completed:
            label = f"{row['benchmark']}/{row['method']}"
            if row["num_cases"] != args.expected_cases:
                failures.append(
                    f"{label}: expected {args.expected_cases} scored cases, "
                    f"found {row['num_cases']}")
            if row["request_errors"] != 0:
                failures.append(f"{label}: request_errors={row['request_errors']}")
            if (args.require_clean
                    and row["normal_termination_rate"] not in (1, 1.0)):
                failures.append(
                    f"{label}: normal_termination_rate={row['normal_termination_rate']}")
            if row["recovery_mode"] != "compression_only":
                if row["restored_raw_kv_tokens"] is None:
                    failures.append(f"{label}: missing restored_raw_kv_tokens")
                if row["duplicate_raw_kv_tokens"] != 0:
                    failures.append(
                        f"{label}: duplicate_raw_kv_tokens={row['duplicate_raw_kv_tokens']}")
        if args.max_c2kv_pool_usage is not None:
            for row in pool_rows:
                if row["max_c2kv_pool_usage"] > args.max_c2kv_pool_usage:
                    failures.append(
                        f"{row['benchmark']}: max_c2kv_pool_usage="
                        f"{row['max_c2kv_pool_usage']:.3f} > "
                        f"{args.max_c2kv_pool_usage:.3f}")
        if failures:
            raise SystemExit("smoke quality gate failed:\n" + "\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
