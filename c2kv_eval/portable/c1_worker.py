"""Run one official non-BFCL harness task against one C1 controller.

This module is intentionally orchestration-only.  It delegates generation,
tool execution, and official scoring to the existing portable benchmark
adapters; C1 detector/recovery state remains entirely in the controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

from .adapters.base import RunContext


def _bare_endpoint(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if parsed.scheme != "http" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"expected a bare HTTP endpoint, got {value!r}")
    return value.rstrip("/").removesuffix("/v1")


def build_context(args: argparse.Namespace) -> RunContext:
    agent = _bare_endpoint(args.agent_base_url)
    user = _bare_endpoint(args.user_base_url)
    if agent == user:
        raise ValueError("agent C1 endpoint and raw Full user endpoint must differ")
    options = {
        "bench_python": args.bench_python,
        "num_workers": 1,
    }
    if args.benchmark == "tau2":
        options.update({
            "tau2_dir": args.benchmark_dir,
            "task_set": args.task_set,
            "tau2_task_ids": args.task_id,
            "tau2_num_trials": 1,
            "tau2_max_steps": args.tau2_max_steps,
            "tau2_timeout": args.task_timeout,
        })
    else:
        options.update({
            "toolsandbox_dir": args.benchmark_dir,
            "ts_scenarios": args.task_id,
            "ts_agent": args.ts_agent,
            "ts_user": args.ts_user,
            "full": False,
        })
    return RunContext(
        base_url=agent,
        user_base_url=user,
        out_dir=args.out,
        model=args.model,
        arm=args.model,
        run_name=args.run_name,
        options=options,
    )


def run(args: argparse.Namespace) -> dict:
    if args.out.exists():
        raise FileExistsError(f"output already exists: {args.out}")
    if not args.benchmark_dir.is_dir():
        raise FileNotFoundError(f"benchmark checkout not found: {args.benchmark_dir}")
    context = build_context(args)
    if args.benchmark == "tau2":
        from .adapters import tau2_adapter as adapter
    else:
        from .adapters import toolsandbox_adapter as adapter
    started = time.monotonic()
    summary = adapter.run(context)
    summary.update({
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "status": "officially_scored",
        "agent_endpoint": context.base_url,
        "user_endpoint": context.user_base_url,
        "wall_time": time.monotonic() - started,
    })
    path = args.out / "official_summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("tau2", "toolsandbox"), required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--agent-base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--bench-python", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="c1_legacy_prefill")
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--tau2-max-steps", type=int)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--ts-agent", default="GPT_4_o_2024_05_13")
    parser.add_argument("--ts-user", default="GPT_4_o_2024_05_13")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if "/" in args.task_id or "\\" in args.task_id or args.task_id in {".", ".."}:
        raise SystemExit("task identity must be one path-safe official ID")
    if args.task_timeout <= 0:
        raise SystemExit("task timeout must be positive")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
