"""ToolSandbox adapter (official-CLI driver).

The ToolSandbox Python API (Scenario/play/play_and_evaluate) does not expose
a Scenario.discover(); the supported entrypoint is the ``tool_sandbox`` CLI,
which runs scenarios against an OpenAI-compatible endpoint configured via
``OPENAI_BASE_URL`` and writes ``result_summary.json`` per run.  This
adapter drives that CLI and parses the summaries.

Verified metrics fields (per scenario): similarity / milestone_similarity /
minefield_similarity / turn_count.  There is no "main_acc".

Usage (benchts venv on the server):
    python benchmarks/adapters/toolsandbox_adapter.py \
        --base-url http://127.0.0.1:34002/v1 --out results/bench/ts_full
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base import RunContext, v1

NAME = "toolsandbox"
TS_DIR = Path(os.environ.get("TS_DIR") or Path.home() / "benchmarks" / "ToolSandbox")
AGENT = "GPT_4_o_2024_05_13"  # openai_api_agent/openai_api_user role keys


def add_arguments(parser) -> None:
    """ToolSandbox-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--toolsandbox-dir", type=Path, default=None,
                        help="ToolSandbox checkout (default $TS_DIR)")
    parser.add_argument("--full", action="store_true",
                        help="toolsandbox: full suite instead of test mode")
    parser.add_argument("--ts-scenarios", default="",
                        help="toolsandbox: comma-separated scenario names "
                             "for subset runs (-s); overrides --full")
    parser.add_argument("--ts-agent", default="",
                        help="toolsandbox: agent role key (default "
                             "GPT_4_o_2024_05_13 -> openai_api_agent)")
    parser.add_argument("--ts-user", default="",
                        help="toolsandbox: user-simulator role key (same default)")


def cli_command(out_dir: Path, agent: str = AGENT, user: str = AGENT,
                test_mode: bool = True,
                scenarios: "list[str] | None" = None,
                parallel: "str | None" = None) -> List[str]:
    """``tool_sandbox`` argv (PINNED).  ``-s names...`` is the subset form
    (the CLI also takes ``-p`` for parallelism, from $TS_PARALLEL); ``-t``
    is test mode, and a subset run overrides it."""
    cmd = ["tool_sandbox", "--user", user, "--agent", agent, "-o", str(out_dir)]
    if scenarios:
        cmd += ["-s"] + list(scenarios)
        if parallel:
            cmd += ["-p", parallel]
    elif test_mode:
        cmd.append("-t")
    return cmd


def split_scenarios(raw) -> "list[str] | None":
    """``--ts-scenarios a,b`` -> ["a", "b"]; empty -> None (full/test mode)."""
    if not raw:
        return None
    items = ([s.strip() for s in raw.split(",")] if isinstance(raw, str)
             else [str(s).strip() for s in raw])
    return [s for s in items if s] or None


def harness_env(base_url: str, user_base_url: str = "") -> Dict[str, str]:
    """``user_base_url`` (default: the raw upstream endpoint) routes the
    user simulator OUT of the arm proxy via TOOLSANDBOX_USER_BASE_URL —
    the patched openai_api_user role reads it.  Routing the simulator
    through the compression arm made every historical TS number an
    agent+user joint degradation (audit BLOCKER)."""
    return {
        **os.environ,
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_API_KEY_USER": "EMPTY",
        "OPENAI_BASE_URL": v1(base_url),
        # default: same endpoint the proxy itself fronts (full mode)
        "TOOLSANDBOX_USER_BASE_URL": v1(user_base_url) if user_base_url
        else os.environ.get("TOOLSANDBOX_USER_BASE_URL", v1(base_url)),
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }


def run(ctx: RunContext) -> Dict[str, Any]:
    """Drive the official ``tool_sandbox`` CLI against the arm proxy.

    No cost join: see ``COST_JOIN`` below.
    """
    scenarios = split_scenarios(ctx.opt("ts_scenarios", ""))
    summary = run_ts(
        ctx.base_url, ctx.out_dir,
        test_mode=not ctx.options.get("full", False),
        agent=ctx.opt("ts_agent", AGENT), user=ctx.opt("ts_user", AGENT),
        # the user simulator must NOT ride the arm proxy: route it to the
        # raw upstream endpoint (tau2 already does the same split)
        user_base_url=ctx.user_base_url,
        scenarios=scenarios,
        # An explicit -s selection has a fixed denominator.  Do not let a
        # missing result row silently turn a finite smoke into a smaller run.
        expected=len(scenarios) if scenarios is not None else None,
        expected_task_ids=scenarios,
        benchmark_dir=ctx.opt("toolsandbox_dir"), python=ctx.opt("bench_python"),
    )
    summary["cost_join"] = COST_JOIN
    return summary


# Why ToolSandbox gets no per-task cost columns: ``result_summary.json`` is
# the only artefact this adapter reads and it carries per-scenario SCORES
# (similarity / milestone_similarity / minefield_similarity / turn_count,
# tool_sandbox/cli/utils.py:196-208), no messages — so nothing here can
# rebuild the message prefix ``proxy.conversation_id`` keys on.
COST_JOIN = ("not joinable: result_summary.json holds per-scenario scores "
             "only, no messages to key the request log by")


def run_ts(base_url: str, out_dir: Path, test_mode: bool = True,
            agent: str = AGENT, user: str = AGENT, expected: int = None,
            benchmark_dir: Path = None, user_base_url: str = "",
            scenarios: "list[str] | None" = None,
            expected_task_ids: "list[str] | None" = None,
            python: "str | None" = None) -> Dict[str, Any]:
    """Run the CLI and collect ``result_summary.json``."""
    if expected_task_ids is not None:
        expected_task_ids = [str(task_id) for task_id in expected_task_ids]
        if len(set(expected_task_ids)) != len(expected_task_ids):
            raise ValueError("ToolSandbox expected_task_ids contains a duplicate scenario")
        if expected is None:
            expected = len(expected_task_ids)
        elif expected != len(expected_task_ids):
            raise ValueError(
                "ToolSandbox expected count does not match expected_task_ids")
    ts_dir = (Path(benchmark_dir) if benchmark_dir else TS_DIR).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = harness_env(base_url, user_base_url)
    # The console script's directory is otherwise first on sys.path, and an
    # editable installation may silently import a different checkout.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ts_dir.resolve()), env.get("PYTHONPATH"))))
    cmd = cli_command(out_dir, agent=agent, user=user, test_mode=test_mode,
                      scenarios=scenarios,
                      parallel=os.environ.get("TS_PARALLEL"))
    cmd[0] = str(Path(python or sys.executable).parent / "tool_sandbox")
    completed = subprocess.run(cmd, cwd=ts_dir, env=env)
    if completed.returncode != 0:
        raise SystemExit(f"FATAL: tool_sandbox CLI exited {completed.returncode}")
    summary = (collect(out_dir, expected_task_ids=expected_task_ids)
               if expected_task_ids is not None else collect(out_dir))
    # terminal-state check (acceptance 1): a scenario that never ran must
    # fail the run, not shrink the denominator
    n_scored = summary.get("n") if isinstance(summary, dict) else None
    if expected is not None and n_scored != expected:
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: n_scored={n_scored} != n_total={expected}")
    if expected is not None:
        print(f"TERMINAL-STATE ts: n_scored={n_scored} n_total={expected}")
    return summary


def _execution_diagnostics(summary_path: Path, scenario_name: str) -> Dict[str, Any]:
    """Read native terminal/tool state without changing ToolSandbox scoring."""
    path = summary_path.parent / "trajectories" / scenario_name / "execution_context.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "conversation_active": None,
            "normal_termination": None,
            "tool_execution_count": None,
            "tool_execution_failure_count": None,
            "tool_execution_success_rate": None,
        }
    sandbox = ((data.get("_dbs") or {}).get("SANDBOX") or [])
    records = [row for row in sandbox if isinstance(row, dict)]
    terminal_rows = [row for row in records
                     if isinstance(row.get("conversation_active"), bool)]
    terminal = max(
        enumerate(terminal_rows),
        key=lambda pair: (pair[1].get("sandbox_message_index", -1), pair[0]),
        default=(None, None),
    )[1]
    active = terminal.get("conversation_active") if terminal else None
    # A failed tool call has ``tool_trace=null`` in ToolSandbox, so selecting
    # only rows with a trace silently discarded every exception and made the
    # success rate read 100%.  Count every agent-facing execution response;
    # user-only ``end_conversation`` rows are addressed to USER and excluded.
    tool_rows = [row for row in records
                 if row.get("sender") == "EXECUTION_ENVIRONMENT"
                 and row.get("recipient") == "AGENT"
                 and row.get("openai_tool_call_id")]
    tool_successes = sum(not row.get("tool_call_exception") for row in tool_rows)
    tool_failures = len(tool_rows) - tool_successes
    return {
        "conversation_active": active,
        "normal_termination": (not active) if isinstance(active, bool) else None,
        "tool_execution_count": len(tool_rows),
        "tool_execution_failure_count": tool_failures,
        "tool_execution_success_rate": (
            tool_successes / len(tool_rows) if tool_rows else None),
    }


def collect(out_dir: Path, expected_task_ids: "list[str] | None" = None) -> Dict[str, Any]:
    summaries = sorted(out_dir.glob("agent_*/result_summary.json"))
    if not summaries:
        raise SystemExit(f"FATAL: no result_summary.json under {out_dir} — "
                         "the CLI produced nothing (check TS run logs)")
    from ..metrics import aggregate

    rows: List[Dict[str, Any]] = []
    crashed: List[str] = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        for scenario in data.get("per_scenario_results") or []:
            if scenario.get("traceback"):
                # a crashed scenario FAILS the run — _mean used to skip
                # these None rows and the upstream recorded a silent 0
                crashed.append(str(scenario.get("name")))
                continue
            task_id = str(scenario.get("name"))
            diagnostics = _execution_diagnostics(path, task_id)
            rows.append({
                "task_id": task_id,
                # official semantic column: dialogue similarity to the
                # reference (milestone-weighted); minefield = violations
                "semantic_score": scenario.get("similarity"),
                "milestone_similarity": scenario.get("milestone_similarity"),
                "minefield_similarity": scenario.get("minefield_similarity"),
                "turn_count": scenario.get("turn_count"),
                "protocol_legal": None,  # TS has no tool-call legality metric
                **diagnostics,
            })
    if crashed:
        raise SystemExit(
            f"FATAL: ts terminal-state check failed: {len(crashed)} scenario(s) "
            f"crashed (traceback in result_summary): {', '.join(crashed[:10])}")
    if expected_task_ids is not None:
        expected_ids = {str(task_id) for task_id in expected_task_ids}
        scored_ids = {str(row["task_id"]) for row in rows}
        if scored_ids != expected_ids:
            missing = sorted(expected_ids - scored_ids)
            unexpected = sorted(scored_ids - expected_ids)
            details = []
            if missing:
                details.append(f"missing={','.join(missing[:20])}")
            if unexpected:
                details.append(f"unexpected={','.join(unexpected[:20])}")
            raise SystemExit(
                "FATAL: ts terminal-state task-id check failed: " + "; ".join(details))
    summary = aggregate(rows, cluster_key="task_id")

    def mean(field: str):
        values = [float(row[field]) for row in rows
                  if isinstance(row.get(field), (int, float))
                  and not isinstance(row.get(field), bool)]
        return sum(values) / len(values) if values else None

    # Preserve ToolSandbox's native official metrics alongside the portable
    # semantic column.  These were previously parsed and then discarded by
    # the common two-column aggregator.
    summary.update({
        "official_metric_kind": "continuous_scenario_similarity",
        "milestone_similarity_mean": mean("milestone_similarity"),
        "minefield_similarity_mean": mean("minefield_similarity"),
        "turn_count_mean": mean("turn_count"),
        "tool_execution_success_rate": mean("tool_execution_success_rate"),
        "tool_execution_count": sum(
            row.get("tool_execution_count") or 0 for row in rows),
        "tool_execution_failure_count": sum(
            row.get("tool_execution_failure_count") or 0 for row in rows),
        "normal_termination_rate": (
            sum(row.get("normal_termination") is True for row in rows) / len(rows)
            if rows else None),
        "premature_termination_count": sum(
            row.get("normal_termination") is not True for row in rows),
        "termination_counts": {
            "normal": sum(row.get("normal_termination") is True for row in rows),
            "max_messages_or_unknown": sum(
                row.get("normal_termination") is not True for row in rows),
        },
        "task_rows": rows,
    })
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true",
                        help="run the full suite instead of test mode")
    parser.add_argument("--agent", default=AGENT,
                        help="agent role key (openai_api_agent config entry)")
    parser.add_argument("--user", default=AGENT,
                        help="user-simulator role key (openai_api_user config entry)")
    parser.add_argument("--ts-dir", type=Path, default=None,
                        help="ToolSandbox checkout (default $TS_DIR or ~/benchmarks/ToolSandbox)")
    args = parser.parse_args()
    summary = run_ts(args.base_url, args.out, test_mode=not args.full,
                     agent=args.agent, user=args.user, benchmark_dir=args.ts_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
