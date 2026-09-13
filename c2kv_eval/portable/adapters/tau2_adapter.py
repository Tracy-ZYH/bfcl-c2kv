"""τ²-bench adapter.

tau2 talks to the agent LLM through LiteLLM, so the arm proxy is plugged in
purely by configuration: an OpenAI-compatible provider entry whose api_base
is the proxy.  The user simulator and any judge calls go to a *separate*
full-mode endpoint (we compress only the agent's view, never the user
simulator, to keep the benchmark semantics intact).

Run recipe (see benchmarks/README.md):
  1. proxy in the requested arm on port P (agent endpoint)
  2. write a litellm provider block + settings JSON
  3. `tau2 run` over the requested task set
  4. `tau2 evaluate-trajs` for the official reward
  5. this adapter parses trajectories + the proxy request log into unified
     rows (per task: official reward as semantic column; protocol columns
     recomputed from raw assistant turns with our shared checker)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..metrics import aggregate, protocol_columns_for_turn
from .base import RunContext, v1

NAME = "tau2"
TAU2_DIR = Path(os.environ.get("TAU2_DIR") or Path.home() / "benchmarks" / "tau2")


def add_arguments(parser) -> None:
    """tau2-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument(
        "--tau2-dir", type=Path, default=None,
        help="tau2 checkout (default: $TAU2_DIR or ~/benchmarks/tau2)",
    )
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--tau2-task-ids", default="",
                        help="comma-separated exact tau2 task ids; keeps every arm on the same cases")
    parser.add_argument("--tau2-num-trials", type=int, default=None,
                        help="tau2: trials per selected task (unset keeps the official default)")
    parser.add_argument("--tau2-max-steps", type=int, default=None,
                        help="tau2: per-task turn cap (unset keeps the official default)")
    parser.add_argument("--tau2-timeout", type=int, default=None,
                        help="tau2: per-task wallclock cap in seconds (unset means no cap)")


def run_command(base_url: str, user_base_url: str, task_set: str, model: str,
                num_workers: int, run_name: str,
                max_tasks: Optional[int] = None,
                num_trials: Optional[int] = None,
                max_steps: Optional[int] = None,
                timeout: Optional[int] = None,
                task_ids: Optional[List[str]] = None,
                python: Optional[str] = None) -> List[str]:
    """``tau2.cli run`` argv — PINNED: the server scripts quote these
    numbers, so any edit here changes what every historical tau2 row means.
    """
    agent_args = json.dumps(
        {"api_base": v1(base_url), "api_key": "EMPTY", "temperature": 0.0}
    )
    user_args = json.dumps(
        {"api_base": v1(user_base_url), "api_key": "EMPTY", "temperature": 0.0}
    )
    cmd = [
        python or sys.executable, "-m", "tau2.cli", "run",
        "--domain", task_set.split("_")[0],
        "--task-set-name", task_set,
        "--agent-llm", f"openai/{model}",
        "--agent-llm-args", agent_args,
        "--user-llm", f"openai/{model}",
        "--user-llm-args", user_args,
        "--max-concurrency", str(num_workers),
        "--save-to", run_name,
        # headless: resume an existing checkpoint without the interactive
        # prompt (a killed run's checkpoint otherwise EOFs the CLI)
        "--auto-resume",
    ]
    if num_trials is not None:
        cmd += ["--num-trials", str(num_trials)]
    if task_ids:
        cmd += ["--task-ids", *[str(task_id) for task_id in task_ids]]
    if max_tasks is not None:
        cmd += ["--num-tasks", str(max_tasks)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    return cmd


def evaluate_command(sims: Path, python: Optional[str] = None) -> List[str]:
    """``tau2.cli evaluate-trajs`` argv (writes updated_results.json)."""
    return [python or sys.executable, "-m", "tau2.cli", "evaluate-trajs",
            "-o", str(sims), str(sims / "results.json")]


def harness_env(tau2_dir: Optional[Path] = None) -> Dict[str, str]:
    env = {**os.environ,
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    if tau2_dir is not None:
        # An existing harness environment may contain an editable tau2 from
        # another account.  Put the explicitly selected checkout first without
        # modifying that environment.
        source = str((Path(tau2_dir).resolve() / "src"))
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (source, env.get("PYTHONPATH"))))
    return env


def split_task_ids(raw: Any) -> List[str]:
    if not raw:
        return []
    values = raw.split(",") if isinstance(raw, str) else raw
    return [str(value).strip() for value in values if str(value).strip()]


def run(ctx: RunContext) -> Dict[str, Any]:
    """Run tau2 with the agent LLM behind the arm proxy.

    The user simulator points at a separate full-mode endpoint (same served
    model, no compression) so only the agent's context is ever compressed.
    ``--tau2-dir``/``$TAU2_DIR`` (or ~/benchmarks/tau2, or
    ``benchmark_dir`` when this module is driven standalone) selects the tau2
    checkout that provides both the CLI and the tool registry.
    Official semantics: ``--save-to NAME`` writes
    ``<tau2_dir>/data/simulations/NAME/results.json``; rewards are computed
    by ``tau2 evaluate-trajs`` into ``updated_results.json``, which collect
    then reads.

    No cost join: see ``COST_JOIN`` below.
    """
    benchmark_dir = ctx.opt("tau2_dir") or ctx.opt("benchmark_dir")
    tau2_dir = Path(benchmark_dir) if benchmark_dir else TAU2_DIR
    task_set = ctx.opt("task_set", "airline")
    max_tasks = ctx.opt("max_tasks")
    task_ids = split_task_ids(ctx.opt("tau2_task_ids", ""))
    python = ctx.opt("bench_python", sys.executable)
    if task_ids and max_tasks is not None:
        raise ValueError("--tau2-task-ids and --max-tasks are mutually exclusive")

    env = harness_env(tau2_dir)
    subprocess.run(
        run_command(ctx.base_url, ctx.user_base_url, task_set, ctx.model,
                    ctx.opt("num_workers", 4), ctx.run_name,
                    max_tasks=max_tasks,
                    num_trials=ctx.opt("tau2_num_trials"),
                    max_steps=ctx.opt("tau2_max_steps"),
                    timeout=ctx.opt("tau2_timeout"),
                    task_ids=task_ids, python=python),
        cwd=tau2_dir, env=env, check=True)
    sims = tau2_dir / "data" / "simulations" / ctx.run_name
    subprocess.run(evaluate_command(sims, python=python),
                   cwd=tau2_dir, env=env, check=True)
    updated = sims / "updated_results.json"
    if not updated.exists():
        raise SystemExit(f"FATAL: tau2 evaluation produced no {updated}")
    # tau2 writes native artifacts inside its checkout.  Keep immutable copies
    # with the portable result so an independently named run directory is
    # self-contained even if tau2's simulations directory is later cleaned.
    native_results_copy = ctx.out_dir / "tau2_results.json"
    native_updated_copy = ctx.out_dir / "tau2_updated_results.json"
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sims / "results.json", native_results_copy)
    shutil.copy2(updated, native_updated_copy)
    # terminal-state gate via the shared checker: infra-error simulations
    # are NOT valid terminal states (the old inline len(sims) check counted
    # them as scored)
    from .. import terminal_check

    code = terminal_check.check_tau2(
        ctx.run_name, len(task_ids) or max_tasks or None,
        task_ids=",".join(task_ids),
        root=tau2_dir / "data" / "simulations",
    )
    if code != 0:
        raise SystemExit(f"FATAL: tau2 terminal-state check failed (rc={code})")
    summary = collect(updated, domain=task_set.split("_")[0])
    summary["simulation_dir"] = str(sims)
    summary["native_results_copy"] = str(native_results_copy)
    summary["native_updated_results_copy"] = str(native_updated_copy)
    summary["selected_task_ids"] = task_ids or None
    summary["cost_join"] = COST_JOIN
    return summary


# Why tau2 gets no per-task cost columns.  ``proxy.conversation_id`` keys on
# the system head + the first two non-system messages OF THE REQUEST AS SENT
# (proxy.py:434-447), so rebuilding it needs the exact wire payload:
#   1. the agent system message is NOT in results.json — tau2 keeps it in
#      LLMAgentState.system_messages and only the conversation messages are
#      serialised (tau2-bench src/tau2/agent/llm_agent.py:101,127);
#   2. even reconstructing it through tau2's own code (LLMAgent.system_prompt
#      + registry policy) leaves litellm between us and the socket: an
#      assistant tool-call message carries ``content: None``
#      (src/tau2/utils/llm_utils.py:191-197), and None vs "" changes
#      _canonical_messages' output (proxy.py:412-414), so the key would
#      silently mismatch.
# This repo holds no captured (results.json, request log) pair to pin either
# question, and a key that is wrong for the steady-state id would attribute
# only each task's FIRST request — worse than no column.  One captured pair
# turns this into a small addition.
COST_JOIN = ("not joinable: the agent system message is not stored in "
             "results.json and the litellm wire form of an assistant "
             "tool-call message is unpinned (see adapters/tau2_adapter.py)")


def collect(results_path: Path, domain: str = "airline") -> Dict[str, Any]:
    """Parse a tau2 results.json into unified rows.

    Verified against real trajectory files: simulations[i].messages carry
    role/content/tool_calls (litellm already parsed our server's tool_calls),
    reward_info.reward is the official semantic score.  Protocol columns are
    recomputed with the shared checker against the domain tool pool.
    """
    from ..metrics import protocol_columns_for_turn

    tools: List[Dict[str, Any]] = []
    try:
        from tau2.registry import registry

        env = registry.get_env_constructor(domain)()
        tools = [tool.openai_schema for tool in env.get_tools()]
    except Exception as error:  # noqa: BLE001 - protocol column degrades
        print(f"WARNING: tau2 tool pool unavailable ({error!r}); "
              "protocol column degrades to unknown", file=sys.stderr)
        tools = []

    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for sim in data.get("simulations") or []:
        turns = [
            protocol_columns_for_turn(m, tools)
            for m in sim.get("messages") or []
            if m.get("role") == "assistant"
        ]
        first_violations = [
            t["first_violation"] for t in turns
            if t["protocol_legal"] is False and t["first_violation"]
        ]
        protocol_legal = None
        if any(t["protocol_legal"] is False for t in turns):
            protocol_legal = False
        elif turns and all(t["protocol_legal"] is True for t in turns):
            protocol_legal = True
        reward_info = sim.get("reward_info") or {}
        termination = sim.get("termination_reason")
        normal_termination = termination in {"agent_stop", "user_stop"}
        rows.append(
            {
                "task_id": str(sim.get("task_id")),
                "semantic_score": reward_info.get("reward"),
                "protocol_legal": protocol_legal,
                "n_turns": len(turns),
                "n_tool_calls": sum(t["n_tool_calls"] for t in turns),
                "n_illegal_turns": len(first_violations),
                "n_unknown_protocol_turns": sum(t["protocol_legal"] is None for t in turns),
                "first_violation": first_violations[0] if first_violations else None,
                "termination": termination,
                "normal_termination": normal_termination,
            }
        )
    from ..metrics import aggregate

    summary = aggregate(rows, cluster_key="task_id")
    summary["task_rows"] = rows
    termination_counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("termination") or "unknown")
        termination_counts[key] = termination_counts.get(key, 0) + 1
    normal_count = sum(row["normal_termination"] for row in rows)
    rewards = [row["semantic_score"] for row in rows
               if isinstance(row.get("semantic_score"), (int, float))]
    binary_rewards = bool(rewards) and all(float(value) in {0.0, 1.0}
                                           for value in rewards)
    summary["official_metric_kind"] = (
        "binary_task_success" if binary_rewards else "official_reward_mean")
    summary["official_success_rate"] = (
        sum(float(value) for value in rewards) / len(rewards)
        if binary_rewards else None)
    summary["normal_termination_rate"] = (
        normal_count / len(rows) if rows else None)
    summary["premature_termination_count"] = len(rows) - normal_count
    summary["termination_counts"] = termination_counts
    summary["protocol_evaluable_tasks"] = sum(row["protocol_legal"] is not None for row in rows)
    summary["protocol_tool_pool_size"] = len(tools)
    return summary


def _task_tools(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = traj.get("tools")
    if isinstance(tools, list):
        return tools
    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--user-base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    parser.add_argument("--model", default="c2kv-agent")
    args = parser.parse_args()
    # one code path: standalone use builds the same RunContext run.py builds
    summary = run(RunContext(
        base_url=args.base_url, user_base_url=args.user_base_url,
        out_dir=args.out, model=args.model, arm="full", run_name=args.run_name,
        options={"benchmark_dir": args.benchmark_dir, "task_set": args.task_set,
                 "num_workers": args.num_workers, "max_tasks": args.max_tasks},
    ))
    print(json.dumps(summary, indent=2))
