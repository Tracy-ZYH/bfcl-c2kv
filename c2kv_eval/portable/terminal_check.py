"""Terminal-state check: every benchmark entry must end with a recorded state.

A run where entries silently vanish (transport drop, killed mid-run, client
exception) must FAIL instead of producing a deceptively small denominator
(acceptance 1: n_scored == n_total).  Pure stdlib; designed to be called by
bench_queue/run_one_task.sh after each benchmark, and by run.py adapters.

Exit codes: 0 = every expected id has a terminal state; 1 = shortfall (the
caller must void the run); 2 = artifacts not found.

Usage:
  python terminal_check.py tau2  --run hr2_tau2_hy3 [--expected 50]
  python terminal_check.py bfcl  [--expected 200] [--run-ids a,b,c]
  python terminal_check.py ts    --run hr2_ts_hy3 [--expected 16]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

TAU2_SIMS = (Path(os.environ.get("TAU2_DIR") or Path.home() / "benchmarks/tau2")
             / "data/simulations")
TS_RESULTS = Path.home() / "bench_results"
GORILLA = Path(
    os.environ.get(
        "BFCL_CHECKOUT",
        str(Path.home() / "benchmarks/gorilla/berkeley-function-call-leaderboard"),
    )
)
DEFAULT_EXPECTED = {"tau2": 50, "bfcl": 200, "ts": None}


def fail(benchmark: str, n_scored: int, n_total: int, missing) -> int:
    print(f"TERMINAL-STATE {benchmark}: n_scored={n_scored} n_total={n_total} "
          f"missing={len(missing)}")
    if missing:
        shown = ",".join(map(str, missing[:20]))
        more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
        print(f"FATAL: {benchmark} run has no terminal state for: {shown}{more}")
        return 1
    return 0


def check_tau2(run: str, expected, task_ids: str = "",
               root: "Path | str | None" = None) -> int:
    simulations_root = Path(root) if root is not None else TAU2_SIMS
    path = simulations_root / run / "results.json"
    if not path.exists():
        print(f"FATAL: no tau2 results at {path}")
        return 2
    sims = (json.loads(path.read_text(encoding="utf-8")).get("simulations")) or []
    got = {str(s.get("task_id")) for s in sims}
    # a simulation that died on infrastructure_error is NOT a valid
    # terminal state — counting it as scored let a 25/25-infrastructure-error
    # chunk pass the gate (the run is void, not complete)
    infra = {str(s.get("task_id")) for s in sims
             if s.get("termination_reason") == "infrastructure_error"}
    if task_ids:
        # id-exact over the pinned subset (ids are task_id strings, not
        # necessarily 0..N when a TASK_IDS subset was run)
        want = {t.strip() for t in task_ids.replace(",", " ").split() if t.strip()}
    else:
        if expected is None:
            expected = DEFAULT_EXPECTED["tau2"]
        want = {str(i) for i in range(expected)}
    missing = sorted((want - got) | (want & infra))
    return fail("tau2", len((got & want) - infra), len(want), missing)


def check_bfcl(expected, run_ids, handler: str = "c2kv-hf",
               category: str = "multi_turn_base",
               root: "Path | str | None" = None) -> int:
    """``handler`` is the BFCL model key the run registered (result dir
    name).  run.py registers one per arm (c2kv-<arm>); hardcoding c2kv-hf
    made every non-default arm read the WRONG directory — or stale files
    when an old c2kv-hf dir was lying around (audit BLOCKER).

    ``category`` is the BFCL test category the run generated; bfcl_eval
    files results per category family (result/<model>/<family>/BFCL_v4_
    <category>_result.json), so the glob is recursive — hard-wiring
    multi_turn/ left every other category (memory, long_context, ...)
    without a terminal gate."""
    project_root = Path(
        root or os.environ.get("BFCL_PROJECT_ROOT") or GORILLA
    )
    pattern = str(project_root / f"result/{handler}/**/*{category}_result.json")
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        print(f"FATAL: no bfcl result file under {pattern}")
        return 2
    got = set()
    with open(hits[-1], "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("id") is not None:
                got.add(str(row["id"]))
    if run_ids:
        # id-exact check when the caller pinned the id list
        want = {r.strip() for r in run_ids.split(",") if r.strip()}
        missing = sorted(want - got)
        return fail("bfcl", len(got & want), len(want), missing)
    # count-based otherwise (bfcl ids are category-prefixed, not numeric)
    expected = expected if expected is not None else DEFAULT_EXPECTED["bfcl"]
    missing = ["?"] * max(0, expected - len(got))
    return fail("bfcl", len(got), expected, missing)


def check_ts(run: str, expected, scenarios: str = "") -> int:
    hits = sorted(glob.glob(str(TS_RESULTS / f"task_{run}" / "*" / "result_summary.json")))
    if not hits:
        print(f"FATAL: no result_summary.json under {TS_RESULTS / f'task_{run}'}")
        return 2
    data = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    per = data.get("per_scenario_results") or []
    got = {str(r.get("name")) for r in per}
    if scenarios:
        want = {s.strip() for s in scenarios.replace(",", " ").split() if s.strip()}
        missing = sorted(want - got)
        return fail("ts", len(got & want), len(want), missing)
    missing: list = []
    if expected is not None:
        missing = ["?"] * max(0, expected - len(got))
    return fail("ts", len(got), expected if expected is not None else len(got), missing)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="benchmark", required=True)
    for name in ("tau2", "ts"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--expected", type=int, default=None)
        p.add_argument("--task-ids", default="",
                       help="tau2: id-exact check over this subset")
        p.add_argument("--scenarios", default="",
                       help="ts: id-exact check over this subset")
        if name == "tau2":
            p.add_argument("--tau2-dir", type=Path, default=None,
                           help="tau2 checkout (default: $TAU2_DIR)")
    p = sub.add_parser("bfcl")
    p.add_argument("--expected", type=int, default=None)
    p.add_argument("--run-ids", default="")
    p.add_argument("--handler", default="c2kv-hf",
                   help="BFCL model key / result-dir name (run.py: c2kv-<arm>)")
    p.add_argument("--category", default="multi_turn_base",
                   help="BFCL test category the run generated")
    args = parser.parse_args(argv)

    if args.benchmark == "tau2":
        root = args.tau2_dir / "data" / "simulations" if args.tau2_dir else None
        code = check_tau2(args.run, args.expected, args.task_ids, root=root)
    elif args.benchmark == "bfcl":
        code = check_bfcl(args.expected, args.run_ids, handler=args.handler,
                          category=args.category)
    else:
        code = check_ts(args.run, args.expected, args.scenarios)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
