"""ACON-runner adapter: AppWorld and 8-objective QA behind the arm proxy.

Both benchmarks are driven through the runners shipped in microsoft/acon
(``experiments/appworld/run_all.py``, ``experiments/smolagents/run.py``) so
the ACON text arm and the KV arms share one agent loop, one prompt and one
scorer.  The runners' agent LLM is ``productive_agents.llm.vLLM``, an
OpenAI client whose endpoint is hard-coded upstream; the vendored
``benchmarks/acon_patches/0001-openai-base-url-env.patch`` makes it read
``ACON_OPENAI_BASE_URL`` / ``ACON_OPENAI_API_KEY``, which this adapter
exports as the arm proxy.  Neither benchmark has a user simulator, so no
second endpoint is involved.

What the proxy sees: ACON forwards its memory as a role-preserving message
list (system, user task, assistant code, user observation, ...), so every
turn before the latest observation is history under the training rule and
IS compressed.  Observations can be long: read ``dropped_docs`` in the
request log (turn packing keeps doc 0 + the last max_doc_num-1 docs); the
summary's ``dropped_docs_total`` / ``n_docs_max`` are the same fact rolled
up over the tasks the cost join matched (``n_cost_joined`` of ``n``).

Semantic column:
* ``qa``       — ACON's own EM/F1 (``predictions.jsonl``: SQuAD-normalised
                 exact match averaged over the 8 questions of a task; F1
                 alongside).  Data = ACON's shipped ``data/nq_multi_8``
                 (100 train / 100 test), used unchanged.
* ``appworld`` — the OFFICIAL scorer ``appworld evaluate <experiment>
                 <split>`` (state-based unit tests).  The runner's per-task
                 ``results.json:success`` only records that the agent called
                 ``complete_task``; it is carried as
                 ``agent_reported_success``, never as the semantic score.

Terminal-state check (acceptance 1): every task of the split (or of the
pinned id list) must have a scored row, else the run FAILS instead of
shrinking the denominator.

Resume semantics: ``run_all.py`` skips a task whose output directory exists
and ``run.py`` rewrites ``predictions.jsonl``; ``benchmarks/run.py`` suffixes
the tag with the git sha, so a code change never resumes into old
trajectories (same rule as tau2 ``--auto-resume``).

Contamination: AppWorld trajectories are 31.5 % of the ckpt-1088 training
pool (fork/task/d-repair-v2 ``inv_1088/a3_train_pool_benchmarks.json``);
AppWorld rows on that checkpoint carry the same CONTAMINATED label as tau2.
NQ / wiki-18 (the QA task) is not in that pool.

Usage (server; the acon venv needs ``smolagents`` for qa, ``appworld`` +
downloaded data for appworld, see the runner READMEs):
    python benchmarks/adapters/acon_adapter.py --kind qa \
        --base-url http://127.0.0.1:34100 --out results/bench/qa_c2kv --limit 5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import proxy, reqlog
from ..metrics import aggregate
from .base import RunContext, v1

NAME = "acon"
NAMES = ("acon_appworld", "acon_qa")  # one module, two --benchmark values
NAME_TO_KIND = {"acon_appworld": "appworld", "acon_qa": "qa"}
ACON_DIR = Path(os.environ.get("ACON_DIR") or Path.home() / "baselines" / "acon")
KINDS = ("appworld", "qa")
HISTORY_FILE = "llm_history.json"  # MemoryManager.dump_history (memory.py:199)
BASE_URL_ENV = "ACON_OPENAI_BASE_URL"  # read by the patched productive_agents.llm.vLLM
API_KEY_ENV = "ACON_OPENAI_API_KEY"

QA_DATA_FOLDER = "data/nq_multi_8"  # ACON's shipped split, byte-for-byte
QA_DEFAULT_SPLIT = "test"
QA_DEFAULT_MAX_ITER = 30  # experiments/smolagents/run.py CLI default
APPWORLD_DEFAULT_SPLIT = "test_normal"  # ACON's evaluation split (paper §8.1)
APPWORLD_DEFAULT_MAX_ITER = 50  # run_all.py default
APPWORLD_SEED = 42  # run_all.py default (ACON §8.3 fixes seed 42)


def add_arguments(parser) -> None:
    """ACON-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument("--acon-dir", type=Path, default=None,
                        help="acon checkout (default $ACON_DIR or ~/baselines/acon)")
    parser.add_argument("--split", default="",
                        help="acon_qa: train|test (default test); acon_appworld: "
                             "train|dev|test_normal|test_challenge (default test_normal)")
    parser.add_argument("--tag", default="",
                        help="ACON runner tag = output dir suffix (default: --run-name, "
                             "sha-suffixed like every other run)")
    parser.add_argument("--task-ids", default="",
                        help="acon_*: comma-separated task id pin list (smoke runs)")


def split_task_ids(raw) -> Optional[List[str]]:
    """``--task-ids a,b`` -> ["a", "b"]; empty -> None (whole split)."""
    if not raw:
        return None
    items = ([t.strip() for t in raw.split(",")] if isinstance(raw, str)
             else [str(t).strip() for t in raw])
    return [t for t in items if t] or None


def _sanitize(name: str) -> str:
    """``experiments/smolagents/run.py:_sanitize_for_path``, same rule."""
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in name)


def runner_env(base_url: str) -> Dict[str, str]:
    """Environment for either runner: the agent LLM endpoint is the arm proxy
    (the patched vLLM client appends nothing, so ``/v1`` is added here)."""
    return {
        **os.environ,
        BASE_URL_ENV: v1(base_url),
        API_KEY_ENV: "EMPTY",
        "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    }


def _terminal_check(benchmark: str, rows: List[Dict[str, Any]], expected: Optional[int]) -> None:
    if expected is None:
        return
    print(f"TERMINAL-STATE {benchmark}: n_scored={len(rows)} n_total={expected}")
    if len(rows) < expected:
        raise SystemExit(
            f"FATAL: {benchmark} terminal-state check failed: "
            f"n_scored={len(rows)} < n_total={expected}")


# ------------------------------------------------------------- cost join (ACON)

def conversation_ids(history_path: Path) -> List[str]:
    """The ``proxy.conversation_id``s ONE ACON task produced.

    Exact, not inferred.  The runner's OpenAI client sends
    ``[{"role": "system", "content": self.system_message}] + session[1:]``
    (llm.py:174-210 ``_build_messages`` on
    ``MemoryManager.get_conversation_history(exclude_system=True)``,
    unified_agent.py:367), and the session's own element 0 is that same
    system message (unified_agent.py:184 -> memory.py:112-121).  So each
    request is a PREFIX of the dumped session, and
    ``proxy.conversation_id`` (system head + first two non-system messages,
    proxy.py:434-447) takes exactly two values per session: the first
    request carries [system, user], every later one [system, user,
    assistant, ...].  Both are returned.

    A wrong key can only match nothing — the join report says so and the
    cost columns stay empty; it can never produce a wrong number.  (Model
    names containing "o1" are one such case: llm.py:179 then folds the
    system prompt into the user message instead of sending it.)
    """
    path = Path(history_path)
    try:
        sessions = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(sessions, list):
        return []
    ids: List[str] = []
    for session in sessions:
        if not isinstance(session, list):
            continue
        for size in (2, 3):
            if len(session) >= size:
                ids.append(proxy.conversation_id(session[:size]))
    return list(dict.fromkeys(ids))


def cost_join(rows: List[Dict[str, Any]], task_dir_of,
              request_log) -> Dict[str, Any]:
    """Fill the per-task cost columns from the proxy request log; returns the
    join REPORT (not just its prose line), because the summary needs the
    numeric denominator of the cost means as well —
    ``reqlog.cost_summary(rows, report)`` turns it into the summary block.
    ``task_dir_of(task_id)`` is the runner's per-task output directory (the
    one ``dump_history`` wrote)."""
    if not request_log:
        return reqlog.not_joinable(rows, "no request log for this run")
    log_rows = reqlog.read_rows(Path(request_log))
    return reqlog.join_by_conversation(
        rows, log_rows,
        lambda row: conversation_ids(Path(task_dir_of(str(row.get("task_id"))))
                                     / HISTORY_FILE))


# ---------------------------------------------------------------- 8-objective QA

def qa_command(python: str, model: str, tag: str, split: str, max_iter: int,
               limit: Optional[int] = None,
               id_list_file: Optional[Path] = None) -> List[str]:
    cmd = [python, "run.py", "--split", split, "--model_name", model, "--tag", tag,
           "--max_iter", str(max_iter), "--data_folder", QA_DATA_FOLDER]
    if limit:
        cmd += ["--limit", str(limit)]
    if id_list_file is not None:
        cmd += ["--id_list_file", str(id_list_file)]
    return cmd


def qa_fold(split: str) -> str:
    """run.py maps dev/validation/val (and anything unknown) to the test fold."""
    fold = (split or "test").lower()
    return "train" if fold == "train" else "test"


def qa_run_dir(acon_dir: Path, model: str, tag: str, split: str) -> Path:
    """run.py: ``outputs/<model>_<tag>/<fold>`` next to the script (the
    ``--output_dir`` flag is ignored upstream)."""
    return (Path(acon_dir) / "experiments" / "smolagents" / "outputs"
            / f"{_sanitize(model)}_{_sanitize(tag)}" / _sanitize(qa_fold(split)))


def qa_sample_dir(run_dir: Path, task_id: str) -> Path:
    """run.py:239 -> run_sample(output_base=<run_dir>/samples); run.py:92
    ``sample_dir = os.path.join(output_base, ex.id)`` is where dump_history
    writes ``llm_history.json``."""
    return Path(run_dir) / "samples" / str(task_id)


def qa_expected(acon_dir: Path, split: str, limit: Optional[int],
                task_ids: Optional[List[str]]) -> int:
    if task_ids:
        return len(task_ids)
    data = (Path(acon_dir) / "experiments" / "smolagents" / QA_DATA_FOLDER
            / f"{qa_fold(split)}.jsonl")
    n = sum(1 for line in data.read_text(encoding="utf-8").splitlines() if line.strip())
    return min(n, limit) if limit else n


def run_qa(base_url: str, out_dir: Path, acon_dir: Optional[Path] = None,
           model: str = "c2kv-agent", tag: str = "c2kv_run",
           split: str = QA_DEFAULT_SPLIT, max_iter: int = QA_DEFAULT_MAX_ITER,
           limit: Optional[int] = None, task_ids: Optional[List[str]] = None,
           python: Optional[str] = None,
           request_log: Optional[Path] = None) -> Dict[str, Any]:
    acon_dir = Path(acon_dir) if acon_dir else ACON_DIR
    python = python or sys.executable
    cwd = acon_dir / "experiments" / "smolagents"
    out_dir.mkdir(parents=True, exist_ok=True)
    id_list_file = None
    if task_ids:
        id_list_file = (out_dir / "qa_task_ids.txt").resolve()
        id_list_file.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    cmd = qa_command(python, model, tag, split, max_iter, limit=limit,
                     id_list_file=id_list_file)
    subprocess.run(cmd, cwd=cwd, env=runner_env(base_url), check=True)
    return collect_qa(qa_run_dir(acon_dir, model, tag, split),
                      expected=qa_expected(acon_dir, split, limit, task_ids),
                      request_log=request_log)


def collect_qa(run_dir: Path, expected: Optional[int] = None,
               request_log: Optional[Path] = None) -> Dict[str, Any]:
    """``predictions.jsonl`` rows (id, em, f1, iterations, success) -> unified
    rows; ``summary.json`` (avg_em / avg_f1 / total) is carried verbatim."""
    predictions = Path(run_dir) / "predictions.jsonl"
    if not predictions.exists():
        raise SystemExit(f"FATAL: no predictions.jsonl under {run_dir} — "
                         "the ACON QA runner produced nothing")
    rows: List[Dict[str, Any]] = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append({
            "task_id": str(rec.get("id")),
            "semantic_score": rec.get("em"),
            "f1": rec.get("f1"),
            "n_turns": rec.get("iterations"),
            "agent_reported_success": rec.get("success"),
            "protocol_legal": None,  # code-action agent: no tool-call schema
        })
    _terminal_check("acon_qa", rows, expected)
    report = cost_join(rows, lambda tid: qa_sample_dir(run_dir, tid), request_log)
    summary = aggregate(rows, cluster_key="task_id")
    # cost_join line + n_cost_joined (the *_mean denominator) + the three
    # joined fields aggregate does not mean
    summary.update(reqlog.cost_summary(rows, report))
    f1 = [float(r["f1"]) for r in rows if r.get("f1") is not None]
    summary["f1_mean"] = sum(f1) / len(f1) if f1 else None
    official = Path(run_dir) / "summary.json"
    if official.exists():
        summary["official_summary"] = json.loads(official.read_text(encoding="utf-8"))
    summary["run_dir"] = str(run_dir)
    return summary


# --------------------------------------------------------------------- AppWorld

def appworld_command(python: str, model: str, tag: str, split: str, max_iter: int,
                     task_ids: Optional[List[str]] = None,
                     seed: int = APPWORLD_SEED) -> List[str]:
    cmd = [python, "run_all.py", "--split", split, "--model_name", model, "--tag", tag,
           "--max_iter", str(max_iter), "--seed", str(seed)]
    if task_ids:
        cmd += ["--task_ids", *task_ids]
    return cmd


def appworld_evaluate_command(appworld_cli: str, model: str, tag: str,
                              split: str) -> List[str]:
    """Official scorer argv (state-based unit tests); the runner's own
    ``success`` flag is NOT a score."""
    return [appworld_cli, "evaluate", appworld_experiment(model, tag), split]


def appworld_task_dir(run_dir: Path, task_id: str) -> Path:
    """run_all.py:193 ``task_output_dir = <run_dir>/task_<id>`` — the dir
    run.py:173 dumps ``llm_history.json`` (and results.json) into."""
    return Path(run_dir) / f"task_{task_id}"


def appworld_experiment(model: str, tag: str) -> str:
    """run_all.py: ``experiment_name = f'{model_name.replace("/","_")}_{tag}'``
    — also the AppWorld experiment the official scorer is pointed at."""
    return f"{model.replace('/', '_')}_{tag}"


def appworld_run_dir(acon_dir: Path, model: str, tag: str, split: str) -> Path:
    """run_all.py per-task output: ``outputs/<experiment>/<split>/task_<id>/``."""
    return (Path(acon_dir) / "experiments" / "appworld" / "outputs"
            / appworld_experiment(model, tag) / split)


def appworld_eval_path(acon_dir: Path, model: str, tag: str, split: str) -> Path:
    """``appworld evaluate`` output, relative to APPWORLD_ROOT (= the runner
    cwd): ``experiments/outputs/<experiment>/evaluations/<split>.json``."""
    return (Path(acon_dir) / "experiments" / "appworld" / "experiments" / "outputs"
            / appworld_experiment(model, tag) / "evaluations" / f"{split}.json")


def _appworld_cli(python: str) -> str:
    sibling = Path(python).parent / "appworld"
    return str(sibling) if sibling.exists() else "appworld"


def appworld_split_size(python: str, split: str, cwd: Path, env: Dict[str, str]) -> int:
    code = ("import sys; from appworld import load_task_ids; "
            "print(len(load_task_ids(sys.argv[1])))")
    proc = subprocess.run([python, "-c", code, split], cwd=cwd, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip().isdigit():
        raise SystemExit(f"FATAL: cannot size AppWorld split {split!r} via "
                         f"appworld.load_task_ids: {proc.stderr.strip()[:500]}")
    return int(proc.stdout.strip())


def prepare_appworld_run(acon_dir: Path, out_dir: Path, split: str,
                         task_ids: Optional[List[str]] = None) -> Path:
    """Isolate mutable runner outputs and the scorer's exact dataset list.

    AppWorld's official CLI evaluates a dataset name, not ``--task_ids``.
    A private dataset file therefore binds generation and scoring to the
    same selected tasks without editing the installed official dataset.
    Large immutable databases and task files remain linked to the data root.
    """
    if not split or Path(split).name != split or split in {".", ".."}:
        raise SystemExit(f"FATAL: invalid AppWorld split: {split!r}")
    source = Path(acon_dir) / "experiments" / "appworld"
    data = Path(os.environ.get("APPWORLD_ROOT", str(source))) / "data"
    dataset = data / "datasets" / f"{split}.txt"
    if not dataset.is_file():
        raise SystemExit(f"FATAL: missing AppWorld dataset {dataset}")
    available = [line.strip() for line in dataset.read_text().splitlines() if line.strip()]
    selected = list(task_ids) if task_ids else available
    if not selected or len(selected) != len(set(selected)):
        raise SystemExit("FATAL: AppWorld task selection is empty or contains duplicates")
    unknown = set(selected) - set(available)
    if unknown:
        raise SystemExit(f"FATAL: AppWorld tasks are outside {split}: {sorted(unknown)}")
    root = Path(out_dir).resolve() / "appworld_harness"
    cwd = root / "experiments" / "appworld"
    shutil.copytree(source, cwd, ignore=shutil.ignore_patterns(
        "data", "outputs", "experiments", "__pycache__"))
    private_data = cwd / "data"
    private_data.mkdir()
    for entry in data.iterdir():
        target = private_data / entry.name
        if entry.name == "datasets":
            shutil.copytree(entry, target)
        else:
            target.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
    (private_data / "datasets" / f"{split}.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8")
    (Path(out_dir) / "selected_tasks.json").write_text(json.dumps({
        "benchmark": "appworld", "source_dataset": str(dataset),
        "split": split, "source_count": len(available), "task_ids": selected,
    }, indent=2) + "\n", encoding="utf-8")
    return root


def run_appworld(base_url: str, out_dir: Path, acon_dir: Optional[Path] = None,
                 model: str = "c2kv-agent", tag: str = "c2kv_run",
                 split: str = APPWORLD_DEFAULT_SPLIT,
                 max_iter: int = APPWORLD_DEFAULT_MAX_ITER,
                 task_ids: Optional[List[str]] = None,
                 python: Optional[str] = None,
                 request_log: Optional[Path] = None) -> Dict[str, Any]:
    acon_dir = Path(acon_dir) if acon_dir else ACON_DIR
    python = python or sys.executable
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = prepare_appworld_run(acon_dir, out_dir, split, task_ids)
    cwd = run_root / "experiments" / "appworld"
    env = {**runner_env(base_url), "APPWORLD_ROOT": str(cwd)}
    subprocess.run(appworld_command(python, model, tag, split, max_iter, task_ids),
                   cwd=cwd, env=env, check=True)
    # official scorer (state-based unit tests); the runner's own success flag
    # is not a score
    subprocess.run(appworld_evaluate_command(_appworld_cli(python), model, tag, split),
                   cwd=cwd, env=env, check=True)
    expected = len(task_ids) if task_ids else appworld_split_size(python, split, cwd, env)
    return collect_appworld(appworld_eval_path(run_root, model, tag, split),
                            appworld_run_dir(run_root, model, tag, split),
                            expected=expected, request_log=request_log,
                            expected_ids=task_ids)


_TASK_SECTIONS = ("individual", "tasks", "per_task", "task_results", "results")
_SUCCESS_KEYS = ("success", "passed", "pass", "is_success", "solved")


def _task_ok(record: Any) -> Optional[bool]:
    if isinstance(record, bool):
        return record
    if isinstance(record, dict):
        for key in _SUCCESS_KEYS:
            if isinstance(record.get(key), bool):
                return record[key]
        fails = record.get("fails")
        if isinstance(fails, list) and isinstance(record.get("passes"), list):
            return len(fails) == 0
    return None


def appworld_per_task(data: Any) -> Dict[str, bool]:
    """Per-task pass/fail from the ``appworld evaluate`` JSON.

    The AppWorld docs pin the content (per-task pass/fail, passes/fails
    arrays, TGC/SGC aggregates) but this repo has not pinned the key names
    against a live file, so the shapes recognised here are explicit and
    anything else FAILS LOUDLY with the top-level keys — the first live run
    then shows the exact layout and the fix is one entry in the two tuples
    above, never a silent empty table."""
    if isinstance(data, dict):
        for key in _TASK_SECTIONS:
            section = data.get(key)
            if isinstance(section, dict) and section:
                out = {str(t): _task_ok(r) for t, r in section.items()}
                if all(v is not None for v in out.values()):
                    return out  # type: ignore[return-value]
            if isinstance(section, list) and section:
                out = {}
                complete = True
                for rec in section:
                    if not isinstance(rec, dict) or rec.get("task_id") is None:
                        complete = False
                        break
                    ok = _task_ok(rec)
                    if ok is None:
                        complete = False
                        break
                    out[str(rec["task_id"])] = ok
                if complete and out:
                    return out
    shape = list(data.keys())[:20] if isinstance(data, dict) else type(data).__name__
    raise SystemExit("FATAL: unrecognised appworld evaluation layout "
                     f"(top-level: {shape}); extend appworld_per_task")


def collect_appworld(eval_path: Path, run_dir: Path,
                     expected: Optional[int] = None,
                     request_log: Optional[Path] = None,
                     expected_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise SystemExit(f"FATAL: appworld evaluate wrote no {eval_path}")
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    scored = appworld_per_task(data)
    if expected_ids is not None and set(scored) != set(expected_ids):
        raise SystemExit("FATAL: AppWorld scored task IDs differ from selected tasks")
    if expected is not None and len(scored) != expected:
        raise SystemExit(f"FATAL: AppWorld n_scored={len(scored)} != n_total={expected}")
    for task_id, ok in sorted(scored.items()):
        row: Dict[str, Any] = {"task_id": task_id,
                               "semantic_score": 1.0 if ok else 0.0,
                               "protocol_legal": None}  # code-action agent
        agent = appworld_task_dir(run_dir, task_id) / "results.json"
        if agent.exists():
            rec = json.loads(agent.read_text(encoding="utf-8"))
            row.update({
                "n_turns": rec.get("iterations"),
                "termination": rec.get("termination_reason"),
                "agent_reported_success": rec.get("success"),
            })
        rows.append(row)
    _terminal_check("acon_appworld", rows, expected)
    report = cost_join(rows, lambda tid: appworld_task_dir(run_dir, tid),
                       request_log)
    summary = aggregate(rows, cluster_key="task_id")
    summary.update(reqlog.cost_summary(rows, report))
    if isinstance(data, dict):
        scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
        summary["official_aggregate"] = scalars or data.get("aggregate")
    summary["evaluation_path"] = str(eval_path)
    summary["run_dir"] = str(run_dir)
    return summary


# ------------------------------------------------------------------- dispatch

def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry for both ``--benchmark acon_qa`` and ``acon_appworld``
    (one module, two registry names — ``ctx.options["benchmark"]`` picks).

    Neither benchmark has a user simulator, so ``ctx.user_base_url`` is
    unused.  Cost columns ARE joined here (see ``conversation_ids``).
    """
    benchmark = str(ctx.options.get("benchmark") or "")
    if benchmark not in NAME_TO_KIND:
        # never guess: a typo must not silently run the OTHER benchmark
        raise SystemExit(f"FATAL: acon_adapter got --benchmark {benchmark!r}; "
                         f"known: {NAMES}")
    kind = NAME_TO_KIND[benchmark]
    common = dict(
        acon_dir=ctx.opt("acon_dir"), model=ctx.model,
        tag=ctx.opt("tag", ctx.run_name),
        task_ids=split_task_ids(ctx.opt("task_ids")),
        python=ctx.opt("bench_python"),
        request_log=ctx.request_log,
    )
    if kind == "qa":
        return run_qa(ctx.base_url, ctx.out_dir,
                      split=ctx.opt("split", QA_DEFAULT_SPLIT),
                      max_iter=ctx.opt("max_iter", QA_DEFAULT_MAX_ITER),
                      limit=ctx.opt("max_tasks"), **common)
    return run_appworld(ctx.base_url, ctx.out_dir,
                        split=ctx.opt("split", APPWORLD_DEFAULT_SPLIT),
                        max_iter=ctx.opt("max_iter", APPWORLD_DEFAULT_MAX_ITER),
                        **common)


def run_kind(kind: str, base_url: str, out_dir: Path, **kwargs) -> Dict[str, Any]:
    """Standalone entry (this module's own CLI): dispatch by kind name."""
    if kind == "qa":
        return run_qa(base_url, out_dir, **kwargs)
    if kind == "appworld":
        kwargs.pop("limit", None)  # run_all.py has no --limit; use --task_ids
        return run_appworld(base_url, out_dir, **kwargs)
    raise SystemExit(f"FATAL: unknown ACON benchmark kind {kind!r}; known: {KINDS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acon-dir", type=Path, default=None,
                        help="acon checkout (default $ACON_DIR or ~/baselines/acon)")
    parser.add_argument("--model", default="c2kv-agent")
    parser.add_argument("--tag", default="c2kv_run")
    parser.add_argument("--split", default=None,
                        help="qa: train|test (default test); appworld: "
                             "train|dev|test_normal|test_challenge (default test_normal)")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="qa only")
    parser.add_argument("--task-ids", default="", help="comma-separated pin list")
    parser.add_argument("--python", default=None,
                        help="python of the acon venv (default: this interpreter)")
    args = parser.parse_args()
    defaults = {"qa": (QA_DEFAULT_SPLIT, QA_DEFAULT_MAX_ITER),
                "appworld": (APPWORLD_DEFAULT_SPLIT, APPWORLD_DEFAULT_MAX_ITER)}
    split, max_iter = defaults[args.kind]
    summary = run_kind(
        args.kind, args.base_url, args.out,
        acon_dir=args.acon_dir, model=args.model, tag=args.tag,
        task_ids=split_task_ids(args.task_ids), python=args.python,
        split=args.split or split, max_iter=args.max_iter or max_iter,
        limit=args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
