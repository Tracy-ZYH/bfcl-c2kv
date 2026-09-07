"""Unified runner: one benchmark x one arm -> unified rows + summary.

Registry dispatch over the per-benchmark adapters (contract:
``adapters/base.py`` — NAME, add_arguments, run(ctx)); this module owns only
what is common to every benchmark: the proxy lifecycle (spawn
benchmarks/proxy.py in the requested arm, tear it down after), the git-sha
suffix on --run-name/--out, and the summary envelope.  The compression ratio
comes from the arm registry (arms.py), NOT from a --ratio flag.

Adding a benchmark = one adapter module + one ADAPTERS entry. Server
scripts use this CLI; checkpoint profiles resolve packing and projection
before the proxy starts, and conflicting explicit overrides are rejected.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .adapters import (
    acebench_adapter, acon_adapter, bfcl_adapter, tau2_adapter,
    toolsandbox_adapter,
)
from .adapters.base import RunContext
from .checkpoint_profile import ProfileError, resolve_checkpoint_profile
from .capabilities import run_preflight
from .arms import get_arm

# --benchmark value -> adapter module.  Two names share acon_adapter (the
# module dispatches on ctx.options["benchmark"]); add_arguments is called
# once per MODULE, so a shared flag would collide in argparse and must stay
# in the core block of main() instead.
ADAPTERS = {
    "tau2": tau2_adapter,
    "bfcl": bfcl_adapter,
    "toolsandbox": toolsandbox_adapter,
    "acon_appworld": acon_adapter,
    "acon_qa": acon_adapter,
    "acebench": acebench_adapter,
}
assert set(acon_adapter.NAMES) <= set(ADAPTERS)
assert all(module.NAME in ADAPTERS or name in getattr(module, "NAMES", ())
           for name, module in ADAPTERS.items())


def start_proxy(upstream: str, arm: str, port: int, log_dir: Path,
                record_reference: str = "", reference: str = "",
                backend: str = "sglang", doc_packing: str = "turn",
                max_doc_length: int = 512, max_doc_num: int = 12,
                query_projection: str | None = None, witness_tokenizer: str = "",
                python_bin: str | None = None, recovery_control: str = ""):
    log_path = log_dir / f"proxy_{arm}_{port}.jsonl"
    out_handle = open(log_dir / f"proxy_{arm}_{port}.out", "w")
    command = [
        python_bin or sys.executable, "-m", "c2kv_eval.portable.proxy",
        "--upstream", upstream, "--arm", arm, "--backend", backend,
        "--port", str(port), "--request-log", str(log_path),
        "--doc-packing", doc_packing,
        "--max-doc-length", str(max_doc_length),
        "--max-doc-num", str(max_doc_num),
    ]
    if record_reference:
        command += ["--record-reference", record_reference]
    if reference:
        command += ["--reference", reference]
    if query_projection:
        command += ["--query-projection", query_projection]
    if witness_tokenizer:
        command += ["--witness-tokenizer", witness_tokenizer]
    if recovery_control:
        command += ["--recovery-control", recovery_control]
    proc = subprocess.Popen(
        command,
        stdout=out_handle,
        stderr=subprocess.STDOUT,
    )
    import urllib.request

    # never route the local health probe through an ambient http_proxy
    # (an inherited proxy env once made every run.py launch fail its own
    # gateway check)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    for _ in range(100):
        try:
            opener.open(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc, log_path
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit(f"proxy did not come up on port {port}")


def _git_short_sha() -> str:
    """Short commit of the running tree: run_name/out-dir suffix so a code
    change can never silently mix old and new trajectories via tau2's
    --auto-resume (which keeps every normally-terminated simulation)."""
    import subprocess
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, timeout=10).stdout.strip()
        return sha or "nogit"
    except Exception:
        return "nogit"


def add_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags run.py itself owns: the proxy/arm/serving knobs, plus the few
    flags MORE THAN ONE adapter reads (argparse refuses a duplicate option
    string, so those cannot live in an adapter's add_arguments)."""
    parser.add_argument("--benchmark", required=True, choices=list(ADAPTERS))
    parser.add_argument("--arm", required=True)
    parser.add_argument("--recovery-control", default="",
                        help="JSON control for one optional pre-return generation retry (append/replace/retry_full)")
    parser.add_argument("--upstream", required=True,
                        help="backend base URL (REQUIRED: a default here once "
                             "silently aimed sglang runs at the hf_server port)")
    parser.add_argument("--user-upstream", default="",
                        help="base URL for user-simulator/judge traffic (defaults to --upstream; only the agent arm proxy compresses)")
    parser.add_argument("--proxy-port", type=int, default=34100)
    parser.add_argument("--proxy-python", default=None,
                        help="proxy interpreter; use the serving environment for compatible checkpoint tokenizers")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exact-out", action="store_true",
                        help="use the supplied output directory verbatim (matrix cells)")
    # shared by tau2 (--max-concurrency) and acebench (--num-threads)
    parser.add_argument("--num-workers", type=int, default=4)
    # shared by tau2 (--num-tasks) and acon_qa (--limit)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    parser.add_argument("--record-reference", default="",
                        help="full-arm run: write the reference trajectory jsonl "
                             "the recover arms diff against")
    parser.add_argument("--reference", default="",
                        help="recover arm: reference trajectory jsonl (from a "
                             "--record-reference full-arm run)")
    parser.add_argument("--backend", default="sglang", choices=["sglang"],
                        help="SGLang serving stack behind the proxy")
    parser.add_argument("--model", default="c2kv-agent",
                        help="served model name at the endpoint (any "
                             "OpenAI-compatible endpoint serves any name; "
                             "tau2 agent/user LLMs and the BFCL handler "
                             "both use it; toolsandbox role keys are "
                             "separate, see --ts-agent)")
    parser.add_argument("--checkpoint", type=Path,
                        help="local checkpoint served by --upstream")
    parser.add_argument("--tokenizer", type=Path,
                        help="tokenizer used by witness selection; defaults to "
                             "--checkpoint only when they are the same directory")
    parser.add_argument("--checkpoint-profile", type=Path,
                        help="explicit checkpoint profile JSON; otherwise discovered beside checkpoint")
    parser.add_argument("--expected-profile-fingerprint",
                        help="matrix binding; reject a profile file changed after planning")
    parser.add_argument("--reference-profile", choices=["checkpoint-1088"],
                        help="explicit historical reference recipe; its missing training provenance remains recorded")
    parser.add_argument("--allow-unprofiled", action="store_true",
                        help="opt into the legacy 512/12 recipe without claiming checkpoint alignment")
    parser.add_argument("--query-projection", choices=["base", "gist"],
                        help="explicit legacy checkpoint query regime; checked against any saved profile")
    parser.add_argument("--capability-features", default="",
                        help="comma-separated verified server/harness capabilities")
    parser.add_argument("--doc-packing", choices=["turn", "message"],
                        help="must agree with the checkpoint profile when supplied")
    parser.add_argument("--max-doc-length", type=int,
                        help="checkpoint profile document token limit")
    parser.add_argument("--max-doc-num", type=int,
                        help="checkpoint profile maximum history documents")
    # shared by acon_* (agent step cap) and acebench (--max-dialog-turns)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="acon_qa/acon_appworld: agent step cap (runner defaults "
                             "30/50); acebench: --max-dialog-turns (default 40)")
    # shared by acon_* and acebench
    parser.add_argument("--bench-python", default="",
                        help="python of the harness venv for acon_*/acebench "
                             "(default: this interpreter)")


def build_parser() -> argparse.ArgumentParser:
    """Core flags + every adapter's own flags (each module once)."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_core_arguments(parser)
    for module in dict.fromkeys(ADAPTERS.values()):
        module.add_arguments(parser)
    return parser


def build_context(args: argparse.Namespace, request_log: Path) -> RunContext:
    """The one object an adapter receives.  ``options`` is the whole parsed
    namespace, so an adapter reads exactly the flags it registered (plus the
    core ones) and run.py needs no per-benchmark knowledge."""
    options = dict(vars(args))
    options["benchmark"] = args.benchmark
    return RunContext(
        base_url=f"http://127.0.0.1:{args.proxy_port}",
        user_base_url=args.user_upstream or args.upstream,
        out_dir=args.out,
        model=args.model,
        arm=args.arm,
        run_name=args.run_name,
        request_log=Path(request_log) if request_log else None,
        options=options,
    )


def resolve_run_profile(args: argparse.Namespace) -> dict:
    """Resolve once and reject conflicting CLI values before any model call."""
    if args.checkpoint is None:
        if args.checkpoint_profile or args.reference_profile:
            raise ProfileError("--checkpoint is required with a checkpoint/reference profile")
        if not args.allow_unprofiled:
            raise ProfileError("supply --checkpoint (and its profile), or explicitly use --allow-unprofiled")
        for key, default in (("doc_packing", "turn"), ("max_doc_length", 512),
                             ("max_doc_num", 12)):
            if getattr(args, key) is None:
                setattr(args, key, default)
        return {"profile_kind": "unprofiled", "serving": {
            key: getattr(args, key) for key in
            ("doc_packing", "max_doc_length", "max_doc_num", "query_projection")},
            "missing": ["checkpoint identity and training alignment are unverified"]}
    profile = resolve_checkpoint_profile(
        args.checkpoint, profile_path=args.checkpoint_profile,
        reference_profile=args.reference_profile,
        query_projection=args.query_projection, require_serving_e2e=True)
    if (args.expected_profile_fingerprint is not None
            and args.expected_profile_fingerprint != profile.get("profile_fingerprint")):
        raise ProfileError("resolved checkpoint profile differs from the planned profile fingerprint")
    serving = profile["serving"]
    for key in ("doc_packing", "max_doc_length", "max_doc_num", "query_projection"):
        explicit = getattr(args, key)
        resolved = serving[key]
        if explicit is not None and explicit != resolved:
            raise ProfileError(f"--{key.replace('_', '-')}={explicit!r} conflicts with checkpoint profile {resolved!r}")
        setattr(args, key, resolved)
    return profile


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        profile = resolve_run_profile(args)
    except ProfileError as exc:
        parser.error(str(exc))
    preflight = run_preflight(args.benchmark, args.arm, args.backend,
                              options=vars(args), profile=profile)

    sha = _git_short_sha()
    if sha not in (args.run_name or ""):
        args.run_name = f"{args.run_name}_{sha}"
    if not args.exact_out and sha not in str(args.out):
        args.out = args.out.with_name(f"{args.out.name}_{sha}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoint_profile.resolved.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "preflight.json").write_text(
        json.dumps(preflight.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    preflight.raise_for_errors()
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    proxy_proc, request_log = start_proxy(
        args.upstream, args.arm, args.proxy_port, log_dir,
        record_reference=args.record_reference, reference=args.reference,
        backend=args.backend, doc_packing=args.doc_packing,
        max_doc_length=args.max_doc_length, max_doc_num=args.max_doc_num,
        query_projection=args.query_projection,
        witness_tokenizer=str(args.tokenizer or args.checkpoint or "")
        if get_arm(args.arm).gold_recovery or args.recovery_control else "",
        python_bin=args.proxy_python, recovery_control=args.recovery_control)
    try:
        # every adapter owns its own "/v1" (adapters/base.py:v1) and its own
        # cwd; run.py hands over the bare proxy URL and nothing else
        ctx = build_context(args, request_log)
        adapter_started = time.perf_counter()
        summary = ADAPTERS[args.benchmark].run(ctx)
        adapter_wall_sec = time.perf_counter() - adapter_started
    finally:
        proxy_proc.terminate()
    summary["arm"] = args.arm
    summary["benchmark"] = args.benchmark
    summary["backend"] = args.backend
    summary["model"] = args.model
    summary["runner_adapter_wall_sec"] = adapter_wall_sec
    summary["runner_wall_scope"] = "adapter invocation including generation, tool execution and official scoring; excludes model server startup"
    summary["num_workers"] = args.num_workers
    summary["checkpoint_profile"] = profile
    summary["preflight"] = preflight.as_dict()
    if get_arm(args.arm).text_policy:
        # text-arm consumers: degeneration and compressor cost surfaced at
        # the RUN level (the per-request stats live in the request log)
        ta_rows = []
        try:
            with open(request_log, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line).get("textarm")
                        if isinstance(row, dict):
                            ta_rows.append(row)
        except OSError:
            pass
        degenerate_requests = sum(1 for t in ta_rows if t.get("degenerate"))
        compressed_requests = sum(
            1 for t in ta_rows if t.get("history_compressed"))
        summary["textarm_summary"] = {
            "textarm_requests": len(ta_rows),
            "degenerate_requests": degenerate_requests,
            "degenerate_arm": bool(
                ta_rows and degenerate_requests == len(ta_rows)),
            "history_compressed_requests": compressed_requests,
            "compressor_calls": sum(int(t.get("n_compressor_calls") or 0)
                                    for t in ta_rows),
            "compressor_prompt_tokens": sum(
                int((t.get("compressor_usage") or {}).get("prompt_tokens") or 0)
                for t in ta_rows),
            "compressor_completion_tokens": sum(
                int((t.get("compressor_usage") or {}).get("completion_tokens") or 0)
                for t in ta_rows),
            "compressor_wall_sec": round(sum(
                float((t.get("compressor_usage") or {}).get("wall_sec") or 0)
                for t in ta_rows), 1),
            "retrieval_calls": sum(int((t.get("retrieval_usage") or {}).get("calls") or 0)
                                   for t in ta_rows),
            "retrieval_prompt_tokens": sum(int((t.get("retrieval_usage") or {}).get("prompt_tokens") or 0)
                                           for t in ta_rows),
            "retrieval_completion_tokens": sum(int((t.get("retrieval_usage") or {}).get("completion_tokens") or 0)
                                               for t in ta_rows),
        }
        if (get_arm(args.arm).text_policy.startswith("acon_hist") and ta_rows
                and compressed_requests == 0):
            print(f"WARNING: arm {args.arm!r} NEVER compressed history in "
                  f"{len(ta_rows)} requests — the trigger never fired; the "
                  f"row is effectively a full arm under the acon label")
        if any(t.get("degenerate") for t in ta_rows):
            print(f"WARNING: arm {args.arm!r} ran DEGENERATE (no Subgoal "
                  f"segments) on {sum(1 for t in ta_rows if t.get('degenerate'))}"
                  f"/{len(ta_rows)} requests — effectively a full arm")
    # every arm: regime facts from the request log (doc drops, projection
    # mode mix, outcome mix, wall percentiles) — a compressed number whose
    # run dropped docs or mixed query_proj modes is not one regime
    from . import reqlog

    summary["request_log_summary"] = reqlog.summarize_file(request_log)
    rl = summary["request_log_summary"]
    if rl.get("dropped_requests"):
        print(f"NOTE: {rl['dropped_requests']}/{rl['n_ok']} requests dropped history "
              f"docs (turn packing, max_doc_num={args.max_doc_num}); mean dropped "
              f"{rl['dropped_docs_mean']}")
    if rl.get("mixed_query_proj"):
        print(f"WARNING: request log mixes c2kv_query_proj modes {rl['c2kv_query_proj']} "
              "— not one serving regime")
    summary["doc_packing"] = args.doc_packing
    summary["max_doc_length"] = args.max_doc_length
    summary["max_doc_num"] = args.max_doc_num
    summary["request_log"] = str(request_log)
    if args.reference:
        summary["reference"] = args.reference
    if args.record_reference:
        summary["record_reference"] = args.record_reference
    (args.out / f"summary_{args.arm}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
