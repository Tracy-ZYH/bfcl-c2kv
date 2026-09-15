"""Merge verified v1 completed arms with their retry-v2 replacements."""
import argparse
import collections
import csv
import json
import math
from pathlib import Path
from c2kv_eval.portable.ops.summarize_compression_agnostic_recovery import _row

METHODS = ["c2kv4", "c2kv4_replace_w2", "c2kv4_append_w2",
           "streamingllm_r25", "streamingllm_r25_replace_w2", "streamingllm_r25_append_w2",
           "h2o_r25", "snapkv_r25", "pyramidkv_r25", "pyramidkv_r25_replace_w2",
           "pyramidkv_r25_append_w2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1", type=Path, required=True)
    p.add_argument("--retry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError(f"Refusing existing output: {a.out}")
    if not (a.retry / "FULL_COMPLETE").exists():
        raise RuntimeError("Retry completion marker missing")
    rows, diagnostics = [], []
    for i, method in enumerate(METHODS):
        root = a.v1 if i < 6 else a.retry
        cell = root / "full/tau2" / method
        paths = list(cell.glob("summary_*.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one summary: {cell}")
        row = _row(paths[0], root)
        result = json.loads((cell / "tau2_updated_results.json").read_text())
        sims = result["simulations"]
        ids = [str(s["task_id"]) for s in sims]
        if ids != [str(i) for i in range(50)] or row["num_cases"] != 50:
            raise RuntimeError(f"Case completeness/order mismatch: {cell}")
        rewards = [(s.get("reward_info") or {}).get("reward") for s in sims]
        if any(r is None for r in rewards):
            raise RuntimeError(f"Missing official reward: {cell}")
        if not math.isclose(sum(rewards) / 50, row["official_task_score"]):
            raise RuntimeError(f"Official reward/summary mismatch: {cell}")
        terminations = collections.Counter(s["termination_reason"] for s in sims)
        if terminations["infrastructure_error"]:
            raise RuntimeError(f"Infrastructure termination: {cell}")
        proxy = next((cell / "logs").glob("proxy_*.jsonl"))
        status = collections.Counter()
        for line in proxy.open():
            status[json.loads(line).get("status")] += 1
        if set(status) != {"ok"} or row["request_errors"] != 0 or row["duplicate_raw_kv_tokens"] != 0:
            raise RuntimeError(f"Request/duplicate-KV gate failed: {cell}: {status}")
        row.update(source_run=root.name, correct_count=sum(r == 1 for r in rewards),
                   timeout_count=terminations["timeout"], max_steps_count=terminations["max_steps"],
                   too_many_errors_count=terminations["too_many_errors"],
                   normal_count=terminations["user_stop"], runtime_hours=row["runtime_sec"] / 3600,
                   score_semantics="official_tau2_reward_mean",
                   recovery_success_semantics="request_local_operator_success; not task_repair_success")
        rows.append(row)
        diagnostics.append(dict(method=method, source_run=str(root), request_status=dict(status),
                                terminations=dict(terminations),
                                failed_task_ids=[s["task_id"] for s,r in zip(sims,rewards) if r < 1],
                                timeout_task_ids=[s["task_id"] for s in sims if s["termination_reason"] == "timeout"],
                                case_order_verified=True))
    a.out.mkdir(parents=True)
    with (a.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    def fmt(v, pattern):
        return pattern.format(v) if v is not None else "—"
    lines = ["# tau2 airline50 — v1 + retry_v2", "",
             "| Method | Source | Correct/50 | Official score | Normal termination | Timeouts | Effective history compression | Calls/request | Runtime (h) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {'v1' if r['source_run']==a.v1.name else 'retry_v2'} | {r['correct_count']}/50 | {r['official_task_score']:.0%} | {r['normal_termination_rate']:.0%} | {r['timeout_count']} | {fmt(r['history_kv_compression'], '{:.2f}×')} | {r['model_calls_per_step']:.3f} | {r['runtime_hours']:.2f} |")
    lines += ["", "All 11 arms scored the same ordered task IDs 0–49. All proxy request errors and reported duplicate raw KV counts are zero.", "",
              "Recovery uses controlled always/first/W2 and is request-local; it does not roll back executed tool/environment state. Operator success is not task-repair success. Calls/request is measured per proxy agent request, not per committed environment step.", "",
              "History compression is measured from available history KV accounting. C2KV recovery uses the post-recovery tensor ratio, rather than its pre-recovery nominal gist ratio; the accounting excludes auxiliary indexes and pool duplication.", "",
              "H2O and SnapKV append/replace remain unsupported because the current shared dense-slot implementation cannot exactly deduplicate their headwise source indices. No scores are fabricated for those four arms.", "",
              "v1 had no fallback output cap; retry_v2 uses 4096 tokens and timeout cancellation. All completed v1 agent responses were below 2.7k tokens in the prior audit, but this remains a mixed-run development comparison. A future formal experiment should freeze the same serving configuration for all arms.", "",
              f"Sum of selected arm harness runtimes: {sum(r['runtime_hours'] for r in rows):.2f} hours. Arms ran in parallel, so this sum is not elapsed wall-clock time. Runtime excludes server startup and shutdown."]
    (a.out / "summary.md").write_text("\n".join(lines)+"\n")
    (a.out / "validation.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    print(a.out)


if __name__ == "__main__":
    main()
