from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ARMS = (
    "full",
    "c2kv",
    "d_sham_mech",
    "d_sham_neutral",
    "d_corr",
    "d_corr_recompute",
    "d_corr_all",
)

PLAN_COST_ARMS = {
    "d_sham_neutral",
    "d_corr",
    "d_corr_recompute",
    "d_corr_all",
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(count: int, total: int) -> float | None:
    return count / total if total else None


def _find_score(root: Path) -> dict[str, Any]:
    matches = sorted((root / "score").rglob("*_score.json"))
    if not matches:
        return {}
    rows = _load_jsonl(matches[0])
    return rows[0] if rows else {}


def summarize(run_root: Path, arms: list[str]) -> list[dict[str, Any]]:
    detail_by_arm: dict[str, list[dict[str, Any]]] = {}
    plan_summary = _load_json(run_root / "logs" / "plan_build_summary.json")
    rows = []
    for arm in arms:
        root = run_root / arm
        summary = _load_json(root / "logs" / "summary.json")
        score = _find_score(root)
        metrics = _load_jsonl(root / "logs" / "metrics.jsonl")
        details = _load_jsonl(root / "logs" / "details.jsonl")
        detail_by_arm[arm] = details
        total = len(details) or int(summary.get("num_examples") or 0)
        drift_steps = [
            step
            for row in details
            for step in (row.get("drift_steps") or [])
            if isinstance(step, dict)
        ]
        executed_drift_ids = {
            row.get("id")
            for row in details
            if any(
                step.get("executed_action_drift") is True
                or step.get("executed_action_matches_reference") is False
                for step in (row.get("drift_steps") or [])
                if isinstance(step, dict)
            )
        }
        state_drift_ids = {
            row.get("id")
            for row in details
            if any(
                step.get("state_drift") is True
                or step.get("state_matches_reference") is False
                for step in (row.get("drift_steps") or [])
                if isinstance(step, dict)
            )
        }
        serialization_mismatch_ids = {
            row.get("id")
            for row in details
            if any(
                step.get("serialization_mismatch") is True
                for step in (row.get("drift_steps") or [])
                if isinstance(step, dict)
            )
        }
        legacy_original = sum(
            int(row.get("history_original_tokens") or 0) for row in metrics
        )
        legacy_effective = sum(
            int(row.get("history_effective_tokens") or 0) for row in metrics
        )
        original = sum(
            int(
                row.get("canonical_full_history_tokens")
                or row.get("history_original_tokens")
                or 0
            )
            for row in metrics
        )
        effective = sum(
            int(
                row.get("physical_history_kv_tokens")
                or row.get("history_effective_tokens")
                or 0
            )
            for row in metrics
        )
        c2kv_gist = sum(int(row.get("c2kv_gist_tokens") or 0) for row in metrics)
        repair_kv = sum(int(row.get("repair_kv_tokens") or 0) for row in metrics)
        recomputed_raw = sum(
            int(row.get("recomputed_raw_tokens") or 0) for row in metrics
        )
        c2kv_extract_work = sum(
            int(row.get("c2kv_extract_recomputed_tokens") or 0) for row in metrics
        )
        repair_extract_work = sum(
            int(row.get("repair_extract_recomputed_tokens") or 0) for row in metrics
        )
        query_prefill_work = sum(
            int(row.get("query_prefill_tokens") or row.get("chat_prompt_tokens") or 0)
            for row in metrics
        )
        decode_work = sum(
            int(row.get("decode_tokens") or row.get("chat_completion_tokens") or 0)
            for row in metrics
        )
        total_actual_work = sum(
            int(row.get("total_actual_recomputed_tokens") or 0)
            for row in metrics
        )
        if not total_actual_work:
            total_actual_work = (
                c2kv_extract_work
                + repair_extract_work
                + query_prefill_work
                + decode_work
            )
        chat_seconds = sum(float(row.get("chat_seconds") or 0.0) for row in metrics)
        extract_seconds = sum(float(row.get("extract_seconds") or 0.0) for row in metrics)
        c2kv_extract_seconds = sum(
            float(row.get("c2kv_extract_seconds") or 0.0) for row in metrics
        )
        repair_extract_seconds = sum(
            float(row.get("repair_extract_seconds") or 0.0) for row in metrics
        )
        tool_execution_seconds = sum(
            float(row.get("tool_execution_seconds") or 0.0) for row in metrics
        )
        observed_e2e_seconds = sum(
            float(row.get("episode_e2e_observed_seconds") or 0.0)
            for row in metrics
        )
        if not observed_e2e_seconds:
            observed_e2e_seconds = chat_seconds + extract_seconds + tool_execution_seconds
        plan_build_seconds = (
            float(plan_summary.get("plan_build_seconds") or 0.0)
            if arm in PLAN_COST_ARMS
            else 0.0
        )
        plan_build_tokenization_tokens = (
            int(plan_summary.get("plan_build_tokenization_tokens") or 0)
            if arm in PLAN_COST_ARMS
            else 0
        )
        observed_e2e_with_plan_seconds = observed_e2e_seconds + plan_build_seconds
        chat_latencies = [
            float(row.get("avg_chat_seconds"))
            for row in metrics
            if row.get("avg_chat_seconds") is not None
        ]
        row = {
            "method": arm,
            "bfcl_accuracy": score.get("accuracy") or score.get("overall_accuracy"),
            "correct_count": score.get("correct_count") or score.get("correct"),
            "num_examples": total,
            "errors": int(summary.get("errors") or 0),
            "chat_calls": int(summary.get("chat_calls") or 0),
            "extract_calls": int(summary.get("extract_calls") or 0),
            "extract_success_rate": summary.get("extract_success_rate"),
            "turn_joint_pass_rate": None,
            "executed_action_drift_rate": _rate(len(executed_drift_ids), total),
            "state_drift_rate": _rate(len(state_drift_ids), total),
            "serialization_mismatch_rate": _rate(len(serialization_mismatch_ids), total),
            "full_history_kv_tokens": original,
            "physical_history_kv_tokens": effective,
            "c2kv_gist_tokens": c2kv_gist,
            "repair_kv_tokens": repair_kv,
            "recomputed_raw_tokens": recomputed_raw,
            "legacy_history_original_tokens": legacy_original,
            "legacy_history_effective_tokens": legacy_effective,
            "history_kv_compression": original / effective if effective else None,
            "peak_physical_kv_tokens": effective,
            "peak_kv_compression": original / effective if effective else None,
            "c2kv_extract_recomputed_tokens": c2kv_extract_work,
            "repair_extract_recomputed_tokens": repair_extract_work,
            "query_prefill_tokens": query_prefill_work,
            "decode_tokens": decode_work,
            "total_actual_recomputed_tokens": total_actual_work,
            "e2e_token_work_ratio": None,
            "chat_seconds": chat_seconds,
            "extract_seconds": extract_seconds,
            "c2kv_extract_seconds": c2kv_extract_seconds,
            "repair_extract_seconds": repair_extract_seconds,
            "tool_execution_seconds": tool_execution_seconds,
            "episode_e2e_observed_seconds": observed_e2e_seconds,
            "avg_episode_e2e_observed_seconds": (
                observed_e2e_seconds / total if total else None
            ),
            "plan_path": plan_summary.get("plan_path") if arm in PLAN_COST_ARMS else None,
            "plan_build_mode": plan_summary.get("mode") if arm in PLAN_COST_ARMS else None,
            "plan_build_seconds": plan_build_seconds,
            "plan_build_tokenization_tokens": plan_build_tokenization_tokens,
            "plan_n_qids": plan_summary.get("n_qids") if arm in PLAN_COST_ARMS else None,
            "plan_budget_gate_passed": (
                plan_summary.get("budget_gate_passed") if arm in PLAN_COST_ARMS else None
            ),
            "plan_neutrality_gate_passed": (
                plan_summary.get("neutrality_gate_passed")
                if arm in PLAN_COST_ARMS
                else None
            ),
            "episode_e2e_observed_with_plan_seconds": observed_e2e_with_plan_seconds,
            "avg_episode_e2e_observed_with_plan_seconds": (
                observed_e2e_with_plan_seconds / total if total else None
            ),
            "avg_chat_latency_sec": _mean(chat_latencies),
            "avg_chat_latency_sec_note": "chat/completions only; excludes extract and tool execution",
            "ttft_sec": None,
            "ttft_note": "unavailable: server response does not expose first-token timestamp",
            "detail_step_rows": len(drift_steps),
        }
        rows.append(row)
    if "c2kv" in detail_by_arm and "d_sham_mech" in detail_by_arm:
        c2kv = {row.get("id"): row.get("result") for row in detail_by_arm["c2kv"]}
        mismatches = []
        for row in detail_by_arm["d_sham_mech"]:
            sample_id = row.get("id")
            if sample_id in c2kv and row.get("result") != c2kv[sample_id]:
                mismatches.append(sample_id)
        if mismatches:
            mismatch_path = run_root / "d_sham_mech_mismatches.json"
            mismatch_path.write_text(
                json.dumps(mismatches, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "d_sham_mech output mismatch vs c2kv for "
                f"{len(mismatches)} samples; see {mismatch_path}"
            )
    full_work = next(
        (
            int(row.get("total_actual_recomputed_tokens") or 0)
            for row in rows
            if row.get("method") == "full"
        ),
        0,
    )
    if full_work:
        for row in rows:
            method_work = int(row.get("total_actual_recomputed_tokens") or 0)
            row["e2e_token_work_ratio"] = (
                full_work / method_work if method_work else None
            )
    return rows


def write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    csv_path = run_root / "kv_repair_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = run_root / "kv_repair_report.md"
    lines = [
        "# BFCL History C2KV KV-Repair Sweep",
        "",
        "| Method | BFCL Acc | Correct | Examples | Errors | History KV Compression | E2E Work Ratio | Executed Drift | State Drift | Avg Chat-only s | Avg Observed E2E s | Plan Build s | Avg E2E+Plan s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc} | {correct} | {num_examples} | {errors} | {comp} | {work} | {edrift} | {sdrift} | {lat} | {e2e_lat} | {plan_s} | {e2e_plan_lat} |".format(
                method=row["method"],
                acc=(
                    f"{float(row['bfcl_accuracy']):.4f}"
                    if row["bfcl_accuracy"] is not None
                    else "-"
                ),
                correct=row["correct_count"] if row["correct_count"] is not None else "-",
                num_examples=row["num_examples"],
                errors=row["errors"],
                comp=(
                    f"{row['history_kv_compression']:.4f}x"
                    if row["history_kv_compression"] is not None
                    else "-"
                ),
                work=(
                    f"{row['e2e_token_work_ratio']:.4f}x"
                    if row["e2e_token_work_ratio"] is not None
                    else "-"
                ),
                edrift=(
                    f"{row['executed_action_drift_rate']:.4f}"
                    if row["executed_action_drift_rate"] is not None
                    else "-"
                ),
                sdrift=(
                    f"{row['state_drift_rate']:.4f}"
                    if row["state_drift_rate"] is not None
                    else "-"
                ),
                lat=(
                    f"{row['avg_chat_latency_sec']:.4f}"
                    if row["avg_chat_latency_sec"] is not None
                    else "-"
                ),
                e2e_lat=(
                    f"{row['avg_episode_e2e_observed_seconds']:.4f}"
                    if row["avg_episode_e2e_observed_seconds"] is not None
                    else "-"
                ),
                plan_s=f"{row['plan_build_seconds']:.4f}",
                e2e_plan_lat=(
                    f"{row['avg_episode_e2e_observed_with_plan_seconds']:.4f}"
                    if row["avg_episode_e2e_observed_with_plan_seconds"] is not None
                    else "-"
                ),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_root / "kv_repair_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    run_root = Path(args.run_root)
    rows = summarize(run_root, arms)
    write_outputs(run_root, rows)
    print(run_root / "kv_repair_summary.csv")


if __name__ == "__main__":
    main()
