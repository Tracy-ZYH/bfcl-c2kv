from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ARMS = (
    "full",
    "c2kv",
    "d_sham_mech",
    "hint_only",
    "d_sham_neutral",
    "d_corr",
    "d_corr_w1",
    "d_corr_w2",
    "d_corr_w4",
    "d_corr_w2_hint",
    "d_corr_w2_oracle_location_hint",
    "d_corr_replace_w2",
    "cacheblend_w2",
    "d_corr_recompute",
    "d_corr_recompute_w2",
    "d_corr_all",
    "raw_all_replace",
    "raw_all_replace_direct",
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


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return int(number)


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
        turn_joint: dict[tuple[Any, Any], bool] = {}
        for step in drift_steps:
            key = (step.get("id"), step.get("turn"))
            if key not in turn_joint:
                turn_joint[key] = True
            if (
                step.get("executed_action_drift") is True
                or step.get("executed_action_matches_reference") is False
                or step.get("state_drift") is True
                or step.get("state_matches_reference") is False
            ):
                turn_joint[key] = False
        turn_joint_pass_rate = (
            sum(int(value) for value in turn_joint.values()) / len(turn_joint)
            if turn_joint
            else None
        )
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
        query_prefill_work = 0
        query_cache_report_missing = 0
        kv_runtime_report_missing = 0
        peak_values: list[int] = []
        for row in metrics:
            if "query_prefill_tokens" in row:
                query_prefill_work += int(row.get("query_prefill_tokens") or 0)
            else:
                query_cache_report_missing += 1
            query_cache_report_missing += int(row.get("chat_cache_report_missing") or 0)
            kv_runtime_report_missing += int(row.get("kv_runtime_report_missing") or 0)
            peak = row.get("peak_physical_kv_tokens")
            if peak is None:
                peak = row.get("kv_peak_resident_tokens")
            peak_int = _positive_int(peak)
            if peak_int is not None:
                peak_values.append(peak_int)
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
        repair_segments = sum(int(row.get("repair_segments") or 0) for row in metrics)
        repair_trigger_count = sum(
            int(row.get("detector_trigger_count") or 0) for row in metrics
        )
        repair_success_count = sum(
            int(row.get("repair_success_count") or 0) for row in metrics
        )
        oracle_harmful_segments = sum(
            int(row.get("oracle_harmful_segments") or 0) for row in metrics
        )
        c2kv_wrong_repair_correct = sum(
            int(row.get("c2kv_wrong_repair_correct") or 0) for row in metrics
        )
        c2kv_wrong_repair_wrong = sum(
            int(row.get("c2kv_wrong_repair_wrong") or 0) for row in metrics
        )
        c2kv_correct_repair_wrong = sum(
            int(row.get("c2kv_correct_repair_wrong") or 0) for row in metrics
        )
        net_repair_gain = sum(int(row.get("net_repair_gain") or 0) for row in metrics)
        repaired_step_count = sum(
            int(row.get("repaired_step_count") or 0) for row in metrics
        )
        repair_changed_action_count = sum(
            int(row.get("repair_changed_action_count") or 0) for row in metrics
        )
        repair_changed_first_token_count = sum(
            int(row.get("repair_changed_first_token_count") or 0)
            for row in metrics
        )
        repair_success_start_correct_num = 0.0
        repair_success_start_correct_den = 0
        repair_success_start_drifted_num = 0.0
        repair_success_start_drifted_den = 0
        for metric in metrics:
            value = metric.get("repair_success_when_start_state_correct")
            den = int(metric.get("detector_trigger_count") or 0)
            if value is not None and den:
                repair_success_start_correct_num += float(value) * den
                repair_success_start_correct_den += den
            value = metric.get("repair_success_when_start_state_already_drifted")
            if value is not None and den:
                repair_success_start_drifted_num += float(value) * den
                repair_success_start_drifted_den += den
        row = {
            "method": arm,
            "bfcl_accuracy": score.get("accuracy") or score.get("overall_accuracy"),
            "correct_count": score.get("correct_count") or score.get("correct"),
            "num_examples": total,
            "errors": int(summary.get("errors") or 0),
            "chat_calls": int(summary.get("chat_calls") or 0),
            "extract_calls": int(summary.get("extract_calls") or 0),
            "extract_success_rate": summary.get("extract_success_rate"),
            "turn_joint_pass_rate": turn_joint_pass_rate,
            "executed_action_drift_rate": _rate(len(executed_drift_ids), total),
            "state_drift_rate": _rate(len(state_drift_ids), total),
            "serialization_mismatch_rate": _rate(len(serialization_mismatch_ids), total),
            "repair_segments": repair_segments,
            "oracle_harmful_segments": oracle_harmful_segments,
            "detector_trigger_count": repair_trigger_count,
            "detector_trigger_rate": (
                repair_trigger_count / repair_segments if repair_segments else None
            ),
            "repair_rate": (
                repair_trigger_count / repair_segments if repair_segments else None
            ),
            "repair_success_count": repair_success_count,
            "repair_success_rate": (
                repair_success_count / repair_trigger_count
                if repair_trigger_count
                else None
            ),
            "repair_segment_success_rate": (
                repair_success_count / oracle_harmful_segments
                if oracle_harmful_segments
                else None
            ),
            "c2kv_wrong_repair_correct": c2kv_wrong_repair_correct,
            "c2kv_wrong_repair_wrong": c2kv_wrong_repair_wrong,
            "c2kv_correct_repair_wrong": c2kv_correct_repair_wrong,
            "net_repair_gain": net_repair_gain,
            "repaired_step_count": repaired_step_count,
            "repair_changed_action_count": repair_changed_action_count,
            "repair_changed_action_rate": (
                repair_changed_action_count / repaired_step_count
                if repaired_step_count
                else None
            ),
            "repair_changed_first_token_count": repair_changed_first_token_count,
            "repair_changed_first_token_rate": (
                repair_changed_first_token_count / repaired_step_count
                if repaired_step_count
                else None
            ),
            "repair_success_when_start_state_correct": (
                repair_success_start_correct_num / repair_success_start_correct_den
                if repair_success_start_correct_den
                else None
            ),
            "repair_success_when_start_state_already_drifted": (
                repair_success_start_drifted_num / repair_success_start_drifted_den
                if repair_success_start_drifted_den
                else None
            ),
            "full_history_kv_tokens": original,
            "physical_history_kv_tokens": effective,
            "c2kv_gist_tokens": c2kv_gist,
            "repair_kv_tokens": repair_kv,
            "recomputed_raw_tokens": recomputed_raw,
            "legacy_history_original_tokens": legacy_original,
            "legacy_history_effective_tokens": legacy_effective,
            "history_kv_compression": original / effective if effective else None,
            "peak_physical_kv_tokens": max(peak_values) if peak_values else None,
            "peak_kv_compression": (
                original / max(peak_values) if peak_values else None
            ),
            "c2kv_extract_recomputed_tokens": c2kv_extract_work,
            "repair_extract_recomputed_tokens": repair_extract_work,
            "query_prefill_tokens": query_prefill_work,
            "query_prefill_tokens_source": "sglang_runtime_cached_tokens",
            "query_cache_report_missing": query_cache_report_missing,
            "kv_runtime_report_missing": kv_runtime_report_missing,
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
                json.dumps(
                    {
                        "note": (
                            "Sample-level closed-loop results differ. This is "
                            "diagnostic only: d_sham_mech validates no-op repair "
                            "plumbing at the per-step token level during runtime."
                        ),
                        "sample_ids": mismatches,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            for row in rows:
                if row.get("method") == "d_sham_mech":
                    row["sample_level_c2kv_mismatch_count"] = len(mismatches)
                    row["sample_level_c2kv_mismatch_note"] = (
                        "diagnostic only; not a fatal no-op failure"
                    )
                    break
    if "d_corr_replace_w2" in detail_by_arm and "append_masked_w2" in detail_by_arm:
        replace = {
            row.get("id"): row.get("result")
            for row in detail_by_arm["d_corr_replace_w2"]
        }
        mismatches = []
        for row in detail_by_arm["append_masked_w2"]:
            sample_id = row.get("id")
            if sample_id in replace and row.get("result") != replace[sample_id]:
                mismatches.append(sample_id)
        if mismatches:
            mismatch_path = run_root / "append_masked_w2_mismatches.json"
            mismatch_path.write_text(
                json.dumps(
                    {
                        "note": (
                            "append_masked_w2 should be deterministic-equivalent "
                            "to d_corr_replace_w2. A mismatch indicates a KV slot "
                            "ordering, position correction, or mask/layout bug."
                        ),
                        "sample_ids": mismatches,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "append_masked_w2 output mismatch vs d_corr_replace_w2 for "
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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = run_root / "kv_repair_report.md"
    lines = [
        "# BFCL History C2KV KV-Repair Sweep",
        "",
        "| Method | BFCL Acc | Correct | Turn Joint | Executed Drift | State Drift | Repair Attempts | Repair Success | Wrong -> Correct | Wrong -> Wrong | Correct -> Wrong | Net Gain | Changed Action | Repair KV | Recomputed Raw | KV Compression | E2E Work | Avg E2E s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc} | {correct} | {turn_joint} | {edrift} | {sdrift} | {attempts} | {repair_success} | {w2c} | {w2w} | {c2w} | {net} | {changed} | {repair_kv} | {recompute} | {comp} | {work} | {e2e_lat} |".format(
                method=row["method"],
                acc=(
                    f"{float(row['bfcl_accuracy']):.4f}"
                    if row["bfcl_accuracy"] is not None
                    else "-"
                ),
                correct=row["correct_count"] if row["correct_count"] is not None else "-",
                turn_joint=(
                    f"{row['turn_joint_pass_rate']:.4f}"
                    if row["turn_joint_pass_rate"] is not None
                    else "-"
                ),
                repair_success=(
                    f"{row['repair_segment_success_rate']:.4f}"
                    if row["repair_segment_success_rate"] is not None
                    else "-"
                ),
                attempts=row["detector_trigger_count"],
                w2c=row["c2kv_wrong_repair_correct"],
                w2w=row["c2kv_wrong_repair_wrong"],
                c2w=row["c2kv_correct_repair_wrong"],
                net=row["net_repair_gain"],
                changed=(
                    f"{row['repair_changed_action_rate']:.4f}"
                    if row["repair_changed_action_rate"] is not None
                    else "-"
                ),
                repair_kv=row["repair_kv_tokens"],
                recompute=row["recomputed_raw_tokens"],
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
                e2e_lat=(
                    f"{row['avg_episode_e2e_observed_seconds']:.4f}"
                    if row["avg_episode_e2e_observed_seconds"] is not None
                    else "-"
                ),
            )
        )
    by_method = {row["method"]: row for row in rows}

    def acc(method: str) -> float | None:
        value = (by_method.get(method) or {}).get("bfcl_accuracy")
        return float(value) if value is not None else None

    def gain(method: str) -> int | None:
        row = by_method.get(method)
        return int(row["net_repair_gain"]) if row is not None else None

    lines.extend(["", "## Automatic Readout", ""])
    comparisons = [
        ("W2_gain_over_W1", gain("d_corr_w2"), gain("d_corr_w1")),
        ("W4_gain_over_W2", gain("d_corr_w4"), gain("d_corr_w2")),
        ("Replace_gain_over_Append", gain("d_corr_replace_w2"), gain("d_corr_w2")),
        ("Hint_gain_over_Corr", acc("d_corr_w2_hint"), acc("d_corr_w2")),
        (
            "OracleHint_gain_over_NormalHint",
            acc("d_corr_w2_oracle_location_hint"),
            acc("d_corr_w2_hint"),
        ),
        ("RawAllReplace_gap_to_Full", acc("full"), acc("raw_all_replace")),
        (
            "RawAllReplaceDirect_gap_to_Full",
            acc("full"),
            acc("raw_all_replace_direct"),
        ),
        ("CorrAllAppend_gap_to_RawAllReplace", acc("raw_all_replace"), acc("d_corr_all")),
    ]
    for name, lhs, rhs in comparisons:
        if lhs is None or rhs is None:
            lines.append(f"- {name}: unavailable")
        elif isinstance(lhs, int) and isinstance(rhs, int):
            lines.append(f"- {name}: {lhs - rhs:+d}")
        else:
            lines.append(f"- {name}: {lhs - rhs:+.4f}")
    full_acc = acc("full")
    raw_replace_acc = acc("raw_all_replace")
    if (
        full_acc is not None
        and raw_replace_acc is not None
        and abs(full_acc - raw_replace_acc) > 0.05
    ):
        lines.append("- RAW_KV_REPLACE_NOT_FULL_EQUIVALENT")
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
