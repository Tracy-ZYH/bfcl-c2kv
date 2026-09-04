from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "full": "Full",
    "c2kv": "C2KV",
    "streamingllm": "StreamingLLM",
    "h2o": "H2O",
    "snapkv_persistent": "SnapKV-Persistent",
    "snapkv": "SnapKV-Persistent",
    "snapkv_refresh": "SnapKV-Refresh",
    "pyramidkv": "PyramidKV",
}


CSV_FIELDS = [
    "Method",
    "BFCL Accuracy",
    "Correct",
    "Total",
    "Turn Joint",
    "Candidate Action Drift",
    "Executed Action Drift",
    "State Drift",
    "Avg Full History KV",
    "Avg Active History KV",
    "Estimated Weighted History-KV Retention",
    "Estimated Weighted History-KV Compression",
    "Measured Weighted History-KV Retention",
    "Measured Weighted History-KV Compression",
    "Memory Report Coverage",
    "Model Calls / Committed Step",
    "Prefill Tokens / Committed Step",
    "Runtime Status",
]


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


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _rate(num: float | int | None, den: float | int | None) -> float | None:
    if num is None:
        return None
    return num / den if den else None


def _score(root: Path, total: int) -> tuple[float | None, int | None]:
    path = root / "score" / "data_multi_turn.csv"
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            value = rows[0].get("Base") or rows[0].get("Multi Turn Overall Acc")
            if isinstance(value, str) and value.endswith("%"):
                acc = float(value[:-1]) / 100.0
                return acc, round(acc * total)
    return None, None


def _flatten_steps(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in details:
        sample_id = row.get("id")
        for step in row.get("drift_steps") or []:
            if isinstance(step, dict):
                item = dict(step)
                item.setdefault("id", sample_id)
                out.append(item)
    return out


def _step_rate(steps: list[dict[str, Any]], drift_key: str, match_key: str) -> float | None:
    if not steps:
        return None
    bad = sum(
        1
        for step in steps
        if step.get(drift_key) is True or step.get(match_key) is False
    )
    return bad / len(steps)


def _turn_joint(details: list[dict[str, Any]]) -> float | None:
    total = 0
    passed = 0
    for row in details:
        turns: dict[int, list[dict[str, Any]]] = {}
        for step in row.get("drift_steps") or []:
            if isinstance(step, dict):
                turns.setdefault(int(step.get("turn") or 0), []).append(step)
        for steps in turns.values():
            total += 1
            if all(
                step.get("executed_action_drift") is not True
                and step.get("executed_action_matches_reference") is not False
                and step.get("state_drift") is not True
                and step.get("state_matches_reference") is not False
                for step in steps
            ):
                passed += 1
    return _rate(passed, total)


def _compression(steps: list[dict[str, Any]]) -> dict[str, Any]:
    estimated_pairs: list[tuple[float, float]] = []
    measured_pairs: list[tuple[float, float]] = []
    measured_reported_steps = 0
    history_bearing_reported_steps = 0
    for step in steps:
        decision = step.get("history_kv_decision")
        if isinstance(decision, dict):
            full = _num(decision.get("history_raw_tokens"))
            active = _num(decision.get("history_active_kv_tokens"))
            if full is not None and active is not None and active > 0:
                estimated_pairs.append((full, active))
        report = step.get("kv_memory_report")
        if isinstance(report, dict):
            full = _num(report.get("full_equivalent_history_tokens"))
            active = _num(report.get("active_history_kv_tokens"))
            # C2KV's legacy report is explicitly a client-side estimate. It
            # must never be promoted into the measured/runtime column.
            measured = report.get("estimated") is False
            # A zero-history decision is a valid runtime report, but it does
            # not contribute to a history-KV compression ratio (0 / 0).
            if full is not None and full > 0:
                history_bearing_reported_steps += 1
            # A no-history step needs no history layout measurement. Count it
            # toward coverage so physical runs are not penalized before their
            # first completed history unit exists.
            if measured or not full:
                measured_reported_steps += 1
            if (
                measured
                and full is not None
                and active is not None
                and full > 0
                and active > 0
            ):
                measured_pairs.append((full, active))
    def weighted(pairs: list[tuple[float, float]]) -> float | None:
        full = sum(a for a, _ in pairs)
        active = sum(b for _, b in pairs)
        return _rate(full, active)
    def retention(pairs: list[tuple[float, float]]) -> float | None:
        full = sum(a for a, _ in pairs)
        active = sum(b for _, b in pairs)
        return _rate(active, full)
    return {
        "avg_full": (
            sum(full for full, _ in estimated_pairs) / len(estimated_pairs)
            if estimated_pairs
            else None
        ),
        "avg_active": (
            sum(active for _, active in estimated_pairs) / len(estimated_pairs)
            if estimated_pairs
            else None
        ),
        "estimated_retention": retention(estimated_pairs),
        "estimated_weighted": weighted(estimated_pairs),
        "measured_retention": retention(measured_pairs),
        "measured_weighted": weighted(measured_pairs),
        # Coverage measures actual runtime layout, not transport of a
        # client-side estimate. Zero-history steps are valid by construction.
        "coverage": _rate(measured_reported_steps, len(steps)),
        "history_bearing_coverage": _rate(
            len(measured_pairs), history_bearing_reported_steps
        ),
    }


def _identity_signature(row: dict[str, Any]) -> dict[str, Any]:
    """Return the deterministic closed-loop surface for identity sanity runs."""
    return {
        "result": row.get("result"),
        "steps": [
            {
                "candidate_raw_text": step.get("candidate_raw_text"),
                "candidate_action": step.get("candidate_action"),
                "executed_action": step.get("executed_action"),
                "state": step.get("state"),
            }
            for step in row.get("drift_steps") or []
            if isinstance(step, dict)
        ],
    }


def assert_identity_against_full(run_root: Path, methods: list[str]) -> None:
    """Fail fast when retention=1 diverges from ordinary Full serving."""
    full_rows = {
        str(row.get("id")): _identity_signature(row)
        for row in _load_jsonl(run_root / "full" / "logs" / "details.jsonl")
        if row.get("id") is not None
    }
    if not full_rows:
        raise RuntimeError("identity assertion requires full/logs/details.jsonl")

    failures: list[str] = []
    for method in methods:
        if method == "full":
            continue
        rows = _load_jsonl(run_root / method / "logs" / "details.jsonl")
        seen = {str(row.get("id")) for row in rows if row.get("id") is not None}
        missing = sorted(set(full_rows) - seen)
        extra = sorted(seen - set(full_rows))
        if missing or extra:
            failures.append(
                f"{method}: episode set mismatch missing={missing} extra={extra}"
            )
        for row in rows:
            sample_id = str(row.get("id"))
            expected = full_rows.get(sample_id)
            if expected is None:
                continue
            if _identity_signature(row) != expected:
                failures.append(f"{method}: retention=1 differs from Full for {sample_id}")
    if failures:
        raise RuntimeError("History-KV identity failure:\n" + "\n".join(failures))


def _summary_estimated_ratios(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    retention = _num(summary.get("estimated_weighted_history_kv_retention_ratio"))
    compression = _num(summary.get("estimated_weighted_history_kv_compression"))
    if retention is None and compression is not None and 0 < compression < 1:
        # Compatibility with early runs that wrote active/full under a
        # compression-ratio field name.
        retention = compression
        compression = 1.0 / retention
    if compression is None and retention is not None and retention > 0:
        compression = 1.0 / retention
    return retention, compression


def summarize_method(run_root: Path, method: str) -> dict[str, Any]:
    root = run_root / method
    details = _load_jsonl(root / "logs" / "details.jsonl")
    summary = _load_json(root / "logs" / "summary.json")
    total = len(details) or int(summary.get("num_examples") or 0)
    steps = _flatten_steps(details)
    acc, correct = _score(root, total)
    metrics = [
        row.get("c2kv_drift_metrics")
        for row in details
        if isinstance(row.get("c2kv_drift_metrics"), dict)
    ]
    chat_calls = sum(int(row.get("chat_calls") or 0) for row in metrics)
    prefill = sum(int(row.get("chat_recomputed_prompt_tokens") or 0) for row in metrics)
    comp = _compression(steps)
    summary_full = _num(summary.get("canonical_full_history_tokens"))
    summary_active = _num(summary.get("physical_history_kv_tokens"))
    summary_steps = _num(summary.get("chat_calls")) or len(steps)
    summary_avg_full = _rate(summary_full, summary_steps)
    summary_avg_active = _rate(summary_active, summary_steps)
    summary_weighted = _rate(summary_full, summary_active)
    summary_retention = _rate(summary_active, summary_full)
    named_summary_retention, named_summary_compression = _summary_estimated_ratios(summary)
    runtime_statuses = {
        str(step.get("kv_memory_report", {}).get("history_kv_runtime_status"))
        for step in steps
        if isinstance(step.get("kv_memory_report"), dict)
        and step["kv_memory_report"].get("history_kv_runtime_status")
    }
    status = ",".join(sorted(runtime_statuses)) if runtime_statuses else "ok"
    if comp["coverage"] is not None and comp["coverage"] < 1.0:
        status = f"{status},memory_report_incomplete"
    if not root.exists():
        status = "missing"
    elif int(summary.get("errors") or 0):
        status = "errors"
    return {
        "Method": METHOD_LABELS.get(method, method),
        "BFCL Accuracy": acc,
        "Correct": correct,
        "Total": total,
        "Turn Joint": _turn_joint(details),
        "Candidate Action Drift": _step_rate(
            steps, "candidate_action_drift", "candidate_action_matches_reference"
        ),
        "Executed Action Drift": _step_rate(
            steps, "executed_action_drift", "executed_action_matches_reference"
        ),
        "State Drift": _step_rate(steps, "state_drift", "state_matches_reference"),
        "Avg Full History KV": summary_avg_full or comp["avg_full"],
        "Avg Active History KV": summary_avg_active or comp["avg_active"],
        "Estimated Weighted History-KV Retention": summary_retention
        or named_summary_retention
        or comp["estimated_retention"],
        "Estimated Weighted History-KV Compression": summary_weighted
        or named_summary_compression
        or comp["estimated_weighted"]
        or summary.get("estimated_weighted_history_kv_compression"),
        "Measured Weighted History-KV Retention": comp["measured_retention"],
        "Measured Weighted History-KV Compression": comp["measured_weighted"],
        "Memory Report Coverage": comp["coverage"],
        "Model Calls / Committed Step": _rate(chat_calls, len(steps)),
        "Prefill Tokens / Committed Step": _rate(prefill, len(steps)),
        "Runtime Status": status,
    }


def write_outputs(run_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = run_root / "history_kv_baseline_summary.csv"
    md_path = run_root / "history_kv_baseline_summary.md"
    json_path = run_root / "history_kv_baseline_summary.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# BFCL History KV Baselines", ""]
    lines.append("| " + " | ".join(CSV_FIELDS) + " |")
    lines.append("| " + " | ".join(["---"] * len(CSV_FIELDS)) + " |")
    for row in rows:
        vals = []
        for col in CSV_FIELDS:
            value = row.get(col)
            if isinstance(value, float):
                vals.append(f"{value:.4f}")
            elif value is None:
                vals.append("-")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--methods", default="full,c2kv,streamingllm,h2o,snapkv_persistent,pyramidkv")
    parser.add_argument(
        "--assert-identity-against-full",
        action="store_true",
        help="Require each non-Full method to match Full episode-by-episode.",
    )
    args = parser.parse_args()
    run_root = Path(args.run_root)
    methods = [method for method in args.methods.split(",") if method]
    if args.assert_identity_against_full:
        assert_identity_against_full(run_root, methods)
    rows = [summarize_method(run_root, method) for method in methods]
    write_outputs(run_root, rows)
    print(json.dumps({"run_root": str(run_root), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
