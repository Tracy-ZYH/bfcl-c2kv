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
    "Avg Active History KV",
    "Estimated Weighted History-KV Compression",
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
            if full is not None and active is not None and active > 0:
                measured_pairs.append((full, active))
    def weighted(pairs: list[tuple[float, float]]) -> float | None:
        full = sum(a for a, _ in pairs)
        active = sum(b for _, b in pairs)
        return _rate(full, active)
    return {
        "avg_active": (
            sum(active for _, active in estimated_pairs) / len(estimated_pairs)
            if estimated_pairs
            else None
        ),
        "estimated_weighted": weighted(estimated_pairs),
        "measured_weighted": weighted(measured_pairs),
        "coverage": _rate(len(measured_pairs), len(steps)),
    }


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
    status = "ok"
    if not root.exists():
        status = "missing"
    elif int(summary.get("errors") or 0):
        status = "errors"
    elif method in {"h2o", "snapkv", "snapkv_persistent", "pyramidkv"} and not summary.get(
        "allow_client_fallback"
    ):
        status = "runtime_eviction_unimplemented"
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
        "Avg Active History KV": comp["avg_active"],
        "Estimated Weighted History-KV Compression": comp["estimated_weighted"]
        or summary.get("estimated_weighted_history_kv_compression"),
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
    args = parser.parse_args()
    run_root = Path(args.run_root)
    rows = [summarize_method(run_root, method) for method in args.methods.split(",") if method]
    write_outputs(run_root, rows)
    print(json.dumps({"run_root": str(run_root), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
