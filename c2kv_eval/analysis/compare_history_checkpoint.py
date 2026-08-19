from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler

from c2kv_eval.analysis.compare_multiturn_modes import (
    DEFAULT_MODEL,
    _analysis_rows,
    _find_first,
    _load_prompts_and_answers,
    _rate,
    _result_count,
    _score_header,
)


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


def _score_ids(mode_root: Path, category: str) -> set[str]:
    result_path = _find_first(mode_root / "result", f"*_{category}_result.json")
    score_path = _find_first(mode_root / "score", f"*_{category}_score.json")
    if result_path is None or score_path is None:
        return set()
    result_ids = {
        str(row["id"])
        for row in _load_jsonl(result_path)
        if row.get("id") is not None
    }
    score_rows = _load_jsonl(score_path)
    invalid_ids = {
        str(row["id"])
        for row in score_rows[1:]
        if row.get("id") is not None and row.get("valid") is False
    }
    explicit_valid = {
        str(row["id"])
        for row in score_rows[1:]
        if row.get("id") is not None and row.get("valid") is True
    }
    return explicit_valid if explicit_valid else result_ids - invalid_ids


def _check_schema(rows: list[dict[str, Any]], name: str) -> set[int]:
    versions = {
        int(row.get("schema_version", 1))
        for row in rows
        if isinstance(row, dict)
    }
    if len(versions) > 1:
        raise ValueError(
            f"Mixed schema versions in {name}: {sorted(versions)}. "
            "Regenerate this mode with one runner version before comparing."
        )
    return versions


def _reference_prompt_tokens(reference_details_path: str) -> dict[tuple[str, int], int]:
    if not reference_details_path:
        return {}
    rows = _load_jsonl(Path(reference_details_path))
    out: dict[tuple[str, int], int] = {}
    for row in rows:
        sample_id = str(row.get("id"))
        global_step = 0
        for turn_counts in row.get("input_token_count") or []:
            if not isinstance(turn_counts, list):
                continue
            for value in turn_counts:
                try:
                    out[(sample_id, global_step)] = int(value)
                except Exception:
                    pass
                global_step += 1
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _auroc(labels: list[bool], scores: list[float]) -> float | None:
    pairs = [(score, label) for score, label in zip(scores, labels)]
    pos = sum(1 for _, label in pairs if label)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(1 for _, label in pairs[index:end] if label)
        index = end
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def _threshold_sweep(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "kv_divergence",
    label_key: str = "candidate_action_drift",
) -> list[dict[str, Any]]:
    pairs = [
        (_as_float(row.get(score_key)), bool(row.get(label_key)))
        for row in rows
        if _as_float(row.get(score_key)) is not None
    ]
    pairs = [(score, label) for score, label in pairs if score is not None]
    if not pairs:
        return []
    positives = sum(1 for _, label in pairs if label)
    thresholds = sorted({score for score, _ in pairs})
    sweep = []
    for threshold in thresholds:
        triggered = [score >= threshold for score, _ in pairs]
        tp = sum(1 for trig, (_, label) in zip(triggered, pairs) if trig and label)
        fp = sum(1 for trig, (_, label) in zip(triggered, pairs) if trig and not label)
        fn = sum(1 for trig, (_, label) in zip(triggered, pairs) if not trig and label)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / positives if positives else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        sweep.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "trigger_rate": sum(triggered) / len(triggered),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "total": len(pairs),
            }
        )
    return sweep


def _threshold_metrics_at(
    rows: list[dict[str, Any]],
    threshold: float | None,
    *,
    score_key: str = "kv_divergence",
    label_key: str = "candidate_action_drift",
) -> dict[str, Any]:
    if threshold is None:
        return {
            "configured_precision": None,
            "configured_recall": None,
            "configured_f1": None,
            "configured_trigger_rate": None,
        }
    pairs = [
        (_as_float(row.get(score_key)), bool(row.get(label_key)))
        for row in rows
        if _as_float(row.get(score_key)) is not None
    ]
    pairs = [(score, label) for score, label in pairs if score is not None]
    if not pairs:
        return {
            "configured_precision": None,
            "configured_recall": None,
            "configured_f1": None,
            "configured_trigger_rate": None,
        }
    triggered = [score >= threshold for score, _ in pairs]
    positives = sum(1 for _, label in pairs if label)
    tp = sum(1 for trig, (_, label) in zip(triggered, pairs) if trig and label)
    fp = sum(1 for trig, (_, label) in zip(triggered, pairs) if trig and not label)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / positives if positives else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "configured_precision": precision,
        "configured_recall": recall,
        "configured_f1": f1,
        "configured_trigger_rate": sum(triggered) / len(triggered),
    }


def _best_sweep_row(sweep: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not sweep:
        return None
    return max(
        sweep,
        key=lambda row: (
            -1.0 if row.get("f1") is None else float(row["f1"]),
            -1.0 if row.get("precision") is None else float(row["precision"]),
        ),
    )


def _kv_calibration(
    *,
    run_root: Path,
    mode: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    usable = [row for row in rows if _as_float(row.get("kv_divergence")) is not None]
    labels = [bool(row.get("candidate_action_drift")) for row in usable]
    kv_scores = [_as_float(row.get("kv_divergence")) for row in usable]
    kl_rows = [row for row in usable if _as_float(row.get("logit_kl")) is not None]
    entropy_rows = [row for row in usable if _as_float(row.get("entropy")) is not None]
    margin_rows = [
        row for row in usable if _as_float(row.get("top1_top2_margin")) is not None
    ]
    sweep = _threshold_sweep(usable)
    best = _best_sweep_row(sweep)
    configured_threshold = None
    for row in rows:
        configured_threshold = _as_float(row.get("verify_threshold"))
        if configured_threshold is not None:
            break
    configured = _threshold_metrics_at(usable, configured_threshold)
    if sweep:
        sweep_path = run_root / mode / "logs" / "kv_threshold_sweep.csv"
        with open(sweep_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
            writer.writeheader()
            writer.writerows(sweep)
    safe_scores = [
        float(row["kv_divergence"]) for row in usable if not row.get("candidate_action_drift")
    ]
    drift_scores = [
        float(row["kv_divergence"]) for row in usable if row.get("candidate_action_drift")
    ]
    return {
        "kv_steps": len(usable),
        "readout_available_rate": _rate(len(usable), len(rows)),
        "kv_divergence_auroc": (
            _auroc(labels, [float(score) for score in kv_scores])
            if kv_scores and all(score is not None for score in kv_scores)
            else None
        ),
        "logit_kl_auroc": _auroc(
            [bool(row.get("candidate_action_drift")) for row in kl_rows],
            [float(row["logit_kl"]) for row in kl_rows],
        )
        if kl_rows
        else None,
        "entropy_auroc": _auroc(
            [bool(row.get("candidate_action_drift")) for row in entropy_rows],
            [float(row["entropy"]) for row in entropy_rows],
        )
        if entropy_rows
        else None,
        "margin_auroc": _auroc(
            [bool(row.get("candidate_action_drift")) for row in margin_rows],
            [-float(row["top1_top2_margin"]) for row in margin_rows],
        )
        if margin_rows
        else None,
        "safe_kv_mean": (
            sum(safe_scores) / len(safe_scores) if safe_scores else None
        ),
        "drift_kv_mean": (
            sum(drift_scores) / len(drift_scores) if drift_scores else None
        ),
        "best_threshold": best.get("threshold") if best else None,
        "best_precision": best.get("precision") if best else None,
        "best_recall": best.get("recall") if best else None,
        "best_f1": best.get("f1") if best else None,
        "best_trigger_rate": best.get("trigger_rate") if best else None,
        "configured_threshold": configured_threshold,
        **configured,
    }


def _summary_row(
    *,
    run_root: Path,
    mode: str,
    category: str,
    turn_rows: list[dict[str, Any]],
    reference_prompt_tokens: dict[tuple[str, int], int],
) -> dict[str, Any]:
    mode_root = run_root / mode
    score = _score_header(mode_root, category)
    total_samples = _result_count(mode_root, category)
    details = _load_jsonl(mode_root / "logs" / "details.jsonl")
    metrics = _load_jsonl(mode_root / "logs" / "checkpoint_metrics.jsonl")
    checkpoint_steps = _load_jsonl(mode_root / "logs" / "checkpoint_steps.jsonl")
    checkpoint_segments = _load_jsonl(
        mode_root / "logs" / "checkpoint_segments.jsonl"
    )
    _check_schema(checkpoint_steps, f"{mode}/checkpoint_steps")
    _check_schema(checkpoint_segments, f"{mode}/checkpoint_segments")
    correct_ids = _score_ids(mode_root, category)

    turn_total = len(turn_rows)
    total_steps = len(checkpoint_steps)
    total_segments = len(checkpoint_segments)
    if checkpoint_segments:
        verify_count = sum(
            1 for row in checkpoint_segments if row.get("verify_triggered")
        )
        refresh_count = sum(
            1 for row in checkpoint_segments if row.get("refresh_triggered")
        )
        rollback_count = sum(
            1 for row in checkpoint_segments if row.get("rollback_triggered")
        )
        total_verify_units = total_segments
    else:
        verify_count = sum(
            1 for row in checkpoint_steps if row.get("verify_triggered")
        )
        refresh_count = sum(
            1 for row in checkpoint_steps if row.get("refresh_triggered")
        )
        rollback_count = sum(
            1 for row in checkpoint_steps if row.get("rollback_triggered")
        )
        total_verify_units = total_steps
    regenerated_steps = sum(int(row.get("regenerated_steps") or 0) for row in checkpoint_steps)
    full_regenerated_tokens = sum(
        int(row.get("full_regenerated_tokens") or 0) for row in checkpoint_steps
    )
    full_verifier_tokens = sum(
        int(row.get("full_probe_prompt_tokens") or 0) for row in checkpoint_steps
    )
    c2kv_verifier_tokens = sum(
        int(row.get("c2kv_probe_prompt_tokens") or 0) for row in checkpoint_steps
    )
    total_candidate_tokens = sum(
        int(row.get("history_prompt_tokens") or 0) for row in checkpoint_steps
    )
    baseline_full_prompt_tokens = 0
    missing_reference_prompt_tokens = 0
    for row in checkpoint_steps:
        key = (str(row.get("id")), int(row.get("global_step") or 0))
        value = reference_prompt_tokens.get(key)
        if value is None:
            value = int(row.get("full_history_tokens") or 0)
            missing_reference_prompt_tokens += 1
        baseline_full_prompt_tokens += int(value or 0)
    refreshed_steps = [row for row in checkpoint_steps if row.get("refresh_triggered")]
    regen_same_count = sum(
        1 for row in refreshed_steps if row.get("regenerated_same_as_candidate")
    )
    candidate_action_drift_ids = {
        str(row.get("id"))
        for row in checkpoint_steps
        if row.get("candidate_action_drift")
    }
    executed_action_drift_ids = {
        str(row.get("id"))
        for row in checkpoint_steps
        if row.get("executed_action_drift")
    }
    state_drift_ids = {
        str(row.get("id")) for row in checkpoint_steps if row.get("state_drift")
    }
    serialization_mismatch_ids = {
        str(row.get("id"))
        for row in checkpoint_steps
        if row.get("serialization_mismatch")
    }
    candidate_readout_reused = sum(
        1 for row in checkpoint_steps if row.get("candidate_readout_reused")
    )
    refreshed_ids = {
        str(row.get("id")) for row in checkpoint_steps if row.get("refresh_triggered")
    }
    original_history_tokens = sum(
        int(row.get("history_original_tokens") or 0) for row in metrics
    )
    effective_history_tokens = sum(
        int(row.get("history_effective_tokens") or 0) for row in metrics
    )
    total_verify_tokens = full_verifier_tokens + c2kv_verifier_tokens
    total_recovery_tokens = full_regenerated_tokens
    e2e_work_tokens = (
        total_candidate_tokens + total_verify_tokens + total_recovery_tokens
    )
    legacy_effective_total = (
        effective_history_tokens + full_regenerated_tokens + full_verifier_tokens
    )
    chat_calls = sum(int(row.get("chat_calls") or 0) for row in metrics)
    chat_seconds = sum(float(row.get("chat_seconds") or 0.0) for row in metrics)
    calibration = _kv_calibration(
        run_root=run_root,
        mode=mode,
        rows=checkpoint_steps,
    )
    (mode_root / "logs" / "kv_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "method": mode,
        "bfcl_accuracy": score.get("accuracy"),
        "correct_count": score.get("correct_count"),
        "total_samples": total_samples,
        "checkpoint_interval": (
            checkpoint_segments[0].get("checkpoint_interval")
            if checkpoint_segments
            else (
                checkpoint_steps[0].get("checkpoint_interval")
                if checkpoint_steps
                else None
            )
        ),
        "turn_joint_pass_rate": _rate(
            sum(1 for row in turn_rows if row.get("joint_pass")),
            turn_total,
        ),
        "candidate_action_drift_rate": _rate(
            len(candidate_action_drift_ids), total_samples
        ),
        "executed_action_drift_rate": _rate(
            len(executed_action_drift_ids), total_samples
        ),
        "state_drift_rate": _rate(len(state_drift_ids), total_samples),
        "serialization_mismatch_rate": _rate(
            len(serialization_mismatch_ids),
            total_samples,
        ),
        "verify_rate": _rate(verify_count, total_verify_units),
        "refresh_rate": _rate(refresh_count, total_verify_units),
        "rollback_rate": _rate(rollback_count, total_verify_units),
        "average_segment_length": (
            sum(float(row.get("segment_length") or 0) for row in checkpoint_segments)
            / total_segments
            if total_segments
            else 1.0
        ),
        "average_rollback_steps": (
            sum(float(row.get("rollback_steps") or 0) for row in checkpoint_segments)
            / total_segments
            if total_segments
            else 0.0
        ),
        "readout_available_rate": calibration.get("readout_available_rate"),
        "candidate_readout_reuse_rate": _rate(
            candidate_readout_reused,
            total_steps,
        ),
        "regenerated_same_as_candidate_rate": _rate(
            regen_same_count, len(refreshed_steps)
        ),
        "recovery_success_rate": _rate(
            len(refreshed_ids & correct_ids),
            len(refreshed_ids),
        ),
        "average_regenerated_steps": (
            regenerated_steps / total_samples if total_samples else None
        ),
        "history_full_tokens": original_history_tokens,
        "history_effective_c2kv_tokens": effective_history_tokens,
        "history_memory_compression_ratio": (
            original_history_tokens / effective_history_tokens
            if effective_history_tokens
            else 1.0
        ),
        "baseline_full_prompt_tokens": baseline_full_prompt_tokens,
        "baseline_full_prompt_tokens_source": (
            "reference_details"
            if reference_prompt_tokens
            else "fallback_full_history_tokens"
        ),
        "missing_reference_prompt_token_steps": missing_reference_prompt_tokens,
        "candidate_prompt_tokens": total_candidate_tokens,
        "candidate_effective_history_tokens": effective_history_tokens,
        "c2kv_verify_prompt_tokens": c2kv_verifier_tokens,
        "full_verify_prompt_tokens": full_verifier_tokens,
        "recovery_prompt_tokens": total_recovery_tokens,
        "recovery_full_history_tokens": full_regenerated_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_verify_tokens": total_verify_tokens,
        "total_recovery_tokens": total_recovery_tokens,
        "full_regenerated_tokens": full_regenerated_tokens,
        "full_verifier_tokens": full_verifier_tokens,
        "c2kv_verifier_tokens": c2kv_verifier_tokens,
        "e2e_token_work_ratio": (
            baseline_full_prompt_tokens / e2e_work_tokens if e2e_work_tokens else 1.0
        ),
        "legacy_effective_compression_ratio": (
            original_history_tokens / legacy_effective_total if legacy_effective_total else 1.0
        ),
        "effective_compression_ratio": (
            original_history_tokens / legacy_effective_total if legacy_effective_total else 1.0
        ),
        "average_chat_latency": chat_seconds / chat_calls if chat_calls else None,
        "kv_divergence_auroc": calibration.get("kv_divergence_auroc"),
        "logit_kl_auroc": calibration.get("logit_kl_auroc"),
        "entropy_auroc": calibration.get("entropy_auroc"),
        "margin_auroc": calibration.get("margin_auroc"),
        "best_threshold": calibration.get("best_threshold"),
        "best_precision": calibration.get("best_precision"),
        "best_recall": calibration.get("best_recall"),
        "best_f1": calibration.get("best_f1"),
        "best_trigger_rate": calibration.get("best_trigger_rate"),
        "configured_threshold": calibration.get("configured_threshold"),
        "configured_precision": calibration.get("configured_precision"),
        "configured_recall": calibration.get("configured_recall"),
        "configured_f1": calibration.get("configured_f1"),
        "configured_trigger_rate": calibration.get("configured_trigger_rate"),
        "verify_count": verify_count,
        "refresh_count": refresh_count,
        "rollback_count": rollback_count,
        "total_segments": total_segments,
        "total_steps": total_steps,
        "regenerated_steps": regenerated_steps,
        "history_original_tokens": original_history_tokens,
        "history_effective_tokens": effective_history_tokens,
        "detail_rows": len(details),
    }


def write_report(run_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# BFCL History Checkpoint Recovery",
        "",
        "| Method | BFCL Acc | Correct | Turn Joint | Candidate Action Drift | Executed Action Drift | State Drift | Serialization Mismatch | Interval | Verify Rate | Rollback Rate | Refresh Rate | Avg Segment | Avg Rollback Steps | Readout | Reuse Readout | KV AUROC | KL AUROC | Ent AUROC | Margin AUROC | Config F1 | Config Thr | Best F1 | Best Thr | Recovery Success | Avg Regen Steps | History KV Compression | Candidate Token Work | Verify Token Work | Recovery Token Work | E2E Token Work Ratio | E2E Source | Missing Ref Token Steps | Legacy Effective Compression | Avg Chat s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {acc} | {correct} | {joint} | {cand} | {execd} | {state} | {serial} | {interval} | {verify} | {rollback} | {refresh} | {avg_segment} | {avg_rollback} | {readout} | {reuse} | {kv_auc} | {kl_auc} | {ent_auc} | {margin_auc} | {config_f1} | {config_threshold} | {best_f1} | {best_threshold} | {recovery} | {regen} | {hist_comp}x | {candidate_tokens} | {verify_tokens} | {recovery_tokens} | {e2e_ratio}x | {e2e_source} | {missing_ref} | {legacy_comp}x | {chat} |".format(
                method=row["method"],
                acc=_fmt(row.get("bfcl_accuracy")),
                correct=_fmt(row.get("correct_count")),
                joint=_fmt(row.get("turn_joint_pass_rate")),
                cand=_fmt(row.get("candidate_action_drift_rate")),
                execd=_fmt(row.get("executed_action_drift_rate")),
                state=_fmt(row.get("state_drift_rate")),
                serial=_fmt(row.get("serialization_mismatch_rate")),
                interval=_fmt(row.get("checkpoint_interval")),
                verify=_fmt(row.get("verify_rate")),
                rollback=_fmt(row.get("rollback_rate")),
                refresh=_fmt(row.get("refresh_rate")),
                avg_segment=_fmt(row.get("average_segment_length")),
                avg_rollback=_fmt(row.get("average_rollback_steps")),
                readout=_fmt(row.get("readout_available_rate")),
                reuse=_fmt(row.get("candidate_readout_reuse_rate")),
                kv_auc=_fmt(row.get("kv_divergence_auroc")),
                kl_auc=_fmt(row.get("logit_kl_auroc")),
                ent_auc=_fmt(row.get("entropy_auroc")),
                margin_auc=_fmt(row.get("margin_auroc")),
                config_f1=_fmt(row.get("configured_f1")),
                config_threshold=_fmt(row.get("configured_threshold")),
                best_f1=_fmt(row.get("best_f1")),
                best_threshold=_fmt(row.get("best_threshold")),
                recovery=_fmt(row.get("recovery_success_rate")),
                regen=_fmt(row.get("average_regenerated_steps")),
                hist_comp=_fmt(row.get("history_memory_compression_ratio")),
                candidate_tokens=_fmt(row.get("total_candidate_tokens")),
                verify_tokens=_fmt(row.get("total_verify_tokens")),
                recovery_tokens=_fmt(row.get("total_recovery_tokens")),
                e2e_ratio=_fmt(row.get("e2e_token_work_ratio")),
                e2e_source=row.get("baseline_full_prompt_tokens_source"),
                missing_ref=_fmt(row.get("missing_reference_prompt_token_steps")),
                legacy_comp=_fmt(row.get("legacy_effective_compression_ratio")),
                chat=_fmt(row.get("average_chat_latency")),
            )
        )
    lines.extend(
        [
            "",
            "Candidate Action Drift is measured before checkpoint recovery.",
            "Executed Action Drift and State Drift are measured after rollback/refresh/regeneration.",
            "Verify Rate and Rollback Rate use segment count as the denominator when `checkpoint_segments.jsonl` is present.",
            "KV AUROC and threshold sweep use Candidate Action Drift as the label.",
            "The full sweep is written to each mode's `logs/kv_threshold_sweep.csv` when KV scores are available.",
            "History KV Compression = Full History Tokens / Effective History KV Tokens.",
            "E2E Token Work Ratio = Full Baseline Prompt Token Work / (Candidate + Verify + Recovery Prompt Token Work).",
            "`Legacy Effective Compression` is the previous mixed accounting and is retained only for continuity.",
        ]
    )
    report_text = "\n".join(lines) + "\n"
    (run_root / "report.md").write_text(report_text, encoding="utf-8")
    (run_root / "checkpoint_summary.md").write_text(report_text, encoding="utf-8")
    if any(str(row.get("method", "")).startswith("multistep_") for row in rows):
        (run_root / "multistep_checkpoint_summary.md").write_text(
            report_text,
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root)
    config = MODEL_CONFIG_MAPPING[args.model]
    decoder = QwenFCHandler(
        model_name=config.model_name,
        temperature=0,
        registry_name=args.model,
        is_fc_model=config.is_fc_model,
    )
    prompt_by_id, answer_by_id = _load_prompts_and_answers(args.category)
    reference_tokens = _reference_prompt_tokens(args.reference_details_path)
    modes = args.modes.split(",") if args.modes else [
        path.name for path in sorted(run_root.iterdir()) if path.is_dir()
    ]
    rows = []
    for mode in modes:
        turn_rows, _ = _analysis_rows(
            run_root,
            mode,
            args.category,
            decoder,
            prompt_by_id,
            answer_by_id,
        )
        rows.append(
            _summary_row(
                run_root=run_root,
                mode=mode,
                category=args.category,
                turn_rows=turn_rows,
                reference_prompt_tokens=reference_tokens,
            )
        )
    summary = {
        "run_root": str(run_root),
        "category": args.category,
        "methods": rows,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with open(run_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_report(run_root, rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--category", default="multi_turn_base")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", default="")
    parser.add_argument("--reference-details-path", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
