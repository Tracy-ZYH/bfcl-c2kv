"""Drive the official BFCL generator and scorer through the portable proxy.

The adapter registers an OpenAI handler for the requested served model and
isolates result, score, and subset files under the run's output directory.

Usage:
    python -m c2kv_eval.portable.adapters.bfcl_adapter \
        --base-url http://127.0.0.1:34000/v1 --categories multi_turn_base

Subset generation uses ``--run-ids`` and scoring uses ``--partial-eval``.
The terminal-state check rejects missing or unscored requested cases.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .base import RunContext, v1
from ..bfcl_gold_recovery import (
    GoldRecoveryController,
    MULTI_EVENT_ORACLE_KIND,
    MULTI_EVENT_ORACLE_VERSION,
    MultiEventGoldRecoveryController,
    ORACLE_KIND,
    ORACLE_VERSION,
    TurnRetry,
    summarize_audit,
)

NAME = "bfcl"
MODEL_NAME = "c2kv-hf"  # BFCL handler key / result-dir name (stable layout)
SERVED_MODEL = "c2kv-agent"  # default served model name at the endpoint


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_arguments(parser) -> None:
    """BFCL-only CLI flags (shared ones live in run.py's core block)."""
    parser.add_argument(
        "--bfcl-dir", type=Path, default=None,
        help="BFCL checkout (default: $BENCH_BFCL_DIR or the checkout containing this package)",
    )
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--run-ids", default="",
                        help="bfcl: comma-separated official case ids for a subset run")
    parser.add_argument(
        "--bfcl-oracle-max-events", type=_positive_int, default=1,
        help="bfcl gold arms: retry budget per task; 1 is the frozen v2 protocol",
    )


def default_bfcl_dir() -> str:
    """Checkout containing the installed BFCL package data."""
    configured = os.environ.get("BENCH_BFCL_DIR")
    if configured:
        return configured
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "bfcl_eval").is_dir():
        return str(package_root)
    return str(Path.home() / "benchmarks" / "gorilla"
               / "berkeley-function-call-leaderboard")


def handler_key(arm: str) -> str:
    """BFCL model key = result-dir name, one per arm so runs never overwrite
    each other.  eval_runner.py:782 un-escapes the result dir with
    ``replace("_", "/")`` — underscores in arm names would corrupt the path,
    so the key uses dashes."""
    return f"c2kv-{(arm or 'full').replace('_', '-')}"


def generate_argv(handler_name: str, categories: str,
                  run_ids: Optional[List[str]] = None,
                  num_threads: int = 1) -> List[str]:
    """``bfcl generate`` argv (PINNED; driven in-process by run_cli)."""
    if num_threads < 1:
        raise ValueError(f"BFCL num_threads must be positive, got {num_threads}")
    argv = [
        "generate", "--model", handler_name, "--test-category", categories,
        "--num-threads", str(num_threads),
    ]
    if run_ids:
        argv.append("--run-ids")
    return argv


def evaluate_argv(handler_name: str, categories: str,
                  run_ids: Optional[List[str]] = None) -> List[str]:
    """``bfcl evaluate`` argv.  A subset EVALUATE needs --partial-eval, NOT
    --run-ids (this vintage scores the full category otherwise)."""
    argv = ["evaluate", "--model", handler_name, "--test-category", categories]
    if run_ids:
        argv.append("--partial-eval")
    return argv


def install_handler(base_url: str, model: str = SERVED_MODEL,
                    handler_name: "str | None" = None,
                    gold_recovery: "str | None" = None,
                    task_audit_path: "Path | str | None" = None,
                    bfcl_oracle_max_events: int = 1) -> None:
    # NOTE: default resolved at CALL time — binding the default to
    # MODEL_NAME at def time made monkeypatched names register the
    # wrong key (val20 evaluate failure)
    handler_name = handler_name or MODEL_NAME
    if (not isinstance(bfcl_oracle_max_events, int)
            or isinstance(bfcl_oracle_max_events, bool)
            or bfcl_oracle_max_events <= 0):
        raise ValueError("bfcl_oracle_max_events must be a positive integer")
    if bfcl_oracle_max_events > 1 and gold_recovery is None:
        raise ValueError("BFCL multi-event recovery requires a gold-recovery arm")
    import httpx
    from openai import OpenAI
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )

    class C2KVHandler(OpenAICompletionsHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bfcl_oracle_max_events == 1:
                self._gold_controller = GoldRecoveryController(
                    gold_recovery, task_audit_path)
            else:
                self._gold_controller = MultiEventGoldRecoveryController(
                    gold_recovery, bfcl_oracle_max_events, task_audit_path)

        def _build_client_kwargs(self):
            return {
                "api_key": "EMPTY",
                "base_url": base_url,
                "timeout": httpx.Timeout(timeout=600.0, connect=8.0),
            }

        def inference(self, test_entry: dict, include_input_log: bool,
                      exclude_state_log: bool):
            controller = self._gold_controller
            guard = (controller.inference_lock
                     if controller.selector is not None else nullcontext())
            with guard:
                state = controller.begin(test_entry)
                original_test_entry = deepcopy(test_entry)
                original_namespace = self.model_name_underline_replaced
                try:
                    if isinstance(controller, MultiEventGoldRecoveryController):
                        active_retry_namespace = None
                        while True:
                            self.model_name_underline_replaced = (
                                active_retry_namespace or original_namespace)
                            next_retry_namespace = None
                            try:
                                result = super().inference(
                                    deepcopy(original_test_entry),
                                    include_input_log,
                                    exclude_state_log,
                                )
                                controller.after_inference(self, result[1])
                                return result
                            except TurnRetry:
                                next_retry_namespace = controller.begin_retry()
                            finally:
                                self.model_name_underline_replaced = original_namespace
                                if active_retry_namespace is not None:
                                    controller.cleanup_retry(active_retry_namespace)
                            active_retry_namespace = next_retry_namespace
                    else:
                        try:
                            working_entry = (deepcopy(original_test_entry)
                                             if controller.selector is not None else test_entry)
                            result = super().inference(
                                working_entry, include_input_log, exclude_state_log)
                            controller.after_inference(self, result[1])
                        except TurnRetry:
                            retry_namespace = controller.begin_retry()
                            self.model_name_underline_replaced = retry_namespace
                            try:
                                result = super().inference(
                                    deepcopy(original_test_entry),
                                    include_input_log,
                                    exclude_state_log,
                                )
                                controller.after_inference(self, result[1])
                            finally:
                                self.model_name_underline_replaced = original_namespace
                                controller.cleanup_retry(retry_namespace)
                        return result
                except Exception as error:
                    controller.record_handler_error(error)
                    raise
                finally:
                    self.model_name_underline_replaced = original_namespace
                    controller.write_audit()
                    controller.clear()

        def _add_next_turn_user_message_FC(
                self, inference_data: dict, user_message: list[dict]) -> dict:
            self._gold_controller.before_next_turn(self)
            return super()._add_next_turn_user_message_FC(
                inference_data, user_message)

        def _parse_query_response_FC(self, api_response: Any) -> dict:
            parsed = super()._parse_query_response_FC(api_response)
            controller = getattr(self, "_gold_controller", None)
            if controller is not None:
                controller.record_parse(self, parsed)
            return parsed

        def _query_FC(self, inference_data: dict):
            controller = self._gold_controller
            replay = controller.replay_response()
            if replay is not None:
                return replay
            kwargs = {
                "messages": inference_data["message"],
                "model": model,
                "temperature": self.temperature,
                "store": False,
                "max_completion_tokens": 4096,
            }
            if inference_data.get("tools"):
                kwargs["tools"] = inference_data["tools"]
            context = controller.request_context()
            if context is not None:
                extra_body: Dict[str, Any] = {"c2kv_eval_context": context}
                oracle = controller.oracle_payload()
                if oracle is not None:
                    extra_body["c2kv_oracle"] = oracle
                kwargs["extra_body"] = extra_body
            inference_data["inference_input_log"] = {
                "message": repr(inference_data["message"]),
                "tools": inference_data["tools"],
            }
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - t0
            controller.record_query(response, latency)
            return response, latency

    MODEL_CONFIG_MAPPING[handler_name] = ModelConfig(
        model_name=model,
        display_name="c2kv-hf",
        url="",
        org="c2kv",
        license="",
        model_handler=C2KVHandler,
        is_fc_model=True,
        underscore_to_dot=False,
    )


def official_category_counts(category: str) -> Dict[str, int]:
    """Resolve a BFCL category/collection with the pinned official loader.

    A collection such as ``multi_turn`` is not a data filename. Memory and
    web-search concrete categories also share source files and are rewritten
    by the official loader, so counting ``BFCL_v4_<argument>.json`` is not a
    general denominator rule.
    """
    try:
        from bfcl_eval.utils import load_dataset_entry, parse_test_category_argument
    except ImportError as error:
        raise RuntimeError(
            "BFCL expected-count resolution requires the pinned bfcl_eval package"
        ) from error

    concrete = parse_test_category_argument([category])
    if not concrete:
        raise ValueError(f"BFCL category resolves to no concrete categories: {category}")
    if "format_sensitivity" in concrete:
        raise ValueError(
            "BFCL format_sensitivity is non-scoring and cannot produce a scored summary"
        )

    counts: Dict[str, int] = {}
    for name in concrete:
        entries = load_dataset_entry(
            name, include_prereq=False, include_language_specific_hint=False)
        ids = [entry.get("id") for entry in entries if isinstance(entry, dict)]
        if not ids or any(item is None for item in ids):
            raise RuntimeError(f"BFCL category has no complete official id set: {name}")
        if len(ids) != len(set(map(str, ids))):
            raise RuntimeError(f"BFCL category has duplicate official ids: {name}")
        counts[name] = len(ids)
    return counts


def expected_count(category: str) -> int:
    """Number of scored tasks after official collection expansion."""
    return sum(official_category_counts(category).values())


def _score_header(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            header = next((json.loads(line) for line in handle if line.strip()), None)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid BFCL score file {path}: {error}") from error
    if not isinstance(header, dict):
        raise RuntimeError(f"BFCL score file has no JSON header: {path}")

    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if (not isinstance(total, int) or isinstance(total, bool) or total <= 0
            or not isinstance(correct, int) or isinstance(correct, bool)
            or correct < 0 or correct > total
            or not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool)
            or not math.isfinite(float(accuracy))):
        raise RuntimeError(f"invalid BFCL score header in {path}: {header}")
    derived = correct / total
    if not math.isclose(float(accuracy), derived, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"BFCL score header accuracy mismatch in {path}: "
            f"accuracy={accuracy} correct_count={correct} total_count={total}")
    return header


def collect_score_summary(project_root: "Path | str", handler_name: str,
                          selected_counts: Mapping[str, int]) -> Dict[str, Any]:
    """Read and validate official per-category score headers without inference."""
    score_root = Path(project_root) / "score" / handler_name
    if not selected_counts:
        raise ValueError("BFCL selected category counts are empty")

    headers = []
    total = 0
    correct = 0
    for category, expected in selected_counts.items():
        if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
            raise ValueError(f"invalid selected count for BFCL/{category}: {expected!r}")
        filename = f"BFCL_v4_{category}_score.json"
        hits = sorted(score_root.rglob(filename)) if score_root.is_dir() else []
        if len(hits) != 1:
            raise RuntimeError(
                f"expected one BFCL score file for {category} under {score_root}, "
                f"found {len(hits)}")
        header = _score_header(hits[0])
        if header["total_count"] != expected:
            raise RuntimeError(
                f"BFCL scorer denominator mismatch for {category}: "
                f"total_count={header['total_count']} selected_expected={expected}")
        total += header["total_count"]
        correct += header["correct_count"]
        headers.append({
            "category": category,
            "path": str(hits[0]),
            "accuracy": float(header["accuracy"]),
            "correct_count": header["correct_count"],
            "total_count": header["total_count"],
        })

    selected_expected = sum(selected_counts.values())
    if total != selected_expected:
        raise RuntimeError(
            f"BFCL scorer coverage mismatch: n_scored={total} n={selected_expected}")
    return {
        "n": selected_expected,
        "n_total": selected_expected,
        "n_scored": total,
        "correct_count": correct,
        "semantic_score": correct / total,
        "bfcl_accuracy": correct / total,
        "official_correct_count": correct,
        "official_total_count": total,
        "official_score": correct / total,
        "official_metric_kind": "bfcl_accuracy",
        "official_score_headers": headers,
        "scored": True,
    }


def run(ctx: RunContext) -> Dict[str, Any]:
    """Adapter entry: run the official CLI from inside the gorilla checkout.

    BFCL reads ``BFCL_PROJECT_ROOT`` when its modules are first imported.
    Point it at this run's output directory before ``run_bfcl`` imports BFCL,
    while keeping cwd in the checkout so the installed package data resolves.
    The handler expects an OpenAI base_url WITH ``/v1``.

    Per-task proxy cost joins through ``c2kv_eval_context.task_id``; handler
    end-to-end wall time is written separately under ``task_audit``.
    """
    project_root = ctx.out_dir.resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    previous_project_root = os.environ.get("BFCL_PROJECT_ROOT")
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    os.chdir(ctx.opt("bfcl_dir") or default_bfcl_dir())
    try:
        from ..arms import get_arm

        gold_recovery = get_arm(ctx.arm).gold_recovery
        num_threads = int(ctx.opt("num_workers", 1))
        oracle_max_events = int(ctx.opt("bfcl_oracle_max_events", 1))
        task_audit_path = (
            project_root / "task_audit" / f"{handler_key(ctx.arm)}.jsonl")
        summary = run_bfcl(
            v1(ctx.base_url),
            categories=ctx.opt("categories", "multi_turn_base"),
            mode=ctx.opt("mode", "both"),
            run_ids=ctx.opt("run_ids"),
            model=ctx.model,
            handler_name=handler_key(ctx.arm),
            project_root=project_root,
            gold_recovery=gold_recovery,
            task_audit_path=task_audit_path,
            num_threads=num_threads,
            bfcl_oracle_max_events=oracle_max_events,
        )
    finally:
        os.chdir(prev_cwd)
        if previous_project_root is None:
            os.environ.pop("BFCL_PROJECT_ROOT", None)
        else:
            os.environ["BFCL_PROJECT_ROOT"] = previous_project_root
    summary["cost_join"] = COST_JOIN
    return summary


# Every real request carries an explicit BFCL task id.  reqlog aggregates it
# into request_log_summary.task_costs, while task_audit measures end-to-end
# handler wall time including tool execution and any privileged gold check.
COST_JOIN = ("explicit c2kv_eval_context.task_id join available in "
             "request_log_summary.task_costs")


def run_bfcl(base_url: str, categories: str = "multi_turn_base",
             mode: str = "both", run_ids: "list[str] | str | None" = None,
             model: str = SERVED_MODEL,
             handler_name: str = MODEL_NAME,
             project_root: "Path | str | None" = None,
             gold_recovery: "str | None" = None,
             task_audit_path: "Path | str | None" = None,
             num_threads: int = 1,
             bfcl_oracle_max_events: int = 1) -> Dict[str, Any]:
    """Register the handler and drive the official generate/evaluate CLI
    in-process.

    ``handler_name`` is the BFCL model key (= result-dir name); ``run``
    passes ``handler_key(arm)`` so arms never overwrite each other's results.

    ``run_ids`` subsets the category: this BFCL vintage implements subsetting
    through <project-root>/test_case_ids_to_generate.json ({"<category>":
    [ids]}) + the boolean --run-ids flag, NOT a CLI value, so the ids are
    written to that file and the flag is passed (comma string or list).

    ``project_root`` is the isolated output root used by BFCL's official
    RESULT_PATH, SCORE_PATH and TEST_IDS_TO_GENERATE_PATH.  It must be set
    before the first official import.

    Terminal-state check (acceptance 1): every selected entry must have a
    result row and, after evaluation, an official score row. Generation-only
    summaries deliberately contain no ``n_scored`` or ``semantic_score``."""
    if mode not in ("generate", "evaluate", "both"):
        raise ValueError(f"invalid BFCL mode: {mode}")
    if (not isinstance(bfcl_oracle_max_events, int)
            or isinstance(bfcl_oracle_max_events, bool)
            or bfcl_oracle_max_events <= 0):
        raise ValueError("bfcl_oracle_max_events must be a positive integer")
    if bfcl_oracle_max_events > 1 and gold_recovery is None:
        raise ValueError("BFCL multi-event recovery requires a gold-recovery arm")
    project_root = Path(
        project_root or os.environ.get("BFCL_PROJECT_ROOT") or Path.cwd()
    ).resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    previous_project_root = os.environ.get("BFCL_PROJECT_ROOT")
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    try:
        audit_path = (Path(task_audit_path) if task_audit_path else
                      project_root / "task_audit" / f"{handler_name}.jsonl")
        install_handler(
            base_url,
            model=model,
            handler_name=handler_name,
            gold_recovery=gold_recovery,
            task_audit_path=audit_path,
            bfcl_oracle_max_events=bfcl_oracle_max_events,
        )
        category_counts = official_category_counts(categories)
        ids: Optional[List[str]] = None
        if run_ids:
            ids = ([i.strip() for i in run_ids.split(",") if i.strip()]
                   if isinstance(run_ids, str) else list(run_ids))
            if len(category_counts) != 1:
                raise ValueError(
                    "BFCL --run-ids requires one concrete category, not collection "
                    f"{categories!r} -> {sorted(category_counts)}")
            if not ids:
                raise ValueError("BFCL --run-ids resolved to an empty id list")
            # atomic write: concurrent runs racing on one file truncated ids
            id_file = project_root / "test_case_ids_to_generate.json"
            tmp = id_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({categories: ids}), encoding="utf-8")
            tmp.replace(id_file)
            selected_counts = {next(iter(category_counts)): len(ids)}
        else:
            selected_counts = category_counts
        expected = sum(selected_counts.values())
        generation_started = time.perf_counter()
        if mode in ("generate", "both"):
            run_cli(generate_argv(
                handler_name, categories, ids, num_threads=num_threads))
        generation_phase_wall_time = time.perf_counter() - generation_started
        official_scoring_time = 0.0
        if mode in ("evaluate", "both"):
            scoring_started = time.perf_counter()
            run_cli(evaluate_argv(handler_name, categories, ids))
            official_scoring_time = time.perf_counter() - scoring_started
        from .. import terminal_check

        ids_str = ",".join(ids or [])
        for category, selected in selected_counts.items():
            code = terminal_check.check_bfcl(
                selected, ids_str, handler=handler_name,
                category=category, root=project_root)
            if code != 0:
                raise SystemExit(
                    f"FATAL: bfcl terminal-state check failed for {category} (rc={code})")

        summary: Dict[str, Any] = {
            "benchmark": "bfcl", "categories": categories, "mode": mode,
            "n_total": expected,
            "bfcl_project_root": str(project_root),
            "bfcl_num_threads": num_threads,
            "bfcl_oracle_max_events": bfcl_oracle_max_events,
            "generation_phase_wall_time": generation_phase_wall_time,
            "official_scoring_time": official_scoring_time,
            "generation_phase_scope": (
                "official BFCL generate phase: model/proxy requests, tool execution, "
                "response parsing and harness overhead; excludes official scoring and server startup"
            ),
        }
        if mode == "generate":
            summary.update({"n_generated": expected, "scored": False})
        else:
            summary.update(collect_score_summary(
                project_root, handler_name, selected_counts))
            if mode == "both":
                summary["n_generated"] = expected
        summary.update(summarize_audit(audit_path))
        if gold_recovery is not None:
            summary["gold_recovery_handler_concurrency"] = 1
            if bfcl_oracle_max_events == 1:
                summary["bfcl_oracle_kind"] = ORACLE_KIND
                summary["bfcl_oracle_version"] = ORACLE_VERSION
                summary["bfcl_oracle_variant"] = "frozen_single_event_v2"
            else:
                summary["bfcl_oracle_kind"] = MULTI_EVENT_ORACLE_KIND
                summary["bfcl_oracle_version"] = MULTI_EVENT_ORACLE_VERSION
                summary["bfcl_oracle_variant"] = "event_budget_sensitivity"
        return summary
    finally:
        if previous_project_root is None:
            os.environ.pop("BFCL_PROJECT_ROOT", None)
        else:
            os.environ["BFCL_PROJECT_ROOT"] = previous_project_root


def run_cli(argv):
    from bfcl_eval.__main__ import cli

    # Typer's standalone mode raises SystemExit(0) after generate, which
    # would skip evaluate, terminal checks and the unified run summary.
    cli(args=argv, prog_name="bfcl", standalone_mode=False)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:34000/v1")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--mode", choices=["generate", "evaluate", "both"], default="both")
    parser.add_argument("--run-ids", default="")
    parser.add_argument("--model", default=SERVED_MODEL,
                        help="served model name at the endpoint")
    parser.add_argument("--handler-name", default=MODEL_NAME,
                        help="BFCL model key / result-dir name")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="BFCL --num-threads value")
    parser.add_argument(
        "--bfcl-oracle-max-events", type=_positive_int, default=1,
        help="gold recovery retries per task; 1 keeps the frozen v2 protocol",
    )
    args = parser.parse_args(argv)
    # one code path: the CLI used to re-implement run() and had drifted
    # (evaluate got --run-ids instead of --partial-eval, no terminal gate)
    ids = [i.strip() for i in args.run_ids.split(",") if i.strip()] or None
    summary = run_bfcl(args.base_url, categories=args.categories, mode=args.mode,
                       run_ids=ids, model=args.model,
                       handler_name=args.handler_name,
                       num_threads=args.num_workers,
                       bfcl_oracle_max_events=args.bfcl_oracle_max_events)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
