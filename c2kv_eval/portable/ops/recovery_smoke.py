"""Run five bounded live checks of request-local C2KV recovery.

The caller supplies an already running SGLang endpoint.  This script starts
one plain ``c2kv4`` proxy child, sends exactly one chat request for each
recovery case, and retains the raw responses, proxy output, request JSONL,
and a machine-readable summary under a new output directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from c2kv_eval.portable.ops.server_smoke import http, synthetic_payload
from c2kv_eval.portable.request_recovery import RECOVERY_CONTROL_FIELD


EXPECTED_HISTORY_DOCS = 5
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
CASES = (
    {
        "name": "append_first",
        "control": {"operation": "append", "triggered": True, "selector": "first"},
        "placement": "append_keep_ledger",
        "retried": True,
    },
    {
        "name": "replace_first",
        "control": {"operation": "replace", "triggered": True, "selector": "first"},
        "placement": "in_place",
        "retried": True,
    },
    {
        "name": "append_witness_weather",
        "control": {
            "operation": "append",
            "triggered": True,
            "selector": "witness",
            "target_values": ["weather"],
        },
        "placement": "append_keep_ledger",
        "retried": True,
    },
    {
        "name": "retry_full",
        "control": {"operation": "retry_full", "triggered": True},
        "placement": None,
        "retried": True,
    },
    {
        "name": "witness_no_match",
        "control": {
            "operation": "append",
            "triggered": True,
            "selector": "witness",
            "target_values": ["c2kv-recovery-smoke-no-match-91f443c0"],
        },
        "placement": "append_keep_ledger",
        "retried": False,
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _free_port(requested: int) -> int:
    if requested:
        return requested
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _upstream_base(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if not value.startswith(("http://", "https://")):
        raise ValueError("--endpoint must start with http:// or https://")
    return value.rstrip("/")


def _wait_until_ready(proxy_url: str, proc: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"proxy exited with {proc.returncode}")
        try:
            http(f"{proxy_url}/health")
            return
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError("proxy startup timeout")
            time.sleep(0.2)


def _wait_for_log_rows(path: Path, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if len(_read_jsonl(path)) >= expected:
                return
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(0.05)


def _require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _usage_sum(metadata: dict[str, Any]) -> dict[str, int]:
    initial = metadata.get("initial_usage") or {}
    retry = metadata.get("retry_usage") or {}
    result = {
        key: int(initial.get(key) or 0) + int(retry.get(key) or 0)
        for key in USAGE_KEYS
    }
    if not result["total_tokens"]:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def _check_usage(
    errors: list[str], response: dict[str, Any], recovery: dict[str, Any], retried: bool
) -> dict[str, Any]:
    usage = response.get("usage") or {}
    observed = {key: usage.get(key) for key in USAGE_KEYS}
    if retried:
        expected = _usage_sum(recovery)
        _require(errors, observed == expected, f"usage is not the sum of both attempts: {observed} != {expected}")
        return {"observed": observed, "expected_sum": expected, "source": "initial_usage+retry_usage"}

    _require(
        errors,
        all(isinstance(observed[key], int) for key in USAGE_KEYS),
        f"single-attempt usage is incomplete: {observed}",
    )
    if all(isinstance(observed[key], int) for key in USAGE_KEYS):
        _require(
            errors,
            observed["total_tokens"] == observed["prompt_tokens"] + observed["completion_tokens"],
            f"single-attempt total_tokens is inconsistent: {observed}",
        )
    return {"observed": observed, "source": "single_response"}


def _check_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    proxy = response.get("c2kv_proxy") or {}
    recovery = proxy.get("request_recovery") or {}
    control = case["control"]
    layout = proxy.get("c2kv_layout")
    _require(errors, isinstance(layout, list), "response.c2kv_proxy.c2kv_layout is missing")
    layout = layout if isinstance(layout, list) else []
    gists = [item for item in layout if item.get("kind") == "gist"]
    repairs = [item for item in layout if item.get("kind") == "repair"]

    choice = (response.get("choices") or [{}])[0]
    _require(errors, choice.get("finish_reason") in ("stop", "length", "tool_calls"),
             f"generation did not finish: {choice.get('finish_reason')!r}")
    _require(errors, proxy.get("arm") == "c2kv4", f"unexpected proxy arm: {proxy.get('arm')!r}")
    _require(errors, proxy.get("ratio") == 4, f"unexpected C2KV ratio: {proxy.get('ratio')!r}")
    _require(errors, proxy.get("c2kv_query_proj_effective") == "base",
             "effective query projection is not base")
    _require(errors, proxy.get("c2kv_query_proj_decode_verified") is True,
             "decode query projection is not verified")
    _require(errors, bool(recovery), "response.c2kv_proxy.request_recovery is missing")
    _require(errors, recovery.get("operation") == control["operation"],
             f"operation mismatch: {recovery.get('operation')!r}")
    _require(errors, recovery.get("triggered") is True, "recovery was not marked triggered")
    _require(errors, recovery.get("selector") == control.get("selector", "first"),
             f"selector mismatch: {recovery.get('selector')!r}")
    _require(errors, recovery.get("candidate_count") == EXPECTED_HISTORY_DOCS,
             f"candidate_count is not {EXPECTED_HISTORY_DOCS}: {recovery.get('candidate_count')!r}")

    if control["operation"] != "retry_full":
        _require(errors, proxy.get("n_gist_messages") == EXPECTED_HISTORY_DOCS,
                 f"proxy gist count is not {EXPECTED_HISTORY_DOCS}: {proxy.get('n_gist_messages')!r}")
        _require(errors, proxy.get("history_packed_candidate_doc_count") == EXPECTED_HISTORY_DOCS,
                 "packed candidate count differs from the synthetic history")
        _require(errors, isinstance(proxy.get("gist_tokens"), int) and proxy["gist_tokens"] > 0,
                 f"gist token count is invalid: {proxy.get('gist_tokens')!r}")
        _require(errors, isinstance(proxy.get("original_tokens"), int) and proxy["original_tokens"] > 0,
                 f"original token count is invalid: {proxy.get('original_tokens')!r}")
        _require(errors, proxy.get("history_retained_fraction") == 1.0,
                 f"synthetic history was not fully retained: {proxy.get('history_retained_fraction')!r}")

    if case["retried"]:
        _require(errors, recovery.get("status") == "retried",
                 f"recovery status is not retried: {recovery.get('status')!r}")
        _require(errors, recovery.get("generation_attempts") == 2,
                 f"generation_attempts is not 2: {recovery.get('generation_attempts')!r}")
        _require(errors, recovery.get("environment_rollback") is False,
                 "environment_rollback is not false")
        _require(errors, isinstance(recovery.get("initial_usage"), dict),
                 "initial_usage is missing")
        _require(errors, isinstance(recovery.get("retry_usage"), dict),
                 "retry_usage is missing")
    else:
        _require(errors, recovery.get("status") == "no_literal_witness",
                 f"no-match status is wrong: {recovery.get('status')!r}")
        # The no-retry branch currently represents its one attempt implicitly:
        # it has no retry usage and may omit generation_attempts entirely.
        attempts = recovery.get("generation_attempts", 1)
        _require(errors, attempts == 1, f"no-match generation attempts is not 1: {attempts!r}")
        _require(errors, "retry_usage" not in recovery, "no-match unexpectedly has retry_usage")

    if control["operation"] in ("append", "replace") and case["retried"]:
        _require(errors, recovery.get("selected_index") == 0,
                 f"first/witness selection did not choose doc 0: {recovery.get('selected_index')!r}")
        _require(errors, isinstance(recovery.get("selected_out_index"), int),
                 "selected_out_index is missing")
        _require(errors, recovery.get("placement") == case["placement"],
                 f"metadata placement mismatch: {recovery.get('placement')!r}")
        _require(errors, len(repairs) == 1, f"expected one repair layout entry, got {len(repairs)}")
        _require(errors, all(item.get("placement") == case["placement"] for item in repairs),
                 f"repair layout placement is not {case['placement']}")
        expected_gists = EXPECTED_HISTORY_DOCS
        if case["placement"] == "in_place":
            expected_gists -= 1
        _require(errors, len(gists) == expected_gists,
                 f"gist layout count is {len(gists)}, expected {expected_gists}")
    elif control["operation"] == "retry_full":
        _require(errors, recovery.get("raw_regenerate") is True,
                 "retry_full was not marked raw_regenerate")
        _require(errors, recovery.get("raw_kv_source") == "full_history_prefill",
                 f"retry_full source mismatch: {recovery.get('raw_kv_source')!r}")
        _require(errors, proxy.get("n_gist_messages") == 0,
                 f"retry_full retained gist messages: {proxy.get('n_gist_messages')!r}")
        _require(errors, proxy.get("gist_tokens") == 0,
                 f"retry_full retained gist tokens: {proxy.get('gist_tokens')!r}")
        _require(errors, not gists and not repairs,
                 "retry_full response still reports injected gist/repair layout")
    else:
        _require(errors, recovery.get("selected_index") is None,
                 "no-match unexpectedly selected a history doc")
        _require(errors, len(gists) == EXPECTED_HISTORY_DOCS,
                 f"no-match gist layout count is {len(gists)}, expected {EXPECTED_HISTORY_DOCS}")
        _require(errors, not repairs, "no-match response contains a repair layout entry")

    usage_check = _check_usage(errors, response, recovery, bool(case["retried"]))
    return {
        "passed": not errors,
        "errors": errors,
        "recovery": recovery,
        "counts": {
            "candidate_count": recovery.get("candidate_count"),
            "n_gist_messages": proxy.get("n_gist_messages"),
            "gist_layout_entries": len(gists),
            "repair_layout_entries": len(repairs),
        },
        "usage_check": usage_check,
    }


def _check_log_row(
    case: dict[str, Any], response: dict[str, Any], rows: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    matching = [
        row for row in rows
        if (row.get("eval_context") or {}).get("recovery_smoke_case") == case["name"]
    ]
    _require(errors, len(matching) == 1,
             f"expected one request-log row for case, got {len(matching)}")
    if len(matching) != 1:
        return errors
    row = matching[0]
    proxy = response.get("c2kv_proxy") or {}
    _require(errors, row.get("status") == "ok", f"request-log status is {row.get('status')!r}")
    _require(errors, row.get("arm") == "c2kv4", f"request-log arm is {row.get('arm')!r}")
    _require(errors, row.get("request_recovery") == proxy.get("request_recovery"),
             "request-log recovery metadata differs from response")
    _require(errors, row.get("usage") == response.get("usage"),
             "request-log usage differs from response")
    _require(errors, row.get("c2kv_layout") == proxy.get("c2kv_layout"),
             "request-log layout differs from response")
    return errors


def _http_error_evidence(error: Exception) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "exception": f"{type(error).__name__}: {error}",
    }
    if isinstance(error, HTTPError):
        try:
            body = error.read().decode("utf-8", errors="replace")
            try:
                evidence["body"] = json.loads(body)
            except json.JSONDecodeError:
                evidence["body"] = body
        except OSError as read_error:
            evidence["body_read_error"] = f"{type(read_error).__name__}: {read_error}"
    return evidence


def run(args: argparse.Namespace) -> dict[str, Any]:
    upstream = _upstream_base(args.endpoint)
    args.out.mkdir(parents=True, exist_ok=False)
    responses_dir = args.out / "responses"
    responses_dir.mkdir()
    request_log = args.out / "requests.jsonl"
    proxy_log = args.out / "proxy.log"
    port = _free_port(args.proxy_port)
    proxy_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "c2kv_eval.portable.proxy",
        "--upstream",
        upstream,
        "--backend",
        "sglang",
        "--arm",
        "c2kv4",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--request-log",
        str(request_log.resolve()),
        "--doc-packing",
        "turn",
        "--max-doc-length",
        str(args.max_doc_length),
        "--max-doc-num",
        str(args.max_doc_num),
        "--query-projection",
        "base",
        "--witness-tokenizer",
        str(args.tokenizer),
    ]
    summary: dict[str, Any] = {
        "kind": "c2kv_request_recovery_live_smoke",
        "started_at": _utc_now(),
        "endpoint": args.endpoint,
        "upstream": upstream,
        "model": args.model,
        "tokenizer": str(args.tokenizer),
        "arm": "c2kv4",
        "query_projection": "base",
        "proxy_port": port,
        "expected_chat_requests": len(CASES),
        "artifacts": {
            "proxy_log": "proxy.log",
            "request_log": "requests.jsonl",
            "responses": "responses",
            "summary": "summary.json",
        },
        "cases": [],
        "passed": False,
    }
    _write_json(args.out / "summary.json", summary)

    responses: dict[str, dict[str, Any]] = {}
    with proxy_log.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        try:
            _wait_until_ready(proxy_url, proc, args.startup_timeout)
            for index, case in enumerate(CASES, start=1):
                payload = deepcopy(synthetic_payload(args.model))
                # Frozen short-literal matching excludes an adjacent period.
                # Use a space after "weather" to exercise a positive match.
                payload["messages"][1]["content"] = "Remember that I enjoy weather activities."
                payload["messages"][2]["content"] = "I will remember your interest in weather activities."
                payload[RECOVERY_CONTROL_FIELD] = deepcopy(case["control"])
                payload["c2kv_eval_context"] = {"recovery_smoke_case": case["name"]}
                response_path = responses_dir / f"{index:02d}_{case['name']}.json"
                result: dict[str, Any] = {
                    "name": case["name"],
                    "response": str(response_path.relative_to(args.out)),
                    "passed": False,
                    "errors": [],
                }
                try:
                    response = http(f"{proxy_url}/v1/chat/completions", payload)
                    _write_json(response_path, response)
                    if not isinstance(response, dict):
                        raise TypeError(f"chat response must be a JSON object, got {type(response).__name__}")
                    responses[case["name"]] = response
                    try:
                        result.update(_check_response(case, response))
                    except Exception as error:
                        result["errors"].append(
                            f"validation {type(error).__name__}: {error}"
                        )
                    _wait_for_log_rows(request_log, index)
                except Exception as error:
                    evidence = _http_error_evidence(error)
                    if not response_path.exists():
                        _write_json(response_path, evidence)
                    result["errors"].append(evidence["exception"])
                summary["cases"].append(result)
                _write_json(args.out / "summary.json", summary)
        except Exception as error:
            summary["setup_error"] = f"{type(error).__name__}: {error}"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    try:
        rows = _read_jsonl(request_log)
    except (json.JSONDecodeError, OSError) as error:
        rows = []
        summary["request_log_error"] = f"{type(error).__name__}: {error}"
    by_name = {result["name"]: result for result in summary["cases"]}
    for case in CASES:
        response = responses.get(case["name"])
        result = by_name.get(case["name"])
        if response is None or result is None:
            continue
        result["errors"].extend(_check_log_row(case, response, rows))
        result["passed"] = not result["errors"]

    summary["finished_at"] = _utc_now()
    summary["observed_chat_request_log_rows"] = len(rows)
    summary["passed"] = (
        "setup_error" not in summary
        and "request_log_error" not in summary
        and len(summary["cases"]) == len(CASES)
        and len(rows) == len(CASES)
        and all(result["passed"] for result in summary["cases"])
    )
    _write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        required=True,
        help="running SGLang base URL (a trailing /v1 or /v1/chat/completions is accepted)",
    )
    parser.add_argument("--model", required=True, help="model name exposed by the running server")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="local tokenizer used to decode exact compressed records for witness selection",
    )
    parser.add_argument("--out", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--proxy-port", type=int, default=0, help="proxy port; 0 selects a free port")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--max-doc-length", type=int, default=512)
    parser.add_argument("--max-doc-num", type=int, default=12)
    args = parser.parse_args()
    if args.max_doc_num < EXPECTED_HISTORY_DOCS:
        parser.error(f"--max-doc-num must be at least {EXPECTED_HISTORY_DOCS} for this smoke")
    summary = run(args)
    for result in summary["cases"]:
        print(json.dumps({
            "case": result["name"],
            "passed": result["passed"],
            "errors": result["errors"],
        }, ensure_ascii=False), flush=True)
    print(json.dumps({
        "passed": summary["passed"],
        "summary": str((args.out / "summary.json").resolve()),
    }, ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
