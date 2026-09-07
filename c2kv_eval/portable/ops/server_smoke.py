"""Exercise the shipped proxy and SGLang KV paths with a synthetic conversation.

This is an integration test, not a benchmark score. It starts only its own
proxy children; the caller supplies an already running, isolated server.
Raw responses and request logs are retained in a new output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib import request

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from c2kv_eval.portable.arms import get_arm

OPENER = request.build_opener(request.ProxyHandler({}))
DEFAULT_ARMS = (
    "full", "c2kv", "c2kv16", "hybrid", "c2kv_repair",
    "c2kv_repair_tail", "c2kv_repair_inplace", "hybrid_repair",
    "history_kv_streamingllm_r312", "history_kv_h2o_r312",
    "history_kv_snapkv_r312", "history_kv_pyramidkv_r312",
    "cacheblend_r16", "cacheblend_r15_k",
)


def http(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with OPENER.open(req, timeout=600) as response:
        raw = response.read()
    if payload is None:
        return raw.decode("utf-8")
    return json.loads(raw) if raw else None


def synthetic_payload(model):
    messages = [{"role": "system", "content":
                 "You are a helpful assistant. Use the supplied tool when asked for a stock price."}]
    for topic in ("weather", "travel", "books", "music", "gardening"):
        messages.extend([
            {"role": "user", "content": f"Remember that I enjoy {topic}."},
            {"role": "assistant", "content": f"I will remember your interest in {topic}."},
        ])
    messages.append({"role": "user", "content": "Get the current stock price for AAPL."})
    return {
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": 48, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "tools": [{"type": "function", "function": {
            "name": "get_stock_price", "description": "Get the current price of a stock.",
            "parameters": {"type": "object", "properties": {
                "symbol": {"type": "string"}}, "required": ["symbol"]},
            "response": {"type": "number"},
        }}],
    }


def check_responses(arm_name, responses, rows, query_projection, server_query_projection=None):
    arm = get_arm(arm_name)
    failures = []
    def require(condition, message):
        if not condition:
            failures.append(message)
    require(len(rows) == len(responses), "missing proxy request-log rows")
    for i, (response, row) in enumerate(zip(responses, rows)):
        prefix = f"request {i}: "
        choice = (response.get("choices") or [{}])[0]
        require(choice.get("finish_reason") in ("stop", "length", "tool_calls"),
                prefix + "generation did not finish")
        require(row.get("status") == "ok", prefix + f"proxy status={row.get('status')}")
        require(isinstance(row.get("kv_resident_tokens"), int), prefix + "KV accounting missing")
        require(row.get("c2kv_query_proj") == (server_query_projection or query_projection),
                prefix + "server projection flag mismatch")
        if arm.compress_history:
            require((row.get("n_gist_messages") or 0) > 0, prefix + "no history compressed")
            require(row.get("c2kv_query_proj_effective") == query_projection,
                    prefix + "effective projection mismatch")
            require(row.get("c2kv_query_proj_decode_verified") is True,
                    prefix + "decode projection unverified")
        if arm.repair:
            require(row.get("repair_frame_delta_status") == "measured", prefix + "repair frame unmeasured")
            require(row.get("repair_frame_delta") == 0, prefix + "repair extraction frame differs")
            layout = row.get("c2kv_layout") or []
            repairs = [item for item in layout if item.get("kind") == "repair"]
            require(bool(repairs), prefix + "raw repair was not injected")
            require(all(item.get("placement") == arm.repair["placement"] for item in repairs),
                    prefix + "repair placement mismatch")
            frame = row.get("repair_frame") or {}
            require(frame.get("ok") is not False, prefix + "repair/gist frame mismatch")
        if arm.kv_reuse or arm.history_kv:
            require(row.get("c2kv_query_proj_effective") == "base",
                    prefix + "raw-KV baseline must use base query projections")
        if arm.history_kv:
            require(row.get("history_kv_method") == arm.history_kv["method"],
                    prefix + "history-KV method not reported")
        if arm.kv_reuse:
            require(row.get("kv_reuse_method") == arm.kv_reuse["method"],
                    prefix + "KV reuse method not reported")
    if len(responses) == 2:
        messages = []
        for response in responses:
            message = (response.get("choices") or [{}])[0].get("message") or {}
            messages.append({"content": message.get("content"),
                             "tools": [call.get("function") for call in
                                       (message.get("tool_calls") or [])]})
        require(messages[0] == messages[1], "cold/warm greedy responses differ")
    return failures


def run_arm(args, arm_name):
    arm_dir = args.out / arm_name
    arm_dir.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = arm_dir / "requests.jsonl"
    command = [sys.executable, "-m", "c2kv_eval.portable.proxy", "--upstream", args.upstream,
               "--arm", arm_name, "--backend", "sglang", "--port", str(port),
               "--request-log", str(log_path), "--doc-packing", "turn",
               "--max-doc-length", str(args.max_doc_length),
               "--max-doc-num", str(args.max_doc_num),
               "--query-projection", args.query_projection]
    responses = []
    errors = []
    with (arm_dir / "proxy.log").open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                cwd=Path(__file__).resolve().parents[3])
        try:
            deadline = time.monotonic() + 30
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"proxy exited with {proc.returncode}")
                try:
                    http(f"http://127.0.0.1:{port}/health")
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError("proxy startup timeout")
                    time.sleep(0.2)
            for _ in range(2):
                responses.append(http(f"http://127.0.0.1:{port}/v1/chat/completions",
                                      synthetic_payload(args.model)))
            # The response is written immediately before the request-log row.
            time.sleep(0.2)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if log_path.exists() else []
    (arm_dir / "responses.json").write_text(json.dumps(responses, indent=2), encoding="utf-8")
    errors.extend(check_responses(arm_name, responses, rows, args.query_projection,
                                   args.server_query_projection))
    if len(responses) != 2:
        errors.append("expected two completed requests")
    return {"arm": arm_name, "passed": not errors, "errors": errors,
            "n_responses": len(responses), "n_log_rows": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model", default="c2kv-agent")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--query-projection", choices=("base", "gist"), default="base")
    parser.add_argument("--server-query-projection", choices=("base", "gist"),
                        help="expected server default, if distinct from the per-request query regime")
    parser.add_argument("--max-doc-length", type=int, default=512)
    parser.add_argument("--max-doc-num", type=int, default=12)
    args = parser.parse_args()
    for name in args.arms:
        get_arm(name)
    args.out.mkdir(parents=True, exist_ok=False)
    summary = {"kind": "synthetic_integration_test", "upstream": args.upstream,
               "query_projection": args.query_projection, "arms": []}
    for name in args.arms:
        result = run_arm(args, name)
        summary["arms"].append(result)
        print(json.dumps(result), flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    raise SystemExit(0 if all(row["passed"] for row in summary["arms"]) else 1)


if __name__ == "__main__":
    main()
