#!/usr/bin/env python3
"""Real-endpoint integration smoke for HiAgent and fixed ACON variants.

This reuses the proxy's text-arm functions in-process. It does not launch a
model server or benchmark, and it writes a small JSON result even when one
case fails. HiAgent full uses a forced first retrieval call, then exercises
the proxy's real bounded retrieval loop without forcing the retry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from c2kv_eval.portable import proxy, textarms  # noqa: E402
from c2kv_eval.portable.arms import Arm, get_arm  # noqa: E402
from c2kv_eval.portable.backends import get_backend  # noqa: E402


USER_SENTINEL = "USER_REQUEST_SENTINEL_KEEP_7319"
RETRIEVAL_SECRET = "RETRIEVED_SECRET_4821"


def _usage(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def _message(data: Dict[str, Any]) -> Dict[str, Any]:
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise AssertionError(f"bad chat response: {json.dumps(data)[:600]}")
    return choices[0]["message"]


def _assert_usage(usage: Dict[str, int], label: str) -> None:
    if usage["prompt_tokens"] <= 0 or usage["completion_tokens"] <= 0:
        raise AssertionError(f"{label} returned empty usage: {usage}")


def _policy_payload(staged: Dict[str, Any], arm: Arm) -> Dict[str, Any]:
    wire = dict(staged)
    wire["messages"], _ = proxy._assemble(staged["messages"], arm)
    wire.update({
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 96,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    return wire


def _chat(staged: Dict[str, Any], arm: Arm) -> Dict[str, Any]:
    return proxy._post_json("/v1/chat/completions", _policy_payload(staged, arm), 600)


def hiagent_messages() -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "Follow the user's request and use tools when instructed."},
        {"role": "user", "content": USER_SENTINEL + ": compare the completed checks."},
        {"role": "assistant", "content": "Subgoal: inspect first account"},
        {"role": "tool", "tool_call_id": "first", "content":
         RETRIEVAL_SECRET + " " + "first-account-detail " * 90},
        {"role": "assistant", "content": "Subgoal: inspect second account"},
        {"role": "tool", "tool_call_id": "second", "content":
         "SECOND_ACCOUNT_VALUE_93 " + "second-account-detail " * 90},
        {"role": "assistant", "content": "Subgoal: compare accounts"},
        {"role": "user", "content": (
            "If the first completed trajectory is hidden, call hiagent_retrieve "
            "for subgoal 1. If a line beginning RETRIEVED_SECRET is visible, "
            "answer with that full line and no tool call.")},
    ]


def acon_history_messages() -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Act on the user's tool results."},
        {"role": "user", "content": USER_SENTINEL + ": retain the account decision."},
    ]
    for index in range(20):
        messages.extend([
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": f"call-{index}", "type": "function",
                              "function": {"name": "inspect",
                                           "arguments": json.dumps({"index": index})}}]},
            {"role": "tool", "tool_call_id": f"call-{index}",
             "content": f"OBSERVATION_{index} " + "account-state-value " * 48},
        ])
    messages.append({"role": "user", "content": "Give a one-sentence status."})
    return messages


def acon_observation_messages() -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": "Act on the user's tool results."},
        {"role": "user", "content": USER_SENTINEL + ": retain the relevant account id."},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "large", "type": "function",
                          "function": {"name": "inspect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "large",
         "content": "ACCOUNT_ID_REQUIRED_884 " + "irrelevant-record-value " * 210},
        {"role": "user", "content": "State the required account id."},
    ]


def _normal_case(name: str, messages: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    textarms.reset_state()
    arm = get_arm(name)
    original = {"model": model, "messages": messages}
    staged, stats = proxy._apply_text_arm(original, arm, "smoke-" + name)
    compressor_usage = stats.get("compressor_usage") or {}
    _assert_usage({
        "prompt_tokens": int(compressor_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(compressor_usage.get("completion_tokens") or 0),
    }, name + " compressor")
    if int(stats.get("n_compressor_calls") or 0) <= 0:
        raise AssertionError(f"{name} did not call its compressor")
    if int(stats.get("out_chars") or 0) >= int(stats.get("raw_chars") or 0):
        raise AssertionError(f"{name} did not shorten history: {stats}")
    if USER_SENTINEL not in json.dumps(staged["messages"], ensure_ascii=False):
        raise AssertionError(f"{name} dropped the user request")
    data = _chat(staged, arm)
    policy_usage = _usage(data)
    _assert_usage(policy_usage, name + " policy")
    content = str(_message(data).get("content") or "").strip()
    if not content:
        raise AssertionError(f"{name} returned empty policy content")
    return {
        "name": name,
        "status": "passed",
        "n_compressor_calls": stats["n_compressor_calls"],
        "compressor_usage": compressor_usage,
        "policy_usage": policy_usage,
        "raw_chars": stats["raw_chars"],
        "out_chars": stats["out_chars"],
        "user_request_preserved": True,
        "history_shortened": True,
        "policy_content_nonempty": True,
    }


def _hiagent_full_case(model: str) -> Dict[str, Any]:
    textarms.reset_state()
    arm = get_arm("hiagent_full")
    original = {"model": model, "messages": hiagent_messages()}
    forced = dict(original)
    forced["tool_choice"] = {
        "type": "function", "function": {"name": textarms.HIAGENT_RETRIEVE_TOOL_NAME}}
    staged, stats = proxy._apply_text_arm(forced, arm, "smoke-hiagent-full")
    if RETRIEVAL_SECRET in json.dumps(staged["messages"], ensure_ascii=False):
        raise AssertionError("HiAgent initial summarized context exposed hidden trajectory")
    initial = _chat(staged, arm)
    ids = textarms.hiagent_retrieval_request(_message(initial))
    if not ids:
        raise AssertionError("model did not emit the required HiAgent retrieval meta-call")

    retry_payloads: List[Dict[str, Any]] = []

    def send(retry_staged: Dict[str, Any]) -> Dict[str, Any]:
        retry_payloads.append(retry_staged)
        return _chat(retry_staged, arm)

    final = proxy._hiagent_retrieval_loop(
        original, arm, "smoke-hiagent-full", initial, stats, send)
    final_message = _message(final)
    if textarms.hiagent_retrieval_request(final_message) is not None:
        raise AssertionError("HiAgent retrieval meta-tool leaked as the final action")
    final_content = str(final_message.get("content") or "").strip()
    if not final_content:
        raise AssertionError("HiAgent full returned empty final content")
    if not retry_payloads or RETRIEVAL_SECRET not in json.dumps(
            retry_payloads[0]["messages"], ensure_ascii=False):
        raise AssertionError("HiAgent retry did not reveal the requested trajectory")
    if USER_SENTINEL not in json.dumps(retry_payloads[0]["messages"], ensure_ascii=False):
        raise AssertionError("HiAgent full dropped the user request")
    if int(stats.get("out_chars") or 0) >= int(stats.get("raw_chars") or 0):
        raise AssertionError(f"HiAgent full did not shorten history: {stats}")
    compressor_usage = stats.get("compressor_usage") or {}
    _assert_usage({
        "prompt_tokens": int(compressor_usage.get("prompt_tokens") or 0),
        "completion_tokens": int(compressor_usage.get("completion_tokens") or 0),
    }, "hiagent_full compressor")
    _assert_usage(_usage(final), "hiagent_full final policy")
    return {
        "name": "hiagent_full",
        "status": "passed",
        "initial_retrieval_ids": ids,
        "retrieved_subgoals": stats.get("retrieved_subgoals"),
        "retrieval_usage": stats.get("retrieval_usage"),
        "compressor_usage": compressor_usage,
        "final_policy_usage": _usage(final),
        "raw_chars": stats.get("raw_chars"),
        "out_chars": stats.get("out_chars"),
        "user_request_preserved": True,
        "history_shortened": True,
        "trajectory_revealed_to_retry": True,
        "meta_tool_leaked": False,
        "final_content_nonempty": True,
    }


def run(base_url: str, model: str = "c2kv-agent") -> Dict[str, Any]:
    proxy.UPSTREAM = base_url.rstrip("/")
    proxy.BACKEND = get_backend("sglang", proxy._post_json)
    proxy.QUERY_PROJECTION = "base"
    cases = []
    specs = [
        ("hiagent_summary", hiagent_messages),
        ("hiagent_full", None),
        *[(f"acon_hist_{guideline}", acon_history_messages)
          for guideline in textarms.ACON_GUIDELINES],
        *[(f"acon_obs_{guideline}", acon_observation_messages)
          for guideline in textarms.ACON_GUIDELINES],
    ]
    for name, builder in specs:
        try:
            result = (_hiagent_full_case(model) if name == "hiagent_full"
                      else _normal_case(name, builder(), model))
        except Exception as error:  # preserve every case outcome in JSON
            result = {
                "name": name,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(limit=8),
            }
        cases.append(result)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": base_url,
        "model": model,
        "query_projection": "base",
        "kind": "synthetic_textarm_integration_smoke",
        "cases": cases,
        "passed": sum(case["status"] == "passed" for case in cases),
        "failed": sum(case["status"] != "passed" for case in cases),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="c2kv-agent")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.base_url, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
