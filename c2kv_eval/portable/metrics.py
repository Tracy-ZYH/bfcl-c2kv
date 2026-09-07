"""Two-column metrics shared by every benchmark adapter.

Protocol column (per model turn):
  - detect tool calls: OpenAI `tool_calls` objects, or `<tool_call>` JSON
    blocks in text (Qwen dialect the checkpoints emit).
  - a call is *legal* iff: name ∈ advertised tool names AND arguments are a
    JSON object satisfying the tool's JSON schema (required keys and types;
    non-schema keys are tolerated, matching the lenient scorer used in the
    BFCL AST checker).
  - a turn with no tool call is legal iff it is plain text (no broken
    tool-call syntax): an unterminated `<tool_call>` or trailing partial JSON
    counts as illegal.

Semantic column: supplied by each benchmark's official scorer; this module
only aggregates it (mean + session/task-cluster bootstrap CI).

Cost columns: TTFT / wall latency / token usage come from the proxy request
log (`benchmarks/proxy.py`), plus KV bytes accounting per arm
(raw prompt tokens vs gist tokens, ResKV-style b = m + r).
"""
from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
BROKEN_CALL_RE = re.compile(r"<tool_call>(?:(?!</tool_call>).)*$", re.DOTALL)


def parse_tool_calls(message: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Return (calls, has_broken_syntax) for one assistant message.

    Handles both the OpenAI shape ({id, function: {name, arguments}}) and
    tau2's flat shape ({name, arguments}).
    """
    calls: List[Dict[str, Any]] = []
    for item in message.get("tool_calls") or []:
        function = item.get("function") or {}
        name = function.get("name") or item.get("name")
        raw_args = function.get("arguments", item.get("arguments"))
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = None
        calls.append({"name": name, "arguments": args})
    text = message.get("content") or ""
    if isinstance(text, list):  # multimodal content blocks
        text = " ".join(
            block.get("text", "") for block in text if isinstance(block, dict)
        )
    for match in TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1))
            calls.append(
                {"name": obj.get("name"), "arguments": obj.get("arguments")}
            )
        except json.JSONDecodeError:
            calls.append({"name": None, "arguments": None})
    broken = bool(BROKEN_CALL_RE.search(text))
    return calls, broken


def _schema_violations(name: str, args: Any, tools: Sequence[Dict[str, Any]]) -> Optional[str]:
    tool = next(
        (t for t in tools if (t.get("function") or {}).get("name") == name), None
    )
    if not tools:
        # no advertised pool: we cannot judge legality — degrade to unknown
        # (None) instead of flagging every call as an unknown-tool violation
        return None
    if tool is None:
        return f"unknown tool name {name!r}"
    if args is None or not isinstance(args, dict):
        return "arguments are not a JSON object"
    schema = (tool.get("function") or {}).get("parameters") or {}
    for key in schema.get("required") or []:
        if key not in args:
            return f"missing required argument {key!r}"
    properties = schema.get("properties") or {}
    for key, value in args.items():
        expected = properties.get(key)
        if expected is None:
            continue
        json_type = expected.get("type")
        if json_type == "integer" and not isinstance(value, int) or (
            json_type == "integer" and isinstance(value, bool)
        ):
            return f"argument {key!r} must be integer"
        if json_type == "number" and not isinstance(value, (int, float)) or (
            json_type == "number" and isinstance(value, bool)
        ):
            return f"argument {key!r} must be number"
        if json_type == "string" and not isinstance(value, str):
            return f"argument {key!r} must be string"
        if json_type == "boolean" and not isinstance(value, bool):
            return f"argument {key!r} must be boolean"
        if json_type == "array" and not isinstance(value, list):
            return f"argument {key!r} must be array"
        if json_type == "object" and not isinstance(value, dict):
            return f"argument {key!r} must be object"
    return None


def protocol_columns_for_turn(
    message: Dict[str, Any], tools: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """One row of the protocol column: legality, call count, first violation."""
    calls, broken = parse_tool_calls(message)
    if calls and not tools and not broken:
        # tool calls exist but no pool was advertised — legality is not
        # computable, degrade to unknown rather than pass/fail. Broken
        # syntax remains decidable even without the advertised schema.
        return {"n_tool_calls": len(calls), "protocol_legal": None,
                "first_violation": "no_tool_pool"}
    violations = [
        _schema_violations(call["name"], call["arguments"], tools) for call in calls
    ]
    violations = [v for v in violations if v]
    if broken:
        violations.append("unterminated <tool_call> syntax")
    legal = not violations
    return {
        "n_tool_calls": len(calls),
        "protocol_legal": legal,
        "first_violation": violations[0] if violations else None,
    }


def aggregate(
    rows: Iterable[Dict[str, Any]],
    cluster_key: str = "task_id",
    bootstrap_reps: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Aggregate per-task rows: means + cluster bootstrap 95% CI.

    Recognized row fields (all optional): semantic_score, protocol_legal,
    ttft_sec, wall_sec, gist_tokens, original_tokens.
    """
    rows = list(rows)
    if not rows:
        return {"n": 0}
    clusters: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(str(row.get(cluster_key) or "_"), []).append(row)

    def _mean(values: Sequence[Any]) -> Optional[float]:
        picked = [float(v) for v in values if v is not None]
        return sum(picked) / len(picked) if picked else None

    def _cluster_ci(field: str) -> Tuple[Optional[float], Optional[float]]:
        keys = list(clusters)
        rng = random.Random(seed)
        means: List[float] = []
        for _ in range(bootstrap_reps):
            flat = [r for k in keys for r in clusters[rng.choice(keys)]]
            mean = _mean([r.get(field) for r in flat])
            if mean is not None:
                means.append(mean)
        if not means:
            return None, None
        means.sort()
        return means[int(0.025 * len(means))], means[min(len(means) - 1, int(0.975 * len(means)))]

    def _pct(field: str, q: float) -> Optional[float]:
        values = sorted(v for v in (r.get(field) for r in rows) if v is not None)
        if not values:
            return None
        return values[min(len(values) - 1, int(q * len(values)))]

    summary: Dict[str, Any] = {"n": len(rows), "n_clusters": len(clusters)}
    summary["semantic_score"] = _mean([r.get("semantic_score") for r in rows])
    summary["protocol_legal_rate"] = _mean(
        [float(r["protocol_legal"]) if r.get("protocol_legal") is not None else None
         for r in rows]
    )
    for field in ("ttft_sec", "wall_sec", "gist_tokens", "original_tokens"):
        summary[f"{field}_mean"] = _mean([r.get(field) for r in rows])
    summary["ttft_sec_p50"] = _pct("ttft_sec", 0.50)
    summary["ttft_sec_p95"] = _pct("ttft_sec", 0.95)
    summary["wall_sec_p95"] = _pct("wall_sec", 0.95)
    lo, hi = _cluster_ci("semantic_score")
    summary["semantic_score_ci95"] = [lo, hi]
    return summary


def effective_compression(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    """ResKV-style b = m + r bookkeeping: raw-equivalent tokens / (gist +
    any raw history retained).  None if the arm kept no compressed history."""
    original = sum(r.get("original_tokens") or 0 for r in rows)
    gist = sum(r.get("gist_tokens") or 0 for r in rows)
    if not original:
        return None
    return original / gist if gist else None
