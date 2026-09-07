"""Run-level summary of one proxy request log (``proxy.py --request-log``).

Every arm gets the same block in ``run.py``'s summary — until now only the
text arms read the log (compressor cost), so the regime facts that decide
whether a compressed number is even readable were left in the jsonl:

* ``n_docs`` / ``dropped_docs`` — turn packing keeps doc 0 + the last
  ``max_doc_num``-1 docs and DROPS the rest (training regime); on long
  trajectories (8-objective QA, AppWorld) this is the first place history
  actually disappears, so the share of requests that dropped anything and
  the mean number dropped are reported per run;
* ``c2kv_query_proj`` — the server FLAG (one value per run); a run mixing
  values is not one regime (docs/c2kv_semantics.md §4).  ``mixed_query_proj``
  keys on that flag.  ``c2kv_query_proj_effective`` is the PER-REQUEST
  provenance the reconciled server reports (which projection the request
  actually ran under, which the flag does not determine — a repair-only
  request flips it), counted per run as ``effective_query_proj_counts``
  with ``"absent"`` for rows that carry no value at all;
* request outcome mix (ok / error kinds), wall p50/p90, gist vs original
  token ledger, kv_resident p50 (sglang).

Pure stdlib.  ``join_by_conversation`` below does the per-task join: the
log is keyed by conversation id, not task id, so a benchmark only gets cost
columns when its own artefacts let us rebuild the message prefixes
``proxy.conversation_id`` keys on (see README "cost columns").
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def read_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _mean(values: Iterable[Any]) -> Optional[float]:
    picked = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(picked) / len(picked) if picked else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = Counter(str(r.get("error_kind") or r.get("status")) for r in rows
                     if r.get("status") != "ok")
    docs = [r.get("n_docs") for r in ok if isinstance(r.get("n_docs"), int)]
    dropped = [r.get("dropped_docs") for r in ok if isinstance(r.get("dropped_docs"), int)]
    walls = [float(r["wall_sec"]) for r in ok
             if isinstance(r.get("wall_sec"), (int, float))]
    kv = [r["kv_resident_tokens"] for r in ok if isinstance(r.get("kv_resident_tokens"), int)]
    gist = sum(int(r.get("gist_tokens") or 0) for r in ok)
    original = sum(int(r.get("original_tokens") or 0) for r in ok)
    modes = sorted({str(r["c2kv_query_proj"]) for r in ok if r.get("c2kv_query_proj")})
    # per-request provenance, NOT the flag: "gist"/"base"/... per row, and
    # "absent" for a row the server reported nothing on (an old server, or a
    # request that never reached the injection path).  mixed_query_proj below
    # stays keyed on the FLAG so the two questions never collapse into one.
    effective = Counter(
        str(r["c2kv_query_proj_effective"]) if r.get("c2kv_query_proj_effective")
        else "absent"
        for r in ok)
    packings = sorted({str(r["doc_packing"]) for r in ok if r.get("doc_packing")})
    summary: Dict[str, Any] = {
        "n_requests": len(rows),
        "n_ok": len(ok),
        "n_error": len(rows) - len(ok),
        "error_kinds": dict(errors),
        "n_conversations": len({r.get("conv_id") for r in ok if r.get("conv_id")}),
        "n_docs_mean": _mean(docs),
        "n_docs_max": max(docs) if docs else None,
        "compressed_requests": sum(1 for d in docs if d > 0),
        "dropped_requests": sum(1 for d in dropped if d > 0),
        "dropped_share": (sum(1 for d in dropped if d > 0) / len(dropped)) if dropped else None,
        "dropped_docs_mean": _mean(dropped),
        "dropped_docs_max": max(dropped) if dropped else None,
        "wall_p50": _pct(walls, 0.50),
        "wall_p90": _pct(walls, 0.90),
        "gist_tokens_total": gist,
        "original_tokens_total": original,
        "logical_over_gist": round(original / gist, 3) if gist else None,
        "kv_resident_p50": _pct([float(v) for v in kv], 0.50),
        "c2kv_query_proj": modes,
        "effective_query_proj_counts": dict(sorted(effective.items())),
        "doc_packing": packings,
    }
    summary["mixed_query_proj"] = len(modes) > 1
    summary["wall_sec_total"] = sum(walls)
    summary["server_total_gpu_kv_bytes_max"] = max(
        (r["total_gpu_kv_bytes"] for r in ok if isinstance(r.get("total_gpu_kv_bytes"), int)),
        default=None)
    summary["server_memory_scope"] = "process allocator snapshot including shared caches and concurrent requests"
    recovery = [r["gold_recovery"] for r in ok if isinstance(r.get("gold_recovery"), dict)]
    summary["gold_recovery_status_counts"] = dict(Counter(r.get("status") for r in recovery))
    summary["recovery_extract_sec_total"] = sum(float(r.get("recovery_extract_sec") or 0) for r in recovery)
    summary["recovery_events"] = len({tuple(r["event_id"]) for r in recovery
                                     if r.get("status") == "appended" and r.get("event_id")})
    tensor_rows = [r["history_tensor_accounting"] for r in ok
                   if isinstance(r.get("history_tensor_accounting"), dict)]
    if tensor_rows:
        summary["history_tensor_accounting"] = {
            "n_requests": len(tensor_rows),
            "before_recovery_bytes_mean": _mean(r.get("before_recovery_bytes") for r in tensor_rows),
            "after_recovery_bytes_mean": _mean(r.get("after_recovery_bytes") for r in tensor_rows),
            "history_ratio_before_mean": _mean(r.get("history_ratio_before") for r in tensor_rows),
            "history_ratio_after_mean": _mean(r.get("history_ratio_after") for r in tensor_rows),
            "scope": tensor_rows[0]["scope"],
        }
    by_task = {}
    for row in rows:
        context = row.get("eval_context") or {}
        task_id = context.get("task_id")
        if task_id is not None:
            by_task.setdefault(str(task_id), []).append(row)
    if by_task:
        summary["task_costs"] = {
            key: {"n_requests": len(values),
                  "n_errors": sum(r.get("status") != "ok" for r in values),
                  "wall_sec_total": sum(float(r.get("wall_sec") or 0) for r in values),
                  "prompt_tokens": sum(int((r.get("usage") or {}).get("prompt_tokens") or 0) for r in values),
                  "completion_tokens": sum(int((r.get("usage") or {}).get("completion_tokens") or 0) for r in values)}
            for key, values in by_task.items()
        }
    return summary


def summarize_file(path: Path) -> Dict[str, Any]:
    return summarize(read_rows(path))


# ---------------------------------------------------------------- cost join

# what the join writes onto a task row (metrics.aggregate already means
# wall_sec / gist_tokens / original_tokens)
SUM_FIELDS = ("wall_sec", "gist_tokens", "original_tokens", "dropped_docs")
COST_FIELDS = SUM_FIELDS + ("n_docs_max", "n_requests")


def _num(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def join_by_conversation(rows: List[Dict[str, Any]],
                         log_rows: List[Dict[str, Any]],
                         key_of_row) -> Dict[str, Any]:
    """Fill the per-task cost columns from the proxy request log.

    The log is keyed by ``proxy.conversation_id`` (system head + the first
    two non-system messages of the request AS THE CLIENT SENT IT), never by
    task id, so the join only exists for a harness whose own artefacts let
    us rebuild those message prefixes exactly.  ``key_of_row(row)`` returns
    the conversation id(s) of one task row — a string, an iterable of
    strings, or ``None`` when the task cannot be keyed.  A conversation id
    normally shifts ONCE per task (the first request carries one non-system
    message, every later one carries two), so a caller usually returns both.

    ``rows`` are mutated in place: a task with at least one matched request
    gets ``wall_sec`` / ``gist_tokens`` / ``original_tokens`` /
    ``dropped_docs`` (sums), ``n_docs_max`` and ``n_requests``; a task with
    none gets ``n_requests = 0`` and no cost field at all (``metrics.
    aggregate`` skips missing fields rather than averaging in a zero).

    ``metrics.aggregate`` means only wall_sec / gist_tokens / original_tokens
    and no caller persists the rows, so pass them through ``cost_summary``
    afterwards or the other three fields reach no artefact.

    Returned report — the caller decides from it whether the numbers are
    quotable, because a WRONG key produces no matches, never wrong sums:

    ``n_rows`` / ``n_keyed`` / ``n_joined``  task rows, keyed, matched;
    ``n_log_ok`` / ``n_log_joined``          ok log rows, attributed;
    ``ambiguous_conv_ids``                   ids claimed by >1 task (never
                                             attributed to any of them);
    ``full_coverage``                        every ok log row landed on a task.
    """
    by_conv: Dict[str, List[Dict[str, Any]]] = {}
    for row in log_rows:
        conv = row.get("conv_id")
        if row.get("status") == "ok" and conv:
            by_conv.setdefault(str(conv), []).append(row)
    n_log_ok = sum(len(v) for v in by_conv.values())

    keys: List[List[str]] = []
    claims: Dict[str, Set[int]] = {}
    for index, row in enumerate(rows):
        raw = key_of_row(row)
        if raw is None:
            ids: List[str] = []
        elif isinstance(raw, str):
            ids = [raw]
        else:
            ids = [str(k) for k in raw if k]
        keys.append(ids)
        for conv in ids:
            claims.setdefault(conv, set()).add(index)
    ambiguous = sorted(c for c, owners in claims.items() if len(owners) > 1)

    joined_rows: Set[int] = set()
    joined_log = 0
    for index, row in enumerate(rows):
        matched: List[Dict[str, Any]] = []
        for conv in dict.fromkeys(keys[index]):
            if conv in ambiguous:
                continue
            matched.extend(by_conv.get(conv, []))
        row["n_requests"] = len(matched)
        if not matched:
            continue
        joined_rows.add(index)
        joined_log += len(matched)
        for field in SUM_FIELDS:
            values = [_num(m.get(field)) for m in matched]
            values = [v for v in values if v is not None]
            if values:
                row[field] = sum(values)
        docs = [m.get("n_docs") for m in matched if isinstance(m.get("n_docs"), int)]
        if docs:
            row["n_docs_max"] = max(docs)
    return {
        "n_rows": len(rows),
        "n_keyed": sum(1 for k in keys if k),
        "n_joined": len(joined_rows),
        "n_log_ok": n_log_ok,
        "n_log_joined": joined_log,
        "ambiguous_conv_ids": ambiguous,
        "full_coverage": bool(n_log_ok) and joined_log == n_log_ok and not ambiguous,
    }


def not_joinable(rows: List[Dict[str, Any]], reason: str) -> Dict[str, Any]:
    """A report for a join that never RAN (no request log, harness whose
    artefacts cannot rebuild the message prefixes): same shape as
    ``join_by_conversation`` returns, with the reason carried verbatim so
    ``cost_join_status`` states it instead of inventing a count."""
    return {
        "n_rows": len(rows),
        "n_keyed": 0,
        "n_joined": 0,
        "n_log_ok": 0,
        "n_log_joined": 0,
        "ambiguous_conv_ids": [],
        "full_coverage": False,
        "status": f"not joinable: {reason}",
    }


def _whole(value: Optional[float]) -> Any:
    """Doc/request counts are integers; the join sums them as floats."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def cost_summary(rows: List[Dict[str, Any]],
                 report: Dict[str, Any]) -> Dict[str, Any]:
    """The summary block for one cost join — call it on the rows AFTER
    ``join_by_conversation`` mutated them, and ``summary.update()`` it.

    ``metrics.aggregate`` means only wall_sec / gist_tokens / original_tokens,
    and nothing persists the per-task rows, so without this the other three
    joined fields would be computed and thrown away.  It also carries the
    NUMERIC denominator of those means: a partial join means the cost means
    cover ``n_cost_joined`` tasks while ``semantic_score`` covers ``n``, and
    a consumer must not have to parse the ``cost_join`` prose to see it.

    ``n_docs_max`` / ``dropped_docs_total`` are ``None`` — never 0 — when no
    task carries the field, because a 0 would read as a measured zero.
    ``n_cost_requests`` is 0 when the join RAN and attributed nothing, and
    ``None`` when it never ran (``not_joinable``).
    """
    counts = [r.get("n_requests") for r in rows
              if isinstance(r.get("n_requests"), int)
              and not isinstance(r.get("n_requests"), bool)]
    docs = [_num(r.get("n_docs_max")) for r in rows]
    docs = [d for d in docs if d is not None]
    dropped = [_num(r.get("dropped_docs")) for r in rows]
    dropped = [d for d in dropped if d is not None]
    return {
        "cost_join": cost_join_status(report),
        "n_cost_joined": int(report.get("n_joined") or 0),
        "n_cost_requests": sum(counts) if counts else None,
        "n_docs_max": _whole(max(docs)) if docs else None,
        "dropped_docs_total": _whole(sum(dropped)) if dropped else None,
    }


def cost_join_status(report: Dict[str, Any]) -> str:
    """The ``cost_join`` line of a summary: what the join actually achieved."""
    if report.get("status"):
        return str(report["status"])  # a join that never ran (not_joinable)
    if not report.get("n_joined"):
        return ("not joinable: no request-log row matched a task conversation id "
                f"({report.get('n_log_ok', 0)} ok rows, "
                f"{report.get('n_keyed', 0)}/{report.get('n_rows', 0)} tasks keyed)")
    base = (f"{report['n_joined']}/{report['n_rows']} tasks, "
            f"{report['n_log_joined']}/{report['n_log_ok']} logged requests")
    if report.get("full_coverage"):
        return f"joined: {base}"
    extra = (f"; {len(report['ambiguous_conv_ids'])} ambiguous conv_id(s)"
             if report.get("ambiguous_conv_ids") else "")
    return f"partial: {base}{extra} — per-task cost columns are incomplete"
