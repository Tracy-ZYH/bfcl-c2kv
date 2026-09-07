"""SGLang c2kv fork backend (kvoffload-sglang-c2kv, branch c2kv-sglang-bfcl).

Wire protocol (verified live against 22fbf3146 on NPU, docs/sglang_migration.md):
* ``POST /v1/c2kv/extract`` — same request/response shape as hf_server
  (``text``/``compression_ratio``/``role`` → ``key_hash``/``gist_len``/
  ``original_seq_len``); failures are HTTP 200 with ``success=false``.
* ``POST /v1/c2kv/repair_extract`` — full-context form (server branch
  task/c2kv-serve-align, c2kv/c2kv_serving_semantics.md): ``messages`` +
  ``target_index`` + ``tools``; the server renders the prefix like a chat
  request and stores the target message's raw K/V (pre-RoPE) with its
  absolute positions in that rendering.  The legacy ``text``+``role`` form
  (standalone encoding, caller-supplied ``position_offset``) is kept only
  for A/B.
* message-level ``c2kv_key_hash`` (gist), ``c2kv_repair_key_hashes`` /
  ``c2kv_repair_only_key_hashes`` (raw KV) and ``c2kv_repair_placement``
  (in_place / append_keep_ledger / append_tail) on chat messages.
* every response carries ``metadata.sglang_runtime.c2kv_query_proj`` (which
  projection the server used for post-gist tokens) and ``c2kv_layout`` (the
  injections with their RoPE positions) for provenance / frame checks.
* constrained decoding via ``response_format={"type": "structural_tag"}``;
  tool schemas need xgrammar-safe repair (``_normalize_tool_schema``,
  moved verbatim from hf_server.py).
* per-request KV accounting in ``metadata.sglang_runtime``
  (kv_resident_tokens / kv_peak_resident_tokens / kv_pool_size).
* history-KV eviction baselines (arms.history_kv, upstream
  ``c2kv_eval.adapters.bfcl_history_kv_baselines``), two server paths:
  - ``repair_extract`` (default): ONE ``/v1/c2kv/repair_extract`` call with
    ``repair_mode="history_kv_<method>"`` + ``history_kv_method`` +
    ``history_kv_retention_ratio``; the server prefills the span, selects the
    surviving token slots (``selected_relative_indices``) and stores them as a
    single repair entry.  The chat request replaces the history text with the
    upstream carrier message (``c2kv_repair_only_key_hashes`` +
    ``c2kv_use_gist_projection: false``) and echoes the accounting in
    ``c2kv_kv_memory_hint``.
  - ``physical_eviction``: no extract call; the history stays raw text and
    ``c2kv_kv_memory_hint.history_kv_eviction`` asks the scheduler to compact
    the request's own KV slots after the history round.  The server resolves
    the token range itself from ``history_message_count``
    (serving_chat._resolve_history_kv_eviction_range), so the proxy never
    sends client-side token offsets.  Needs ``--disable-radix-cache`` and,
    with ``persistent_session``, ``--enable-streaming-session``.
  Both echo back in ``metadata.kv_memory_report`` (method / runtime status /
  freed slots / kept tokens), which normalize_response turns into
  ``history_kv_*`` cost columns.

Regime notes: ``--c2kv-query-proj base`` follows the original lowercase-qkv
algorithm; ``gist`` selects the later local fork's query rule. Choose from
checkpoint provenance (docs/c2kv_semantics.md). The proxy supports turn
packing and explicit repair placement independently of that choice.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import Backend, BackendError


def _normalize_tool_schema(value: Any, defs: Dict[str, Any] = None,
                           seen: frozenset = frozenset()) -> Any:
    """Schema repair before grammar compilation (moved verbatim from
    benchmarks/hf_server.py — the SGLang/xgrammar path needs it just as
    much as the old server did).

    Benchmark tool schemas use loose types ("type": "dict"/"any"/...,
    {"type": "list"} without items) and $ref/$defs indirection, neither
    of which xgrammar's JSON-schema converter accepts.  $refs are inlined
    (cycle-guarded), then types are mapped and unsupported keywords stripped.

    CONFOUND, not a separate channel: SGLang derives the grammar from
    ``request.tools``, which is also what the chat template renders into the
    prompt, so the repaired schema is what the MODEL SEES too.  The
    ``cd_full`` / ``cd_c2kv`` arms therefore differ from ``full`` / ``c2kv``
    by prompt AND grammar; the H1 comparison cannot separate the two (see
    arms.py and README "Constrained-decoding arms").
    """
    if isinstance(value, dict):
        if "$ref" in value and defs is not None:
            name = str(value["$ref"]).split("/")[-1]
            if name in defs and name not in seen:
                return _normalize_tool_schema(
                    defs[name], defs, seen | {name}
                )
            return {"type": "object"}
        out = {}
        for key, item in value.items():
            if key in ("$schema", "$id", "$defs", "$ref"):
                continue
            if key == "type":
                if isinstance(item, str):
                    item = {
                        "dict": "object",
                        "any": "object",
                        "int": "integer",
                        "float": "number",
                        "str": "string",
                        "bool": "boolean",
                        "list": "array",
                        "tuple": "array",
                        "None": "null",
                    }.get(item, item)
                elif isinstance(item, list):
                    mapped = []
                    for t in item:
                        t = {
                            "dict": "object", "any": "object", "int": "integer",
                            "float": "number", "str": "string", "bool": "boolean",
                            "list": "array", "tuple": "array", "None": "null",
                        }.get(t, t)
                        if t not in mapped:
                            mapped.append(t)
                    if not mapped:
                        continue
                    item = mapped
            out[key] = _normalize_tool_schema(item, defs, seen)
        return out
    if isinstance(value, list):
        return [_normalize_tool_schema(v, defs, seen) for v in value]
    return value


def _inline_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    defs = schema.get("$defs") or schema.get("definitions") or {}
    return _normalize_tool_schema(schema, defs)


class SglangBackend(Backend):
    name = "sglang"
    needs_repair_plan = True
    # the history-KV arms need the proxy's history/current split and its
    # per-conversation streaming-session id
    wants_request_context = True

    def __init__(self, post_json):
        self._post_json = post_json  # (path, payload, timeout) -> dict

    # ---- primitives ----
    def extract(self, text: str, role: str, ratio: int,
                tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload = {
            "text": text,
            "compression_ratio": ratio,
            "role": role,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            # the server renders tools into the chat template exactly as
            # the serving path does, so original_seq_len measures the TRUE
            # system block (fixes the position_offset short-by-tools bug)
            payload["tools"] = tools
        result = self._post_json("/v1/c2kv/extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "extract_failed",
                f"c2kv extract failed: {result.get('error') or json.dumps(result)[:500]}")
        return result

    def repair_extract(self, text: str, role: str, span_start: int,
                       span_end: Optional[int], position_offset: int,
                       source_doc_index: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "text": text,
            "role": role,
            "chat_template_kwargs": {"enable_thinking": False},
            "span_start": span_start,
            "position_offset": position_offset,
            "repair_mode": "d_corr",
            "source_doc_index": source_doc_index,
        }
        if span_end is not None:
            payload["span_end"] = span_end
        result = self._post_json("/v1/c2kv/repair_extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "repair_failed",
                f"c2kv repair_extract failed: {result.get('error') or json.dumps(result)[:500]}")
        return result

    def repair_extract_messages(self, messages: List[Dict[str, Any]],
                                target_index: int,
                                tools: Optional[List[Dict[str, Any]]],
                                source_doc_index: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "messages": [
                {"role": m.get("role") or "user", "content": m.get("content") or ""}
                for m in messages
            ],
            "target_index": int(target_index),
            "chat_template_kwargs": {"enable_thinking": False},
            "repair_mode": "d_corr",
            "source_doc_index": source_doc_index,
            # storage form (reconciled server, c2kv_serving_semantics.md §3):
            # pre-RoPE entries can take every placement; a rotated entry makes
            # append_tail fail with C2KV_APPEND_TAIL_REQUIRES_PRE_ROPE
            "raw_kv_position_mode": "pre_rope",
        }
        if tools:
            payload["tools"] = tools
        result = self._post_json("/v1/c2kv/repair_extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "repair_failed",
                f"c2kv repair_extract(messages) failed: "
                f"{result.get('error') or json.dumps(result)[:500]}")
        return result

    def history_kv_extract(self, history_text: str, system_text: str,
                           tools: Optional[List[Dict[str, Any]]],
                           spec: Dict[str, Any]) -> Dict[str, Any]:
        """One history-KV eviction extract (upstream ``repair_extract``
        backend, c2kv_eval.adapters.bfcl_history_kv_baselines
        ``_build_runtime_history_kv``).

        Form: the FULL-CONTEXT ``messages``/``target_index`` request, so the
        server renders system + tools + the history block exactly like a chat
        prompt, prefills it, and captures the history block's raw K/V at its
        true absolute positions (``raw_kv_position_mode="rotated"``, as
        upstream sends).  The budget travels as ``history_kv_retention_ratio``
        and is resolved against the span the SERVER measured
        (qwen3.generate_raw_repair_kv: ``ceil(requested_span_tokens * ratio)``)
        — the upstream client instead multiplied its own tokenizer's history
        length and sent absolute ``history_kv_target_tokens``.  Deviation
        recorded in README "History-KV eviction arms".
        """
        method = str(spec["method"])
        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": history_text})
        target_index = len(messages) - 1
        payload: Dict[str, Any] = {
            "messages": messages,
            "target_index": target_index,
            "chat_template_kwargs": {"enable_thinking": False},
            # the legacy placement rule maps any "history_kv_" repair mode to
            # in_place (scheduler._resolve_c2kv_repair_placement), i.e. the
            # compressed span stands in for the history unit it replaces
            "repair_mode": f"history_kv_{method}",
            # upstream stores the history entry post-RoPE at its own absolute
            # positions; it is never re-placed, so "rotated" is exact
            "raw_kv_position_mode": "rotated",
            "extract_source": "model_prefill",
            "source_doc_index": 0,
            "history_kv_method": method,
            "history_kv_recent_window": int(spec["recent_window"]),
            "history_kv_kernel_size": int(spec["kernel_size"]),
            "history_kv_pooling": str(spec["pooling"]),
            "history_kv_h2o_recent_fraction": float(spec["h2o_recent_fraction"]),
        }
        if spec.get("target_tokens") is not None:
            payload["history_kv_target_tokens"] = int(spec["target_tokens"])
        else:
            payload["history_kv_retention_ratio"] = float(spec["retention_ratio"])
        if tools:
            payload["tools"] = tools
        result = self._post_json("/v1/c2kv/repair_extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "history_kv_extract_failed",
                f"c2kv history-KV extract ({method}) failed: "
                f"{result.get('error') or json.dumps(result)[:500]}")
        # strict: a server that ignored the history_kv_* fields would return a
        # plain uncompressed repair entry, and the run would silently be a
        # full-history arm wearing a baseline's name (upstream
        # --strict-runtime-eviction)
        echoed = str(result.get("history_kv_method") or "")
        if echoed != method:
            raise BackendError(
                "history_kv_extract_failed",
                f"server did not apply history_kv_method={method!r} "
                f"(echoed {echoed!r}); refusing to report an uncompressed "
                "request as a history-KV baseline")
        return result

    def open_history_session(self, session_id: str, timeout: int = 600) -> str:
        """Open the streaming session the physical-eviction arms need.

        Same call the upstream client makes
        (``_open_persistent_history_session``); the server hands the id back as
        a bare JSON string and refuses a duplicate id."""
        result = self._post_json(
            "/open_session",
            {
                "capacity_of_str_len": 0,
                "session_id": session_id,
                "streaming": True,
                "timeout": float(timeout),
            },
            timeout,
        )
        if result != session_id:
            raise BackendError(
                "history_kv_session_failed",
                f"open_session did not return {session_id!r}: "
                f"{json.dumps(result)[:500]}")
        return session_id

    # ---- history-KV request shaping ----
    @staticmethod
    def _history_kv_carrier(method: str, key_hash: str) -> Dict[str, Any]:
        """The upstream carrier message, verbatim (bfcl_history_kv_baselines
        ``_build_runtime_history_kv``).

        ``c2kv_use_gist_projection: false`` is sent because upstream sends it;
        which projection the request ACTUALLY ran under is a server decision
        (the message value overrides the ``--c2kv-query-proj`` default) and
        must be read per row from ``c2kv_query_proj_effective`` /
        ``c2kv_query_proj_source``, never assumed from this field."""
        return {
            "role": "user",
            "content": f"[runtime {method} compressed history kv]",
            "c2kv_repair_only_key_hashes": [key_hash],
            "c2kv_use_gist_projection": False,
        }

    def _apply_history_kv(self, messages: List[Dict[str, Any]],
                          history: Dict[str, Any],
                          tools: Optional[List[Dict[str, Any]]]
                          ) -> tuple:
        """Return (messages, hint, session_id) for a history-KV arm."""
        spec = history["spec"]
        method = str(spec["method"])
        indices = [int(i) for i in history.get("history_out_indices") or []]
        session_id = history.get("session_id")
        if not indices or not history.get("history_text"):
            # first turn of a conversation: nothing completed to compress.
            # Upstream returns the current block unchanged and issues no
            # extract; no hint is sent, so such a row simply carries no
            # history_kv_* cost columns.
            return list(messages), None, session_id

        if str(spec["backend"]) == "physical_eviction":
            count = int(history["history_message_count"])
            target = int(spec["target_tokens"])
            eviction = {
                "method": method,
                # the server resolves the token range itself in its own frame
                "history_message_count": count,
                "target_tokens": target,
                "retention_ratio": spec.get("retention_ratio"),
                "history_kv_recent_window": int(spec["recent_window"]),
                "history_kv_kernel_size": int(spec["kernel_size"]),
                "history_kv_pooling": str(spec["pooling"]),
                "history_kv_h2o_recent_fraction": float(spec["h2o_recent_fraction"]),
                "persistent_session": bool(session_id),
            }
            hint: Dict[str, Any] = {
                # left at 0 on purpose: serving_chat._resolve_history_kv_
                # eviction_range overwrites it with the server's own exact
                # history token count
                "full_equivalent_history_tokens": 0,
                # Active counters are server-owned measurements.  The target
                # remains in history_kv_eviction until physical eviction
                # reports the number of slots it actually kept.
                "active_history_kv_tokens": 0,
                "active_full_raw_tokens": 0,
                "active_c2kv_gist_tokens": 0,
                "history_kv_method": method,
                "estimated": True,
                "history_kv_backend": "physical_eviction",
                "history_kv_eviction": eviction,
            }
            if session_id:
                hint["persistent_history_session"] = {"enabled": True}
            return list(messages), hint, session_id

        record = self.history_kv_extract(
            history["history_text"], history.get("system_text") or "", tools, spec)
        kept = int(record.get("selected_token_count")
                   or record.get("token_len") or 0)
        span = int(record.get("requested_span_tokens") or 0)
        keep = set(indices)
        out: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            if index == indices[0]:
                out.append(self._history_kv_carrier(method, record["key_hash"]))
            if index in keep:
                continue
            out.append(message)
        hint = {
            "full_equivalent_history_tokens": span,
            # The scheduler increments these counters after the repair block
            # is actually installed.  Seeding them here counts it twice.
            "active_history_kv_tokens": 0,
            "active_full_raw_tokens": 0,
            "active_c2kv_gist_tokens": 0,
            "active_raw_repair_tokens": 0,
            "history_kv_method": method,
            "estimated": False,
            # provenance beyond the upstream hint; the scheduler copies the
            # whole hint into kv_memory_report, so these come back on the
            # response and become request-log columns
            "history_kv_backend": "repair_extract",
            "history_kv_requested_span_tokens": span,
            "history_kv_selected_token_count": kept,
            "history_selection_metadata": record.get("history_selection_metadata"),
        }
        return out, hint, session_id

    # ---- KV reuse (CacheBlend) request shaping ----
    def kv_reuse_extract(self, history_docs: List[Dict[str, Any]],
                         system_text: str,
                         tools: Optional[List[Dict[str, Any]]],
                         spec: Dict[str, Any]) -> Dict[str, Any]:
        """One CacheBlend extract (reconciled server,
        c2kv_serving_semantics.md section 10).

        Form: the FULL-CONTEXT multi-message ``messages`` /
        ``target_index``..``target_end_index`` request -- system + tools +
        one message PER HISTORY DOC -- so the server renders the prologue and
        the docs exactly like a chat prompt, takes each doc's rendered message
        as one chunk (``chunking="doc"``; ``"grid"`` sends
        ``cacheblend_chunk_tokens`` instead and the server cuts a fixed token
        grid across the span), computes every chunk's KV standalone, and
        recomputes the ``recomp_ratio`` highest-deviation tokens in context.
        The entry is the WHOLE span (post-RoPE at its absolute positions);
        the chat request places it ``in_place``.
        """
        method = str(spec["method"])
        messages: List[Dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        target_index = len(messages)
        for doc in history_docs:
            messages.append({"role": str(doc.get("role") or "user"),
                             "content": str(doc.get("content") or "")})
        target_end_index = len(messages) - 1
        payload: Dict[str, Any] = {
            "messages": messages,
            "target_index": target_index,
            "target_end_index": target_end_index,
            "chat_template_kwargs": {"enable_thinking": False},
            # "cacheblend" resolves to in_place (scheduler
            # _C2KV_IN_PLACE_REPAIR_MODE_PREFIXES): the blended span stands in
            # for the history it was extracted from
            "repair_mode": method,
            # the entry is post-RoPE at its native positions; the server
            # forces this for kv_reuse anyway, sent explicitly for the log
            "raw_kv_position_mode": "rotated",
            "extract_source": "model_prefill",
            "source_doc_index": 0,
            "kv_reuse_method": method,
            "cacheblend_recomp_ratio": float(spec["recomp_ratio"]),
            "cacheblend_check_layer": int(spec["check_layer"]),
            "cacheblend_metric": str(spec["metric"]),
            "cacheblend_mask": str(spec["mask"]),
        }
        if str(spec["chunking"]) == "grid":
            # a grid ignores the per-message boundaries: target_end_index
            # still defines the span, the server cuts chunk_tokens across it
            payload["cacheblend_chunk_tokens"] = int(spec["chunk_tokens"])
        if tools:
            payload["tools"] = tools
        result = self._post_json("/v1/c2kv/repair_extract", payload, 600)
        if not result.get("success", True) or not result.get("key_hash"):
            raise BackendError(
                "kv_reuse_extract_failed",
                f"c2kv KV-reuse extract ({method}) failed: "
                f"{result.get('error') or json.dumps(result)[:500]}")
        # strict: a server that ignored kv_reuse_method would return a plain
        # full raw-history entry, and the run would silently be a full-history
        # arm wearing CacheBlend's name (same rule as history_kv_extract)
        echoed = str(result.get("kv_reuse_method") or "")
        if echoed != method:
            raise BackendError(
                "kv_reuse_extract_failed",
                f"server did not apply kv_reuse_method={method!r} "
                f"(echoed {echoed!r}); refusing to report a full-history "
                "request as a KV-reuse baseline")
        acct = result.get("cacheblend")
        if not isinstance(acct, dict):
            raise BackendError(
                "kv_reuse_extract_failed",
                f"server echoed no cacheblend accounting for {method!r}")
        return result

    @staticmethod
    def _kv_reuse_carrier(method: str, key_hash: str) -> Dict[str, Any]:
        """Repair-only carrier for the blended history span; ``in_place`` so
        the query continues at the span's absolute end (the history-KV
        carrier relies on the legacy prefix rule for the same effect).
        CacheBlend reuses base-model KV, so its query must also use the base
        projections even when this endpoint serves a gist-query checkpoint.
        """
        return {
            "role": "user",
            "content": f"[{method} reused history kv]",
            "c2kv_repair_only_key_hashes": [key_hash],
            "c2kv_repair_placement": "in_place",
            "c2kv_use_gist_projection": False,
        }

    def _apply_kv_reuse(self, messages: List[Dict[str, Any]],
                        reuse: Dict[str, Any],
                        tools: Optional[List[Dict[str, Any]]]) -> tuple:
        """Return (messages, hint) for a KV-reuse arm."""
        spec = reuse["spec"]
        method = str(spec["method"])
        indices = [int(i) for i in reuse.get("history_out_indices") or []]
        docs = reuse.get("history_docs") or []
        if not indices or not docs:
            # first turn: nothing completed to reuse; no extract, no hint, so
            # such a row carries no cacheblend_* cost columns
            return list(messages), None
        record = self.kv_reuse_extract(
            docs, reuse.get("system_text") or "", tools, spec)
        acct = record.get("cacheblend") or {}
        span = int(record.get("requested_span_tokens")
                   or record.get("token_len") or 0)
        recomputed = acct.get("recomputed_tokens")
        keep = set(indices)
        out: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            if index == indices[0]:
                out.append(self._kv_reuse_carrier(method, record["key_hash"]))
            if index in keep:
                continue
            out.append(message)
        hint = {
            # CacheBlend keeps the WHOLE span resident: the saving is compute
            "full_equivalent_history_tokens": span,
            # Resident counters are filled by the scheduler after injection.
            "active_history_kv_tokens": 0,
            "active_full_raw_tokens": 0,
            "active_c2kv_gist_tokens": 0,
            "active_raw_repair_tokens": 0,
            "active_recomputed_raw_tokens": int(recomputed or 0),
            "estimated": False,
            # provenance: the scheduler copies the whole hint into
            # kv_memory_report, so these come back on the response
            "kv_reuse_method": method,
            "kv_reuse_backend": "repair_extract",
            "cacheblend_span_tokens": span,
            "cacheblend_recomputed_tokens": recomputed,
            "cacheblend_effective_recomp_ratio": acct.get("effective_recomp_ratio"),
            "cacheblend_recomp_ratio": float(spec["recomp_ratio"]),
            "cacheblend_check_layer": int(spec["check_layer"]),
            "cacheblend_metric": str(spec["metric"]),
            "cacheblend_mask": str(spec["mask"]),
            "cacheblend_chunking": str(spec["chunking"]),
            "cacheblend_chunk_count": acct.get("chunk_count"),
            "cacheblend_deviation_max": acct.get("deviation_max"),
            "cacheblend_deviation_selected_min": acct.get("deviation_selected_min"),
            "cacheblend_cache_hit": bool(acct.get("cache_hit", False)),
        }
        return out, hint

    # ---- chat shaping ----
    def prepare_chat(self, payload: Dict[str, Any], arm,
                     repair_plan: Optional[Dict[str, Any]],
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = dict(payload)
        out.pop("c2kv_repair", None)  # request-level repair is hf_server-only
        if getattr(arm, "history_kv", None) or getattr(arm, "kv_reuse", None):
            # The raw-KV baselines have a fixed query regime. Override both
            # the server default and a profile supplied by the proxy.
            out["c2kv_use_gist_projection"] = False
        # Same chat_template_kwargs as the two FRAME-DEFINING endpoints
        # (extract() and repair_extract_messages() above both send
        # enable_thinking=False).  Every position this bench measures --
        # original_seq_len, position_start, rendered_prefix_len -- was computed
        # with thinking off; the served prompt must be rendered the same way or
        # the two renderings differ by whatever the checkpoint's template does
        # with enable_thinking.  setdefault, so an explicit client value still
        # wins (BFCL's OpenAI client sends none) -- but such a request is then
        # outside the measured frame.
        raw_kwargs = out.get("chat_template_kwargs")
        template_kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        template_kwargs.setdefault("enable_thinking", False)
        out["chat_template_kwargs"] = template_kwargs
        messages = list(out.get("messages") or [])
        if getattr(arm, "history_kv", None):
            history = (context or {}).get("history_kv")
            if not history:
                raise BackendError(
                    "history_kv_failed",
                    f"arm {arm.name!r} is a history-KV arm but the proxy sent "
                    "no history context")
            messages, hint, session_id = self._apply_history_kv(
                messages, history, out.get("tools"))
            if hint is not None:
                out["c2kv_kv_memory_hint"] = hint
            if session_id:
                params = dict(out.get("session_params") or {})
                params["id"] = session_id
                out["session_params"] = params
        if getattr(arm, "kv_reuse", None):
            reuse = (context or {}).get("kv_reuse")
            if not reuse:
                raise BackendError(
                    "kv_reuse_failed",
                    f"arm {arm.name!r} is a KV-reuse arm but the proxy sent "
                    "no history context")
            messages, hint = self._apply_kv_reuse(messages, reuse, out.get("tools"))
            if hint is not None:
                out["c2kv_kv_memory_hint"] = hint
        if arm.constrain_tools:
            # structural_tag constrained decoding; the grammar input needs
            # xgrammar-safe schemas.  SGLang compiles the grammar from
            # request.tools, the SAME field the chat template renders, so the
            # repaired schema also reaches the prompt (known cd_* confound,
            # see the module docstring and arms.py)
            out["response_format"] = {"type": "structural_tag"}
            tools = out.get("tools")
            if tools:
                out["tools"] = [
                    {**tool, "function": {
                        **tool.get("function", {}),
                        "parameters": _inline_refs(
                            tool.get("function", {}).get("parameters")
                            or {"type": "object"}),
                    }} if isinstance(tool, dict) else tool
                    for tool in tools
                ]
        if repair_plan and (arm.repair or getattr(arm, "gold_recovery", None)):
            placement = str(repair_plan.get("placement") or "append_keep_ledger")
            key_hash = repair_plan["repair_key_hash"]
            index = repair_plan.get("target_out_index", repair_plan.get("message_index"))
            if index is None or not 0 <= index < len(messages):
                raise BackendError(
                    "repair_failed",
                    f"repair plan target index {index!r} out of range")
            if placement == "in_place":
                # the raw span REPLACES the target doc's gist: the message
                # becomes repair-only (no gist), the server re-anchors the
                # query to the span's absolute end
                message = dict(messages[index])
                if not message.get("c2kv_key_hash"):
                    raise BackendError(
                        "repair_failed",
                        f"in_place repair target {index} has no c2kv_key_hash")
                message.pop("c2kv_key_hash", None)
                message["c2kv_repair_only_key_hashes"] = [key_hash]
                message["c2kv_repair_placement"] = placement
                messages[index] = message
            elif placement in ("append_keep_ledger", "append_tail"):
                # the raw span is appended to the end of history (after all
                # gists and the raw hybrid tail, before the current turn) as a
                # repair-only message; the gist of the target doc stays
                insert_at = repair_plan.get("current_start_out_index")
                if insert_at is None or not 0 <= insert_at <= len(messages):
                    insert_at = len(messages)
                messages.insert(int(insert_at), {
                    "role": "user", "content": "",
                    "c2kv_repair_only_key_hashes": [key_hash],
                    "c2kv_repair_placement": placement,
                })
            else:
                raise BackendError("repair_failed", f"unknown placement {placement!r}")
        out["messages"] = messages
        return out

    # ---- response normalization ----
    @staticmethod
    def _history_kv_cost(data: Dict[str, Any]) -> Dict[str, Any]:
        """The server's history-KV echo, flattened into cost columns.

        ``metadata.kv_memory_report`` is the scheduler's per-request layout
        report (scheduler._init_c2kv_kv_memory_report + _apply_history_kv_
        eviction): it carries back the hint the proxy sent plus, on the
        physical path, the MEASURED eviction result.  Every column is the
        server's number, never a proxy estimate."""
        report = ((data.get("metadata") or {}).get("kv_memory_report")) or {}
        if not isinstance(report, dict) or not report:
            return {}
        physical = report.get("history_kv_physical_eviction")
        physical = physical if isinstance(physical, dict) else {}
        columns = {
            "history_kv_method": report.get("history_kv_method") or physical.get("method"),
            "history_kv_backend": report.get("history_kv_backend"),
            "history_kv_runtime_status": report.get("history_kv_runtime_status")
            or physical.get("runtime_status"),
            "history_kv_full_equivalent_tokens": report.get("full_equivalent_history_tokens"),
            "history_kv_full_equivalent_tokens_source": report.get(
                "full_equivalent_history_tokens_source"
            ),
            "history_kv_active_tokens": report.get("active_history_kv_tokens"),
            "history_kv_active_tokens_source": report.get(
                "active_history_kv_tokens_source"
            ),
            # repair_extract path (echoed hint)
            "history_kv_span_tokens": report.get("history_kv_requested_span_tokens"),
            "history_kv_selected_tokens": report.get("history_kv_selected_token_count"),
            "history_kv_selection": report.get("history_selection_metadata"),
            # physical path (measured by PhysicalHistoryKVEvictor)
            "history_kv_eviction_ok": physical.get("success"),
            "history_kv_eviction_error": physical.get("error") or None,
            "history_kv_kept_tokens": physical.get("kept_history_tokens"),
            "history_kv_history_tokens": physical.get("history_tokens"),
            "history_kv_freed_slots": physical.get(
                "freed_physical_slots", report.get("physical_slots_freed")),
            "history_kv_freed_bytes": physical.get("freed_kv_bytes"),
            "history_kv_selection_reason": report.get("selection_reason"),
        }
        return {k: v for k, v in columns.items() if v is not None}

    @staticmethod
    def _kv_reuse_cost(data: Dict[str, Any]) -> Dict[str, Any]:
        """The server's KV-reuse (CacheBlend) echo, flattened into cost
        columns.  ``metadata.kv_memory_report`` carries the extract provenance
        from the proxy plus scheduler-owned resident counters.
        ``cacheblend_recomputed_tokens`` /
        ``cacheblend_span_tokens`` are the compute-saving columns; resident
        KV is the whole span by construction."""
        report = ((data.get("metadata") or {}).get("kv_memory_report")) or {}
        if not isinstance(report, dict) or not report.get("kv_reuse_method"):
            return {}
        keys = (
            "kv_reuse_method", "kv_reuse_backend",
            "cacheblend_span_tokens", "cacheblend_recomputed_tokens",
            "cacheblend_effective_recomp_ratio", "cacheblend_recomp_ratio",
            "cacheblend_check_layer", "cacheblend_metric", "cacheblend_mask",
            "cacheblend_chunking", "cacheblend_chunk_count",
            "cacheblend_deviation_max", "cacheblend_deviation_selected_min",
            "cacheblend_cache_hit",
        )
        columns = {k: report.get(k) for k in keys}
        columns["kv_reuse_active_tokens"] = report.get("active_history_kv_tokens")
        columns["kv_reuse_active_tokens_source"] = report.get(
            "active_history_kv_tokens_source"
        )
        columns["kv_reuse_recomputed_tokens"] = report.get("active_recomputed_raw_tokens")
        return {k: v for k, v in columns.items() if v is not None}

    def normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Response -> (content, tool_calls, finish_reason, usage, cost).

        Injection-time failures are the ones that must NOT pass as answers:
        the reconciled server aborts the request and reports WHY in
        ``metadata.sglang_runtime.c2kv_injection_error`` plus
        ``metadata.finish_message`` (the finish_reason message).  Both are
        read here, because a plain ``finish_reason == "abort"`` body used to
        be classified ``finish_abort`` with a truncated JSON dump as the only
        evidence — and an abort caused by a pool eviction was therefore never
        retried.  A ``C2KV_CACHE_MISS`` in either field is raised as
        ``cache_miss`` so proxy.CacheMiss's re-extract-and-retry path fires.

        Admission-time misses do not arrive here at all: the server answers
        them with HTTP 400 and ``proxy._post_json`` raises CacheMiss directly.
        The ``data["error"]`` classifier below is still reachable — the top
        guard only raises for an error body WITHOUT choices — so it stays.
        """
        if data.get("object") == "error" or (data.get("error") and not data.get("choices")):
            raise BackendError("upstream", f"sglang error body: {data.get('error')}")
        choice = (data.get("choices") or [{}])[0] or {}
        finish = choice.get("finish_reason")
        metadata = data.get("metadata") or {}
        runtime = (metadata.get("sglang_runtime") or {}) if isinstance(metadata, dict) else {}
        injection_error = str(runtime.get("c2kv_injection_error") or "").strip()
        finish_message = str(
            (metadata.get("finish_message") if isinstance(metadata, dict) else "") or "").strip()
        if finish == "abort":
            detail = " | ".join(t for t in (injection_error, finish_message) if t)
            if "C2KV_CACHE_MISS" in injection_error or "C2KV_CACHE_MISS" in finish_message:
                raise BackendError("cache_miss", detail)
            raise BackendError("finish_abort", detail or json.dumps(data)[:500])
        message = choice.get("message") or {}
        error_text = str(data.get("error") or "")
        if "C2KV_CACHE_MISS" in error_text or "C2KV cache miss" in error_text:
            raise BackendError("cache_miss", error_text)
        cost = {k: runtime[k] for k in (
            "kv_resident_tokens", "kv_peak_resident_tokens", "kv_pool_size",
            "bytes_per_kv_token", "physical_main_kv_bytes", "physical_c2kv_pool_bytes",
            "total_gpu_kv_bytes", "peak_total_gpu_kv_bytes", "page_size_tokens",
            "physical_main_kv_slots", "physical_c2kv_pool_slots",
            # c2kv_query_proj = the server FLAG (one value per run; reqlog's
            # mixed-mode check keys on it); _effective / _source / _decode_
            # verified are the per-request provenance of the reconciled server
            "c2kv_query_proj", "c2kv_query_proj_effective",
            "c2kv_query_proj_source", "c2kv_query_proj_decode_verified",
            "c2kv_tools_dump",
            "c2kv_gist_seen", "c2kv_position_correction", "c2kv_layout",
            # injection provenance on a request the server DID serve (an
            # injection error that aborted the request raised above): the row
            # is not a clean measurement and must say so in its own column
            "c2kv_injection_error")
            if k in runtime}
        if isinstance(metadata, dict) and metadata.get("finish_message") is not None:
            cost["finish_message"] = metadata["finish_message"]
        cost.update(self._history_kv_cost(data))
        cost.update(self._kv_reuse_cost(data))
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": finish,
            "usage": data.get("usage"),
            "cost": cost,
        }
