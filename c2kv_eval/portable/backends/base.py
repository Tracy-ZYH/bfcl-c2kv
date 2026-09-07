"""Backend abstraction for the arm proxy.

The proxy owns arm semantics (compression bookkeeping, oracle recover,
reference recording); a backend owns ONLY the wire protocol of the serving
stack it fronts.  Two implementations:

* ``hfserver`` — the in-repo Flask hf_server (request-level
  ``c2kv_repair``, cost block in the response ``c2kv`` field).
* ``sglang``   — the kvoffload-sglang-c2kv fork (message-level
  ``c2kv_repair_key_hashes`` via ``/v1/c2kv/repair_extract``, cost block
  in ``metadata.sglang_runtime``, structural_tag constrained decoding).

Adding a backend must never require benchmark-side changes; adding an arm
stays arms.py-only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class BackendError(RuntimeError):
    """A backend-level failure that must surface as an infrastructure
    error, never as a model answer (B1/B2 lesson: HTTP-200-with-error-body
    and abort finishes are failures too)."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"[{kind}] {detail[:2000]}")
        self.kind = kind  # extract_failed | repair_failed | finish_abort | cache_miss | upstream
        self.detail = detail


class Backend:
    name: str = "base"
    # True when the backend needs the proxy-resolved repair plan
    # (sglang: repair_extract + message-level hashes).  hf_server resolves
    # the policy server-side, so planning (and its system extract) is
    # skipped entirely there — no needless c2kv-pool pollution.
    needs_repair_plan: bool = False
    # True when prepare_chat needs the per-request proxy context (the
    # history/current split, and the per-conversation streaming-session id
    # for the history-KV eviction arms).  The proxy passes ``context=`` ONLY
    # to backends that declare this, so a backend that never asked for it
    # keeps its 3-argument prepare_chat signature.
    wants_request_context: bool = False

    # ---- KV primitives ----
    def extract(self, text: str, role: str, ratio: int,
                tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Compress one message; returns {key_hash, gist_len, original_seq_len}.

        ``tools`` (when given) is rendered into the chat template the same
        way the serving path renders the request's tools — required for
        measuring the TRUE system-block length (Qwen templates put the
        tool schemas in the system block).  Backends that cannot render
        tools must raise, not silently return a short length."""
        raise NotImplementedError

    def repair_extract(self, text: str, role: str, span_start: int,
                       span_end: Optional[int], position_offset: int,
                       source_doc_index: int) -> Dict[str, Any]:
        """LEGACY standalone form: the message text alone is encoded (no
        context) and rotated to ``position_offset``.  Lower fidelity than
        ``repair_extract_messages`` (docs/c2kv_semantics.md "Raw KV"); kept
        for A/B only."""
        raise NotImplementedError

    def repair_extract_messages(self, messages: List[Dict[str, Any]],
                                target_index: int,
                                tools: Optional[List[Dict[str, Any]]],
                                source_doc_index: int) -> Dict[str, Any]:
        """FULL-CONTEXT form: the backend renders ``messages[:target_index+1]``
        like a chat request (with ``tools``) and stores the raw KV of message
        ``target_index`` computed inside that context.  Returns the entry
        record: key_hash, token_len, position_start, position_end,
        already_rotated."""
        raise NotImplementedError

    def history_kv_extract(self, history_text: str, system_text: str,
                           tools: Optional[List[Dict[str, Any]]],
                           spec: Dict[str, Any]) -> Dict[str, Any]:
        """HISTORY-KV EVICTION form: compress one completed-history span by
        token eviction instead of gist compression (StreamingLLM / H2O /
        SnapKV / PyramidKV).  The backend prefills the span in its serving
        frame, the SERVER selects the surviving token slots under ``spec``
        (arms.history_kv_spec) and stores them as one repair entry.  Returns
        the entry record: key_hash, requested_span_tokens,
        selected_token_count, selected_relative_indices, history_kv_method."""
        raise NotImplementedError

    def open_history_session(self, session_id: str, timeout: int = 600) -> str:
        """Open a streaming session that the physical-eviction history-KV
        arms need (the compacted KV must survive across turns).  Returns the
        session id the server acknowledged."""
        raise NotImplementedError

    def kv_reuse_extract(self, history_docs: List[Dict[str, Any]],
                         system_text: str,
                         tools: Optional[List[Dict[str, Any]]],
                         spec: Dict[str, Any]) -> Dict[str, Any]:
        """KV REUSE form (CacheBlend): the completed history as per-chunk
        out-of-context KV with the ``spec["recomp_ratio"]`` highest-deviation
        tokens recomputed in context by the SERVER (arms.kv_reuse_spec).  One
        chunk per ``history_docs`` entry (or a token grid).  Returns the entry
        record: key_hash, requested_span_tokens, kv_reuse_method and the
        ``cacheblend`` accounting (chunk_count, recomputed_tokens, ...)."""
        raise NotImplementedError

    # ---- chat shaping ----
    def prepare_chat(self, payload: Dict[str, Any], arm,
                     repair_plan: Optional[Dict[str, Any]],
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Backend-specific request shaping for one chat request (copy in,
        copy out): constrained-decoding fields, repair fields.  ``arm`` is
        the active arms.Arm; ``repair_plan`` (from repair_policy + the
        proxy ledger) describes the target doc when arm.repair is set.
        ``context`` is passed only to backends with
        ``wants_request_context`` (see the class attribute)."""
        raise NotImplementedError

    def normalize_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flat, backend-agnostic view of one chat response:
        content / tool_calls / finish_reason / usage / cost{...}.
        Raises BackendError for 200-with-error-body finishes."""
        raise NotImplementedError

    # ---- helpers shared by implementations ----
    @staticmethod
    def _message_choice(data: Dict[str, Any]) -> Dict[str, Any]:
        choices = data.get("choices") or [{}]
        return (choices[0] or {}).get("message") or {}
