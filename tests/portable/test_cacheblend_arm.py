"""CPU-only tests for the CacheBlend KV-reuse arms (``cacheblend_r16``,
``cacheblend_r15_k``): the arm registry, the proxy's per-doc history split,
and the request contract with the reconciled server's
``/v1/c2kv/repair_extract kv_reuse_method="cacheblend"`` route
(``task/c2kv-cacheblend``, ``c2kv/c2kv_serving_semantics.md`` section 10).

Everything here shapes requests against a FAKE ``post_json``.  NOTHING in
this file has been executed against a live server or a model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# proxy.py imports its siblings as top-level modules (script-style)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from c2kv_eval.portable.arms import ARMS, Arm, get_arm, kv_reuse_spec
from c2kv_eval.portable import proxy as proxy_mod
from c2kv_eval.portable.backends.base import BackendError
from c2kv_eval.portable.backends.sglang import SglangBackend


REUSE_ARMS = ("cacheblend_r16", "cacheblend_r15_k")


class FakePost:
    """Records (path, payload, timeout); returns canned bodies per path."""

    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def __call__(self, path, payload, timeout, retries=2):
        self.calls.append({"path": path, "payload": payload, "timeout": timeout})
        body = self.responses.get(path)
        if callable(body):
            return body(payload)
        if body is None:
            raise AssertionError(f"unexpected POST to {path}")
        return body

    def paths(self):
        return [call["path"] for call in self.calls]


def _cacheblend_ok(payload):
    """A plausible server response: the span is measured by the server, the
    whole span is stored, int(span * ratio) tokens were recomputed."""
    span = 900
    ratio = float(payload.get("cacheblend_recomp_ratio") or 0.16)
    recomputed = max(1, int(span * ratio))
    n_chunks = payload["target_end_index"] - payload["target_index"] + 1
    return {
        "success": True,
        "key_hash": "cb-" + str(payload.get("kv_reuse_method")),
        "token_len": span,
        "requested_span_tokens": span,
        "selected_token_count": span,
        "kv_reuse_method": payload.get("kv_reuse_method"),
        "cacheblend": {
            "chunk_count": n_chunks,
            "chunk_bounds": [[0, 450], [450, 900]][:n_chunks],
            "recomputed_tokens": recomputed,
            "effective_recomp_ratio": recomputed / span,
            "deviation_max": 3.5,
            "deviation_selected_min": 0.7,
            "check_layer": payload.get("cacheblend_check_layer"),
            "metric": payload.get("cacheblend_metric"),
        },
        "repair_mode": payload.get("repair_mode"),
        "already_rotated": True,
        "span_start": 40,
        "span_end": 40 + span,
        "rendered_prefix_len": 40,
    }


def _messages():
    return [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "content": "obs1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "current question"},
    ]


def _tools():
    return [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"a": {"type": "int"}}}}}]


def _context(messages, arm):
    out, counts = proxy_mod._assemble(messages, arm)
    return out, counts, proxy_mod._kv_reuse_context(out, counts, arm)


# ---------------------------------------------------------------- registry

class TestRegistry:
    def test_arms_registered_with_artifact_defaults(self):
        r16 = kv_reuse_spec(get_arm("cacheblend_r16"))
        assert get_arm("cacheblend_r16").compress_history is False
        assert r16["method"] == "cacheblend"
        assert r16["recomp_ratio"] == 0.16      # EuroSys artifact recomp_ratio
        assert r16["check_layer"] == 1          # artifact check_layers=[1]
        assert r16["metric"] == "v"             # artifact: V-deviation
        assert r16["mask"] == "causal"          # exact causality by default
        assert r16["chunking"] == "doc"
        assert r16["chunk_tokens"] is None
        k15 = kv_reuse_spec(get_arm("cacheblend_r15_k"))
        assert k15["metric"] == "k" and k15["recomp_ratio"] == 0.15

    def test_existing_arms_untouched(self):
        for name, arm in ARMS.items():
            if name.startswith("cacheblend_"):
                continue
            assert arm.kv_reuse is None
            assert kv_reuse_spec(arm) is None

    def test_spec_validation(self):
        def spec(**cfg):
            return kv_reuse_spec(Arm(name="a", compress_history=False, kv_reuse=cfg))

        with pytest.raises(ValueError):
            spec(method="lmcache")
        with pytest.raises(ValueError):
            spec(method="cacheblend", recomp_ratio=1.5)
        with pytest.raises(ValueError):
            spec(method="cacheblend", check_layer=-1)
        with pytest.raises(ValueError):
            spec(method="cacheblend", metric="q")
        with pytest.raises(ValueError):
            spec(method="cacheblend", mask="diag")
        with pytest.raises(ValueError):
            spec(method="cacheblend", chunking="grid")       # needs chunk_tokens
        with pytest.raises(ValueError):
            spec(method="cacheblend", chunk_tokens=256)      # doc chunking ignores it
        with pytest.raises(ValueError):
            spec(method="cacheblend", bogus=1)
        grid = spec(method="cacheblend", chunking="grid", chunk_tokens=256)
        assert grid["chunk_tokens"] == 256

    def test_exclusive_with_other_mechanisms(self):
        with pytest.raises(ValueError):
            Arm(name="a", compress_history=True, ratio=8,
                kv_reuse={"method": "cacheblend"}).validate()
        with pytest.raises(ValueError):
            Arm(name="a", compress_history=False, text_policy="hiagent",
                kv_reuse={"method": "cacheblend"}).validate()
        with pytest.raises(ValueError):
            Arm(name="a", compress_history=False,
                history_kv={"method": "h2o", "retention_ratio": 0.3},
                kv_reuse={"method": "cacheblend"}).validate()


# ------------------------------------------------------------ proxy context

class TestContext:
    def test_docs_kept_separate_system_outside(self):
        arm = get_arm("cacheblend_r16")
        out, counts, ctx = _context(_messages(), arm)
        assert ctx is not None
        assert ctx["method"] == "cacheblend" and ctx["chunking"] == "doc"
        assert ctx["system_text"] == "sys prompt"
        # history = q1 / tool-call / obs1 / a1 (out indices 1..4); the current
        # question stays raw
        assert ctx["history_out_indices"] == [1, 2, 3, 4]
        assert ctx["current_start_out_index"] == 5
        # two turn docs: (q1 + action) and (obs1 + a1), same packing the c2kv
        # arm gists, each one chunk
        assert ctx["n_history_docs"] == 2
        docs = ctx["history_docs"]
        assert [d["role"] for d in docs] == ["user", "user"]
        assert docs[0]["content"].startswith("Previous turn\n[User query]\nq1")
        assert "<tool_call>" in docs[0]["content"]
        assert "obs1" in docs[1]["content"] and "a1" in docs[1]["content"]

    def test_non_reuse_arm_has_no_context(self):
        out, counts = proxy_mod._assemble(_messages(), get_arm("full"))
        assert proxy_mod._kv_reuse_context(out, counts, get_arm("full")) is None

    def test_first_turn_has_no_docs(self):
        arm = get_arm("cacheblend_r16")
        first = [_messages()[0], {"role": "user", "content": "first question"}]
        out, counts, ctx = _context(first, arm)
        assert ctx["history_docs"] == [] and ctx["history_out_indices"] == []


# ------------------------------------------------------------------- wire

class TestWire:
    def _prepare(self, arm_name="cacheblend_r16", responses=None, messages=None):
        arm = get_arm(arm_name)
        out, counts, ctx = _context(messages or _messages(), arm)
        post = FakePost(responses if responses is not None
                        else {"/v1/c2kv/repair_extract": _cacheblend_ok})
        backend = SglangBackend(post)
        # Simulate a G checkpoint profile on the shared endpoint. The raw-KV
        # baseline must override this query mode at both protocol levels.
        payload = {"model": "c2kv-agent", "messages": out, "tools": _tools(),
                   "c2kv_use_gist_projection": True}
        prepared = backend.prepare_chat(
            payload, arm, None,
            context={"conversation_id": "c", "history_kv": None, "kv_reuse": ctx})
        return post, prepared, ctx

    def test_extract_request_contract(self):
        post, prepared, ctx = self._prepare()
        assert post.paths() == ["/v1/c2kv/repair_extract"]
        req = post.calls[0]["payload"]
        # multi-message form: system + one message per doc, span = the docs
        assert [m["role"] for m in req["messages"]] == ["system", "user", "user"]
        assert req["messages"][0]["content"] == "sys prompt"
        assert req["messages"][1]["content"] == ctx["history_docs"][0]["content"]
        assert req["target_index"] == 1 and req["target_end_index"] == 2
        assert req["kv_reuse_method"] == "cacheblend"
        assert req["repair_mode"] == "cacheblend"
        assert req["raw_kv_position_mode"] == "rotated"
        assert req["extract_source"] == "model_prefill"
        assert req["cacheblend_recomp_ratio"] == 0.16
        assert req["cacheblend_check_layer"] == 1
        assert req["cacheblend_metric"] == "v"
        assert req["cacheblend_mask"] == "causal"
        assert "cacheblend_chunk_tokens" not in req      # doc chunking
        assert req["tools"] == _tools()
        assert req["chat_template_kwargs"] == {"enable_thinking": False}
        # no history_kv_* field rides along: the server treats the two as
        # exclusive and would refuse the request
        assert not any(k.startswith("history_kv_") for k in req)

    def test_chat_request_carries_entry_in_place(self):
        post, prepared, ctx = self._prepare()
        msgs = prepared["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "user"]
        carrier = msgs[1]
        assert carrier["c2kv_repair_only_key_hashes"] == ["cb-cacheblend"]
        assert carrier["c2kv_repair_placement"] == "in_place"
        assert carrier["c2kv_use_gist_projection"] is False
        assert prepared["c2kv_use_gist_projection"] is False
        assert msgs[2]["content"] == "current question"
        hint = prepared["c2kv_kv_memory_hint"]
        # the whole span stays resident; the saving is compute
        assert hint["full_equivalent_history_tokens"] == 900
        assert hint["active_history_kv_tokens"] == 0
        assert hint["active_raw_repair_tokens"] == 0
        assert hint["active_recomputed_raw_tokens"] == int(900 * 0.16)
        assert hint["kv_reuse_method"] == "cacheblend"
        assert hint["cacheblend_chunk_count"] == 2
        assert hint["cacheblend_recomputed_tokens"] == int(900 * 0.16)
        assert hint["cacheblend_metric"] == "v" and hint["cacheblend_chunking"] == "doc"
        assert hint["estimated"] is False
        assert prepared["chat_template_kwargs"] == {"enable_thinking": False}

    def test_k_lineage_arm_sends_its_knobs(self):
        post, prepared, ctx = self._prepare("cacheblend_r15_k")
        req = post.calls[0]["payload"]
        assert req["cacheblend_metric"] == "k"
        assert req["cacheblend_recomp_ratio"] == 0.15
        assert prepared["c2kv_kv_memory_hint"]["cacheblend_recomputed_tokens"] == int(900 * 0.15)

    def test_grid_chunking_sends_chunk_tokens(self):
        arm = Arm(name="cb_grid", compress_history=False,
                  kv_reuse={"method": "cacheblend", "chunking": "grid", "chunk_tokens": 256})
        out, counts = proxy_mod._assemble(_messages(), arm)
        ctx = proxy_mod._kv_reuse_context(out, counts, arm)
        post = FakePost({"/v1/c2kv/repair_extract": _cacheblend_ok})
        SglangBackend(post).prepare_chat(
            {"messages": out, "tools": _tools()}, arm, None,
            context={"conversation_id": "c", "kv_reuse": ctx})
        req = post.calls[0]["payload"]
        assert req["cacheblend_chunk_tokens"] == 256
        assert req["target_end_index"] == 2   # the span is still the docs

    def test_strict_echo_required(self):
        # a server that ignored kv_reuse_method returns a plain repair entry
        def plain(payload):
            body = _cacheblend_ok(payload)
            body.pop("kv_reuse_method")
            body.pop("cacheblend")
            return body

        with pytest.raises(BackendError) as info:
            self._prepare(responses={"/v1/c2kv/repair_extract": plain})
        assert info.value.kind == "kv_reuse_extract_failed"

        def no_accounting(payload):
            body = _cacheblend_ok(payload)
            body.pop("cacheblend")
            return body

        with pytest.raises(BackendError) as info:
            self._prepare(responses={"/v1/c2kv/repair_extract": no_accounting})
        assert info.value.kind == "kv_reuse_extract_failed"

        def failed(payload):
            return {"success": False, "error": "Invalid cacheblend config: x"}

        with pytest.raises(BackendError) as info:
            self._prepare(responses={"/v1/c2kv/repair_extract": failed})
        assert info.value.kind == "kv_reuse_extract_failed"

    def test_first_turn_no_extract_no_hint(self):
        first = [_messages()[0], {"role": "user", "content": "first question"}]
        post, prepared, ctx = self._prepare(messages=first, responses={})
        assert post.paths() == []
        assert "c2kv_kv_memory_hint" not in prepared
        assert [m["content"] for m in prepared["messages"]] == ["sys prompt", "first question"]

    def test_missing_context_fails_loud(self):
        arm = get_arm("cacheblend_r16")
        backend = SglangBackend(FakePost({}))
        with pytest.raises(BackendError) as info:
            backend.prepare_chat({"messages": _messages()}, arm, None, context={})
        assert info.value.kind == "kv_reuse_failed"

    def test_cost_columns_from_server_echo(self):
        backend = SglangBackend(FakePost({}))
        data = {
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "metadata": {
                "sglang_runtime": {"kv_resident_tokens": 1000, "c2kv_layout": []},
                "kv_memory_report": {
                    "kv_reuse_method": "cacheblend",
                    "kv_reuse_backend": "repair_extract",
                    "active_history_kv_tokens": 900,
                    "active_history_kv_tokens_source": "scheduler_runtime",
                    "active_recomputed_raw_tokens": 144,
                    "cacheblend_span_tokens": 900,
                    "cacheblend_recomputed_tokens": 144,
                    "cacheblend_effective_recomp_ratio": 0.16,
                    "cacheblend_chunk_count": 2,
                    "cacheblend_metric": "v",
                    "cacheblend_cache_hit": False,
                },
            },
        }
        cost = backend.normalize_response(data)["cost"]
        assert cost["kv_reuse_method"] == "cacheblend"
        assert cost["cacheblend_recomputed_tokens"] == 144
        assert cost["cacheblend_span_tokens"] == 900
        assert cost["kv_reuse_active_tokens"] == 900
        assert cost["kv_reuse_active_tokens_source"] == "scheduler_runtime"
        assert cost["kv_reuse_recomputed_tokens"] == 144
        assert cost["cacheblend_cache_hit"] is False
        # a report without kv_reuse_method adds nothing
        data["metadata"]["kv_memory_report"] = {"history_kv_method": "h2o"}
        assert "kv_reuse_method" not in backend.normalize_response(data)["cost"]
