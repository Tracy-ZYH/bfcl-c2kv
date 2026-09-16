"""CPU-only tests for the history-KV eviction arms (StreamingLLM / H2O /
SnapKV / PyramidKV), ported from the upstream kvoffload-sglang client
``c2kv_eval.adapters.bfcl_history_kv_baselines`` + ``run_history_kv_baselines.sh``.

Everything here shapes requests against a FAKE ``post_json``: the asserted
payloads are the contract with the reconciled server
(``/v1/c2kv/repair_extract`` history_kv_* fields,
``c2kv_kv_memory_hint.history_kv_eviction``, ``/open_session``).  NOTHING in
this file has been executed against a live server or a model.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# proxy.py imports its siblings as top-level modules (script-style)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from c2kv_eval.portable.arms import ARMS, Arm, get_arm, history_kv_spec
from c2kv_eval.portable import proxy as proxy_mod
from c2kv_eval.portable.backends.base import BackendError
from c2kv_eval.portable.backends.sglang import SglangBackend


HISTORY_ARMS = (
    "history_kv_streamingllm_r312",
    "history_kv_h2o_r312",
    "history_kv_snapkv_r312",
    "history_kv_pyramidkv_r312",
)


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


def _repair_extract_ok(payload):
    """A plausible server response: the server measured the span itself and
    kept ceil(span * ratio) of it."""
    span = 1000
    ratio = float(payload.get("history_kv_retention_ratio") or 1.0)
    target = payload.get("history_kv_target_tokens")
    kept = int(target) if target is not None else int(-(-span * ratio // 1))
    return {
        "success": True,
        "key_hash": "hk-" + str(payload.get("history_kv_method")),
        "token_len": kept,
        "requested_span_tokens": span,
        "selected_token_count": kept,
        "selected_relative_indices": list(range(span - kept, span)),
        "history_kv_method": payload.get("history_kv_method"),
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
    return out, counts, proxy_mod._history_kv_context(out, counts, arm)


# ---------------------------------------------------------------- registry

class TestRegistry:
    def test_arms_registered_with_upstream_defaults(self):
        # defaults mirror run_history_kv_baselines.sh HISTORY_KV_* env values
        for name in HISTORY_ARMS:
            arm = get_arm(name)
            spec = history_kv_spec(arm)
            assert arm.compress_history is False
            assert spec["retention_ratio"] == 0.312   # r312 = per-mille
            assert spec["backend"] == "repair_extract"
            assert spec["recent_window"] == 64
            assert spec["kernel_size"] == 5
            assert spec["pooling"] == "avgpool"
            assert spec["h2o_recent_fraction"] == 0.5
            assert spec["persistent_session"] is False
        assert history_kv_spec(get_arm(HISTORY_ARMS[0]))["method"] == "streamingllm"
        assert history_kv_spec(get_arm(HISTORY_ARMS[1]))["method"] == "h2o"
        # the server normalizes snapkv -> snapkv_persistent, pyramid -> pyramidkv
        assert history_kv_spec(get_arm(HISTORY_ARMS[2]))["method"] == "snapkv_persistent"
        assert history_kv_spec(get_arm(HISTORY_ARMS[3]))["method"] == "pyramidkv"

    def test_method_aliases(self):
        assert history_kv_spec(Arm(
            name="a", compress_history=False,
            history_kv={"method": "snapkv", "retention_ratio": 0.5},
        ))["method"] == "snapkv_persistent"
        assert history_kv_spec(Arm(
            name="a", compress_history=False,
            history_kv={"method": "pyramid", "retention_ratio": 0.5},
        ))["method"] == "pyramidkv"

    def test_r25_names_mean_exactly_twenty_five_percent_retention(self):
        for method in ("streamingllm", "h2o", "snapkv_persistent", "pyramidkv"):
            spec = history_kv_spec(get_arm(f"history_kv_{method}_r25"))
            assert spec["retention_ratio"] == 0.25

    def test_r25_persistent_names_use_physical_sessions(self):
        for method in ("streamingllm", "h2o", "snapkv_persistent", "pyramidkv"):
            spec = history_kv_spec(
                get_arm(f"history_kv_{method}_r25_persistent"))
            assert spec["retention_ratio"] == 0.25
            assert spec["target_tokens"] is None
            assert spec["backend"] == "physical_eviction"
            assert spec["persistent_session"] is True

    @pytest.mark.parametrize("method", ["h2o", "snapkv_persistent"])
    def test_headwise_recovery_retains_original_backend_and_budget(self, method):
        arm = Arm(
            name="unsafe", compress_history=False,
            history_kv={"method": method, "retention_ratio": 0.25,
                        "recovery_mode": "replace", "recovery_window": 2})
        spec = history_kv_spec(arm)
        assert spec["method"] == method
        assert spec["retention_ratio"] == 0.25
        assert spec["recovery_mode"] == "replace"

    def test_existing_arms_untouched(self):
        for name, arm in ARMS.items():
            if name.startswith("history_kv_"):
                continue
            assert arm.history_kv is None
            assert history_kv_spec(arm) is None

    @pytest.mark.parametrize("config", [
        {"method": "snapkv_refresh", "retention_ratio": 0.312},  # client-side only
        {"method": "streamingllm"},                              # no budget
        {"method": "streamingllm", "retention_ratio": 0.0},
        {"method": "streamingllm", "retention_ratio": 1.5},
        {"method": "streamingllm", "retention_ratio": 0.3, "pooling": "sum"},
        {"method": "streamingllm", "retention_ratio": 0.3, "recent_window": 0},
        {"method": "streamingllm", "retention_ratio": 0.3, "backend": "client"},
        {"method": "streamingllm", "retention_ratio": 0.3, "typo": 1},
        # a streaming session only exists on the physical path
        {"method": "h2o", "retention_ratio": 0.3, "persistent_session": True},
    ])
    def test_rejected_specs(self, config):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=False, history_kv=config).validate()

    def test_exclusive_with_other_mechanisms(self):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=True,
                history_kv={"method": "h2o", "retention_ratio": 0.3}).validate()
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=False, text_policy="hiagent",
                history_kv={"method": "h2o", "retention_ratio": 0.3}).validate()


# ------------------------------------------------------------ proxy split

class TestProxySplit:
    def test_history_span_excludes_system_and_current(self):
        arm = get_arm("history_kv_h2o_r312")
        out, counts, ctx = _context(_messages(), arm)
        cutoff = counts["current_start_out_index"]
        assert out[0]["role"] == "system"
        # history = everything before the current block, minus the system msg
        assert ctx["history_out_indices"] == list(range(1, cutoff))
        assert ctx["history_message_count"] == cutoff
        assert ctx["system_text"] == "sys prompt"
        assert ctx["n_history_messages"] == cutoff - 1
        # turn-doc packing of the completed history (same text the c2kv arm
        # gists); the assistant tool-call turn is in the training dialect
        assert ctx["history_text"].startswith("Previous turn\n[User query]\nq1")
        assert "<tool_call>" in ctx["history_text"]
        assert "current question" not in ctx["history_text"]
        assert "sys prompt" not in ctx["history_text"]
        assert ctx["method"] == "h2o" and ctx["backend"] == "repair_extract"

    def test_no_context_for_other_arms(self):
        for name in ("full", "c2kv", "hiagent"):
            arm = get_arm(name)
            assert proxy_mod._history_kv_context(
                [], {"current_start_out_index": 0}, arm) is None

    def test_first_turn_has_no_completed_history(self):
        arm = get_arm("history_kv_h2o_r312")
        out, counts, ctx = _context(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "first question"}], arm)
        assert ctx["history_out_indices"] == []
        assert ctx["history_text"] == ""


# ------------------------------------------------- repair_extract protocol

class TestRepairExtractPath:
    def _prepare(self, arm_name="history_kv_snapkv_r312", tools=None,
                 responses=None):
        arm = get_arm(arm_name)
        messages = _messages()
        out, counts, ctx = _context(messages, arm)
        post = FakePost(responses or {"/v1/c2kv/repair_extract": _repair_extract_ok})
        backend = SglangBackend(post)
        payload = {"messages": out, "model": "m"}
        if tools:
            payload["tools"] = tools
        prepared = backend.prepare_chat(payload, arm, None,
                                        context={"conversation_id": "c",
                                                 "history_kv": ctx})
        return arm, ctx, post, prepared

    def test_repair_extract_payload_is_exact(self):
        arm, ctx, post, _ = self._prepare(tools=_tools())
        assert post.paths() == ["/v1/c2kv/repair_extract"]
        call = post.calls[0]
        assert call["timeout"] == 600
        assert call["payload"] == {
            "messages": [
                {"role": "system", "content": "sys prompt"},
                {"role": "user", "content": ctx["history_text"]},
            ],
            "target_index": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "repair_mode": "history_kv_snapkv_persistent",
            "raw_kv_position_mode": "rotated",
            "extract_source": "model_prefill",
            "source_doc_index": 0,
            "history_kv_method": "snapkv_persistent",
            "history_kv_recent_window": 64,
            "history_kv_kernel_size": 5,
            "history_kv_pooling": "avgpool",
            "history_kv_h2o_recent_fraction": 0.5,
            "history_kv_retention_ratio": 0.312,
            "tools": _tools(),
        }

    def test_absolute_budget_replaces_the_ratio(self):
        arm = Arm(name="hk_abs", compress_history=False,
                  history_kv={"method": "streamingllm", "target_tokens": 128})
        out, counts, ctx = _context(_messages(), arm)
        post = FakePost({"/v1/c2kv/repair_extract": _repair_extract_ok})
        SglangBackend(post).prepare_chat({"messages": out}, arm, None,
                                         context={"history_kv": ctx})
        payload = post.calls[0]["payload"]
        assert payload["history_kv_target_tokens"] == 128
        assert "history_kv_retention_ratio" not in payload

    def test_recovery_refuses_a_server_that_silently_ignored_it(self):
        arm = Arm(
            name="stream_restore", compress_history=False,
            history_kv={"method": "streamingllm", "retention_ratio": 0.25,
                        "recovery_mode": "replace", "recovery_window": 2})
        spec = history_kv_spec(arm)
        spec["recovery_char_range"] = [0, 4]
        post = FakePost({"/v1/c2kv/repair_extract": _repair_extract_ok})
        with pytest.raises(BackendError, match="did not acknowledge"):
            SglangBackend(post).history_kv_extract(
                "abcd-rest", "system", _tools(), spec)

    def test_chat_request_carries_the_upstream_carrier(self):
        arm, ctx, post, prepared = self._prepare()
        messages = prepared["messages"]
        # system stays raw, history collapses into ONE carrier, current stays
        assert messages[0]["role"] == "system"
        carrier = messages[1]
        assert carrier == {
            "role": "user",
            "content": "[runtime snapkv_persistent compressed history kv]",
            "c2kv_repair_only_key_hashes": ["hk-snapkv_persistent"],
            "c2kv_use_gist_projection": False,
        }
        assert messages[-1]["content"] == "current question"
        # 4 history messages collapsed into 1 carrier: system + carrier + current
        assert ctx["history_out_indices"] == [1, 2, 3, 4]
        assert len(messages) == 3
        assert all("c2kv_key_hash" not in m for m in messages)
        # the frame-defining chat_template_kwargs are still set
        assert prepared["chat_template_kwargs"] == {"enable_thinking": False}
        assert "session_params" not in prepared

    def test_hint_leaves_active_accounting_to_the_scheduler(self):
        arm, ctx, post, prepared = self._prepare()
        kept = 312  # ceil(1000 * 0.312) from the fake server
        assert prepared["c2kv_kv_memory_hint"] == {
            "full_equivalent_history_tokens": 1000,
            "active_history_kv_tokens": 0,
            "active_full_raw_tokens": 0,
            "active_c2kv_gist_tokens": 0,
            "active_raw_repair_tokens": 0,
            "history_kv_method": "snapkv_persistent",
            "estimated": False,
            "history_kv_backend": "repair_extract",
            "history_kv_requested_span_tokens": 1000,
            "history_kv_selected_token_count": kept,
            "history_selection_metadata": None,
        }

    def test_first_turn_issues_no_extract(self):
        arm = get_arm("history_kv_h2o_r312")
        out, counts, ctx = _context(
            [{"role": "system", "content": "s"},
             {"role": "user", "content": "first"}], arm)
        post = FakePost({})
        prepared = SglangBackend(post).prepare_chat(
            {"messages": out}, arm, None, context={"history_kv": ctx})
        assert post.calls == []
        assert "c2kv_kv_memory_hint" not in prepared
        assert [m["content"] for m in prepared["messages"]] == ["s", "first"]

    def test_server_that_ignores_the_method_is_a_hard_failure(self):
        def ignored(payload):
            body = _repair_extract_ok(payload)
            body["history_kv_method"] = None
            return body
        with pytest.raises(BackendError) as excinfo:
            self._prepare(responses={"/v1/c2kv/repair_extract": ignored})
        assert excinfo.value.kind == "history_kv_extract_failed"

    def test_failed_extract_is_a_hard_failure(self):
        with pytest.raises(BackendError) as excinfo:
            self._prepare(responses={"/v1/c2kv/repair_extract": {
                "success": False, "error": "boom", "key_hash": ""}})
        assert excinfo.value.kind == "history_kv_extract_failed"

    def test_history_arm_without_context_is_a_hard_failure(self):
        arm = get_arm("history_kv_h2o_r312")
        with pytest.raises(BackendError):
            SglangBackend(FakePost({})).prepare_chat(
                {"messages": []}, arm, None, context=None)


# ---------------------------------------------- physical-eviction protocol

class TestPhysicalEvictionPath:
    ARM = Arm(name="hk_phys", compress_history=False,
              history_kv={"method": "h2o", "target_tokens": 256,
                          "backend": "physical_eviction"})
    SESSION_ARM = Arm(name="hk_phys_sess", compress_history=False,
                      history_kv={"method": "pyramidkv", "target_tokens": 256,
                                  "backend": "physical_eviction",
                                  "persistent_session": True})

    def test_hint_shape_and_no_extract_call(self):
        out, counts, ctx = _context(_messages(), self.ARM)
        post = FakePost({})
        prepared = SglangBackend(post).prepare_chat(
            {"messages": out}, self.ARM, None, context={"history_kv": ctx})
        assert post.calls == []          # the history stays raw text
        assert prepared["messages"] == out
        hint = prepared["c2kv_kv_memory_hint"]
        assert hint["history_kv_eviction"] == {
            "method": "h2o",
            "history_message_count": counts["current_start_out_index"],
            "target_tokens": 256,
            "retention_ratio": None,
            "history_kv_recent_window": 64,
            "history_kv_kernel_size": 5,
            "history_kv_pooling": "avgpool",
            "history_kv_h2o_recent_fraction": 0.5,
            "persistent_session": False,
        }
        assert hint["estimated"] is True
        assert hint["active_history_kv_tokens"] == 0
        assert hint["full_equivalent_history_tokens"] == 0  # server overwrites
        assert "persistent_history_session" not in hint
        assert "session_params" not in prepared

    def test_persistent_session_rides_on_session_params(self):
        out, counts, ctx = _context(_messages(), self.SESSION_ARM)
        ctx["session_id"] = "sess-1"
        prepared = SglangBackend(FakePost({})).prepare_chat(
            {"messages": out}, self.SESSION_ARM, None, context={"history_kv": ctx})
        assert prepared["session_params"] == {"id": "sess-1"}
        hint = prepared["c2kv_kv_memory_hint"]
        assert hint["persistent_history_session"] == {
            "enabled": True, "session_id": "sess-1"}
        assert hint["history_kv_eviction"]["persistent_session"] is True

    def test_ratio_budget_is_deferred_to_server_tokenization(self):
        arm = get_arm("history_kv_h2o_r25_persistent")
        out, counts, ctx = _context(_messages(), arm)
        ctx["session_id"] = "sess-ratio"
        prepared = SglangBackend(FakePost({})).prepare_chat(
            {"messages": out}, arm, None, context={"history_kv": ctx})
        eviction = prepared["c2kv_kv_memory_hint"]["history_kv_eviction"]
        assert eviction["retention_ratio"] == 0.25
        assert "target_tokens" not in eviction

    def test_open_session_payload(self):
        post = FakePost({"/open_session": lambda p: p["session_id"]})
        assert SglangBackend(post).open_history_session("sess-2", timeout=300) == "sess-2"
        assert post.calls[0]["path"] == "/open_session"
        assert post.calls[0]["payload"] == {
            "capacity_of_str_len": 0,
            "session_id": "sess-2",
            "streaming": True,
            "timeout": 300.0,
        }

    def test_open_session_mismatch_raises(self):
        post = FakePost({"/open_session": "someone-elses-id"})
        with pytest.raises(BackendError) as excinfo:
            SglangBackend(post).open_history_session("sess-3")
        assert excinfo.value.kind == "history_kv_session_failed"

    def test_close_session_payload(self):
        post = FakePost({"/close_session": {}})
        SglangBackend(post).close_history_session("sess-4")
        assert post.calls == [{
            "path": "/close_session",
            "payload": {"session_id": "sess-4"},
            "timeout": 60,
        }]

    def test_proxy_reuses_one_session_per_conversation(self, monkeypatch):
        opened = []

        class Fake(SglangBackend):
            def open_history_session(self, session_id, timeout=600):
                opened.append(session_id)
                return session_id

            def close_history_session(self, session_id, timeout=60):
                pass

        monkeypatch.setattr(proxy_mod, "BACKEND", Fake(FakePost({})))
        monkeypatch.setattr(proxy_mod.STATE, "history_sessions", {})
        monkeypatch.setattr(proxy_mod.STATE, "history_session_last_seen", {})
        monkeypatch.setattr(proxy_mod.STATE, "max_history_sessions", 2)
        first = proxy_mod._history_session_id("conv-a")
        again = proxy_mod._history_session_id("conv-a")
        other = proxy_mod._history_session_id("conv-b")
        assert first == again and first != other
        assert opened == [first, other]
        assert first.startswith("c2kv-bench-history-conv-a")

    def test_proxy_lru_closes_completed_conversation(self, monkeypatch):
        opened, closed = [], []

        class Fake(SglangBackend):
            def open_history_session(self, session_id, timeout=600):
                opened.append(session_id)
                return session_id

            def close_history_session(self, session_id, timeout=60):
                closed.append(session_id)

        monkeypatch.setattr(proxy_mod, "BACKEND", Fake(FakePost({})))
        monkeypatch.setattr(proxy_mod.STATE, "history_sessions", {})
        monkeypatch.setattr(proxy_mod.STATE, "history_session_last_seen", {})
        monkeypatch.setattr(proxy_mod.STATE, "max_history_sessions", 1)
        first = proxy_mod._history_session_id("conv-a")
        second = proxy_mod._history_session_id("conv-b")
        assert first != second
        assert closed == [first]
        proxy_mod._close_all_history_sessions()
        assert closed == [first, second]


# ------------------------------------------------------------ proxy wiring

class TestProxyWiring:
    """One chat request through ProxyHandler.do_POST: the proxy must build the
    history context, hand it to prepare_chat, and reproduce the upstream
    sequence (repair_extract then chat) on the wire."""

    def test_one_chat_request_reproduces_the_upstream_sequence(
            self, monkeypatch, tmp_path):
        arm = get_arm("history_kv_streamingllm_r312")
        chat_response = {
            "choices": [{"message": {"content": "done", "tool_calls": None},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 2},
            "metadata": {"sglang_runtime": {"kv_resident_tokens": 7},
                         "kv_memory_report": {
                             "history_kv_method": "streamingllm",
                             "history_kv_backend": "repair_extract",
                             "history_kv_selected_token_count": 312,
                             "active_history_kv_tokens": 312}},
        }
        wire = []

        def fake_post(path, payload, timeout, retries=2):
            wire.append((path, payload))
            if path == "/v1/c2kv/repair_extract":
                return _repair_extract_ok(payload)
            return dict(chat_response)

        log = tmp_path / "req.jsonl"
        monkeypatch.setattr(proxy_mod, "ARM", arm)
        monkeypatch.setattr(proxy_mod, "BACKEND", SglangBackend(fake_post))
        monkeypatch.setattr(proxy_mod, "_post_json", fake_post)
        monkeypatch.setattr(proxy_mod, "REQUEST_LOG_PATH", str(log))
        monkeypatch.setattr(proxy_mod.STATE, "recover", None)
        monkeypatch.setattr(proxy_mod.STATE, "reference_log_path", "")

        body = json.dumps({"messages": _messages(), "tools": _tools(),
                           "model": "m"}).encode("utf-8")
        handler = proxy_mod.ProxyHandler.__new__(proxy_mod.ProxyHandler)
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        sent = {}
        handler._send_json = lambda code, obj: sent.update(code=code, obj=obj)
        handler.do_POST()

        assert sent["code"] == 200
        assert [path for path, _ in wire] == [
            "/v1/c2kv/repair_extract", "/v1/chat/completions"]
        extract_payload = wire[0][1]
        assert extract_payload["history_kv_method"] == "streamingllm"
        assert extract_payload["history_kv_retention_ratio"] == 0.312
        assert extract_payload["tools"] == _tools()
        chat_payload = wire[1][1]
        assert [m["role"] for m in chat_payload["messages"]] == [
            "system", "user", "user"]
        assert chat_payload["messages"][1]["c2kv_repair_only_key_hashes"] == [
            "hk-streamingllm"]
        assert chat_payload["c2kv_kv_memory_hint"]["history_kv_method"] == "streamingllm"
        row = json.loads(log.read_text(encoding="utf-8").strip())
        assert row["status"] == "ok"
        assert row["history_kv"]["method"] == "streamingllm"
        assert row["history_kv_selected_tokens"] == 312


# ------------------------------------------------------------ cost columns

class TestCostColumns:
    @staticmethod
    def _response(report):
        return {
            "choices": [{"message": {"content": "ok", "tool_calls": None},
                         "finish_reason": "stop"}],
            "usage": {"completion_tokens": 3},
            "metadata": {"sglang_runtime": {"kv_resident_tokens": 10,
                                            "c2kv_query_proj": "gist"},
                         "kv_memory_report": report},
        }

    def test_repair_extract_echo(self):
        normalized = SglangBackend(FakePost({})).normalize_response(self._response({
            "history_kv_method": "snapkv_persistent",
            "history_kv_backend": "repair_extract",
            "full_equivalent_history_tokens": 1000,
            "full_equivalent_history_tokens_source": "request_hint",
            "active_history_kv_tokens": 312,
            "active_history_kv_tokens_source": "scheduler_runtime",
            "history_kv_requested_span_tokens": 1000,
            "history_kv_selected_token_count": 312,
            "source": "sglang_c2kv_runtime_injection",
        }))
        cost = normalized["cost"]
        assert cost["history_kv_method"] == "snapkv_persistent"
        assert cost["history_kv_backend"] == "repair_extract"
        assert cost["history_kv_selected_tokens"] == 312
        assert cost["history_kv_span_tokens"] == 1000
        assert cost["history_kv_active_tokens"] == 312
        assert cost["history_kv_active_tokens_source"] == "scheduler_runtime"
        assert cost["history_kv_full_equivalent_tokens_source"] == "request_hint"
        assert cost["kv_resident_tokens"] == 10   # existing columns survive

    def test_physical_eviction_echo(self):
        normalized = SglangBackend(FakePost({})).normalize_response(self._response({
            "history_kv_method": "h2o",
            "history_kv_runtime_status": "physical_eviction_ok",
            "active_history_kv_tokens": 256,
            "full_equivalent_history_tokens": 900,
            "physical_slots_freed": 640,
            "selection_reason": "h2o_heavy_hitter_recent",
            "history_kv_physical_eviction": {
                "success": True, "error": "", "method": "h2o",
                "runtime_status": "physical_eviction_ok",
                "old_physical_kv_slots": 1024, "new_physical_kv_slots": 384,
                "freed_physical_slots": 640, "history_tokens": 900,
                "kept_history_tokens": 256, "freed_kv_bytes": 12345,
            },
        }))
        cost = normalized["cost"]
        assert cost["history_kv_eviction_ok"] is True
        assert cost["history_kv_freed_slots"] == 640
        assert cost["history_kv_kept_tokens"] == 256
        assert cost["history_kv_history_tokens"] == 900
        assert cost["history_kv_freed_bytes"] == 12345
        assert cost["history_kv_runtime_status"] == "physical_eviction_ok"
        assert cost["history_kv_selection_reason"] == "h2o_heavy_hitter_recent"
        assert "history_kv_eviction_error" not in cost

    def test_failed_eviction_is_visible(self):
        cost = SglangBackend(FakePost({})).normalize_response(self._response({
            "history_kv_runtime_status": "physical_eviction_attention_scores_unavailable",
            "history_kv_physical_eviction": {
                "success": False,
                "error": "ATTENTION_SCORE_SELECTION_UNAVAILABLE_IN_NORMAL_PREFILL",
            },
        }))["cost"]
        assert cost["history_kv_eviction_ok"] is False
        assert cost["history_kv_eviction_error"].startswith("ATTENTION_SCORE")

    def test_no_history_columns_for_other_arms(self):
        cost = SglangBackend(FakePost({})).normalize_response(
            self._response(None))["cost"]
        assert not [k for k in cost if k.startswith("history_kv_")]
        assert cost["c2kv_query_proj"] == "gist"

    def test_columns_reach_the_request_log(self, tmp_path, monkeypatch):
        """cost columns are flattened into the jsonl row by _log_request."""
        log = tmp_path / "req.jsonl"
        monkeypatch.setattr(proxy_mod, "REQUEST_LOG_PATH", str(log))
        monkeypatch.setattr(proxy_mod, "ARM", get_arm("history_kv_h2o_r312"))
        monkeypatch.setattr(proxy_mod, "BACKEND", SglangBackend(FakePost({})))
        normalized = SglangBackend(FakePost({})).normalize_response(
            self._response({
                "history_kv_method": "h2o",
                "history_kv_physical_eviction": {
                    "success": True, "error": "", "method": "h2o",
                    "freed_physical_slots": 640, "kept_history_tokens": 256},
            }))
        handler = proxy_mod.ProxyHandler.__new__(proxy_mod.ProxyHandler)
        counts = {"history_kv": {"method": "h2o", "backend": "repair_extract",
                                 "n_history_messages": 4, "n_history_docs": 2,
                                 "history_message_count": 5}}
        proxy_mod.ProxyHandler._log_request(
            handler, {"messages": []}, normalized, counts,
            fingerprint="fp", conv="c", turn=1)
        row = json.loads(log.read_text(encoding="utf-8").strip())
        assert row["history_kv_method"] == "h2o"
        assert row["history_kv_freed_slots"] == 640
        assert row["history_kv_kept_tokens"] == 256
        assert row["history_kv"]["n_history_docs"] == 2
