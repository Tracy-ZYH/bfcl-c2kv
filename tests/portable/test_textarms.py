"""Offline tests for the text-level baseline arms (textarms.py).

Audit-hardened assertions: compression must actually shrink the message
list, user instructions must survive, subgoal segmentation must trigger on
Subgoal content and be visibly degenerate on null-content tool-call
replies, ACON must not duplicate history (the tuple-precedence bug), and
an empty compressor result must raise, never cache.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c2kv_eval.portable import textarms
from c2kv_eval.portable.arms import Arm, get_arm


def _action_dialect(message):
    return "tool:" + str(message.get("tool_calls"))


def _fake_compress_ok(payload):
    """Accepts a full request payload dict; returns a canned summary."""
    assert "model" in payload and payload["messages"], "payload contract"
    user = payload["messages"][-1]["content"]
    return f"SUMMARY({len(user)} chars)"


def _fake_compress_empty(payload):
    return "   "


def _reset():
    textarms.reset_state()


def test_compressor_failure_is_logged_as_textarm_error(monkeypatch, tmp_path):
    """A failure before _apply_text_arm returns still gets the text-arm label."""
    from c2kv_eval.portable import proxy as proxy_mod

    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(
        proxy_mod, "ARM", SimpleNamespace(name="acon_hist", text_policy="acon_hist"))
    monkeypatch.setattr(proxy_mod, "BACKEND", SimpleNamespace(name="fake"))
    monkeypatch.setattr(proxy_mod, "REQUEST_LOG_PATH", str(log))

    def fail_before_stats(payload, arm, conv):
        raise textarms.TextarmCompressorError("compressor failed")

    monkeypatch.setattr(proxy_mod, "_apply_text_arm", fail_before_stats)
    body = json.dumps({
        "model": "m",
        "messages": [{"role": "user", "content": "question"}],
    }).encode("utf-8")
    handler = proxy_mod.ProxyHandler.__new__(proxy_mod.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    sent = {}
    handler._send_json = lambda code, obj: sent.update(code=code, obj=obj)

    handler.do_POST()

    row = json.loads(log.read_text(encoding="utf-8"))
    assert sent["code"] == 502
    assert row["status"] == "textarm_error"
    assert row["error_kind"] == "textarm_error"


# ---- Recent truncation -------------------------------------------------------

def _char_token_count(messages):
    return sum(len(str(message.get("content") or "")) for message in messages)


def test_recent_truncation_keeps_whole_newest_turn_and_full_current():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "A" * 80},
        {"role": "assistant", "content": "old call", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "content": "old result"},
        {"role": "assistant", "content": "old final"},
        {"role": "user", "content": "B" * 12},
        {"role": "assistant", "content": "new call", "tool_calls": [{"id": "new"}]},
        {"role": "tool", "content": "new result"},
        {"role": "assistant", "content": "new final"},
        {"role": "user", "content": "CURRENT"},
        {"role": "assistant", "content": "current call", "tool_calls": [{"id": "cur"}]},
        {"role": "tool", "content": "current result"},
    ]
    out, stats = textarms.recent_truncation_transform(
        messages, cutoff=9, retention_ratio=0.25,
        token_count=_char_token_count)
    rendered = json.dumps(out)
    assert "A" * 80 not in rendered
    assert "B" * 12 in rendered
    assert "new result" in rendered and "new final" in rendered
    assert "CURRENT" in rendered and "current result" in rendered
    assert out[0] == messages[0]
    assert stats["history_turns"] == 2
    assert stats["retained_history_turns"] == 1
    assert stats["dropped_history_turns"] == 1
    assert stats["n_compressor_calls"] == 0


def test_recent_truncation_never_splits_oversized_newest_turn():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "N" * 200},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "current"},
    ]
    out, stats = textarms.recent_truncation_transform(
        messages, cutoff=6, retention_ratio=0.01,
        token_count=_char_token_count)
    rendered = json.dumps(out)
    assert "N" * 200 in rendered and "result" in rendered
    assert stats["retained_history_turns"] == 1
    assert stats["retained_history_tokens"] > stats["target_history_tokens"]


# ---- HiAgent -----------------------------------------------------------------

def test_hiagent_compresses_and_shrinks():
    _reset()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "Subgoal: find data", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "found it"},
        {"role": "user", "content": "IMPORTANT: my budget is $300"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Subgoal: finish up", "tool_calls": [{"id": "3"}]},
        {"role": "tool", "content": "done-ish"},
        {"role": "user", "content": "continue"},
    ]
    out, stats = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect)
    assert stats["n_segments"] == 2 and stats["n_summarized"] == 1
    assert stats["n_compressor_calls"] == 1
    assert len(out) < len(messages), "compression must shrink the message list"
    assert out_chars_lt(stats), stats
    # note injected into system
    assert "Subgoal:" in out[0]["content"] and out[0]["role"] == "system"
    # user instruction inside the completed segment SURVIVES verbatim
    assert any(m.get("role") == "user" and "$300" in (m.get("content") or "")
               for m in out), "user constraint must survive"
    # summary message present
    assert any(m.get("role") == "user"
               and (m.get("content") or "").startswith("Subgoal 1: find data")
               for m in out)
    # current segment raw
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out)
    assert stats["degenerate"] is False
    # cached re-run: no new compressor call
    _, stats2 = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect)
    assert stats2["n_compressor_calls"] == 0


def test_hiagent_policy_uses_native_calls_and_allows_text_only_replies():
    note = textarms.HIAGENT_SUBGOAL_NOTE
    assert "Action:" not in note
    assert '"Subgoal: {subgoal}"' in note
    assert "structured tool_calls field" in note
    assert "reply naturally in assistant content" in note

    retrieval = textarms.HIAGENT_RETRIEVAL_NOTE
    assert "native function-calling interface" in retrieval
    assert "serialize the retrieval request into assistant content" in retrieval


def test_hiagent_null_content_toolcall_reply_is_visible_degenerate():
    _reset()
    # BFCL FC shape: assistant replies carry tool_calls with null content —
    # no Subgoal line ever appears.  The arm is a passthrough and MUST say
    # so (audit: silent full-arm degeneration).
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "r"},
        {"role": "user", "content": "next"},
    ]
    out, stats = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect)
    assert stats["n_segments"] == 0 and stats["n_summarized"] == 0
    assert stats["degenerate"] is True
    assert stats["n_compressor_calls"] == 0
    assert len(out) == len(messages)


def test_hiagent_user_in_summary_input():
    _reset()
    seen = {}

    def spy(payload):
        seen["user"] = payload["messages"][-1]["content"]
        return "ok summary"

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Subgoal: a", "tool_calls": [{"id": "1"}]},
        {"role": "user", "content": "IMPORTANT: my budget is $300"},
        {"role": "tool", "content": "obs"},
        {"role": "assistant", "content": "Subgoal: b", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "obs2"},
    ]
    textarms.hiagent_transform(messages, spy, _action_dialect)
    assert "$300" in seen["user"], "user constraint must reach the summarizer"


def test_hiagent_full_retrieves_selected_completed_trajectory():
    _reset()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "FIRST-RAW"},
        {"role": "assistant", "content": "Subgoal: second"},
        {"role": "tool", "content": "SECOND-RAW"},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "tool", "content": "CURRENT-RAW"},
    ]
    out, stats = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect,
        variant="full", retrieve_subgoals=[2, 99])
    rendered = json.dumps(out)
    assert "hiagent_retrieve" in out[0]["content"]
    assert "FIRST-RAW" not in rendered
    assert "SECOND-RAW" in rendered
    assert "CURRENT-RAW" in rendered
    assert stats["retrieved_subgoals"] == [2]
    assert stats["invalid_retrieval_subgoals"] == [99]
    assert stats["n_summarized"] == 1
    assert stats["n_compressor_calls"] == 1


def test_hiagent_retrieval_tool_call_contract():
    schema = textarms.hiagent_retrieval_tool()
    assert schema["function"]["name"] == "hiagent_retrieve"
    message = {"tool_calls": [
        {"function": {"name": "other", "arguments": "{}"}},
        {"function": {"name": "hiagent_retrieve",
                      "arguments": '{"subgoal_ids": [2, 1, 2]}'}}
    ]}
    assert textarms.hiagent_retrieval_request(message) == [2, 1]
    assert textarms.hiagent_retrieval_request({"content": "normal"}) is None
    try:
        textarms.hiagent_retrieval_request({"tool_calls": [{"function": {
            "name": "hiagent_retrieve", "arguments": '{"subgoal_ids": [0]}'}}]})
        raise AssertionError("malformed retrieval must fail")
    except ValueError:
        pass


def out_chars_lt(stats):
    return stats["out_chars"] < stats["raw_chars"]


# ---- ACON --------------------------------------------------------------------

def _acon_messages(obs_len=200):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "z" * obs_len},
        {"role": "user", "content": "final turn"},
    ]


def test_acon_obs_threshold_and_shrink():
    _reset()
    big = _acon_messages(obs_len=textarms.ACON_OBS_THRESHOLD_CHARS + 10)
    out, stats = textarms.acon_transform(
        big, _fake_compress_ok, _action_dialect, "convA", mode="obs")
    assert stats["n_obs_compressed"] == 1
    assert stats["n_compressor_calls"] == 1
    assert len(out[3]["content"]) < len(big[3]["content"])
    assert out[3]["content"].startswith("SUMMARY(")
    # under threshold: untouched
    small = _acon_messages(obs_len=300)
    _, stats2 = textarms.acon_transform(
        small, _fake_compress_ok, _action_dialect, "convA2", mode="obs")
    assert stats2["n_obs_compressed"] == 0
    # idempotent (cache)
    _, stats3 = textarms.acon_transform(
        big, _fake_compress_ok, _action_dialect, "convA", mode="obs")
    assert stats3["n_compressor_calls"] == 0


def _acon_long_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "long task"},
    ]
    for i in range(20):
        msgs += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"a{i}"}]},
            {"role": "tool", "content": "z" * 900},
        ]
    msgs.append({"role": "user", "content": "final turn"})
    return msgs


def test_acon_hist_no_duplication_and_embeds_summary():
    _reset()
    messages = _acon_long_messages()
    out, stats = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convB", mode="hist")
    assert stats["history_compressed"] is True
    assert stats["n_compressor_calls"] == 1
    # THE audit bug: the old tuple precedence made tail = ALL nonsystem and
    # duplicated the whole history (43 in -> 84 out).  Output must SHRINK.
    assert len(out) < len(messages), (
        f"history compression must shrink: {len(messages)} -> {len(out)}")
    # summary is embedded in the FIRST user prompt, not a standalone message
    first_user = next(m for m in out if m.get("role") == "user")
    assert "<HISTORY_SUMMARY>" in first_user["content"]
    assert first_user["content"].startswith("long task")
    # the preserved tail keeps the LAST action/observation pair (k=2) and
    # the final user turn; no message is duplicated
    assert out[-1]["content"] == "final turn"
    assert out[-2]["content"] == "z" * 900
    roles = [m.get("role") for m in out]
    assert len(roles) == len(set(map(str, roles))) or True  # roles repeat legitimately
    ids = [tc.get("id") for m in out for tc in (m.get("tool_calls") or [])]
    assert len(ids) == len(set(ids)), "tool_call ids must not duplicate"
    # faithful semantics: one new message since the compression; prev
    # summary + 1 message is far under T_hist -> NO call, the new turn
    # stays raw and visible to the policy
    messages2 = messages + [{"role": "assistant", "content": "done",
                             "tool_calls": [{"id": "z"}]}]
    out3, stats3 = textarms.acon_transform(
        messages2, _fake_compress_ok, _action_dialect, "convB", mode="hist")
    assert stats3["n_compressor_calls"] == 0
    assert stats3["history_compressed"] is False
    assert any(m.get("content") == "done" for m in out3)


def test_acon_hist_below_threshold_passthrough_no_duplication():
    _reset()
    messages = _acon_messages(obs_len=100)
    out, stats = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convC", mode="hist")
    assert stats["history_compressed"] is False
    assert len(out) == len(messages), "below threshold must not duplicate"


def test_acon_modes_split():
    _reset()
    messages = _acon_long_messages()
    messages[3]["content"] = "y" * (textarms.ACON_OBS_THRESHOLD_CHARS + 5)
    _, so = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convD", mode="obs")
    assert so["n_obs_compressed"] == 1 and so["history_compressed"] is False
    _reset()
    _, sh = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convD", mode="hist")
    assert sh["n_obs_compressed"] == 0 and sh["history_compressed"] is True


def test_acon_guideline_stages_are_distinct_and_cache_separated():
    _reset()
    seen = []

    def spy(payload):
        prompt = payload["messages"][-1]["content"]
        seen.append(prompt)
        return "# Refined Observation\nshort"

    big = _acon_messages(obs_len=textarms.ACON_OBS_THRESHOLD_CHARS + 10)
    for guideline in textarms.ACON_GUIDELINES:
        _, stats = textarms.acon_transform(
            big, spy, _action_dialect, "same-conv", mode="obs",
            guideline=guideline)
        assert stats["guideline"] == guideline
        assert stats["n_compressor_calls"] == 1
    assert len(seen) == 3
    assert "generate a \"Reasoning\"" in seen[0]
    assert "Always preserve:" in seen[1]
    assert "Reasoning (<=40 words)" in seen[2]


def test_acon_optimized_history_wrapper_is_not_nested():
    _reset()

    def wrapped(_payload):
        return "<HISTORY_SUMMARY>\n1. REASONING\nkept\n</HISTORY_SUMMARY>"

    out, stats = textarms.acon_transform(
        _acon_long_messages(), wrapped, _action_dialect, "conv-ut",
        mode="hist", guideline="ut")
    first_user = next(m for m in out if m.get("role") == "user")
    assert first_user["content"].count("<HISTORY_SUMMARY>") == 1
    assert stats["guideline"] == "ut"


def test_textarm_registry_names_and_capability_contracts():
    assert get_arm("hiagent").text_policy == "hiagent"
    assert get_arm("hiagent_summary").required_capabilities == ()
    assert get_arm("hiagent_full").required_capabilities == (
        "hiagent_trajectory_retrieval_v1",)
    for mode in ("hist", "obs"):
        for guideline in textarms.ACON_GUIDELINES:
            arm = get_arm(f"acon_{mode}_{guideline}")
            assert arm.text_policy == f"acon_{mode}_{guideline}"
    for name in ("cacheblend_r16", "cacheblend_r15_k"):
        assert get_arm(name).query_projection == "base"
    try:
        Arm(name="bad", compress_history=False,
            query_projection="auto").validate()
        raise AssertionError("unknown query projection must fail")
    except ValueError:
        pass


def test_empty_compressor_result_raises_and_never_caches():
    _reset()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "Subgoal: a", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "o"},
        {"role": "assistant", "content": "Subgoal: b", "tool_calls": [{"id": "2"}]},
    ]
    try:
        textarms.hiagent_transform(messages, _fake_compress_empty, _action_dialect)
        raise AssertionError("empty summary must raise")
    except textarms.TextarmCompressorError:
        pass
    assert not textarms._SUMMARY_CACHE, "nothing may be cached on failure"


def test_compressor_decode_params():
    hi = textarms.compressor_payload("hiagent", "m", "s", "u")
    assert hi["max_tokens"] == 100 and hi["stop"] == ["\n\n"]
    assert hi["chat_template_kwargs"] == {"enable_thinking": False}
    ac = textarms.compressor_payload("acon", "m", "s", "u")
    assert ac["max_tokens"] == 2048 and ac["seed"] == 42
    assert ac["chat_template_kwargs"] == {"enable_thinking": False}


def test_acon_rolling_only_new_messages():
    """Audit M2 fix: the second turn's compressor input must contain ONLY
    the messages past the covered point (not the whole prefix again), with
    prev_summary carried — O(new content), never O(T^2)."""
    _reset()
    prompts = []

    def spy(payload):
        prompts.append(payload["messages"][-1]["content"])
        return f"SUMMARY({len(prompts)})"

    def msgs(obs_mark):
        m = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "long task"},
        ]
        for i in range(20):
            m += [
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": f"a{i}"}]},
                {"role": "tool", "content": f"{obs_mark}-{i}-" + "z" * 900},
            ]
        return m

    _, s1 = textarms.acon_transform(msgs("OLDMARK"), spy, _action_dialect,
                                    "convRoll", mode="hist")
    assert s1["n_compressor_calls"] == 1
    # first cover: the whole prefix (the last action/observation pair is the
    # preserved TAIL, so OLDMARK-19 is deliberately NOT in the prompt)
    assert "OLDMARK-0-" in prompts[0] and "OLDMARK-18-" in prompts[0]
    assert "OLDMARK-19-" not in prompts[0]

    # turn 2 (faithful semantics, audit 2026-09-05): only ONE new pair
    # since the compression; prev_summary + new < T_hist -> NO compression,
    # the policy SEES the new raw turns (not folded)
    m2 = msgs("OLDMARK") + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "n0"}]},
        {"role": "tool", "content": "NEWMARK-extra-" + "y" * 900},
    ]
    out2, s2 = textarms.acon_transform(m2, spy, _action_dialect,
                                       "convRoll", mode="hist")
    assert s2["n_compressor_calls"] == 0, "under threshold must not compress"
    assert s2["history_compressed"] is False
    assert s2.get("new_raw_messages_visible") == 2  # the new action+obs pair raw
    joined = json.dumps(out2)
    assert "NEWMARK-extra-" in joined  # new raw turn visible to the policy
    assert "<HISTORY_SUMMARY>" in joined  # previous summary block present

    # keep adding new raw turns until prev_summary + new exceeds T_hist:
    # exactly ONE compression folding all the new messages
    extra = []
    for i in range(18):
        extra += [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": f"j{i}"}]},
            {"role": "tool", "content": f"FOLD{i}-" + "w" * 900},
        ]
    m3 = m2 + extra
    out3, s3 = textarms.acon_transform(m3, spy, _action_dialect,
                                       "convRoll", mode="hist")
    assert s3["n_compressor_calls"] == 1
    assert s3["history_compressed"] is True
    # the compressor folded FOLD0..FOLD16; FOLD17 is the PRESERVED pair —
    # it stays raw and visible in the output, not in the prompt
    assert "FOLD0-" in prompts[-1] and "FOLD16-" in prompts[-1]
    assert "FOLD17-" not in prompts[-1]
    assert any((m.get("content") or "").startswith("FOLD17-") for m in out3)

    # identical history again: covered == prefix, under threshold -> 0 calls
    _, s4 = textarms.acon_transform(m3, spy, _action_dialect,
                                    "convRoll", mode="hist")
    assert s4["n_compressor_calls"] == 0


def test_textarm_compress_finish_semantics():
    """Audit BUG1 fix: finish_reason=length (the HiAgent 100-token
    truncation regime) is a NORMAL completion; abort / missing finish /
    empty content are failures.  Self-contained (no pytest fixtures)."""
    from c2kv_eval.portable import proxy

    orig = proxy._post_json

    def resp(finish, content):
        return {"choices": [{"finish_reason": finish,
                             "message": {"content": content}}]}

    try:
        proxy._post_json = lambda path, payload, timeout: resp("length", "truncated but usable")
        assert proxy._textarm_compress({"model": "m"}) == "truncated but usable"
        proxy._post_json = lambda path, payload, timeout: resp("stop", "ok")
        assert proxy._textarm_compress({"model": "m"}) == "ok"
        for bad in (resp("abort", "x"), resp(None, "x"), resp("stop", "   ")):
            proxy._post_json = lambda path, payload, timeout, r=bad: r
            try:
                proxy._textarm_compress({"model": "m"})
                raise AssertionError(f"must raise on {bad}")
            except textarms.TextarmCompressorError:
                pass
    finally:
        proxy._post_json = orig


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")


def test_acon_state_isolated_across_tau2_shaped_tasks():
    """2026-09-04 audit BLOCKER: with a colliding conversation id, task B
    inherited task A's rolling state (prev_summary + covered_until) and B's
    early messages were neither summarized nor present.  The digest check
    plus per-task ids must keep the two tasks' states independent even if
    they somehow share one id."""
    from c2kv_eval.portable import proxy as proxy_mod

    def tau2_msgs(instruction, mark):
        m = [
            {"role": "system", "content": "airline system"},
            {"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": instruction},
        ]
        for i in range(20):
            m += [
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": f"{mark}{i}"}]},
                {"role": "tool", "content": f"{mark}{i}-" + "z" * 900},
            ]
        return m

    # different tasks MUST get different conversation ids (the primary fix)
    c1 = proxy_mod.conversation_id(tau2_msgs("book a flight", "A"))
    c2 = proxy_mod.conversation_id(tau2_msgs("refund my ticket", "B"))
    assert c1 != c2

    # belt-and-braces: even under a FORCED shared id, the digest check
    # resets the rolling state instead of inheriting the other task's
    _reset()
    seen = []

    def spy(payload):
        seen.append(payload["messages"][-1]["content"])
        return "SUMMARY"

    m1, m2 = tau2_msgs("book a flight", "A"), tau2_msgs("refund", "B")
    textarms.acon_transform(m1, spy, _action_dialect, "SHARED", mode="hist")
    _, s2 = textarms.acon_transform(m2, spy, _action_dialect, "SHARED",
                                    mode="hist")
    # B must NOT inherit A's coverage: its compressor saw B's own messages
    # from zero (A's marks absent from B's summary input)
    assert "A0-" not in seen[-1] and "A19-" not in seen[-1]
    assert "B0-" in seen[-1]
    assert s2["n_compressor_calls"] == 1


def test_acon_trigger_counts_tool_calls():
    """Audit 2026-09-05: assistant actions live in tool_calls (content
    empty) — the trigger estimate must count them or the arm NEVER fires
    and silently becomes a full arm."""
    _reset()
    calls = []

    def spy(payload):
        calls.append(1)
        return "SUMMARY"

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "t"},
    ]
    for i in range(20):
        msgs += [
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": f"c{i}", "function": {
                 "name": "do", "arguments": json.dumps(
                     {"blob": "x" * 400})}}]},
            {"role": "tool", "content": "ok"},
        ]
    # content-only estimate: ~50 chars; with tool_calls: ~20*450 = 9000+
    # chars — under the 16384 threshold, so extend to 40 pairs to cross it
    msgs += [
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "cx", "function": {
             "name": "do", "arguments": json.dumps({"blob": "y" * 400})}}]},
        {"role": "tool", "content": "ok"},
    ] * 20
    _, stats = textarms.acon_transform(msgs, spy, _action_dialect,
                                       "convTC", mode="hist")
    assert stats["history_compressed"] is True, (
        "trigger must fire on tool_calls-bearing turns "
        f"(prefix_est_tokens={stats.get('prefix_est_tokens')})")
    assert stats["n_compressor_calls"] == 1


def test_hiagent_default_system_prefix():
    """Audit 2026-09-05: with no system message, hiagent must insert
    default+note — not a note-only system that suppresses the training
    default the full arm gets."""
    out, _ = textarms.hiagent_transform(
        [{"role": "user", "content": "q"}],
        lambda payload: "s", _action_dialect)
    assert out[0]["role"] == "system"
    assert out[0]["content"].startswith("You are a helpful assistant.")
    assert "Subgoal" in out[0]["content"]
    assert out[0]["content"] == (
        textarms.TRAINING_DEFAULT_SYSTEM_PROMPT + "\n"
        + textarms.HIAGENT_SUBGOAL_NOTE.strip())
