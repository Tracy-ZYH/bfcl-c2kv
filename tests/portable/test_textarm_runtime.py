"""Exercise internal HiAgent retrieval without executing its meta-tool outside the proxy."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2kv_eval.portable import proxy, textarms
from c2kv_eval.portable.arms import get_arm


def _original():
    return {"model": "c2kv-agent", "messages": [
        {"role": "user", "content": "Compare the two accounts."},
        {"role": "assistant", "content": "Subgoal: inspect first account"},
        {"role": "tool", "content": "account A secret detail: 193", "tool_call_id": "call_a"},
        {"role": "assistant", "content": "Subgoal: inspect second account"},
        {"role": "user", "content": "Now compare."},
    ]}


def _retrieval(ids):
    return {"choices": [{"message": {"role": "assistant",
        "content": "Subgoal: compare the two accounts", "tool_calls": [
        {"id": "meta", "type": "function", "function": {
            "name": textarms.HIAGENT_RETRIEVE_TOOL_NAME,
            "arguments": json.dumps({"subgoal_ids": ids})}}]}}],
        "usage": {"prompt_tokens": 71, "completion_tokens": 9}}


def test_retrieval_reveals_trajectory_and_charges_extra_generation(monkeypatch):
    textarms.reset_state()
    monkeypatch.setattr(proxy, "_textarm_compress", lambda *a, **k: "Account inspected.")
    original = _original()
    arm = get_arm("hiagent_full")
    staged, stats = proxy._apply_text_arm(original, arm, "test")
    assert "193" not in json.dumps(staged["messages"])
    assert staged["tools"][-1]["function"]["name"] == textarms.HIAGENT_RETRIEVE_TOOL_NAME
    sent = []
    final_call = {"id": "environment", "type": "function", "function": {
        "name": "report_comparison", "arguments": json.dumps({"value": 193})}}
    final = {"choices": [{"message": {"role": "assistant",
        "content": "Subgoal: report the comparison",
        "tool_calls": [final_call]}}]}

    def send(payload):
        sent.append(payload)
        return final

    result = proxy._hiagent_retrieval_loop(original, arm, "test", _retrieval([1]), stats, send)
    assert result == final
    assert len(sent) == 1 and "193" in json.dumps(sent[0]["messages"])
    assert result["choices"][0]["message"]["tool_calls"] == [final_call]
    assert stats["retrieval_usage"] == {"calls": 1, "prompt_tokens": 71, "completion_tokens": 9}
    assert stats["retrieved_subgoals"] == [1]
    assert "tools" not in original


def test_nonexistent_retrieval_is_not_leaked_to_executor(monkeypatch):
    textarms.reset_state()
    monkeypatch.setattr(proxy, "_textarm_compress", lambda *a, **k: "Account inspected.")
    original = _original()
    arm = get_arm("hiagent_full")
    _, stats = proxy._apply_text_arm(original, arm, "test")
    with pytest.raises(ValueError, match="nonexistent"):
        proxy._hiagent_retrieval_loop(original, arm, "test", _retrieval([99]), stats,
                                      lambda payload: pytest.fail("invalid retrieval must not generate"))


@pytest.mark.parametrize("policy,mode,guideline", [
    ("acon_hist_ut_co", "hist", "ut_co"), ("acon_obs_ut", "obs", "ut"),
    ("acon_hist", "hist", "base")])
def test_acon_guideline_reaches_compressor(monkeypatch, policy, mode, guideline):
    seen = {}
    def transform(messages, *args, **kwargs):
        seen.update(kwargs)
        return messages, {}
    monkeypatch.setattr(textarms, "acon_transform", transform)
    proxy._apply_text_arm(_original(), get_arm(policy), "test")
    assert seen["mode"] == mode and seen["guideline"] == guideline
