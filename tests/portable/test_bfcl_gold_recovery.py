from __future__ import annotations

import json
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c2kv_eval.portable import bfcl_gold_recovery as recovery
from c2kv_eval.portable.adapters import bfcl_adapter


def _response(name, *, proxy_status=None, gist_tokens=None):
    tool_call = SimpleNamespace(
        id=f"call-{name}",
        type="function",
        function=SimpleNamespace(name=name, arguments="{}"),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            role="assistant", content=None, tool_calls=[tool_call]))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
    )
    if proxy_status is not None or gist_tokens is not None:
        response.c2kv_proxy = {}
        if proxy_status is not None:
            response.c2kv_proxy["gold_recovery"] = {"status": proxy_status}
        if gist_tokens is not None:
            response.c2kv_proxy["gist_tokens"] = gist_tokens
    return response


def _prefix_checker(decoded, ground_truth, test_entry):
    del test_entry
    for turn, expected in enumerate(ground_truth):
        actual_names = [
            item.split("(", 1)[0]
            for step in decoded[turn]
            for item in step
        ]
        expected_names = [item.split("(", 1)[0] for item in expected]
        if actual_names != expected_names:
            return {
                "valid": False,
                "multi_turn": {
                    "valid": False,
                    "error_type": "toy:mismatch",
                    "turn": turn,
                },
                "irrelevance": {"valid": True},
            }
    return {
        "valid": True,
        "multi_turn": {"valid": True},
        "irrelevance": {"valid": True},
    }


def _install_toy_handler(
        monkeypatch, tmp_path, selector, choose_response, *, max_events=1):
    mapping = {}
    calls = []

    class ModelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Client:
        class Completions:
            def create(inner_self, **kwargs):
                del inner_self
                calls.append(deepcopy(kwargs))
                return choose_response(kwargs)

        def __init__(self):
            self.chat = SimpleNamespace(completions=self.Completions())

    class OpenAICompletionsHandler:
        def __init__(self, model_name, temperature, registry_name, is_fc_model,
                     **kwargs):
            del kwargs
            self.model_name = model_name
            self.temperature = temperature
            self.registry_name = registry_name
            self.is_fc_model = is_fc_model
            self.model_name_underline_replaced = "c2kv_agent"
            self.client = Client()

        def inference(self, test_entry, include_input_log, exclude_state_log):
            del include_input_log, exclude_state_log
            if self.model_name_underline_replaced.startswith("c2kv_gold_retry_"):
                utils = sys.modules[
                    "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"]
                setattr(
                    utils,
                    f"{self.model_name_underline_replaced}_task_Class_instance",
                    object(),
                )
            inference_data = {"message": [], "tools": [{
                "type": "function",
                "function": {"name": "available_tool", "parameters": {}},
            }]}
            all_responses = []
            for turn, user_messages in enumerate(test_entry["question"]):
                if turn == 0:
                    inference_data["message"].extend(user_messages)
                else:
                    inference_data = self._add_next_turn_user_message_FC(
                        inference_data, user_messages)
                turn_responses = []
                steps = test_entry.get("_steps_per_turn", [1] * len(test_entry["question"]))[turn]
                for _step in range(steps):
                    api_response, _ = self._query_FC(inference_data)
                    parsed = self._parse_query_response_FC(api_response)
                    inference_data["message"].append(
                        parsed["model_responses_message_for_chat_history"])
                    turn_responses.append(parsed["model_responses"])
                all_responses.append(turn_responses)
            metadata = {"inference_log": []}
            if test_entry.get("force_quit"):
                metadata["inference_log"].append({
                    "role": "handler_log",
                    "content": "Model has been forced to quit after 20 steps.",
                })
            return all_responses, metadata

        def _parse_query_response_FC(self, api_response):
            message = api_response.choices[0].message
            return {
                "model_responses": [
                    {call.function.name: call.function.arguments}
                    for call in message.tool_calls
                ],
                "model_responses_message_for_chat_history": message,
                "tool_call_ids": [call.id for call in message.tool_calls],
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }

        def decode_execute(self, result, has_tool_call_tag):
            del has_tool_call_tag
            return [f"{next(iter(call))}()" for call in result]

        def _add_next_turn_user_message_FC(self, inference_data, user_message):
            inference_data["message"].extend(user_message)
            return inference_data

    modules = {
        "bfcl_eval": ModuleType("bfcl_eval"),
        "bfcl_eval.constants": ModuleType("bfcl_eval.constants"),
        "bfcl_eval.constants.model_config": ModuleType(
            "bfcl_eval.constants.model_config"),
        "bfcl_eval.model_handler": ModuleType("bfcl_eval.model_handler"),
        "bfcl_eval.model_handler.api_inference": ModuleType(
            "bfcl_eval.model_handler.api_inference"),
        "bfcl_eval.model_handler.api_inference.openai_completion": ModuleType(
            "bfcl_eval.model_handler.api_inference.openai_completion"),
        "bfcl_eval.eval_checker": ModuleType("bfcl_eval.eval_checker"),
        "bfcl_eval.eval_checker.multi_turn_eval": ModuleType(
            "bfcl_eval.eval_checker.multi_turn_eval"),
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils": ModuleType(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils"),
        "openai": ModuleType("openai"),
        "httpx": ModuleType("httpx"),
    }
    for name in (
        "bfcl_eval", "bfcl_eval.constants", "bfcl_eval.model_handler",
        "bfcl_eval.model_handler.api_inference",
        "bfcl_eval.eval_checker", "bfcl_eval.eval_checker.multi_turn_eval",
    ):
        modules[name].__path__ = []
    config_module = modules["bfcl_eval.constants.model_config"]
    config_module.MODEL_CONFIG_MAPPING = mapping
    config_module.ModelConfig = ModelConfig
    completion_module = modules[
        "bfcl_eval.model_handler.api_inference.openai_completion"]
    completion_module.OpenAICompletionsHandler = OpenAICompletionsHandler
    modules["openai"].OpenAI = object
    modules["httpx"].Timeout = lambda **kwargs: kwargs
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    audit_path = tmp_path / "task.jsonl"
    bfcl_adapter.install_handler(
        "http://proxy/v1",
        handler_name="c2kv-test",
        gold_recovery=selector,
        task_audit_path=audit_path,
        bfcl_oracle_max_events=max_events,
    )
    config = mapping["c2kv-test"]
    handler = config.model_handler(
        model_name=config.model_name,
        temperature=0.001,
        registry_name="c2kv-test",
        is_fc_model=True,
    )
    handler._gold_controller.prefix_checker = _prefix_checker
    return handler, calls, audit_path


def _entry(task_id="multi_turn_base_7", turns=2):
    return {
        "id": task_id,
        "question": [[{"role": "user", "content": f"question {turn}"}]
                     for turn in range(turns)],
        "function": [],
        "initial_config": {},
        "involved_classes": [],
    }


@pytest.mark.parametrize("selector", ["witness", "random"])
def test_final_turn_retry_replays_history_without_http_or_prompt_gold(
        monkeypatch, tmp_path, selector):
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        if context["user_turn"] == 0:
            return _response("good0")
        if context["attempt"] == 0:
            return _response("bad1", gist_tokens=7)
        return _response("good1", proxy_status="appended")

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, selector, choose)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1(x='secret-value')"]]

    result, _ = handler.inference(_entry(), False, True)

    assert [[next(iter(step[0])) for step in turn] for turn in result] == [
        ["good0"], ["good1"]]
    assert len(calls) == 3
    contexts = [call["extra_body"]["c2kv_eval_context"] for call in calls]
    assert [(item["user_turn"], item["step"], item["attempt"])
            for item in contexts] == [(0, 0, 0), (1, 0, 0), (1, 0, 1)]
    assert all(item["task_id"] == "multi_turn_base_7" for item in contexts)
    assert "c2kv_oracle" not in calls[0]["extra_body"]
    assert "c2kv_oracle" not in calls[1]["extra_body"]
    oracle = calls[2]["extra_body"]["c2kv_oracle"]
    assert oracle == {
        "kind": "bfcl_gold_turn_v2",
        "version": 2,
        "task_id": "multi_turn_base_7",
        "turn": 1,
        "retry_start_step": 0,
        "selector": selector,
        "values": ["good1", "secret-value"],
    }
    assert "secret-value" not in repr(calls[2]["messages"])
    assert "secret-value" not in json.dumps(calls[2]["tools"])
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["trigger_turn"] == 1
    assert row["retry_start_step"] == 0
    assert row["retry_eligibility"] == {
        "turn": 1, "eligible": True, "retry_start_step": 0,
        "observation_source": "response.c2kv_proxy.gist_tokens>0",
    }
    assert row["attempts"] == 2
    assert row["replay_count"] == 1
    assert row["retry_instances_cleaned"] == 1
    assert row["recovered"] is True
    assert row["did_intervene"] is True


def test_retry_failure_returns_retry_trace_and_never_tries_again(
        monkeypatch, tmp_path):
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        if context["user_turn"] == 0:
            return _response("good0")
        return _response("bad1", proxy_status="appended", gist_tokens=7)

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"]]

    result, _ = handler.inference(_entry(), False, True)

    assert next(iter(result[1][0][0])) == "bad1"
    assert len(calls) == 3
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["status"] == "retry_failed"
    assert row["attempts"] == 2
    assert row["recovered"] is False


def test_audit_serializes_nested_checker_diagnostics_without_changing_return(
        monkeypatch, tmp_path):
    class DirectoryLike:
        def __repr__(self):
            return "DirectoryLike('/sandbox/records')"

    def checker(decoded, ground_truth, test_entry):
        outcome = _prefix_checker(decoded, ground_truth, test_entry)
        if not outcome["valid"]:
            outcome["multi_turn"]["filesystem"] = {
                "nested": [DirectoryLike(), {"cwd": DirectoryLike()}],
            }
        return outcome

    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        if context["user_turn"] == 0:
            return _response("good0")
        if context["attempt"] == 0:
            return _response("bad1", gist_tokens=7)
        return _response("good1", proxy_status="appended")

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"]]
    handler._gold_controller.prefix_checker = checker

    result, _ = handler.inference(_entry(), False, True)

    assert [[next(iter(step[0])) for step in turn] for turn in result] == [
        ["good0"], ["good1"]]
    assert len(calls) == 3
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["check_error"]["valid"] is False
    assert row["recovered"] is True
    assert row["retry_final_valid"] is True
    diagnostic = row["check_error"]["multi_turn"]["filesystem"]["nested"]
    expected = {
        "type": f"{DirectoryLike.__module__}.{DirectoryLike.__qualname__}",
        "repr": "DirectoryLike('/sandbox/records')",
    }
    assert diagnostic == [expected, {"cwd": expected}]


def test_turn_zero_without_observed_gist_is_not_retried_and_keeps_base(
        monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness",
        lambda kwargs: _response(
            "bad0" if kwargs["extra_body"]["c2kv_eval_context"]["user_turn"] == 0
            else "good1"),
    )
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"]]

    result, _ = handler.inference(_entry(), False, True)

    assert len(result) == 2
    assert len(calls) == 2
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["trigger_turn"] == 0
    assert row["attempts"] == 1
    assert row["skip_reason"] == "no_compressed_history"
    assert row["retry_start_step"] is None
    assert row["retry_eligibility"]["eligible"] is False


def test_turn_zero_replays_precompression_step_then_recovers_target_suffix(
        monkeypatch, tmp_path):
    """A failed first user turn is repairable only from its observed gist step."""
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        assert context["user_turn"] == 0
        if context["attempt"] == 0 and context["step"] == 0:
            return _response("base_step0")
        if context["attempt"] == 0 and context["step"] == 1:
            return _response("fullwrong", gist_tokens=9)
        if context["attempt"] == 0 and context["step"] == 2:
            return _response("base_after_failure", gist_tokens=9)
        assert context in ({
            "benchmark": "bfcl", "task_id": "multi_turn_base_7",
            "user_turn": 0, "step": 1, "attempt": 1,
        }, {
            "benchmark": "bfcl", "task_id": "multi_turn_base_7",
            "user_turn": 0, "step": 2, "attempt": 1,
        })
        return _response(
            "gold_step1" if context["step"] == 1 else "gold_step2",
            proxy_status="appended")

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["base_step0()", "gold_step1(x='from-gold')", "gold_step2()"]]
    entry = _entry(turns=1)
    entry["_steps_per_turn"] = [3]

    result, _ = handler.inference(entry, False, True)

    assert [[next(iter(step[0])) for step in turn] for turn in result] == [
        ["base_step0", "gold_step1", "gold_step2"]]
    # Base calls are steps 0..2.  Retry step 0 is a latency-zero replay;
    # step 1 starts recovery and later target steps keep the same payload.
    assert [(call["extra_body"]["c2kv_eval_context"]["step"],
             call["extra_body"]["c2kv_eval_context"]["attempt"])
            for call in calls] == [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
    oracle = calls[3]["extra_body"]["c2kv_oracle"]
    assert calls[4]["extra_body"]["c2kv_oracle"] == oracle
    assert oracle["kind"] == "bfcl_gold_turn_v2"
    assert oracle["retry_start_step"] == 1
    # BFCL gold is turn-scoped, including the replayed call; it is still
    # independent of the failed model action at step 1.
    assert oracle["values"] == [
        "base_step0", "gold_step1", "from-gold", "gold_step2"]
    assert "fullwrong" not in oracle["values"]
    assert "from-gold" not in repr(calls[3]["messages"])

    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["retry_start_step"] == 1
    assert row["replay_count"] == 1
    assert row["first_compressed_request_steps"] == [{
        "user_turn": 0, "step": 1,
        "observation_source": "response.c2kv_proxy.gist_tokens>0",
    }]
    assert row["recovered"] is True


def test_nonliteral_gold_records_interface_failure_and_keeps_base_trace(
        monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness",
        lambda kwargs: _response(
            "good0" if kwargs["extra_body"]["c2kv_eval_context"]["user_turn"] == 0
            else "bad1", gist_tokens=7),
    )
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1(x=make_value())"]]

    result, _ = handler.inference(_entry(), False, True)

    assert next(iter(result[1][0][0])) == "bad1"
    assert len(calls) == 2
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["status"] == "unrepairable_interface_unsupported"
    assert row["skip_reason"] == "interface_unsupported_nonliteral_gold"
    assert row["nonliteral"]["call_index"] == 0
    assert row["nonliteral"]["field"] == "kwargs.x"


def test_empty_irrelevance_gold_is_not_treated_as_a_witness_target(
        monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness",
        lambda kwargs: _response(
            "good0" if kwargs["extra_body"]["c2kv_eval_context"]["user_turn"] == 0
            else "unexpected_call", gist_tokens=7),
    )
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], []]

    handler.inference(_entry(), False, True)

    assert len(calls) == 2
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["skip_reason"] == "interface_unsupported_empty_gold_turn"
    assert row["did_intervene"] is False


def test_full_arm_only_adds_eval_context_and_never_loads_gold(
        monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, None, lambda kwargs: _response("any"))
    handler._gold_controller.ground_truth_loader = lambda entry: pytest.fail(
        "full arm must not read possible_answer")

    result, _ = handler.inference(_entry(), False, True)

    assert len(result) == 2
    assert [(call["extra_body"]["c2kv_eval_context"]["user_turn"],
             call["extra_body"]["c2kv_eval_context"]["attempt"])
            for call in calls] == [(0, 0), (1, 0)]
    assert all("c2kv_oracle" not in call["extra_body"] for call in calls)
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["schema"] == "bfcl_task_telemetry_v1"
    assert row["selector"] is None
    assert row["oracle_enabled"] is False
    assert row["status"] == "completed"
    assert row["total_wall_seconds"] >= 0


def test_force_quit_is_a_failed_final_prefix(monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness",
        lambda kwargs: _response(
            "good0" if kwargs["extra_body"]["c2kv_eval_context"]["user_turn"] == 0
            else "good1", proxy_status="appended", gist_tokens=7),
    )
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"]]
    entry = _entry()
    entry["force_quit"] = True

    handler.inference(entry, False, True)

    assert len(calls) == 3
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["base_error"]["type"] == "force_quit"
    assert row["status"] == "retry_failed"
    assert row["recovered"] is False


def test_official_checker_cleanup_preserves_live_namespace(monkeypatch):
    checker_module = ModuleType(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    utils_module = ModuleType(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    utils_module.live_model_task_Class_instance = object()

    def main(decoded, ground_truth, test_entry, category, namespace):
        del decoded, ground_truth, test_entry, category
        setattr(utils_module, f"{namespace}_eval_task_Class_instance", object())
        setattr(
            utils_module,
            f"{namespace}_ground_truth_eval_task_Class_instance",
            object(),
        )
        return {"valid": True}

    checker_module.multi_turn_checker = main
    checker_module.multi_turn_irrelevance_checker = (
        lambda decoded, ground_truth: {"valid": True})
    monkeypatch.setitem(sys.modules, checker_module.__name__, checker_module)
    monkeypatch.setitem(sys.modules, utils_module.__name__, utils_module)

    outcome = recovery.official_prefix_check(
        [[[]]], [[]], {"id": "multi_turn_base_1"})

    assert outcome["valid"] is True
    assert utils_module.live_model_task_Class_instance is not None
    assert [name for name in vars(utils_module)
            if name.startswith("c2kv_gold_check_")] == []


def test_thread_local_contexts_do_not_cross_tasks():
    controller = recovery.GoldRecoveryController(None)
    barrier = threading.Barrier(2)
    seen = {}

    def worker(task_id):
        controller.begin(_entry(task_id=task_id))
        barrier.wait()
        seen[task_id] = controller.request_context()
        controller.clear()

    threads = [
        threading.Thread(target=worker, args=(f"multi_turn_base_{index}",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["multi_turn_base_0"]["task_id"] == "multi_turn_base_0"
    assert seen["multi_turn_base_1"]["task_id"] == "multi_turn_base_1"


def test_replay_has_zero_latency_for_official_accounting():
    controller = recovery.GoldRecoveryController(
        "witness", ground_truth_loader=lambda entry: [["good0()"], ["good1()"]])
    state = controller.begin(_entry())
    cached_response = object()
    state.response_cache[0].append((cached_response, 7.5))
    state.trigger_turn = 1
    controller.begin_retry()

    assert controller.replay_response() == (cached_response, 0.0)


def test_audit_summary_separates_no_witness_from_unrepairable(tmp_path):
    path = tmp_path / "audit.jsonl"
    rows = [
        {"selector": "witness", "trigger_turn": 1,
         "recovered": True, "did_intervene": False,
         "intervention_statuses": ["no_literal_witness"], "skip_reason": None},
        {"selector": "witness", "trigger_turn": 0,
         "recovered": None, "did_intervene": False,
         "intervention_statuses": [],
         "skip_reason": "earlier_unrepairable_turn_failure"},
        {"selector": "random", "trigger_turn": 2,
         "recovered": False, "did_intervene": True,
         "intervention_statuses": ["appended"], "skip_reason": None},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")

    summary = recovery.summarize_audit(path)

    assert summary["n_oracle_tasks"] == 3
    assert summary["n_oracle_recovered"] == 1
    assert summary["n_oracle_intervened"] == 1
    assert summary["n_no_witness"] == 1
    assert summary["n_eligible_unrepairable"] == 1
    assert summary["n_interface_unsupported"] == 0
    assert summary["n_earlier_unrepairable_turn_failure"] == 1


def test_audit_summary_counts_controller_no_compressed_skip_once(tmp_path):
    path = tmp_path / "audit.jsonl"
    row = {
        "selector": "witness",
        "trigger_turn": 0,
        "recovered": None,
        "did_intervene": False,
        "intervention_statuses": [],
        "skip_reason": "no_compressed_history",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = recovery.summarize_audit(path)

    assert summary["n_no_compressed_history"] == 1
    assert summary["n_eligible_unrepairable"] == 1
    assert summary["eligible_unrepairable_by_reason"] == {
        "no_compressed_history": 1}


def test_v3_recovers_two_distinct_turns_from_latest_repaired_path(
        monkeypatch, tmp_path):
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        turn, step, attempt = (
            context["user_turn"], context["step"], context["attempt"])
        if attempt == 0:
            if turn == 0:
                return _response("good0")
            if step == 0:
                return _response("kept1")
            return _response("bad1", gist_tokens=7)
        if attempt == 1:
            if turn == 1:
                return _response("good1", proxy_status="appended")
            return _response("bad2", gist_tokens=9)
        assert (turn, step, attempt) == (2, 0, 2)
        return _response("good2", proxy_status="appended", gist_tokens=10)

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose, max_events=4)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"],
        ["kept1()", "good1(x='first-secret')"],
        ["good2(x='second-secret')"],
    ]
    entry = _entry(turns=3)
    entry["_steps_per_turn"] = [1, 2, 1]

    result, _ = handler.inference(entry, False, True)

    assert [[next(iter(step[0])) for step in turn] for turn in result] == [
        ["good0"], ["kept1", "good1"], ["good2"]]
    contexts = [call["extra_body"]["c2kv_eval_context"] for call in calls]
    assert [(item["user_turn"], item["step"], item["attempt"])
            for item in contexts] == [
                (0, 0, 0), (1, 0, 0), (1, 1, 0),
                (1, 1, 1), (2, 0, 1), (2, 0, 2),
            ]
    target_calls = [calls[3], calls[5]]
    assert [call["extra_body"]["c2kv_oracle"]["turn"]
            for call in target_calls] == [1, 2]
    assert all(call["extra_body"]["c2kv_oracle"]["kind"]
               == "bfcl_gold_turn_v3" for call in target_calls)
    assert all(call["extra_body"]["c2kv_oracle"]["version"] == 3
               for call in target_calls)
    assert "c2kv_oracle" not in calls[4]["extra_body"]
    assert "good1" in repr(calls[5]["messages"])
    assert "bad1" not in repr(calls[5]["messages"])
    assert all(secret not in repr(call["messages"])
               for call in target_calls
               for secret in ("first-secret", "second-secret"))

    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["oracle_kind"] == "bfcl_gold_turn_v3"
    assert row["legacy_scalar_fields_scope"].startswith("not_applicable")
    assert row["trigger_turn"] is None and row["recovered"] is None
    assert [(event["turn"], event["start_step"], event["success"], event["status"])
            for event in row["events"]] == [
                (1, 1, True, "target_recovered"),
                (2, 0, True, "target_recovered"),
            ]
    assert [event["extract_statuses"] for event in row["events"]] == [
        ["appended"], ["appended"]]
    assert [event["retry_attempt"] for event in row["events"]] == [1, 2]
    assert [event["c2kv_oracle_event_key"] for event in row["events"]] == [
        ["bfcl_gold_turn_v3", "3", "multi_turn_base_7", 1],
        ["bfcl_gold_turn_v3", "3", "multi_turn_base_7", 2],
    ]
    assert [(event["cost"]["http_call_count"],
             event["cost"]["replay_count"],
             event["cost"]["checker_calls"])
            for event in row["events"]] == [(1, 2, 2), (1, 3, 3)]
    assert all(event["cost"]["checker_seconds"] >= 0
               and event["cost"]["retry_wall_seconds"] >= 0
               and event["cost"]["base_failure_detector_checker_included"] is False
               and event["cost"]["replayed_original_http_latency_included"] is False
               for event in row["events"])
    assert row["retry_count"] == 2
    assert row["attempts"] == 3
    assert row["http_call_count"] == 6
    assert row["replay_count"] == 5
    assert row["retry_instances_cleaned"] == 2
    assert row["retry_namespaces_cleaned"] == 2
    assert row["event_cost_totals"]["events_with_cost"] == 2
    assert row["event_cost_totals"]["http_call_count"] == 2
    assert row["event_cost_totals"]["replay_count"] == 5
    assert row["event_cost_totals"]["checker_calls"] == 5
    # Turn 1's base observation was discarded with its failed suffix. The
    # repaired suffix reported no gist, so only the latest turn-2 observation
    # remains in the final trajectory cache.
    assert row["first_compressed_request_steps"] == [
        {"user_turn": 2, "step": 0,
         "observation_source": "response.c2kv_proxy.gist_tokens>0"},
    ]
    summary = recovery.summarize_audit(audit_path)
    assert summary["n_oracle_events"] == 2
    assert summary["n_oracle_event_retries"] == 2
    assert summary["n_oracle_event_targets_recovered"] == 2
    assert summary["total_oracle_retries"] == 2
    assert summary["total_handler_http_calls"] == 6


def test_v3_failed_retry_target_blocks_later_events_but_finishes_trajectory(
        monkeypatch, tmp_path):
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        if context["user_turn"] == 0:
            return _response("good0")
        if context["user_turn"] == 1:
            return _response("bad1", proxy_status="appended", gist_tokens=7)
        assert context["attempt"] == 1
        assert "c2kv_oracle" not in kwargs["extra_body"]
        return _response("good2", gist_tokens=8)

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose, max_events=4)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"], ["good2()"]]

    result, _ = handler.inference(_entry(turns=3), False, True)

    assert next(iter(result[1][0][0])) == "bad1"
    assert next(iter(result[2][0][0])) == "good2"
    assert len(calls) == 4
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(row["events"]) == 1
    assert row["events"][0]["success"] is False
    assert row["events"][0]["status"] == "target_failed"
    assert row["skip_reason"] == "retry_target_failed"
    assert row["retry_count"] == 1


def test_v3_invalid_replayed_prefix_aborts_event_without_gold_or_second_retry(
        monkeypatch, tmp_path):
    checker_calls = 0

    def checker(decoded, ground_truth, test_entry):
        nonlocal checker_calls
        del decoded, ground_truth, test_entry
        checker_calls += 1
        valid = checker_calls == 1
        return {
            "valid": valid,
            "multi_turn": {
                "valid": valid,
                "error_type": None if valid else "toy:replay_changed",
            },
            "irrelevance": {"valid": True},
        }

    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        turn, attempt = context["user_turn"], context["attempt"]
        if attempt == 0:
            return _response("good0" if turn == 0 else "bad1", gist_tokens=7)
        assert "c2kv_oracle" not in kwargs["extra_body"]
        return _response(f"ordinary{turn}", gist_tokens=7)

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose, max_events=4)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"], ["good2()"]]
    handler._gold_controller.prefix_checker = checker

    result, _ = handler.inference(_entry(turns=3), False, True)

    assert len(result) == 3
    assert [(call["extra_body"]["c2kv_eval_context"]["user_turn"],
             call["extra_body"]["c2kv_eval_context"]["attempt"])
            for call in calls] == [(0, 0), (1, 0), (1, 1), (2, 1)]
    assert all("c2kv_oracle" not in call["extra_body"] for call in calls[2:])
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(row["events"]) == 1
    event = row["events"][0]
    assert event["status"] == "replayed_prefix_invalid"
    assert event["success"] is False
    assert event["abort_turn"] == 0
    assert event["retry_check_error"]["multi_turn"]["error_type"] == (
        "toy:replay_changed")
    assert event["cost"]["http_call_count"] == 0
    assert event["cost"]["replay_count"] == 1
    assert event["cost"]["checker_calls"] == 1
    assert row["skip_reason"] == "replayed_prefix_invalid"
    assert row["retry_count"] == 1
    assert row["event_aggregation"]["target_recovered"] == 0
    assert row["event_aggregation"]["target_failed"] == 1
    assert row["http_call_count"] == 4


def test_v3_budget_exhaustion_records_event_and_keeps_latest_trace(
        monkeypatch, tmp_path):
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        turn, attempt = context["user_turn"], context["attempt"]
        names = {
            (0, 0): "good0",
            (1, 0): "bad1",
            (1, 1): "good1",
            (2, 1): "bad2",
            (2, 2): "good2",
            (3, 2): "bad3",
        }
        name = names[(turn, attempt)]
        recovered = (turn, attempt) in {(1, 1), (2, 2)}
        return _response(
            name,
            proxy_status="appended" if recovered else None,
            gist_tokens=7 if turn else None,
        )

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose, max_events=2)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"], ["good2()"], ["good3()"]]

    result, _ = handler.inference(_entry(turns=4), False, True)

    assert next(iter(result[3][0][0])) == "bad3"
    assert len(calls) == 6
    assert "c2kv_oracle" not in calls[-1]["extra_body"]
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert [event["status"] for event in row["events"]] == [
        "target_recovered", "target_recovered", "event_budget_exhausted"]
    assert row["events"][-1]["eligibility"] == {
        "eligible": False,
        "reason": "event_budget_exhausted",
        "observation_source": "response.c2kv_proxy.gist_tokens>0",
    }
    assert row["retry_count"] == 2
    assert row["skip_reason"] == "event_budget_exhausted"


def test_v3_no_compressed_failure_never_sends_privileged_metadata(
        monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch,
        tmp_path,
        "witness",
        lambda kwargs: _response(
            "bad0" if kwargs["extra_body"]["c2kv_eval_context"]["user_turn"] == 0
            else "good1"),
        max_events=4,
    )
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"]]

    handler.inference(_entry(), False, True)

    assert len(calls) == 2
    assert all("c2kv_oracle" not in call["extra_body"] for call in calls)
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["retry_count"] == 0
    assert row["events"][0]["status"] == "no_compressed_history"
    assert row["events"][0]["eligibility"]["eligible"] is False


def test_v3_unsupported_gold_keeps_full_task_without_retry(monkeypatch, tmp_path):
    handler, calls, audit_path = _install_toy_handler(
        monkeypatch,
        tmp_path,
        "witness",
        lambda kwargs: _response(
            "bad0" if kwargs["extra_body"]["c2kv_eval_context"]["user_turn"] == 0
            else "good1",
            gist_tokens=7,
        ),
        max_events=4,
    )
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0(x=make_value())"], ["good1()"]]

    result, _ = handler.inference(_entry(), False, True)

    assert len(result) == 2
    assert len(calls) == 2
    assert all("c2kv_oracle" not in call["extra_body"] for call in calls)
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["retry_count"] == 0
    assert row["events"][0]["status"] == (
        "interface_unsupported_nonliteral_gold")


def test_v3_no_witness_retry_failure_does_not_retry_later_turn(
        monkeypatch, tmp_path):
    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        if context["user_turn"] == 0:
            return _response("good0")
        if context["user_turn"] == 1:
            return _response(
                "bad1",
                proxy_status=("no_literal_witness"
                              if context["attempt"] == 1 else None),
                gist_tokens=7,
            )
        assert "c2kv_oracle" not in kwargs["extra_body"]
        return _response("good2", gist_tokens=8)

    handler, calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose, max_events=4)
    handler._gold_controller.ground_truth_loader = lambda entry: [
        ["good0()"], ["good1()"], ["good2()"]]

    result, _ = handler.inference(_entry(turns=3), False, True)

    assert len(result) == 3
    assert len(calls) == 4
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(row["events"]) == 1
    assert row["events"][0]["extract_statuses"] == ["no_literal_witness"]
    assert row["events"][0]["status"] == "target_failed"


def test_v3_event_audit_serializes_checker_objects(monkeypatch, tmp_path):
    class Diagnostic:
        def __repr__(self):
            return "Diagnostic('v3')"

    def choose(kwargs):
        context = kwargs["extra_body"]["c2kv_eval_context"]
        if context["attempt"] == 0:
            return _response("bad0", gist_tokens=5)
        return _response("good0", proxy_status="appended", gist_tokens=5)

    handler, _calls, audit_path = _install_toy_handler(
        monkeypatch, tmp_path, "witness", choose, max_events=4)
    handler._gold_controller.ground_truth_loader = lambda entry: [["good0()"]]

    def checker(decoded, ground_truth, test_entry):
        outcome = _prefix_checker(decoded, ground_truth, test_entry)
        if not outcome["valid"]:
            outcome["multi_turn"]["diagnostic"] = Diagnostic()
        return outcome

    handler._gold_controller.prefix_checker = checker
    handler.inference(_entry(turns=1), False, True)

    row = json.loads(audit_path.read_text(encoding="utf-8"))
    assert row["events"][0]["check_error"]["multi_turn"]["diagnostic"] == {
        "type": f"{Diagnostic.__module__}.{Diagnostic.__qualname__}",
        "repr": "Diagnostic('v3')",
    }
