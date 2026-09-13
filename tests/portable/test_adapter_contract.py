"""The adapter contract (adapters/base.py) and the PINNED external commands.

Two things this file protects:

1. every adapter module really implements NAME / add_arguments / run(ctx),
   and ``v1`` / ``RunContext.opt`` behave as the adapters assume;
2. **the argv handed to each external harness is byte-identical to what the
   pre-registry code sent.**  Those command lines are the experiment: a
   changed flag silently redefines every number produced with it, so each is
   asserted against a literal list rather than against the code that built
   it.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c2kv_eval.portable import run
from c2kv_eval.portable.adapters import (
    acebench_adapter, acon_adapter, bfcl_adapter, base, tau2_adapter,
    toolsandbox_adapter,
)
from c2kv_eval.portable.adapters.base import RunContext, v1

MODULES = [tau2_adapter, bfcl_adapter, toolsandbox_adapter, acon_adapter,
           acebench_adapter]


# ---- contract ---------------------------------------------------------------

def test_every_adapter_module_implements_the_contract():
    for module in MODULES:
        assert isinstance(module.NAME, str) and module.NAME
        assert callable(module.add_arguments)
        assert callable(module.run)
        names = getattr(module, "NAMES", (module.NAME,))
        assert all(name in run.ADAPTERS for name in names), module.NAME


def test_v1_appends_exactly_one_suffix():
    assert v1("http://127.0.0.1:34100") == "http://127.0.0.1:34100/v1"
    assert v1("http://127.0.0.1:34100/") == "http://127.0.0.1:34100/v1"
    assert v1("http://h:1///") == "http://h:1/v1"


def _ctx(**options):
    return RunContext(base_url="http://p", user_base_url="http://u",
                      out_dir=Path("out"), model="m", arm="c2kv",
                      options=options)


def test_run_context_defaults_and_opt():
    ctx = _ctx(split="", task_ids=[], max_iter=None, num_workers=0, full=False,
               tag="mytag")
    assert ctx.run_name == "c2kv_run" and ctx.request_log is None
    # argparse's empty forms fall back to the adapter default...
    assert ctx.opt("split", "test") == "test"
    assert ctx.opt("task_ids") is None
    assert ctx.opt("max_iter", 30) == 30
    assert ctx.opt("missing", "d") == "d"
    # ...but 0 and False are values, not emptiness
    assert ctx.opt("num_workers", 4) == 0
    assert ctx.opt("full", True) is False
    assert ctx.opt("tag", "fallback") == "mytag"


def test_add_arguments_registers_only_that_adapters_flags():
    import argparse

    owned = {
        tau2_adapter: {"--tau2-dir", "--task-set", "--tau2-task-ids", "--tau2-num-trials",
                       "--tau2-max-steps", "--tau2-timeout"},
        bfcl_adapter: {
            "--bfcl-dir", "--categories", "--run-ids",
            "--bfcl-oracle-max-events"},
        toolsandbox_adapter: {"--full", "--ts-scenarios", "--ts-agent", "--ts-user", "--toolsandbox-dir"},
        acon_adapter: {"--acon-dir", "--split", "--tag", "--task-ids"},
        acebench_adapter: {"--acebench-dir", "--acebench-category",
                           "--acebench-language", "--acebench-task-ids", "--user-model"},
    }
    for module, flags in owned.items():
        parser = argparse.ArgumentParser()
        module.add_arguments(parser)
        registered = {a.option_strings[0] for a in parser._actions if a.option_strings}
        assert registered - {"-h"} == flags, module.NAME


# ---- tau2: `tau2.cli run` / `evaluate-trajs` --------------------------------

def test_tau2_run_command_is_byte_identical():
    cmd = tau2_adapter.run_command(
        "http://127.0.0.1:34100", "http://127.0.0.1:35000", "airline",
        "c2kv-agent", 4, "c2kv_run_ab12", python="/py")
    assert cmd == [
        "/py", "-m", "tau2.cli", "run",
        "--domain", "airline",
        "--task-set-name", "airline",
        "--agent-llm", "openai/c2kv-agent",
        "--agent-llm-args",
        '{"api_base": "http://127.0.0.1:34100/v1", "api_key": "EMPTY", "temperature": 0.0}',
        "--user-llm", "openai/c2kv-agent",
        "--user-llm-args",
        '{"api_base": "http://127.0.0.1:35000/v1", "api_key": "EMPTY", "temperature": 0.0}',
        "--max-concurrency", "4",
        "--save-to", "c2kv_run_ab12",
        "--auto-resume",
    ]


def test_tau2_run_command_domain_and_num_tasks():
    cmd = tau2_adapter.run_command("http://p/", "http://u/", "telecom_small",
                                   "m", 2, "r", max_tasks=5, python="/py")
    # --domain is the task set's first underscore-separated token
    assert cmd[cmd.index("--domain") + 1] == "telecom"
    assert cmd[cmd.index("--task-set-name") + 1] == "telecom_small"
    assert cmd[-2:] == ["--num-tasks", "5"]
    assert cmd[-3] == "--auto-resume"


def test_tau2_run_command_forwards_bounded_smoke_knobs():
    cmd = tau2_adapter.run_command("http://p", "http://u", "airline", "m", 1,
                                   "smoke", max_tasks=1, num_trials=1,
                                   max_steps=12, timeout=300, python="/py")
    for flag, value in (("--num-tasks", "1"), ("--num-trials", "1"),
                        ("--max-steps", "12"), ("--timeout", "300")):
        assert cmd[cmd.index(flag) + 1] == value


def test_tau2_run_command_forwards_exact_task_ids():
    cmd = tau2_adapter.run_command(
        "http://p", "http://u", "airline", "m", 1, "fixed",
        task_ids=["0", "3"], python="/py")
    index = cmd.index("--task-ids")
    assert cmd[index:index + 3] == ["--task-ids", "0", "3"]


def test_tau2_harness_env_prefers_selected_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/old/editable")
    env = tau2_adapter.harness_env(tmp_path)
    assert env["PYTHONPATH"].split(":") == [str(tmp_path.resolve() / "src"),
                                               "/old/editable"]


def test_tau2_evaluate_command_is_byte_identical(tmp_path):
    sims = tmp_path / "sims"
    cmd = tau2_adapter.evaluate_command(sims, python="/py")
    assert cmd == ["/py", "-m", "tau2.cli", "evaluate-trajs", "-o", str(sims),
                   str(sims / "results.json")]


def test_tau2_default_python_is_this_interpreter():
    assert tau2_adapter.run_command("http://p", "http://u", "airline", "m", 1,
                                    "r")[0] == sys.executable
    assert tau2_adapter.evaluate_command(Path("s"))[0] == sys.executable


# ---- bfcl: `bfcl generate` / `bfcl evaluate` --------------------------------

def test_bfcl_argv_is_byte_identical():
    assert bfcl_adapter.generate_argv("c2kv-full", "multi_turn_base") == [
        "generate", "--model", "c2kv-full", "--test-category", "multi_turn_base",
        "--num-threads", "1"]
    assert bfcl_adapter.evaluate_argv("c2kv-full", "multi_turn_base") == [
        "evaluate", "--model", "c2kv-full", "--test-category", "multi_turn_base"]


def test_bfcl_subset_argv_uses_run_ids_then_partial_eval():
    ids = ["multi_turn_base_0", "multi_turn_base_3"]
    # generate takes the BOOLEAN --run-ids (the ids go in
    # test_case_ids_to_generate.json); evaluate takes --partial-eval instead
    assert bfcl_adapter.generate_argv("c2kv-full", "memory", ids) == [
        "generate", "--model", "c2kv-full", "--test-category", "memory",
        "--num-threads", "1", "--run-ids"]
    assert bfcl_adapter.evaluate_argv("c2kv-full", "memory", ids) == [
        "evaluate", "--model", "c2kv-full", "--test-category", "memory",
        "--partial-eval"]


def test_bfcl_generate_argv_forwards_worker_count_to_pinned_flag():
    assert bfcl_adapter.generate_argv(
        "c2kv-full", "multi_turn_base", num_threads=4)[-2:] == [
            "--num-threads", "4"]
    with pytest.raises(ValueError, match="must be positive"):
        bfcl_adapter.generate_argv(
            "c2kv-full", "multi_turn_base", num_threads=0)


def test_bfcl_handler_key_dashes_the_arm():
    assert bfcl_adapter.handler_key("c2kv") == "c2kv-c2kv"
    assert bfcl_adapter.handler_key("c2kv_repair_tail") == "c2kv-c2kv-repair-tail"
    assert bfcl_adapter.handler_key("") == "c2kv-full"


# ---- toolsandbox: `tool_sandbox` -------------------------------------------

def test_toolsandbox_command_is_byte_identical(tmp_path):
    assert toolsandbox_adapter.cli_command(tmp_path) == [
        "tool_sandbox", "--user", "GPT_4_o_2024_05_13",
        "--agent", "GPT_4_o_2024_05_13", "-o", str(tmp_path), "-t"]
    assert toolsandbox_adapter.cli_command(tmp_path, test_mode=False) == [
        "tool_sandbox", "--user", "GPT_4_o_2024_05_13",
        "--agent", "GPT_4_o_2024_05_13", "-o", str(tmp_path)]


def test_toolsandbox_subset_command_overrides_test_mode(tmp_path):
    cmd = toolsandbox_adapter.cli_command(tmp_path, scenarios=["a", "b"],
                                          parallel="4")
    assert cmd == ["tool_sandbox", "--user", "GPT_4_o_2024_05_13",
                   "--agent", "GPT_4_o_2024_05_13", "-o", str(tmp_path),
                   "-s", "a", "b", "-p", "4"]
    assert "-t" not in cmd  # a subset run is never also test mode


def test_toolsandbox_env_splits_agent_and_user():
    env = toolsandbox_adapter.harness_env("http://127.0.0.1:34100",
                                          "http://127.0.0.1:35000")
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:34100/v1"
    assert env["TOOLSANDBOX_USER_BASE_URL"] == "http://127.0.0.1:35000/v1"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


def test_toolsandbox_selected_scenarios_bind_exact_terminal_denominator(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import os

    selected = tmp_path / "selected-source"
    selected.mkdir()
    python = tmp_path / "selected-env" / "bin" / "python"
    seen = {}
    monkeypatch.setattr(toolsandbox_adapter.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0))
    monkeypatch.setattr(toolsandbox_adapter, "collect", lambda out, **kwargs:
                        seen.update(kwargs) or {"n": 2})
    toolsandbox_adapter.run_ts("http://agent", tmp_path / "out", scenarios=["a", "b"],
                               expected=2, expected_task_ids=["a", "b"],
                               benchmark_dir=selected, python=str(python))
    assert seen["expected_task_ids"] == ["a", "b"]


def test_toolsandbox_collect_rejects_a_missing_selected_scenario(tmp_path):
    summary = tmp_path / "agent_mock" / "result_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"per_scenario_results": [{
        "name": "present", "traceback": None, "similarity": 1.0,
        "milestone_similarity": 1.0, "minefield_similarity": 0.0,
        "turn_count": 1,
    }]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing=absent"):
        toolsandbox_adapter.collect(tmp_path, expected_task_ids=["present", "absent"])


def test_toolsandbox_collect_rejects_an_unexpected_scenario(tmp_path):
    summary = tmp_path / "agent_mock" / "result_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"per_scenario_results": [{
        "name": "unexpected", "traceback": None, "similarity": 1.0,
        "milestone_similarity": 1.0, "minefield_similarity": 0.0,
        "turn_count": 1,
    }]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="unexpected=unexpected"):
        toolsandbox_adapter.collect(tmp_path, expected_task_ids=["present"])


def test_toolsandbox_collect_preserves_native_metrics(tmp_path):
    path = tmp_path / "agent_mock" / "result_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"per_scenario_results": [{
        "name": "a", "traceback": None, "similarity": 0.75,
        "milestone_similarity": 0.5, "minefield_similarity": 1.0,
        "turn_count": 4,
    }]}), encoding="utf-8")
    trajectory = path.parent / "trajectories" / "a"
    trajectory.mkdir(parents=True)
    (trajectory / "execution_context.json").write_text(json.dumps({
        "_dbs": {"SANDBOX": [
            {"sandbox_message_index": 1, "conversation_active": True},
            {"sandbox_message_index": 2, "sender": "EXECUTION_ENVIRONMENT",
             "recipient": "AGENT", "openai_tool_call_id": "call_ok",
             "conversation_active": False, "tool_trace": ["ok"],
             "tool_call_exception": None},
        ]}
    }), encoding="utf-8")
    summary = toolsandbox_adapter.collect(tmp_path, expected_task_ids=["a"])
    assert summary["milestone_similarity_mean"] == 0.5
    assert summary["minefield_similarity_mean"] == 1.0
    assert summary["turn_count_mean"] == 4.0
    assert summary["task_rows"][0]["task_id"] == "a"
    assert summary["task_rows"][0]["normal_termination"] is True
    assert summary["normal_termination_rate"] == 1.0
    assert summary["premature_termination_count"] == 0
    assert summary["tool_execution_count"] == 1
    assert summary["tool_execution_failure_count"] == 0
    assert summary["tool_execution_success_rate"] == 1.0


def test_toolsandbox_tool_failures_without_trace_are_counted(tmp_path):
    path = tmp_path / "agent_mock" / "result_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"per_scenario_results": [{
        "name": "a", "traceback": None, "similarity": 0.5,
        "milestone_similarity": 0.5, "minefield_similarity": 0.0,
        "turn_count": 2,
    }]}), encoding="utf-8")
    trajectory = path.parent / "trajectories" / "a"
    trajectory.mkdir(parents=True)
    (trajectory / "execution_context.json").write_text(json.dumps({
        "_dbs": {"SANDBOX": [
            {"sandbox_message_index": 1, "sender": "EXECUTION_ENVIRONMENT",
             "recipient": "AGENT", "openai_tool_call_id": "call_failed",
             "conversation_active": True, "tool_trace": None,
             "tool_call_exception": "ConnectionError"},
            {"sandbox_message_index": 2, "sender": "EXECUTION_ENVIRONMENT",
             "recipient": "AGENT", "openai_tool_call_id": "call_ok",
             "conversation_active": False, "tool_trace": ["ok"],
             "tool_call_exception": None},
        ]}
    }), encoding="utf-8")
    summary = toolsandbox_adapter.collect(tmp_path, expected_task_ids=["a"])
    assert summary["tool_execution_count"] == 2
    assert summary["tool_execution_failure_count"] == 1
    assert summary["tool_execution_success_rate"] == 0.5


# ---- acon: `run.py` / `run_all.py` / `appworld evaluate` --------------------

def test_acon_qa_command_is_byte_identical():
    assert acon_adapter.qa_command("/py", "c2kv-agent", "r_ab12", "test", 30) == [
        "/py", "run.py", "--split", "test", "--model_name", "c2kv-agent",
        "--tag", "r_ab12", "--max_iter", "30",
        "--data_folder", "data/nq_multi_8"]


def test_acon_appworld_command_is_byte_identical():
    assert acon_adapter.appworld_command("/py", "c2kv-agent", "r_ab12",
                                         "test_normal", 50) == [
        "/py", "run_all.py", "--split", "test_normal",
        "--model_name", "c2kv-agent", "--tag", "r_ab12",
        "--max_iter", "50", "--seed", "42"]


def test_acon_appworld_evaluate_command_is_byte_identical():
    assert acon_adapter.appworld_evaluate_command(
        "/venv/bin/appworld", "org/model", "r_ab12", "test_normal") == [
        "/venv/bin/appworld", "evaluate", "org_model_r_ab12", "test_normal"]


# ---- acebench: `generate.py` / `eval_main.py` ------------------------------

def test_acebench_commands_are_byte_identical(tmp_path):
    assert acebench_adapter.generate_command(
        "/py", tmp_path, "c2kv-agent", "agent", "en", 4, 40, "c2kv-agent",
        0.0, 1.0, 1200) == [
        "/py", str(tmp_path / "generate.py"),
        "--model", "c2kv-agent", "--category", "agent", "--language", "en",
        "--num-threads", "4", "--max-dialog-turns", "40",
        "--user-model", "c2kv-agent",
        "--temperature", "0.0", "--top-p", "1.0", "--max-tokens", "1200"]
    assert acebench_adapter.eval_command(
        "/py", tmp_path, "c2kv-agent", "agent", "en") == [
        "/py", str(tmp_path / "eval_main.py"), "--model", "c2kv-agent",
        "--category", "agent", "--language", "en"]


def test_toolsandbox_uses_selected_environment_and_checkout(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import os
    monkeypatch.chdir(tmp_path)
    selected = tmp_path / "selected-source"
    selected.mkdir()
    python = tmp_path / "selected-env" / "bin" / "python"
    calls = []
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stale-source"))
    monkeypatch.setattr(toolsandbox_adapter.subprocess, "run", lambda cmd, **kw:
                        calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(toolsandbox_adapter, "collect", lambda out: {"n": 1})
    toolsandbox_adapter.run_ts("http://agent", Path("out"),
        benchmark_dir=selected, python=str(python), user_base_url="http://user")
    cmd, kwargs = calls[0]
    assert cmd[0] == str(python.parent / "tool_sandbox")
    assert cmd[cmd.index("-o") + 1] == str(tmp_path / "out")
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(selected.resolve())
    assert kwargs["env"]["TOOLSANDBOX_USER_BASE_URL"] == "http://user/v1"


# ---- cost-join declarations -------------------------------------------------

@pytest.mark.parametrize("module", [tau2_adapter,
                                    toolsandbox_adapter, acebench_adapter])
def test_unjoinable_adapters_declare_a_reason(module):
    assert module.COST_JOIN.startswith("not joinable: ")
    assert len(module.COST_JOIN) > len("not joinable: ")


def test_bfcl_declares_explicit_task_cost_join():
    assert "c2kv_eval_context.task_id" in bfcl_adapter.COST_JOIN
    assert "request_log_summary.task_costs" in bfcl_adapter.COST_JOIN


def test_base_module_is_importable_as_a_package_member():
    assert base.RunContext is RunContext


def test_adapter_modules_run_as_package_modules(tmp_path):
    """Installed adapters retain their standalone ``--help`` entry points."""
    import subprocess

    package_root = Path(run.__file__).resolve().parents[2]
    for module in MODULES:
        done = subprocess.run([sys.executable, "-m", module.__name__, "--help"],
                              capture_output=True, text=True, cwd=package_root)
        assert done.returncode == 0, (module.__name__, done.stderr[-400:])
