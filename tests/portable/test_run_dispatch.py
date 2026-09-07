"""run.py registry dispatch + the CLI surface the server scripts call.

Two jobs:

1. **CLI compatibility.** run.py is driven by CLI only, so every flag's
   name, default and type is pinned here as a table.  A flag that moved from
   run.py into an adapter's ``add_arguments`` must still parse identically.
2. **RunContext plumbing.** The knobs must reach the adapters under the
   right names with the right defaults — the likeliest place for a silent
   mis-wiring (and the adapters, not run.py, now own "/v1" and cwd).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c2kv_eval.portable import run
from c2kv_eval.portable.adapters import (
    acebench_adapter, acon_adapter, bfcl_adapter, tau2_adapter,
    toolsandbox_adapter,
)
from c2kv_eval.portable.adapters.base import RunContext

# (flag, default, required) exactly as the pre-registry run.py accepted it.
# Server scripts quote these; changing one silently re-points a run.
CLI_SURFACE = [
    ("--benchmark", None, True),
    ("--arm", None, True),
    ("--recovery-control", "", False),
    ("--upstream", None, True),
    ("--user-upstream", "", False),
    ("--proxy-port", 34100, False),
    ("--proxy-python", None, False),
    ("--out", None, True),
    ("--task-set", "airline", False),
    ("--tau2-num-trials", None, False),
    ("--tau2-max-steps", None, False),
    ("--tau2-timeout", None, False),
    ("--categories", "multi_turn_base", False),
    ("--run-ids", "", False),
    ("--bfcl-oracle-max-events", 1, False),
    ("--num-workers", 4, False),
    ("--max-tasks", None, False),
    ("--run-name", "c2kv_run", False),
    ("--full", False, False),
    ("--record-reference", "", False),
    ("--reference", "", False),
    ("--backend", "sglang", False),
    ("--model", "c2kv-agent", False),
    ("--ts-scenarios", "", False),
    ("--ts-agent", "", False),
    ("--ts-user", "", False),
    ("--toolsandbox-dir", None, False),
    ("--doc-packing", None, False),
    ("--max-doc-length", None, False),
    ("--max-doc-num", None, False),
    ("--checkpoint", None, False),
    ("--tokenizer", None, False),
    ("--checkpoint-profile", None, False),
    ("--expected-profile-fingerprint", None, False),
    ("--reference-profile", None, False),
    ("--allow-unprofiled", False, False),
    ("--query-projection", None, False),
    ("--capability-features", "", False),
    ("--exact-out", False, False),
    ("--acon-dir", None, False),
    ("--acebench-dir", None, False),
    ("--bfcl-dir", None, False),
    ("--tau2-dir", None, False),
    ("--split", "", False),
    ("--tag", "", False),
    ("--max-iter", None, False),
    ("--task-ids", "", False),
    ("--acebench-category", "agent", False),
    ("--acebench-language", "en", False),
    ("--acebench-task-ids", "", False),
    ("--user-model", "", False),
    ("--bench-python", "", False),
]


def _actions():
    return {a.option_strings[0]: a for a in run.build_parser()._actions
            if a.option_strings}


def test_cli_surface_is_unchanged():
    actions = _actions()
    for flag, default, required in CLI_SURFACE:
        assert flag in actions, f"{flag} disappeared from run.py's CLI"
        assert actions[flag].default == default, flag
        assert actions[flag].required is required, flag
    # no flag was invented either (-h aside)
    assert set(actions) - {"-h"} == {flag for flag, _, _ in CLI_SURFACE}


def test_cli_choices_and_types_are_unchanged():
    actions = _actions()
    assert actions["--benchmark"].choices == list(run.ADAPTERS)
    assert set(actions["--benchmark"].choices) == {
        "tau2", "bfcl", "toolsandbox", "acon_appworld", "acon_qa", "acebench"}
    assert actions["--backend"].choices == ["sglang"]
    assert actions["--doc-packing"].choices == ["turn", "message"]
    assert actions["--acebench-language"].choices == ["en", "zh"]
    for flag in ("--out", "--acon-dir", "--acebench-dir", "--bfcl-dir",
                 "--tau2-dir", "--checkpoint", "--tokenizer"):
        assert actions[flag].type is Path, flag
    for flag in ("--proxy-port", "--num-workers", "--max-tasks", "--max-iter",
                 "--max-doc-length", "--max-doc-num"):
        assert actions[flag].type is int, flag
    assert actions["--bfcl-oracle-max-events"].type is bfcl_adapter._positive_int


def test_bfcl_event_budget_cli_requires_a_positive_integer():
    with pytest.raises(SystemExit):
        _args(["--benchmark", "bfcl", *BASE_ARGV,
               "--bfcl-oracle-max-events", "0"])


def test_registry_covers_every_adapter_module():
    assert run.ADAPTERS["tau2"] is tau2_adapter
    assert run.ADAPTERS["bfcl"] is bfcl_adapter
    assert run.ADAPTERS["toolsandbox"] is toolsandbox_adapter
    assert run.ADAPTERS["acebench"] is acebench_adapter
    # one module, two --benchmark names
    assert run.ADAPTERS["acon_qa"] is acon_adapter
    assert run.ADAPTERS["acon_appworld"] is acon_adapter


# ---- RunContext plumbing ----------------------------------------------------

def _args(argv):
    return run.build_parser().parse_args(argv)


BASE_ARGV = ["--arm", "c2kv", "--upstream", "http://up:35000", "--out", "o"]


def test_build_context_defaults(tmp_path):
    args = _args(["--benchmark", "tau2", *BASE_ARGV])
    args.out = tmp_path
    ctx = run.build_context(args, tmp_path / "proxy.jsonl")
    # the proxy URL is handed over BARE: adapters own the "/v1"
    assert ctx.base_url == "http://127.0.0.1:34100"
    assert ctx.user_base_url == "http://up:35000"  # --user-upstream defaults
    assert ctx.model == "c2kv-agent" and ctx.arm == "c2kv"
    assert ctx.out_dir == tmp_path
    assert ctx.request_log == tmp_path / "proxy.jsonl"
    assert ctx.options["benchmark"] == "tau2"
    assert ctx.options["task_set"] == "airline"


def test_g_profile_drives_query_and_document_geometry(tmp_path):
    checkpoint = tmp_path / "g_hist_s43" / "checkpoint-588"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "gist_param": "qkv", "gist_type": "dynamic-interleave",
        "gist_overlap": 64, "gist_residual_type": "embed-mean"}), encoding="utf-8")
    (checkpoint.parent / "run_config.json").write_text(json.dumps({
        "doc_mode": "history_only", "tools_in_system": True,
        "max_doc_length": 768, "max_doc_num": 16, "ratios": "8,8,4,16"}), encoding="utf-8")
    (checkpoint.parent / "train_manifest_used.json").write_text(json.dumps({
        "doc_mode": "history_only", "tools_in_system": True}), encoding="utf-8")
    args = _args(["--benchmark", "tau2", *BASE_ARGV, "--checkpoint", str(checkpoint),
                  "--query-projection", "gist"])
    profile = run.resolve_run_profile(args)
    assert (args.doc_packing, args.max_doc_length, args.max_doc_num, args.query_projection) == (
        "turn", 768, 16, "gist")
    assert profile["profile_kind"] == "legacy_artifacts"
    args.expected_profile_fingerprint = "outdated-plan"
    with pytest.raises(run.ProfileError, match="planned profile fingerprint"):
        run.resolve_run_profile(args)
    args.expected_profile_fingerprint = profile["profile_fingerprint"]
    args.max_doc_length = 512
    with pytest.raises(run.ProfileError, match="conflicts with checkpoint profile"):
        run.resolve_run_profile(args)


def test_run_requires_explicit_legacy_opt_in():
    args = _args(["--benchmark", "tau2", *BASE_ARGV])
    with pytest.raises(run.ProfileError, match="allow-unprofiled"):
        run.resolve_run_profile(args)
    args.allow_unprofiled = True
    assert run.resolve_run_profile(args)["profile_kind"] == "unprofiled"
    assert (args.max_doc_length, args.max_doc_num) == (512, 12)


def test_build_context_user_upstream_split():
    args = _args(["--benchmark", "acebench", "--user-upstream", "http://raw:35000",
                  *BASE_ARGV])
    ctx = run.build_context(args, None)
    assert ctx.user_base_url == "http://raw:35000"
    assert ctx.request_log is None


def test_bfcl_subset_cli_reaches_context(tmp_path):
    args = _args(["--benchmark", "bfcl", "--model", "Qwen/Frozen-FC",
                  "--run-ids", "multi_turn_base_1", *BASE_ARGV])
    args.out = tmp_path
    ctx = run.build_context(args, None)
    assert ctx.model == "Qwen/Frozen-FC"
    assert ctx.options["run_ids"] == "multi_turn_base_1"


def _ctx(benchmark, **options):
    base = {"benchmark": benchmark}
    base.update(options)
    return RunContext(base_url="http://127.0.0.1:34100",
                      user_base_url="http://raw:35000", out_dir=Path("out"),
                      model="c2kv-agent", arm="c2kv", run_name="r_ab12",
                      options=base)


def _capture(monkeypatch, module, name):
    calls = {}

    def fake(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"n": 0}

    monkeypatch.setattr(module, name, fake)
    return calls


def test_acon_qa_dispatch_defaults(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, acon_adapter, "run_qa")
    ctx = _ctx("acon_qa", tag="", split="", max_iter=None, task_ids="",
               max_tasks=5, acon_dir=None, bench_python="")
    ctx.out_dir = tmp_path
    acon_adapter.run(ctx)
    assert calls["args"] == ("http://127.0.0.1:34100", tmp_path)
    kw = calls["kwargs"]
    assert kw["split"] == acon_adapter.QA_DEFAULT_SPLIT
    assert kw["max_iter"] == acon_adapter.QA_DEFAULT_MAX_ITER
    assert kw["limit"] == 5 and kw["tag"] == "r_ab12" and kw["model"] == "c2kv-agent"
    assert kw["task_ids"] is None and kw["python"] is None
    assert kw["request_log"] is None


def test_acon_appworld_dispatch_overrides(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, acon_adapter, "run_appworld")
    ctx = _ctx("acon_appworld", tag="mytag", split="dev", max_iter=20,
               task_ids="t1", acon_dir=Path("/acon"),
               bench_python="/venv/bin/python")
    ctx.model = "m"
    ctx.request_log = tmp_path / "proxy.jsonl"
    acon_adapter.run(ctx)
    kw = calls["kwargs"]
    assert kw["split"] == "dev" and kw["max_iter"] == 20
    assert kw["tag"] == "mytag" and kw["task_ids"] == ["t1"]
    assert kw["acon_dir"] == Path("/acon") and kw["python"] == "/venv/bin/python"
    assert kw["request_log"] == tmp_path / "proxy.jsonl"
    assert "limit" not in kw  # run_all.py has no --limit


def test_acebench_dispatch_routes_user_simulator_to_raw_upstream(monkeypatch):
    calls = _capture(monkeypatch, acebench_adapter, "run_acebench")
    ctx = _ctx("acebench", num_workers=4, max_iter=None, acebench_category="",
               acebench_language="", user_model="", acebench_dir=None,
               bench_python="")
    ctx.arm = "full"
    summary = acebench_adapter.run(ctx)
    assert calls["args"] == ("http://127.0.0.1:34100", "http://raw:35000",
                             Path("out"))
    kw = calls["kwargs"]
    assert kw["category"] == acebench_adapter.DEFAULT_CATEGORY
    assert kw["language"] == acebench_adapter.DEFAULT_LANGUAGE
    assert kw["user_model"] is None and kw["num_threads"] == 4
    assert kw["max_dialog_turns"] == acebench_adapter.DEFAULT_MAX_DIALOG_TURNS
    assert summary["cost_join"].startswith("not joinable:")


def test_tau2_dispatch_passes_run_name_and_workers(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.setdefault("cmds", []).append(cmd)

        class _P:
            returncode = 0

        return _P()

    monkeypatch.setattr(tau2_adapter.subprocess, "run", fake_run)
    monkeypatch.setattr(tau2_adapter, "collect", lambda *a, **k: {"n": 1})
    sims = tmp_path / "data" / "simulations" / "r_ab12"
    sims.mkdir(parents=True)
    (sims / "updated_results.json").write_text("{}", encoding="utf-8")
    from c2kv_eval.portable import terminal_check

    def fake_check_tau2(*args, **kwargs):
        seen["terminal_args"] = args
        seen["terminal_kwargs"] = kwargs
        return 0

    monkeypatch.setattr(terminal_check, "check_tau2", fake_check_tau2)
    ctx = _ctx("tau2", task_set="mock_domain", num_workers=7, max_tasks=3,
               tau2_dir=tmp_path)
    summary = tau2_adapter.run(ctx)
    run_cmd = seen["cmds"][0]
    assert run_cmd[run_cmd.index("--max-concurrency") + 1] == "7"
    assert run_cmd[run_cmd.index("--save-to") + 1] == "r_ab12"
    assert run_cmd[run_cmd.index("--num-tasks") + 1] == "3"
    assert seen["terminal_kwargs"]["root"] == tmp_path / "data" / "simulations"
    assert summary["cost_join"].startswith("not joinable:")


def test_toolsandbox_dispatch_splits_scenarios(monkeypatch):
    calls = _capture(monkeypatch, toolsandbox_adapter, "run_ts")
    ctx = _ctx("toolsandbox", full=False, ts_agent="", ts_user="",
               ts_scenarios=" a , b ")
    summary = toolsandbox_adapter.run(ctx)
    kw = calls["kwargs"]
    assert kw["scenarios"] == ["a", "b"]
    assert kw["expected"] == 2
    assert kw["expected_task_ids"] == ["a", "b"]
    assert kw["agent"] == toolsandbox_adapter.AGENT
    assert kw["user"] == toolsandbox_adapter.AGENT
    assert kw["user_base_url"] == "http://raw:35000"
    assert kw["test_mode"] is True
    assert summary["cost_join"].startswith("not joinable:")


def test_bfcl_dispatch_adds_v1_and_chdirs(monkeypatch, tmp_path):
    calls = _capture(monkeypatch, bfcl_adapter, "run_bfcl")
    seen = {}
    monkeypatch.setenv("BFCL_PROJECT_ROOT", "previous-root")
    monkeypatch.setattr(bfcl_adapter.os, "chdir",
                        lambda path: seen.setdefault("cwds", []).append(str(path)))
    ctx = _ctx("bfcl", categories="memory", bfcl_dir=str(tmp_path),
               run_ids="memory_1", num_workers=4, bfcl_oracle_max_events=3)
    ctx.arm = "c2kv4_gold_witness"
    summary = bfcl_adapter.run(ctx)
    # the handler needs /v1; run.py hands over the bare URL
    assert calls["args"] == ("http://127.0.0.1:34100/v1",)
    assert calls["kwargs"]["categories"] == "memory"
    assert calls["kwargs"]["run_ids"] == "memory_1"
    assert calls["kwargs"]["project_root"] == ctx.out_dir.resolve()
    assert calls["kwargs"]["num_threads"] == 4
    assert calls["kwargs"]["bfcl_oracle_max_events"] == 3
    # underscores in an arm name would corrupt the result dir path
    assert calls["kwargs"]["handler_name"] == "c2kv-c2kv4-gold-witness"
    assert seen["cwds"][0] == str(tmp_path)
    assert len(seen["cwds"]) == 2  # chdir in, chdir back (try/finally)
    assert bfcl_adapter.os.environ["BFCL_PROJECT_ROOT"] == "previous-root"
    assert "request_log_summary.task_costs" in summary["cost_join"]


def test_h200_matrix_delegates_to_generic_runner_with_smoke_flags():
    pytest.skip("deployment matrix remains in the consuming repository")


def test_cli_accepts_new_benchmarks():
    parser_choices = None

    class _Stop(Exception):
        pass

    def fake_parse(self, argv=None):  # capture choices without running
        nonlocal parser_choices
        for action in self._actions:
            if action.dest == "benchmark":
                parser_choices = list(action.choices)
        raise _Stop

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(argparse.ArgumentParser, "parse_args", fake_parse)
        with pytest.raises(_Stop):
            run.main(["--benchmark", "acon_qa", "--arm", "full", "--upstream", "x", "--out", "o"])
    assert {"acon_appworld", "acon_qa", "acebench"} <= set(parser_choices)


# ---- main(): proxy lifecycle + summary envelope -----------------------------

class _FakeProc:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _stub_run(monkeypatch, tmp_path, summary, arm="c2kv", extra_argv=()):
    """main() with the proxy stubbed out and one adapter replaced."""
    proc = _FakeProc()
    log = tmp_path / "proxy.jsonl"
    log.write_text(json.dumps({"status": "ok", "conv_id": "a", "wall_sec": 1.0,
                               "n_docs": 3, "dropped_docs": 0,
                               "c2kv_query_proj": "gist"}) + "\n",
                   encoding="utf-8")
    seen = {}

    def fake_start_proxy(upstream, arm_, port, log_dir, **kwargs):
        seen["start"] = {"upstream": upstream, "arm": arm_, "port": port,
                         "log_dir": log_dir, **kwargs}
        return proc, log

    monkeypatch.setattr(run, "start_proxy", fake_start_proxy)
    monkeypatch.setattr(run, "_git_short_sha", lambda: "ab12cd3")
    monkeypatch.setattr(run, "run_preflight", lambda *a, **k:
                        type("Ready", (), {"raise_for_errors": lambda self: None,
                                            "as_dict": lambda self: {"ok": True}})())
    monkeypatch.setitem(run.ADAPTERS, "tau2",
                        type("Stub", (), {
                            "NAME": "tau2",
                            "add_arguments": staticmethod(lambda p: None),
                            "run": staticmethod(
                                lambda ctx: (seen.update(ctx=ctx), dict(summary))[1]),
                        }))
    run.main(["--benchmark", "tau2", "--arm", arm, "--upstream", "http://up:35000",
              "--out", str(tmp_path / "outdir"), "--allow-unprofiled", *extra_argv])
    seen["proc"] = proc
    return seen


def test_main_writes_the_summary_envelope(monkeypatch, tmp_path):
    seen = _stub_run(monkeypatch, tmp_path, {"n": 4, "semantic_score": 0.5,
                                             "cost_join": "joined: 4/4 tasks"})
    out = tmp_path / "outdir_ab12cd3"
    written = json.loads((out / "summary_c2kv.json").read_text(encoding="utf-8"))
    assert written["arm"] == "c2kv" and written["benchmark"] == "tau2"
    assert written["backend"] == "sglang" and written["model"] == "c2kv-agent"
    assert written["doc_packing"] == "turn"
    assert written["max_doc_length"] == 512 and written["max_doc_num"] == 12
    assert written["request_log"].endswith("proxy.jsonl")
    assert written["request_log_summary"]["n_ok"] == 1
    assert written["semantic_score"] == 0.5  # the adapter's own keys survive
    assert written["cost_join"] == "joined: 4/4 tasks"
    assert "textarm_summary" not in written  # only for text arms
    assert seen["proc"].terminated is True


def test_main_sha_suffixes_run_name_and_out(monkeypatch, tmp_path):
    seen = _stub_run(monkeypatch, tmp_path, {"n": 0})
    ctx = seen["ctx"]
    assert ctx.run_name == "c2kv_run_ab12cd3"
    assert ctx.out_dir == tmp_path / "outdir_ab12cd3"
    assert ctx.out_dir.is_dir()
    # the adapter gets the proxy log for the cost join
    assert ctx.request_log == tmp_path / "proxy.jsonl"
    # the proxy really is started on the requested arm/upstream
    assert seen["start"]["upstream"] == "http://up:35000"
    assert seen["start"]["arm"] == "c2kv" and seen["start"]["port"] == 34100
    assert seen["start"]["doc_packing"] == "turn"


def test_main_text_arm_adds_textarm_summary(monkeypatch, tmp_path):
    _stub_run(monkeypatch, tmp_path, {"n": 0}, arm="hiagent")
    written = json.loads((tmp_path / "outdir_ab12cd3" / "summary_hiagent.json")
                         .read_text(encoding="utf-8"))
    assert written["textarm_summary"]["textarm_requests"] == 0


def test_main_records_reference_flags_in_the_summary(monkeypatch, tmp_path):
    _stub_run(monkeypatch, tmp_path, {"n": 0},
              extra_argv=["--record-reference", "ref.jsonl"])
    written = json.loads((tmp_path / "outdir_ab12cd3" / "summary_c2kv.json")
                         .read_text(encoding="utf-8"))
    assert written["record_reference"] == "ref.jsonl"
    assert "reference" not in written


def test_acon_adapter_refuses_an_unknown_benchmark_name():
    with pytest.raises(SystemExit, match="acon_adapter got --benchmark"):
        acon_adapter.run(_ctx("acon_typo"))
    assert acon_adapter.NAME_TO_KIND == {"acon_appworld": "appworld",
                                         "acon_qa": "qa"}
