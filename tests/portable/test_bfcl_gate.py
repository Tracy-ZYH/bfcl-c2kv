"""BFCL terminal gate + expected-count derivation (offline, no bfcl_eval)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c2kv_eval.portable import terminal_check, textarms
from c2kv_eval.portable.adapters import bfcl_adapter


def test_embedded_cli_returns_to_run_evaluation(monkeypatch):
    typer = pytest.importorskip("typer")
    cli = typer.Typer()
    completed = []

    @cli.command()
    def generate():
        completed.append("generate")

    @cli.command()
    def evaluate():
        completed.append("evaluate")

    monkeypatch.setitem(sys.modules, "bfcl_eval.__main__", SimpleNamespace(cli=cli))
    original_argv = list(sys.argv)
    bfcl_adapter.run_cli(["generate"])
    bfcl_adapter.run_cli(["evaluate"])
    assert completed == ["generate", "evaluate"]
    assert sys.argv == original_argv


def test_hiagent_native_response_matches_bfcl_fc_handler_contract(monkeypatch):
    """A Subgoal content line coexists with a BFCL native function call."""
    mapping = {}

    class ModelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class OpenAICompletionsHandler:
        # Pinned bfcl_eval OpenAICompletionsHandler response contract. The
        # real package is available only in the server's benchmark venv.
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

    package = ModuleType("bfcl_eval")
    package.__path__ = []
    constants = ModuleType("bfcl_eval.constants")
    constants.__path__ = []
    model_config = ModuleType("bfcl_eval.constants.model_config")
    model_config.MODEL_CONFIG_MAPPING = mapping
    model_config.ModelConfig = ModelConfig
    handlers = ModuleType("bfcl_eval.model_handler")
    handlers.__path__ = []
    api_inference = ModuleType("bfcl_eval.model_handler.api_inference")
    api_inference.__path__ = []
    openai_completion = ModuleType(
        "bfcl_eval.model_handler.api_inference.openai_completion")
    openai_completion.OpenAICompletionsHandler = OpenAICompletionsHandler
    openai_module = ModuleType("openai")
    openai_module.OpenAI = object
    httpx_module = ModuleType("httpx")
    httpx_module.Timeout = lambda **kwargs: kwargs
    for name, module in {
        "bfcl_eval": package,
        "bfcl_eval.constants": constants,
        "bfcl_eval.constants.model_config": model_config,
        "bfcl_eval.model_handler": handlers,
        "bfcl_eval.model_handler.api_inference": api_inference,
        "bfcl_eval.model_handler.api_inference.openai_completion": openai_completion,
        "openai": openai_module,
        "httpx": httpx_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    bfcl_adapter.install_handler(
        "http://proxy/v1", handler_name="c2kv-hiagent-full")
    config = mapping["c2kv-hiagent-full"]
    handler = config.model_handler.__new__(config.model_handler)
    message = SimpleNamespace(
        content="Subgoal: inspect the workspace",
        tool_calls=[SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="ls", arguments='{"a": true}'))],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=5),
    )

    parsed = handler._parse_query_response_FC(response)

    assert config.is_fc_model is True
    assert parsed["model_responses"] == [{"ls": '{"a": true}'}]
    assert parsed["tool_call_ids"] == ["call-1"]
    assert parsed["model_responses_message_for_chat_history"] is message
    assert "Action:" not in textarms.HIAGENT_SUBGOAL_NOTE


def _results(root: Path, handler: str, family: str, category: str, ids):
    path = root / "result" / handler / family / f"BFCL_v4_{category}_result.json"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps({"id": i}) + "\n" for i in ids), encoding="utf-8")


def _score(root: Path, handler: str, family: str, category: str, header):
    path = root / "score" / handler / family / f"BFCL_v4_{category}_score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    return path


def test_check_bfcl_finds_non_base_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_check, "GORILLA", tmp_path)
    _results(tmp_path, "c2kv-hybrid", "memory", "memory", ["memory_0", "memory_1", "memory_2"])
    assert terminal_check.check_bfcl(3, "", handler="c2kv-hybrid", category="memory") == 0
    assert terminal_check.check_bfcl(4, "", handler="c2kv-hybrid", category="memory") == 1
    # the default category still resolves the multi_turn family
    _results(tmp_path, "c2kv-hybrid", "multi_turn", "multi_turn_base", ["multi_turn_base_0"])
    assert terminal_check.check_bfcl(1, "", handler="c2kv-hybrid") == 0
    # a category never generated is "artifacts not found", not "complete"
    assert terminal_check.check_bfcl(200, "", handler="c2kv-hybrid",
                                     category="multi_turn_long_context") == 2


def test_check_bfcl_id_exact_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(terminal_check, "GORILLA", tmp_path)
    _results(tmp_path, "c2kv-full", "memory", "memory", ["memory_0", "memory_2"])
    assert terminal_check.check_bfcl(None, "memory_0,memory_2", handler="c2kv-full",
                                     category="memory") == 0
    assert terminal_check.check_bfcl(None, "memory_0,memory_1", handler="c2kv-full",
                                     category="memory") == 1


def test_check_bfcl_explicit_root_ignores_stale_shared_results(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    isolated = tmp_path / "isolated"
    monkeypatch.setattr(terminal_check, "GORILLA", shared)
    _results(shared, "c2kv-full", "memory", "memory", ["memory_0"])
    _results(isolated, "c2kv-full", "memory", "memory", ["memory_1"])

    assert terminal_check.check_bfcl(
        None, "memory_1", handler="c2kv-full", category="memory",
        root=isolated,
    ) == 0
    assert terminal_check.check_bfcl(
        None, "memory_0", handler="c2kv-full", category="memory",
        root=isolated,
    ) == 1


def test_run_bfcl_sets_official_root_before_import_and_writes_ids_there(
        tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    isolated = tmp_path / "isolated"
    shared.mkdir()
    monkeypatch.chdir(shared)
    monkeypatch.delenv("BFCL_PROJECT_ROOT", raising=False)
    seen = {"argv": []}

    def install(base_url, model, handler_name, **kwargs):
        seen["root_at_import"] = os.environ.get("BFCL_PROJECT_ROOT")
        seen["install_kwargs"] = kwargs

    def check(expected, run_ids, **kwargs):
        seen["check"] = (expected, run_ids, kwargs)
        return 0

    monkeypatch.setattr(bfcl_adapter, "install_handler", install)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_counts", lambda category: {category: 99})
    monkeypatch.setattr(bfcl_adapter, "run_cli", seen["argv"].append)
    monkeypatch.setattr(terminal_check, "check_bfcl", check)
    monkeypatch.setattr(
        bfcl_adapter, "collect_score_summary",
        lambda root, handler, counts: {
            "n": sum(counts.values()), "n_total": sum(counts.values()),
            "n_scored": sum(counts.values()), "correct_count": 1,
            "semantic_score": 1.0, "official_score_headers": [], "scored": True,
        })

    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", categories="multi_turn_base",
        run_ids=["multi_turn_base_7"],
        handler_name="c2kv-full", project_root=isolated,
    )

    assert seen["root_at_import"] == str(isolated.resolve())
    assert not (shared / "test_case_ids_to_generate.json").exists()
    assert json.loads((isolated / "test_case_ids_to_generate.json").read_text()) == {
        "multi_turn_base": ["multi_turn_base_7"]}
    assert seen["check"][2]["root"] == isolated.resolve()
    assert summary["n"] == summary["n_scored"] == 1
    assert summary["semantic_score"] == 1.0
    assert summary["bfcl_project_root"] == str(isolated.resolve())
    assert summary["bfcl_oracle_max_events"] == 1
    assert seen["install_kwargs"]["gold_recovery"] is None
    assert seen["install_kwargs"]["bfcl_oracle_max_events"] == 1
    assert seen["install_kwargs"]["task_audit_path"] == (
        isolated.resolve() / "task_audit" / "c2kv-full.jsonl")
    assert summary["task_telemetry_path"] == str(
        isolated.resolve() / "task_audit" / "c2kv-full.jsonl")
    assert "BFCL_PROJECT_ROOT" not in os.environ


def test_expected_count_uses_official_collection_mapping(monkeypatch):
    package = ModuleType("bfcl_eval")
    package.__path__ = []
    utils = ModuleType("bfcl_eval.utils")
    utils.parse_test_category_argument = lambda categories: ["part_a", "part_b"]
    utils.load_dataset_entry = lambda category, **kwargs: [
        {"id": f"{category}_{index}"}
        for index in range(2 if category == "part_a" else 3)
    ]
    monkeypatch.setitem(sys.modules, "bfcl_eval", package)
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", utils)

    assert bfcl_adapter.official_category_counts("collection") == {
        "part_a": 2, "part_b": 3}
    assert bfcl_adapter.expected_count("collection") == 5


def test_non_scoring_category_is_rejected(monkeypatch):
    package = ModuleType("bfcl_eval")
    package.__path__ = []
    utils = ModuleType("bfcl_eval.utils")
    utils.parse_test_category_argument = lambda categories: ["format_sensitivity"]
    utils.load_dataset_entry = lambda category, **kwargs: [{"id": "unused"}]
    monkeypatch.setitem(sys.modules, "bfcl_eval", package)
    monkeypatch.setitem(sys.modules, "bfcl_eval.utils", utils)

    with pytest.raises(ValueError, match="non-scoring"):
        bfcl_adapter.expected_count("format_sensitivity")


def test_collect_score_summary_reads_and_aggregates_official_headers(tmp_path):
    first = _score(tmp_path, "c2kv-full", "multi_turn", "part_a", {
        "accuracy": 0.5, "correct_count": 1, "total_count": 2})
    second = _score(tmp_path, "c2kv-full", "multi_turn", "part_b", {
        "accuracy": 1.0, "correct_count": 1, "total_count": 1})

    summary = bfcl_adapter.collect_score_summary(
        tmp_path, "c2kv-full", {"part_a": 2, "part_b": 1})

    assert summary["n"] == summary["n_scored"] == 3
    assert summary["correct_count"] == 2
    assert summary["semantic_score"] == pytest.approx(2 / 3)
    assert [row["path"] for row in summary["official_score_headers"]] == [
        str(first), str(second)]


def test_collect_score_summary_rejects_missing_or_wrong_denominator(tmp_path):
    with pytest.raises(RuntimeError, match="found 0"):
        bfcl_adapter.collect_score_summary(
            tmp_path, "c2kv-full", {"multi_turn_base": 1})

    _score(tmp_path, "c2kv-full", "multi_turn", "multi_turn_base", {
        "accuracy": 0.5, "correct_count": 1, "total_count": 2})
    with pytest.raises(RuntimeError, match="denominator mismatch"):
        bfcl_adapter.collect_score_summary(
            tmp_path, "c2kv-full", {"multi_turn_base": 1})


def test_collect_score_summary_rejects_inconsistent_accuracy(tmp_path):
    _score(tmp_path, "c2kv-full", "multi_turn", "multi_turn_base", {
        "accuracy": 1.0, "correct_count": 0, "total_count": 1})
    with pytest.raises(RuntimeError, match="accuracy mismatch"):
        bfcl_adapter.collect_score_summary(
            tmp_path, "c2kv-full", {"multi_turn_base": 1})


def test_generate_mode_does_not_claim_scored_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(bfcl_adapter, "install_handler", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bfcl_adapter, "official_category_counts",
        lambda category: {"multi_turn_base": 1})
    argv = []
    monkeypatch.setattr(bfcl_adapter, "run_cli", argv.append)
    monkeypatch.setattr(terminal_check, "check_bfcl", lambda *args, **kwargs: 0)

    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1", mode="generate", project_root=tmp_path)

    assert summary["n_generated"] == summary["n_total"] == 1
    assert summary["scored"] is False
    assert "n" not in summary and "n_scored" not in summary
    assert "semantic_score" not in summary
    assert argv == [bfcl_adapter.generate_argv("c2kv-hf", "multi_turn_base")]


def test_run_bfcl_passes_opt_in_v3_budget_to_handler(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        bfcl_adapter, "install_handler",
        lambda *args, **kwargs: seen.update(kwargs),
    )
    monkeypatch.setattr(
        bfcl_adapter, "official_category_counts",
        lambda category: {"multi_turn_base": 1},
    )
    monkeypatch.setattr(bfcl_adapter, "run_cli", lambda argv: None)
    monkeypatch.setattr(terminal_check, "check_bfcl", lambda *args, **kwargs: 0)

    summary = bfcl_adapter.run_bfcl(
        "http://proxy/v1",
        mode="generate",
        project_root=tmp_path,
        gold_recovery="witness",
        bfcl_oracle_max_events=4,
    )

    assert seen["bfcl_oracle_max_events"] == 4
    assert seen["gold_recovery"] == "witness"
    assert summary["bfcl_oracle_kind"] == "bfcl_gold_turn_v3"
    assert summary["bfcl_oracle_version"] == 3
    assert summary["bfcl_oracle_variant"] == "event_budget_sensitivity"


def test_run_bfcl_rejects_multi_event_budget_without_gold_arm(tmp_path):
    with pytest.raises(ValueError, match="gold-recovery arm"):
        bfcl_adapter.run_bfcl(
            "http://proxy/v1",
            mode="generate",
            project_root=tmp_path,
            bfcl_oracle_max_events=4,
        )
