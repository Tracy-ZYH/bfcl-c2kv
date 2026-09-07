"""Offline tests for adapters/acebench_adapter.py (no subprocess, no network).

Covers: category expansion from a checkout's category.py, the agent/user
endpoint split and role-history marker, the per-run working directory, the
terminal-state gate, and the score-file parsing for both failure-row
conventions (index-keyed agent rows, id-keyed normal rows) including the
turn-group clustering of normal_multi_turn_*.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2kv_eval.portable.adapters import acebench_adapter as B


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _checkout(tmp_path: Path) -> Path:
    root = tmp_path / "acebench"
    root.mkdir()
    (root / "category.py").write_text(
        "ACE_DATA_CATEGORY = {\n"
        "  'agent': ['agent_multi_step', 'agent_multi_turn'],\n"
        "  'multi_turn': ['normal_multi_turn_user_adjust', 'normal_multi_turn_user_switch'],\n"
        "}\n", encoding="utf-8")
    (root / "data_all" / "data_en").mkdir(parents=True)
    return root


# ---- categories / env / workdir ----------------------------------------------

def test_expand_categories_from_checkout(tmp_path):
    cat_map = B.load_category_map(_checkout(tmp_path))
    assert B.expand_categories("agent", cat_map) == ["agent_multi_step", "agent_multi_turn"]
    # a bare test name passes through (eval_main.py rule)
    assert B.expand_categories("agent_multi_turn", cat_map) == ["agent_multi_turn"]


def test_harness_env_splits_agent_and_user(monkeypatch):
    monkeypatch.setenv("GPT_BASE_URL", "https://api.openai.com/v1")  # must not matter
    env = B.harness_env("http://127.0.0.1:34100", "http://127.0.0.1:35000", "c2kv-agent")
    assert env[B.AGENT_BASE_URL_ENV] == "http://127.0.0.1:34100/v1"
    assert env[B.USER_BASE_URL_ENV] == "http://127.0.0.1:35000/v1"
    assert env[B.MODELS_ENV] == "c2kv-agent"
    assert env[B.ROLE_HISTORY_ENV] == "1"
    assert B.CAPABILITY_FEATURES == ("acebench_role_history_v1",)
    assert env[B.AGENT_API_KEY_ENV] == env[B.USER_API_KEY_ENV] == "EMPTY"
    # no separate user endpoint given: the simulator falls back to the SAME
    # url as the agent (standalone use), never to an OpenAI default
    assert B.harness_env("http://a", "", "m")[B.USER_BASE_URL_ENV] == "http://a/v1"


def test_vendored_patch_uses_structured_agent_history_only():
    patch = (Path(B.__file__).resolve().parents[1] / "patches" / "acebench" /
             "0001-endpoint-env-and-model-registry.patch").read_text(encoding="utf-8")
    assert "ACEBENCH_ROLE_HISTORY_V1" in patch
    assert "message_history if" in patch
    assert "model_inference/role_history.py" in patch
    assert 'roles = {"user": "user", "agent": "assistant", "execution": "tool"}' in patch
    assert "history.split" not in patch and "inference_data.split" not in patch
    # User-simulator endpoint routing is retained, while structured-history
    # conversion applies only to the evaluated agent and scorer code is absent.
    assert "ACEBENCH_USER_BASE_URL" in patch
    assert "eval_main.py" not in patch
    assert "not joinable" in B.COST_JOIN


def test_generate_and_eval_commands_pin_protocol_knobs(tmp_path):
    root = _checkout(tmp_path)
    gen = B.generate_command("py", root, "c2kv-agent", "agent", "en", 4, 40,
                             "c2kv-agent", 0.0, 1.0, 1200)
    assert gen[1] == str(root / "generate.py")
    assert gen[gen.index("--user-model") + 1] == "c2kv-agent"
    assert gen[gen.index("--temperature") + 1] == "0.0"
    assert gen[gen.index("--max-dialog-turns") + 1] == "40"
    ev = B.eval_command("py", root, "c2kv-agent", "agent", "en")
    assert ev[1] == str(root / "eval_main.py") and ev[-1] == "en"


def test_prepare_workdir_links_data_all_once(tmp_path):
    root = _checkout(tmp_path)
    (root / "data_all" / "data_en" / "data_x.json").write_text("{}\n")
    out = tmp_path / "out"
    work = B.prepare_workdir(out, root)
    assert work == out / "acebench_work"
    assert (work / "data_all" / "data_en" / "data_x.json").exists()
    assert B.prepare_workdir(out, root) == work  # idempotent


def test_prepare_workdir_materializes_exact_official_subset_with_provenance(tmp_path):
    root = _checkout(tmp_path)
    source = root / "data_all" / "data_en" / "data_agent_multi_step.json"
    _write_jsonl(source, [{"id": "agent_multi_step_0"}, {"id": "agent_multi_step_1"}])
    _write_jsonl(source.parent / "possible_answer" / source.name,
                 [{"id": "agent_multi_step_0", "ground_truth": []},
                  {"id": "agent_multi_step_1", "ground_truth": []}])
    work = B.prepare_workdir(tmp_path / "out", root, language="en",
                             tests=["agent_multi_step"], task_ids="agent_multi_step_1")
    selected = B._jsonl(work / "data_all" / "data_en" / "data_agent_multi_step.json")
    assert selected == [{"id": "agent_multi_step_1"}]
    assert B._jsonl(source) == [{"id": "agent_multi_step_0"}, {"id": "agent_multi_step_1"}]
    record = json.loads((work / "selected_tasks.json").read_text(encoding="utf-8"))
    assert record["sources"][0]["selected_ids"] == ["agent_multi_step_1"]
    assert record["sources"][0]["source_rows"] == 2
    assert B._jsonl(work / "data_all" / "data_en" / "possible_answer" /
                    "data_agent_multi_step.json") == [{"id": "agent_multi_step_1", "ground_truth": []}]


def test_prepare_workdir_max_tasks_selects_first_source_row(tmp_path):
    root = _checkout(tmp_path)
    _write_jsonl(root / "data_all" / "data_en" / "data_agent_multi_step.json",
                 [{"id": "agent_multi_step_0"}, {"id": "agent_multi_step_1"}])
    _write_jsonl(root / "data_all" / "data_en" / "possible_answer" /
                 "data_agent_multi_step.json",
                 [{"id": "agent_multi_step_0"}, {"id": "agent_multi_step_1"}])
    work = B.prepare_workdir(tmp_path / "out", root, language="en",
                             tests=["agent_multi_step"], max_tasks=1)
    assert B._jsonl(work / "data_all" / "data_en" / "data_agent_multi_step.json") == [
        {"id": "agent_multi_step_0"}]


def test_subset_harness_remaps_category_without_changing_upstream_checkout(tmp_path):
    root = _checkout(tmp_path)
    (root / "generate.py").write_text("# upstream generator\n", encoding="utf-8")
    source_category = (root / "category.py").read_text(encoding="utf-8")
    work = tmp_path / "out" / "acebench_work"
    work.mkdir(parents=True)
    harness = B.prepare_subset_harness(work, root, "agent", ["agent_multi_step"])
    assert (harness / "generate.py").read_text(encoding="utf-8") == "# upstream generator\n"
    assert "'agent': ['agent_multi_step']" in (harness / "category.py").read_text(encoding="utf-8")
    assert (root / "category.py").read_text(encoding="utf-8") == source_category


def test_prepare_score_dir_matches_official_model_normalization(tmp_path):
    path = B.prepare_score_dir(tmp_path, "en", "org/model")
    assert path == tmp_path / "score_all" / "score_en" / "org_model"
    assert path.is_dir()


# ---- terminal-state gate ----------------------------------------------------

def test_check_terminal_missing_ids_is_fatal(tmp_path):
    work = tmp_path / "w"
    _write_jsonl(work / "data_all" / "data_en" / "data_agent_multi_turn.json",
                 [{"id": f"agent_multi_turn_{i}"} for i in range(3)])
    _write_jsonl(work / "result_all" / "result_en" / "m" / "data_agent_multi_turn_result.json",
                 [{"id": "agent_multi_turn_0"}, {"id": "agent_multi_turn_2"}])
    with pytest.raises(SystemExit, match="agent_multi_turn_1"):
        B.check_terminal(work, "en", "m", ["agent_multi_turn"])
    # complete -> passes silently
    _write_jsonl(work / "result_all" / "result_en" / "m" / "data_agent_multi_turn_result.json",
                 [{"id": f"agent_multi_turn_{i}"} for i in range(3)])
    B.check_terminal(work, "en", "m", ["agent_multi_turn"])


# ---- score parsing ----------------------------------------------------------

def test_collect_agent_failures_are_index_keyed(tmp_path):
    work = tmp_path / "w"
    results = [{"id": f"agent_multi_turn_{i}", "result": [], "process": []} for i in range(4)]
    _write_jsonl(work / "result_all" / "result_en" / "m" / "data_agent_multi_turn_result.json", results)
    _write_jsonl(work / "score_all" / "score_en" / "m" / "data_agent_multi_turn_score.json", [
        {"end_to_end_accuracy": 0.5, "process_accuracy": 0.7, "correct_count": 2, "total_count": 4},
        {"id": 1, "valid": False, "error": ["x"], "error_type": "wrong number of class"},
        {"id": 3, "valid": False, "error": ["y"], "error_type": "..."},
    ])
    summary = B.collect(work, "en", "m", ["agent_multi_turn"])
    assert summary["n"] == 4 and summary["semantic_score"] == pytest.approx(0.5)
    assert summary["per_category"]["agent_multi_turn"]["end_to_end_accuracy"] == 0.5
    assert summary["per_category"]["agent_multi_turn"]["total_count"] == 4


def test_collect_normal_multi_turn_clusters_by_turn_group(tmp_path):
    work = tmp_path / "w"
    ids = ["normal_multi_turn_user_adjust_0_0", "normal_multi_turn_user_adjust_0_1",
           "normal_multi_turn_user_adjust_1_0"]
    _write_jsonl(work / "result_all" / "result_en" / "m" / "data_normal_multi_turn_user_adjust_result.json",
                 [{"id": i, "result": "[f(a=1)]"} for i in ids])
    _write_jsonl(work / "score_all" / "score_en" / "m" / "data_normal_multi_turn_user_adjust_score.json", [
        {"accuracy": 0.5, "correct_count": 1, "total_count": 3, "process_accuracy": 0.67},
        {"id": "normal_multi_turn_user_adjust_0_1", "turn": "0", "valid": False, "error": ["e"]},
    ])
    summary = B.collect(work, "en", "m", ["normal_multi_turn_user_adjust"])
    assert summary["n"] == 3
    assert summary["n_clusters"] == 2  # turn groups 0 and 1, the official unit
    assert summary["semantic_score"] == pytest.approx(2 / 3)


def test_failed_task_ids_conventions():
    results = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    failures = [{"id": 2}, {"id": "a"}, {"id": None}, {"id": 99}]
    assert B.failed_task_ids(results, failures) == {"c", "a"}


def test_collect_missing_or_empty_score_is_fatal(tmp_path):
    work = tmp_path / "w"
    _write_jsonl(work / "result_all" / "result_en" / "m" / "data_agent_multi_turn_result.json",
                 [{"id": "agent_multi_turn_0"}])
    with pytest.raises(SystemExit, match="wrote no"):
        B.collect(work, "en", "m", ["agent_multi_turn"])
    (work / "score_all" / "score_en" / "m").mkdir(parents=True)
    (work / "score_all" / "score_en" / "m" / "data_agent_multi_turn_score.json").write_text("")
    with pytest.raises(SystemExit, match="empty score file"):
        B.collect(work, "en", "m", ["agent_multi_turn"])


def test_cluster_id_rule():
    assert B.cluster_id("normal_multi_turn_user_switch", "normal_multi_turn_user_switch_3_2") == \
        "normal_multi_turn_user_switch_3"
    assert B.cluster_id("agent_multi_turn", "agent_multi_turn_7") == "agent_multi_turn_7"
