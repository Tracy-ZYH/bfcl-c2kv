"""Offline tests for adapters/acon_adapter.py (no subprocess, no network).

Covers: runner command lines and env, output-path derivation (must match
ACON's run.py / run_all.py rules verbatim), the QA and AppWorld collectors,
the terminal-state gate, and the loud failure on an unrecognised AppWorld
evaluation layout.
"""
from __future__ import annotations

import json
import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2kv_eval.portable.adapters import acon_adapter as A


def test_bm25_patch_applies_and_imports_without_dense_stack(tmp_path, monkeypatch):
    project_root = Path(A.__file__).resolve().parents[1]
    default_acon_root = project_root.parent / "tmp" / "baselines" / "acon"
    acon_root = Path(os.environ.get("ACON_ROOT", default_acon_root))
    source_rel = Path("experiments/smolagents/search/retriever_server.py")
    if not (acon_root / source_rel).is_file():
        pytest.skip("set ACON_ROOT to the pinned microsoft/acon checkout")

    staged_root = tmp_path / "acon"
    staged_source = staged_root / source_rel
    staged_source.parent.mkdir(parents=True)
    shutil.copy2(acon_root / source_rel, staged_source)
    patch_path = project_root / "patches" / "acon" / "0003-bm25-lazy-imports.patch"
    subprocess.run(
        ["git", "apply", "--ignore-space-change", str(patch_path)],
        cwd=staged_root,
        check=True,
        capture_output=True,
        text=True,
    )

    stubs = tmp_path / "stubs"
    (stubs / "pyserini" / "search").mkdir(parents=True)
    (stubs / "pyserini" / "__init__.py").write_text("")
    (stubs / "pyserini" / "search" / "__init__.py").write_text("")
    (stubs / "pyserini" / "search" / "lucene.py").write_text(
        "class _Doc:\n"
        "    def raw(self): return '{}'\n"
        "class LuceneSearcher:\n"
        "    def __init__(self, path): self.path = path\n"
        "    def doc(self, idx): return _Doc()\n"
    )
    (stubs / "uvicorn.py").write_text("def run(*args, **kwargs): pass\n")
    (stubs / "fastapi.py").write_text(
        "class FastAPI:\n"
        "    def post(self, path): return lambda fn: fn\n"
    )
    (stubs / "pydantic.py").write_text("class BaseModel: pass\n")

    monkeypatch.syspath_prepend(str(stubs))
    for module in ("datasets", "faiss", "numpy", "torch", "tqdm", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setattr(sys, "argv", [str(staged_source), "--index_path", "dummy-index"])
    namespace = runpy.run_path(str(staged_source), run_name="acon_retriever_import_test")
    assert type(namespace["retriever"]).__name__ == "BM25Retriever"
    assert namespace["retriever"].contain_doc is True


def test_pyserini_sparse_patch_removes_only_dense_exports(tmp_path):
    site_packages = tmp_path / "site-packages"
    init_path = site_packages / "pyserini" / "search" / "lucene" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    prefix = "".join(f"# line {i}\n" for i in range(1, 24))
    source = (
        "JBagOfWordsQueryGenerator = autoclass('io.anserini.search.query.BagOfWordsQueryGenerator')\n"
        "JDisjunctionMaxQueryGenerator = autoclass('io.anserini.search.query.DisjunctionMaxQueryGenerator')\n"
        "JCovid19QueryGenerator = autoclass('io.anserini.search.query.Covid19QueryGenerator')\n"
        "\n"
        "from ._impact_searcher import LuceneImpactSearcher, SlimSearcher\n"
        "from ._searcher import LuceneSearcher, LuceneFusionSearcher, LuceneSimilarities\n"
        "from ._hnsw_searcher import LuceneHnswDenseSearcher, LuceneFlatDenseSearcher\n"
    )
    init_path.write_text(prefix + source)
    patch_path = (
        Path(A.__file__).resolve().parents[1]
        / "patches"
        / "acon"
        / "0004-pyserini-sparse-imports.patch"
    )
    subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=site_packages,
        check=True,
        capture_output=True,
        text=True,
    )
    patched = init_path.read_text()
    assert "from ._searcher import LuceneSearcher" in patched
    assert "_impact_searcher" not in patched
    assert "_hnsw_searcher" not in patched
    assert patched.count("\n") == (prefix + source).count("\n") - 2


def test_smolagents_execution_error_reaches_next_agent_turn(tmp_path, monkeypatch):
    """The exact QA action must execute and its TypeError must become feedback."""
    project_root = Path(A.__file__).resolve().parents[1]
    default_acon_root = project_root.parent / "tmp" / "baselines" / "acon"
    acon_root = Path(os.environ.get("ACON_ROOT", default_acon_root))
    env_rel = Path("src/productive_agents/env/smolagents/env.py")
    agent_rel = Path("src/productive_agents/agents/smolagents/agent.py")
    if not (acon_root / env_rel).is_file():
        pytest.skip("set ACON_ROOT to the pinned microsoft/acon checkout")

    staged_root = tmp_path / "acon"
    for source_rel in (env_rel, agent_rel):
        target = staged_root / source_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(acon_root / source_rel, target)
    patch_path = (
        project_root
        / "patches"
        / "acon"
        / "0005-smolagents-error-feedback.patch"
    )
    subprocess.run(
        ["git", "apply", "--ignore-space-change", str(patch_path)],
        cwd=staged_root,
        check=True,
        capture_output=True,
        text=True,
    )

    class FakeExecutor:
        def __init__(self, **_kwargs):
            self.calls = []

        def send_tools(self, _tools):
            pass

        def __call__(self, code):
            self.calls.append(code)
            raise TypeError(
                "WikipediaRetrieverTool.forward() missing 1 required "
                "positional argument: 'n_results'"
            )

    class FakeConfig:
        verbose = False
        max_interactions = 4

    class FakeWikipediaTool:
        name = "wikipedia_search"

    class FakeFinalAnswerTool:
        name = "final_answer"

    class FakeActionProcessor:
        def __init__(self, logger):
            self.logger = logger

    class FakePromptBuilder:
        def __init__(self, prompt_dict=None, working_dir="."):
            self.prompt_dict = prompt_dict or {}
            self.working_dir = working_dir

    def module(name, **attrs):
        value = types.ModuleType(name)
        for key, item in attrs.items():
            setattr(value, key, item)
        return value

    stubs = {
        "smolagents": module("smolagents", LocalPythonExecutor=FakeExecutor),
        "smolagents.local_python_executor": module(
            "smolagents.local_python_executor", fix_final_answer_code=lambda code: code
        ),
        "smolagents.utils": module(
            "smolagents.utils",
            parse_code_blobs=lambda code, _tags: code,
            truncate_content=str,
            extract_code_from_text=lambda *_args: "",
        ),
        "productive_agents": module("productive_agents"),
        "productive_agents.env": module("productive_agents.env"),
        "productive_agents.env.smolagents": module("productive_agents.env.smolagents"),
        "productive_agents.env.base": module(
            "productive_agents.env.base", BaseLanguageBasedEnv=object
        ),
        "productive_agents.env.smolagents.config": module(
            "productive_agents.env.smolagents.config", SmolagentsEnvConfig=FakeConfig
        ),
        "productive_agents.env.smolagents.tool": module(
            "productive_agents.env.smolagents.tool",
            WikipediaRetrieverTool=FakeWikipediaTool,
            FinalAnswerTool=FakeFinalAnswerTool,
        ),
        "productive_agents.utils": module(
            "productive_agents.utils", all_seed=lambda _seed: None
        ),
        "productive_agents.agents": module("productive_agents.agents"),
        "productive_agents.agents.smolagents": module(
            "productive_agents.agents.smolagents"
        ),
        "productive_agents.agents.unified_agent": module(
            "productive_agents.agents.unified_agent",
            UnifiedAgent=object,
            UnifiedPromptBuilder=FakePromptBuilder,
            UnifiedActionProcessor=FakeActionProcessor,
        ),
    }
    for name, value in stubs.items():
        monkeypatch.setitem(sys.modules, name, value)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        spec.loader.exec_module(loaded)
        return loaded

    env_module = load("productive_agents.env.smolagents.env", staged_root / env_rel)
    agent_module = load(
        "productive_agents.agents.smolagents.agent", staged_root / agent_rel
    )
    captured_response = '''Thought: I will search the first question.

```python
yam_food_storage = wikipedia_search(query="where is the food stored in a yam plant?")
print("Food storage in yam plant:", yam_food_storage)
```'''
    action = agent_module.SmolagentsActionProcessor(None).extract_action(
        captured_response
    )
    assert action.startswith("yam_food_storage = wikipedia_search")
    assert "```" not in action

    env = env_module.SmolagentsEnv(config=FakeConfig())
    env.reset(42, "captured QA task")
    observation, reward, done, info = env.step(action)

    assert env.python_executor.calls == [action]
    assert "missing 1 required positional argument: 'n_results'" in observation
    assert (reward, done, info["reason"]) == (0.0, False, "execution_error")
    assert env.observation == observation
    assert env.trajectory == [{
        "action": action,
        "observation": observation,
        "reward": 0.0,
        "done": False,
        "info": info,
    }]
    builder = object.__new__(agent_module.SmolagentsPromptBuilder)
    builder.env = env
    assert builder.build_prompt(env, []) == observation


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---- env / commands ---------------------------------------------------------

def test_runner_env_points_agent_at_proxy_v1(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://corp:3128")
    env = A.runner_env("http://127.0.0.1:34100/")
    assert env[A.BASE_URL_ENV] == "http://127.0.0.1:34100/v1"
    assert env[A.API_KEY_ENV] == "EMPTY"
    assert env["NO_PROXY"] == "127.0.0.1,localhost"


def test_qa_command_uses_shipped_split_and_pins():
    cmd = A.qa_command("py", "c2kv-agent", "run_ab12", "test", 30, limit=5,
                       id_list_file=Path("/x/ids.txt"))
    assert cmd[:2] == ["py", "run.py"]
    assert cmd[cmd.index("--data_folder") + 1] == "data/nq_multi_8"
    assert cmd[cmd.index("--limit") + 1] == "5"
    assert cmd[cmd.index("--id_list_file") + 1] == str(Path("/x/ids.txt"))
    assert "--output_dir" not in cmd  # ignored upstream; path derived instead


def test_appworld_command_and_experiment_name():
    cmd = A.appworld_command("py", "org/model", "run_ab12", "test_normal", 50,
                             task_ids=["t1", "t2"])
    assert cmd[cmd.index("--split") + 1] == "test_normal"
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert cmd[-3:] == ["--task_ids", "t1", "t2"]
    # run_all.py: model_name.replace("/", "_") + "_" + tag
    assert A.appworld_experiment("org/model", "run_ab12") == "org_model_run_ab12"


def test_output_paths_follow_runner_rules(tmp_path):
    # run.py: sanitised model/tag, dev -> test fold
    run_dir = A.qa_run_dir(tmp_path, "c2kv agent", "run/ab12", "dev")
    assert run_dir == (tmp_path / "experiments" / "smolagents" / "outputs"
                       / "c2kv-agent_run-ab12" / "test")
    # run_all.py output + appworld evaluate output (relative to runner cwd)
    assert A.appworld_run_dir(tmp_path, "m", "t", "test_normal") == (
        tmp_path / "experiments" / "appworld" / "outputs" / "m_t" / "test_normal")
    assert A.appworld_eval_path(tmp_path, "m", "t", "test_normal") == (
        tmp_path / "experiments" / "appworld" / "experiments" / "outputs" / "m_t"
        / "evaluations" / "test_normal.json")


def test_qa_expected_counts_shipped_file(tmp_path):
    data = tmp_path / "experiments" / "smolagents" / A.QA_DATA_FOLDER / "test.jsonl"
    _write_jsonl(data, [{"id": f"nq_multi8_test_{i}"} for i in range(7)])
    assert A.qa_expected(tmp_path, "test", None, None) == 7
    assert A.qa_expected(tmp_path, "dev", 3, None) == 3
    assert A.qa_expected(tmp_path, "test", 3, ["a", "b"]) == 2


# ---- QA collector -----------------------------------------------------------

def _qa_rows():
    return [
        {"id": "nq_multi8_test_1", "em": 0.5, "f1": 0.6, "iterations": 9, "success": True},
        {"id": "nq_multi8_test_2", "em": 0.0, "f1": 0.1, "iterations": 30, "success": False},
        {"id": "nq_multi8_test_3", "em": 1.0, "f1": 1.0, "iterations": 12, "success": True},
    ]


def test_collect_qa_rows_and_official_summary(tmp_path):
    _write_jsonl(tmp_path / "predictions.jsonl", _qa_rows())
    (tmp_path / "summary.json").write_text(json.dumps({"avg_em": 0.5, "avg_f1": 0.5667, "total": 3}))
    summary = A.collect_qa(tmp_path, expected=3)
    assert summary["n"] == 3 and summary["n_clusters"] == 3
    assert summary["semantic_score"] == pytest.approx(0.5)
    assert summary["f1_mean"] == pytest.approx((0.6 + 0.1 + 1.0) / 3)
    assert summary["official_summary"]["total"] == 3
    assert summary["protocol_legal_rate"] is None  # code agent: no schema column


def test_collect_qa_terminal_gate(tmp_path):
    _write_jsonl(tmp_path / "predictions.jsonl", _qa_rows())
    with pytest.raises(SystemExit, match="n_scored=3 < n_total=4"):
        A.collect_qa(tmp_path, expected=4)


def test_collect_qa_missing_predictions_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="predictions.jsonl"):
        A.collect_qa(tmp_path)


# ---- AppWorld collector -----------------------------------------------------

def test_appworld_per_task_recognised_shapes():
    by_dict = {"aggregate": {"tgc": 0.5}, "individual": {"t1": {"success": True}, "t2": {"success": False}}}
    assert A.appworld_per_task(by_dict) == {"t1": True, "t2": False}
    by_bool = {"tasks": {"t1": True, "t2": False}}
    assert A.appworld_per_task(by_bool) == {"t1": True, "t2": False}
    by_arrays = {"tasks": {"t1": {"passes": ["a"], "fails": []}, "t2": {"passes": [], "fails": ["x"]}}}
    assert A.appworld_per_task(by_arrays) == {"t1": True, "t2": False}
    by_list = {"results": [{"task_id": "t1", "passed": True}, {"task_id": "t2", "passed": False}]}
    assert A.appworld_per_task(by_list) == {"t1": True, "t2": False}


def test_appworld_per_task_unknown_layout_is_fatal():
    with pytest.raises(SystemExit, match="unrecognised appworld evaluation layout"):
        A.appworld_per_task({"tgc": 0.4, "sgc": 0.2})
    with pytest.raises(SystemExit):
        A.appworld_per_task({"individual": {"t1": {"score": 0.3}}})  # no boolean


def test_collect_appworld_joins_runner_results(tmp_path):
    eval_path = tmp_path / "evaluations" / "test_normal.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps({
        "tgc": 0.5, "sgc": 0.0,
        "individual": {"t1": {"success": True}, "t2": {"success": False}},
    }))
    run_dir = tmp_path / "outputs" / "m_t" / "test_normal"
    (run_dir / "task_t1").mkdir(parents=True)
    (run_dir / "task_t1" / "results.json").write_text(json.dumps(
        {"success": True, "iterations": 7, "termination_reason": "task_completed"}))
    (run_dir / "task_t2").mkdir(parents=True)
    # agent claimed success but the official scorer disagrees: semantic wins
    (run_dir / "task_t2" / "results.json").write_text(json.dumps(
        {"success": True, "iterations": 50, "termination_reason": "max_iterations"}))
    summary = A.collect_appworld(eval_path, run_dir, expected=2)
    assert summary["n"] == 2 and summary["semantic_score"] == pytest.approx(0.5)
    assert summary["official_aggregate"] == {"tgc": 0.5, "sgc": 0.0}


def test_collect_appworld_terminal_gate_and_missing_eval(tmp_path):
    eval_path = tmp_path / "test_normal.json"
    eval_path.write_text(json.dumps({"individual": {"t1": {"success": True}}}))
    with pytest.raises(SystemExit, match="n_scored=1 != n_total=168"):
        A.collect_appworld(eval_path, tmp_path / "none", expected=168)
    with pytest.raises(SystemExit, match="wrote no"):
        A.collect_appworld(tmp_path / "missing.json", tmp_path)


def test_appworld_subset_is_private_and_shared_by_runner_and_scorer(tmp_path, monkeypatch):
    source = tmp_path / "acon" / "experiments" / "appworld"
    source.mkdir(parents=True)
    (source / "run_all.py").write_text("# official runner\n")
    data_root = tmp_path / "official"
    datasets = data_root / "data" / "datasets"
    datasets.mkdir(parents=True)
    dataset = datasets / "test_normal.txt"
    dataset.write_text("task_1\ntask_2\n")
    monkeypatch.setenv("APPWORLD_ROOT", str(data_root))
    out = tmp_path / "run"
    out.mkdir()
    root = A.prepare_appworld_run(tmp_path / "acon", out, "test_normal", ["task_2"])
    private = root / "experiments" / "appworld"
    assert (private / "run_all.py").is_file()
    assert (private / "data" / "datasets" / "test_normal.txt").read_text() == "task_2\n"
    assert dataset.read_text() == "task_1\ntask_2\n"
    assert json.loads((out / "selected_tasks.json").read_text())["task_ids"] == ["task_2"]
    with pytest.raises(SystemExit, match="outside test_normal"):
        A.prepare_appworld_run(tmp_path / "acon", tmp_path / "other", "test_normal", ["task_3"])
    with pytest.raises(FileExistsError):
        A.prepare_appworld_run(tmp_path / "acon", out, "test_normal", ["task_2"])


def test_appworld_score_must_match_selected_ids(tmp_path):
    path = tmp_path / "score.json"
    path.write_text(json.dumps({"individual": {"task_1": {"success": True}}}))
    with pytest.raises(SystemExit, match="scored task IDs"):
        A.collect_appworld(path, tmp_path, expected=1, expected_ids=["task_2"])


def test_run_dispatch_rejects_unknown_kind(tmp_path):
    with pytest.raises(SystemExit, match="unknown ACON benchmark kind"):
        A.run_kind("nope", "http://x", tmp_path)


# ---- cost join (the only adapter pair whose artefacts key the request log) ---

from c2kv_eval.portable import proxy, reqlog


def _session(task: str, turns: int = 2):
    """One ACON session as MemoryManager.dump_history writes it:
    [system, user, assistant, user, assistant, ...] (memory.py:112-168)."""
    session = [{"role": "system", "content": "SYSTEM PROMPT"},
               {"role": "user", "content": f"task {task}"}]
    for i in range(turns):
        session.append({"role": "assistant", "content": f"code {task} {i}"})
        session.append({"role": "user", "content": f"obs {task} {i}"})
    return session


def _dump_history(task_dir: Path, session):
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / A.HISTORY_FILE).write_text(json.dumps([session]), encoding="utf-8")


def _log_row(messages, **extra):
    row = {"status": "ok", "conv_id": proxy.conversation_id(messages)}
    row.update(extra)
    return row


def test_conversation_ids_are_the_two_the_proxy_sees(tmp_path):
    session = _session("t1")
    _dump_history(tmp_path, session)
    ids = A.conversation_ids(tmp_path / A.HISTORY_FILE)
    # first request: [system, user]; every later one: [system, user, assistant, ...]
    assert ids == [proxy.conversation_id(session[:2]),
                   proxy.conversation_id(session[:3])]
    # and the id really is stable once the assistant turn exists
    assert proxy.conversation_id(session[:5]) == ids[1]
    assert proxy.conversation_id(session) == ids[1]


def test_conversation_ids_missing_or_broken_history_is_empty(tmp_path):
    assert A.conversation_ids(tmp_path / "nope.json") == []
    (tmp_path / A.HISTORY_FILE).write_text("not json", encoding="utf-8")
    assert A.conversation_ids(tmp_path / A.HISTORY_FILE) == []
    (tmp_path / A.HISTORY_FILE).write_text('{"a": 1}', encoding="utf-8")
    assert A.conversation_ids(tmp_path / A.HISTORY_FILE) == []


def test_collect_qa_joins_cost_columns_from_the_request_log(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "predictions.jsonl", _qa_rows())
    sessions = {}
    for rec in _qa_rows():
        sessions[rec["id"]] = _session(rec["id"])
        _dump_history(A.qa_sample_dir(run_dir, rec["id"]), sessions[rec["id"]])
    first, second = "nq_multi8_test_1", "nq_multi8_test_2"
    log = tmp_path / "proxy.jsonl"
    rows = [
        _log_row(sessions[first][:2], wall_sec=1.0, gist_tokens=10,
                 original_tokens=100, n_docs=0, dropped_docs=0),
        _log_row(sessions[first][:3], wall_sec=2.0, gist_tokens=30,
                 original_tokens=300, n_docs=4, dropped_docs=1),
        _log_row(sessions[second][:2], wall_sec=0.5, gist_tokens=5,
                 original_tokens=50, n_docs=0, dropped_docs=0),
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = A.collect_qa(run_dir, expected=3, request_log=log)
    # task 3 made no request -> no cost fields, so the means cover 2 tasks
    assert summary["wall_sec_mean"] == pytest.approx((3.0 + 0.5) / 2)
    assert summary["gist_tokens_mean"] == pytest.approx((40 + 5) / 2)
    assert summary["original_tokens_mean"] == pytest.approx((400 + 50) / 2)
    assert summary["cost_join"] == "joined: 2/3 tasks, 3/3 logged requests"
    # ...and the denominator of those means is a NUMBER in the summary, not
    # only prose: semantic_score covers 3 tasks, the cost means cover 2
    assert summary["n"] == 3 and summary["n_cost_joined"] == 2
    # the three joined fields metrics.aggregate does not mean
    assert summary["n_cost_requests"] == 3
    assert summary["n_docs_max"] == 4
    assert summary["dropped_docs_total"] == 1


def test_collect_qa_without_request_log_sets_no_cost_columns(tmp_path):
    _write_jsonl(tmp_path / "predictions.jsonl", _qa_rows())
    summary = A.collect_qa(tmp_path, expected=3)
    assert summary["wall_sec_mean"] is None
    assert summary["gist_tokens_mean"] is None
    assert summary["cost_join"] == "not joinable: no request log for this run"
    assert summary["n_cost_joined"] == 0
    # nothing was measured: None, never a zero that reads as a measurement
    assert summary["n_cost_requests"] is None
    assert summary["n_docs_max"] is None
    assert summary["dropped_docs_total"] is None


def test_cost_join_returns_the_report_not_only_its_prose_line(tmp_path):
    """The summary needs the numeric denominator too, so cost_join hands back
    the whole reqlog report."""
    rows = [{"task_id": "t1"}, {"task_id": "t2"}]
    report = A.cost_join(rows, lambda tid: tmp_path / tid, None)
    assert report["n_rows"] == 2 and report["n_joined"] == 0
    assert (reqlog.cost_join_status(report)
            == "not joinable: no request log for this run")


def test_collect_qa_reports_an_unmatched_log_instead_of_a_number(tmp_path):
    """A history that does not describe what was sent must yield NOTHING."""
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "predictions.jsonl", _qa_rows())
    for rec in _qa_rows():
        _dump_history(A.qa_sample_dir(run_dir, rec["id"]), _session(rec["id"]))
    log = tmp_path / "proxy.jsonl"
    log.write_text(json.dumps({"status": "ok", "conv_id": "somethingelse",
                               "wall_sec": 9.0}) + "\n", encoding="utf-8")
    summary = A.collect_qa(run_dir, expected=3, request_log=log)
    assert summary["wall_sec_mean"] is None
    assert summary["cost_join"].startswith("not joinable: ")


def test_collect_appworld_joins_cost_columns(tmp_path):
    eval_path = tmp_path / "evaluations" / "test_normal.json"
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps(
        {"individual": {"t1": {"success": True}, "t2": {"success": False}}}))
    run_dir = tmp_path / "outputs" / "m_t" / "test_normal"
    sessions = {t: _session(t) for t in ("t1", "t2")}
    for task, session in sessions.items():
        _dump_history(A.appworld_task_dir(run_dir, task), session)
    log = tmp_path / "proxy.jsonl"
    rows = [
        _log_row(sessions["t1"][:2], wall_sec=1.0, gist_tokens=8, n_docs=0),
        _log_row(sessions["t1"][:3], wall_sec=3.0, gist_tokens=24, n_docs=11,
                 dropped_docs=2),
        _log_row(sessions["t2"][:3], wall_sec=2.0, gist_tokens=16, n_docs=5),
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    summary = A.collect_appworld(eval_path, run_dir, expected=2, request_log=log)
    assert summary["wall_sec_mean"] == pytest.approx((4.0 + 2.0) / 2)
    assert summary["cost_join"] == "joined: 2/2 tasks, 3/3 logged requests"
    assert summary["n_cost_joined"] == 2 and summary["n_cost_requests"] == 3
    # t1's second request dropped 2 docs out of 11 — the fact that says the
    # task's own history was truncated by turn packing
    assert summary["n_docs_max"] == 11 and summary["dropped_docs_total"] == 2
