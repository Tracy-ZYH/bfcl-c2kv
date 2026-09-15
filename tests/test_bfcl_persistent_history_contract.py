"""Model-free BFCL request/session contracts, using mocked HTTP only."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import traceback

import pytest

ADAPTERS = Path(__file__).resolve().parents[1] / "c2kv_eval/adapters"


def method(filename, cls, name, namespace):
    tree = ast.parse((ADAPTERS / filename).read_text())
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls
                for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    # Preserve postponed annotations without importing the evaluator/model stack.
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ast.fix_missing_locations(future)
    exec(compile(ast.Module(body=[future, node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class Stats:
    def __init__(self, *args):
        self.errors = []

    def __getattr__(self, _):
        return 0

    def as_dict(self):
        return {"errors": self.errors}


def test_session_id_is_sent_on_every_chat_request_full_is_unchanged():
    calls = []
    def post(base, path, payload, timeout):
        calls.append((path, deepcopy(payload)))
        return {"choices": [{"message": {"content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    query = method("bfcl_history_drift.py", "HistoryDriftRunner", "_query", {
        "_token_count": lambda *a: 1, "time": time, "_post_json": post,
        "_tool_calls_to_text": lambda _: "", "_actual_cached_tokens_from_response": lambda _: 0,
        "_kv_runtime_stats_from_response": lambda _: None, "_kv_memory_report_from_response": lambda _: None})
    runner = SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda *a, **kw: [1]),
        max_completion_tokens=1, model="mock", temperature=0, base_url="mock://not-http", timeout=1,
        _persistent_history_session_id="episode-session", _last_kv_memory_hint={})
    for _ in range(2):
        query(runner, [{"role": "user", "content": "mock"}], [], Stats())
    assert len(calls) == 2 and all(p == "/v1/chat/completions" for p, _ in calls)
    assert all(body['session_params']['id'] == 'episode-session' for _, body in calls)
    runner._persistent_history_session_id = None
    query(runner, [], [], Stats())
    assert 'session_params' not in calls[-1][1]


@pytest.mark.parametrize("fail", [False, True])
def test_episode_closes_session_even_when_generation_raises(fail):
    run_sample = method("bfcl_history_kv_baselines.py", "HistoryKVBaselineRunner", "run_sample", {
        "DriftStats": Stats, "_tool_payload": lambda x: x, "deepcopy": deepcopy,
        "traceback": traceback, "RUNTIME_EVICTION_METHODS": {"h2o"}})
    events = []
    runner = SimpleNamespace(mode="mock", ratio=4, _persistent_session_enabled=lambda: True,
                             strict_runtime_eviction=False, history_kv_method="h2o")
    def open_session(sample):
        events.append(("open", sample)); runner._persistent_history_session_id = 'same-session'
    def close_session():
        events.append(("close", runner._persistent_history_session_id)); runner._persistent_history_session_id = None
    def run_impl(*args):
        if fail:
            raise RuntimeError("mock generation failed")
        return [], {}
    runner._open_persistent_history_session = open_session
    runner._close_persistent_history_session = close_session
    runner._run_sample_impl = run_impl
    row = run_sample(runner, {"id": "episode", "function": []})
    assert events == [('open', 'episode'), ('close', 'same-session')]
    assert runner._persistent_history_session_id is None
    assert row['id'] == 'episode'


def test_persistent_mode_forbids_repair_extract_even_with_empty_history():
    build = method("bfcl_history_kv_baselines.py", "HistoryKVCompressor", "_build_runtime_history_kv", {})
    self = SimpleNamespace(runner=SimpleNamespace(_persistent_session_enabled=lambda: True))
    with pytest.raises(RuntimeError, match="REPAIR_EXTRACT_FORBIDDEN"):
        build(self, [], [], 0, Stats())


def test_compression_disabled_and_full_never_open_persistent_history_session():
    enabled = method("bfcl_history_kv_baselines.py", "HistoryKVBaselineRunner", "_persistent_session_enabled",
                     {"RUNTIME_EVICTION_METHODS": {'streamingllm', 'h2o', 'snapkv_persistent', 'pyramidkv'}})
    runner = SimpleNamespace(persistent_history_kv_session=True, runtime_history_kv_backend='physical_eviction',
        history_kv_method='full', history_kv_retention_ratio=.312, history_kv_target_compression=0)
    assert not enabled(runner)
    runner.history_kv_method = 'h2o'
    assert enabled(runner)
    runner.history_kv_retention_ratio = 1
    assert not enabled(runner)
