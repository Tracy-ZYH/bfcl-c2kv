"""Regressions for task identity and unknown tau2 protocol coverage."""
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c2kv_eval.portable.adapters import tau2_adapter


@pytest.mark.parametrize('tool_name,available,legal', [
    ('lookup', True, True), ('missing_tool', True, False), ('lookup', False, None),
])
def test_tool_registry_instance_and_unknown_protocol(tmp_path, monkeypatch, tool_name, available, legal):
    registry_module = ModuleType('tau2.registry')
    schema = {'type': 'function', 'function': {
        'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}}

    def get_constructor(domain):
        assert domain == 'airline'
        if not available:
            raise RuntimeError('tool pool unavailable')
        return lambda: SimpleNamespace(get_tools=lambda: [SimpleNamespace(openai_schema=schema)])

    registry_module.registry = SimpleNamespace(get_env_constructor=get_constructor)
    parent = ModuleType('tau2')
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, 'tau2', parent)
    monkeypatch.setitem(sys.modules, 'tau2.registry', registry_module)
    path = tmp_path / 'updated_results.json'
    path.write_text(json.dumps({'simulations': [{
        'task_id': 'fixed-task', 'reward_info': {'reward': 0.0},
        'termination_reason': 'agent_stop',
        'messages': [{'role': 'assistant', 'content': None, 'tool_calls': [{
            'type': 'function', 'function': {'name': tool_name, 'arguments': '{}'}}]}],
    }]}), encoding='utf-8')
    result = tau2_adapter.collect(path)
    assert result['semantic_score'] == 0.0
    assert result['task_rows'][0]['task_id'] == 'fixed-task'
    assert result['task_rows'][0]['protocol_legal'] is legal
    assert result['protocol_evaluable_tasks'] == int(legal is not None)
    assert result['task_rows'][0]['n_illegal_turns'] == int(legal is False)
    assert result['task_rows'][0]['n_unknown_protocol_turns'] == int(legal is None)
    assert result['task_rows'][0]['normal_termination'] is True
    assert result['normal_termination_rate'] == 1.0
    assert result['premature_termination_count'] == 0
    assert result['official_metric_kind'] == 'binary_task_success'


def test_tau2_collect_separates_premature_zero_from_normal_wrong_zero(
        tmp_path, monkeypatch):
    registry_module = ModuleType('tau2.registry')
    registry_module.registry = SimpleNamespace(
        get_env_constructor=lambda domain: lambda: SimpleNamespace(get_tools=lambda: []))
    parent = ModuleType('tau2')
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, 'tau2', parent)
    monkeypatch.setitem(sys.modules, 'tau2.registry', registry_module)
    path = tmp_path / 'updated_results.json'
    path.write_text(json.dumps({'simulations': [
        {'task_id': 'normal-wrong', 'reward_info': {'reward': 0.0},
         'termination_reason': 'user_stop', 'messages': []},
        {'task_id': 'premature', 'reward_info': {'reward': 0.0},
         'termination_reason': 'max_steps', 'messages': []},
    ]}), encoding='utf-8')
    result = tau2_adapter.collect(path)
    assert result['official_success_rate'] == 0.0
    assert result['normal_termination_rate'] == 0.5
    assert result['premature_termination_count'] == 1
    assert result['termination_counts'] == {'user_stop': 1, 'max_steps': 1}
