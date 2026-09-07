"""BFCL first-failed-turn gold recovery controller.

The controller observes the official multi-turn handler through overridable
hooks.  It never changes BFCL responses or places ground truth in the prompt.
"""
from __future__ import annotations

import ast
import copy
import importlib
import json
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .agent.d_witness_core import target_values


# v1 skipped a failed first user turn unconditionally.  v2 instead retries
# exactly from that turn's first request which demonstrably compressed history.
ORACLE_KIND = "bfcl_gold_turn_v2"
ORACLE_VERSION = 2
MULTI_EVENT_ORACLE_KIND = "bfcl_gold_turn_v3"
MULTI_EVENT_ORACLE_VERSION = 3
COMPRESSION_OBSERVATION_SOURCE = "response.c2kv_proxy.gist_tokens>0"


def _audit_json_default(value: Any) -> Dict[str, str]:
    """Represent non-JSON checker diagnostics without changing checker state.

    Official checker failures may retain simulator objects such as a
    ``GorillaFileSystem.Directory``.  They are diagnostics only: preserve the
    JSON-native telemetry structure and turn each unsupported leaf into a
    traceable type/repr record.  Do not catch serialization or file I/O
    errors here; an audit write failure still needs to surface to the caller.
    """
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


class TurnRetry(RuntimeError):
    """Internal control flow from the final BFCL loop to ``inference``."""

    def __init__(self, turn: int):
        super().__init__(f"retry BFCL user turn {turn}")
        self.turn = turn


class GoldLiteralError(ValueError):
    """A gold call contains syntax the oracle interface cannot represent."""

    def __init__(self, call_index: int, field: str, detail: str):
        super().__init__(
            f"gold call {call_index} field {field} is not literal: {detail}")
        self.call_index = call_index
        self.field = field
        self.detail = detail


def _literal(node: ast.AST, call_index: int, field: str) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as error:
        raise GoldLiteralError(call_index, field, str(error)) from error


def gold_target_values(call_strings: Sequence[str]) -> List[str]:
    """Extract tool names and literal argument leaves without executing data."""
    values: List[str] = []
    for call_index, call_string in enumerate(call_strings):
        if not isinstance(call_string, str):
            raise GoldLiteralError(
                call_index, "call", f"expected str, got {type(call_string).__name__}")
        try:
            expression = ast.parse(call_string, mode="eval").body
        except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as error:
            raise GoldLiteralError(call_index, "call", str(error)) from error
        if not isinstance(expression, ast.Call):
            raise GoldLiteralError(call_index, "call", "expected a function call")
        if not isinstance(expression.func, ast.Name):
            raise GoldLiteralError(call_index, "function", "expected a simple tool name")

        positional = [
            _literal(node, call_index, f"args[{index}]")
            for index, node in enumerate(expression.args)
        ]
        keywords: Dict[str, Any] = {}
        for keyword in expression.keywords:
            if keyword.arg is None:
                raise GoldLiteralError(
                    call_index, "kwargs", "** expansion is not supported")
            keywords[keyword.arg] = _literal(
                keyword.value, call_index, f"kwargs.{keyword.arg}")
        arguments: Any = {"args": positional, "kwargs": keywords}
        values.extend(target_values(expression.func.id, arguments))
    return list(dict.fromkeys(values))


def _force_quit(metadata: Mapping[str, Any]) -> bool:
    stack: List[Any] = [metadata.get("inference_log")]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if (item.get("role") == "handler_log"
                    and "forced to quit" in str(item.get("content", ""))):
                return True
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def _sanitize_namespace(value: str) -> str:
    return re.sub(r"[-./:]", "_", value)


def cleanup_namespace_instances(namespace: str) -> int:
    """Delete only BFCL simulator instances owned by a unique namespace."""
    utils_module = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    prefix = _sanitize_namespace(namespace)
    names = [
        name for name in vars(utils_module)
        if name.startswith(prefix) and name.endswith("_instance")
    ]
    for name in names:
        vars(utils_module).pop(name, None)
    return len(names)


def _proxy_metadata(response: Any) -> Optional[Mapping[str, Any]]:
    proxy = getattr(response, "c2kv_proxy", None)
    if proxy is None and isinstance(getattr(response, "model_extra", None), dict):
        proxy = response.model_extra.get("c2kv_proxy")
    return proxy if isinstance(proxy, Mapping) else None


def _proxy_gold_status(response: Any) -> Optional[str]:
    proxy = _proxy_metadata(response)
    if proxy is None:
        return None
    recovery = proxy.get("gold_recovery")
    if not isinstance(recovery, Mapping):
        return None
    status = recovery.get("status")
    return str(status) if status is not None else None


def _has_proxy_gist_tokens(response: Any) -> bool:
    """Return only the observed compressed-history signal for a request.

    The controller deliberately does not infer compression from a turn number,
    message count, or a later request.  It is eligible only when the proxy
    attached a positive ``gist_tokens`` count to this exact base request.
    """
    proxy = _proxy_metadata(response)
    if proxy is None:
        return False
    gist_tokens = proxy.get("gist_tokens")
    return (isinstance(gist_tokens, (int, float))
            and not isinstance(gist_tokens, bool)
            and gist_tokens > 0)


def official_prefix_check(
    decoded_turns: Sequence[Sequence[Sequence[str]]],
    ground_truth: Sequence[Sequence[str]],
    test_entry: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run both official prefix checkers in a private disposable namespace."""
    checker_module = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    utils_module = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    namespace = f"c2kv_gold_check_{uuid.uuid4().hex}"
    namespace_prefix = _sanitize_namespace(namespace)
    before = set(vars(utils_module))
    try:
        main = checker_module.multi_turn_checker(
            copy.deepcopy(list(decoded_turns)),
            copy.deepcopy(list(ground_truth)),
            copy.deepcopy(dict(test_entry)),
            str(test_entry["id"]).rsplit("_", 1)[0],
            namespace,
        )
        irrelevance = checker_module.multi_turn_irrelevance_checker(
            copy.deepcopy(list(decoded_turns)),
            copy.deepcopy(list(ground_truth)),
        )
        return {
            "valid": bool(main.get("valid")) and bool(irrelevance.get("valid")),
            "multi_turn": main,
            "irrelevance": irrelevance,
        }
    finally:
        for name in set(vars(utils_module)) - before:
            if name.startswith(namespace_prefix) and name.endswith("_instance"):
                vars(utils_module).pop(name, None)


def load_task_ground_truth(test_entry: Mapping[str, Any]) -> List[List[str]]:
    """Load one official ``possible_answer`` row by task id."""
    from bfcl_eval.utils import load_ground_truth_entry

    task_id = str(test_entry["id"])
    category = task_id.rsplit("_", 1)[0]
    matches = [
        row for row in load_ground_truth_entry(category)
        if isinstance(row, dict) and str(row.get("id")) == task_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one BFCL ground-truth row for {task_id}, found {len(matches)}")
    ground_truth = matches[0].get("ground_truth")
    if (not isinstance(ground_truth, list)
            or any(not isinstance(turn, list) for turn in ground_truth)):
        raise RuntimeError(f"invalid BFCL ground truth for {task_id}")
    return copy.deepcopy(ground_truth)


@dataclass
class TaskState:
    task_id: str
    selector: Optional[str]
    original_test_entry: Dict[str, Any]
    ground_truth: Optional[List[List[str]]]
    started_at: float
    attempt: int = 0
    current_turn: int = 0
    step: int = 0
    decoded_turns: List[List[List[str]]] = field(default_factory=list)
    response_cache: Dict[int, List[Tuple[Any, float]]] = field(
        default_factory=lambda: defaultdict(list))
    replay_positions: Dict[int, int] = field(default_factory=dict)
    first_compressed_request_steps: Dict[int, int] = field(default_factory=dict)
    trigger_turn: Optional[int] = None
    retry_start_step: Optional[int] = None
    retry_eligibility: Optional[Dict[str, Any]] = None
    oracle_values: List[str] = field(default_factory=list)
    oracle_decided: bool = False
    recovered: Optional[bool] = None
    retry_final_valid: Optional[bool] = None
    replay_count: int = 0
    checker_seconds: float = 0.0
    base_error: Optional[Dict[str, Any]] = None
    check_error: Optional[Dict[str, Any]] = None
    retry_check_error: Optional[Dict[str, Any]] = None
    status: str = "base_running"
    skip_reason: Optional[str] = None
    nonliteral: Optional[Dict[str, Any]] = None
    intervention_statuses: List[str] = field(default_factory=list)
    retry_instances_cleaned: int = 0


@dataclass
class MultiEventTaskState(TaskState):
    """Latest-trajectory state for the opt-in bounded v3 sensitivity arm."""

    max_events: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)
    active_event_index: Optional[int] = None
    retried_turns: set[int] = field(default_factory=set)
    retries_started: int = 0
    checker_calls: int = 0
    http_call_count: int = 0
    retry_namespaces_cleaned: int = 0
    final_valid: Optional[bool] = None
    blocked_reason: Optional[str] = None


class GoldRecoveryController:
    """Thread-local state machine used by one BFCL handler instance."""

    def __init__(
        self,
        selector: Optional[str],
        audit_path: "Path | str | None" = None,
        *,
        ground_truth_loader: Callable[[Mapping[str, Any]], List[List[str]]] = (
            load_task_ground_truth),
        prefix_checker: Callable[
            [Sequence[Sequence[Sequence[str]]], Sequence[Sequence[str]], Mapping[str, Any]],
            Dict[str, Any],
        ] = official_prefix_check,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if selector not in (None, "witness", "random"):
            raise ValueError(f"unknown BFCL gold selector {selector!r}")
        self.selector = selector
        self.audit_path = Path(audit_path) if audit_path else None
        self.ground_truth_loader = ground_truth_loader
        self.prefix_checker = prefix_checker
        self.clock = clock
        self.local = threading.local()
        self.inference_lock = threading.RLock()
        self.audit_lock = threading.Lock()

    @property
    def state(self) -> Optional[TaskState]:
        return getattr(self.local, "state", None)

    def begin(self, test_entry: Mapping[str, Any]) -> TaskState:
        ground_truth = None
        if self.selector is not None:
            ground_truth = self.ground_truth_loader(test_entry)
        state = TaskState(
            task_id=str(test_entry["id"]),
            selector=self.selector,
            original_test_entry=copy.deepcopy(dict(test_entry)),
            ground_truth=ground_truth,
            started_at=self.clock(),
        )
        self.local.state = state
        return state

    def clear(self) -> None:
        if hasattr(self.local, "state"):
            del self.local.state

    def request_context(self) -> Optional[Dict[str, Any]]:
        state = self.state
        if state is None:
            return None
        return {
            "benchmark": "bfcl",
            "task_id": state.task_id,
            "user_turn": state.current_turn,
            "step": state.step,
            "attempt": state.attempt,
        }

    def oracle_payload(self) -> Optional[Dict[str, Any]]:
        state = self.state
        if (state is None or state.attempt != 1 or state.trigger_turn is None
                or state.current_turn != state.trigger_turn
                or state.retry_start_step is None
                or state.step < state.retry_start_step):
            return None
        return {
            "kind": ORACLE_KIND,
            "version": ORACLE_VERSION,
            "task_id": state.task_id,
            "turn": state.trigger_turn,
            "retry_start_step": state.retry_start_step,
            "selector": state.selector,
            "values": list(state.oracle_values),
        }

    def replay_response(self) -> Optional[Tuple[Any, float]]:
        state = self.state
        if state is None or state.attempt != 1 or state.trigger_turn is None:
            return None
        should_replay = state.current_turn < state.trigger_turn
        if state.current_turn == state.trigger_turn:
            if state.retry_start_step is None:
                raise RuntimeError("BFCL retry lacks retry_start_step")
            should_replay = state.step < state.retry_start_step
        if not should_replay:
            return None
        position = state.replay_positions.get(state.current_turn, 0)
        cached = state.response_cache.get(state.current_turn, [])
        if position >= len(cached):
            raise RuntimeError(
                f"missing cached BFCL response for {state.task_id} "
                f"turn={state.current_turn} step={position}")
        state.replay_positions[state.current_turn] = position + 1
        state.replay_count += 1
        state.step += 1
        response, _original_latency = cached[position]
        return response, 0.0

    def record_query(self, response: Any, latency: float) -> None:
        state = self.state
        if state is None:
            return
        if state.attempt == 0 and state.selector is not None:
            state.response_cache[state.current_turn].append((response, latency))
            if (_has_proxy_gist_tokens(response)
                    and state.current_turn not in state.first_compressed_request_steps):
                state.first_compressed_request_steps[state.current_turn] = state.step
        if (state.attempt == 1 and state.trigger_turn is not None
                and state.current_turn == state.trigger_turn):
            status = _proxy_gold_status(response)
            if status is not None:
                state.intervention_statuses.append(status)
        state.step += 1

    def record_parse(self, handler: Any, parsed: Mapping[str, Any]) -> None:
        state = self.state
        if state is None:
            return
        while len(state.decoded_turns) <= state.current_turn:
            state.decoded_turns.append([])
        try:
            decoded = handler.decode_execute(
                parsed["model_responses"], has_tool_call_tag=False)
        except Exception as error:  # BFCL itself treats decode failure as an empty turn.
            decoded = []
            if state.base_error is None:
                state.base_error = {
                    "type": "decode_error",
                    "turn": state.current_turn,
                    "step": max(0, state.step - 1),
                    "message": str(error),
                }
        state.decoded_turns[state.current_turn].append(copy.deepcopy(decoded))

    def before_next_turn(self, handler: Any) -> None:
        state = self.state
        if state is None:
            return
        if state.selector is not None:
            self._check_completed_prefix(handler, force_quit=False)
        state.current_turn += 1
        state.step = 0

    def after_inference(self, handler: Any, metadata: Mapping[str, Any]) -> None:
        state = self.state
        if state is None:
            return
        if state.selector is None:
            state.status = "completed"
            return
        forced = _force_quit(metadata)
        if forced and state.base_error is None:
            state.base_error = {
                "type": "force_quit",
                "turn": state.current_turn,
            }
        self._check_completed_prefix(handler, force_quit=forced)
        if state.attempt == 0 and not state.oracle_decided:
            state.status = "base_valid"
        elif state.attempt == 1:
            if state.recovered and state.retry_final_valid:
                state.status = "retry_recovered"
            elif state.recovered:
                state.status = "retry_target_recovered_later_failed"
            else:
                state.status = "retry_failed"

    def _check_completed_prefix(self, handler: Any, *, force_quit: bool) -> None:
        state = self.state
        assert state is not None and state.ground_truth is not None
        turn = state.current_turn
        prefix_length = turn + 1
        if len(state.decoded_turns) < prefix_length:
            outcome = {
                "valid": False,
                "multi_turn": {
                    "valid": False,
                    "error_type": "c2kv:missing_decoded_turn",
                },
                "irrelevance": {"valid": False},
            }
        else:
            started = self.clock()
            outcome = self.prefix_checker(
                copy.deepcopy(state.decoded_turns[:prefix_length]),
                copy.deepcopy(state.ground_truth[:prefix_length]),
                copy.deepcopy(state.original_test_entry),
            )
            state.checker_seconds += self.clock() - started
        valid = bool(outcome.get("valid")) and not force_quit
        if force_quit:
            outcome = {
                **outcome,
                "valid": False,
                "force_quit": True,
                "error_type": "multi_turn:force_quit",
            }

        if state.attempt == 1:
            state.retry_final_valid = valid
            if turn == state.trigger_turn:
                state.recovered = valid
            if not valid and state.retry_check_error is None:
                state.retry_check_error = copy.deepcopy(outcome)
            return
        if valid or state.oracle_decided:
            return

        state.oracle_decided = True
        state.trigger_turn = turn
        state.check_error = copy.deepcopy(outcome)
        retry_start_step = state.first_compressed_request_steps.get(turn)
        state.retry_eligibility = {
            "turn": turn,
            "eligible": retry_start_step is not None,
            "retry_start_step": retry_start_step,
            "observation_source": COMPRESSION_OBSERVATION_SOURCE,
        }
        if retry_start_step is None:
            state.status = "unrepairable_no_compressed_history"
            state.skip_reason = "no_compressed_history"
            return
        state.retry_start_step = retry_start_step
        try:
            state.oracle_values = gold_target_values(state.ground_truth[turn])
        except GoldLiteralError as error:
            state.status = "unrepairable_interface_unsupported"
            state.skip_reason = "interface_unsupported_nonliteral_gold"
            state.nonliteral = {
                "call_index": error.call_index,
                "field": error.field,
                "detail": error.detail,
            }
            return
        if not state.oracle_values:
            state.status = "unrepairable_interface_unsupported"
            state.skip_reason = "interface_unsupported_empty_gold_turn"
            return
        state.status = "retry_triggered"
        raise TurnRetry(turn)

    def begin_retry(self) -> str:
        state = self.state
        if state is None or state.trigger_turn is None:
            raise RuntimeError("BFCL retry requested without a trigger turn")
        state.attempt = 1
        state.current_turn = 0
        state.step = 0
        state.decoded_turns = []
        state.replay_positions = {}
        state.status = "retry_running"
        return f"c2kv_gold_retry_{state.task_id}_{uuid.uuid4().hex}"

    def cleanup_retry(self, namespace: str) -> None:
        state = self.state
        cleaned = cleanup_namespace_instances(namespace)
        if state is not None:
            state.retry_instances_cleaned += cleaned

    def record_handler_error(self, error: BaseException) -> None:
        state = self.state
        if state is None:
            return
        state.status = "handler_error"
        if state.base_error is None:
            state.base_error = {
                "type": type(error).__name__,
                "turn": state.current_turn,
                "step": state.step,
                "message": str(error),
            }

    def audit_row(self) -> Optional[Dict[str, Any]]:
        state = self.state
        if state is None:
            return None
        return {
            "schema": "bfcl_task_telemetry_v1",
            "oracle_kind": ORACLE_KIND if state.selector is not None else None,
            "oracle_version": ORACLE_VERSION if state.selector is not None else None,
            "task_id": state.task_id,
            "selector": state.selector,
            "oracle_enabled": state.selector is not None,
            "trigger_turn": state.trigger_turn,
            "retry_start_step": state.retry_start_step,
            "retry_eligibility": copy.deepcopy(state.retry_eligibility),
            "first_compressed_request_steps": [
                {"user_turn": turn, "step": step,
                 "observation_source": COMPRESSION_OBSERVATION_SOURCE}
                for turn, step in sorted(state.first_compressed_request_steps.items())
            ],
            "attempts": state.attempt + 1,
            "status": state.status,
            "skip_reason": state.skip_reason,
            "base_error": state.base_error,
            "check_error": state.check_error,
            "retry_check_error": state.retry_check_error,
            "nonliteral": state.nonliteral,
            "recovered": state.recovered,
            "retry_final_valid": state.retry_final_valid,
            "did_intervene": "appended" in state.intervention_statuses,
            "intervention_statuses": list(state.intervention_statuses),
            "replay_count": state.replay_count,
            "retry_instances_cleaned": state.retry_instances_cleaned,
            "gold_checker_seconds": state.checker_seconds,
            "total_wall_seconds": self.clock() - state.started_at,
        }

    def write_audit(self) -> Optional[Dict[str, Any]]:
        row = self.audit_row()
        if row is None or self.audit_path is None:
            return row
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_lock:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(
                    row, ensure_ascii=False, default=_audit_json_default) + "\n")
                handle.flush()
        return row


class MultiEventGoldRecoveryController(GoldRecoveryController):
    """Bounded v3 recovery over distinct failed user turns.

    Each retry starts a fresh BFCL simulator namespace. Responses before the
    failed suffix are replayed from the latest accepted trajectory, while new
    responses replace that suffix in the cache for any later retry.
    """

    def __init__(
        self,
        selector: str,
        max_events: int,
        audit_path: "Path | str | None" = None,
        *,
        ground_truth_loader: Callable[[Mapping[str, Any]], List[List[str]]] = (
            load_task_ground_truth),
        prefix_checker: Callable[
            [Sequence[Sequence[Sequence[str]]], Sequence[Sequence[str]], Mapping[str, Any]],
            Dict[str, Any],
        ] = official_prefix_check,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not isinstance(max_events, int) or isinstance(max_events, bool) or max_events <= 1:
            raise ValueError("BFCL multi-event recovery requires max_events > 1")
        if selector not in ("witness", "random"):
            raise ValueError("BFCL multi-event recovery requires a gold selector")
        super().__init__(
            selector,
            audit_path,
            ground_truth_loader=ground_truth_loader,
            prefix_checker=prefix_checker,
            clock=clock,
        )
        self.max_events = max_events

    @property
    def state(self) -> Optional[MultiEventTaskState]:
        return getattr(self.local, "state", None)

    def begin(self, test_entry: Mapping[str, Any]) -> MultiEventTaskState:
        state = MultiEventTaskState(
            task_id=str(test_entry["id"]),
            selector=self.selector,
            original_test_entry=copy.deepcopy(dict(test_entry)),
            ground_truth=self.ground_truth_loader(test_entry),
            started_at=self.clock(),
            max_events=self.max_events,
        )
        self.local.state = state
        return state

    def _active_event(self) -> Optional[Dict[str, Any]]:
        state = self.state
        if state is None or state.active_event_index is None:
            return None
        return state.events[state.active_event_index]

    def oracle_payload(self) -> Optional[Dict[str, Any]]:
        state = self.state
        event = self._active_event()
        if (state is None or event is None
                or state.current_turn != event["turn"]
                or state.step < event["start_step"]):
            return None
        return {
            "kind": MULTI_EVENT_ORACLE_KIND,
            "version": MULTI_EVENT_ORACLE_VERSION,
            "task_id": state.task_id,
            "turn": event["turn"],
            "retry_start_step": event["start_step"],
            "selector": state.selector,
            "values": list(event["oracle_values"]),
        }

    def replay_response(self) -> Optional[Tuple[Any, float]]:
        state = self.state
        event = self._active_event()
        if state is None or event is None:
            return None
        should_replay = state.current_turn < event["turn"]
        if state.current_turn == event["turn"]:
            should_replay = state.step < event["start_step"]
        if not should_replay:
            return None
        position = state.replay_positions.get(state.current_turn, 0)
        cached = state.response_cache.get(state.current_turn, [])
        if position >= len(cached):
            raise RuntimeError(
                f"missing cached BFCL response for {state.task_id} "
                f"turn={state.current_turn} step={position}")
        state.replay_positions[state.current_turn] = position + 1
        state.replay_count += 1
        state.step += 1
        response, _original_latency = cached[position]
        return response, 0.0

    def record_query(self, response: Any, latency: float) -> None:
        state = self.state
        if state is None:
            return
        state.http_call_count += 1
        cached = state.response_cache[state.current_turn]
        if len(cached) == state.step:
            cached.append((response, latency))
        elif len(cached) > state.step:
            cached[state.step] = (response, latency)
            del cached[state.step + 1:]
        else:
            raise RuntimeError(
                f"non-contiguous BFCL response cache for {state.task_id} "
                f"turn={state.current_turn} step={state.step}")

        if (_has_proxy_gist_tokens(response)
                and state.current_turn not in state.first_compressed_request_steps):
            state.first_compressed_request_steps[state.current_turn] = state.step

        event = self._active_event()
        if event is not None and state.current_turn == event["turn"]:
            status = _proxy_gold_status(response)
            if status is not None:
                event["extract_statuses"].append(status)
        state.step += 1

    def before_next_turn(self, handler: Any) -> None:
        state = self.state
        if state is None:
            return
        self._check_completed_prefix(force_quit=False)
        state.current_turn += 1
        state.step = 0

    def after_inference(self, handler: Any, metadata: Mapping[str, Any]) -> None:
        del handler
        state = self.state
        if state is None:
            return
        forced = _force_quit(metadata)
        if forced and state.base_error is None:
            state.base_error = {
                "type": "force_quit",
                "turn": state.current_turn,
            }
        state.final_valid = self._check_completed_prefix(force_quit=forced)
        if not state.events:
            state.status = "base_valid"
        elif state.blocked_reason is not None:
            state.status = f"completed_{state.blocked_reason}"
        elif state.final_valid:
            state.status = "multi_event_completed_valid"
        else:
            state.status = "multi_event_completed_invalid"

    def _prefix_outcome(self, *, force_quit: bool) -> Tuple[Dict[str, Any], bool]:
        state = self.state
        assert state is not None and state.ground_truth is not None
        prefix_length = state.current_turn + 1
        if len(state.decoded_turns) < prefix_length:
            outcome = {
                "valid": False,
                "multi_turn": {
                    "valid": False,
                    "error_type": "c2kv:missing_decoded_turn",
                },
                "irrelevance": {"valid": False},
            }
        else:
            started = self.clock()
            outcome = self.prefix_checker(
                copy.deepcopy(state.decoded_turns[:prefix_length]),
                copy.deepcopy(state.ground_truth[:prefix_length]),
                copy.deepcopy(state.original_test_entry),
            )
            state.checker_seconds += self.clock() - started
            state.checker_calls += 1
        valid = bool(outcome.get("valid")) and not force_quit
        if force_quit:
            outcome = {
                **outcome,
                "valid": False,
                "force_quit": True,
                "error_type": "multi_turn:force_quit",
            }
        return outcome, valid

    def _ineligible_event(
        self,
        outcome: Mapping[str, Any],
        *,
        start_step: Optional[int],
        reason: str,
        nonliteral: Optional[Dict[str, Any]] = None,
    ) -> None:
        state = self.state
        assert state is not None
        state.events.append({
            "turn": state.current_turn,
            "start_step": start_step,
            "eligibility": {
                "eligible": False,
                "reason": reason,
                "observation_source": COMPRESSION_OBSERVATION_SOURCE,
            },
            "success": None,
            "status": reason,
            "extract_statuses": [],
            "check_error": copy.deepcopy(outcome),
            "retry_check_error": None,
            "nonliteral": copy.deepcopy(nonliteral),
            "retry_attempt": None,
            "c2kv_oracle_event_key": None,
            "cost": None,
        })
        state.blocked_reason = reason

    def _finish_event_cost(self, event: Dict[str, Any]) -> None:
        state = self.state
        assert state is not None
        start = event.pop("_cost_start", None)
        started_at = event.pop("_retry_started_at", None)
        if not isinstance(start, dict) or started_at is None:
            return
        event["cost"] = {
            "scope": (
                "retry_begin_through_target_prefix_check_or_abort; includes "
                "retry replay, tool execution, target HTTP, and retry-prefix "
                "checker work; excludes the base failure-detector check and "
                "any post-abort ordinary trajectory"),
            "http_call_count": state.http_call_count - start["http_call_count"],
            "replay_count": state.replay_count - start["replay_count"],
            "checker_calls": state.checker_calls - start["checker_calls"],
            "checker_seconds": state.checker_seconds - start["checker_seconds"],
            "retry_wall_seconds": self.clock() - started_at,
            "base_failure_detector_checker_included": False,
            "replayed_original_http_latency_included": False,
        }

    def _check_completed_prefix(self, *, force_quit: bool) -> bool:
        state = self.state
        assert state is not None and state.ground_truth is not None
        outcome, valid = self._prefix_outcome(force_quit=force_quit)
        event = self._active_event()
        if event is not None:
            if state.current_turn < event["turn"]:
                if not valid:
                    event["success"] = False
                    event["status"] = "replayed_prefix_invalid"
                    event["abort_turn"] = state.current_turn
                    event["retry_check_error"] = copy.deepcopy(outcome)
                    state.blocked_reason = "replayed_prefix_invalid"
                    self._finish_event_cost(event)
                    state.active_event_index = None
                return valid
            event["success"] = valid
            if valid:
                event["status"] = "target_recovered"
            else:
                event["status"] = "target_failed"
                event["retry_check_error"] = copy.deepcopy(outcome)
                state.blocked_reason = "retry_target_failed"
            self._finish_event_cost(event)
            state.active_event_index = None
            return valid

        if valid or state.blocked_reason is not None:
            return valid
        turn = state.current_turn
        if turn in state.retried_turns:
            state.blocked_reason = "retry_target_failed"
            return valid

        start_step = state.first_compressed_request_steps.get(turn)
        if state.retries_started >= state.max_events:
            self._ineligible_event(
                outcome, start_step=start_step, reason="event_budget_exhausted")
            return valid
        if start_step is None:
            self._ineligible_event(
                outcome, start_step=None, reason="no_compressed_history")
            return valid

        try:
            oracle_values = gold_target_values(state.ground_truth[turn])
        except GoldLiteralError as error:
            self._ineligible_event(
                outcome,
                start_step=start_step,
                reason="interface_unsupported_nonliteral_gold",
                nonliteral={
                    "call_index": error.call_index,
                    "field": error.field,
                    "detail": error.detail,
                },
            )
            return valid
        if not oracle_values:
            self._ineligible_event(
                outcome,
                start_step=start_step,
                reason="interface_unsupported_empty_gold_turn",
            )
            return valid

        state.events.append({
            "turn": turn,
            "start_step": start_step,
            "eligibility": {
                "eligible": True,
                "reason": None,
                "observation_source": COMPRESSION_OBSERVATION_SOURCE,
            },
            "success": None,
            "status": "retry_triggered",
            "extract_statuses": [],
            "check_error": copy.deepcopy(outcome),
            "retry_check_error": None,
            "nonliteral": None,
            "oracle_values": oracle_values,
            "retry_attempt": None,
            "c2kv_oracle_event_key": [
                MULTI_EVENT_ORACLE_KIND,
                str(MULTI_EVENT_ORACLE_VERSION),
                state.task_id,
                turn,
            ],
            "cost": None,
        })
        state.active_event_index = len(state.events) - 1
        state.retried_turns.add(turn)
        raise TurnRetry(turn)

    def begin_retry(self) -> str:
        state = self.state
        event = self._active_event()
        if state is None or event is None:
            raise RuntimeError("BFCL retry requested without an active event")
        turn = int(event["turn"])
        start_step = int(event["start_step"])
        event["_cost_start"] = {
            "http_call_count": state.http_call_count,
            "replay_count": state.replay_count,
            "checker_calls": state.checker_calls,
            "checker_seconds": state.checker_seconds,
        }
        event["_retry_started_at"] = self.clock()
        state.response_cache[turn] = state.response_cache[turn][:start_step]
        for later_turn in [item for item in state.response_cache if item > turn]:
            del state.response_cache[later_turn]
        state.first_compressed_request_steps = {
            cached_turn: step
            for cached_turn, step in state.first_compressed_request_steps.items()
            if cached_turn < turn
        }
        state.attempt += 1
        state.retries_started += 1
        event["retry_attempt"] = state.attempt
        state.current_turn = 0
        state.step = 0
        state.decoded_turns = []
        state.replay_positions = {}
        event["status"] = "retry_running"
        state.status = "multi_event_retry_running"
        return (
            f"c2kv_gold_retry_v3_{state.task_id}_{state.retries_started}_"
            f"{uuid.uuid4().hex}")

    def record_handler_error(self, error: BaseException) -> None:
        super().record_handler_error(error)
        state = self.state
        event = self._active_event()
        if state is None or event is None:
            return
        event["success"] = False
        event["status"] = "handler_error"
        event["abort_turn"] = state.current_turn
        event["retry_check_error"] = {
            "valid": False,
            "error_type": type(error).__name__,
            "message": str(error),
        }
        state.blocked_reason = "handler_error"
        self._finish_event_cost(event)
        state.active_event_index = None

    def cleanup_retry(self, namespace: str) -> None:
        state = self.state
        cleaned = cleanup_namespace_instances(namespace)
        if state is not None:
            state.retry_namespaces_cleaned += 1
            state.retry_instances_cleaned += cleaned

    def audit_row(self) -> Optional[Dict[str, Any]]:
        state = self.state
        if state is None:
            return None
        extract_statuses = [
            status
            for event in state.events
            for status in event.get("extract_statuses", [])
        ]
        event_rows = copy.deepcopy(state.events)
        for event in event_rows:
            event.pop("oracle_values", None)
            for key in [name for name in event if name.startswith("_")]:
                event.pop(key, None)
        completed_costs = [
            event["cost"] for event in state.events
            if isinstance(event.get("cost"), Mapping)
        ]
        return {
            "schema": "bfcl_task_telemetry_v2",
            "oracle_kind": MULTI_EVENT_ORACLE_KIND,
            "oracle_version": MULTI_EVENT_ORACLE_VERSION,
            "task_id": state.task_id,
            "selector": state.selector,
            "oracle_enabled": True,
            "event_budget": state.max_events,
            "events": event_rows,
            "event_aggregation": {
                "scope": "distinct_failed_user_turns_with_latest_trajectory_replay",
                "detected_failures": len(state.events),
                "retries_started": state.retries_started,
                "target_recovered": sum(
                    event.get("success") is True for event in state.events),
                "target_failed": sum(
                    event.get("success") is False for event in state.events),
            },
            "event_cost_totals": {
                "scope": "sum of completed or aborted retry event costs",
                "events_with_cost": len(completed_costs),
                "http_call_count": sum(
                    item["http_call_count"] for item in completed_costs),
                "replay_count": sum(
                    item["replay_count"] for item in completed_costs),
                "checker_calls": sum(
                    item["checker_calls"] for item in completed_costs),
                "checker_seconds": sum(
                    item["checker_seconds"] for item in completed_costs),
                "retry_wall_seconds": sum(
                    item["retry_wall_seconds"] for item in completed_costs),
                "base_failure_detector_checker_included": False,
                "replayed_original_http_latency_included": False,
            },
            "legacy_scalar_fields_scope": "not_applicable_for_multi_event; use events",
            "trigger_turn": None,
            "retry_start_step": None,
            "retry_eligibility": None,
            "check_error": None,
            "retry_check_error": None,
            "nonliteral": None,
            "recovered": None,
            "retry_final_valid": None,
            "first_compressed_request_steps": [
                {"user_turn": turn, "step": step,
                 "observation_source": COMPRESSION_OBSERVATION_SOURCE}
                for turn, step in sorted(state.first_compressed_request_steps.items())
            ],
            "attempts": state.attempt + 1,
            "retry_count": state.retries_started,
            "status": state.status,
            "skip_reason": state.blocked_reason,
            "base_error": state.base_error,
            "final_valid": state.final_valid,
            "did_intervene": "appended" in extract_statuses,
            "intervention_statuses": extract_statuses,
            "replay_count": state.replay_count,
            "http_call_count": state.http_call_count,
            "gold_checker_calls": state.checker_calls,
            "retry_instances_cleaned": state.retry_instances_cleaned,
            "retry_namespaces_cleaned": state.retry_namespaces_cleaned,
            "gold_checker_seconds": state.checker_seconds,
            "total_wall_seconds": self.clock() - state.started_at,
        }


def summarize_audit(path: "Path | str") -> Dict[str, Any]:
    """Return counts without changing the official BFCL score denominator."""
    audit_path = Path(path)
    rows = []
    if audit_path.exists():
        with audit_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    oracle_rows = [row for row in rows if row.get("selector") is not None]
    multi_event_rows = [
        row for row in oracle_rows
        if row.get("oracle_kind") == MULTI_EVENT_ORACLE_KIND
    ]
    event_rows = [
        event
        for row in multi_event_rows
        for event in (row.get("events") or [])
        if isinstance(event, dict)
    ]
    reasons: Dict[str, int] = {}
    for row in oracle_rows:
        if row.get("oracle_kind") == MULTI_EVENT_ORACLE_KIND:
            for event in row.get("events") or []:
                eligibility = event.get("eligibility") or {}
                reason = eligibility.get("reason")
                if reason:
                    reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        else:
            reason = row.get("skip_reason")
            if reason:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    def _triggered(row: Mapping[str, Any]) -> bool:
        if row.get("oracle_kind") == MULTI_EVENT_ORACLE_KIND:
            return bool(row.get("events"))
        return row.get("trigger_turn") is not None

    def _recovered(row: Mapping[str, Any]) -> bool:
        if row.get("oracle_kind") == MULTI_EVENT_ORACLE_KIND:
            return any(event.get("success") is True
                       for event in row.get("events") or [])
        return row.get("recovered") is True

    def _final_valid(row: Mapping[str, Any]) -> bool:
        if row.get("oracle_kind") == MULTI_EVENT_ORACLE_KIND:
            return row.get("final_valid") is True
        return row.get("retry_final_valid") is True

    def _intervened(row: Mapping[str, Any]) -> bool:
        return row.get("did_intervene") is True

    return {
        "task_telemetry_path": str(audit_path),
        "n_task_telemetry": len(rows),
        "n_oracle_tasks": len(oracle_rows),
        "n_oracle_triggered": sum(
            _triggered(row) for row in oracle_rows),
        "n_oracle_recovered": sum(
            _recovered(row) for row in oracle_rows),
        "n_retry_final_valid": sum(
            _final_valid(row) for row in oracle_rows),
        "n_oracle_intervened": sum(
            _intervened(row) for row in oracle_rows),
        "n_no_witness": sum(
            "no_literal_witness" in (row.get("intervention_statuses") or [])
            for row in oracle_rows),
        "n_no_compressed_history": sum(
            ("no_compressed_history" in (row.get("intervention_statuses") or [])
             or row.get("skip_reason") == "no_compressed_history")
            for row in oracle_rows),
        "n_eligible_unrepairable": sum(reasons.values()),
        "n_interface_unsupported": sum(
            count for reason, count in reasons.items()
            if reason.startswith("interface_unsupported_")),
        "n_earlier_unrepairable_turn_failure": reasons.get(
            "earlier_unrepairable_turn_failure", 0),
        "eligible_unrepairable_by_reason": reasons,
        "oracle_summary_scope": {
            "legacy_task_counts": (
                "v2 single-event task rows; v3 counts tasks with any matching event"),
            "event_counts": "v3 event rows only",
        },
        "n_oracle_events": len(event_rows),
        "n_oracle_event_retries": sum(
            event.get("eligibility", {}).get("eligible") is True
            for event in event_rows),
        "n_oracle_event_targets_recovered": sum(
            event.get("success") is True for event in event_rows),
        "n_oracle_event_targets_failed": sum(
            event.get("success") is False for event in event_rows),
        "n_oracle_event_budget_exhausted": sum(
            event.get("status") == "event_budget_exhausted"
            for event in event_rows),
        "total_oracle_retries": sum(
            int(row.get("retry_count", 0))
            if row.get("oracle_kind") == MULTI_EVENT_ORACLE_KIND
            else int(row.get("attempts", 1)) - 1
            for row in oracle_rows),
        "total_gold_checker_seconds": sum(
            float(row.get("gold_checker_seconds", 0.0)) for row in oracle_rows),
        "total_handler_http_calls": sum(
            int(row.get("http_call_count", 0)) for row in multi_event_rows),
        "total_handler_http_calls_scope": "v3 task rows only",
    }
