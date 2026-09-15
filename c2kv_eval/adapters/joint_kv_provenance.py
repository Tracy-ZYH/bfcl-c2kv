"""Model-free joint KV planning; not an evaluator or a compression algorithm.

Boundaries MUST come from the authoritative existing renderer/span mapper.
No tokenizer, chat-template inference or global-mask conversion occurs here.
This module is not yet wired to serving: see JOINT_KV_STATUS.md.
"""
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SemanticSpan:
    kind: str
    unit_id: str
    token_start: int
    token_end: int


@dataclass(frozen=True)
class RetainedPositions:
    frame_id: str
    layout: str
    # Axis = (layer, KV head); shared-layout callers use a single explicit axis.
    axes: Mapping[tuple[int, int], tuple[int, ...]]


def validate_spans(spans, full_tokens):
    cursor = 0
    units = set()
    for span in spans:
        if span.kind not in {"tool", "history", "protected"}:
            raise ValueError("Unknown semantic span kind")
        if span.token_start != cursor or not cursor < span.token_end <= full_tokens:
            raise ValueError("Provenance must partition the exact full token frame")
        key = (span.kind, span.unit_id)
        if key in units:
            raise ValueError("Duplicate semantic unit ID")
        units.add(key)
        cursor = span.token_end
    if cursor != full_tokens:
        raise ValueError("Unmapped full-context tokens")


def merge_retained(spans, full_tokens, frame_id, *, tool=None, history=None,
                   compress_tools=False, compress_history=False):
    """Merge existing algorithm selections independently PER LAYER/HEAD.

    Selections contain absolute original token positions, not compact offsets.
    Layout mismatch fails closed rather than unioning heads into a global mask.
    """
    spans = list(spans)
    validate_spans(spans, full_tokens)
    groups = {kind: {p for s in spans if s.kind == kind
                     for p in range(s.token_start, s.token_end)}
              for kind in ("tool", "history", "protected")}
    selections = []
    for kind, enabled, selection in (("tool", compress_tools, tool),
                                     ("history", compress_history, history)):
        if not enabled:
            continue
        if selection is None or selection.frame_id != frame_id or not selection.axes:
            raise ValueError("Missing selection or mismatched full-context frame")
        for positions in selection.axes.values():
            if len(positions) != len(set(positions)) or not set(positions) <= groups[kind]:
                raise ValueError("Selection escapes semantic span or duplicates tokens")
        selections.append(selection)
    if selections:
        layout, axes = selections[0].layout, set(selections[0].axes)
        if any(s.layout != layout or set(s.axes) != axes for s in selections):
            raise ValueError("Incompatible layer/head layouts; no global-mask fallback")
    else:
        layout, axes = "shared_token_indices_across_layers_heads", {(0, 0)}
    result = {}
    for axis in axes:
        keep = set(groups["protected"])
        keep.update(tool.axes[axis] if compress_tools else groups["tool"])
        keep.update(history.axes[axis] if compress_history else groups["history"])
        result[axis] = tuple(sorted(keep))
    return RetainedPositions(frame_id, layout, result)


def reusable_chunks(spans, full_tokens, *, compress_tools, compress_history):
    """CacheBlend plan only: fresh spans are never made eviction candidates.

    Returns authoritative chunks with a reuse flag. Noncontiguous reusable
    spans require a multi-span serving interface, not the current single span.
    """
    spans = list(spans)
    validate_spans(spans, full_tokens)
    return [(s, (s.kind == "tool" and compress_tools)
             or (s.kind == "history" and compress_history)) for s in spans]


@dataclass
class JointKVAccumulator:
    full_tool_kv: int = 0
    active_tool_kv: int = 0
    full_history_kv: int = 0
    active_history_kv: int = 0
    protected_kv: int = 0
    joint_active_kv: int = 0

    def add(self, spans, full_tokens, retained):
        spans = list(spans)
        validate_spans(spans, full_tokens)
        for positions in retained.axes.values():
            keep = set(positions)
            if len(keep) != len(positions) or not keep <= set(range(full_tokens)):
                raise ValueError("Invalid resident positions")
            for span in spans:
                full = span.token_end - span.token_start
                active = sum(p in keep for p in range(span.token_start, span.token_end))
                if span.kind == "protected":
                    if active != full:
                        raise ValueError("System/current scaffold must remain full")
                    self.protected_kv += full
                else:
                    setattr(self, f"full_{span.kind}_kv", getattr(self, f"full_{span.kind}_kv") + full)
                    setattr(self, f"active_{span.kind}_kv", getattr(self, f"active_{span.kind}_kv") + active)
            self.joint_active_kv += len(keep)

    def summary(self):
        def ratio(full, active):
            return full / active if full and active else None
        full = self.full_tool_kv + self.full_history_kv
        active = self.active_tool_kv + self.active_history_kv
        return {**vars(self), "tool_compression_ratio": ratio(self.full_tool_kv, self.active_tool_kv),
                "history_compression_ratio": ratio(self.full_history_kv, self.active_history_kv),
                "joint_compression_ratio": ratio(full, active),
                "end_to_end_kv_retention": self.joint_active_kv / (full + self.protected_kv)
                if full + self.protected_kv else None,
                "count_unit": "token slots summed across represented layer/head axes"}
