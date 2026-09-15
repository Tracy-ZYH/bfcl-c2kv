import pytest
from c2kv_eval.adapters.joint_kv_provenance import (
    SemanticSpan as S, RetainedPositions as R, merge_retained,
    reusable_chunks, JointKVAccumulator,
)

SPANS = [S("protected", "system", 0, 2), S("tool", "tools", 2, 6),
         S("history", "turn0", 6, 10), S("protected", "current", 10, 12)]
AXES = {(0, 0): (2,), (0, 1): (5,)}
TOOL = R("full1", "per_head", AXES)
HISTORY = R("full1", "per_head", {(0, 0): (6, 8), (0, 1): (7, 9)})


def test_disabled_is_full():
    r = merge_retained(SPANS, 12, "full1")
    assert r.axes[(0, 0)] == tuple(range(12))


@pytest.mark.parametrize("tools,history", [(True, False), (False, True), (True, True)])
def test_scopes_preserve_axes_protected_and_original_positions(tools, history):
    r = merge_retained(SPANS, 12, "full1", tool=TOOL, history=HISTORY,
                       compress_tools=tools, compress_history=history)
    for axis, keep in r.axes.items():
        assert {0, 1, 10, 11} <= set(keep)
        assert set(keep) & set(range(2, 6)) == set(TOOL.axes[axis] if tools else range(2, 6))
        assert set(keep) & set(range(6, 10)) == set(HISTORY.axes[axis] if history else range(6, 10))


def test_layout_frame_and_boundary_mismatches_fail():
    for bad in [R("other", "per_head", HISTORY.axes),
                R("full1", "shared", {(0, 0): (6,)}),
                R("full1", "per_head", {(0, 0): (0,), (0, 1): (7,)})]:
        with pytest.raises(ValueError):
            merge_retained(SPANS, 12, "full1", tool=TOOL, history=bad,
                           compress_tools=True, compress_history=True)
    with pytest.raises(ValueError):
        merge_retained(SPANS[1:], 12, "full1")


def test_cacheblend_does_not_evict_protected_chunks():
    chunks = reusable_chunks(SPANS, 12, compress_tools=True, compress_history=True)
    assert [reuse for _, reuse in chunks] == [False, True, True, False]


def test_metrics_use_cumulative_counts_not_mean_ratios():
    r = merge_retained(SPANS, 12, "full1", tool=TOOL, history=HISTORY,
                       compress_tools=True, compress_history=True)
    acc = JointKVAccumulator()
    acc.add(SPANS, 12, r)
    acc.add(SPANS, 12, merge_retained(SPANS, 12, "full1"))
    out = acc.summary()
    assert out["joint_compression_ratio"] == 24 / 14
    assert out["end_to_end_kv_retention"] == 26 / 36
