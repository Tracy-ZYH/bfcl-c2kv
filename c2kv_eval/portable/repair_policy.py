"""Repair-policy parsing and span selection (stdlib-only, unit-tested).

Shared by hf_server (the corr-append arm) and the CPU tests.  Grammar
(docs/hybrid_spec.md "Repair interaction"):

* ``first``       — doc 0 (the D harness corr@first)
* ``offset:<j>``  — DOC j: one history message; the span is ALL extract
                    chunks of that doc.  This matches
                    agent/d_kv_intervene.py --corr_k_policy offset:<j>,
                    which indexes docs (the pre-v2 server counted CHUNKS —
                    the two agreed only at j=0).
* ``chunk:<i>``   — explicit extract-chunk index across the compressed
                    history (768-token units; one long doc = many chunks).

Selection raises on out-of-range indices and on a target doc that has no
extract chunks — a repair target that resolves to nothing must fail loudly,
never silently concatenate the whole scratch cache.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def parse_policy(policy: str) -> Dict[str, Optional[int]]:
    """'first' | 'offset:<j>' | 'chunk:<i>' -> {kind, index}.

    Raises ValueError on anything else (the caller maps it to a 4xx).
    """
    policy = str(policy or "").strip()
    if policy == "first":
        return {"kind": "doc", "index": 0}
    if policy.startswith("offset:"):
        try:
            value = int(policy.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"bad offset policy {policy!r}") from None
        if value < 0:
            raise ValueError(f"negative offset {value}")
        return {"kind": "doc", "index": value}
    if policy.startswith("chunk:"):
        try:
            value = int(policy.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"bad chunk policy {policy!r}") from None
        if value < 0:
            raise ValueError(f"negative chunk index {value}")
        return {"kind": "chunk", "index": value}
    raise ValueError(f"unknown policy {policy!r}")


def span_selection(doc_chunk_counts: List[int],
                   kind: str, index: int) -> Tuple[int, int, int]:
    """Which (doc, first_chunk, n_chunks) the corr append must span.

    ``doc_chunk_counts[i]`` = number of extract chunks of compressed doc i.
    Returns (doc_index, first_chunk_index, chunk_count).  Raises ValueError
    when the index is out of range or the selected doc has zero chunks
    (a doc whose extract produced no blocks can never be a repair target).
    """
    if kind not in ("doc", "chunk"):
        raise ValueError(f"unknown selection kind {kind!r}")
    total_chunks = sum(doc_chunk_counts)
    if kind == "doc":
        if not 0 <= index < len(doc_chunk_counts):
            raise ValueError(
                f"offset {index} out of range (0..{len(doc_chunk_counts) - 1} docs)")
        if doc_chunk_counts[index] <= 0:
            raise ValueError(f"doc {index} has no extract chunks")
        first = sum(doc_chunk_counts[:index])
        return index, first, doc_chunk_counts[index]
    if not 0 <= index < total_chunks:
        raise ValueError(f"chunk {index} out of range (0..{total_chunks - 1})")
    cumulative = 0
    for doc_index, count in enumerate(doc_chunk_counts):
        if cumulative + count > index:
            # the span is exactly the ONE requested chunk
            return doc_index, index, 1
        cumulative += count
    raise ValueError("unreachable")
