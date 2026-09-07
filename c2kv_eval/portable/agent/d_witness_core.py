"""Witness-IDF k* selection — prereg v2.2, frozen algorithm (2026-08-30).

Pure functions, deliberately torch-free (unit-testable on the Windows box,
importable by the server-side selector that needs the tokenizer).

Frozen semantics (do not modify without amending the prereg):

```python
texts = [tokenizer.decode(ids) for ids in doc_ids]      # decoded grid rows
values = [target_tool_name] + leaves(target_args)       # tool name + arg leaves

def occurs(v, t):
    s = str(v)
    return s in t if len(s) >= 8 else \\
           bool(re.search(rf"(?<![\\w.]){re.escape(s)}(?![\\w.])", t))

df    = {v: sum(occurs(v, t) for t in texts) for v in values}
score = [sum(1 / df[v] for v in values if occurs(v, texts[i])) for i in range(n)]

k_star = argmax(score) if max(score) > 0 else None
```

1. ``1/df`` is the ENTIRE localization power — no additional filtering.
   Values present in every doc add the same 1/n to each block and cancel in
   argmax; values present in exactly one doc add 1.0.
2. ``texts`` are the DECODED grid rows (post-``max_doc_length`` truncation,
   post chat-template rendering) — the text the model actually saw — never
   the raw dataset JSON.
3. ``k_star is None`` is a RESULT, not an exception: qids whose target has
   no literal witness in history (synthesized free-text arguments) are
   marked by the algorithm itself.

Implementation notes (frozen alongside):
- ``leaves`` yields JSON leaf values in JSON literal form: strings verbatim,
   numbers via str(), booleans as ``true``/``false``, null as ``null``
   (str(True)="True" would never match rendered JSON).
- duplicate values are deduplicated before scoring (a repeated value must
   not double its 1/df contribution); empty strings are dropped (zero
   information).
- ties resolve to the LOWEST index (deterministic argmax).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


def leaves(value) -> List[str]:
    """JSON leaf values as strings, depth-first, object keys not included."""
    if isinstance(value, dict):
        out: List[str] = []
        for v in value.values():
            out.extend(leaves(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(leaves(v))
        return out
    if isinstance(value, bool):  # before int: bool is an int subclass
        return ["true" if value else "false"]
    if value is None:
        return ["null"]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def target_values(tool_name: Optional[str], target_args) -> List[str]:
    """[tool_name] + leaves(args), deduplicated, order-preserving, no ''."""
    raw: List[str] = ([str(tool_name)] if tool_name else []) + leaves(target_args)
    return list(dict.fromkeys(v for v in raw if v))


def occurs(value: str, text: str) -> bool:
    s = str(value)
    if len(s) >= 8:
        return s in text
    return bool(re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", text))


def witness_scores(
    texts: Sequence[str],
    values: Sequence[str],
) -> Tuple[Dict[str, int], List[float]]:
    """Document frequency per value + per-doc IDF localization score."""
    df = {v: sum(1 for t in texts if occurs(v, t)) for v in values}
    scores = [
        sum(1.0 / df[v] for v in values if df[v] > 0 and occurs(v, t))
        for t in texts
    ]
    return df, scores


def select_k_star(texts: Sequence[str], values: Sequence[str]) -> Optional[int]:
    """argmax of the witness score; None when no value occurs anywhere."""
    if not texts:
        return None
    _, scores = witness_scores(texts, values)
    if max(scores) <= 0:
        return None
    best = max(range(len(scores)), key=lambda i: scores[i])  # first max on ties
    return int(best)
