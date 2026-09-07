"""Backend factory.  proxy.py selects one via --backend; benchmarks and
adapters stay backend-agnostic."""
from __future__ import annotations

from typing import Callable

from .base import Backend, BackendError
from .sglang import SglangBackend

BACKENDS = {"sglang": SglangBackend}


def get_backend(name: str, post_json: Callable) -> Backend:
    try:
        cls = BACKENDS[name]
    except KeyError:
        raise SystemExit(
            f"FATAL: unknown backend {name!r}; known: {', '.join(sorted(BACKENDS))}")
    return cls(post_json)
