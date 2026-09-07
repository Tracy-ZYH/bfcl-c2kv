"""One adapter contract for every benchmark.

``benchmarks/run.py`` owns the proxy lifecycle, the run-name/out-dir sha
rule and the summary envelope; everything benchmark-specific lives behind
this contract, so adding a benchmark is one file plus one registry entry.

An adapter module exposes three names:

``NAME``            the ``--benchmark`` value it answers to (``NAMES`` when
                    one module serves several, e.g. the ACON runners);
``add_arguments``   registers *its own* CLI flags on run.py's parser — a
                    flag shared by two adapters stays in run.py's core
                    block, because argparse refuses a duplicate option
                    string;
``run(ctx)``        does the work and returns the summary dict.

The adapter owns everything about how the harness is reached: the ``/v1``
suffix (``v1()`` here is the single implementation) and any working
directory change (inside the adapter's own ``try/finally``).  run.py hands
it a :class:`RunContext` and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def v1(url: str) -> str:
    """OpenAI-compatible base url: exactly one ``/v1``, no trailing slash."""
    return url.rstrip("/") + "/v1"


@dataclass
class RunContext:
    """Everything an adapter is given.

    ``base_url``      the arm proxy (agent traffic; compressed).
    ``user_base_url`` the raw upstream (user simulator / judge; never
                      compressed) — benchmarks without a simulator ignore it.
    ``out_dir``       the run's output directory (already sha-suffixed and
                      created by run.py).
    ``model``         served model name at the endpoint.
    ``arm``           the arm the proxy runs (adapters need it only for
                      naming, never for semantics).
    ``run_name``      sha-suffixed run name (tau2 --save-to, ACON tag, ...).
    ``request_log``   the proxy request log being written for this run, for
                      the per-task cost join; ``None`` disables the join.
    ``options``       the parsed CLI namespace as a dict — the flags each
                      adapter registered in its own ``add_arguments``.
    """

    base_url: str
    user_base_url: str
    out_dir: Path
    model: str
    arm: str
    run_name: str = "c2kv_run"
    request_log: Optional[Path] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def opt(self, name: str, default: Any = None) -> Any:
        """CLI option, with run.py's "empty means default" convention.

        ``argparse`` gives absent string/list flags ``""`` / ``[]``; every
        adapter used to spell ``kwargs.get(x) or DEFAULT`` by hand.  ``0``
        and ``False`` are values, not emptiness, and survive.
        """
        value = self.options.get(name, default)
        if value is None or value == "" or value == []:
            return default
        return value
