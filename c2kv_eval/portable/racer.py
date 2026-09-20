"""Frozen T02 Racer bridge; no training or backend-specific detector."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
from typing import Any, Mapping

ARTIFACT = Path("/home/zhuyuhan/project/c2kv/experiments/history_system/artifacts/c1_risk.t02_v1.json")
_RUNTIME = Path("/home/zhuyuhan/project/c2kv/experiments/history_system/runtime")
_RUNTIME_PYTHON = _RUNTIME / "python"
for _root in (_RUNTIME_PYTHON, _RUNTIME):
    if str(_root) not in sys.path: sys.path.insert(0, str(_root))
from benchmarks.memory_runtime.recovery.set_models import C1RiskSelector

class RacerError(RuntimeError): pass

class FrozenT02Racer:
    threshold = 0.5
    artifact_path = str(ARTIFACT)
    def __init__(self, artifact: str | Path = ARTIFACT):
        self.artifact_path = str(Path(artifact).resolve())
        self.selector = C1RiskSelector(self.artifact_path, threshold=self.threshold)
        self.calls = 0
    def score(self, *, hidden: Any, logprobs: list[float], is_stop: bool, parse_ok: bool) -> dict[str, Any]:
        if not isinstance(hidden, list) or not hidden: raise RacerError("prompt-last hidden unavailable")
        vals = [float(x) for x in logprobs if x is not None and math.isfinite(float(x))]
        if not vals: raise RacerError("draft logprobs unavailable")
        context = {"prefill_hidden": hidden, "draft_logprobs": vals,
                   "is_stop": bool(is_stop), "parse_ok": bool(parse_ok),
                   "prefill_contract": self.selector.artifact["feature_contract"]["prefill_contract"]}
        result = self.selector.model.predict_risk(context)
        self.calls += 1
        if not result.available or result.score is None: raise RacerError(result.reason or "T02 score unavailable")
        return {"risk_score": float(result.score), "threshold": self.threshold,
                "triggered": float(result.score) > self.threshold,
                "detector_artifact": self.artifact_path, "detector_calls": 1}

def semantic_units(messages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for i,m in enumerate(messages):
        if m.get("role") in {"user","assistant","tool"}:
            text=m.get("content") or ""
            if isinstance(text, str) and text:
                out.append({"unit_id": f"turn-{i}", "turn_id": i, "role": m.get("role"), "content": text, "original_token_span": None})
    return out
