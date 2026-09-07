"""Arm registry for the modular benchmark layer.

An arm fully describes how conversation history reaches the model, so the
proxy and the metrics layer share one source of truth.  Hybrid semantics are
defined once in docs/hybrid_spec.md (canonical gist_first layout):

* full    — every message is sent as raw text.
* c2kv    — every history message is compressed via /v1/c2kv/extract and
            referenced by c2kv_key_hash; the current turn stays raw.
* hybrid  — top-k tail of history stays raw, the rest is compressed
            (hybrid / hybrid_k1 / hybrid_k5 = k 3/1/5).
* *_repair — corr@first on the compressed prefix (docs/hybrid_spec.md
            "Repair interaction", docs/c2kv_semantics.md "Repair placement"):
            the raw KV of the policy-selected compressed doc, computed in its
            full context, is injected with an explicit placement:
              append_keep_ledger (default) = D-harness corr / keepG, the
                doc's gist stays, the span keeps its original RoPE phase;
              append_tail (*_repair_tail) = D-harness raw_erratum_tail, the
                span is re-rotated to the end of history and the ledger
                advances;
              in_place (*_repair_inplace) = D-harness replaceG, the span
                replaces the doc's gist.
* history_kv_<method>_r<per-mille> — the upstream history-KV eviction
            baselines (StreamingLLM / H2O / SnapKV / PyramidKV) ported from
            kvoffload-sglang ``c2kv_eval.adapters.bfcl_history_kv_baselines``.
            No gist compression: the completed history is prefilled once by
            ``/v1/c2kv/repair_extract`` with ``history_kv_method`` +
            ``history_kv_retention_ratio``, the server selects the surviving
            token slots and stores them as ONE repair entry, and the chat
            request carries that entry on a repair-only carrier message in
            place of the history text.  See README "History-KV eviction arms"
            for the deviations from the upstream client.
* *_recover — step-level oracle recover (docs/hybrid_spec.md "Oracle
            recover"): the proxy diffs every generated action against a
            full-arm reference trajectory keyed by message fingerprint; at
            the first divergence it re-sends the identical payload assembled
            full-raw (the reference KV regime) and returns the regenerated
            step.  One repair per conversation; later mismatches are logged
            as re_diverged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


REPAIR_PLACEMENTS = ("append_keep_ledger", "append_tail", "in_place")

# ---- history-KV eviction baselines (upstream parity) --------------------
# Names and defaults mirror kvoffload-sglang
# ``scripts/run_history_kv_baselines.sh`` (HISTORY_KV_* env defaults) and
# ``c2kv_eval.adapters.bfcl_history_kv_baselines.parse_args``.  The server
# accepts the two aliases below and normalizes them the same way
# (scheduler._build_history_kv_eviction_rounds, qwen3.generate_raw_repair_kv).
HISTORY_KV_METHODS = ("streamingllm", "h2o", "snapkv_persistent", "pyramidkv")
HISTORY_KV_METHOD_ALIASES = {"snapkv": "snapkv_persistent", "pyramid": "pyramidkv"}
# ``--runtime-history-kv-backend`` in the upstream runner.
HISTORY_KV_BACKENDS = ("repair_extract", "physical_eviction")
HISTORY_KV_POOLINGS = ("avgpool", "maxpool")
HISTORY_KV_DEFAULTS: Dict[str, Any] = {
    "backend": "repair_extract",
    "retention_ratio": None,      # --history-kv-retention-ratio (0.312)
    "target_tokens": None,        # absolute budget; overrides the ratio
    "recent_window": 64,          # --history-kv-recent-window
    "kernel_size": 5,             # --history-kv-kernel-size
    "pooling": "avgpool",         # --history-kv-pooling
    "h2o_recent_fraction": 0.5,   # --history-kv-h2o-recent-fraction
    "persistent_session": False,  # --persistent-history-kv-session
}


def history_kv_spec(arm: "Arm") -> Optional[Dict[str, Any]]:
    """Canonical, fully defaulted history-KV spec of ``arm`` (or None).

    Raises ValueError on an unusable spec, so an arm can never reach the wire
    with a method or a budget the server would reject.
    """
    config = getattr(arm, "history_kv", None)
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"arm {arm.name!r}: history_kv must be a dict")
    unknown = set(config) - set(HISTORY_KV_DEFAULTS) - {"method"}
    if unknown:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv keys {sorted(unknown)}")
    spec: Dict[str, Any] = dict(HISTORY_KV_DEFAULTS)
    spec.update(config)
    method = str(spec.get("method") or "").strip().lower()
    method = HISTORY_KV_METHOD_ALIASES.get(method, method)
    if method not in HISTORY_KV_METHODS:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv method {spec.get('method')!r}; "
            f"expected one of {HISTORY_KV_METHODS} (aliases: "
            f"{sorted(HISTORY_KV_METHOD_ALIASES)})")
    spec["method"] = method
    backend = str(spec["backend"])
    if backend not in HISTORY_KV_BACKENDS:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv backend {backend!r}; "
            f"expected one of {HISTORY_KV_BACKENDS}")
    if str(spec["pooling"]) not in HISTORY_KV_POOLINGS:
        raise ValueError(
            f"arm {arm.name!r}: unknown history_kv pooling {spec['pooling']!r}")
    ratio = spec["retention_ratio"]
    target = spec["target_tokens"]
    if ratio is None and target is None:
        raise ValueError(
            f"arm {arm.name!r}: history_kv needs retention_ratio or target_tokens")
    if ratio is not None and not 0.0 < float(ratio) <= 1.0:
        raise ValueError(
            f"arm {arm.name!r}: history_kv retention_ratio must be in (0, 1]")
    if target is not None and int(target) < 1:
        raise ValueError(f"arm {arm.name!r}: history_kv target_tokens must be >= 1")
    if int(spec["recent_window"]) < 1 or int(spec["kernel_size"]) < 1:
        raise ValueError(
            f"arm {arm.name!r}: history_kv recent_window/kernel_size must be >= 1")
    if not 0.0 <= float(spec["h2o_recent_fraction"]) <= 1.0:
        raise ValueError(
            f"arm {arm.name!r}: history_kv h2o_recent_fraction must be in [0, 1]")
    if backend == "physical_eviction" and target is None:
        # The physical path takes an ABSOLUTE budget: the scheduler reads
        # config["target_tokens"] only (scheduler.py
        # _select_history_kv_eviction_indices, mem_cache/history_kv_eviction.py
        # PhysicalHistoryKVEvictor.evict) and never derives it from a ratio,
        # and this proxy has no tokenizer with which to convert one.  See
        # README "History-KV eviction arms".
        raise ValueError(
            f"arm {arm.name!r}: history_kv backend 'physical_eviction' needs an "
            "absolute target_tokens (the server has no retention_ratio on that "
            "path and the proxy has no tokenizer)")
    if spec["persistent_session"] and backend != "physical_eviction":
        raise ValueError(
            f"arm {arm.name!r}: history_kv persistent_session requires backend "
            "'physical_eviction' (upstream run_history_kv_baselines.sh)")
    return spec


# ---- KV reuse with selective recompute (CacheBlend) ---------------------
# CacheBlend (Yao et al., EuroSys 2025, arXiv 2405.16444) served through the
# reconciled server's /v1/c2kv/repair_extract ``kv_reuse_method="cacheblend"``
# (c2kv_serving_semantics.md section 10, mem_cache/cacheblend.py).  Defaults
# mirror the EuroSys artifact (vllm_blend llama.py cache_fuse_metadata:
# check_layers=[1], recomp_ratio=0.16, V-deviation); ``metric="k"`` with
# ratio 0.15 is the LMCache-monorepo lineage (blender.py).
KV_REUSE_METHODS = ("cacheblend",)
KV_REUSE_METRICS = ("v", "k")
KV_REUSE_MASKS = ("causal", "bottom_right")
KV_REUSE_CHUNKINGS = ("doc", "grid")
KV_REUSE_DEFAULTS: Dict[str, Any] = {
    "recomp_ratio": 0.16,     # cacheblend_recomp_ratio (artifact recomp_ratio)
    "check_layer": 1,         # cacheblend_check_layer  (artifact check_layers[0])
    "metric": "v",            # cacheblend_metric: v (artifact) | k (LMCache)
    "mask": "causal",         # cacheblend_mask: exact causality | artifact bottom_right
    "chunking": "doc",        # doc = one chunk per history doc | grid = chunk_tokens
    "chunk_tokens": None,     # cacheblend_chunk_tokens (grid only)
}


def kv_reuse_spec(arm: "Arm") -> Optional[Dict[str, Any]]:
    """Canonical, fully defaulted KV-reuse spec of ``arm`` (or None).

    Raises ValueError on an unusable spec, so an arm can never reach the wire
    with a method or a knob the server would reject.
    """
    config = getattr(arm, "kv_reuse", None)
    if not config:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"arm {arm.name!r}: kv_reuse must be a dict")
    unknown = set(config) - set(KV_REUSE_DEFAULTS) - {"method"}
    if unknown:
        raise ValueError(
            f"arm {arm.name!r}: unknown kv_reuse keys {sorted(unknown)}")
    spec: Dict[str, Any] = dict(KV_REUSE_DEFAULTS)
    spec.update(config)
    method = str(spec.get("method") or "").strip().lower()
    if method not in KV_REUSE_METHODS:
        raise ValueError(
            f"arm {arm.name!r}: unknown kv_reuse method {spec.get('method')!r}; "
            f"expected one of {KV_REUSE_METHODS}")
    spec["method"] = method
    ratio = float(spec["recomp_ratio"])
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"arm {arm.name!r}: kv_reuse recomp_ratio must be in [0, 1]")
    spec["recomp_ratio"] = ratio
    if int(spec["check_layer"]) < 0:
        raise ValueError(f"arm {arm.name!r}: kv_reuse check_layer must be >= 0")
    spec["check_layer"] = int(spec["check_layer"])
    spec["metric"] = str(spec["metric"]).lower()
    if spec["metric"] not in KV_REUSE_METRICS:
        raise ValueError(
            f"arm {arm.name!r}: kv_reuse metric must be one of {KV_REUSE_METRICS}")
    spec["mask"] = str(spec["mask"]).lower()
    if spec["mask"] not in KV_REUSE_MASKS:
        raise ValueError(
            f"arm {arm.name!r}: kv_reuse mask must be one of {KV_REUSE_MASKS}")
    spec["chunking"] = str(spec["chunking"]).lower()
    if spec["chunking"] not in KV_REUSE_CHUNKINGS:
        raise ValueError(
            f"arm {arm.name!r}: kv_reuse chunking must be one of {KV_REUSE_CHUNKINGS}")
    if spec["chunking"] == "grid":
        if spec["chunk_tokens"] is None or int(spec["chunk_tokens"]) < 1:
            raise ValueError(
                f"arm {arm.name!r}: kv_reuse chunking 'grid' needs chunk_tokens >= 1")
        spec["chunk_tokens"] = int(spec["chunk_tokens"])
    elif spec["chunk_tokens"] is not None:
        raise ValueError(
            f"arm {arm.name!r}: kv_reuse chunk_tokens only applies to chunking 'grid'")
    return spec


@dataclass(frozen=True)
class Arm:
    name: str
    compress_history: bool
    ratio: int = 8
    hybrid_top_k: int = 0  # 0 = none raw; >0 = keep this many tail messages raw
    constrain_tools: bool = False  # H1: xgrammar structural-tag decoding on <tool_call>
    # Repair-arm extension (docs/hybrid_spec.md "Repair interaction"):
    # {"policy": "first" | "offset:<j>" | "chunk:<i>"} — the proxy forwards
    # it as the request-level "c2kv_repair" field and hf_server appends the
    # raw KV of the selected COMPRESSED history block at its original
    # logical offset (offset:<j> indexes docs like the D harness
    # --corr_k_policy; chunk:<i> indexes the server's extract chunks).
    repair: Optional[Dict[str, object]] = None
    # Oracle-recover extension (docs/hybrid_spec.md "Oracle recover"):
    # {"once": True} — the proxy loads the full-arm reference trajectory
    # (--reference) and repairs the first divergence per conversation by
    # regenerating the step with full-raw history KV.  Mutually exclusive
    # with repair (both replace the same step's generation).
    recover: Optional[Dict[str, object]] = None
    # Text-level baseline arms (benchmarks/textarms.py): the proxy rewrites
    # the request history per the named paper's policy instead of any KV
    # compression; compressor calls go to the same served endpoint (same
    # model as policy — the papers' own protocol). Variant names keep
    # HiAgent retrieval and ACON guideline stages explicit.
    text_policy: Optional[str] = None
    # Capabilities supplied by benchmark/proxy plumbing rather than by the
    # model endpoint itself. A matrix preflight must reject an arm when one
    # of these markers is absent.
    required_capabilities: tuple[str, ...] = ()
    # Per-arm query projection override. CacheBlend is a training-free base
    # projection baseline even if the shared server was launched with gist as
    # its general default.
    query_projection: Optional[str] = None
    # History-KV eviction baselines (upstream
    # c2kv_eval.adapters.bfcl_history_kv_baselines): no gist compression at
    # all -- the completed history is compressed by TOKEN EVICTION inside the
    # server.  {"method": streamingllm|h2o|snapkv_persistent|pyramidkv,
    #  "retention_ratio": float | "target_tokens": int,
    #  "backend": "repair_extract" (default) | "physical_eviction",
    #  "recent_window", "kernel_size", "pooling", "h2o_recent_fraction",
    #  "persistent_session"}.  history_kv_spec() fills the defaults and
    # rejects anything the server would refuse.
    history_kv: Optional[Dict[str, object]] = None
    # KV reuse with selective recompute (CacheBlend): no gist compression;
    # the completed history is served as per-chunk out-of-context KV with the
    # highest-deviation ``recomp_ratio`` of its tokens recomputed in context
    # by the server.  {"method": "cacheblend", "recomp_ratio", "check_layer",
    #  "metric": v|k, "mask": causal|bottom_right, "chunking": doc|grid,
    #  "chunk_tokens"}.  kv_reuse_spec() fills the defaults and rejects
    # anything the server would refuse.
    kv_reuse: Optional[Dict[str, object]] = None
    # Benchmark-owned turn-end correctness oracle. The proxy only selects
    # and appends KV after an explicit privileged repair request.
    gold_recovery: Optional[str] = None
    native_messages: bool = False
    description: str = ""

    def validate(self) -> None:
        if self.gold_recovery not in (None, "witness", "random"):
            raise ValueError(f"unknown gold recovery selector {self.gold_recovery!r}")
        if self.gold_recovery and (not self.compress_history or self.repair or self.recover):
            raise ValueError("gold recovery requires plain gist compression")
        if self.history_kv:
            if self.compress_history or self.text_policy or self.repair or self.recover:
                raise ValueError(
                    f"arm {self.name!r}: history_kv is exclusive with gist "
                    "compression, text_policy, repair and recover")
            history_kv_spec(self)
        if self.kv_reuse:
            if (self.compress_history or self.text_policy or self.repair
                    or self.recover or self.history_kv):
                raise ValueError(
                    f"arm {self.name!r}: kv_reuse is exclusive with gist "
                    "compression, text_policy, repair, recover and history_kv")
            kv_reuse_spec(self)
        if self.compress_history and self.ratio < 2:
            raise ValueError(f"arm {self.name!r}: compression ratio must be >= 2")
        if not self.compress_history and (self.hybrid_top_k or self.repair or self.recover):
            raise ValueError(f"arm {self.name!r}: hybrid/repair/recover knobs need compression")
        if self.repair and self.recover:
            raise ValueError(f"arm {self.name!r}: repair and recover are mutually exclusive")
        if self.text_policy and self.compress_history:
            raise ValueError(f"arm {self.name!r}: text_policy and KV compression are exclusive")
        text_policies = {
            "hiagent", "hiagent_summary", "hiagent_full",
            "acon_hist", "acon_obs",
            "acon_hist_base", "acon_hist_ut", "acon_hist_ut_co",
            "acon_obs_base", "acon_obs_ut", "acon_obs_ut_co",
        }
        if self.text_policy and self.text_policy not in text_policies:
            raise ValueError(f"arm {self.name!r}: unknown text_policy {self.text_policy!r}")
        if self.query_projection not in (None, "base", "gist"):
            raise ValueError(
                f"arm {self.name!r}: unknown query_projection "
                f"{self.query_projection!r}")
        if self.repair:
            placement = str(self.repair.get("placement") or "append_keep_ledger")
            if placement not in REPAIR_PLACEMENTS:
                raise ValueError(
                    f"arm {self.name!r}: unknown repair placement {placement!r}; "
                    f"expected one of {REPAIR_PLACEMENTS}")


ARMS: Dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm(
            name="full",
            compress_history=False,
            description="raw text history, no compression (upper reference)",
        ),
        Arm(
            name="hiagent",
            compress_history=False,
            text_policy="hiagent",
            description="Compatibility alias for hiagent_summary; observation summarization only",
        ),
        Arm(
            name="full_native", compress_history=False, native_messages=True,
            description="Uncompressed original model with native chat messages and tool-call history",
        ),
        Arm(
            name="c2kv4", compress_history=True, ratio=4,
            description="C2KV history gist at nominal 4x before recovery",
        ),
        Arm(
            name="c2kv4_gold_witness", compress_history=True, ratio=4,
            gold_recovery="witness",
            description="4x gist + BFCL turn-end gold detector + gold witness k* + one-turn append_keep_ledger retry",
        ),
        Arm(
            name="c2kv8_gold_witness", compress_history=True, ratio=8,
            gold_recovery="witness",
            description="8x gist + BFCL turn-end gold detector + gold witness k* + one-turn append_keep_ledger retry",
        ),
        Arm(
            name="c2kv4_gold_random", compress_history=True, ratio=4,
            gold_recovery="random",
            description="4x gist + identical gold trigger and retry, random history block control",
        ),
        Arm(
            name="c2kv8_gold_random", compress_history=True, ratio=8,
            gold_recovery="random",
            description="8x gist + identical gold trigger and retry, random history block control",
        ),
        Arm(
            name="hiagent_summary",
            compress_history=False,
            text_policy="hiagent_summary",
            description="HiAgent (2408.09559) summary-only ablation: subgoal protocol and completed-subgoal observation summaries, without Trajectory Retrieval",
        ),
        Arm(
            name="hiagent_full",
            compress_history=False,
            text_policy="hiagent_full",
            required_capabilities=("hiagent_trajectory_retrieval_v1",),
            description="HiAgent full integrated variant: subgoal summaries plus proxy-intercepted Trajectory Retrieval; tool-native adaptation with no claim of reproducing the paper's benchmark numbers; requires hiagent_trajectory_retrieval_v1",
        ),
        Arm(
            name="acon_hist",
            compress_history=False,
            text_policy="acon_hist",
            description="Compatibility alias for acon_hist_base",
        ),
        Arm(
            name="acon_obs",
            compress_history=False,
            text_policy="acon_obs",
            description="Compatibility alias for acon_obs_base",
        ),
        Arm(
            name="acon_hist_base",
            compress_history=False,
            text_policy="acon_hist_base",
            description="ACON (2510.00615) history compression with the official initial base guideline; threshold 4096 tokens and last action-observation pair preserved",
        ),
        Arm(
            name="acon_hist_ut",
            compress_history=False,
            text_policy="acon_hist_ut",
            description="ACON history compression with the paper Appendix D guideline after one utility-maximization (UT) step",
        ),
        Arm(
            name="acon_hist_ut_co",
            compress_history=False,
            text_policy="acon_hist_ut_co",
            description="ACON history compression with the paper Appendix D guideline after UT then compression-maximization (CO)",
        ),
        Arm(
            name="acon_obs_base",
            compress_history=False,
            text_policy="acon_obs_base",
            description="ACON observation compression with the official initial base guideline and 1024-token threshold",
        ),
        Arm(
            name="acon_obs_ut",
            compress_history=False,
            text_policy="acon_obs_ut",
            description="ACON observation compression with the paper Appendix D guideline after one utility-maximization (UT) step",
        ),
        Arm(
            name="acon_obs_ut_co",
            compress_history=False,
            text_policy="acon_obs_ut_co",
            description="ACON observation compression with the paper Appendix D guideline after UT then compression-maximization (CO)",
        ),
        Arm(
            name="c2kv",
            compress_history=True,
            ratio=8,
            description="all history as gist KV at 8x, current turn raw",
        ),
        Arm(
            name="c2kv16",
            compress_history=True,
            ratio=16,
            description="all history as gist KV at 16x (compression ladder)",
        ),
        Arm(
            name="hybrid",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            description="hybrid k=3: top-3 tail messages raw, earlier history gist (docs/hybrid_spec.md)",
        ),
        Arm(
            name="hybrid_k1",
            compress_history=True,
            ratio=8,
            hybrid_top_k=1,
            description="hybrid k=1: top-1 tail message raw, earlier history gist",
        ),
        Arm(
            name="hybrid_k5",
            compress_history=True,
            ratio=8,
            hybrid_top_k=5,
            description="hybrid k=5: top-5 tail messages raw, earlier history gist",
        ),
        Arm(
            name="c2kv_repair",
            compress_history=True,
            ratio=8,
            repair={"policy": "first", "placement": "append_keep_ledger"},
            description="c2kv 8x + corr@first (D-harness corr/keepG): raw KV of doc 0 computed in context, appended, gist kept, original RoPE phase",
        ),
        Arm(
            name="c2kv_repair_tail",
            compress_history=True,
            ratio=8,
            repair={"policy": "first", "placement": "append_tail"},
            description="c2kv 8x + erratum_tail@first (D-harness raw_erratum_tail): raw KV of doc 0 re-rotated to the end of history, ledger advanced",
        ),
        Arm(
            name="c2kv_repair_inplace",
            compress_history=True,
            ratio=8,
            repair={"policy": "first", "placement": "in_place"},
            description=(
                "c2kv 8x + replaceG@first: raw KV of doc 0 replaces its gist in place. "
                "REGIME BREAK vs the Sep-2 serve-align runs: policy 'first' on a "
                "single-doc turn leaves the request with zero gist segments, and the "
                "reconciled server (D7, c2kv_serving_semantics.md #1) serves such a "
                "request with GIST query projections where serve-align served base. "
                "Read c2kv_query_proj_effective + c2kv_gist_seen per row; the "
                "layout frame check is also unavailable on this arm "
                "(repair_frame.ok_reason = not_measured_in_place_first_doc). "
                "frame_delta IS measured on BFCL FC: the prologue is measured "
                "against the assembled list, which always carries proxy."
                "DEFAULT_SYSTEM_PROMPT when the client sends no system message."
            ),
        ),
        Arm(
            name="hybrid_repair",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            repair={"policy": "first", "placement": "append_keep_ledger"},
            description="hybrid k=3 + corr@first on the compressed prefix",
        ),
        Arm(
            name="hybrid_repair_tail",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            repair={"policy": "first", "placement": "append_tail"},
            description="hybrid k=3 + erratum_tail@first on the compressed prefix",
        ),
        Arm(
            name="c2kv_recover",
            compress_history=True,
            ratio=8,
            recover={"once": True},
            description="c2kv 8x + step-level oracle recover: first action divergence vs the full-arm reference is regenerated with full-raw history KV (once per conversation)",
        ),
        Arm(
            name="hybrid_recover",
            compress_history=True,
            ratio=8,
            hybrid_top_k=3,
            recover={"once": True},
            description="hybrid k=3 + step-level oracle recover (once per conversation)",
        ),
        # ---- history-KV eviction baselines --------------------------
        # retention 0.312 = HISTORY_KV_RETENTION_RATIO in the upstream
        # run_history_kv_baselines.sh; the name suffix is that ratio in
        # per-mille (r312 = 0.312).  NOT RUN against a live server yet.
        Arm(
            name="history_kv_streamingllm_r312",
            compress_history=False,
            history_kv={"method": "streamingllm", "retention_ratio": 0.312},
            description="StreamingLLM history-KV eviction (recent-suffix keep) at "
                        "retention 0.312; server-side selection via "
                        "/v1/c2kv/repair_extract history_kv_method=streamingllm",
        ),
        Arm(
            name="history_kv_h2o_r312",
            compress_history=False,
            history_kv={"method": "h2o", "retention_ratio": 0.312},
            description="H2O heavy-hitter + recent history-KV eviction at retention "
                        "0.312 (h2o_recent_fraction 0.5, recent_window 64); the "
                        "attention scores come from the server's extraction prefill",
        ),
        Arm(
            name="history_kv_snapkv_r312",
            compress_history=False,
            history_kv={"method": "snapkv_persistent", "retention_ratio": 0.312},
            description="SnapKV-persistent history-KV eviction at retention 0.312 "
                        "(observation window 64, avgpool kernel 5); the compressed "
                        "entry is reused, not reselected per turn",
        ),
        Arm(
            name="history_kv_pyramidkv_r312",
            compress_history=False,
            history_kv={"method": "pyramidkv", "retention_ratio": 0.312},
            description="PyramidKV history-KV eviction at retention 0.312; the "
                        "per-layer funnel budget is realised as a layer-union keep "
                        "set on one shared page table (server-side globalisation)",
        ),
        # KNOWN CONFOUND on both cd_* arms: constrain_tools=True makes the
        # sglang backend rewrite request.tools with _inline_refs ($refs
        # inlined, loose types mapped, unsupported keywords stripped) so
        # xgrammar can compile them.  SGLang derives the grammar FROM
        # request.tools and the chat template renders the SAME field into the
        # prompt — there is no separate grammar channel — so a cd_* row
        # differs from its unconstrained twin by prompt AND grammar.  H1
        # therefore measures "structural-tag decoding + normalised tool
        # schemas", not decoding alone.  Not fixable proxy-side.
        Arm(
            name="cd_full",
            compress_history=False,
            constrain_tools=True,
            description=(
                "H1: full raw history + xgrammar structural-tag tool-call decoding. "
                "CONFOUND vs 'full': the tool schemas are also rewritten "
                "(_inline_refs) and SGLang renders that rewritten `tools` field "
                "into the prompt, so the prologue differs too."
            ),
        ),
        Arm(
            name="cd_c2kv",
            compress_history=True,
            ratio=8,
            constrain_tools=True,
            description=(
                "H1: 8x gist history + xgrammar structural-tag tool-call decoding. "
                "CONFOUND vs 'c2kv': same prologue rewrite as cd_full — the "
                "grammar and the rendered tool definitions come from one field."
            ),
        ),
        # ---- KV reuse with selective recompute (CacheBlend) ---------------
        # r16 = recomp_ratio 0.16 (per cent), the EuroSys artifact default;
        # the served KV is the WHOLE history span (resident bytes = 1x raw,
        # the Pareto anchor of the baseline table), the saving is compute.
        # NOT RUN against a live server yet.
        Arm(
            name="cacheblend_r16",
            compress_history=False,
            kv_reuse={"method": "cacheblend", "recomp_ratio": 0.16},
            query_projection="base",
            description="CacheBlend (EuroSys artifact lineage): one chunk per "
                        "history doc, standalone chunk KV, V-deviation at layer "
                        "index 1, 16% highest-deviation tokens recomputed in "
                        "context; server-side via /v1/c2kv/repair_extract "
                        "kv_reuse_method=cacheblend",
        ),
        Arm(
            name="cacheblend_r15_k",
            compress_history=False,
            kv_reuse={"method": "cacheblend", "recomp_ratio": 0.15, "metric": "k"},
            query_projection="base",
            description="CacheBlend (LMCache-monorepo lineage): same mechanism "
                        "with K-deviation and recomp_ratio 0.15 (blender.py)",
        ),
    )
}

# Explicit nominal history budgets; keep historical r312 arms unchanged.
for _method in ("h2o", "snapkv_persistent", "streamingllm", "pyramidkv"):
    for _suffix, _retention in (("r250", 0.25), ("r125", 0.125)):
        _name = f"history_kv_{_method}_{_suffix}"
        ARMS[_name] = Arm(
            name=_name, compress_history=False,
            history_kv={"method": _method, "retention_ratio": _retention},
            description=f"{_method} history KV at retention {_retention}; selection provenance must be read from server metadata",
        )


def get_arm(name: str) -> Arm:
    try:
        arm = ARMS[name]
    except KeyError:
        known = ", ".join(sorted(ARMS))
        raise SystemExit(f"FATAL: unknown arm {name!r}; known arms: {known}")
    arm.validate()
    return arm
