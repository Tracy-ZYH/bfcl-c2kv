# Joint Tool/History KV status

This is an incomplete prerequisite implementation, NOT a runnable joint
baseline. No CLI arms, server, model evaluation, or network calls were added.
Existing evaluator and standalone paths are unchanged.

Implemented in `adapters/joint_kv_provenance.py`:

- Exact full-frame semantic partition validation (`tool/history/protected`).
- Original-position merge independently on each supplied layer/KV-head axis.
- Frame, duplicate, protected-token and incompatible-layout checks.
- CacheBlend reusable-chunk planning (no token eviction).
- Cumulative full/active counts and ratios, including end-to-end retention.

The module accepts authoritative boundaries; it intentionally does NOT
invent Qwen tool chat-template boundaries or call a tokenizer.

## Integration blockers found in current source

1. `c2kv/agent/eval_agent_tool_definition_hybrid_router.py`,
   `_build_full_tool_cache_with_spans`, wraps each tool as a standalone user
   message (`Tool definition {index}: ...`). Its offsets are not offsets into
   the BFCL native `request.tools` prologue.
2. `c2kv/python/inference/compress_kv/compressor.py`, `compress_kv_snapkv`,
   returns per-KV-head retained indices. Current BFCL/SGLang physical history
   eviction uses a shared token request table. Combining them into a global
   union would violate the requested standalone/per-head equivalence.
3. CacheBlend serving `generate_cacheblend_kv` accepts one contiguous reuse
   span. General protected scaffold + tool docs + history chunks require
   authoritative multi-span rendering and injection; compressing the entire
   prologue would incorrectly include protected system/scaffold tokens.
4. No H2O/PyramidKV Tool entrypoint was found in the examined C2KV tool reuse
   baseline files. The BFCL history adaptations cannot be silently called
   identical existing Tool baselines.

Required next implementation: native-tool authoritative renderer provenance,
layer/head-aware KV serving (or explicit approval for a shared-mask adaptation),
multi-span CacheBlend reuse, then existing BFCL runner wiring and mocked end-to-
end contracts. Until then do not publish 2/52/200 joint launch commands or
claim tool_only/history_only equivalence from the planner's unit tests.

CPU checks only:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/zhuyuhan/project/bfcl-c2kv \
  /home/liuyancheng/envs/c2kv/bin/python -m pytest -q -p no:cacheprovider \
  /home/zhuyuhan/project/bfcl-c2kv/tests/test_joint_kv_provenance.py
```

No NPU/GPU/SGLang server was started and no inference endpoint was contacted.
