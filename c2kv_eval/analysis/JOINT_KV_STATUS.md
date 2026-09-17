# Joint Tool/History KV status

The native BFCL serving path supports `tool_only`, `history_only`, and `joint`
scopes for H2O, SnapKV-persistent, and PyramidKV at 25% retention. It keeps
one persistent session per BFCL task and performs one physical cache
compaction per turn.

## Data flow

1. The SGLang OpenAI serving layer renders the native request once.
2. The Tool span is the minimal changed token interval between the same native
   chat-template rendering with and without `request.tools`.
3. Completed-History boundaries continue to use the existing authoritative
   message-prefix renderer.
4. Each enabled semantic scope gets its own budget and baseline selection.
5. Selected Tool/History positions are unioned with every protected position
   between them, then `PhysicalHistoryKVEvictor` materialises the union once.
6. On later turns canonical scope positions are mapped through the persistent
   resident-position ledger; evicted positions cannot reappear.

Registered arms:

- `joint_{h2o,snapkv_persistent,pyramidkv}_tool_only_r25`
- `joint_{h2o,snapkv_persistent,pyramidkv}_history_only_r25`
- `joint_{h2o,snapkv_persistent,pyramidkv}_joint_r25`
- `joint_cacheblend_history_only_r16`

The response/request log includes `full_tool_kv`, `active_tool_kv`,
`full_history_kv`, `active_history_kv`, `joint_active_kv`, the per-scope
selection records, and cumulative run-level retention/compression ratios.

## Known backend limitations

- The physical SGLang request table is shared across layers/KV heads. The
  implementation preserves the existing BFCL physical-baseline semantics:
  H2O/SnapKV aggregate attention scores and PyramidKV reports
  `physical_eviction_globalized`. It does not claim equivalence to a truly
  per-head request table.
- CacheBlend is not token eviction. Its existing safe serving API reuses one
  contiguous completed-History span. Native Qwen Tool definitions and History
  are non-contiguous around protected scaffold, and the API has no per-chunk
  fresh/reuse mask. `history_only` remains supported; `tool_only` and `joint`
  deliberately fail closed instead of recomputing/compressing System tokens or
  duplicating the native Tool prologue.
- C2KV joint arms are absent because checkpoint-1088 was not trained for
  Tool+History joint gist compression.

No model server or inference endpoint was used while implementing this path.
