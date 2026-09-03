# BFCL History KV Compression Baselines

This runner adds a common BFCL multi-turn closed-loop entry point for history
KV compression baselines:

- `full`
- `c2kv`
- `streamingllm`
- `h2o`
- `snapkv_persistent`
- `snapkv_refresh`
- `pyramidkv`

The experiment keeps system prompt, tool definitions, and the current turn full.
Only completed history is part of the compression decision.

## Implemented Strict Paths

- `full`: no history compression.
- `c2kv`: uses the existing SGLang C2KV extraction and gist injection path.
- `streamingllm`: SGLang runtime extracts full-causal history KV, keeps the
  recent suffix under the token budget, stores it as a C2KVPool repair entry,
  and injects that entry into the later chat request.
- `h2o`: SGLang runtime extracts full-causal history KV, scores history tokens
  with recent-query attention, keeps heavy hitters plus a recent suffix, stores
  the surviving KV slots as a repair entry, and injects that entry.
- `snapkv_persistent`: SGLang runtime extracts full-causal history KV, pools
  recent-query attention scores following the SnapKV selection idea, keeps
  pooled top-k history tokens plus recent tokens, stores the surviving KV slots
  as a repair entry, and injects that entry.
- `pyramidkv`: SGLang runtime computes PyramidKV-style per-layer budgets and
  per-layer attention selections, then injects the union of all selected token
  positions. This is a shared-page-table approximation required by the current
  SGLang `req_to_token` layout; the reported active KV tokens are the actual
  union size, not the configured average budget.

These runtime baselines are not client-side text truncation. The generated
request only sees the compressed runtime KV entry for completed history.
System/tools/current turn remain full.

## Current Runtime Boundary

The current implementation builds compressed history KV entries through a
separate full-causal extraction request (`/v1/c2kv/repair_extract`) and then
injects the selected KV into the chat request. It therefore measures true
attention-visible active KV for generation, but its maintenance prefill work is
separate from a future live in-place eviction implementation.

Standard PyramidKV uses layer-wise KV capacities. The current implementation
cannot expose different active history lengths to different layers because
SGLang's request page table is shared across layers. The runtime therefore uses
a conservative union-of-layer-selections approximation and records
`shared_page_table_approximation=true` in the extract metadata.

`snapkv_refresh` can still be run as an explicit diagnostic client fallback, but
it reselects from full textual history on every call and is not the persistent
SnapKV baseline.

## Key Outputs

- `history_kv_baseline_summary.csv`
- `history_kv_baseline_summary.md`
- per-method `logs/details.jsonl`
- per-method BFCL official score directory

Measured runtime compression and estimated client-side compression are kept in
separate columns. Missing SGLang runtime reports are never silently used as
measured compression.
