# Portable C2KV benchmark client

This package connects the C2KV SGLang serving extensions to the official BFCL,
tau2, ToolSandbox, ACEBench, ACON AppWorld, and ACON QA harnesses. It is kept
under `c2kv_eval.portable` so the existing `c2kv_eval` BFCL experiment code and
entry points remain unchanged.

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

Run the unified client from a checkout or an installed wheel:

```bash
python -m c2kv_eval.portable.run --help
# or, after installation
c2kv-bench --help
```

The arm registry in `arms.py` is the single source of request semantics:

- `c2kv_repair`, `c2kv_repair_tail`, and `c2kv_repair_inplace` expose the
  append-at-original-position, append-at-tail, and replace-in-place KV
  operations.
- `c2kv_recover` and `hybrid_recover` replay one diverged step with full raw
  history against a reference recorded by a full-arm run.
- `c2kv4_gold_witness` and related gold controls are BFCL-only. The official
  BFCL handler owns the turn-end correctness trigger; `bfcl_gold_recovery.py`
  supplies the privileged trigger payload, while the proxy only selects and
  materializes the requested KV block.
- `history_kv_*` exposes StreamingLLM, H2O, SnapKV-persistent, and PyramidKV
  through the SGLang history-KV contract.
- `hiagent*`, `acon_*`, and `cacheblend_*` use the same SGLang endpoint and
  account for their extra model calls or selective recomputation.

`--checkpoint` identifies the served weights and resolves the packing profile.
`--tokenizer` independently identifies the base tokenizer used to decode the
exact extracted rows for witness selection. When both live in the same model
directory, omitting `--tokenizer` falls back to `--checkpoint`. A BFCL gold arm
fails preflight if neither directory is available.

Harness checkouts are selected by explicit flags or environment variables:
`--bfcl-dir`/`BENCH_BFCL_DIR`, `--tau2-dir`/`TAU2_DIR`,
`--toolsandbox-dir`/`TS_DIR`, `--acebench-dir`/`ACEBENCH_DIR`, and
`--acon-dir`/`ACON_DIR`. The BFCL adapter also detects the BFCL package beside
this package when run from this repository.

The `patches/` directory contains the endpoint and role-history changes needed
by the pinned upstream harness versions. Each subdirectory includes application
instructions and a copy of the upstream license. Apply patches only to the
matching pinned revision described there.

The benchmark and proxy may use different Python environments. Run this CLI
with the benchmark's interpreter and pass `--proxy-python /path/to/python`
when its environment cannot load the tokenizer. All inference and compression
model calls still use `--upstream`; no model weights load in the client.
`checkpoint-1088` is a reconstructed compatibility preset. For another
checkpoint, provide its saved `--checkpoint-profile` instead of reusing that
preset without checking the packing and projection settings.

For a finite ACEBench run, combine the endpoint/model/checkpoint options
with `--benchmark acebench --acebench-dir /path/to/ACEBench
--acebench-category agent_multi_step --max-tasks 1 --max-iter 4`. Select
`--arm c2kv4` or `--arm hiagent_full`; the latter also requires
`--capability-features hiagent_trajectory_retrieval_v1`. HiAgent summaries
and trajectory retrieval are separate variants. If a short task never reaches
a compression trigger, the request log records that fact.

H2O and SnapKV are history-boundary prefill adaptations here; SnapKV-persistent
does not implement online decode refresh. PyramidKV uses the shared page-table
approximation. CacheBlend retains the full KV span and reduces recomputation.
These variant boundaries are part of the serving contract.

## Retry before returning a generation

The portable proxy accepts a detector-independent control on a plain C2KV arm:

```bash
python -m c2kv_eval.portable.run --benchmark tau2 --arm c2kv4 \
  --backend sglang --upstream http://127.0.0.1:30000 \
  --model YOUR_SERVED_MODEL --checkpoint /path/to/checkpoint \
  --tokenizer /path/to/base-tokenizer --reference-profile checkpoint-1088 \
  --recovery-control '{"operation":"replace","triggered":true,"selector":"first"}' \
  --task-set airline --max-tasks 1 --out /path/to/new-output
```

An individual chat request may instead carry `c2kv_recovery_control` with the
same JSON object. This field is removed before every upstream model call.
`triggered: false` leaves the first candidate unchanged. A true trigger retains
the initial candidate inside the proxy, performs at most one retry, and returns
only the final candidate to the harness. The CLI default applies to every
request; a request field overrides it. This is an explicit trigger interface,
not a correctness detector.

Operations are `append` (raw KV at its original positions), `replace` (replace
the selected gist slot with raw KV), and `retry_full` (generate from full raw
history). Append and replace support `selector: first`, `selector: index` with
`target_index`, or `selector: witness` with explicit `target_values`. For example:

```json
{"operation":"append","triggered":true,"selector":"witness","target_values":["AAPL"]}
```

Witness uses the shipped frozen IDF selector on exact tokenizer-decoded source
rows. Values shorter than eight characters exclude adjacent word characters
or periods; longer values use case-sensitive substring matching.
A missing literal match skips the retry; it does not select a recent row
as a fallback. Full retry takes no witness values or target index. Empty
compressed history also skips a KV repair. Request logs record both generations'
usage and the chosen operation, without logging witness target values.

This interface operates before the harness receives a generation. BFCL tool
environment rollback stays in the existing BFCL entry points; this package does
not roll back tau2, ACEBench, AppWorld, or ToolSandbox environments.
