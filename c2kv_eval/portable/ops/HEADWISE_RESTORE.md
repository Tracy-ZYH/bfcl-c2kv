# Minimal history-KV restore

H2O and SnapKV keep their original layer/head selection and rotated raw keys.
On request-local append/replace retry, recover the selected W2's exact raw
token rows from the existing full-context repair prefill. Each head uses its
own deduplicated retained/target union. The existing shared dense entry needs
the same length across all layers/heads. Fill shorter unions to the maximum
union length with additional, unique, most-recent raw source tokens. These
are actual K/V rows, not padding; no new masks, cache backend or compression
units are introduced. K/V rows are gathered together and retain original
RoPE. The shared position vector is ledger-only, preserving the original
history end, exactly as in the existing compression-only headwise path.

This is W2 restore with a raw-token superset for dense alignment. It is NOT
target-only restore and NOT full-history retry, although restoring W2 may
cover the whole history in a short conversation. Additional raw tokens and
the resulting compression cost must be reported. Append and replace produce
the same deduplicated raw selection for raw-token eviction baselines.

The response includes `recovery_semantics=headwise_raw_union_dense_completion_v1`,
`recovery_target_coverage=1`, `dense_alignment_extra_tokens_per_head_mean/max`,
`target_restored_tokens_per_head_mean`, source-index previews, and before/after
active token counts. The client rejects responses missing this acknowledgement.
Proxy logs and summaries retain the accounting. A returned retry is operator
success, not task-repair success. The baseline representation is still one
selected history span, not independently replaceable history units.

## PyramidKV development audit

In retry_v2, first/W2 always recovery changed six correct tasks to incorrect
(3,19,31,38,47,48), and changed task 28 from incorrect to correct. All seven
terminated with user_stop, so the net -5/50 is not explained by timeouts.
Trajectories include missing checks/actions and accepting false user claims.
No verified position-mapping bug was identified. Baseline and recovery use
the same existing shared-index PyramidKV approximation and full-context raw
KV extraction; its metadata explicitly says it is not the official algorithm.

`RECOVERY_SELECTOR=last` now selects the latest W2 completed-history docs.
This tests whether stale first/W2 recovery caused some failures. Selection of
retained tokens is unchanged. It is an experimental protocol change, not a
proven accuracy fix. Use new compression-only controls and official scores;
do not tune this choice against test rewards or overwrite earlier results.

## Run protocol

The tau2 full runner now includes all 15 compression/append/replace arms,
including h2o_r25_replace_w2, snapkv_r25_replace_w2, h2o_r25_append_w2,
snapkv_r25_append_w2. First validate one task across three baselines and both
operations, then run airline task 0..49 with the same selector/config.
Use DEVICES=3,5 and PORTS=35603,35605; the runner starts a private fresh server
per arm and cleans that server's process group. Existing servers are not stopped.
The full runner supports METHOD_FILTER, TAU2_TASK_IDS and EXPECTED_CASES.
No live NPU test was launched during this implementation. CPU selector and
client contracts pass; NPU attention, ledger and official task scores remain
to be verified by user-run smoke.
