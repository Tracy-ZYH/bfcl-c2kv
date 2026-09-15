# BFCL persistent History KV — lifecycle 修复

## 审计结论

修复前，BFCL 默认 `repair_extract`。StreamingLLM、H2O、SnapKV（包括
`snapkv_persistent` alias）、PyramidKV 每个 generation request 都将完整
completed-history 做 full-causal prefill，然后 selection，存入 C2KVPool 并
reinjection。旧 eviction token 下次可以再次参与候选集合。这是
`stateless_reselect`，不是 episode 内 persistent eviction。

旧 physical/session 代码存在但被 runner 和 shell 两处禁用；BFCL chat
请求没有传 session_params；multi-round prefill 绕过 session match；canonical
history 边界不能直接用于 compacted physical cache；分页 slots 不能等同于
logical tokens。缓存释放还混用了逻辑页序号和实际 allocator 页 ID。

## 修复后的默认 BFCL 路径

一个 episode open 一个 streaming session，每个 generation request 复用同一个
session ID，episode finally close。一个 BFCL turn 可以包含多个 tool/generation
request，不能每个 request 或每个 turn 重新 open。

客户端仍发送完整 OpenAI messages，服务端仍 tokenize 完整 canonical prompt
用于验证工具序列化/prefix/边界。这 **不等于** full KV prefill：进入 model
prefill 的只有 canonical append delta。SessionSlot 恢复上一请求的实际 KV
table，ledger 将 canonical 边界映射到 resident cache，再从 resident+new KV
做 selection 和 physical compaction。未显式恢复的已 eviction position 永远
不能回到候选集合。

System/tool prologue 和当前未完成 turn 受保护。turn 内请求的 completed-history
boundary 可能落在已缓存 prefix 内部，不能要求它大于上一 request 的完整
prompt 长度。原始 decode suffix 在请求结束时丢弃；下一 request 中 structured
assistant/tool-call 的 canonical 序列化作为新增 delta prefill。这是新生成输出
的规范序列化，不是重新恢复之前被 eviction 的旧 completed-history。

K/V compaction 不重新计算/旋转 retained keys；position correction 保存到
SessionSlot，新增 Q/K 的位置继续使用原始逻辑坐标。页 allocator 只释放不再
包含 resident prompt 的物理页，prompt 所在尾页保持所有权。

## 各方法与不能夸大的限制

| 方法 | 新的 lifecycle | 算法特有的限制 |
|---|---|---|
| StreamingLLM | previous resident recent history + new KV → recent selection → physical cache | 保留现有 recent-only history policy；system/tools 是 protected prologue，不是新增可配置的 history sink 算法 |
| H2O | resident keys + newly available queries → 累计 resident attention score → heavy+recent eviction | 跨请求持久化 score；目前为 recent-prefill-query attention 的跨 head/layer 聚合，不是原版逐 decode、独立 per-head heavy-hitter |
| SnapKV | previous resident keys + new KV → attention pooling+recent → physical cache | 物理 mapping 为共享 token indices，不是独立 per-head/layer Top-K；不能声称精确原版 SnapKV |
| PyramidKV | previous resident keys + new KV → layer-weighted scores → shared eviction | per-layer budget 用于 score 权重；最终共享 keep set，不是各层独立 capacity，严格 layer-wise budget 目标尚未实现 |

这次完成的是 persistent **shared-index** backend 的生命周期修复，不能将
它包装为原版 H2O/SnapKV/PyramidKV artifact。要实现严格独立 head/layer cache
需要 attention/page-table layout 改造；没有在本次最小修复中强行改写。
CPU tests 不能证明真实 Ascend kernel、分页 allocator 或官方 scoring 已跑通。

## 保留旧 ablation

`--runtime-history-kv-backend stateless_reselect` 是 `repair_extract` 的明确 alias；
`repair_extract` 名称、接口、selection/reinjection 路径均保留。设
`RUNTIME_HISTORY_KV_BACKEND=repair_extract PERSISTENT_HISTORY_KV_SESSION=0`
可运行旧模式。`physical_eviction + --no-persistent-history-kv-session` 也可用于
旧 per-request physical diagnostic，但不是正式 persistent baseline。

Full 和 retention=1、无更强 compression override 的 identity 请求不 open
history session，不走 eviction。KIVI-QDQ 保留 repair_extract，不改变 2-bit QDQ。
原有结果不覆盖；新 runner 拒绝已有 RUN_ROOT。

## 6 号卡运行（由用户执行）

入口会启动 **当前本地修改后的** kvoffload-sglang-c2kv 服务，依次运行四个
baseline，再进行官方 BFCL generation/evaluation 与日志检查，最后停止它自己
启动的进程组。无需先启动另一个 SGLang；不会复用已占用的端口。

使用 sgl Python `/home/liuyancheng/envs/sgl/bin/python`，源码优先级为
`/home/zhuyuhan/project/kvoffload-sglang-c2kv/python`。服务端关键参数：
`--enable-streaming-session --disable-radix-cache --page-size 128
--disable-cuda-graph --device npu --attention-backend ascend`，物理 eviction
通过 BFCL 请求 hint 指定。脚本 source CANN 8.5.0 和 ATB，只影响子进程。

Checkpoint:
`/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088`

Tokenizer:
`/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507`

数据:
`/home/zhuyuhan/benchmarks/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data`

2 条 smoke（从冻结 stable52 取前两条，顺序遵循 BFCL sort_key）：

```bash
env EXAMPLES=2 bash /home/zhuyuhan/project/bfcl-c2kv/c2kv_eval/scripts/run_bfcl_persistent_history_card6.sh
```

52 条（复用 detector_features.csv 中冻结的 52 个 IDs，不重新筛选 Full 成功）：

```bash
env EXAMPLES=52 bash /home/zhuyuhan/project/bfcl-c2kv/c2kv_eval/scripts/run_bfcl_persistent_history_card6.sh
```

完整 200 条（不使用 stable52/correct IDs 或 Full-success filter）：

```bash
env EXAMPLES=200 bash /home/zhuyuhan/project/bfcl-c2kv/c2kv_eval/scripts/run_bfcl_persistent_history_card6.sh
```

默认 port 35786，history retention 0.312，约 3.2×名义压缩；
`env EXAMPLES=2 HISTORY_KV_RETENTION_RATIO=0.25 ...` 可改为 4×组。
新目录自动为 `/home/zhuyuhan/runs/bfcl_persistent_history_<N>_card6_<timestamp>`；
`env RUN_ROOT=<不存在的新目录> EXAMPLES=2 ...` 可显式指定目录。

## 如何检查真正复用

- `<method>/logs/details.jsonl` 每个 episode 的 `history_kv_lifecycle`：
  一个 session_open，一个 session_close，全部请求同一 session_id。
- `event=session_prompt_saved` 是实际完成 prefill 后保存的 prompt cache，不是
  计划值；includes protected system/current，明确记录 count_scope。
- 下一请求 `previous_resident_position_summary.sha256` 必须等于上一请求
  `resident_position_summary.sha256`；count 也必须相等。客户端和 validator
  会检查这一点。
- `resident_tokens_before_append + new_turn_tokens = resident_tokens_after_append`。
- `resident_tokens_after_append - evicted_tokens_this_turn = resident_tokens_after_eviction`。
- `history_prefill_tokens` 是新 canonical delta 中属于 completed-history 的部分；
  `canonical_delta_prefill_tokens` 包含新增 current/query/tool serialization。
  `full_history_tokens` 是完整 history 逻辑长度，不能用它当实际 prefill work。
- `full_history_reprefill_performed=false`；server missing resident slot/prefix
  mismatch 时 fail closed，不自动 fallback repair_extract/full history。
- 服务端 `HISTORY_KV_SESSION_CLOSED` 记录实际 session resources 释放。
- `<run>/logs/lifecycle_check.log` 四个方法都应 PASS；该检查在官方评估后自动
  运行，包含相同 case IDs/order、cache 连续性和服务端 close 记录。

遇到 assertion/缺少 report，不要绕过检查直接跑 52/200。先读 runner.log、
server_6_35786.log 和 lifecycle_check.log。正常 episode 的 close 路径已做 mock
检查；请求崩溃/超时期间的真实异步关闭仍需 NPU smoke 验证，不能仅凭 HTTP
close 返回就声称物理显存释放干净。

## CPU 验证

已增加 model-free BFCL 请求/session tests 和 server ledger/paired-KV/page
ownership/cached-key-scoring tests。PyTorch 测试显式禁用 torch_npu backend
自动加载，使用 CPU tensors；不 source CANN、不加载 checkpoint、不连接 HTTP。

```bash
env TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 /home/liuyancheng/envs/c2kv/bin/python -m pytest -q /home/zhuyuhan/project/kvoffload-sglang-c2kv/test/registered/unit/test_persistent_history_lifecycle.py /home/zhuyuhan/project/kvoffload-sglang-c2kv/test/registered/unit/test_history_kv_selection.py
```
