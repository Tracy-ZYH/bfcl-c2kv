# KV 算法与 benchmark 评测

这组改动让不同算法共用一个 SGLang 服务，并通过统一入口运行 benchmark。

- **算法**：C2KV、HiAgent、ACON、CacheBlend，以及修正后的 H2O/SnapKV 等历史 KV baseline。
- **评测**：BFCL、ACEBench、tau2、ToolSandbox、AppWorld、QA。
- **恢复操作**：append、replace、witness 选块和生成前重试；BFCL 工具环境回滚保留在原入口。

## 怎么运行

先准备好带 C2KV 扩展的 SGLang 服务、模型 checkpoint，以及对应 benchmark 的依赖和数据。服务需启用 `--enable-c2kv`；客户端的 `--model` 应与服务的模型名一致。checkpoint profile 的用途与兼容要求见[英文 README](README.md#portable-c2kv-benchmark-client)。

统一入口是：

```bash
python -m c2kv_eval.portable.run --help
```

## Compression-agnostic recovery（4× smoke）

retention 后缀统一表示保留比例：`r312=31.2%≈3.2×`、`r25=25%=4×`、
`r125=12.5%=8×`。旧的 `r250` 仅保留为 `r25` 的兼容别名，新结果使用
`r25`。

正式多轮 history-KV baseline 使用带 `_persistent` 后缀的 arm，例如
`history_kv_h2o_r25_persistent`。它们走 `physical_eviction`，同一 case 复用
一个 streaming session；服务端按实际 tokenizer 得到的 completed-history span
计算 25% budget，每轮只 prefill canonical 新增 delta。原来的 `*_r25`/`*_r312`
继续表示 `repair_extract` 的每轮 full-history reselect，仅用于旧结果复现和
request-local append/replace ablation，不能标成 persistent。

portable proxy 将 session 数限制为 harness worker 数：新 case 开始时关闭 LRU
已完成 case，proxy 退出时关闭全部剩余 session。正式结果必须在 request log 中
同时看到 `history_kv_backend=physical_eviction`、同一 case 不变的 session id、
`history_kv_lifecycle.full_history_reprefill_performed=false`，且下一轮的
`previous_resident_position_summary` 对应上一轮的 `resident_position_summary`。

受控 smoke 使用 always-trigger、`selector=first`、`window=2`，只用于隔离
Selective Raw-KV Recovery operator；它是 request-local generation/KV retry，
不会回滚 tau2 或 ToolSandbox 已执行的工具/环境状态。

- C2KV Replace W2：删除两个目标 doc 的 gist，并在原始 position 恢复其 raw KV。
- C2KV Append W2：保留 gist，补入目标 doc 的 raw KV；raw token 本身不重复。
- StreamingLLM/PyramidKV：服务端在原 selection 上与 W2 raw source-token index
  做去重并集。由于 retention cache 已是 exact raw token 子集，此时 Replace 和
  deduplicated Append 的最终 KV 集合相同，输出会明确标注 operator equivalence。
- H2O/SnapKV-persistent：当前逐 layer/KV-head 选择不同 source token，而注入层
  只有共享 dense slot。无法在不增加 per-head mask/ragged layout 的情况下严格
  去重，因此 recovery arm 会 fail-fast；不会用重复 KV 冒充成功结果。

双卡一条 tau2 + 一条 ToolSandbox smoke：

```bash
bash c2kv_eval/portable/ops/run_compression_agnostic_recovery_smoke.sh
```

默认 tau2 使用卡 5、ToolSandbox 使用卡 6；结果写入新的
`/home/zhuyuhan/runs/compression_agnostic_recovery_4x_smoke_<timestamp>`。
ToolSandbox 默认使用信息完整但仍需多次工具调用的
`send_message_with_contact_content_cellular_off`；`*_multiple_user_turn`
依赖本地用户模拟器质量，不适合作为 harness/recovery 基础设施 smoke。
若 tau2 已通过，可设置 `RUN_TAU2=0 RUN_TOOLSANDBOX=1` 只重跑
ToolSandbox，也可用 `TS_SCENARIO=<name>` 显式覆盖场景。
每个 SGLang server 在独立 process group 中启动；正常结束、失败或中断时，
脚本会先 TERM 并等待整个 group，必要时才 KILL，避免 scheduler 残留占用 NPU。

tau2 airline 全部 50 个 task 的双卡 4x recovery matrix 使用：

```bash
bash c2kv_eval/portable/ops/run_compression_agnostic_recovery_tau2_full.sh
```

默认卡 5/6 按 method 分片并行；每个 method 都使用固定 `0..49` task ID
及相同顺序。H2O/SnapKV 的 recovery 仍因 headwise dedup 语义不成立而不运行。

例如，用 HiAgent 在 ACEBench 上跑一个短样例。将 `/path/to/...` 替换为实际路径；checkpoint profile 用于指定该模型的 packing 和 projection 配置。

```bash
python -m c2kv_eval.portable.run \
  --benchmark acebench --arm hiagent_full \
  --upstream http://127.0.0.1:30000 --model c2kv-agent \
  --checkpoint /path/to/checkpoint \
  --checkpoint-profile /path/to/checkpoint_profile.json \
  --acebench-dir /path/to/ACEBench \
  --acebench-category agent_multi_step \
  --capability-features hiagent_trajectory_retrieval_v1 \
  --max-tasks 1 --max-iter 4 --num-workers 1 \
  --out results/acebench_hiagent
```

正式评测时调整 task 数和步骤上限。汇总结果保存在输出目录的 `summary_<arm>.json`，逐请求记录保存在 `logs/`。

## 怎么换算法和评测

通过 `--arm` 选择算法：

| 算法 | 参数值 |
|---|---|
| C2KV，4× 压缩配置 | `c2kv4` |
| HiAgent，含 trajectory retrieval | `hiagent_full` |
| ACON history compression | `acon_hist` |
| CacheBlend，16% recompute 配置 | `cacheblend_r16` |
| H2O / SnapKV | `history_kv_h2o_r312` / `history_kv_snapkv_r312` |

CacheBlend 的 Qwen3 服务端支持跨请求 standalone chunk KV 复用：每个模型/
worker 独立维护 CPU LRU，默认上限 256 MiB，可在启动 server 的命令中设置
`SGLANG_CACHEBLEND_CHUNK_CACHE_BYTES`（字节数；0 禁用，用于旧路径 ablation）。
缓存保存 pre-RoPE KV，以精确 token 内容匹配，重载权重时清空；容量不足、
chunk 内容变化或 server 重启会产生 miss。命中无需旧 chunk standalone prefill，
但仍需 CPU/device 搬运、早期层全 token blend 和后续层选择性重计算。
因此 r16 不是总 prefill 工作量只剩 16%，也不是 16% KV retention。
逐请求 cost 输出 `cacheblend_chunk_cache_hit_tokens`、
`cacheblend_chunk_prefill_tokens`、`cacheblend_chunk_cache_bytes` 和
`cacheblend_dense_blend_prefix_tokens`；最后一项是早期 blend 层的输入 token
数，不是所有层计算量之和。真实加速仍需运行后测量。

`--benchmark` 可选 `bfcl`、`acebench`、`tau2`、`toolsandbox`、`acon_appworld`、`acon_qa`。更换 benchmark 时，同时设置它的目录和 task 选择参数：

- ACEBench、tau2、QA：`--max-tasks`。
- BFCL：`--run-ids`；AppWorld：`--task-ids`；ToolSandbox：`--ts-scenarios`。
- QA 还需要启动 BM25 retriever；外部 benchmark 所需补丁见 `patches/`。

恢复操作通过 `--recovery-control` 指定，详见[英文 README 的重试说明](README.md#retry-before-returning-a-generation)。H2O/SnapKV 当前是 history-boundary prefill 版本。

已有有限 NPU smoke 和六类 benchmark 的官方单例评分验证（preliminary, n=1）；完整效果评测需另行运行。
