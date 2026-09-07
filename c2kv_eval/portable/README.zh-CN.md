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

`--benchmark` 可选 `bfcl`、`acebench`、`tau2`、`toolsandbox`、`acon_appworld`、`acon_qa`。更换 benchmark 时，同时设置它的目录和 task 选择参数：

- ACEBench、tau2、QA：`--max-tasks`。
- BFCL：`--run-ids`；AppWorld：`--task-ids`；ToolSandbox：`--ts-scenarios`。
- QA 还需要启动 BM25 retriever；外部 benchmark 所需补丁见 `patches/`。

恢复操作通过 `--recovery-control` 指定，详见[英文 README 的重试说明](README.md#retry-before-returning-a-generation)。H2O/SnapKV 当前是 history-boundary prefill 版本。

已有有限 NPU smoke 和六类 benchmark 的官方单例评分验证（preliminary, n=1）；完整效果评测需另行运行。
