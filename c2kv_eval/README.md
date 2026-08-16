# C2KV × BFCL Evaluation

Custom evaluation utilities for integrating C2KV/SGLang with BFCL.

## Structure

- `scripts/`: Full, C2KV, and Hybrid benchmark runners
- `adapters/`: BFCL-to-C2KV request adapters
- `analysis/`: Evaluation and error-analysis utilities

## Current baseline

Model:
Qwen/Qwen3-4B-Instruct-2507-FC

Backend:
SGLang + Ascend NPU

BFCL simple_python Full baseline:
- Python Simple AST Accuracy: 95.00%

## Multi-Turn Tool-Definition Compression

Run the 3-card `multi_turn_base` comparison:

```bash
cd /home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard
bash c2kv_eval/scripts/run_bfcl_multiturn_3cards.sh
```

Defaults:

- NPU 5 / port 32000: Full
- NPU 6 / port 32001: C2KV@4
- NPU 7 / port 32002: Hybrid@4 Top-3
- Dataset: `multi_turn_base`
- Samples: 200
- Tool document mode: `per_tool`

Outputs:

```text
/home/zhuyuhan/project/gorilla/bfcl_runs/tooldef_hardneg_multi_turn_base_200/
├── full/
├── c2kv/
├── hybrid/
├── summary.json
└── report.md
```

For smoke tests:

```bash
MAX_EXAMPLES=1 RUN_ROOT=/home/zhuyuhan/project/gorilla/bfcl_runs/smoke_bfcl_c2kv \
  bash c2kv_eval/scripts/run_bfcl_multiturn_3cards.sh
```
