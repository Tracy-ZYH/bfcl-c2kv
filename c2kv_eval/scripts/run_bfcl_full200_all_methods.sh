#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/liuyancheng/envs/bench/bin/python}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhuyuhan/runs/bfcl_full200_v1}"

export ROOT BFCL_PYTHON SGLANG_ROOT SGLANG_PYTHON OUTPUT_ROOT
export CATEGORY=multi_turn_base MAX_EXAMPLES=200
export MODEL_ID=Qwen/Qwen3-4B-Instruct-2507-FC
export MODEL_PATH=/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088
export TOKENIZER_PATH=/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507
export DEVICES="5,6" PORTS="35605,35606"
export DEVICE=5 PORT=35605
export SGLANG_EXTRA_ARGS="--disable-cuda-graph"
export RATIO=4 CHECKPOINT_INTERVAL=4 TEMPERATURE=0
export COMPRESSION_METHODS="full,c2kv,streamingllm,h2o,snapkv_persistent,pyramidkv,kivi"
export ROLLBACK_DEPTHS=2
export RECOVERY_ARMS="d_corr_replace_w2,d_corr_w2,d_corr_w2_hint"
export PYTHONPATH="/home/zhuyuhan/project/python_overlay:${ROOT}:${SGLANG_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
start_time=$(date +%s)
echo "[$(date '+%F %T')] full200 all-method experiment started"
echo "output=${OUTPUT_ROOT}"

bash "${ROOT}/c2kv_eval/scripts/run_bfcl_full200_stage1.sh"
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_full200_stage2.sh"

end_time=$(date +%s)
printf 'total_runtime_seconds=%s\n' "$((end_time-start_time))" > "${OUTPUT_ROOT}/total_runtime.txt"
echo "[$(date '+%F %T')] full200 all-method experiment completed"
echo "summary=${OUTPUT_ROOT}/unified_full200.csv"
