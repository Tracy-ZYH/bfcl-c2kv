#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhuyuhan/runs/bfcl_full200_v1}"
C2KV_ROOT="${C2KV_ROOT:-/home/zhuyuhan/project/c2kv}"
CLIENT_OVERLAY="${CLIENT_OVERLAY:-/home/zhuyuhan/zh_restore/zh_client_overlay}"
REUSE_FULL_REFERENCE="${REUSE_FULL_REFERENCE:-1}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"
FIXED_FULL_REFERENCE_ROOT="${FIXED_FULL_REFERENCE_ROOT:-/home/zhuyuhan/runs/bfcl_full200_v2/full_reference}"
FULL_REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS:-${FIXED_FULL_REFERENCE_ROOT}/full/logs/details.jsonl}"

export ROOT BFCL_PYTHON SGLANG_ROOT SGLANG_PYTHON OUTPUT_ROOT C2KV_ROOT
export REUSE_FULL_REFERENCE ALLOW_EXISTING_OUTPUT FIXED_FULL_REFERENCE_ROOT FULL_REFERENCE_DETAILS
# This entry point is intentionally Full-200. Do not inherit stale stable52
# controls from a long-lived tmux shell.
export CATEGORY=multi_turn_base MAX_EXAMPLES=200
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"
export MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507}"
export DEVICES="${DEVICES:-5,6}" PORTS="${PORTS:-35605,35606}"
export DEVICE="${DEVICE:-5}" PORT="${PORT:-35605}"
export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS:---disable-cuda-graph}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.45}"
export C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.05}"
export RATIO="${RATIO:-4}" CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}" TEMPERATURE="${TEMPERATURE:-0}"
# Full is supplied by FIXED_FULL_REFERENCE_ROOT and must not be generated here.
export COMPRESSION_METHODS="${COMPRESSION_METHODS:-c2kv,streamingllm,h2o,snapkv_persistent,pyramidkv,kivi}"
export ROLLBACK_DEPTHS="${ROLLBACK_DEPTHS:-1,2,4}"
export RECOVERY_ARMS="${RECOVERY_ARMS:-d_corr_w1,d_corr_w2,d_corr_w4,d_corr_w2_hint,d_corr_replace_w1,d_corr_replace_w2,d_corr_replace_w4}"
export PYTHONPATH="${ROOT}:${SGLANG_ROOT}/python:${CLIENT_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -d "${OUTPUT_ROOT}" ] && [ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ] \
  && [ "${ALLOW_EXISTING_OUTPUT}" != "1" ]; then
  echo "Refusing to overwrite non-empty output directory: ${OUTPUT_ROOT}"
  echo "Choose a new OUTPUT_ROOT (recommended), or explicitly set ALLOW_EXISTING_OUTPUT=1."
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
start_time=$(date +%s)
echo "[$(date '+%F %T')] full200 all-method experiment started"
echo "output=${OUTPUT_ROOT}"

if [ "${REUSE_FULL_REFERENCE}" = "1" ]; then
  echo "Reusing read-only Full reference: ${FIXED_FULL_REFERENCE_ROOT}"
else
  bash "${ROOT}/c2kv_eval/scripts/run_bfcl_full200_stage1.sh"
  FIXED_FULL_REFERENCE_ROOT="${OUTPUT_ROOT}/full_reference"
  FULL_REFERENCE_DETAILS="${FIXED_FULL_REFERENCE_ROOT}/full/logs/details.jsonl"
  export FIXED_FULL_REFERENCE_ROOT FULL_REFERENCE_DETAILS
fi
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_full200_stage2.sh"

end_time=$(date +%s)
printf 'total_runtime_seconds=%s\n' "$((end_time-start_time))" > "${OUTPUT_ROOT}/total_runtime.txt"
echo "[$(date '+%F %T')] full200 all-method experiment completed"
echo "summary=${OUTPUT_ROOT}/unified_full200.csv"
