#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/multi_turn_base_full200}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"

RATIO="${RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
DEVICES="${DEVICES:-5,6,7}"
PORTS="${PORTS:-35050,35060,35070}"

FULL_REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS:-${OUTPUT_ROOT}/full_reference/full/logs/details.jsonl}"
if [ ! -s "${FULL_REFERENCE_DETAILS}" ]; then
  echo "Full reference details missing: ${FULL_REFERENCE_DETAILS}"
  echo "Run c2kv_eval/scripts/run_bfcl_full200_stage1.sh first."
  exit 1
fi

mkdir -p \
  "${OUTPUT_ROOT}/compression_baselines" \
  "${OUTPUT_ROOT}/recovery_reference_oracle" \
  "${OUTPUT_ROOT}/recovery_task_oracle" \
  "${OUTPUT_ROOT}/summaries"

"${BFCL_PYTHON}" - "${OUTPUT_ROOT}/experiment_config.json" <<PY
import json, pathlib
path = pathlib.Path("${OUTPUT_ROOT}/experiment_config.json")
config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
config.update({
    "model_id": "${MODEL_ID}",
    "model_path": "${MODEL_PATH}",
    "tokenizer_path": "${TOKENIZER_PATH}",
    "ids_path": "",
    "reference_details_path": "${FULL_REFERENCE_DETAILS}",
    "ratio": int("${RATIO}"),
    "checkpoint_interval": int("${CHECKPOINT_INTERVAL}"),
    "temperature": float("${TEMPERATURE}"),
    "stage2_recovery_trigger": "oracle_reference_drift",
})
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

METHODS=full,c2kv,streamingllm,h2o,snapkv_persistent,pyramidkv,kivi \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS_PATH="${FULL_REFERENCE_DETAILS}" \
RATIO="${RATIO}" \
TEMPERATURE="${TEMPERATURE}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
RUN_ROOT="${OUTPUT_ROOT}/compression_baselines" \
CLEAN_OUTPUT=1 \
RUN_COMPARE=1 \
MODEL_ID="${MODEL_ID}" \
MODEL_PATH="${MODEL_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_history_kv_baselines.sh"

VERIFIERS=oracle \
ROLLBACK_DEPTHS=1,2,4 \
ROLLBACK_POLICY=fixed_depth \
ROLLBACK_BACKEND=kv_restore_strict \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS="${FULL_REFERENCE_DETAILS}" \
COMPRESSION_RATIO="${RATIO}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
RUN_ROOT="${OUTPUT_ROOT}/recovery_reference_oracle/rollback" \
CLEAN_OUTPUT=1 \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_fixed_depth_sweep.sh"

ARMS=d_corr_replace_w1,d_corr_replace_w2,d_corr_replace_w4,d_corr_replace_all,d_corr_recompute_w2,d_corr_w2,cacheblend_w2 \
REPAIR_TRIGGER=oracle \
CATEGORY="${CATEGORY}" \
MAX_EXAMPLES="${MAX_EXAMPLES}" \
IDS_PATH="__NONE__" \
REFERENCE_DETAILS_PATH="${FULL_REFERENCE_DETAILS}" \
RATIO="${RATIO}" \
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
DEVICES="${DEVICES}" \
PORTS="${PORTS}" \
RUN_ROOT="${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair" \
CLEAN_OUTPUT=1 \
RUN_COMPARE=1 \
MODEL_ID="${MODEL_ID}" \
MODEL_PATH="${MODEL_PATH}" \
TOKENIZER_PATH="${TOKENIZER_PATH}" \
bash "${ROOT}/c2kv_eval/scripts/run_bfcl_kv_repair_sweep.sh"

"${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_full200_unified \
  --output-root "${OUTPUT_ROOT}" \
  --category "${CATEGORY}" \
  --model "${MODEL_ID}" \
  --compression-run-root "${OUTPUT_ROOT}/compression_baselines" \
  "Full:full:${OUTPUT_ROOT}/compression_baselines/full" \
  "C2KV:c2kv:${OUTPUT_ROOT}/compression_baselines/c2kv" \
  "StreamingLLM:streamingllm:${OUTPUT_ROOT}/compression_baselines/streamingllm" \
  "H2O:h2o:${OUTPUT_ROOT}/compression_baselines/h2o" \
  "SnapKV:snapkv_persistent:${OUTPUT_ROOT}/compression_baselines/snapkv_persistent" \
  "PyramidKV:pyramidkv:${OUTPUT_ROOT}/compression_baselines/pyramidkv" \
  "KIVI-QDQ:kivi:${OUTPUT_ROOT}/compression_baselines/kivi" \
  "Rollback D1:rollback_d1:${OUTPUT_ROOT}/recovery_reference_oracle/rollback/fixed_depth_i${CHECKPOINT_INTERVAL}_oracle_d1_kv_restore_strict" \
  "Rollback D2:rollback_d2:${OUTPUT_ROOT}/recovery_reference_oracle/rollback/fixed_depth_i${CHECKPOINT_INTERVAL}_oracle_d2_kv_restore_strict" \
  "Rollback D4:rollback_d4:${OUTPUT_ROOT}/recovery_reference_oracle/rollback/fixed_depth_i${CHECKPOINT_INTERVAL}_oracle_d4_kv_restore_strict" \
  "Replace W1:d_corr_replace_w1:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_w1" \
  "Replace W2:d_corr_replace_w2:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_w2" \
  "Replace W4:d_corr_replace_w4:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_w4" \
  "Replace All:d_corr_replace_all:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_replace_all" \
  "Recompute W2:d_corr_recompute_w2:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_recompute_w2" \
  "Append W2:d_corr_w2:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/d_corr_w2" \
  "CacheBlend 15%:cacheblend_w2:${OUTPUT_ROOT}/recovery_reference_oracle/kv_repair/cacheblend_w2"

echo "Stage 2 complete: ${OUTPUT_ROOT}/unified_full200.csv"
