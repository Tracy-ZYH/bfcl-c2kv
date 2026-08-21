#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"
CATEGORY="${CATEGORY:-multi_turn_base}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:?REFERENCE_DETAILS is required}"
MODES="${MODES:-multistep_i2_oracle_whole_segment,multistep_i2_oracle_first_bad,multistep_i2_oracle_first_bad_suffix,multistep_i4_oracle_whole_segment,multistep_i4_oracle_first_bad,multistep_i4_oracle_first_bad_suffix}"

cd "${ROOT}"
exec "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_checkpoint.py \
  --run-root "${RUN_ROOT}" \
  --category "${CATEGORY}" \
  --model "${MODEL_ID}" \
  --reference-details-path "${REFERENCE_DETAILS}" \
  --modes "${MODES}"
