#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard"
BFCL_PYTHON="/home/zhuyuhan/miniconda3/envs/bfcl/bin/python"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_drift_full_success_59}"
IDS_PATH="${IDS_PATH:-}"
MODE="${MODE:-history_c2kv4_closed_loop}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/analysis/${MODE}_after_first_drift}"

cd "${ROOT}"
exec "${BFCL_PYTHON}" c2kv_eval/analysis/analyze_drift_after_first.py \
  --run-root "${RUN_ROOT}" \
  --mode "${MODE}" \
  --ids-path "${IDS_PATH}" \
  --output-dir "${OUTPUT_DIR}"
