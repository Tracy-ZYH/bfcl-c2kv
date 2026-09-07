#!/usr/bin/env bash
set -Ee -o pipefail

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
FEATURES_CSV="${FEATURES_CSV:-/home/zhuyuhan/zh_restore/runs/detector_v2_regen/detector_cv/feature_benchmark/detector_features.csv}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/zh_runs/detector_v3_fast_dev_$(date +%Y%m%d_%H%M%S)}"

env \
  ROOT="${ROOT}" \
  FEATURES_CSV="${FEATURES_CSV}" \
  RUN_ROOT="${RUN_ROOT}" \
  DETECTOR_VERSION=3 \
  FAST_DEV=1 \
  FOLDS="${FOLDS:-1}" \
  MAX_EXAMPLES="${MAX_EXAMPLES:-12}" \
  REPAIR_ARM=d_corr_replace_w2 \
  REPAIR_WINDOW=2 \
  CLEAN_OUTPUT=0 \
  REQUEST_CANDIDATE_LOGPROBS="${REQUEST_CANDIDATE_LOGPROBS:-0}" \
  bash "${ROOT}/c2kv_eval/scripts/run_bfcl_combined_logistic_v2_replace_w2.sh"
