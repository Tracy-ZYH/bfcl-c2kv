#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard"
BFCL_PYTHON="/home/zhuyuhan/miniconda3/envs/bfcl/bin/python"

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_detector_signal_benchmark_${STAMP}}"
MODE="${MODE:-detector_signal_k4_c2kv_segments}"
RUN_COLLECTION="${RUN_COLLECTION:-1}"
DEVICE="${DEVICE:-7}"
PORT="${PORT:-33700}"

IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"
SEGMENTS_PATH="${SEGMENTS_PATH:-${RUN_ROOT}/${MODE}/logs/checkpoint_segments.jsonl}"

mkdir -p "${RUN_ROOT}/${MODE}/logs" "${RUN_ROOT}/detector_benchmark"

if [[ "${RUN_COLLECTION}" == "1" || "${RUN_COLLECTION}" == "true" ]]; then
  (
    cd "${ROOT}"
    RUN_ROOT="${RUN_ROOT}" \
    MODE="${MODE}" \
    DEVICE="${DEVICE}" \
    PORT="${PORT}" \
    IDS_PATH="${IDS_PATH}" \
    REFERENCE_DETAILS="${REFERENCE_DETAILS}" \
    CHECKPOINT_INTERVAL=4 \
    VERIFIER=oracle \
    ROLLBACK_POLICY=fixed_depth \
    ROLLBACK_DEPTH=4 \
    RECOVERY_MODE=first_bad_suffix \
    RECOVERY_HORIZON=suffix \
    ROLLBACK_BACKEND=kv_restore_strict \
    COLLECT_CANDIDATE_DETECTOR_SIGNALS=1 \
    CANDIDATE_LOGPROBS_TOP_K="${CANDIDATE_LOGPROBS_TOP_K:-20}" \
    CANDIDATE_HIDDEN_READOUT="${CANDIDATE_HIDDEN_READOUT:-0}" \
    CANDIDATE_ATTENTION_SUMMARY="${CANDIDATE_ATTENTION_SUMMARY:-0}" \
    RUN_COMPARE=0 \
    CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}" \
    bash c2kv_eval/scripts/run_bfcl_history_multistep_checkpoint.sh
  )
fi

(
  cd "${ROOT}"
  exec "${BFCL_PYTHON}" c2kv_eval/analysis/benchmark_detector_signals.py \
    --segments-path "${SEGMENTS_PATH}" \
    --output-dir "${RUN_ROOT}/detector_benchmark" \
    --rule-detector-threshold "${RULE_DETECTOR_THRESHOLD:-5.0}"
) | tee "${RUN_ROOT}/detector_benchmark/analyze.log"

echo "${RUN_ROOT}/detector_benchmark/detector_comparison.csv"
