#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-52}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
ROLLBACK_DEPTHS="${ROLLBACK_DEPTHS:-1,2,4}"
ROLLBACK_POLICY="${ROLLBACK_POLICY:-fixed_depth}"
VERIFIERS="${VERIFIERS:-oracle,rule}"
ROLLBACK_BACKEND="${ROLLBACK_BACKEND:-kv_restore_strict}"
RULE_DETECTOR_THRESHOLD="${RULE_DETECTOR_THRESHOLD:-5}"
DEVICES="${DEVICES:-4,7}"
PORTS="${PORTS:-33640,33670}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_fixed_depth_sweep_$(date '+%Y%m%d_%H%M%S')}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"

log_info() {
  echo "[$(date '+%F %T')] $*"
}

split_csv() {
  local value="$1"
  IFS=',' read -r -a _split_items <<< "${value}"
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

run_job() {
  local verifier="$1"
  local depth="$2"
  local device="$3"
  local port="$4"
  local mode_depth="d${depth}"
  if [ "${ROLLBACK_POLICY}" = "rule_depth" ]; then
    mode_depth="rule_depth"
  fi
  local mode="${ROLLBACK_POLICY}_i${CHECKPOINT_INTERVAL}_${verifier}_${mode_depth}_${ROLLBACK_BACKEND}"
  log_info "[start] mode=${mode} device=${device} port=${port}"
  (
    cd "${ROOT}"
    CATEGORY="${CATEGORY}" \
    MAX_EXAMPLES="${MAX_EXAMPLES}" \
    COMPRESSION_RATIO="${COMPRESSION_RATIO}" \
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
    VERIFIER="${verifier}" \
    RECOVERY_MODE="first_bad_suffix" \
    RECOVERY_HORIZON="suffix" \
    ATTRIBUTION="auto" \
    ATTRIBUTION_SAFETY_MARGIN="0" \
    ROLLBACK_POLICY="${ROLLBACK_POLICY}" \
    ROLLBACK_DEPTH="${depth}" \
    ROLLBACK_BACKEND="${ROLLBACK_BACKEND}" \
    RULE_DETECTOR_THRESHOLD="${RULE_DETECTOR_THRESHOLD}" \
    DEVICE="${device}" \
    PORT="${port}" \
    IDS_PATH="${IDS_PATH}" \
    REFERENCE_DETAILS="${REFERENCE_DETAILS}" \
    RUN_ROOT="${RUN_ROOT}" \
    MODE="${mode}" \
    CLEAN_OUTPUT="${CLEAN_OUTPUT}" \
    RUN_COMPARE="0" \
    bash c2kv_eval/scripts/run_bfcl_history_multistep_checkpoint.sh
  )
  log_info "[done] mode=${mode}"
}

main() {
  split_csv "${DEVICES}"
  local devices=("${_split_items[@]}")
  split_csv "${PORTS}"
  local ports=("${_split_items[@]}")
  if [ "${#devices[@]}" -ne "${#ports[@]}" ]; then
    echo "DEVICES and PORTS must have the same length."
    exit 1
  fi

  split_csv "${VERIFIERS}"
  local verifiers=("${_split_items[@]}")
  split_csv "${ROLLBACK_DEPTHS}"
  local depths=("${_split_items[@]}")
  if [ "${ROLLBACK_POLICY}" = "rule_depth" ]; then
    depths=("1")
  fi

  mkdir -p "${RUN_ROOT}"
  log_info "RUN_ROOT=${RUN_ROOT}"
  log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES}"
  log_info "K=${CHECKPOINT_INTERVAL} ROLLBACK_POLICY=${ROLLBACK_POLICY} DEPTHS=${ROLLBACK_DEPTHS}"
  log_info "VERIFIERS=${VERIFIERS} BACKEND=${ROLLBACK_BACKEND}"
  log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
  log_info "IDS_PATH=${IDS_PATH}"
  log_info "REFERENCE_DETAILS=${REFERENCE_DETAILS}"

  local modes=()
  local pids=()
  local slot=0
  local verifier depth device port mode
  for verifier in "${verifiers[@]}"; do
    for depth in "${depths[@]}"; do
      device="${devices[$slot]}"
      port="${ports[$slot]}"
      mode_depth="d${depth}"
      if [ "${ROLLBACK_POLICY}" = "rule_depth" ]; then
        mode_depth="rule_depth"
      fi
      mode="${ROLLBACK_POLICY}_i${CHECKPOINT_INTERVAL}_${verifier}_${mode_depth}_${ROLLBACK_BACKEND}"
      modes+=("${mode}")
      run_job "${verifier}" "${depth}" "${device}" "${port}" &
      pids+=("$!")
      slot=$(( (slot + 1) % ${#devices[@]} ))
      if [ "${#pids[@]}" -ge "${#devices[@]}" ]; then
        for pid in "${pids[@]}"; do
          wait "${pid}"
        done
        pids=()
      fi
    done
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  local mode_csv
  mode_csv="$(join_by_comma "${modes[@]}")"
  log_info "[compare] modes=${mode_csv}"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_checkpoint.py \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --model "Qwen/Qwen3-4B-Instruct-2507-FC" \
      --reference-details-path "${REFERENCE_DETAILS}" \
      --modes "${mode_csv}"
  ) > "${RUN_ROOT}/compare.log" 2>&1
  log_info "Report: ${RUN_ROOT}/report.md"
}

main "$@"
