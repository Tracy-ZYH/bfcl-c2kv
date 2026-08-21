#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
DEVICES_CSV="${DEVICES:-6,7}"
PORTS_CSV="${PORTS:-34006,34007}"
CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
COMPRESSION_RATIO="${COMPRESSION_RATIO:-4}"
VERIFIER="${VERIFIER:-oracle}"
ATTRIBUTION_SAFETY_MARGIN="${ATTRIBUTION_SAFETY_MARGIN:-0}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_phase3_phase4_stable52_$(date +%Y%m%d_%H%M%S)}"
STABLE_ROOT="${STABLE_ROOT:-$(ls -td /home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_* | head -n 1)}"
IDS_PATH="${IDS_PATH:-${STABLE_ROOT}/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-${STABLE_ROOT}/frozen_reference/details.jsonl}"

IFS=',' read -r -a DEVICES <<< "${DEVICES_CSV}"
IFS=',' read -r -a PORTS <<< "${PORTS_CSV}"
if [ "${#DEVICES[@]}" -ne "${#PORTS[@]}" ]; then
  echo "DEVICES and PORTS must have the same length."
  exit 1
fi

cd "${ROOT}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "STABLE_ROOT=${STABLE_ROOT}"
echo "IDS_PATH=${IDS_PATH}"
echo "REFERENCE_DETAILS=${REFERENCE_DETAILS}"
echo "DEVICES=${DEVICES_CSV}"
echo "PORTS=${PORTS_CSV}"

PIDS=()
PID_LABELS=()
NEXT_SLOT=0
FAILED=0

wait_one() {
  local pid="${PIDS[0]}"
  local label="${PID_LABELS[0]}"
  if wait "${pid}"; then
    echo "[done] ${label}"
  else
    echo "[failed] ${label}" >&2
    FAILED=1
  fi
  PIDS=("${PIDS[@]:1}")
  PID_LABELS=("${PID_LABELS[@]:1}")
}

launch_one() {
  local interval="$1"
  local attribution="$2"
  local backend="$3"
  local recovery_mode="$4"
  local mode="$5"
  local slot=$((NEXT_SLOT % ${#DEVICES[@]}))
  local device="${DEVICES[$slot]}"
  local port="${PORTS[$slot]}"
  NEXT_SLOT=$((NEXT_SLOT + 1))

  echo "[launch] ${mode} device=${device} port=${port}"
  (
    CHECKPOINT_INTERVAL="${interval}" \
    RECOVERY_MODE="${recovery_mode}" \
    ATTRIBUTION="${attribution}" \
    ATTRIBUTION_SAFETY_MARGIN="${ATTRIBUTION_SAFETY_MARGIN}" \
    ROLLBACK_BACKEND="${backend}" \
    MODE="${mode}" \
    VERIFIER="${VERIFIER}" \
    COMPRESSION_RATIO="${COMPRESSION_RATIO}" \
    MAX_EXAMPLES="${MAX_EXAMPLES}" \
    DEVICE="${device}" \
    PORT="${port}" \
    RUN_ROOT="${RUN_ROOT}" \
    IDS_PATH="${IDS_PATH}" \
    REFERENCE_DETAILS="${REFERENCE_DETAILS}" \
    CLEAN_OUTPUT=1 \
    RUN_COMPARE=0 \
    bash c2kv_eval/scripts/run_bfcl_history_multistep_checkpoint.sh
  ) &
  PIDS+=("$!")
  PID_LABELS+=("${mode}")
  if [ "${#PIDS[@]}" -ge "${#DEVICES[@]}" ]; then
    wait_one
  fi
}

for interval in 2 4; do
  launch_one "${interval}" whole_segment message_replay whole_segment \
    "phase34_k${interval}_whole_message"
  launch_one "${interval}" oracle_first_bad message_replay first_bad_suffix \
    "phase34_k${interval}_oracle_first_bad_message"
  launch_one "${interval}" oracle_first_bad kv_restore first_bad_suffix \
    "phase34_k${interval}_oracle_first_bad_kv"
  launch_one "${interval}" heuristic message_replay first_bad_suffix \
    "phase34_k${interval}_heuristic_message"
  launch_one "${interval}" heuristic kv_restore first_bad_suffix \
    "phase34_k${interval}_heuristic_kv"
done

while [ "${#PIDS[@]}" -gt 0 ]; do
  wait_one
done

MODES=phase34_k2_whole_message,phase34_k2_oracle_first_bad_message,phase34_k2_oracle_first_bad_kv,phase34_k2_heuristic_message,phase34_k2_heuristic_kv,phase34_k4_whole_message,phase34_k4_oracle_first_bad_message,phase34_k4_oracle_first_bad_kv,phase34_k4_heuristic_message,phase34_k4_heuristic_kv \
RUN_ROOT="${RUN_ROOT}" \
REFERENCE_DETAILS="${REFERENCE_DETAILS}" \
bash c2kv_eval/scripts/merge_bfcl_history_multistep_checkpoint.sh

echo "Report: ${RUN_ROOT}/multistep_checkpoint_summary.md"
cat "${RUN_ROOT}/multistep_checkpoint_summary.md"

exit "${FAILED}"
