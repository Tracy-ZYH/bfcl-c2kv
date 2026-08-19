#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/zhuyuhan/miniconda3/envs/bfcl/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/zhuyuhan/miniconda3/envs/sglang/bin/python}"

MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
RATIO="${RATIO:-4}"
DEVICES="${DEVICES:-3,4,5,6,7}"
PORTS="${PORTS:-33500,33501,33502,33503,33504}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_unified_full_success_54_rerun}"
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/history_full_closed_loop/logs/details.jsonl}"
REFERENCE_MODE="${REFERENCE_MODE:-current}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

IFS=',' read -r FULL_DEVICE PURE_DEVICE STRICT_DEVICE C2KV_DEVICE CKPT_DEVICE _ <<< "${DEVICES}"
IFS=',' read -r FULL_PORT PURE_PORT STRICT_PORT C2KV_PORT CKPT_PORT _ <<< "${PORTS}"

SERVER_PIDS=""
LAST_SERVER_PID=""

log_info() {
  echo "[$(date '+%F %T')] $*"
}

source_env_file() {
  local path="$1"
  local rc
  log_info "source ${path}"
  set +e +u
  source "${path}"
  rc=$?
  set -e
  set +u
  if [ "${rc}" -ne 0 ]; then
    log_info "source failed: ${path} rc=${rc}"
    return "${rc}"
  fi
}

cleanup() {
  local status=$?
  set +e
  for pid in ${SERVER_PIDS}; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

require_values() {
  if [ -z "${FULL_DEVICE}" ] || [ -z "${PURE_DEVICE}" ] || [ -z "${STRICT_DEVICE}" ] || [ -z "${C2KV_DEVICE}" ] || [ -z "${CKPT_DEVICE}" ]; then
    echo "DEVICES must contain five comma-separated NPU ids, e.g. DEVICES=3,4,5,6,7"
    exit 1
  fi
  if [ -z "${FULL_PORT}" ] || [ -z "${PURE_PORT}" ] || [ -z "${STRICT_PORT}" ] || [ -z "${C2KV_PORT}" ] || [ -z "${CKPT_PORT}" ]; then
    echo "PORTS must contain five comma-separated ports, e.g. PORTS=33500,33501,33502,33503,33504"
    exit 1
  fi
}

prepare_dirs() {
  local modes=(
    history_full_closed_loop
    pure_full_replay
    c2kv4_oracle_correct_all_strict
    history_c2kv4_closed_loop
    ckpt_i1_oracle_current_step
  )
  if [ "${CLEAN_OUTPUT}" = "1" ]; then
    for mode in "${modes[@]}"; do
      rm -rf "${RUN_ROOT}/${mode}/result" "${RUN_ROOT}/${mode}/score"
    done
  fi
  for mode in "${modes[@]}"; do
    mkdir -p "${RUN_ROOT}/${mode}/result" "${RUN_ROOT}/${mode}/score" "${RUN_ROOT}/${mode}/logs"
  done
}

check_port_free() {
  local port="$1"
  set +e
  "${BFCL_PYTHON}" - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
try:
    result = sock.connect_ex(("127.0.0.1", port))
finally:
    sock.close()
sys.exit(0 if result != 0 else 1)
PY
  local rc=$?
  set -e
  if [ "${rc}" -ne 0 ]; then
    echo "Port ${port} is already in use. Stop the existing server or change PORTS."
    exit 1
  fi
}

start_server() {
  local device="$1"
  local port="$2"
  local mode="$3"
  local log="${RUN_ROOT}/${mode}/logs/server_${device}_${port}.log"
  (
    cd "${SGLANG_ROOT}"
    SGLANG_DEBUG_MEMORY_POOL=1 \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_EMPTY_CACHE_INTERVAL=1 \
    ASCEND_LAUNCH_BLOCKING=1 \
    TASK_QUEUE_ENABLE=1 \
    no_proxy='*' \
    NO_PROXY='*' \
    http_proxy='' \
    https_proxy='' \
    HTTP_PROXY='' \
    HTTPS_PROXY='' \
    ASCEND_RT_VISIBLE_DEVICES="${device}" \
    exec "${SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${MODEL_ID}" \
      --model-impl sglang \
      --device npu \
      --attention-backend ascend \
      --tool-call-parser qwen25 \
      --enable-c2kv \
      --dtype bfloat16 \
      --mem-fraction-static 0.55 \
      --host 127.0.0.1 \
      --port "${port}"
  ) > "${log}" 2>&1 &
  local pid=$!
  SERVER_PIDS="${SERVER_PIDS} ${pid}"
  log_info "[server] mode=${mode} device=${device} port=${port} pid=${pid} log=${log}"
  LAST_SERVER_PID="${pid}"
}

wait_health() {
  local pid="$1"
  local port="$2"
  local mode="$3"
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      log_info "[server] crashed: ${mode}"
      tail -n 120 "${RUN_ROOT}/${mode}/logs"/server_*.log || true
      exit 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      log_info "[server] healthy: ${mode} port=${port}"
      return
    fi
    if [ $((attempt % 15)) -eq 0 ]; then
      log_info "[server] waiting: ${mode} port=${port} attempt=${attempt}/900"
    fi
    sleep 2
  done
  log_info "[server] health timeout: ${mode}"
  tail -n 120 "${RUN_ROOT}/${mode}/logs"/server_*.log || true
  exit 1
}

run_drift_mode() {
  local mode="$1"
  local port="$2"
  local ids_path="$3"
  local reference_details="$4"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_drift \
      --mode "${mode}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${ids_path}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${port}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${reference_details}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --details-path "${RUN_ROOT}/${mode}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${mode}/logs/drift_metrics.jsonl" \
      --summary-path "${RUN_ROOT}/${mode}/logs/run_summary.json" \
      --ratio "${RATIO}"
  ) > "${RUN_ROOT}/${mode}/logs/run.log" 2>&1
  log_info "[runner] done: ${mode}"
}

run_oracle_mode() {
  local mode="$1"
  local base_url="$2"
  local ids_path="$3"
  local reference_details="$4"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_oracle \
      --oracle-mode "${mode}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${ids_path}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "${base_url}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${reference_details}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --details-path "${RUN_ROOT}/${mode}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${mode}/logs/oracle_metrics.jsonl" \
      --summary-path "${RUN_ROOT}/${mode}/logs/run_summary.json" \
      --ratio "${RATIO}"
  ) > "${RUN_ROOT}/${mode}/logs/run.log" 2>&1
  log_info "[runner] done: ${mode}"
}

run_checkpoint_mode() {
  local ids_path="$1"
  local reference_details="$2"
  local mode="ckpt_i1_oracle_current_step"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.eval_bfcl_history_checkpoint \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${ids_path}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${CKPT_PORT}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${reference_details}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --details-path "${RUN_ROOT}/${mode}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${mode}/logs/checkpoint_metrics.jsonl" \
      --step-metrics-path "${RUN_ROOT}/${mode}/logs/checkpoint_steps.jsonl" \
      --summary-path "${RUN_ROOT}/${mode}/logs/run_summary.json" \
      --compression-ratio "${RATIO}" \
      --checkpoint-interval 1 \
      --verifier oracle \
      --verify-threshold 0 \
      --recovery-mode current_step
  ) > "${RUN_ROOT}/${mode}/logs/run.log" 2>&1
  log_info "[runner] done: ${mode}"
}

evaluate_mode() {
  local mode="$1"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner \
      --model "${MODEL_ID}" \
      --test-category "${CATEGORY}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --score-dir "${RUN_ROOT}/${mode}/score" \
      --partial-eval
  ) > "${RUN_ROOT}/${mode}/logs/eval.log" 2>&1
  log_info "[eval] done: ${mode}"
}

export_success_ids() {
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" c2kv_eval/analysis/export_success_ids.py \
      --run-root "${RUN_ROOT}" \
      --mode history_full_closed_loop \
      --category "${CATEGORY}" \
      --output-path "${RUN_ROOT}/current_full_success_ids.txt"
  ) > "${RUN_ROOT}/history_full_closed_loop/logs/export_success_ids.log" 2>&1
  log_info "[ids] current Full success ids: ${RUN_ROOT}/current_full_success_ids.txt"
}

write_reports() {
  local reference_details="$1"
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_drift.py \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --model "${MODEL_ID}" \
      --modes history_full_closed_loop,history_c2kv4_closed_loop
    cp "${RUN_ROOT}/report.md" "${RUN_ROOT}/report_history_drift.md"

    "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_oracle.py \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --model "${MODEL_ID}" \
      --modes pure_full_replay,c2kv4_oracle_correct_all_strict
    cp "${RUN_ROOT}/report.md" "${RUN_ROOT}/report_history_oracle.md"

    "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_checkpoint.py \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --model "${MODEL_ID}" \
      --modes ckpt_i1_oracle_current_step \
      --reference-details-path "${reference_details}"
    cp "${RUN_ROOT}/report.md" "${RUN_ROOT}/report_history_checkpoint.md"

    {
      echo "# BFCL History Unified Full-Success 54 Rerun"
      echo
      echo "- Drift report: report_history_drift.md"
      echo "- Oracle report: report_history_oracle.md"
      echo "- Checkpoint report: report_history_checkpoint.md"
      echo
      echo "## Drift"
      echo
      sed '1d' "${RUN_ROOT}/report_history_drift.md"
      echo
      echo "## Oracle"
      echo
      sed '1d' "${RUN_ROOT}/report_history_oracle.md"
      echo
      echo "## Checkpoint"
      echo
      sed '1d' "${RUN_ROOT}/report_history_checkpoint.md"
    } > "${RUN_ROOT}/report.md"
  ) > "${RUN_ROOT}/compare.log" 2>&1
  log_info "[compare] done"
}

main() {
  require_values
  log_info "BFCL history unified 5-card run starting"
  log_info "RUN_ROOT=${RUN_ROOT}"
  log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
  log_info "IDS_PATH=${IDS_PATH}"
  log_info "REFERENCE_DETAILS=${REFERENCE_DETAILS}"
  log_info "REFERENCE_MODE=${REFERENCE_MODE}"

  source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
  source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

  prepare_dirs
  check_port_free "${FULL_PORT}"
  check_port_free "${STRICT_PORT}"
  check_port_free "${C2KV_PORT}"
  check_port_free "${CKPT_PORT}"

  local full_pid strict_pid c2kv_pid ckpt_pid
  start_server "${FULL_DEVICE}" "${FULL_PORT}" history_full_closed_loop
  full_pid="${LAST_SERVER_PID}"
  start_server "${STRICT_DEVICE}" "${STRICT_PORT}" c2kv4_oracle_correct_all_strict
  strict_pid="${LAST_SERVER_PID}"
  start_server "${C2KV_DEVICE}" "${C2KV_PORT}" history_c2kv4_closed_loop
  c2kv_pid="${LAST_SERVER_PID}"
  start_server "${CKPT_DEVICE}" "${CKPT_PORT}" ckpt_i1_oracle_current_step
  ckpt_pid="${LAST_SERVER_PID}"

  wait_health "${full_pid}" "${FULL_PORT}" history_full_closed_loop
  wait_health "${strict_pid}" "${STRICT_PORT}" c2kv4_oracle_correct_all_strict
  wait_health "${c2kv_pid}" "${C2KV_PORT}" history_c2kv4_closed_loop
  wait_health "${ckpt_pid}" "${CKPT_PORT}" ckpt_i1_oracle_current_step

  run_drift_mode history_full_closed_loop "${FULL_PORT}" "${IDS_PATH}" "${REFERENCE_DETAILS}"
  evaluate_mode history_full_closed_loop

  local downstream_ids_path="${IDS_PATH}"
  local downstream_reference_details="${REFERENCE_DETAILS}"
  if [ "${REFERENCE_MODE}" = "current" ]; then
    export_success_ids
    downstream_ids_path="${RUN_ROOT}/current_full_success_ids.txt"
    downstream_reference_details="${RUN_ROOT}/history_full_closed_loop/logs/details.jsonl"
    log_info "Downstream reference switched to current Full rerun."
    log_info "downstream_ids_path=${downstream_ids_path}"
    log_info "downstream_reference_details=${downstream_reference_details}"
  fi

  local runner_pids=""
  run_oracle_mode pure_full_replay "http://127.0.0.1:1" "${downstream_ids_path}" "${downstream_reference_details}" &
  runner_pids="${runner_pids} $!"
  run_oracle_mode c2kv4_oracle_correct_all_strict "http://127.0.0.1:${STRICT_PORT}" "${downstream_ids_path}" "${downstream_reference_details}" &
  runner_pids="${runner_pids} $!"
  run_drift_mode history_c2kv4_closed_loop "${C2KV_PORT}" "${downstream_ids_path}" "${downstream_reference_details}" &
  runner_pids="${runner_pids} $!"
  run_checkpoint_mode "${downstream_ids_path}" "${downstream_reference_details}" &
  runner_pids="${runner_pids} $!"
  for pid in ${runner_pids}; do
    wait "${pid}"
  done

  evaluate_mode pure_full_replay
  evaluate_mode c2kv4_oracle_correct_all_strict
  evaluate_mode history_c2kv4_closed_loop
  evaluate_mode ckpt_i1_oracle_current_step
  write_reports "${downstream_reference_details}"

  log_info "done: ${RUN_ROOT}"
}

main "$@"
