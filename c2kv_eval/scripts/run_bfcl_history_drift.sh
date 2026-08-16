#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard"
SGLANG_ROOT="/home/zhuyuhan/project/kvoffload-sglang"
BFCL_PYTHON="/home/zhuyuhan/miniconda3/envs/bfcl/bin/python"
SGLANG_PYTHON="/home/zhuyuhan/miniconda3/envs/sglang/bin/python"
MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
RATIO="${RATIO:-4}"
RECENT_FULL_UNITS="${RECENT_FULL_UNITS:-2}"
DEVICES="${DEVICES:-2,3,4,5}"
PORTS="${PORTS:-33000,33001,33002,33003}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_drift_multi_turn_base_200}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
IDS_PATH="${IDS_PATH:-}"

IFS=',' read -r FULL_DEVICE TEACHER_DEVICE C2KV_DEVICE RECENT_DEVICE _ <<< "${DEVICES}"
IFS=',' read -r FULL_PORT TEACHER_PORT C2KV_PORT RECENT_PORT _ <<< "${PORTS}"

SERVER_PIDS=()
RUNNER_PIDS=()

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
  log_info "source ok: ${path}"
}

cleanup() {
  local status=$?
  for pid in "${RUNNER_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  for pid in "${SERVER_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

log_info "BFCL history drift run starting"
log_info "ROOT=${ROOT}"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES} RATIO=${RATIO} RECENT_FULL_UNITS=${RECENT_FULL_UNITS}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
if [ -n "${IDS_PATH}" ]; then
  log_info "IDS_PATH=${IDS_PATH}"
fi
log_info "No --exclude-state-log is used; state logs are retained."

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

MODES=(
  history_full_closed_loop
  history_c2kv4_teacher_forced
  history_c2kv4_closed_loop
  history_recent2_full_rest_c2kv4
)

if [ -z "${FULL_DEVICE}" ] || [ -z "${TEACHER_DEVICE}" ] || [ -z "${C2KV_DEVICE}" ] || [ -z "${RECENT_DEVICE}" ]; then
  echo "DEVICES must contain four comma-separated NPU ids, e.g. DEVICES=2,3,4,5"
  exit 1
fi

if [ -z "${FULL_PORT}" ] || [ -z "${TEACHER_PORT}" ] || [ -z "${C2KV_PORT}" ] || [ -z "${RECENT_PORT}" ]; then
  echo "PORTS must contain four comma-separated ports, e.g. PORTS=33000,33001,33002,33003"
  exit 1
fi

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  log_info "Cleaning previous result/score/summary under ${RUN_ROOT}"
  for mode in "${MODES[@]}"; do
    rm -rf "${RUN_ROOT}/${mode}/result" "${RUN_ROOT}/${mode}/score"
  done
  rm -f "${RUN_ROOT}/summary.json" "${RUN_ROOT}/summary.csv" "${RUN_ROOT}/report.md"
fi
for mode in "${MODES[@]}"; do
  mkdir -p "${RUN_ROOT}/${mode}/result" "${RUN_ROOT}/${mode}/score" "${RUN_ROOT}/${mode}/logs"
done

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
    echo "Port ${port} is already in use. Stop the existing server or set PORTS to free ports."
    return 1
  fi
}

start_server() {
  local name="$1"
  local device="$2"
  local port="$3"
  local log="${RUN_ROOT}/${name}/logs/server_${device}_${port}.log"
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
  SERVER_PIDS+=("$!")
  log_info "[server:${name}] device=${device} port=${port} pid=$! log=${log}"
}

wait_health() {
  local name="$1"
  local port="$2"
  local pid="$3"
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      log_info "[server:${name}] crashed; last log:"
      tail -n 120 "${RUN_ROOT}/${name}/logs/server_"*"_${port}.log" || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      log_info "[server:${name}] healthy on port ${port}"
      return 0
    fi
    if [ $((attempt % 15)) -eq 0 ]; then
      log_info "[server:${name}] waiting for health on port ${port} (attempt ${attempt}/900)"
    fi
    sleep 2
  done
  log_info "[server:${name}] health check timed out"
  tail -n 120 "${RUN_ROOT}/${name}/logs/server_"*"_${port}.log" || true
  return 1
}

run_mode() {
  local mode="$1"
  local port="$2"
  local reference_path="${3:-}"
  local log="${RUN_ROOT}/${mode}/logs/run.log"
  local extra=()
  if [ -n "${reference_path}" ]; then
    extra+=(--reference-details-path "${reference_path}")
  fi
  if [ -n "${IDS_PATH}" ]; then
    extra+=(--ids-path "${IDS_PATH}")
  fi
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_drift \
      --mode "${mode}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${port}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --details-path "${RUN_ROOT}/${mode}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${mode}/logs/drift_metrics.jsonl" \
      --summary-path "${RUN_ROOT}/${mode}/logs/run_summary.json" \
      --ratio "${RATIO}" \
      --recent-full-units "${RECENT_FULL_UNITS}" \
      "${extra[@]}"
  ) > "${log}" 2>&1
  log_info "[runner:${mode}] done log=${log}"
}

evaluate_mode() {
  local mode="$1"
  local cmd=(
    "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner
    --model "${MODEL_ID}"
    --test-category "${CATEGORY}"
    --result-dir "${RUN_ROOT}/${mode}/result"
    --score-dir "${RUN_ROOT}/${mode}/score"
  )
  if [ "${MAX_EXAMPLES}" -lt 200 ] || [ -n "${IDS_PATH}" ]; then
    cmd+=(--partial-eval)
  fi
  (
    cd "${ROOT}"
    exec "${cmd[@]}"
  ) > "${RUN_ROOT}/${mode}/logs/eval.log" 2>&1
  log_info "[eval:${mode}] done"
}

check_port_free "${FULL_PORT}"
check_port_free "${TEACHER_PORT}"
check_port_free "${C2KV_PORT}"
check_port_free "${RECENT_PORT}"

start_server history_full_closed_loop "${FULL_DEVICE}" "${FULL_PORT}"
start_server history_c2kv4_teacher_forced "${TEACHER_DEVICE}" "${TEACHER_PORT}"
start_server history_c2kv4_closed_loop "${C2KV_DEVICE}" "${C2KV_PORT}"
start_server history_recent2_full_rest_c2kv4 "${RECENT_DEVICE}" "${RECENT_PORT}"

wait_health history_full_closed_loop "${FULL_PORT}" "${SERVER_PIDS[0]}"
wait_health history_c2kv4_teacher_forced "${TEACHER_PORT}" "${SERVER_PIDS[1]}"
wait_health history_c2kv4_closed_loop "${C2KV_PORT}" "${SERVER_PIDS[2]}"
wait_health history_recent2_full_rest_c2kv4 "${RECENT_PORT}" "${SERVER_PIDS[3]}"

run_mode history_full_closed_loop "${FULL_PORT}"
REFERENCE_DETAILS="${RUN_ROOT}/history_full_closed_loop/logs/details.jsonl"

run_mode history_c2kv4_teacher_forced "${TEACHER_PORT}" "${REFERENCE_DETAILS}" &
RUNNER_PIDS+=("$!")
run_mode history_c2kv4_closed_loop "${C2KV_PORT}" "${REFERENCE_DETAILS}" &
RUNNER_PIDS+=("$!")
run_mode history_recent2_full_rest_c2kv4 "${RECENT_PORT}" "${REFERENCE_DETAILS}" &
RUNNER_PIDS+=("$!")

failed=0
for pid in "${RUNNER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
RUNNER_PIDS=()
if [ "${failed}" -ne 0 ]; then
  log_info "At least one history drift runner failed. Check ${RUN_ROOT}/*/logs/run.log"
  exit 1
fi

for mode in "${MODES[@]}"; do
  evaluate_mode "${mode}"
done

(
  cd "${ROOT}"
  exec "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_drift.py \
    --run-root "${RUN_ROOT}" \
    --category "${CATEGORY}"
) > "${RUN_ROOT}/compare.log" 2>&1

log_info "History drift run complete"
log_info "Report: ${RUN_ROOT}/report.md"
log_info "Summary: ${RUN_ROOT}/summary.json"
