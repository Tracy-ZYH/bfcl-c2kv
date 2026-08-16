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
DEVICES="${DEVICES:-2,3,4,5}"
PORTS="${PORTS:-33200,33201,33202,33203}"
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/history_full_closed_loop/logs/details.jsonl}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_oracle_full_success}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

MODES=(
  c2kv4_oracle_correct1
  c2kv4_oracle_correct2
  c2kv4_oracle_correct4
  c2kv4_oracle_correct_all
)

IFS=',' read -r DEVICE0 DEVICE1 DEVICE2 DEVICE3 _ <<< "${DEVICES}"
IFS=',' read -r PORT0 PORT1 PORT2 PORT3 _ <<< "${PORTS}"
DEV_ARRAY=("${DEVICE0}" "${DEVICE1}" "${DEVICE2}" "${DEVICE3}")
PORT_ARRAY=("${PORT0}" "${PORT1}" "${PORT2}" "${PORT3}")
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

log_info "BFCL history oracle run starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES} RATIO=${RATIO}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
log_info "IDS_PATH=${IDS_PATH}"
log_info "REFERENCE_DETAILS=${REFERENCE_DETAILS}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  log_info "Cleaning previous oracle result/score under ${RUN_ROOT}"
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
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
    finally:
        sock.close()
except PermissionError as exc:
    print(f"[WARN] socket permission check failed for port {port}: {exc}", file=sys.stderr)
    sys.exit(0)
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
  local mode="$1"
  local device="$2"
  local port="$3"
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
  SERVER_PIDS+=("$!")
  log_info "[server:${mode}] device=${device} port=${port} pid=$! log=${log}"
}

wait_health() {
  local mode="$1"
  local port="$2"
  local pid="$3"
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      log_info "[server:${mode}] crashed; last log:"
      tail -n 120 "${RUN_ROOT}/${mode}/logs/server_"*"_${port}.log" || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      log_info "[server:${mode}] healthy on port ${port}"
      return 0
    fi
    if [ $((attempt % 15)) -eq 0 ]; then
      log_info "[server:${mode}] waiting for health on port ${port} (attempt ${attempt}/900)"
    fi
    sleep 2
  done
  log_info "[server:${mode}] health check timed out"
  tail -n 120 "${RUN_ROOT}/${mode}/logs/server_"*"_${port}.log" || true
  return 1
}

run_mode() {
  local mode="$1"
  local port="$2"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_oracle \
      --oracle-mode "${mode}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${IDS_PATH}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${port}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${REFERENCE_DETAILS}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --details-path "${RUN_ROOT}/${mode}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${mode}/logs/oracle_metrics.jsonl" \
      --summary-path "${RUN_ROOT}/${mode}/logs/run_summary.json" \
      --ratio "${RATIO}"
  ) > "${RUN_ROOT}/${mode}/logs/run.log" 2>&1
  log_info "[runner:${mode}] done"
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
  log_info "[eval:${mode}] done"
}

for i in 0 1 2 3; do
  if [ -z "${DEV_ARRAY[$i]}" ] || [ -z "${PORT_ARRAY[$i]}" ]; then
    echo "DEVICES and PORTS must contain four comma-separated values."
    exit 1
  fi
  check_port_free "${PORT_ARRAY[$i]}"
done

for i in 0 1 2 3; do
  start_server "${MODES[$i]}" "${DEV_ARRAY[$i]}" "${PORT_ARRAY[$i]}"
done
for i in 0 1 2 3; do
  wait_health "${MODES[$i]}" "${PORT_ARRAY[$i]}" "${SERVER_PIDS[$i]}"
done

for i in 0 1 2 3; do
  run_mode "${MODES[$i]}" "${PORT_ARRAY[$i]}" &
  RUNNER_PIDS+=("$!")
done

failed=0
for pid in "${RUNNER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
RUNNER_PIDS=()
if [ "${failed}" -ne 0 ]; then
  log_info "At least one oracle runner failed. Check ${RUN_ROOT}/*/logs/run.log"
  exit 1
fi

for mode in "${MODES[@]}"; do
  evaluate_mode "${mode}"
done

(
  cd "${ROOT}"
  exec "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_oracle.py \
    --run-root "${RUN_ROOT}" \
    --category "${CATEGORY}" \
    --model "${MODEL_ID}"
) > "${RUN_ROOT}/compare.log" 2>&1

log_info "History oracle run complete"
log_info "Report: ${RUN_ROOT}/report.md"
log_info "Summary: ${RUN_ROOT}/summary.json"
