#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="/home/zhuyuhan/project/gorilla/berkeley-function-call-leaderboard"
SGLANG_ROOT="/home/zhuyuhan/project/kvoffload-sglang"
BFCL_PYTHON="/home/zhuyuhan/miniconda3/envs/bfcl/bin/python"
SGLANG_PYTHON="/home/zhuyuhan/miniconda3/envs/sglang/bin/python"
MODEL_PATH="/home/zhuyuhan/project/c2kv/checkpoints/qwen3-4b-agent-tooldoc-hardneg-npu"
TOKENIZER_PATH="/home/zhuyuhan/project/c2kv/models/Qwen3-4B-Instruct-2507"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507-FC"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
RATIO="${RATIO:-4}"
HYBRID_TOP_K="${HYBRID_TOP_K:-3}"
DEVICES="${DEVICES:-1,2,3}"
PORTS="${PORTS:-32000,32001,32002}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/tooldef_hardneg_multi_turn_base_200}"
NUM_THREADS="${NUM_THREADS:-1}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

IFS=',' read -r FULL_DEVICE C2KV_DEVICE HYBRID_DEVICE <<< "${DEVICES}"
IFS=',' read -r FULL_PORT C2KV_PORT HYBRID_PORT <<< "${PORTS}"

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

log_info "BFCL C2KV multi-turn run starting"
log_info "ROOT=${ROOT}"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES} RATIO=${RATIO} HYBRID_TOP_K=${HYBRID_TOP_K}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
log_info "Sourcing Ascend environment"
source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh
log_info "Ascend environment sourced"

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  log_info "Cleaning previous result/score/summary under ${RUN_ROOT}"
  rm -rf "${RUN_ROOT}"/{full,c2kv,hybrid}/{result,score}
  rm -f "${RUN_ROOT}/summary.json" "${RUN_ROOT}/summary.csv" "${RUN_ROOT}/report.md"
fi
mkdir -p "${RUN_ROOT}"/{full,c2kv,hybrid}/{result,score,logs}

check_port_free() {
  local port="$1"
  local rc
  log_info "Checking port ${port}"
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
    sys.exit(2)
sys.exit(0 if result != 0 else 1)
PY
  rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    return 0
  fi
  if [ "${rc}" -eq 2 ]; then
    log_info "Port ${port} check skipped because socket permission was denied"
    return 0
  fi
  echo "Port ${port} is already in use. Stop the existing server or set PORTS to free ports."
  return 1
}

check_port_free "${FULL_PORT}"
check_port_free "${C2KV_PORT}"
check_port_free "${HYBRID_PORT}"

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
  local log="${RUN_ROOT}/${mode}/logs/server_health_${port}.log"
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      log_info "[server:${mode}] crashed; last log:"
      tail -n 120 "${RUN_ROOT}/${mode}/logs/server_"*"_${port}.log" || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >> "${log}" 2>&1; then
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
  local log="${RUN_ROOT}/${mode}/logs/run.log"
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_c2kv_multiturn \
      --mode "${mode}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${port}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --details-path "${RUN_ROOT}/${mode}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${mode}/logs/c2kv_metrics.jsonl" \
      --summary-path "${RUN_ROOT}/${mode}/logs/run_summary.json" \
      --ratio "${RATIO}" \
      --hybrid-top-k "${HYBRID_TOP_K}" \
      --router-scope last_user \
      --tool-document-mode per_tool \
      --exclude-state-log
  ) > "${log}" 2>&1 &
  RUNNER_PIDS+=("$!")
  log_info "[runner:${mode}] port=${port} pid=$! log=${log}"
}

evaluate_mode() {
  local mode="$1"
  local extra=()
  if [ "${MAX_EXAMPLES}" -lt 200 ]; then
    extra+=(--partial-eval)
  fi
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner \
      --model "${MODEL_ID}" \
      --test-category "${CATEGORY}" \
      --result-dir "${RUN_ROOT}/${mode}/result" \
      --score-dir "${RUN_ROOT}/${mode}/score" \
      "${extra[@]}"
  ) > "${RUN_ROOT}/${mode}/logs/eval.log" 2>&1
  log_info "[eval:${mode}] done"
}

start_server full "${FULL_DEVICE}" "${FULL_PORT}"
start_server c2kv "${C2KV_DEVICE}" "${C2KV_PORT}"
start_server hybrid "${HYBRID_DEVICE}" "${HYBRID_PORT}"

wait_health full "${FULL_PORT}" "${SERVER_PIDS[0]}"
wait_health c2kv "${C2KV_PORT}" "${SERVER_PIDS[1]}"
wait_health hybrid "${HYBRID_PORT}" "${SERVER_PIDS[2]}"

run_mode full "${FULL_PORT}"
run_mode c2kv "${C2KV_PORT}"
run_mode hybrid "${HYBRID_PORT}"

failed=0
for pid in "${RUNNER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
RUNNER_PIDS=()
if [ "${failed}" -ne 0 ]; then
  log_info "At least one runner failed. Check ${RUN_ROOT}/*/logs/run.log"
  exit 1
fi

evaluate_mode full
evaluate_mode c2kv
evaluate_mode hybrid

(
  cd "${ROOT}"
  exec "${BFCL_PYTHON}" c2kv_eval/analysis/compare_multiturn_modes.py \
    --run-root "${RUN_ROOT}" \
    --category "${CATEGORY}"
) | tee "${RUN_ROOT}/logs_compare.txt"

for pid in "${SERVER_PIDS[@]}"; do
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
  fi
done
wait >/dev/null 2>&1 || true
trap - INT TERM EXIT

log_info "Done. Summary: ${RUN_ROOT}/summary.json"
log_info "Report: ${RUN_ROOT}/report.md"
