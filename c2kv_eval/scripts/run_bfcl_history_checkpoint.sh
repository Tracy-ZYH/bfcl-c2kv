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
COMPRESSION_RATIO="${COMPRESSION_RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
VERIFIER="${VERIFIER:-oracle}"
VERIFY_THRESHOLD="${VERIFY_THRESHOLD:-0}"
VERIFY_LAYERS="${VERIFY_LAYERS:-25%,50%,75%,last}"
ONLINE_VERIFY="${ONLINE_VERIFY:-0}"
REUSE_CANDIDATE_READOUT="${REUSE_CANDIDATE_READOUT:-1}"
RECOVERY_MODE="${RECOVERY_MODE:-current_step}"
DEVICE="${DEVICE:-4}"
PORT="${PORT:-33300}"
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/history_full_closed_loop/logs/details.jsonl}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_checkpoint_full_success_54}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
MODE="${MODE:-ckpt_i${CHECKPOINT_INTERVAL}_${VERIFIER}_${RECOVERY_MODE}}"

SERVER_PID=""

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
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

log_info "BFCL history checkpoint run starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "MODE=${MODE}"
log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES} COMPRESSION_RATIO=${COMPRESSION_RATIO}"
log_info "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL} VERIFIER=${VERIFIER} VERIFY_THRESHOLD=${VERIFY_THRESHOLD} RECOVERY_MODE=${RECOVERY_MODE}"
log_info "VERIFY_LAYERS=${VERIFY_LAYERS} ONLINE_VERIFY=${ONLINE_VERIFY} REUSE_CANDIDATE_READOUT=${REUSE_CANDIDATE_READOUT}"
log_info "DEVICE=${DEVICE} PORT=${PORT}"
log_info "IDS_PATH=${IDS_PATH}"
log_info "REFERENCE_DETAILS=${REFERENCE_DETAILS}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  log_info "Cleaning previous result/score under ${RUN_ROOT}/${MODE}"
  rm -rf "${RUN_ROOT}/${MODE}/result" "${RUN_ROOT}/${MODE}/score"
fi
mkdir -p "${RUN_ROOT}/${MODE}/result" "${RUN_ROOT}/${MODE}/score" "${RUN_ROOT}/${MODE}/logs"

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
    echo "Port ${port} is already in use. Stop the existing server or set PORT to a free port."
    return 1
  fi
}

start_server() {
  local log="${RUN_ROOT}/${MODE}/logs/server_${DEVICE}_${PORT}.log"
  local server_cmd=(
    "${SGLANG_PYTHON}" -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --served-model-name "${MODEL_ID}"
    --model-impl sglang
    --device npu
    --attention-backend ascend
    --tool-call-parser qwen25
    --enable-c2kv
  )
  if [ "${VERIFIER}" = "kv_divergence" ] || [ "${VERIFIER}" = "instant_kv" ] || [ "${VERIFIER}" = "cumulative_kv" ]; then
    server_cmd+=(--enable-return-hidden-states)
  fi
  server_cmd+=(
    --dtype bfloat16
    --mem-fraction-static 0.55
    --host 127.0.0.1
    --port "${PORT}"
  )
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
    ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
    exec "${server_cmd[@]}"
  ) > "${log}" 2>&1 &
  SERVER_PID="$!"
  log_info "[server] device=${DEVICE} port=${PORT} pid=${SERVER_PID} log=${log}"
}

wait_health() {
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      log_info "[server] crashed; last log:"
      tail -n 120 "${RUN_ROOT}/${MODE}/logs/server_"*"_${PORT}.log" || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      log_info "[server] healthy on port ${PORT}"
      return 0
    fi
    if [ $((attempt % 15)) -eq 0 ]; then
      log_info "[server] waiting for health on port ${PORT} (attempt ${attempt}/900)"
    fi
    sleep 2
  done
  log_info "[server] health check timed out"
  tail -n 120 "${RUN_ROOT}/${MODE}/logs/server_"*"_${PORT}.log" || true
  return 1
}

run_eval() {
  local eval_cmd=(
    "${BFCL_PYTHON}" -m c2kv_eval.adapters.eval_bfcl_history_checkpoint
    --category "${CATEGORY}"
    --max-examples "${MAX_EXAMPLES}"
    --ids-path "${IDS_PATH}"
    --model "${MODEL_ID}"
    --served-model-name "${MODEL_ID}"
    --base-url "http://127.0.0.1:${PORT}"
    --tokenizer-path "${TOKENIZER_PATH}"
    --reference-details-path "${REFERENCE_DETAILS}"
    --result-dir "${RUN_ROOT}/${MODE}/result"
    --details-path "${RUN_ROOT}/${MODE}/logs/details.jsonl"
    --metrics-path "${RUN_ROOT}/${MODE}/logs/checkpoint_metrics.jsonl"
    --step-metrics-path "${RUN_ROOT}/${MODE}/logs/checkpoint_steps.jsonl"
    --summary-path "${RUN_ROOT}/${MODE}/logs/run_summary.json"
    --compression-ratio "${COMPRESSION_RATIO}"
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
    --verifier "${VERIFIER}"
    --verify-threshold "${VERIFY_THRESHOLD}"
    --verify-layers "${VERIFY_LAYERS}"
    --recovery-mode "${RECOVERY_MODE}"
  )
  if [ "${ONLINE_VERIFY}" = "1" ]; then
    eval_cmd+=(--online-verify)
  fi
  if [ "${REUSE_CANDIDATE_READOUT}" = "0" ]; then
    eval_cmd+=(--no-reuse-candidate-readout)
  else
    eval_cmd+=(--reuse-candidate-readout)
  fi
  (
    cd "${ROOT}"
    exec "${eval_cmd[@]}"
  ) > "${RUN_ROOT}/${MODE}/logs/run.log" 2>&1
  log_info "[runner:${MODE}] done"
}

evaluate_mode() {
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner \
      --model "${MODEL_ID}" \
      --test-category "${CATEGORY}" \
      --result-dir "${RUN_ROOT}/${MODE}/result" \
      --score-dir "${RUN_ROOT}/${MODE}/score" \
      --partial-eval
  ) > "${RUN_ROOT}/${MODE}/logs/eval.log" 2>&1
  log_info "[eval:${MODE}] done"
}

compare() {
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_checkpoint.py \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --model "${MODEL_ID}" \
      --modes "${MODE}"
  ) > "${RUN_ROOT}/${MODE}/logs/compare.log" 2>&1
  log_info "[compare:${MODE}] done"
}

check_port_free "${PORT}"
start_server
wait_health
run_eval
evaluate_mode
compare

log_info "History checkpoint run complete"
log_info "Mode report is included in: ${RUN_ROOT}/report.md"
log_info "Mode logs: ${RUN_ROOT}/${MODE}/logs"
