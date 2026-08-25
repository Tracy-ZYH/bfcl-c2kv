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
RECOVERY_MODE="${RECOVERY_MODE:-since_checkpoint}"
RECOVERY_HORIZON="${RECOVERY_HORIZON:-auto}"
ATTRIBUTION="${ATTRIBUTION:-auto}"
ATTRIBUTION_SAFETY_MARGIN="${ATTRIBUTION_SAFETY_MARGIN:-0}"
ROLLBACK_BACKEND="${ROLLBACK_BACKEND:-message_replay}"
ENABLE_HICACHE="${ENABLE_HICACHE:-auto}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"
VERIFY_THRESHOLD="${VERIFY_THRESHOLD:-0}"
RULE_DETECTOR_THRESHOLD="${RULE_DETECTOR_THRESHOLD:-5}"
LOGISTIC_DETECTOR_FEATURES_CSV="${LOGISTIC_DETECTOR_FEATURES_CSV:-}"
LOGISTIC_DETECTOR_THRESHOLD="${LOGISTIC_DETECTOR_THRESHOLD:--1}"
ROLLBACK_POLICY="${ROLLBACK_POLICY:-attribution}"
ROLLBACK_DEPTH="${ROLLBACK_DEPTH:-1}"
COLLECT_CANDIDATE_DETECTOR_SIGNALS="${COLLECT_CANDIDATE_DETECTOR_SIGNALS:-0}"
CANDIDATE_LOGPROBS_TOP_K="${CANDIDATE_LOGPROBS_TOP_K:-20}"
CANDIDATE_HIDDEN_READOUT="${CANDIDATE_HIDDEN_READOUT:-0}"
CANDIDATE_ATTENTION_SUMMARY="${CANDIDATE_ATTENTION_SUMMARY:-0}"
DEVICE="${DEVICE:-3}"
PORT="${PORT:-33400}"
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_closed_loop_multi_turn_base_200/history_full_closed_loop/logs/details.jsonl}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_multistep_checkpoint_full_success_54}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
MODE="${MODE:-multistep_i${CHECKPOINT_INTERVAL}_${VERIFIER}_${RECOVERY_MODE}}"
RUN_COMPARE="${RUN_COMPARE:-1}"

if [[ "${RECOVERY_HORIZON}" != "auto" && "${MODE}" == *one_step* && "${RECOVERY_HORIZON}" != "one_step" ]]; then
  echo "MODE=${MODE} implies RECOVERY_HORIZON=one_step, but RECOVERY_HORIZON=${RECOVERY_HORIZON}."
  exit 1
fi
if [[ "${RECOVERY_HORIZON}" != "auto" && "${MODE}" == *suffix* && "${RECOVERY_HORIZON}" != "suffix" ]]; then
  echo "MODE=${MODE} implies RECOVERY_HORIZON=suffix, but RECOVERY_HORIZON=${RECOVERY_HORIZON}."
  exit 1
fi
if [[ "${RECOVERY_HORIZON}" != "auto" && "${MODE}" == *whole_segment* && "${RECOVERY_HORIZON}" != "whole_segment" ]]; then
  echo "MODE=${MODE} implies RECOVERY_HORIZON=whole_segment, but RECOVERY_HORIZON=${RECOVERY_HORIZON}."
  exit 1
fi

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
    echo "Port ${port} is already in use. Stop the existing server or set PORT."
    return 1
  fi
}

start_server() {
  local log="${RUN_ROOT}/${MODE}/logs/server_${DEVICE}_${PORT}.log"
  local hicache_args=()
  if [[ "${ENABLE_HICACHE}" == "1" || "${ENABLE_HICACHE}" == "true" || ( "${ENABLE_HICACHE}" == "auto" && ( "${ROLLBACK_BACKEND}" == "kv_restore" || "${ROLLBACK_BACKEND}" == "kv_restore_strict" ) ) ]]; then
    hicache_args+=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO}")
  fi
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
      --enable-cache-report \
      "${hicache_args[@]}" \
      --host 127.0.0.1 \
      --port "${PORT}"
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
      log_info "[server] waiting for health on port ${PORT} (${attempt}/900)"
    fi
    sleep 2
  done
  log_info "[server] health check timed out"
  tail -n 120 "${RUN_ROOT}/${MODE}/logs/server_"*"_${PORT}.log" || true
  return 1
}

run_eval() {
  local detector_signal_args=()
  if [[ "${COLLECT_CANDIDATE_DETECTOR_SIGNALS}" == "1" || "${COLLECT_CANDIDATE_DETECTOR_SIGNALS}" == "true" ]]; then
    detector_signal_args+=(--collect-candidate-detector-signals)
  fi
  if [[ "${CANDIDATE_HIDDEN_READOUT}" == "1" || "${CANDIDATE_HIDDEN_READOUT}" == "true" ]]; then
    detector_signal_args+=(--candidate-hidden-readout)
  fi
  if [[ "${CANDIDATE_ATTENTION_SUMMARY}" == "1" || "${CANDIDATE_ATTENTION_SUMMARY}" == "true" ]]; then
    detector_signal_args+=(--candidate-attention-summary)
  fi
  (
    cd "${ROOT}"
    exec "${BFCL_PYTHON}" -m c2kv_eval.adapters.eval_bfcl_history_checkpoint \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${IDS_PATH}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${PORT}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${REFERENCE_DETAILS}" \
      --result-dir "${RUN_ROOT}/${MODE}/result" \
      --details-path "${RUN_ROOT}/${MODE}/logs/details.jsonl" \
      --metrics-path "${RUN_ROOT}/${MODE}/logs/checkpoint_metrics.jsonl" \
      --step-metrics-path "${RUN_ROOT}/${MODE}/logs/checkpoint_steps.jsonl" \
      --segment-metrics-path "${RUN_ROOT}/${MODE}/logs/checkpoint_segments.jsonl" \
      --summary-path "${RUN_ROOT}/${MODE}/logs/run_summary.json" \
      --compression-ratio "${COMPRESSION_RATIO}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --verifier "${VERIFIER}" \
      --verify-threshold "${VERIFY_THRESHOLD}" \
      --rule-detector-threshold "${RULE_DETECTOR_THRESHOLD}" \
      --logistic-detector-features-csv "${LOGISTIC_DETECTOR_FEATURES_CSV}" \
      --logistic-detector-threshold "${LOGISTIC_DETECTOR_THRESHOLD}" \
      --recovery-mode "${RECOVERY_MODE}" \
      --recovery-horizon "${RECOVERY_HORIZON}" \
      --attribution "${ATTRIBUTION}" \
      --attribution-safety-margin "${ATTRIBUTION_SAFETY_MARGIN}" \
      --rollback-policy "${ROLLBACK_POLICY}" \
      --rollback-depth "${ROLLBACK_DEPTH}" \
      --rollback-backend "${ROLLBACK_BACKEND}" \
      --candidate-logprobs-top-k "${CANDIDATE_LOGPROBS_TOP_K}" \
      "${detector_signal_args[@]}"
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
      --reference-details-path "${REFERENCE_DETAILS}" \
      --modes "${MODE}"
  ) > "${RUN_ROOT}/${MODE}/logs/compare.log" 2>&1
  log_info "[compare:${MODE}] done"
}

log_info "BFCL history multi-step checkpoint run starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "MODE=${MODE}"
log_info "CATEGORY=${CATEGORY} MAX_EXAMPLES=${MAX_EXAMPLES}"
log_info "INTERVAL=${CHECKPOINT_INTERVAL} VERIFIER=${VERIFIER} RECOVERY_MODE=${RECOVERY_MODE}"
log_info "RECOVERY_HORIZON=${RECOVERY_HORIZON}"
log_info "ATTRIBUTION=${ATTRIBUTION} ATTRIBUTION_SAFETY_MARGIN=${ATTRIBUTION_SAFETY_MARGIN}"
log_info "ROLLBACK_POLICY=${ROLLBACK_POLICY} ROLLBACK_DEPTH=${ROLLBACK_DEPTH}"
log_info "RULE_DETECTOR_THRESHOLD=${RULE_DETECTOR_THRESHOLD}"
log_info "ROLLBACK_BACKEND=${ROLLBACK_BACKEND}"
log_info "COLLECT_CANDIDATE_DETECTOR_SIGNALS=${COLLECT_CANDIDATE_DETECTOR_SIGNALS} CANDIDATE_LOGPROBS_TOP_K=${CANDIDATE_LOGPROBS_TOP_K}"
log_info "ENABLE_HICACHE=${ENABLE_HICACHE} HICACHE_RATIO=${HICACHE_RATIO}"
log_info "DEVICE=${DEVICE} PORT=${PORT}"
log_info "RUN_COMPARE=${RUN_COMPARE}"
log_info "IDS_PATH=${IDS_PATH}"
log_info "REFERENCE_DETAILS=${REFERENCE_DETAILS}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  rm -rf "${RUN_ROOT:?}/${MODE}/result" "${RUN_ROOT:?}/${MODE}/score"
fi
mkdir -p "${RUN_ROOT}/${MODE}/result" "${RUN_ROOT}/${MODE}/score" "${RUN_ROOT}/${MODE}/logs"

check_port_free "${PORT}"
start_server
wait_health
run_eval
evaluate_mode
if [ "${RUN_COMPARE}" = "1" ]; then
  compare
else
  log_info "[compare:${MODE}] skipped; run a final merged compare after parallel jobs finish"
fi

log_info "History multi-step checkpoint run complete"
log_info "Report: ${RUN_ROOT}/report.md"
log_info "Logs: ${RUN_ROOT}/${MODE}/logs"
