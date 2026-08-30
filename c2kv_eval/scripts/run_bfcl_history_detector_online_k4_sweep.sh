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
COMPRESSION_RATIO="${COMPRESSION_RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
ROLLBACK_DEPTH="${ROLLBACK_DEPTH:-4}"
ROLLBACK_POLICY="${ROLLBACK_POLICY:-fixed_depth}"
ROLLBACK_BACKEND="${ROLLBACK_BACKEND:-kv_restore_strict}"
RECOVERY_MODE="${RECOVERY_MODE:-first_bad_suffix}"
RECOVERY_HORIZON="${RECOVERY_HORIZON:-suffix}"
RULE_DETECTOR_THRESHOLD="${RULE_DETECTOR_THRESHOLD:-5}"
LOGISTIC_DETECTOR_THRESHOLD="${LOGISTIC_DETECTOR_THRESHOLD:--1}"
DEVICE="${DEVICE:-7}"
PORT="${PORT:-33707}"
ENABLE_HICACHE="${ENABLE_HICACHE:-auto}"
HICACHE_RATIO="${HICACHE_RATIO:-2}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
CANDIDATE_LOGPROBS_TOP_K="${CANDIDATE_LOGPROBS_TOP_K:-20}"

IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"
DETECTOR_BENCHMARK_ROOT="${DETECTOR_BENCHMARK_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_detector_signal_benchmark_20260825_134109/detector_benchmark}"
LOGISTIC_DETECTOR_FEATURES_CSV="${LOGISTIC_DETECTOR_FEATURES_CSV:-${DETECTOR_BENCHMARK_ROOT}/detector_features.csv}"
DETECTOR_COMPARISON_CSV="${DETECTOR_COMPARISON_CSV:-${DETECTOR_BENCHMARK_ROOT}/detector_comparison.csv}"

STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_detector_online_k4_d${ROLLBACK_DEPTH}_stable52_${STAMP}}"
DETECTORS="${DETECTORS:-logistic,rule,max_risk_score,rule_detector_max_risk,max_observation_anomaly,mean_risk_score,max_hard_error,max_generation_nll,mean_generation_nll}"
RUN_COMPARE="${RUN_COMPARE:-1}"

SERVER_PID=""
MODES=()

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

threshold_for_signal() {
  local signal="$1"
  "${BFCL_PYTHON}" - "${DETECTOR_COMPARISON_CSV}" "${signal}" <<'PY'
import csv
import sys

path, signal = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("signal") == signal:
            value = row.get("best_threshold")
            if value not in (None, ""):
                print(value)
                raise SystemExit(0)
raise SystemExit(f"missing best_threshold for {signal} in {path}")
PY
}

start_server() {
  local log="${RUN_ROOT}/server_${DEVICE}_${PORT}.log"
  local hicache_args=()
  if [[ "${ENABLE_HICACHE}" == "1" || "${ENABLE_HICACHE}" == "true" || ( "${ENABLE_HICACHE}" == "auto" && ( "${ROLLBACK_BACKEND}" == "kv_restore" || "${ROLLBACK_BACKEND}" == "kv_restore_strict" ) ) ]]; then
    hicache_args+=(--enable-hierarchical-cache --hicache-ratio "${HICACHE_RATIO}")
  fi
  mkdir -p "${RUN_ROOT}"
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
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --enable-cache-report \
      "${hicache_args[@]}" \
      --host 127.0.0.1 \
      --port "${PORT}"
  ) >"${log}" 2>&1 &
  SERVER_PID="$!"
  log_info "[server] device=${DEVICE} port=${PORT} pid=${SERVER_PID} log=${log}"
}

wait_health() {
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      log_info "[server] crashed; last log:"
      tail -n 120 "${RUN_ROOT}/server_${DEVICE}_${PORT}.log" || true
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
  tail -n 120 "${RUN_ROOT}/server_${DEVICE}_${PORT}.log" || true
  return 1
}

run_detector() {
  local detector="$1"
  local verifier
  local mode
  local signal_name=""
  local signal_threshold="0"
  local extra_args=()

  case "${detector}" in
    logistic)
      verifier="logistic"
      mode="online_k${CHECKPOINT_INTERVAL}_d${ROLLBACK_DEPTH}_logistic"
      extra_args+=(--logistic-detector-features-csv "${LOGISTIC_DETECTOR_FEATURES_CSV}")
      extra_args+=(--logistic-detector-threshold "${LOGISTIC_DETECTOR_THRESHOLD}")
      ;;
    rule)
      verifier="rule"
      mode="online_k${CHECKPOINT_INTERVAL}_d${ROLLBACK_DEPTH}_rule"
      ;;
    *)
      verifier="feature_signal"
      signal_name="${detector}"
      signal_threshold="$(threshold_for_signal "${signal_name}")"
      mode="online_k${CHECKPOINT_INTERVAL}_d${ROLLBACK_DEPTH}_signal_${signal_name}"
      extra_args+=(--detector-signal-name "${signal_name}")
      extra_args+=(--detector-signal-threshold "${signal_threshold}")
      ;;
  esac

  local mode_root="${RUN_ROOT}/${mode}"
  mkdir -p "${mode_root}/result" "${mode_root}/score" "${mode_root}/logs"
  log_info "[runner] start detector=${detector} verifier=${verifier} mode=${mode} threshold=${signal_threshold}"
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.adapters.eval_bfcl_history_checkpoint \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${IDS_PATH}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${PORT}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${REFERENCE_DETAILS}" \
      --result-dir "${mode_root}/result" \
      --details-path "${mode_root}/logs/details.jsonl" \
      --metrics-path "${mode_root}/logs/checkpoint_metrics.jsonl" \
      --step-metrics-path "${mode_root}/logs/checkpoint_steps.jsonl" \
      --segment-metrics-path "${mode_root}/logs/checkpoint_segments.jsonl" \
      --summary-path "${mode_root}/logs/run_summary.json" \
      --compression-ratio "${COMPRESSION_RATIO}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --verifier "${verifier}" \
      --rule-detector-threshold "${RULE_DETECTOR_THRESHOLD}" \
      --recovery-mode "${RECOVERY_MODE}" \
      --recovery-horizon "${RECOVERY_HORIZON}" \
      --rollback-policy "${ROLLBACK_POLICY}" \
      --rollback-depth "${ROLLBACK_DEPTH}" \
      --rollback-backend "${ROLLBACK_BACKEND}" \
      --candidate-logprobs-top-k "${CANDIDATE_LOGPROBS_TOP_K}" \
      --collect-candidate-detector-signals \
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
      --temperature 0 \
      "${extra_args[@]}"
  ) >"${mode_root}/logs/run.log" 2>&1

  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner \
      --model "${MODEL_ID}" \
      --test-category "${CATEGORY}" \
      --result-dir "${mode_root}/result" \
      --score-dir "${mode_root}/score" \
      --partial-eval
  ) >"${mode_root}/logs/eval.log" 2>&1
  log_info "[runner] done detector=${detector} mode=${mode}"
  MODES+=("${mode}")
}

merge_report() {
  local mode_csv
  mode_csv="$(IFS=,; echo "${MODES[*]}")"
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" c2kv_eval/analysis/compare_history_checkpoint.py \
      --run-root "${RUN_ROOT}" \
      --category "${CATEGORY}" \
      --model "${MODEL_ID}" \
      --reference-details-path "${REFERENCE_DETAILS}" \
      --modes "${mode_csv}"
  ) >"${RUN_ROOT}/compare.log" 2>&1
  log_info "[compare] done: ${RUN_ROOT}/report.md"
}

log_info "BFCL online detector K=${CHECKPOINT_INTERVAL} D=${ROLLBACK_DEPTH} sweep"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "DEVICE=${DEVICE} PORT=${PORT}"
log_info "DETECTORS=${DETECTORS}"
log_info "LOGISTIC_DETECTOR_FEATURES_CSV=${LOGISTIC_DETECTOR_FEATURES_CSV}"
log_info "DETECTOR_COMPARISON_CSV=${DETECTOR_COMPARISON_CSV}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

check_port_free "${PORT}"
start_server
wait_health

IFS=',' read -r -a DETECTOR_LIST <<< "${DETECTORS}"
for detector in "${DETECTOR_LIST[@]}"; do
  run_detector "${detector}"
done

if [ "${RUN_COMPARE}" = "1" ]; then
  merge_report
fi

log_info "Done. RUN_ROOT=${RUN_ROOT}"
