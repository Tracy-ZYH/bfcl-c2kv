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
MAX_EXAMPLES="${MAX_EXAMPLES:-52}"
RATIO="${RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
REPAIR_ARM="${REPAIR_ARM:-d_corr_replace_w2}"
REPAIR_WINDOW="${REPAIR_WINDOW:-2}"
RULE_DETECTOR_THRESHOLD="${RULE_DETECTOR_THRESHOLD:-5}"
CANDIDATE_LOGPROBS_TOP_K="${CANDIDATE_LOGPROBS_TOP_K:-20}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.10}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"

IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"
DETECTOR_BENCHMARK_ROOT="${DETECTOR_BENCHMARK_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_detector_signal_benchmark_20260825_134109/detector_benchmark}"
LOGISTIC_DETECTOR_FEATURES_CSV="${LOGISTIC_DETECTOR_FEATURES_CSV:-${DETECTOR_BENCHMARK_ROOT}/detector_features.csv}"
DETECTOR_COMPARISON_CSV="${DETECTOR_COMPARISON_CSV:-${DETECTOR_BENCHMARK_ROOT}/detector_comparison.csv}"

DETECTORS="${DETECTORS:-never_trigger,oracle,combined_logistic_best_f1,combined_logistic_high_recall,max_risk_score,rule_trigger,always_trigger}"
DEVICES="${DEVICES:-4,7}"
PORTS="${PORTS:-34704,34707}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/detector_closed_loop_replace_w2_stable52_${STAMP}}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

SERVER_PIDS=()
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
  for pid in "${SERVER_PIDS[@]}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
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
    echo "Port ${port} is already in use. Stop the existing server or change PORTS."
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
  local device="$1"
  local port="$2"
  local log="${RUN_ROOT}/server_${device}_${port}.log"
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
    ASCEND_RT_VISIBLE_DEVICES="${device}" \
    exec "${SGLANG_PYTHON}" -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --served-model-name "${MODEL_ID}" \
      --model-impl sglang \
      --device npu \
      --attention-backend ascend \
      --tool-call-parser qwen25 \
      --enable-c2kv \
      --c2kv-pool-fraction "${C2KV_POOL_FRACTION}" \
      --dtype bfloat16 \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --enable-cache-report \
      --host 127.0.0.1 \
      --port "${port}"
  ) >"${log}" 2>&1 &
  local pid="$!"
  SERVER_PIDS+=("${pid}")
  log_info "[server] device=${device} port=${port} pid=${pid} log=${log}"
}

wait_health() {
  local port="$1"
  local pid="$2"
  local log="$3"
  local attempt
  for attempt in $(seq 1 900); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      log_info "[server] crashed on port ${port}; last log:"
      tail -n 120 "${log}" || true
      return 1
    fi
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      log_info "[server] healthy on port ${port}"
      return 0
    fi
    if [ $((attempt % 15)) -eq 0 ]; then
      log_info "[server] waiting for port ${port} (${attempt}/900)"
    fi
    sleep 2
  done
  tail -n 120 "${log}" || true
  return 1
}

run_detector() {
  local detector="$1"
  local port="$2"
  local mode="detector_${detector}"
  local mode_root="${RUN_ROOT}/${mode}"
  local signal_threshold="${DETECTOR_SIGNAL_THRESHOLD:-5}"
  local extra_args=()

  if [ "${detector}" = "max_risk_score" ]; then
    signal_threshold="$(threshold_for_signal max_risk_score)"
  fi
  if [[ "${detector}" == combined_logistic* ]]; then
    extra_args+=(--logistic-detector-features-csv "${LOGISTIC_DETECTOR_FEATURES_CSV}")
  fi

  mkdir -p "${mode_root}/result" "${mode_root}/score" "${mode_root}/logs"
  log_info "[runner] start detector=${detector} port=${port} mode=${mode}"
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_kv_repair \
      --arm "${REPAIR_ARM}" \
      --detector-arm "${detector}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${IDS_PATH}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --base-url "http://127.0.0.1:${port}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --reference-details-path "${REFERENCE_DETAILS}" \
      --result-dir "${mode_root}/result" \
      --details-path "${mode_root}/logs/details.jsonl" \
      --metrics-path "${mode_root}/logs/metrics.jsonl" \
      --summary-path "${mode_root}/logs/summary.json" \
      --ratio "${RATIO}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --repair-window "${REPAIR_WINDOW}" \
      --repair-extract-source "${REPAIR_EXTRACT_SOURCE:-auto}" \
      --repair-trigger oracle \
      --rule-detector-threshold "${RULE_DETECTOR_THRESHOLD}" \
      --detector-signal-threshold "${signal_threshold}" \
      --candidate-logprobs-top-k "${CANDIDATE_LOGPROBS_TOP_K}" \
      --collect-candidate-detector-signals \
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
      --temperature "${TEMPERATURE}" \
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
  log_info "[runner] done detector=${detector}"
}

merge_report() {
  local mode_csv
  mode_csv="$(IFS=,; echo "${MODES[*]}")"
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_detector_closed_loop \
      --run-root "${RUN_ROOT}" \
      --modes "${mode_csv}"
  ) >"${RUN_ROOT}/compare_detector_closed_loop.log" 2>&1
  log_info "[compare] wrote ${RUN_ROOT}/detector_closed_loop_summary.csv"
}

if [ "${CLEAN_OUTPUT}" = "1" ] && [ -d "${RUN_ROOT}" ]; then
  rm -rf "${RUN_ROOT}"
fi
mkdir -p "${RUN_ROOT}"

log_info "BFCL detector closed-loop Replace-W2 sweep"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "DETECTORS=${DETECTORS}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
log_info "REPAIR_ARM=${REPAIR_ARM} K=${CHECKPOINT_INTERVAL} ratio=${RATIO}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
IFS=',' read -r -a PORT_LIST <<< "${PORTS}"
if [ "${#DEVICE_LIST[@]}" -ne "${#PORT_LIST[@]}" ]; then
  echo "DEVICES and PORTS must have the same length"
  exit 1
fi
for port in "${PORT_LIST[@]}"; do
  check_port_free "${port}"
done
for idx in "${!DEVICE_LIST[@]}"; do
  start_server "${DEVICE_LIST[$idx]}" "${PORT_LIST[$idx]}"
done
for idx in "${!PORT_LIST[@]}"; do
  wait_health "${PORT_LIST[$idx]}" "${SERVER_PIDS[$idx]}" "${RUN_ROOT}/server_${DEVICE_LIST[$idx]}_${PORT_LIST[$idx]}.log"
done

IFS=',' read -r -a DETECTOR_LIST <<< "${DETECTORS}"
PIDS=()
for detector in "${DETECTOR_LIST[@]}"; do
  MODES+=("detector_${detector}")
done

run_worker() {
  local worker_idx="$1"
  local port="${PORT_LIST[$worker_idx]}"
  local idx
  for idx in "${!DETECTOR_LIST[@]}"; do
    if [ $((idx % ${#PORT_LIST[@]})) -ne "${worker_idx}" ]; then
      continue
    fi
    run_detector "${DETECTOR_LIST[$idx]}" "${port}"
  done
}

for worker_idx in "${!PORT_LIST[@]}"; do
  run_worker "${worker_idx}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  log_info "At least one detector run failed."
  exit "${status}"
fi

merge_report
log_info "Done. RUN_ROOT=${RUN_ROOT}"
