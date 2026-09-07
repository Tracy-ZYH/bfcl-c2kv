#!/usr/bin/env bash
set -Ee -o pipefail
set +u

ROOT="${ROOT:-/home/zhuyuhan/project/bfcl-c2kv}"
SGLANG_ROOT="${SGLANG_ROOT:-/home/zhuyuhan/project/kvoffload-sglang-c2kv}"
CLIENT_OVERLAY="${CLIENT_OVERLAY:-/home/zhuyuhan/zh_restore/zh_client_overlay}"
BFCL_PYTHON="${BFCL_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-/home/liuyancheng/envs/sgl/bin/python}"
export PYTHONPATH="${ROOT}:${SGLANG_ROOT}/python:${CLIENT_OVERLAY}:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/home/zhuyuhan/project/model/checkpoints/qwen3-4b-agent-history-c2kv-toolcall-npu-v2/checkpoint-1088}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/zhuyuhan/project/model/models/Qwen3-4B-Instruct-2507}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507-FC}"

CATEGORY="${CATEGORY:-multi_turn_base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-52}"
RATIO="${RATIO:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
TEMPERATURE="${TEMPERATURE:-0}"
REPAIR_ARM="${REPAIR_ARM:-d_corr_replace_w2}"
REPAIR_WINDOW="${REPAIR_WINDOW:-2}"
REPAIR_EXTRACT_SOURCE="${REPAIR_EXTRACT_SOURCE:-auto}"
FOLDS="${FOLDS:-5}"
SEED="${SEED:-20260905}"
ORACLE_TARGET_FRACTION="${ORACLE_TARGET_FRACTION:-0.90}"
DETECTOR_VERSION="${DETECTOR_VERSION:-2}"
FAST_DEV="${FAST_DEV:-0}"

IDS_PATH="${IDS_PATH:-/home/zhuyuhan/recovery_zhuyuhan/toolsnorm52/c2kv-toolsnorm52.NIhbBb/inputs/correct_ids.txt}"
REFERENCE_DETAILS="${REFERENCE_DETAILS:-/home/zhuyuhan/recovery_zhuyuhan/toolsnorm52/c2kv-toolsnorm52.NIhbBb/inputs/details.jsonl}"
FEATURES_CSV="${FEATURES_CSV:-}"

DEVICES="${DEVICES:-4,7}"
PORTS="${PORTS:-35704,35707}"
STAMP="${STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/combined_logistic_v2_replace_w2_stable52_${STAMP}}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.10}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"
CANDIDATE_LOGPROBS_TOP_K="${CANDIDATE_LOGPROBS_TOP_K:-20}"
REQUEST_CANDIDATE_LOGPROBS="${REQUEST_CANDIDATE_LOGPROBS:-0}"
RULE_DETECTOR_THRESHOLD="${RULE_DETECTOR_THRESHOLD:-5}"
SHADOW_NO_TRIGGER_THRESHOLD="${SHADOW_NO_TRIGGER_THRESHOLD:-1.000001}"

SERVER_PIDS=()

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

start_server() {
  local device="$1"
  local port="$2"
  local log="${RUN_ROOT}/server_${device}_${port}.log"
  mkdir -p "${RUN_ROOT}"
  (
    cd "${SGLANG_ROOT}"
    PYTHONPATH="${SGLANG_ROOT}/python:${ROOT}:${PYTHONPATH:-}" \
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
      --tokenizer-path "${TOKENIZER_PATH}" \
      --served-model-name "${MODEL_ID}" \
      --model-impl sglang \
      --device npu \
      --attention-backend ascend \
      --tool-call-parser qwen25 \
      --enable-c2kv \
      --c2kv-pool-fraction "${C2KV_POOL_FRACTION}" \
      --dtype bfloat16 \
      --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      --host 127.0.0.1 \
      --port "${port}" \
      ${SGLANG_EXTRA_ARGS:-}
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
    sleep 2
  done
  tail -n 120 "${log}" || true
  return 1
}

run_eval_case() {
  local detector_label="$1"
  local detector_arm="$2"
  local fold="$3"
  local ids_path="$4"
  local mode_root="$5"
  local port="$6"
  local threshold="$7"
  local model_json="$8"
  local extra_args=()

  mkdir -p "${mode_root}/result" "${mode_root}/score" "${mode_root}/logs"
  if [ ! -s "${ids_path}" ]; then
    log_info "[runner] skip detector=${detector_label} fold=${fold}: empty ids"
    echo '{"num_examples":0,"skipped_empty_ids":true}' > "${mode_root}/logs/summary.json"
    return 0
  fi

  if [ "${detector_arm}" = "combined_logistic_v2" ] || [ "${detector_arm}" = "combined_logistic_v3" ]; then
    extra_args+=(--logistic-v2-model-json "${model_json}")
    if [ "${threshold}" != "" ]; then
      extra_args+=(--logistic-detector-threshold "${threshold}")
    fi
  fi
  if [ "${REQUEST_CANDIDATE_LOGPROBS}" = "1" ]; then
    extra_args+=(--request-candidate-logprobs)
  fi

  log_info "[runner] start detector=${detector_label} arm=${detector_arm} fold=${fold} threshold=${threshold} ids=${ids_path} port=${port}"
  set +e
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_kv_repair \
      --arm "${REPAIR_ARM}" \
      --detector-arm "${detector_arm}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${ids_path}" \
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
      --repair-extract-source "${REPAIR_EXTRACT_SOURCE}" \
      --repair-trigger oracle \
      --rule-detector-threshold "${RULE_DETECTOR_THRESHOLD}" \
      --candidate-logprobs-top-k "${CANDIDATE_LOGPROBS_TOP_K}" \
      --collect-candidate-detector-signals \
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      "${extra_args[@]}"
  ) >"${mode_root}/logs/run.log" 2>&1
  local adapter_rc=$?
  set -e
  if [ "${adapter_rc}" -ne 0 ]; then
    log_info "[runner] FAILED adapter detector=${detector_label} fold=${fold} threshold=${threshold} rc=${adapter_rc} log=${mode_root}/logs/run.log"
    return "${adapter_rc}"
  fi

  set +e
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner \
      --model "${MODEL_ID}" \
      --test-category "${CATEGORY}" \
      --result-dir "${mode_root}/result" \
      --score-dir "${mode_root}/score" \
      --partial-eval
  ) >"${mode_root}/logs/eval.log" 2>&1
  local eval_rc=$?
  set -e
  if [ "${eval_rc}" -ne 0 ]; then
    log_info "[runner] FAILED evaluator detector=${detector_label} fold=${fold} threshold=${threshold} rc=${eval_rc} log=${mode_root}/logs/eval.log"
    return "${eval_rc}"
  fi
  log_info "[runner] done detector=${detector_label} fold=${fold} threshold=${threshold}"
}

run_job_list() {
  local -n jobs_ref="$1"
  local worker_idx
  local pids=()

  run_worker() {
    local local_worker_idx="$1"
    local port="${PORT_LIST[$local_worker_idx]}"
    local idx
    for idx in "${!jobs_ref[@]}"; do
      if [ $((idx % ${#PORT_LIST[@]})) -ne "${local_worker_idx}" ]; then
        continue
      fi
      IFS='|' read -r detector_label detector_arm fold ids_path mode_root threshold model_json <<< "${jobs_ref[$idx]}"
      if ! run_eval_case "${detector_label}" "${detector_arm}" "${fold}" "${ids_path}" "${mode_root}" "${port}" "${threshold}" "${model_json}"; then
        log_info "[worker ${local_worker_idx}] FAILED job=${jobs_ref[$idx]}"
        return 1
      fi
    done
  }

  for worker_idx in "${!PORT_LIST[@]}"; do
    run_worker "${worker_idx}" &
    pids+=("$!")
  done
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [ "${status}" -ne 0 ]; then
    log_info "At least one job failed."
    exit "${status}"
  fi
}

threshold_label() {
  local threshold="$1"
  "${BFCL_PYTHON}" - "${threshold}" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "_"))
PY
}

selected_v2_threshold_for_fold() {
  local fold="$1"
  "${BFCL_PYTHON}" - "${RUN_ROOT}/detector_cv/combined_logistic_v2_selected_thresholds.json" "${fold}" <<'PY'
import json
import sys
path, fold = sys.argv[1], sys.argv[2]
payload = json.load(open(path, encoding="utf-8"))
print(payload["fold_thresholds"][fold]["combined_logistic_v2"])
PY
}

selected_v3_threshold_for_fold() {
  local fold="$1"
  "${BFCL_PYTHON}" - "${RUN_ROOT}/detector_cv/fold_${fold}/combined_logistic_v3_selected_model.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["threshold"])
PY
}

prepare_feature_csv() {
  if [ "${FEATURES_CSV}" != "" ] && [ -s "${FEATURES_CSV}" ]; then
    log_info "Using existing FEATURES_CSV=${FEATURES_CSV}"
    echo "${FEATURES_CSV}" > "${RUN_ROOT}/detector_cv/features_csv_path.txt"
    return 0
  fi

  local feature_root="${RUN_ROOT}/feature_source_never_trigger"
  local segments_path="${RUN_ROOT}/detector_cv/feature_source_segments.jsonl"
  local benchmark_root="${RUN_ROOT}/detector_cv/feature_benchmark"
  run_eval_case "feature_source_never_trigger" "never_trigger" "all" "${IDS_PATH}" "${feature_root}" "${PORT_LIST[0]}" "" ""
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.extract_repair_detector_segments \
      --details-path "${feature_root}/logs/details.jsonl" \
      --output-path "${segments_path}"
  ) >"${RUN_ROOT}/detector_cv/extract_feature_segments.log" 2>&1
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.benchmark_detector_signals \
      --segments-path "${segments_path}" \
      --output-dir "${benchmark_root}" \
      --rule-detector-threshold "${RULE_DETECTOR_THRESHOLD}"
  ) >"${RUN_ROOT}/detector_cv/benchmark_feature_segments.log" 2>&1
  FEATURES_CSV="${benchmark_root}/detector_features.csv"
  if [ ! -s "${FEATURES_CSV}" ]; then
    echo "Failed to create detector feature CSV: ${FEATURES_CSV}"
    exit 1
  fi
  echo "${FEATURES_CSV}" > "${RUN_ROOT}/detector_cv/features_csv_path.txt"
  log_info "Generated FEATURES_CSV=${FEATURES_CSV}"
}

train_v2_models() {
  local trainer="c2kv_eval.analysis.train_combined_logistic_v2"
  if [ "${DETECTOR_VERSION}" = "3" ]; then
    trainer="c2kv_eval.analysis.train_combined_logistic_v3"
  fi
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m "${trainer}" \
      --ids-path "${IDS_PATH}" \
      --features-csv "${FEATURES_CSV}" \
      --output-dir "${RUN_ROOT}/detector_cv" \
      --folds "${FOLDS}" \
      --max-examples "${MAX_EXAMPLES}" \
      --seed "${SEED}"
  ) >"${RUN_ROOT}/detector_cv/train_combined_logistic_v2.log" 2>&1
}

select_v2_thresholds() {
  local selector="c2kv_eval.analysis.select_combined_logistic_v2_thresholds"
  if [ "${DETECTOR_VERSION}" = "3" ]; then
    selector="c2kv_eval.analysis.select_combined_logistic_v3_thresholds"
  fi
  if [ "${DETECTOR_VERSION}" = "3" ]; then
    ( cd "${ROOT}"; "${BFCL_PYTHON}" -m "${selector}" \
        --run-root "${RUN_ROOT}" --cv-dir "${RUN_ROOT}/detector_cv" \
        --folds "${FOLDS}" --target-fraction "${ORACLE_TARGET_FRACTION}" \
    ) >"${RUN_ROOT}/detector_cv/select_combined_logistic_v3_thresholds.log" 2>&1
  else
    ( cd "${ROOT}"; "${BFCL_PYTHON}" -m "${selector}" \
        --run-root "${RUN_ROOT}" --cv-dir "${RUN_ROOT}/detector_cv" \
        --folds "${FOLDS}" --target-fraction "${ORACLE_TARGET_FRACTION}" \
        --output-json "${RUN_ROOT}/detector_cv/combined_logistic_v2_selected_thresholds.json" \
    ) >"${RUN_ROOT}/detector_cv/select_combined_logistic_v2_thresholds.log" 2>&1
  fi
}

merge_report() {
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_combined_logistic_v2_online \
      --run-root "${RUN_ROOT}" \
      --cv-dir "${RUN_ROOT}/detector_cv" \
      --folds "${FOLDS}"
  ) >"${RUN_ROOT}/compare_combined_logistic_v2_online.log" 2>&1
}

if [ "${CLEAN_OUTPUT}" = "1" ] && [ -d "${RUN_ROOT}" ]; then
  rm -rf "${RUN_ROOT}"
fi
mkdir -p "${RUN_ROOT}/detector_cv"

log_info "BFCL combined_logistic_v2 Replace-W2 nested 5-fold"
log_info "DETECTOR_VERSION=${DETECTOR_VERSION}"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
log_info "ORACLE_TARGET_FRACTION=${ORACLE_TARGET_FRACTION}"
log_info "REPAIR_ARM=${REPAIR_ARM} REPAIR_WINDOW=${REPAIR_WINDOW}"
log_info "REFERENCE_DETAILS=${REFERENCE_DETAILS}"
log_info "SHADOW_NO_TRIGGER_THRESHOLD=${SHADOW_NO_TRIGGER_THRESHOLD}"

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

prepare_feature_csv
train_v2_models

TRAIN_JOBS=()
if [ "${DETECTOR_VERSION}" = "3" ] && [ "${ONLINE_SHADOW:-1}" = "1" ]; then
  for fold in $(seq 0 $((FOLDS - 1))); do
    shadow_ids="${RUN_ROOT}/detector_cv/fold_${fold}/calibration_ids.txt"
    shadow_model="${RUN_ROOT}/detector_cv/fold_${fold}/combined_logistic_v3_model.json"
    TRAIN_JOBS+=("combined_logistic_v3_online_shadow|combined_logistic_v3|${fold}|${shadow_ids}|${RUN_ROOT}/logistic_v3_online_shadow/fold_${fold}|${SHADOW_NO_TRIGGER_THRESHOLD}|${shadow_model}")
  done
  run_job_list TRAIN_JOBS
  for fold in $(seq 0 $((FOLDS - 1))); do
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.prepare_combined_logistic_v3_online_thresholds \
      --shadow-root "${RUN_ROOT}/logistic_v3_online_shadow/fold_${fold}" \
      --fold-dir "${RUN_ROOT}/detector_cv/fold_${fold}" \
      --target-rates "0.40,0.45,0.50,0.55,0.60"
  done
fi
TRAIN_JOBS=()
for fold in $(seq 0 $((FOLDS - 1))); do
  train_ids="${RUN_ROOT}/detector_cv/fold_${fold}/outer_train_ids.txt"
  model_json="${RUN_ROOT}/detector_cv/fold_${fold}/combined_logistic_v2_model.json"
  sweep_root="${RUN_ROOT}/logistic_v2_train_sweep/fold_${fold}"
  if [ "${DETECTOR_VERSION}" = "3" ]; then
    train_ids="${RUN_ROOT}/detector_cv/fold_${fold}/calibration_ids.txt"
    model_json="${RUN_ROOT}/detector_cv/fold_${fold}/combined_logistic_v3_model.json"
    sweep_root="${RUN_ROOT}/logistic_v3_calibration/fold_${fold}"
  fi
  TRAIN_JOBS+=("combined_logistic_v2_train_oracle|oracle|${fold}|${train_ids}|${sweep_root}/oracle||")
  IFS=',' read -r -a FOLD_THRESHOLDS < "${RUN_ROOT}/detector_cv/fold_${fold}/threshold_candidates.txt"
  for threshold in "${FOLD_THRESHOLDS[@]}"; do
    label="$(threshold_label "${threshold}")"
    TRAIN_JOBS+=("combined_logistic_v3_calibration_${label}|combined_logistic_v3|${fold}|${train_ids}|${sweep_root}/threshold_${label}|${threshold}|${model_json}")
  done
done
run_job_list TRAIN_JOBS

if [ "${DETECTOR_VERSION}" = "3" ]; then
  select_v2_thresholds
  FINAL_JOBS=()
  for fold in $(seq 0 $((FOLDS - 1))); do
    test_ids="${RUN_ROOT}/detector_cv/fold_${fold}/test_ids.txt"
    selected_model="${RUN_ROOT}/detector_cv/fold_${fold}/combined_logistic_v3_selected_model.json"
    threshold="$(selected_v3_threshold_for_fold "${fold}")"
    FINAL_JOBS+=("combined_logistic_v3|combined_logistic_v3|${fold}|${test_ids}|${RUN_ROOT}/detector_combined_logistic_v3/fold_${fold}|${threshold}|${selected_model}")
  done
  run_job_list FINAL_JOBS
  select_v2_thresholds
  log_info "V3 held-out evaluation complete"
  exit 0
fi

select_v2_thresholds

FINAL_JOBS=()
for fold in $(seq 0 $((FOLDS - 1))); do
  test_ids="${RUN_ROOT}/detector_cv/fold_${fold}/test_ids.txt"
  FINAL_JOBS+=("oracle|oracle|${fold}|${test_ids}|${RUN_ROOT}/detector_oracle/fold_${fold}||")
  threshold="$(selected_v2_threshold_for_fold "${fold}")"
  selected_model="${RUN_ROOT}/detector_cv/fold_${fold}/combined_logistic_v2_selected_model.json"
  FINAL_JOBS+=("combined_logistic_v2|combined_logistic_v2|${fold}|${test_ids}|${RUN_ROOT}/detector_combined_logistic_v2/fold_${fold}|${threshold}|${selected_model}")
done
run_job_list FINAL_JOBS

merge_report

log_info "Done. Summary: ${RUN_ROOT}/combined_logistic_v2_summary.csv"
log_info "Done. Fold diagnostics: ${RUN_ROOT}/combined_logistic_v2_fold_diagnostics.csv"
log_info "Done. Threshold sweep: ${RUN_ROOT}/combined_logistic_v2_threshold_sweep.csv"
