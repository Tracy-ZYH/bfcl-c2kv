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
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4}"
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS_PATH="${REFERENCE_DETAILS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_kv_repair_stable52_$(date +%Y%m%d_%H%M%S)}"
PLAN_PATH="${PLAN_PATH:-}"
USE_REPAIR_PLAN="${USE_REPAIR_PLAN:-0}"
AUTO_BUILD_PLAN="${AUTO_BUILD_PLAN:-0}"
NEUTRAL_CORPUS_PATH="${NEUTRAL_CORPUS_PATH:-/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_neutral_corpus.txt}"
SHARE_PLAN_MODULE="${SHARE_PLAN_MODULE:-/home/zhuyuhan/project/c2kv/share/d-kv-repair/d_sham_plan.py}"

DEVICES="${DEVICES:-4,5,6}"
PORTS="${PORTS:-34200,34201,34202}"
ARMS="${ARMS:-full,c2kv,d_sham_mech,d_sham_neutral,d_corr,d_corr_recompute,d_corr_all}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

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

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
IFS=',' read -r -a PORT_LIST <<< "${PORTS}"
IFS=',' read -r -a ARM_LIST <<< "${ARMS}"

if [ "${#DEVICE_LIST[@]}" -ne "${#PORT_LIST[@]}" ]; then
  echo "DEVICES and PORTS must have the same length."
  exit 1
fi

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
    echo "Port ${port} is already in use."
    return 1
  fi
}

start_server() {
  local slot="$1"
  local device="${DEVICE_LIST[$slot]}"
  local port="${PORT_LIST[$slot]}"
  local log="${RUN_ROOT}/server_${device}_${port}.log"
  check_port_free "${port}"
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
      --mem-fraction-static "${MEM_FRACTION_STATIC:-0.55}" \
      --host 127.0.0.1 \
      --port "${port}"
  ) >"${log}" 2>&1 &
  SERVER_PIDS+=("$!")
  log_info "server slot=${slot} device=${device} port=${port} pid=${SERVER_PIDS[-1]} log=${log}"
}

wait_health() {
  local port="$1"
  local deadline=$((SECONDS + 1800))
  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  echo "Server on port ${port} did not become healthy."
  return 1
}

run_arm() {
  local arm="$1"
  local slot="$2"
  local port="${PORT_LIST[$slot]}"
  local arm_root="${RUN_ROOT}/${arm}"
  mkdir -p "${arm_root}/result" "${arm_root}/score" "${arm_root}/logs"
  log_info "runner start arm=${arm} slot=${slot} port=${port}"
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_kv_repair \
      --arm "${arm}" \
      --category "${CATEGORY}" \
      --max-examples "${MAX_EXAMPLES}" \
      --ids-path "${IDS_PATH}" \
      --reference-details-path "${REFERENCE_DETAILS_PATH}" \
      --plan-path "${PLAN_PATH}" \
      --base-url "http://127.0.0.1:${port}" \
      --model "${MODEL_ID}" \
      --served-model-name "${MODEL_ID}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --ratio "${RATIO}" \
      --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
      --repair-trigger oracle \
      --result-dir "${arm_root}/result" \
      --details-path "${arm_root}/logs/details.jsonl" \
      --metrics-path "${arm_root}/logs/metrics.jsonl" \
      --summary-path "${arm_root}/logs/summary.json" \
      --temperature 0
  ) >"${arm_root}/logs/runner.log" 2>&1
  log_info "runner done arm=${arm}"
}

evaluate_arm() {
  local arm="$1"
  local arm_root="${RUN_ROOT}/${arm}"
  local cmd=(
    "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner
    --model "${MODEL_ID}"
    --test-category "${CATEGORY}"
    --result-dir "${arm_root}/result"
    --score-dir "${arm_root}/score"
  )
  if [ "${MAX_EXAMPLES}" -lt 200 ] || [ -n "${IDS_PATH}" ]; then
    cmd+=(--partial-eval)
  fi
  (
    cd "${ROOT}"
    exec "${cmd[@]}"
  ) >"${arm_root}/logs/eval.log" 2>&1
  log_info "eval done arm=${arm}"
}

build_plan_if_needed() {
  if [[ "${USE_REPAIR_PLAN}" != "1" && "${USE_REPAIR_PLAN}" != "true" ]]; then
    return 0
  fi
  if [[ "${ARMS}" != *d_sham_neutral* && "${ARMS}" != *d_corr* ]]; then
    return 0
  fi
  if [[ -z "${PLAN_PATH}" ]]; then
    PLAN_PATH="${RUN_ROOT}/logs/d_sham_plan.json"
  fi
  if [[ "${AUTO_BUILD_PLAN}" != "1" && "${AUTO_BUILD_PLAN}" != "true" && -f "${PLAN_PATH}" ]]; then
    "${BFCL_PYTHON}" - "${PLAN_PATH}" "${RUN_ROOT}/logs/plan_build_summary.json" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
summary = {
    "mode": "existing_plan",
    "plan_path": str(plan_path),
    "plan_build_seconds": 0.0,
    "n_qids": plan.get("n_qids"),
    "typed_tokens_total": (plan.get("budget") or {}).get("typed_tokens_total"),
    "sham_tokens_total": (plan.get("budget") or {}).get("sham_tokens_total"),
    "budget_gate_passed": (plan.get("budget") or {}).get("gate_passed"),
    "neutrality_gate_passed": (plan.get("neutrality") or {}).get("gate_passed"),
    "doc_table_total_tokens": plan.get("doc_table_total_tokens"),
    "neutral_corpus_tokens": plan.get("neutral_corpus_tokens"),
    "plan_build_tokenization_tokens": plan.get("plan_build_tokenization_tokens"),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
    return 0
  fi

  log_info "build repair plan: ${PLAN_PATH}"
  local start
  local end
  start=$("${BFCL_PYTHON}" -c 'import time; print(time.perf_counter())')
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.scripts.build_bfcl_kv_repair_plan \
      --category "${CATEGORY}" \
      --ids-path "${IDS_PATH}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --share-plan-module "${SHARE_PLAN_MODULE}" \
      --neutral-corpus "${NEUTRAL_CORPUS_PATH}" \
      --out "${PLAN_PATH}"
  ) >"${RUN_ROOT}/logs/plan_build.log" 2>&1
  end=$("${BFCL_PYTHON}" -c 'import time; print(time.perf_counter())')
  "${BFCL_PYTHON}" - "${PLAN_PATH}" "${RUN_ROOT}/logs/plan_build_summary.json" "${start}" "${end}" <<'PY'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
start = float(sys.argv[3])
end = float(sys.argv[4])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
summary = {
    "mode": "built_in_sweep",
    "plan_path": str(plan_path),
    "plan_build_seconds": end - start,
    "n_qids": plan.get("n_qids"),
    "typed_tokens_total": (plan.get("budget") or {}).get("typed_tokens_total"),
    "sham_tokens_total": (plan.get("budget") or {}).get("sham_tokens_total"),
    "budget_gate_passed": (plan.get("budget") or {}).get("gate_passed"),
    "neutrality_gate_passed": (plan.get("neutrality") or {}).get("gate_passed"),
    "doc_table_total_tokens": plan.get("doc_table_total_tokens"),
    "neutral_corpus_tokens": plan.get("neutral_corpus_tokens"),
    "plan_build_tokenization_tokens": plan.get("plan_build_tokenization_tokens"),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
PY
}

log_info "BFCL KV repair sweep starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "ARMS=${ARMS}"
log_info "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
log_info "IDS_PATH=${IDS_PATH}"
log_info "PLAN_PATH=${PLAN_PATH}"

source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  rm -rf "${RUN_ROOT}"
fi
mkdir -p "${RUN_ROOT}/logs"

build_plan_if_needed

for slot in "${!DEVICE_LIST[@]}"; do
  start_server "${slot}"
done
for port in "${PORT_LIST[@]}"; do
  wait_health "${port}"
done

slot=0
for arm in "${ARM_LIST[@]}"; do
  run_arm "${arm}" "${slot}" &
  RUNNER_PIDS+=("$!")
  slot=$(( (slot + 1) % ${#DEVICE_LIST[@]} ))
  if [ "${#RUNNER_PIDS[@]}" -ge "${#DEVICE_LIST[@]}" ]; then
    wait "${RUNNER_PIDS[0]}"
    RUNNER_PIDS=("${RUNNER_PIDS[@]:1}")
  fi
done
for pid in "${RUNNER_PIDS[@]}"; do
  wait "${pid}"
done
RUNNER_PIDS=()

for arm in "${ARM_LIST[@]}"; do
  evaluate_arm "${arm}"
done

(
  cd "${ROOT}"
  "${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_kv_repair_sweep \
    --run-root "${RUN_ROOT}" \
    --arms "${ARMS}"
)

log_info "summary: ${RUN_ROOT}/kv_repair_summary.csv"
log_info "report: ${RUN_ROOT}/kv_repair_report.md"
