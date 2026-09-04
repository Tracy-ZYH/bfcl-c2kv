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
IDS_PATH="${IDS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/correct_ids.txt}"
REFERENCE_DETAILS_PATH="${REFERENCE_DETAILS_PATH:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_full_temp0_stability_20260819_172725/frozen_reference/details.jsonl}"

RATIO="${RATIO:-4}"
HISTORY_KV_RETENTION_RATIO="${HISTORY_KV_RETENTION_RATIO:-0.312}"
HISTORY_KV_TARGET_COMPRESSION="${HISTORY_KV_TARGET_COMPRESSION:-0}"
HISTORY_KV_RECENT_WINDOW="${HISTORY_KV_RECENT_WINDOW:-64}"
HISTORY_KV_KERNEL_SIZE="${HISTORY_KV_KERNEL_SIZE:-5}"
HISTORY_KV_POOLING="${HISTORY_KV_POOLING:-avgpool}"
HISTORY_KV_H2O_RECENT_FRACTION="${HISTORY_KV_H2O_RECENT_FRACTION:-0.5}"
RUNTIME_HISTORY_KV_BACKEND="${RUNTIME_HISTORY_KV_BACKEND:-repair_extract}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-4096}"

METHODS="${METHODS:-full,c2kv,streamingllm,h2o,snapkv_persistent,pyramidkv}"
DEVICES="${DEVICES:-4,5,6,7}"
PORTS="${PORTS:-34740,34750,34760,34770}"
RUN_ROOT="${RUN_ROOT:-/home/zhuyuhan/project/gorilla/bfcl_runs/history_kv_baselines_stable52_$(date +%Y%m%d_%H%M%S)}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
ASSERT_IDENTITY="${ASSERT_IDENTITY:-0}"
STRICT_RUNTIME_EVICTION="${STRICT_RUNTIME_EVICTION:-1}"
ALLOW_CLIENT_FALLBACK="${ALLOW_CLIENT_FALLBACK:-0}"
# Leave this unset to use the correct default for each backend: physical
# eviction is meaningful only when the compacted KV survives across turns.
# Set explicitly to 0 only for the legacy per-request diagnostic path.
PERSISTENT_HISTORY_KV_SESSION="${PERSISTENT_HISTORY_KV_SESSION:-}"

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.55}"
C2KV_POOL_FRACTION="${C2KV_POOL_FRACTION:-0.06}"
PAGE_SIZE="${PAGE_SIZE:-}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"

if [ "${RUNTIME_HISTORY_KV_BACKEND}" = "physical_eviction" ]; then
  # Physical compaction overwrites request-owned slots. Do not permit a prefix
  # cache node to share those slots with another request.
  DISABLE_RADIX_CACHE=1
  # Ascend fused paged attention requires a no-quant block size aligned to 16.
  # The physical evictor is page-aware, so it does not require page_size=1.
  if [ -z "${PAGE_SIZE}" ]; then
    # Ascend fused attention requires a block-aligned page.  Use the same
    # 128-token page size as the established BFCL/SGLang serving runs; the
    # physical evictor is page-aware and does not require page_size=1.
    PAGE_SIZE=128
  fi
  if [ -z "${PERSISTENT_HISTORY_KV_SESSION}" ]; then
    PERSISTENT_HISTORY_KV_SESSION=1
  fi
elif [ -z "${PERSISTENT_HISTORY_KV_SESSION}" ]; then
  PERSISTENT_HISTORY_KV_SESSION=0
fi

if [ "${PERSISTENT_HISTORY_KV_SESSION}" = "1" ] && [ "${RUNTIME_HISTORY_KV_BACKEND}" != "physical_eviction" ]; then
  echo "PERSISTENT_HISTORY_KV_SESSION requires RUNTIME_HISTORY_KV_BACKEND=physical_eviction."
  exit 1
fi

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
IFS=',' read -r -a PORT_LIST <<< "${PORTS}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS}"
SERVER_PIDS=()
RUNNER_PIDS=()
RUNNER_STATUS=0

log_info() {
  echo "[$(date '+%F %T')] $*"
}

source_env_file() {
  local path="$1"
  if [ -f "${path}" ]; then
    local had_errexit=0
    case "$-" in
      *e*) had_errexit=1 ;;
    esac
    set +e
    set +u
    source "${path}"
    local source_status=$?
    set +u
    if [ "${had_errexit}" = "1" ]; then
      set -e
    fi
    if [ "${source_status}" -ne 0 ]; then
      log_info "warning: source ${path} returned ${source_status}; continuing"
    fi
  fi
  return 0
}

cleanup() {
  local status=$?
  for pid in "${SERVER_PIDS[@]}" "${RUNNER_PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup INT TERM EXIT

if [ "${#DEVICE_LIST[@]}" -ne "${#PORT_LIST[@]}" ]; then
  echo "DEVICES and PORTS must have the same length."
  exit 1
fi

write_manifest() {
  mkdir -p "${RUN_ROOT}"
  local bfcl_commit
  local sglang_commit
  local c2kv_commit
  bfcl_commit="$(git -C "${ROOT}" rev-parse HEAD)"
  sglang_commit="$(git -C "${SGLANG_ROOT}" rev-parse HEAD)"
  c2kv_commit="$(git -C /home/zhuyuhan/project/c2kv rev-parse HEAD)"
  "${BFCL_PYTHON}" - "${RUN_ROOT}/run_manifest.json" <<PY
import json, pathlib
manifest = {
    "bfcl_git_commit": "${bfcl_commit}",
    "sglang_git_commit": "${sglang_commit}",
    "c2kv_git_commit": "${c2kv_commit}",
    "model_path": "${MODEL_PATH}",
    "tokenizer_path": "${TOKENIZER_PATH}",
    "ids_path": "${IDS_PATH}",
    "reference_details_path": "${REFERENCE_DETAILS_PATH}",
    "category": "${CATEGORY}",
    "temperature": float("${TEMPERATURE}"),
    "c2kv_ratio": int("${RATIO}"),
    "history_kv_retention_ratio": float("${HISTORY_KV_RETENTION_RATIO}"),
    "history_kv_target_compression": float("${HISTORY_KV_TARGET_COMPRESSION}"),
    "history_kv_recent_window": int("${HISTORY_KV_RECENT_WINDOW}"),
    "history_kv_kernel_size": int("${HISTORY_KV_KERNEL_SIZE}"),
    "history_kv_pooling": "${HISTORY_KV_POOLING}",
    "history_kv_h2o_recent_fraction": float("${HISTORY_KV_H2O_RECENT_FRACTION}"),
    "runtime_history_kv_backend": "${RUNTIME_HISTORY_KV_BACKEND}",
    "persistent_history_kv_session": "${PERSISTENT_HISTORY_KV_SESSION}" == "1",
    "page_size": "${PAGE_SIZE}",
    "disable_radix_cache": "${DISABLE_RADIX_CACHE}" == "1",
    "methods": "${METHODS}".split(","),
    "strict_runtime_eviction": "${STRICT_RUNTIME_EVICTION}" == "1",
    "allow_client_fallback": "${ALLOW_CLIENT_FALLBACK}" == "1",
}
path = pathlib.Path("${RUN_ROOT}/run_manifest.json")
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False))
PY
}

start_server() {
  local slot="$1"
  local device="${DEVICE_LIST[$slot]}"
  local port="${PORT_LIST[$slot]}"
  local log="${RUN_ROOT}/server_${device}_${port}.log"
  log_info "server start device=${device} port=${port}"
  (
    cd "${SGLANG_ROOT}"
    SGLANG_DEBUG_MEMORY_POOL=1 \
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
    SGLANG_EMPTY_CACHE_INTERVAL=1 \
    ASCEND_LAUNCH_BLOCKING=1 \
    TASK_QUEUE_ENABLE=1 \
    no_proxy='*' NO_PROXY='*' http_proxy='' https_proxy='' HTTP_PROXY='' HTTPS_PROXY='' \
    ASCEND_RT_VISIBLE_DEVICES="${device}" \
    server_args=(
      "${SGLANG_PYTHON}" -m sglang.launch_server
      --model-path "${MODEL_PATH}"
      --served-model-name "${MODEL_ID}"
      --model-impl sglang
      --device npu
      --attention-backend ascend
      --tool-call-parser qwen25
      --enable-c2kv
      --c2kv-pool-fraction "${C2KV_POOL_FRACTION}"
      --dtype bfloat16
      --mem-fraction-static "${MEM_FRACTION_STATIC}"
      --host 127.0.0.1
      --port "${port}"
    )
    if [ -n "${PAGE_SIZE}" ]; then
      server_args+=(--page-size "${PAGE_SIZE}")
    fi
    if [ "${DISABLE_RADIX_CACHE}" = "1" ]; then
      server_args+=(--disable-radix-cache)
    fi
    if [ "${PERSISTENT_HISTORY_KV_SESSION}" = "1" ]; then
      server_args+=(--enable-streaming-session)
    fi
    exec "${server_args[@]}"
  ) >"${log}" 2>&1 &
  SERVER_PIDS+=("$!")
}

wait_health() {
  local port="$1"
  for attempt in $(seq 1 900); do
    if curl -fsS --noproxy '*' "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      log_info "server healthy port=${port}"
      return 0
    fi
    sleep 2
  done
  log_info "server health timeout port=${port}"
  return 1
}

flush_server_cache() {
  local port="$1"
  curl -fsS --noproxy '*' -X POST "http://127.0.0.1:${port}/flush_cache" >/dev/null 2>&1 || true
}

run_method() {
  local method="$1"
  local slot="$2"
  local port="${PORT_LIST[$slot]}"
  local method_root="${RUN_ROOT}/${method}"
  mkdir -p "${method_root}/result" "${method_root}/score" "${method_root}/logs"
  log_info "runner start method=${method} port=${port}"
  flush_server_cache "${port}"
  (
    cd "${ROOT}"
    args=(
      "${BFCL_PYTHON}" -m c2kv_eval.adapters.bfcl_history_kv_baselines
      --history-kv-method "${method}"
      --history-kv-retention-ratio "${HISTORY_KV_RETENTION_RATIO}"
      --history-kv-target-compression "${HISTORY_KV_TARGET_COMPRESSION}"
      --history-kv-recent-window "${HISTORY_KV_RECENT_WINDOW}"
      --history-kv-kernel-size "${HISTORY_KV_KERNEL_SIZE}"
      --history-kv-pooling "${HISTORY_KV_POOLING}"
      --history-kv-h2o-recent-fraction "${HISTORY_KV_H2O_RECENT_FRACTION}"
      --runtime-history-kv-backend "${RUNTIME_HISTORY_KV_BACKEND}"
      --category "${CATEGORY}"
      --max-examples "${MAX_EXAMPLES}"
      --ids-path "${IDS_PATH}"
      --reference-details-path "${REFERENCE_DETAILS_PATH}"
      --base-url "http://127.0.0.1:${port}"
      --model "${MODEL_ID}"
      --served-model-name "${MODEL_ID}"
      --tokenizer-path "${TOKENIZER_PATH}"
      --ratio "${RATIO}"
      --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
      --result-dir "${method_root}/result"
      --details-path "${method_root}/logs/details.jsonl"
      --metrics-path "${method_root}/logs/metrics.jsonl"
      --summary-path "${method_root}/logs/summary.json"
      --temperature "${TEMPERATURE}"
    )
    if [ "${STRICT_RUNTIME_EVICTION}" = "1" ]; then
      args+=(--strict-runtime-eviction)
    fi
    if [ "${ALLOW_CLIENT_FALLBACK}" = "1" ]; then
      args+=(--allow-client-fallback)
    fi
    if [ "${PERSISTENT_HISTORY_KV_SESSION}" = "1" ]; then
      args+=(--persistent-history-kv-session)
    fi
    exec "${args[@]}"
  ) >"${method_root}/logs/runner.log" 2>&1
  log_info "runner done method=${method}"
}

evaluate_method() {
  local method="$1"
  local method_root="${RUN_ROOT}/${method}"
  if [ ! -s "${method_root}/logs/details.jsonl" ]; then
    log_info "skip eval method=${method}; no details.jsonl"
    return 0
  fi
  local cmd=(
    "${BFCL_PYTHON}" -m bfcl_eval.eval_checker.eval_runner
    --model "${MODEL_ID}"
    --test-category "${CATEGORY}"
    --result-dir "${method_root}/result"
    --score-dir "${method_root}/score"
  )
  if [ "${MAX_EXAMPLES}" -lt 200 ] || [ -n "${IDS_PATH}" ]; then
    cmd+=(--partial-eval)
  fi
  (
    cd "${ROOT}"
    exec "${cmd[@]}"
  ) >"${method_root}/logs/eval.log" 2>&1
  log_info "eval done method=${method}"
}

log_info "History KV baselines starting"
log_info "RUN_ROOT=${RUN_ROOT}"
log_info "METHODS=${METHODS}"
log_info "DEVICES=${DEVICES} PORTS=${PORTS}"
source_env_file /usr/local/Ascend/cann-8.5.0/set_env.sh
source_env_file /usr/local/Ascend/nnal/atb/set_env.sh

if [ "${CLEAN_OUTPUT}" = "1" ]; then
  rm -rf "${RUN_ROOT}"
fi
mkdir -p "${RUN_ROOT}/logs"
write_manifest >"${RUN_ROOT}/logs/manifest_write.log"

for slot in "${!DEVICE_LIST[@]}"; do
  start_server "${slot}"
done
for port in "${PORT_LIST[@]}"; do
  wait_health "${port}"
done

slot=0
for method in "${METHOD_LIST[@]}"; do
  run_method "${method}" "${slot}" &
  RUNNER_PIDS+=("$!")
  slot=$(( (slot + 1) % ${#DEVICE_LIST[@]} ))
  if [ "${#RUNNER_PIDS[@]}" -ge "${#DEVICE_LIST[@]}" ]; then
    if ! wait "${RUNNER_PIDS[0]}"; then
      RUNNER_STATUS=1
    fi
    RUNNER_PIDS=("${RUNNER_PIDS[@]:1}")
  fi
done
for pid in "${RUNNER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    RUNNER_STATUS=1
  fi
done
RUNNER_PIDS=()

if [ "${RUNNER_STATUS}" -ne 0 ]; then
  log_info "one or more runners failed; leaving partial outputs in ${RUN_ROOT}"
  exit "${RUNNER_STATUS}"
fi

for method in "${METHOD_LIST[@]}"; do
  evaluate_method "${method}"
done

if [ "${RUN_COMPARE}" = "1" ]; then
  compare_args=(
    --run-root "${RUN_ROOT}"
    --methods "${METHODS}"
  )
  if [ "${ASSERT_IDENTITY}" = "1" ]; then
    compare_args+=(--assert-identity-against-full)
  fi
  (
    cd "${ROOT}"
    "${BFCL_PYTHON}" -m c2kv_eval.analysis.compare_history_kv_baselines \
      "${compare_args[@]}"
  )
fi

log_info "summary: ${RUN_ROOT}/history_kv_baseline_summary.csv"
log_info "report: ${RUN_ROOT}/history_kv_baseline_summary.md"
